"""手法非依存の性能比較ハーネス（サブプロセス隔離・タイミング・ピークRSS計測）。

各 `performance/compare_<method>.py` は、手法固有の「変わる部分」を
`PerfAdapter` にまとめてこのモジュールの `run_cli()` に渡す。ハーネス側は
以下の「手法によらず共通の骨格」を担う。

- **`engine` は必ずreleaseビルドで計測する**: `uv run maturin develop`（デフォルト、
  debugビルド）と `--release` とで実行時間が最大140倍変わることを実測で確認済み
  （`docs/performance/ols.md`「最重要の教訓」）。debugビルドのままだと
  「engineがリファレンス実装より大幅に遅い」という誤った結論に至る。実行前に必ず
  `uv run maturin develop --release` を実行すること。`_worker()` 内で `.so` ファイル
  サイズによる簡易チェックを行い、debugビルドの疑いがある場合は警告を出す。
- **メモリ計測に `tracemalloc` は使わない**: `tracemalloc` はPythonのpymallocフックのみを
  追跡するため、`engine` 内部（faerの行列確保等、Rustのヒープ）やnumpy配列のバッファの
  ようなネイティブメモリ確保を捕捉できない（設計行列だけで80MB相当のケースでも
  3.8KB程度しか検知しないことを実測で確認）。リファレンス実装側も同様に過小評価される
  ため公平な比較にならない。→ **プロセス単位のピークRSS**
  （`resource.getrusage(RUSAGE_SELF).ru_maxrss`）を使う。
- **サブプロセス隔離**: 1計測点＝1サブプロセス。同一プロセス内で複数ライブラリ・
  複数計測点を連続実行すると、アロケータが解放済みメモリを保持したままになり
  （OSに返却されない）後続の計測のRSSが汚染される。サブプロセスならピークRSSが
  その計測点だけの値になる。
- **ウォームアップ**: 計測対象のライブラリ・cov_typeで1回ウォームアップ実行してから
  タイミング計測に入る（初回呼び出し特有の一回性オーバーヘッドを除外するため）。
- **実行時間**: ウォームアップ後、`repeats` 回実行し中央値を採用する
  （`time.perf_counter()`。外れ値の影響を避けるため平均ではなく中央値）。
- **DataFrame→pandas変換は計測区間の外**: リファレンス実装（statsmodels/linearmodels）は
  pandas入力を前提とするため `df.to_pandas()` が必要だが、これはライブラリ本体の
  仕事ではなく変換コストなので、`_worker()` が計測ループの外で1回だけ実施し、
  `FitContext.pandas_df` として渡す（`.claude/rules/testing-policy.md`「パフォーマンス
  比較（ベンチマーク）の方法論」）。
- **HACのラグ数を明示的に揃える**: ライブラリごとに自動ラグ選択式が異なりうるため、
  `benchmark.common.hac_auto_lag(n)`（engineの自動選択式と同じ）で計算した同一の
  ラグ数を `FitContext.hac_lags` で渡す。ラグ選択方式自体の違いではなく、
  Newey-West計算そのものの性能差を見るため。
- **スレッド数を1に固定する**: `_run_isolated()` がワーカーサブプロセスの環境変数で
  engine（faer/rayon）・リファレンス実装（numpy/BLAS）とも1スレッドに固定する
  （`_SINGLE_THREAD_ENV`）。多コア機で線形代数バックエンドのスレッドプールが負荷下で
  競合し、engine の classical n=1,000,000 の実行時間が単一スレッド時の20倍以上
  （約0.15秒→3〜4秒）に膨れ上がり計測が不安定になる現象を実測で確認したため
  （`docs/performance/ols.md`「既知の限界」）。単一スレッドに揃えることで
  「Rustコアの計算効率 vs Python+BLAS」という比較の主目的を、スレッドプール挙動の
  環境差から切り離す。

## 既知の限界

- 単一スレッド固定のため、線形代数バックエンドのマルチスレッド化による高速化は
  この比較には現れない（多コアでの実利用の性能特性とは別軸）。engine 側の
  マルチスレッド時の不安定性そのものは別途エンジン側で調査する
  （`docs/planning/specs/refactoring-candidates.md`）。
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from benchmark.common import hac_auto_lag

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

# debugビルドの `_lib*.so` は約900MB、releaseビルドは約33MBだった（実測）。
# 大きな余裕を持ってこの間の閾値でdebugビルドの疑いを警告する。
_DEBUG_BUILD_SO_SIZE_THRESHOLD_BYTES = 200 * 1024 * 1024

# ワーカーサブプロセスに渡す、線形代数バックエンドのスレッド数を1に固定する
# 環境変数（モジュール docstring「スレッド数を1に固定する」参照）。numpy が MKL /
# OpenBLAS / Accelerate のどれとリンクされていても効くよう主要な変数を網羅する。
_SINGLE_THREAD_ENV = {
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}


@dataclass(frozen=True)
class FitContext:
    """`PerfAdapter.fit_once` に渡す、手法によらず固定のシグネチャ。

    Attributes:
        library: 計測対象ライブラリ（"engine" またはリファレンス実装名）。
        df: `PerfAdapter.build_dataframe` が生成した計測用DataFrame。
            engine にはこれをそのまま（ゼロコピーのArrow経由で）渡す。
        pandas_df: `df.to_pandas()` の結果。ハーネスが計測区間の外で1回だけ
            変換する。`library == "engine"` のときは `None`。
        x_cols: 説明変数の列名（["x1", ..., "xk"]）。
        y_col: 被説明変数の列名。
        cov_type: 計測対象の分散推定タイプ。
        hac_lags: `hac_auto_lag(n)` の値。cov_type が HAC 以外なら無視してよい。
        cluster_col: クラスターロバスト用のグループ列名。使わない手法・
            cov_type では `None`。
        weight_col: WLS の重み列名。重みを使わない手法（OLS/Logit 等）では
            `None`。
        method: 推定 method。Logit/Probit の "newton"/"bfgs"/"lbfgs"、IV の
            "2sls"/"gmm" 等。n/k スイープでは `PerfAdapter.default_method`、
            method 軸では `PerfAdapter.extra_methods` の各値が入る。method 軸を
            持たない手法では常に `default_method`（`fit_once` 側で無視してよい）。
    """

    library: str
    df: pl.DataFrame
    pandas_df: object | None
    x_cols: list[str]
    y_col: str
    cov_type: str
    hac_lags: int
    cluster_col: str | None
    weight_col: str | None
    method: str


@dataclass(frozen=True)
class PerfAdapter:
    """1手法分の性能比較の「変わる部分」をまとめたアダプタ。

    ハーネス（本モジュール）が持つ計測ロジック（サブプロセス隔離・ウォームアップ＋
    中央値・ピークRSS・n/k スイープ）に対して、手法ごとのデータ生成・ライブラリ別
    fit 呼び出し・軸の刻みを注入する。既存の `benchmark.common.driver.run_fixture_cli`
    と同じ「callable＋設定を渡す」スタイル。

    Attributes:
        method: 手法名（"ols" 等）。レポートのタイトル・`_meta`・
            `docs/performance/<method>.md` のパス生成に使う。
        module: このアダプタを所有するスクリプトのドット付きモジュールパス
            （"performance.compare_ols"）。1計測点をサブプロセスで再実行する際の
            `python -m <module> --worker ...` の呼び出し先。
        libraries: 計測対象ライブラリ。先頭は必ず "engine"。以降は
            README「Verification accuracy」表の primary reference のみ
            （OLS/WLS/Logit/Probit: "statsmodels"、IV: "linearmodels"）。
        cov_types: 計測する cov_type。`.claude/rules/testing-policy.md`
            「パフォーマンス比較（ベンチマーク）の方法論」に従い、代表2点
            （最も軽い classical と、最も計算コストの重いもの）で足りる。
        reference_versions: `{"statsmodels_version": "0.14.6", ...}` を返す
            callable。`_meta` に記録する（testing-policy.md「必須事項」）。
            engine 分は git ハッシュで足りるため不要。
        build_dataframe: `(n, k, seed) -> pl.DataFrame`。cluster 列など手法側で
            必要な派生列の付与もここで行う。
        fit_once: `FitContext -> object`。1回の推定を実行する（計測区間内で
            呼ばれる）。返り値は使われないが、**リファレンス実装が結果を遅延評価
            （lazy property）で計算する設計の場合、この関数内で明示的にアクセスして
            計算を確定させること**（engine は係数・標準誤差と同じ呼び出しで適合度
            統計量まで常に一括計算するため、揃えないと不公平な比較になる）。
        n_sweep / n_sweep_fixed_k: n 軸スイープの n の刻みと、その際固定する k。
        k_sweep / k_sweep_fixed_n: k 軸スイープの k の刻みと、その際固定する n。
        cluster_col: `FitContext.cluster_col` に渡す列名。cluster を計測しない
            手法では `None` のまま。
        weight_col: `FitContext.weight_col` に渡す列名。WLS のみ設定する
            （`build_dataframe` がその列を含む DataFrame を返す前提）。
        default_method: n/k スイープで使う既定 method（`FitContext.method` に
            入る）。Logit/Probit は "newton"、IV は "2sls"。method 軸を持たない
            手法でも `fit_once` がこの値を無視すれば影響はない。
        extra_methods: method 軸で追加計測する method（Logit/Probit の
            `("bfgs", "lbfgs")`、IV の `("gmm",)` 等）。`default_method` は n/k
            スイープに含まれるため列挙しない。空なら method 軸なし。method 軸は
            cov_type=cov_types[0]・k=n_sweep_fixed_k・n=n_sweep[-1] の1点でのみ
            回す（testing-policy.md「方法論」＝代表点で足りる）。
        default_repeats / default_seed: CLI 引数のデフォルト。
    """

    method: str
    module: str
    libraries: Sequence[str]
    cov_types: Sequence[str]
    reference_versions: Callable[[], dict[str, str]]
    build_dataframe: Callable[[int, int, int], pl.DataFrame]
    fit_once: Callable[[FitContext], object]

    n_sweep: Sequence[int] = (1_000, 10_000, 100_000, 1_000_000)
    n_sweep_fixed_k: int = 5
    k_sweep: Sequence[int] = (5, 20)
    k_sweep_fixed_n: int = 10_000
    cluster_col: str | None = None
    weight_col: str | None = None
    default_method: str = "newton"
    extra_methods: Sequence[str] = ()
    default_repeats: int = 3
    default_seed: int = 42


def _warn_if_debug_build() -> None:
    """`_lib` のインストール済み `.so` サイズからdebugビルドの疑いを警告する。

    モジュール docstring「engineは必ずreleaseビルドで計測する」参照。誤検知を
    許容するヒューリスティックなチェックであり、確実な判定ではないため警告に
    留め、実行は止めない。
    """
    from econometricsmodels import _lib

    so_path = Path(_lib.__file__)
    size = so_path.stat().st_size
    if size > _DEBUG_BUILD_SO_SIZE_THRESHOLD_BYTES:
        print(
            f"WARNING: {so_path} is {size / 1024 / 1024:.0f}MB, which looks "
            "like a debug build (release build is ~33MB). Run "
            "`uv run maturin develop --release` before benchmarking, "
            "otherwise timing results will be misleadingly slow.",
            file=sys.stderr,
        )


def _worker(
    adapter: PerfAdapter,
    library: str,
    cov_type: str,
    n: int,
    k: int,
    seed: int,
    repeats: int,
    method: str = "newton",
) -> dict:
    df = adapter.build_dataframe(n, k, seed)
    x_cols = [f"x{j + 1}" for j in range(k)]
    pandas_df = None if library == "engine" else df.to_pandas()
    ctx = FitContext(
        library=library,
        df=df,
        pandas_df=pandas_df,
        x_cols=x_cols,
        y_col="y",
        cov_type=cov_type,
        hac_lags=hac_auto_lag(n),
        cluster_col=adapter.cluster_col,
        weight_col=adapter.weight_col,
        method=method,
    )

    if library == "engine":
        _warn_if_debug_build()

    adapter.fit_once(ctx)  # warmup

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        adapter.fit_once(ctx)
        times.append(time.perf_counter() - t0)

    return {
        "time_median_s": statistics.median(times),
        "time_all_s": times,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _run_isolated(
    adapter: PerfAdapter,
    library: str,
    cov_type: str,
    n: int,
    k: int,
    seed: int,
    repeats: int,
    method: str = "newton",
) -> dict:
    """1計測点をサブプロセスで実行する（プロセスRSS隔離のため、モジュール docstring参照）。"""
    # ワーカーは `-m` でリポジトリルートを cwd にして起動する。DGP 等を
    # `import benchmark.*` で解決するため（Initiative A でパッケージ化）、
    # ファイルパス直接起動だと解決できない。cwd=リポジトリルートなら `-m` が
    # ルートを sys.path に載せるため PYTHONPATH に依存しない。
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            adapter.module,
            "--worker",
            "--library",
            library,
            "--cov-type",
            cov_type,
            "--n",
            str(n),
            "--k",
            str(k),
            "--seed",
            str(seed),
            "--repeats",
            str(repeats),
            "--method",
            method,
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, **_SINGLE_THREAD_ENV},
    )
    return json.loads(proc.stdout)


def _measure_point(
    adapter: PerfAdapter,
    axis: str,
    n: int,
    k: int,
    cov_type: str,
    library: str,
    repeats: int,
    seed: int,
    method: str = "newton",
) -> dict:
    measured = _run_isolated(
        adapter, library, cov_type, n, k, seed, repeats, method
    )
    row = {
        "axis": axis,
        "n": n,
        "k": k,
        "cov_type": cov_type,
        "method": method,
        "library": library,
        **measured,
    }
    print(
        f"[{axis}-sweep] n={n} k={k} cov_type={cov_type} method={method} "
        f"library={library}: time_median={row['time_median_s']:.4f}s "
        f"peak_rss={row['peak_rss_kb'] / 1024:.1f}MB",
        file=sys.stderr,
    )
    return row


def run_n_sweep(adapter: PerfAdapter, repeats: int, seed: int) -> list[dict]:
    """n 軸のスイープ（k は `adapter.n_sweep_fixed_k` 固定）。"""
    results = []
    for cov_type in adapter.cov_types:
        for n in adapter.n_sweep:
            for library in adapter.libraries:
                results.append(
                    _measure_point(
                        adapter,
                        "n",
                        n,
                        adapter.n_sweep_fixed_k,
                        cov_type,
                        library,
                        repeats,
                        seed,
                        adapter.default_method,
                    )
                )
    return results


def run_k_sweep(adapter: PerfAdapter, repeats: int, seed: int) -> list[dict]:
    """k 軸のスイープ（n は `adapter.k_sweep_fixed_n` 固定）。"""
    results = []
    for k in adapter.k_sweep:
        for cov_type in adapter.cov_types:
            for library in adapter.libraries:
                results.append(
                    _measure_point(
                        adapter,
                        "k",
                        adapter.k_sweep_fixed_n,
                        k,
                        cov_type,
                        library,
                        repeats,
                        seed,
                        adapter.default_method,
                    )
                )
    return results


def run_method_sweep(
    adapter: PerfAdapter, repeats: int, seed: int
) -> list[dict]:
    """method 軸（`adapter.extra_methods` を代表点1つで計測）。

    cov_type=cov_types[0]・k=n_sweep_fixed_k・n=n_sweep[-1] の1点で、
    追加 method（bfgs/lbfgs、gmm 等）× ライブラリを回す。`default_method` は
    n/k スイープに含まれるため対象外。`extra_methods` が空なら何もしない。
    """
    if not adapter.extra_methods:
        return []
    cov_type = adapter.cov_types[0]
    n = adapter.n_sweep[-1]
    k = adapter.n_sweep_fixed_k
    results = []
    for method in adapter.extra_methods:
        for library in adapter.libraries:
            results.append(
                _measure_point(
                    adapter,
                    "method",
                    n,
                    k,
                    cov_type,
                    library,
                    repeats,
                    seed,
                    method,
                )
            )
    return results


def build_report(adapter: PerfAdapter, repeats: int, seed: int) -> dict:
    n_results = run_n_sweep(adapter, repeats, seed)
    k_results = run_k_sweep(adapter, repeats, seed)
    method_results = run_method_sweep(adapter, repeats, seed)
    return {
        "_meta": {
            "method": adapter.method,
            "purpose": (
                f"{adapter.method}の推定のエンドツーエンド実行時間・ピークRSSを"
                f"{'/'.join(adapter.libraries[1:])}と比較する"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            **adapter.reference_versions(),
            "n_sweep": list(adapter.n_sweep),
            "n_sweep_fixed_k": adapter.n_sweep_fixed_k,
            "k_sweep": list(adapter.k_sweep),
            "k_sweep_fixed_n": adapter.k_sweep_fixed_n,
            "cov_types": list(adapter.cov_types),
            "libraries": list(adapter.libraries),
            "default_method": adapter.default_method,
            "extra_methods": list(adapter.extra_methods),
            "method_sweep_n": (
                adapter.n_sweep[-1] if adapter.extra_methods else None
            ),
            "repeats": repeats,
            "seed": seed,
        },
        "results": n_results + k_results + method_results,
    }


def run_cli(
    adapter: PerfAdapter,
    doc: str | None = None,
    argv: list[str] | None = None,
) -> None:
    """各 `compare_<method>.py` の `__main__` から呼ぶエントリポイント。

    `--worker` 指定時は1計測点をこのプロセス内で実行し JSON を標準出力する
    （標準出力は JSON のみ。警告等は標準エラーへ）。それ以外はフルスイープを
    実行し、`--output` があればそこへ、無ければ標準出力へレポート JSON を書く。
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument(
        "--worker",
        action="store_true",
        help="内部用: 1計測点をこのプロセス内で実行しJSONを標準出力する",
    )
    parser.add_argument("--library", choices=list(adapter.libraries))
    parser.add_argument("--cov-type", choices=list(adapter.cov_types))
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int, default=adapter.default_seed)
    parser.add_argument("--repeats", type=int, default=adapter.default_repeats)
    parser.add_argument(
        "--method",
        default=adapter.default_method,
        choices=[adapter.default_method, *adapter.extra_methods],
    )
    parser.add_argument(
        "--output",
        default=None,
        help="結果JSONの出力先。省略時は標準出力のみ",
    )
    args = parser.parse_args(argv)

    if args.worker:
        output = _worker(
            adapter,
            args.library,
            args.cov_type,
            args.n,
            args.k,
            args.seed,
            args.repeats,
            args.method,
        )
        print(json.dumps(output))
        return

    report = build_report(adapter, args.repeats, args.seed)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        print(f"wrote {output_path}", file=sys.stderr)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
