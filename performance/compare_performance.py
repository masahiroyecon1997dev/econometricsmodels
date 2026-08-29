"""OLSの実行時間・メモリ使用量を、statsmodels/pyfixestと比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`OLS(...).fit()`全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

## 計測方法

- **`engine`は必ずreleaseビルドで計測する**: `uv run maturin develop`（デフォルト、debug
  ビルド）と`uv run maturin develop --release`とで、`.so`ファイルサイズが924MB→32.7MB、
  実行時間が最大44倍（HAC, n=100,000）変わることを実測で確認した。debugビルドのままだと
  「engineがstatsmodels/pyfixestより大幅に遅い」という誤った結論に至る
  （実際にはreleaseビルドではclassical/HC1/clusterで同等以上、HACもn=1,000,000で
  ほぼ互角になる）。本スクリプトの実行前に必ず`uv run maturin develop --release`を
  実行すること。`_worker()`内で`.so`ファイルサイズによる簡易チェックを行い、
  debugビルドの疑いがある場合は警告を出す。
- **メモリ計測に`tracemalloc`は使わない**: `tracemalloc`はPythonのpymallocフックのみを
  追跡するため、`engine`内部（faerの行列確保等、Rustのヒープ）やnumpy配列のバッファの
  ようなネイティブメモリ確保を捕捉できない。実測でn=2,000,000, k=5（設計行列だけで
  80MB相当）でも3.8KB程度しか検知しないことを確認した。statsmodels/pyfixestも同様に
  numpyバッファは過小評価されるため、公平な比較にならない。
  → **プロセス単位のピークRSS**（`resource.getrusage(RUSAGE_SELF).ru_maxrss`）を使う。
- **サブプロセス隔離**: 1計測点＝1サブプロセス。同一プロセス内で複数ライブラリ・
  複数計測点を連続実行すると、アロケータが解放済みメモリを保持したままになり
  （OSに返却されない）、後続の計測のRSSが汚染される。サブプロセスなら
  ピークRSSがその計測点だけの値になる。
- **ウォームアップ**: 計測対象のライブラリ・cov_typeで1回ウォームアップ実行してから
  タイミング計測に入る（pyfixestのnumba JITコンパイル等、初回呼び出し特有の
  一回性オーバーヘッドを除外するため）。
- **実行時間**: ウォームアップ後、`repeats`回実行し中央値を採用する
  （`time.perf_counter()`。外れ値の影響を避けるため平均ではなく中央値）。
- **HACのラグ数を明示的に揃える**: 3ライブラリとも自動ラグ選択式が異なりうるため、
  `_common.hac_auto_lag(n)`（engineの自動選択式と同じ）で計算した同一のラグ数を明示指定する。
  ラグ選択方式自体の違いではなく、Newey-West計算そのものの性能差を見るため。

## 既知の限界

- BLAS/線形代数バックエンドのスレッド数は制御していない（numpy/statsmodelsが
  内部で使うBLASのマルチスレッド化とfaerのスレッド数設定が異なりうる）。
  観測された差にはこの要因も混ざりうる。

使用例（リポジトリルートから）:
    # 一括実行（n軸・k軸両方、結果をJSONに保存）
    python -m performance.compare_performance \\
        --output docs/spec/_ols_performance_results.json

    # 単体計測（デバッグ用）
    python -m performance.compare_performance \\
        --worker --library engine --cov-type hac --n 1000 --k 5
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from benchmark.common import hac_auto_lag
from benchmark.linear.datasets import generate_linear_dataset

LIBRARIES = ["engine", "statsmodels", "pyfixest"]
COV_TYPES = ["classical", "hc1", "cluster", "hac"]

N_CLUSTERS = 50

N_SWEEP = [1_000, 10_000, 100_000, 1_000_000]
# HACもreleaseビルドであれば他のcov_typeと同程度の速度になるため、
# 別のn範囲に絞る必要はない（上記「engineは必ずreleaseビルドで計測する」参照）。
HAC_N_SWEEP = N_SWEEP
N_SWEEP_FIXED_K = 5

K_SWEEP = [2, 5, 10, 20]
K_SWEEP_FIXED_N = 10_000

DEFAULT_REPEATS = 3
DEFAULT_SEED = 42

THIS_FILE = Path(__file__).resolve()


# debugビルドの`_lib*.so`は約924MB、releaseビルドは約32.7MBだった（実測）。
# 大きな余裕を持ってこの間の閾値でdebugビルドの疑いを警告する。
_DEBUG_BUILD_SO_SIZE_THRESHOLD_BYTES = 200 * 1024 * 1024


def _warn_if_debug_build() -> None:
    """`_lib`のインストール済み`.so`サイズからdebugビルドの疑いを警告する。

    モジュール docstring「engineは必ずreleaseビルドで計測する」参照。
    誤検知を許容するヒューリスティックなチェックであり、確実な判定では
    ないため警告に留め、実行は止めない。
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


def _build_dataframe(n: int, k: int, seed: int) -> pl.DataFrame:
    df, _ = generate_linear_dataset("baseline", n=n, k=k, seed=seed)
    return df.with_row_index("time_id").with_columns(
        (pl.col("time_id") % N_CLUSTERS).alias("cluster_group")
    )


def _fit_once_engine(
    df: pl.DataFrame, x_cols: list[str], cov_type: str, lag: int
):
    from econometricsmodels import OLS, OLSOptions

    if cov_type == "classical":
        options = OLSOptions(cov_type="classical")
    elif cov_type == "hc1":
        options = OLSOptions(cov_type="hc1")
    elif cov_type == "cluster":
        options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    elif cov_type == "hac":
        options = OLSOptions(cov_type="hac", hac_lags=lag)
    else:
        raise ValueError(f"unknown cov_type: {cov_type!r}")
    return OLS(df, y="y", x=x_cols, options=options).fit()


def _fit_once_statsmodels(pdf, formula: str, cov_type: str, lag: int):
    import statsmodels.formula.api as smf

    sm_cov_type = {
        "classical": "nonrobust",
        "hc1": "HC1",
        "cluster": "cluster",
        "hac": "hac",
    }[cov_type]
    fit_kwargs: dict = {"cov_type": sm_cov_type, "use_t": True}
    if cov_type == "cluster":
        fit_kwargs["cov_kwds"] = {"groups": pdf["cluster_group"]}
    elif cov_type == "hac":
        fit_kwargs["cov_kwds"] = {"maxlags": lag}
    return smf.ols(formula, data=pdf).fit(**fit_kwargs)


def _fit_once_pyfixest(pdf, formula: str, cov_type: str, lag: int):
    import pyfixest as pf

    if cov_type == "classical":
        return pf.feols(formula, data=pdf, vcov="iid")
    elif cov_type == "hc1":
        return pf.feols(formula, data=pdf, vcov="HC1")
    elif cov_type == "cluster":
        return pf.feols(formula, data=pdf, vcov={"CRV1": "cluster_group"})
    elif cov_type == "hac":
        return pf.feols(
            formula,
            data=pdf,
            vcov="NW",
            vcov_kwargs={"time_id": "time_id", "lag": lag},
        )
    else:
        raise ValueError(f"unknown cov_type: {cov_type!r}")


def _worker(
    library: str, cov_type: str, n: int, k: int, seed: int, repeats: int
) -> dict:
    df = _build_dataframe(n, k, seed)
    x_cols = [f"x{j + 1}" for j in range(k)]
    formula = "y ~ " + " + ".join(x_cols)
    lag = hac_auto_lag(n)

    if library == "engine":
        _warn_if_debug_build()

        def fit_once():
            return _fit_once_engine(df, x_cols, cov_type, lag)
    elif library == "statsmodels":
        pdf = df.to_pandas()

        def fit_once():
            return _fit_once_statsmodels(pdf, formula, cov_type, lag)
    elif library == "pyfixest":
        pdf = df.to_pandas()

        def fit_once():
            return _fit_once_pyfixest(pdf, formula, cov_type, lag)
    else:
        raise ValueError(f"unknown library: {library!r}")

    fit_once()  # warmup

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fit_once()
        times.append(time.perf_counter() - t0)

    return {
        "time_median_s": statistics.median(times),
        "time_all_s": times,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _run_isolated(
    library: str, cov_type: str, n: int, k: int, seed: int, repeats: int
) -> dict:
    """1計測点をサブプロセスで実行する（プロセスRSS隔離のため、モジュール docstring参照）。"""
    # ワーカーは `-m` でリポジトリルートを cwd にして起動する。DGP 等を
    # `import benchmark.*` で解決するため（Initiative A でパッケージ化）、
    # ファイルパス直接起動だと解決できない。cwd=リポジトリルートなら `-m` が
    # ルートを sys.path に載せるため PYTHONPATH に依存しない。
    repo_root = THIS_FILE.parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "performance.compare_performance",
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
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(repo_root),
    )
    return json.loads(proc.stdout)


def _measure_point(
    axis: str,
    n: int,
    k: int,
    cov_type: str,
    library: str,
    repeats: int,
    seed: int,
) -> dict:
    measured = _run_isolated(library, cov_type, n, k, seed, repeats)
    row = {
        "axis": axis,
        "n": n,
        "k": k,
        "cov_type": cov_type,
        "library": library,
        **measured,
    }
    print(
        f"[{axis}-sweep] n={n} k={k} cov_type={cov_type} library={library}: "
        f"time_median={row['time_median_s']:.4f}s "
        f"peak_rss={row['peak_rss_kb'] / 1024:.1f}MB",
        file=sys.stderr,
    )
    return row


def run_n_sweep(
    cov_types: list[str],
    libraries: list[str],
    repeats: int,
    seed: int,
    fixed_k: int,
) -> list[dict]:
    """n軸のスイープ。cov_typeごとに異なるn候補を許容する構造にしてある
    （`HAC_N_SWEEP`定義のコメント参照。現状は全cov_typeで同じ値だが、
    将来再び特定cov_typeだけ上限を下げる必要が生じた場合に備えて残す）。
    """
    results = []
    for cov_type in cov_types:
        n_values = HAC_N_SWEEP if cov_type == "hac" else N_SWEEP
        for n in n_values:
            for library in libraries:
                results.append(
                    _measure_point(
                        "n", n, fixed_k, cov_type, library, repeats, seed
                    )
                )
    return results


def run_k_sweep(
    cov_types: list[str],
    libraries: list[str],
    repeats: int,
    seed: int,
    fixed_n: int,
) -> list[dict]:
    results = []
    for k in K_SWEEP:
        for cov_type in cov_types:
            for library in libraries:
                results.append(
                    _measure_point(
                        "k", fixed_n, k, cov_type, library, repeats, seed
                    )
                )
    return results


def build_report(
    cov_types: list[str], libraries: list[str], repeats: int, seed: int
) -> dict:
    n_results = run_n_sweep(
        cov_types, libraries, repeats, seed, N_SWEEP_FIXED_K
    )
    k_results = run_k_sweep(
        cov_types, libraries, repeats, seed, K_SWEEP_FIXED_N
    )
    return {
        "_meta": {
            "purpose": (
                "OLS(...).fit()のエンドツーエンド実行時間・ピークRSSを"
                "statsmodels/pyfixestと比較する"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "n_sweep": N_SWEEP,
            "hac_n_sweep": HAC_N_SWEEP,
            "n_sweep_fixed_k": N_SWEEP_FIXED_K,
            "k_sweep": K_SWEEP,
            "k_sweep_fixed_n": K_SWEEP_FIXED_N,
            "cov_types": cov_types,
            "libraries": libraries,
            "repeats": repeats,
            "seed": seed,
        },
        "results": n_results + k_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        action="store_true",
        help="内部用: 1計測点をこのプロセス内で実行しJSONを標準出力する",
    )
    parser.add_argument("--library", choices=LIBRARIES)
    parser.add_argument("--cov-type", choices=COV_TYPES)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--output",
        default=None,
        help="結果JSONの出力先。省略時は標準出力のみ",
    )
    args = parser.parse_args()

    if args.worker:
        output = _worker(
            args.library,
            args.cov_type,
            args.n,
            args.k,
            args.seed,
            args.repeats,
        )
        print(json.dumps(output))
    else:
        report = build_report(COV_TYPES, LIBRARIES, args.repeats, args.seed)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False)
            )
            print(f"wrote {output_path}", file=sys.stderr)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
