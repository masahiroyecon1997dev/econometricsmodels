"""Tobit の実行時間・ピークRSS を計測するベンチマークスクリプト（engine 単独）。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`Tobit(...).fit()` 全体（Python API 呼び出し、Arrow 変換・PyO3 オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・release ビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは Tobit 固有のアダプタのみを定義する。

## engine 単独で計測する（インプロセス計測できるリファレンス実装が無い）

OLS/WLS/Logit/Probit は statsmodels、IV は linearmodels をリファレンスに並べて
相対比較するが、Tobit にはインプロセス計測できるリファレンス実装を置かない。

- 正確性検証の主リファレンスは R `AER::tobit`（＝ `survreg`）だが、R は共通ハーネス
  のインプロセス計測モデル（`fit_once(ctx)` を計測ループ内で呼ぶ）に乗らない。
- statsmodels にネイティブ Tobit は無い。
- 以前は py4etrics（`statsmodels.GenericLikelihoodModel` ベースの Tobit）を
  リファレンスに使っていたが、(1) 実質メンテ停止（最終リリース 2024-01、依存ピン
  無し）で statsmodels 0.15.0 更新時に import 不能になった、(2) 数値微分ヘッシアン
  のため k>=8 で事実上フリーズし k 軸は元々 engine 単独だった、(3) 「engine の
  解析的スコア／ヘッシアン vs statsmodels の有限差分」という交絡を含み Rust 化
  そのものの寄与を単独では取り出せない、という理由で 2026-09（statsmodels 0.15.0
  更新に合わせて）撤去した。撤去前の最終クロス比較スナップショットは
  `docs/performance/tobit.md`「[アーカイブ]」節に凍結してある。

したがって本スクリプトは engine の Tobit が軸ごとに「時間がかかりすぎていないか」を
観察する（他手法ページのような相対比較ではなく、engine 単独の絶対値とスケーリング
の推移）。正式な数値照合は従来どおり R ベースの
`tests/nonlinear/test_tobit_reference.py`（`AER::tobit`）・`test_tobit_crosscheck.py`
（`censReg`）が担う。

## cov_type の範囲

`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に
従い、代表2点のみ計測する: 最も軽い `classical` と、最も計算コストの重い
`cluster`。n=100,000・k=5 で全 cov_type を実測して確認済み（classical 0.138s <
opg 0.148s < hc1 0.150s < hc0 0.154s < cluster 0.162s、engine・newton）。省略する
`opg`/`hc0`/`hc1` は classical と cluster の間に収まる。cluster の疑似グループ数は
50 固定。

## スイープ軸

- **n 軸**（k=5 固定、newton）: classical / cluster とも n=1,000〜100,000。加えて
  **classical のみ n=200,000 / 1,000,000**（`n_sweep_engine_only`、seed は
  `default_seed=42` のみ。全 cov_type で大 n を回すと CI 時間がかさむため classical
  に絞る。下記「## n 軸の大標本点」参照）。
- **k 軸**（n=10,000 固定、newton）: k=5・20、classical / cluster。
- **method 軸**: 下記「## method（オプティマイザ）の範囲」参照。

## n 軸の大標本点（classical のみ n=200,000 / 1,000,000）

- **n=1,000,000（seed=42）が Issue #291 の再現点**。#291 は乱数 β の
  `moderate_censoring` DGP で engine が
  `ComputationError: the Hessian is singular and cannot be inverted` になっていた
  バグで、seed=42 では n=1,000,000 で失敗（n=500,000 までは成功）。Probit #284 と
  同系統。修正は `d797f9b` / `5b79ffe`（`FaerNewton` の停滞収束判定）。
  `_perf_harness._run_isolated` は `check=True` なので、**再発すれば engine の
  `.fit()` が例外を投げて benchmark ジョブが失敗する**。
- **n=200,000 は n スケーリングのデータ点＋安価な早期警告**。#291 は seed 依存で、
  seed=1 は n=200,000 で既に破綻していたが、この guard が回す seed=42 では
  n=200,000 は #291 を再現しない。
- **このガードの限界**（深いカバレッジは docstring 末尾「今後」および
  `docs/performance/tobit.md`「今後の検討事項」の凍結フィクスチャ + `AER::tobit` に委ねる）:
  - **単一 seed**（42）。#291 が示した seed 感度（seed=1 は n=200,000 で破綻）は
    `run_cli` の `--seed` がレポート全体で1つのため、この仕組みでは検査できない。
  - **捕捉できるのは再発時の「例外」のみ**。#291 の修正が持ち込みうる新しい失敗様式
    （非最適点で収束宣言＝silently-wrong な収束）は、finite な結果さえ返れば
    `check=True` を通ってしまうため、この性能スクリプトでは構造上検知できない。
  - **発火はリリース単位**。`benchmark_performance.yml` はタグ push（`v*`）+
    `workflow_dispatch` のみで、per-PR では回らない。solver 回帰が
    リリースブランチにマージされても次のタグまで捕捉されない。
- 単体テスト（`engine/src/nonlinear/common.rs` の
  `run_solver_newton_converges_when_cost_hits_floating_point_floor_above_gradient_tol`
  ほか）は停滞判定ロジックの模擬。実 Tobit 経路を大標本で通す自動チェックはこの
  guard が唯一（上記の限界つき）。実行時間も `docs/performance/tobit.md` に記録され、
  停滞検出器の誤作動による反復数増（速度劣化）は可視化される。

## method（オプティマイザ）の範囲

既定は Newton-Raphson で、n/k スイープは newton で回す。加えて `lbfgs` を
**method 軸**として代表点1つ（cov_type=classical・k=5・n=100,000）で計測する。
**`bfgs` は現状 method 軸から除外している**: engine の Tobit BFGS 経路は n>=10,000
で `MoreThuenteLineSearch: NaN or Inf` により発散する（Issue #292。
`_perf_harness._run_isolated` は `check=True` なので、そのまま入れると benchmark
ジョブごと失敗する）。#292 解消後に `extra_methods` へ戻す。

quasi-Newton のパフォーマンス劣化の早期検知として、`check_report`
（`_check_method_ratios`）で engine の `lbfgs/newton` 実行時間比を計算し、5x を
超えたら job summary に `> [!WARNING]` を出す（CI failure にはしない。実時間の
絶対値ではなく同一ジョブ内の比を見るため、共有ランナーの速度差に影響されない。
#285 と同系統の劣化のガード）。

使用例（リポジトリルートから）:
    # 一括実行（n軸・k軸両方、結果をJSONに保存）
    python -m performance.compare_tobit \\
        --output docs/performance/results/tobit.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する
    # （一括実行では `_perf_harness._run_isolated` が自動で設定する）。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_tobit \\
        --worker --library engine --cov-type cluster --n 1000 --k 5
"""

from __future__ import annotations

import polars as pl

from benchmark.nonlinear.datasets import generate_censored_regression_dataset
from performance._perf_harness import FitContext, PerfAdapter, run_cli

# クラスターロバストSE計測用の疑似グループ数（compare_logit.py / compare_probit.py
# と同じ）。
_N_CLUSTERS = 50

# 計測に使う打ち切りシナリオ。左打ち切り ~35%（Tobit テストの BASELINE_SCENARIO と
# 同じ。`benchmark/nonlinear/datasets.py` の `_TOBIT_SCENARIO_CONFIG`）。
_SCENARIO = "moderate_censoring"


def _build_dataframe(n: int, k: int, seed: int):
    df, _beta, _bounds = generate_censored_regression_dataset(
        _SCENARIO, n=n, k=k, seed=seed
    )
    # cluster cov_type 用に行番号ベースの疑似グループ列を付ける。
    return df.with_columns(
        (pl.int_range(pl.len()) % _N_CLUSTERS).alias("cluster_group")
    )


def _lower_bound(y: pl.Series) -> float:
    """左打ち切り閾値。`_SCENARIO` は左打ち切りで、打ち切られた観測は閾値へ厳密に
    セットされ、非打ち切り観測は閾値より真に大きいため、標本最小値が閾値と一致する。
    """
    return float(y.min())


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import Tobit, TobitOptions

    lower = _lower_bound(ctx.df[ctx.y_col])
    if ctx.cov_type == "classical":
        options = TobitOptions(
            lower=lower, upper=None, cov_type="classical", method=ctx.method
        )
    elif ctx.cov_type == "cluster":
        options = TobitOptions(
            lower=lower,
            upper=None,
            cov_type="cluster",
            cluster_col=ctx.cluster_col,
            method=ctx.method,
        )
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return Tobit(ctx.df, y=ctx.y_col, x=ctx.x_cols, options=options).fit()


# engine の quasi-Newton（lbfgs）が newton のこの倍数より遅ければ警告する。
# 実測（n=100,000, k=5, classical）は lbfgs/newton ~3x（run 間で 3.2〜3.6x）
# なので、劣化して初めて発火する余裕を持たせた値（module docstring「method の
# 範囲」参照）。
_QUASI_NEWTON_RATIO_LIMIT = 5.0


def _check_method_ratios(report: dict) -> list[str]:
    """engine の method 軸（lbfgs）が newton の `_QUASI_NEWTON_RATIO_LIMIT` 倍より
    遅ければ警告文字列を返す（`PerfAdapter.check_report`）。

    基準の newton は n 軸の classical・n=method_sweep_n・engine の行を使う
    （`_perf_harness.run_method_sweep` が method 軸を回す条件と同じ）。
    """
    meta = report["_meta"]
    n = meta.get("method_sweep_n")
    if n is None:
        return []
    default_method = meta.get("default_method", "newton")
    rows = report["results"]
    newton = next(
        (
            r
            for r in rows
            if r["axis"] == "n"
            and r["library"] == "engine"
            and r["cov_type"] == "classical"
            and r["n"] == n
            and r["method"] == default_method
        ),
        None,
    )
    if newton is None or newton["time_median_s"] <= 0.0:
        return []
    warnings: list[str] = []
    for r in rows:
        if r["axis"] != "method" or r["library"] != "engine":
            continue
        ratio = r["time_median_s"] / newton["time_median_s"]
        if ratio > _QUASI_NEWTON_RATIO_LIMIT:
            warnings.append(
                f"engine method={r['method']} が newton の {ratio:.1f}x 遅い "
                f"(n={n:,}, k={r['k']}, classical; 想定上限 "
                f"{_QUASI_NEWTON_RATIO_LIMIT:.0f}x)。quasi-Newton 実装の"
                f"パフォーマンス劣化の可能性（#285 参照）。"
            )
    return warnings


TOBIT_ADAPTER = PerfAdapter(
    method="tobit",
    module="performance.compare_tobit",
    # インプロセス計測できるリファレンス実装が無いため engine 単独
    # （module docstring「engine 単独で計測する」参照）。
    libraries=("engine",),
    cov_types=("classical", "cluster"),
    # engine は git ハッシュで足りるため記録するリファレンス版は無い。
    reference_versions=dict,
    build_dataframe=_build_dataframe,
    fit_once=_fit_once_engine,
    cluster_col="cluster_group",
    # classical / cluster とも n=1,000〜100,000。
    n_sweep=(1_000, 10_000, 100_000),
    # classical のみ追加する大標本点。n=1,000,000（seed=42）が Issue #291（大標本
    # Hessian 特異エラー）の再現点で、修正（d797f9b / 5b79ffe）の回帰検知を担う。
    # n=200,000 は n スケーリングのデータ点＋早期警告（seed=42 では #291 を再現
    # しない）。全 cov_type で回すと CI 時間がかさむため classical に絞る。詳細・
    # 限界（単一 seed / 例外のみ捕捉 / リリース単位で発火）は docstring
    # 「## n 軸の大標本点」参照。
    n_sweep_engine_only=(200_000, 1_000_000),
    # method 軸: lbfgs のみ（bfgs は #292 で発散するため除外）。代表点は
    # classical・k=5・n=100,000。既定の newton は n/k スイープに含まれる。
    extra_methods=("lbfgs",),
    # quasi-Newton の劣化ガード（lbfgs/newton 比が 5x 超で job summary に警告）。
    check_report=_check_method_ratios,
)


if __name__ == "__main__":
    run_cli(TOBIT_ADAPTER, doc=__doc__)
