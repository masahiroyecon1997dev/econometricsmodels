"""Logitの実行時間・メモリ使用量を statsmodels と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`Logit(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・releaseビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは Logit 固有のアダプタ（データセット生成・ライブラリ別 `fit_once`）のみを
定義する。`compare_ols.py` とほぼ同型。

## 計測範囲の対称性（重要）

engine（`engine::nonlinear` の Logit）は係数・標準誤差と同じ呼び出しの中で
対数尤度・切片のみモデルの対数尤度・尤度比統計量・そのp値・McFadden擬似R²・
AIC・BIC までを**常に一括計算**する。一方 statsmodels の `LogitResults` はこれらを
`cached_value`（遅延評価プロパティ）として実装しており、**アクセスして初めて
計算される**。そのため `_fit_once_statsmodels` では `.fit()` 直後にこれらの
プロパティへ明示的にアクセスし、engine と同じ処理範囲で計測する。

## 軸と cov_type の範囲

n 軸（10,000 / 100,000 / 1,000,000、k=5）と k 軸（n=10,000）は既定の標準誤差
（classical）だけを回し、標準誤差の種類は **cov 軸**（n=100,000・k=5、newton）で
全て回す: `classical`・`opg`・`hc0`・`hc1`、cluster の 2×2（G=50 / G=n/10 ×
random / sorted）。Logit/Probit は OLS/WLS と違い HAC を持たない。

**`opg` と `hc1` は engine 単独で測る**: statsmodels の discrete model
（`Logit.fit`）は `opg` を `cov_type` 引数としてネイティブに受け付けず
（`score_obs` からの手計算）、`hc1` は暗黙に `hc0` へフォールバックして実体が
`hc0` と同じになる（どちらも `benchmark/nonlinear/references/statsmodels_ref.py`
参照）。同等のリファレンスが無いため、`hc0` と cluster だけを statsmodels と
比較する。

## method（オプティマイザ）の範囲

engine・statsmodels とも既定は Newton-Raphson で、n/k スイープは newton で回す。
加えて `bfgs`/`lbfgs` を **method 軸**として代表点1つ（classical・
k=5・n=1,000,000）で計測する（`PerfAdapter.extra_methods`）。正確性検証
（`test_logit_reference.py`）も newton を主軸に bfgs/lbfgs は代表ケースのみ、
という絞り方に合わせている。

使用例（リポジトリルートから）:
    # 一括実行（n軸・k軸両方、結果をJSONに保存）
    python -m performance.compare_logit \\
        --output docs/performance/results/logit.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する
    # （一括実行では `_perf_harness._run_isolated` が自動で設定する）。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_logit \\
        --worker --library engine --cov-case cluster_g50_random --n 1000 --k 5
"""

from __future__ import annotations

import statsmodels

from benchmark.nonlinear.datasets import generate_binary_choice_dataset
from performance._perf_harness import (
    CovCase,
    FitContext,
    PerfAdapter,
    cluster_cases,
    run_cli,
)

_ENGINE_COV_TYPES = ("classical", "opg", "hc0", "hc1")

_COV_CASES = (
    CovCase("classical", "classical"),
    # statsmodels の discrete model は opg をネイティブに受け付けず（score_obs
    # からの手計算）、hc1 は暗黙に hc0 へフォールバックして実体が hc0 と同じに
    # なるため（`benchmark/nonlinear/references/statsmodels_ref.py` 参照）、
    # 同等のリファレンスが無いこの2つは engine 単独で測る。
    CovCase("opg", "opg", libraries=("engine",)),
    CovCase("hc0", "hc0"),
    CovCase("hc1", "hc1", libraries=("engine",)),
    *cluster_cases(),
)


def _build_dataframe(n: int, k: int, seed: int):
    df, _ = generate_binary_choice_dataset(
        "baseline", link="logit", n=n, k=k, seed=seed
    )
    return df


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import Logit, LogitOptions

    if ctx.cov_type in _ENGINE_COV_TYPES:
        options = LogitOptions(cov_type=ctx.cov_type, solver=ctx.method)
    elif ctx.cov_type == "cluster":
        options = LogitOptions(
            cov_type="cluster",
            cluster=ctx.cluster,
            solver=ctx.method,
        )
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return Logit(ctx.df, y=ctx.y_col, x=ctx.x_cols, options=options).fit()


def _fit_once_statsmodels(ctx: FitContext):
    import statsmodels.formula.api as smf

    formula = f"{ctx.y_col} ~ " + " + ".join(ctx.x_cols)
    fit_kwargs: dict = {
        "disp": 0,
        "method": ctx.method,
        "cov_type": {"classical": "nonrobust"}.get(ctx.cov_type, ctx.cov_type),
    }
    if ctx.cov_type == "cluster":
        fit_kwargs["cov_kwds"] = {"groups": ctx.pandas_df[ctx.cluster]}
    res = smf.logit(formula, data=ctx.pandas_df).fit(**fit_kwargs)
    # engine と計測範囲を揃えるため、遅延評価プロパティを明示的に確定させる
    # （モジュール docstring「計測範囲の対称性」参照）。
    _ = (
        res.llf,
        res.llnull,
        res.llr,
        res.llr_pvalue,
        res.prsquared,
        res.aic,
        res.bic,
    )
    return res


def _fit_once(ctx: FitContext):
    if ctx.library == "engine":
        return _fit_once_engine(ctx)
    if ctx.library == "statsmodels":
        return _fit_once_statsmodels(ctx)
    raise ValueError(f"unknown library: {ctx.library!r}")


LOGIT_ADAPTER = PerfAdapter(
    method="logit",
    module="performance.compare_logit",
    libraries=("engine", "statsmodels"),
    cov_cases=_COV_CASES,
    reference_versions=lambda: {
        "statsmodels_version": statsmodels.__version__
    },
    build_dataframe=_build_dataframe,
    fit_once=_fit_once,
    # method 軸: bfgs/lbfgs を代表点（classical・k=5・n=1,000,000）で計測する。
    # 既定の newton は n/k スイープに含まれる。
    extra_methods=("bfgs", "lbfgs"),
)


if __name__ == "__main__":
    run_cli(LOGIT_ADAPTER, doc=__doc__)
