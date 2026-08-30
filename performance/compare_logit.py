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

## cov_type の範囲

`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に
従い、代表2点のみ計測する: 最も軽い `classical` と、最も計算コストの重い
`cluster`。Logit/Probit は OLS/WLS と違い HAC を持たず、classical/hc0/cluster を
n=100,000 で軽く実測したところ cluster が最重だった（`docs/spec/
logit-performance-notes.md`「計測方法」）。

**`opg` は計測対象外**: statsmodels の discrete model（`Logit.fit`）は `opg` を
`cov_type` 引数としてネイティブに受け付けず、`score_obs` からの手計算
（`benchmark/nonlinear/references/statsmodels_ref.py` 参照）になる。engine の
ネイティブ OPG との比較は「計測対象の処理範囲を対称に揃える」方針に反するため
除外する。

使用例（リポジトリルートから）:
    # 一括実行（n軸・k軸両方、結果をJSONに保存）
    python -m performance.compare_logit \\
        --output docs/spec/_logit_performance_results.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する
    # （一括実行では `_perf_harness._run_isolated` が自動で設定する）。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_logit \\
        --worker --library engine --cov-type cluster --n 1000 --k 5
"""

from __future__ import annotations

import polars as pl
import statsmodels

from benchmark.nonlinear.datasets import generate_binary_choice_dataset
from performance._perf_harness import FitContext, PerfAdapter, run_cli

# クラスターロバストSE計測用の疑似グループ数（旧 compare_performance.py と同じ）。
_N_CLUSTERS = 50


def _build_dataframe(n: int, k: int, seed: int):
    df, _ = generate_binary_choice_dataset(
        "baseline", link="logit", n=n, k=k, seed=seed
    )
    # cluster cov_type 用に行番号ベースの疑似グループ列を付ける。
    return df.with_columns(
        (pl.int_range(pl.len()) % _N_CLUSTERS).alias("cluster_group")
    )


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import Logit, LogitOptions

    if ctx.cov_type == "classical":
        options = LogitOptions(cov_type="classical")
    elif ctx.cov_type == "cluster":
        options = LogitOptions(cov_type="cluster", cluster_col=ctx.cluster_col)
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return Logit(ctx.df, y=ctx.y_col, x=ctx.x_cols, options=options).fit()


def _fit_once_statsmodels(ctx: FitContext):
    import statsmodels.formula.api as smf

    formula = f"{ctx.y_col} ~ " + " + ".join(ctx.x_cols)
    fit_kwargs: dict = {
        "disp": 0,
        "method": "newton",
        "cov_type": "nonrobust" if ctx.cov_type == "classical" else "cluster",
    }
    if ctx.cov_type == "cluster":
        fit_kwargs["cov_kwds"] = {"groups": ctx.pandas_df[ctx.cluster_col]}
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
    cov_types=("classical", "cluster"),
    reference_versions=lambda: {
        "statsmodels_version": statsmodels.__version__
    },
    build_dataframe=_build_dataframe,
    fit_once=_fit_once,
    cluster_col="cluster_group",
)


if __name__ == "__main__":
    run_cli(LOGIT_ADAPTER, doc=__doc__)
