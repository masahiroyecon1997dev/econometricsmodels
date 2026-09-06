"""WLSの実行時間・メモリ使用量を statsmodels と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`WLS(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・releaseビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは WLS 固有のアダプタ（データセット生成・ライブラリ別 `fit_once`）のみを
定義する。`compare_ols.py` とほぼ同型で、違いは重み列を engine の `weight` 引数と
statsmodels の `weights=` に渡す点のみ。

## 重みの定義

engine（`WLS`、`docs/spec/wls-spec.md`「API引数」）と statsmodels（`smf.wls` の
`weights=`）はいずれも重みを analytic weight（分散の逆数に比例、正規化不要）として
扱うため、`generate_linear_dataset` が返す `weight` 列をそのまま両方に渡せる。

## 計測範囲の対称性（重要）

engine（`engine::linear` の WLS）は係数・標準誤差と同じ呼び出しの中で
R²・調整済みR²・対数尤度・AIC・BIC・F統計量・F検定のp値までを**常に一括計算**
する。一方 statsmodels の `RegressionResults` はこれらを `cached_value`（遅延評価
プロパティ）として実装しており、**アクセスして初めて計算される**。そのため
`_fit_once_statsmodels` では `.fit()` 直後にこれらのプロパティへ明示的にアクセスし、
engine と同じ「フルセットの適合度統計量込み」で計測する。

## cov_type の範囲

`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に
従い、代表2点のみ計測する: 最も軽い `classical` と、最も計算コストの重い `hac`
（Newey-West、三重ループ）。WLS で classical/hc1/cluster/hac を n=100,000 で軽く
実測し、OLS と同じく HAC が最重であることを確認した
（`docs/performance/wls.md`「計測方法」）。

使用例（リポジトリルートから）:
    # 一括実行（n軸・k軸両方、結果をJSONに保存）
    python -m performance.compare_wls \\
        --output docs/performance/results/wls.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する
    # （一括実行では `_perf_harness._run_isolated` が自動で設定する）。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_wls \\
        --worker --library engine --cov-type hac --n 1000 --k 5
"""

from __future__ import annotations

import statsmodels

from benchmark.linear.datasets import generate_linear_dataset
from performance._perf_harness import FitContext, PerfAdapter, run_cli


def _build_dataframe(n: int, k: int, seed: int):
    # `generate_linear_dataset` は列 y, x1..xk, weight を返す（weight は
    # 0.5〜1.5 の一様乱数）。WLS 用の追加加工は不要。
    df, _ = generate_linear_dataset("baseline", n=n, k=k, seed=seed)
    return df


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import WLS, OLSOptions

    if ctx.cov_type == "classical":
        options = OLSOptions(cov_type="classical")
    elif ctx.cov_type == "hac":
        options = OLSOptions(cov_type="hac", hac_lags=ctx.hac_lags)
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return WLS(
        ctx.df,
        y=ctx.y_col,
        x=ctx.x_cols,
        weight=ctx.weight_col,
        options=options,
    ).fit()


def _fit_once_statsmodels(ctx: FitContext):
    import statsmodels.formula.api as smf

    formula = f"{ctx.y_col} ~ " + " + ".join(ctx.x_cols)
    sm_cov_type = {"classical": "nonrobust", "hac": "hac"}[ctx.cov_type]
    fit_kwargs: dict = {"cov_type": sm_cov_type, "use_t": True}
    if ctx.cov_type == "hac":
        fit_kwargs["cov_kwds"] = {"maxlags": ctx.hac_lags}
    res = smf.wls(
        formula,
        data=ctx.pandas_df,
        weights=ctx.pandas_df[ctx.weight_col],
    ).fit(**fit_kwargs)
    # engine と計測範囲を揃えるため、遅延評価プロパティを明示的に確定させる
    # （モジュール docstring「計測範囲の対称性」参照）。
    _ = (
        res.rsquared,
        res.rsquared_adj,
        res.fvalue,
        res.f_pvalue,
        res.aic,
        res.bic,
        res.llf,
    )
    return res


def _fit_once(ctx: FitContext):
    if ctx.library == "engine":
        return _fit_once_engine(ctx)
    if ctx.library == "statsmodels":
        return _fit_once_statsmodels(ctx)
    raise ValueError(f"unknown library: {ctx.library!r}")


WLS_ADAPTER = PerfAdapter(
    method="wls",
    module="performance.compare_wls",
    libraries=("engine", "statsmodels"),
    cov_types=("classical", "hac"),
    reference_versions=lambda: {
        "statsmodels_version": statsmodels.__version__
    },
    build_dataframe=_build_dataframe,
    fit_once=_fit_once,
    weight_col="weight",
)


if __name__ == "__main__":
    run_cli(WLS_ADAPTER, doc=__doc__)
