"""IVの実行時間・メモリ使用量を linearmodels と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`IV(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・releaseビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは IV 固有のアダプタ（データセット生成・ライブラリ別 `fit_once`）のみを
定義する。

## リファレンス実装

公開ページ（`docs/guide/verification.md`）の primary reference に従い linearmodels のみと
比較する（`benchmark/iv/references/linearmodels_ref.py` と同じ主リファレンス）。
2SLS は `linearmodels.iv.IV2SLS`、GMM は `linearmodels.iv.IVGMM` に対応させる
（engine cov_type ↔ linearmodels の対応は `linearmodels_ref.py` の `_COV_TYPE_MAP`
と同じ）。

## 軸と method（2sls / gmm）の範囲

n 軸（10,000 / 100,000 / 1,000,000、k=5）と k 軸（n=10,000）は既定 method の
**2SLS**・既定の標準誤差（classical）だけを回す。GMM は **method 軸**として代表点
1つ（classical, k=5, n=1,000,000）で計測し、標準誤差・重み行列の違いは cov 軸で見る。
GMM × hac（HAC の重み行列、または HAC の標準誤差）は linearmodels 側の `IVGMM` +
kernel が病的に遅く（n=100,000 で約40秒、engine の600倍以上）計測が非現実的なため
対象外とする（engine 側の問題ではない。`docs/performance/iv.md`「既知の限界」）。

## cov 軸（n=100,000・k=5）

2SLS: classical・hc0・hc1・hac、cluster の 2×2（G=50 / G=n/10 × random / sorted）
を linearmodels と比較する。`hc2`/`hc3` は linearmodels 側に対応実装が無いため
（`linearmodels_ref.py` docstring 参照）engine 単独で測る。

GMM（linearmodels `IVGMM`）の代表 2 ケースも cov 軸に載せる（どちらも `variant`
で区別する）:

- `gmm_cluster_weight`: 重み行列も標準誤差もクラスター（G=50, random）の 2 ステップ GMM
- `gmm_iterated`: 反復 GMM（robust の重み行列・hc1 の標準誤差。engine と
  linearmodels で上限 100 回・許容誤差 1e-6 を同じ数値で指定する。収束判定の尺度は
  異なるが、baseline で反復回数が一致することを確認した）。classical の重み行列では
  等分散の下で 2 回で収束してしまい 2 ステップ GMM と変わらないため、反復が実際に
  回る robust にする（baseline で両ライブラリとも 3 回で収束することを確認済み）

linearmodels の `IVGMM` は重み行列の既定が `robust` で、engine の既定
（`gmm_weight_type="classical"`）と異なる。method 軸の GMM も含め、重み行列を
明示して揃える。

## 計測範囲の対称性

engine（`engine::iv`）は係数・標準誤差と同じ `.fit()` の中で、R²・調整済みR²・
F統計量・過剰識別検定（Sargan / Hansen J）・弱操作変数F統計量・Wu-Hausman検定・
第一段階回帰まで**常に一括計算**する。一方 linearmodels の `IVResults` は
これらを遅延評価プロパティにしており、特に `first_stage.diagnostics` は
アクセス時に**第一段階回帰をフル再fitする**（OLS の `rsquared` や Logit の
`llnull` と同じ位置づけ）。`_fit_once_linearmodels` は `.fit()` 直後に
`params`/`std_errors`/`tstats`/`pvalues`/`rsquared`/`rsquared_adj`/
`f_statistic`/`first_stage.diagnostics`（2SLS は加えて `sargan`・`wu_hausman()`、
GMM は `j_stat`）へ明示アクセスし、engine と同じ処理範囲で計測する。

使用例（リポジトリルートから）:
    python -m performance.compare_iv \\
        --output docs/performance/results/iv.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_iv \\
        --worker --library engine --cov-case hac --n 1000 --k 5
"""

from __future__ import annotations

import linearmodels

from benchmark.iv.datasets import generate_iv_dataset
from performance._perf_harness import (
    CLUSTER_N_GROUPS,
    CovCase,
    FitContext,
    PerfAdapter,
    cluster_cases,
    run_cli,
)

# baseline DGP を過剰識別（k_instruments > k_endog）で回し、engine が常に計算する
# 過剰識別検定（Sargan / Hansen J）を計測範囲に確実に含める。x_exog の本数だけを
# n/k スイープの `k` で振る（列は y, x1..xk, endog1, z1, z2）。
_ENDOG_COLS = ["endog1"]
_INSTRUMENT_COLS = ["z1", "z2"]


# engine の cov_type と linearmodels の (cov_type, debiased) の対応
# （`benchmark/iv/references/linearmodels_ref.py` の `_COV_TYPE_MAP` と同じ。
# hac は kernel の設定が要るため `_fit_once_linearmodels` で別に扱う）。
_LINEARMODELS_COV_TYPES = {
    "classical": ("unadjusted", True),
    "hc0": ("robust", False),
    "hc1": ("robust", True),
    "cluster": ("clustered", True),
}
_ENGINE_COV_TYPES = ("classical", "hc0", "hc1", "hc2", "hc3")

# GMM の cov 軸ケースを区別する `CovCase.variant`。
_VARIANT_CLUSTER_WEIGHT = "cluster_weight"
_VARIANT_ITERATED = "iterated"

# 反復 GMM の上限回数と許容誤差。両ライブラリに同じ数値を明示する。収束判定の
# 尺度は異なる（engine は係数ごとの相対・絶対混合変化、linearmodels は共分散で
# 正規化した二次形式）ため同じ停止条件にはならないが、baseline では両者とも
# 3 回で収束することを確認している。
_GMM_MAX_ITER = 100
_GMM_TOL = 1e-6

_COV_CASES = (
    CovCase("classical", "classical"),
    CovCase("hc0", "hc0"),
    CovCase("hc1", "hc1"),
    # linearmodels に対応する HC2/HC3 が無いため engine 単独。
    CovCase("hc2", "hc2", libraries=("engine",)),
    CovCase("hc3", "hc3", libraries=("engine",)),
    CovCase("hac", "hac"),
    *cluster_cases(),
    CovCase(
        "gmm_cluster_weight",
        "cluster",
        n_clusters=CLUSTER_N_GROUPS,
        cluster_layout="random",
        method="gmm",
        variant=_VARIANT_CLUSTER_WEIGHT,
    ),
    CovCase(
        "gmm_iterated",
        "hc1",
        method="gmm",
        variant=_VARIANT_ITERATED,
    ),
)


def _build_dataframe(n: int, k: int, seed: int):
    df, _ = generate_iv_dataset(
        "baseline",
        n=n,
        k_exog=k,
        k_endog=len(_ENDOG_COLS),
        k_instruments=len(_INSTRUMENT_COLS),
        seed=seed,
    )
    return df


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import IV, IVOptions

    kwargs: dict = {"estimator": ctx.method, "cov_type": ctx.cov_type}
    if ctx.cov_type == "hac":
        kwargs.update(hac_lags=ctx.hac_lags, hac_time=ctx.hac_time)
    elif ctx.cov_type == "cluster":
        kwargs.update(cluster=ctx.cluster)
    elif ctx.cov_type not in _ENGINE_COV_TYPES:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    if ctx.variant == _VARIANT_CLUSTER_WEIGHT:
        kwargs.update(gmm_weight_type="cluster")
    elif ctx.variant == _VARIANT_ITERATED:
        kwargs.update(
            gmm_type="iterated",
            gmm_weight_type="robust",
            gmm_max_iter=_GMM_MAX_ITER,
            gmm_tol=_GMM_TOL,
        )
    elif ctx.variant is not None:
        raise ValueError(f"unknown variant: {ctx.variant!r}")
    options = IVOptions(**kwargs)
    return IV(
        ctx.df,
        y=ctx.y_col,
        x_exog=ctx.x_cols,
        x_endog=_ENDOG_COLS,
        instruments=_INSTRUMENT_COLS,
        options=options,
    ).fit()


def _fit_once_linearmodels(ctx: FitContext):
    from linearmodels.iv import IV2SLS, IVGMM

    formula = (
        f"{ctx.y_col} ~ 1 + "
        + " + ".join(ctx.x_cols)
        + f" + [{' + '.join(_ENDOG_COLS)} ~ {' + '.join(_INSTRUMENT_COLS)}]"
    )
    if ctx.cov_type == "hac":
        cov_type = "kernel"
        cov_config = {
            "debiased": False,
            "kernel": "bartlett",
            "bandwidth": ctx.hac_lags,
        }
    elif ctx.cov_type in _LINEARMODELS_COV_TYPES:
        cov_type, debiased = _LINEARMODELS_COV_TYPES[ctx.cov_type]
        cov_config = {"debiased": debiased}
        if ctx.cov_type == "cluster":
            cov_config["clusters"] = ctx.pandas_df[ctx.cluster]
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")

    if ctx.method == "2sls":
        res = IV2SLS.from_formula(formula, ctx.pandas_df).fit(
            cov_type=cov_type, **cov_config
        )
    else:
        # 重み行列を engine の既定（classical）に揃える。linearmodels の既定は
        # robust（モジュール docstring 参照）。
        weight_type, weight_config, fit_kwargs = "unadjusted", {}, {}
        if ctx.variant == _VARIANT_CLUSTER_WEIGHT:
            weight_type = "clustered"
            weight_config = {"clusters": ctx.pandas_df[ctx.cluster]}
        elif ctx.variant == _VARIANT_ITERATED:
            weight_type = "robust"
            fit_kwargs = {"iter_limit": _GMM_MAX_ITER, "tol": _GMM_TOL}
        elif ctx.variant is not None:
            raise ValueError(f"unknown variant: {ctx.variant!r}")
        res = IVGMM.from_formula(
            formula,
            ctx.pandas_df,
            weight_type=weight_type,
            **weight_config,
        ).fit(cov_type=cov_type, **fit_kwargs, **cov_config)
    # engine と計測範囲を揃えるため、遅延評価プロパティを明示的に確定させる
    # （モジュール docstring「計測範囲の対称性」参照）。
    _ = (
        res.params,
        res.std_errors,
        res.tstats,
        res.pvalues,
        res.rsquared,
        res.rsquared_adj,
        res.f_statistic.stat,
        res.first_stage.diagnostics,
    )
    if ctx.method == "2sls":
        _ = (res.sargan.stat, res.sargan.pval, res.wu_hausman().stat)
    else:
        _ = res.j_stat.stat
    return res


def _fit_once(ctx: FitContext):
    if ctx.library == "engine":
        return _fit_once_engine(ctx)
    if ctx.library == "linearmodels":
        return _fit_once_linearmodels(ctx)
    raise ValueError(f"unknown library: {ctx.library!r}")


IV_ADAPTER = PerfAdapter(
    method="iv",
    module="performance.compare_iv",
    libraries=("engine", "linearmodels"),
    cov_cases=_COV_CASES,
    reference_versions=lambda: {
        "linearmodels_version": linearmodels.__version__
    },
    build_dataframe=_build_dataframe,
    fit_once=_fit_once,
    default_method="2sls",
    extra_methods=("gmm",),
)


if __name__ == "__main__":
    run_cli(IV_ADAPTER, doc=__doc__)
