"""IVの実行時間・メモリ使用量を linearmodels と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`IV(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・releaseビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは IV 固有のアダプタ（データセット生成・ライブラリ別 `fit_once`）のみを
定義する。

## リファレンス実装

README「Verification accuracy」の primary reference に従い linearmodels のみと
比較する（`benchmark/iv/references/linearmodels_ref.py` と同じ主リファレンス）。
2SLS は `linearmodels.iv.IV2SLS`、GMM は `linearmodels.iv.IVGMM` に対応させる
（engine cov_type ↔ linearmodels の対応は `linearmodels_ref.py` の `_COV_TYPE_MAP`
と同じ）。

## method（2sls / gmm）の範囲

n/k スイープは既定 method の **2SLS** で回す。GMM は **method 軸**として代表点
1つ（cov_type=classical, k=5, n=1,000,000）でのみ計測する。GMM × hac は
linearmodels 側の `IVGMM` + kernel が病的に遅く（n=100,000 で約40秒、engine の
600倍以上）大規模計測が非現実的なため対象外とする（engine 側の問題ではない。
`docs/performance/iv.md`「既知の限界」）。正確性検証
（`test_iv_reference.py`）も 2SLS 主軸・GMM は代表シナリオのみ、という絞り方に
合わせる。

## cov_type の範囲

`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に
従い、代表2点のみ計測する: 最も軽い `classical` と、最も計算コストの重い `hac`
（Newey-West、bartlett kernel）。IV で classical/hc1/cluster/hac を n=100,000 で
軽く実測し、OLS/WLS と同じく HAC が最重であることを確認した
（`docs/performance/iv.md`「計測方法」）。`hc2`/`hc3` は linearmodels 側に対応実装が
無いため（`linearmodels_ref.py` docstring 参照）、性能比較でも扱わない。

## 計測範囲の対称性（Issue #98）

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
        --worker --library engine --cov-type hac --n 1000 --k 5
"""

from __future__ import annotations

import linearmodels

from benchmark.iv.datasets import generate_iv_dataset
from performance._perf_harness import FitContext, PerfAdapter, run_cli

# baseline DGP を過剰識別（k_instruments > k_endog）で回し、engine が常に計算する
# 過剰識別検定（Sargan / Hansen J）を計測範囲に確実に含める。x_exog の本数だけを
# n/k スイープの `k` で振る（列は y, x1..xk, endog1, z1, z2）。
_ENDOG_COLS = ["endog1"]
_INSTRUMENT_COLS = ["z1", "z2"]


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
    from econometricsmodels import IV, IvOptions

    if ctx.cov_type == "classical":
        options = IvOptions(method=ctx.method, cov_type="classical")
    elif ctx.cov_type == "hac":
        options = IvOptions(
            method=ctx.method, cov_type="hac", hac_lags=ctx.hac_lags
        )
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
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
    model_cls = IV2SLS if ctx.method == "2sls" else IVGMM
    if ctx.cov_type == "classical":
        cov_type, cov_config = "unadjusted", {"debiased": True}
    elif ctx.cov_type == "hac":
        cov_config = {
            "debiased": False,
            "kernel": "bartlett",
            "bandwidth": ctx.hac_lags,
        }
        cov_type = "kernel"
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")

    res = model_cls.from_formula(formula, ctx.pandas_df).fit(
        cov_type=cov_type, **cov_config
    )
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
    cov_types=("classical", "hac"),
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
