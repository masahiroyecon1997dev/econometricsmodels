"""FEの実行時間・メモリ使用量を linearmodels と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`FE(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・releaseビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは FE 固有のアダプタ（データセット生成・ライブラリ別 `fit_once`）のみを
定義する。

## リファレンス実装

FE/RE共通のPython主リファレンス（`docs/planning/specs/panel-api-design.md`5.1節・
`benchmark/panel/references/linearmodels_ref.py`と同じ）である
`linearmodels.panel.PanelOLS`と比較する。

## N×T分解（サンプルサイズ軸のスイープ方針）

パネルはサンプルサイズが entity数(N) × 時点数(T) の2軸に分解されるが、ハーネスの
`build_dataframe(n, k, seed)` は単一の `n` しか持たない。本スクリプトは
**T（時点数）を`_N_PERIODS_FIXED=6`に固定し、Nをスイープする**（ミクロパネル—
企業・個人パネルでN大・T小が典型—を想定した設計、`benchmark/panel/datasets.py`の
baselineシナリオの既定値`_DEFAULT_N_PERIODS=6`と同値に揃えている。ユーザー確認
済み・2026-09-21）。`n_entities = n // _N_PERIODS_FIXED`で整数除算するため、実際の
観測数（`n_entities * _N_PERIODS_FIXED`）は要求した`n`と若干ずれる（例:
n=1,000 → n_entities=166 → 実際は996観測）。性能比較の目的はスケーリング傾向の
把握であり、この程度のずれは許容する。

## cov_type の範囲

`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に
従い、代表2点のみ計測する。FEの`classical`/`hc1`/`hc2`/`hc3`/`cluster`/`hac`を
n_entities=16,666・n_periods=6・k=5で実測した結果、`hac`（Driscoll-Kraay）が
最重量（中央値91.6ms、classicalの82.3msに対し+11%）だったため、`classical`と
`hac`を採用する（OLS/WLS/IVと同じ組み合わせ）。DKのバンド幅は時点数`T`ベース
（`engine::panel::fe::resolve_dk_bandwidth`）であり、ハーネスの`ctx.hac_lags`は
総観測数`n`ベース（Newey-West用）で基準が異なるため使わない——
`hac_auto_lag(_N_PERIODS_FIXED)`をモジュール定数として別途計算し、engine・
linearmodels双方のバンド幅指定に使う。

## 2-way固定効果の計測（method軸の流用）

2-way FE（entity+time）はIVのgmm・Logit/Probitのbfgs/lbfgsと同じ`extra_methods`
仕組みを流用し、代表点1つ（cov_type=classical・k=n_sweep_fixed_k・
n=n_sweep[-1]）だけ追加計測する（`default_method="one_way"`,
`extra_methods=("two_way",)`。ユーザー確認済み・2026-09-21）。`FitContext.method`
の意味が他手法の推定アルゴリズム選択（newton等）からFEの固定効果構造選択に
変わる点で既存パターンからの意図的な逸脱だが、「cov_type・n・k軸を共有し代表点
1つだけ追加計測する」という構造自体はIV/Logit/Probitと同型。

## k軸はclassicalのみ計測する（`k_sweep_cov_types`）

DK HAC（`cov_type="hac"`）のバンド幅は時点数`T`ベース（`_N_PERIODS_FIXED=6`・
バンド幅2）だが、k軸スイープの`k=20`（`k_sweep=(5, 20)`は全手法共通の既定値）で
実測すると、F検定用の共分散行列の部分行列がほぼ特異になり
`ComputationError`で失敗することが判明した（`T=6`に対して`k=20`は次元過多——
DKの`S`行列はバンド幅内の時点ペアからの寄与の和で、実効ランクが時点数`T`に
制約されるため）。RE（`compare_re.py`）は同じ`k=20`・`hac`で問題なく成功する
——RE自身のF統計量はFEの`wald_f_test`（部分行列の反転）とは異なる定義
（変換済みyの単純平均を基準にしたSST/SSR比較）を使うため、この特異性の
影響を受けない（`engine/src/panel/CLAUDE.md`「F統計量（Issue #337）」参照）。
FEのみ`k_sweep_cov_types=("classical",)`でk軸のcov_typeをclassicalに絞る
（n軸はk=5固定のため`hac`込みで問題なく計測できる。`_perf_harness.py`の
`PerfAdapter.k_sweep_cov_types`参照）。

## MultiIndex構築は計測区間の外（ハーネス拡張）

`linearmodels.panel.PanelOLS`は呼び出し前に`MultiIndex(entity, time)`の構築が
必須だが、engineはentity/timeをプレーン列として受け取るためこの手順が不要。
実務ではMultiIndex構築もpolars→pandas変換と同じ「データ準備段階で一度だけ行う
処理」であるため、`PerfAdapter.build_pandas_df`（`_perf_harness.py`への拡張、
ユーザー確認済み・2026-09-21）でウォームアップ前に1回だけ構築し、計測ループの
外に置く。

## 計測範囲の対称性

`engine`（`FeEstimator::fit`）は係数・標準誤差と同じ`.fit()`の中でパネル固有R²
（within/between/overall）・F統計量まで常に一括計算する。一方linearmodelsの
`PanelResults`はこれらを遅延評価プロパティにしているため、`_fit_once_linearmodels`
は`.fit()`直後に明示アクセスして確定させる。F統計量は`f_statistic_robust`
（`cov_type`に連動する版）を使う——`benchmark/panel/references/linearmodels_ref.py`
の`run()`と同じ理由（`f_statistic`は常にhomoskedastic固定のF検定で、engineの
`f_statistic`と対応しない）。`aic`/`bic`はlinearmodelsが提供しないため対称性を
取る対象に含めない（同ファイル参照）。

使用例（リポジトリルートから）:
    python -m performance.compare_fe \\
        --output docs/performance/results/fe.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_fe \\
        --worker --library engine --cov-type hac --n 10000 --k 5 --method one_way
"""

from __future__ import annotations

import linearmodels
import pandas as pd
import polars as pl

from benchmark.common import hac_auto_lag
from benchmark.panel.datasets import generate_fe_dataset
from performance._perf_harness import FitContext, PerfAdapter, run_cli

_ENTITY_COL = "entity"
_TIME_COL = "time"

# T（時点数）を固定しNをスイープする（モジュールdocstring「N×T分解」参照）。
# `benchmark/panel/datasets.py`のbaselineシナリオの既定値と同値。
_N_PERIODS_FIXED = 6

# DKバンド幅は時点数Tベース（ハーネスの`ctx.hac_lags`は総観測数nベースで基準が
# 異なるため使わない、モジュールdocstring「cov_typeの範囲」参照）。
_DK_BANDWIDTH = hac_auto_lag(_N_PERIODS_FIXED)


def _build_dataframe(n: int, k: int, seed: int) -> pl.DataFrame:
    n_entities = max(1, n // _N_PERIODS_FIXED)
    df, _ = generate_fe_dataset(
        "baseline",
        n_entities=n_entities,
        n_periods=_N_PERIODS_FIXED,
        k=k,
        seed=seed,
    )
    return df


def _build_pandas_df(df: pl.DataFrame) -> pd.DataFrame:
    """MultiIndex(entity, time)を構築する（計測区間の外、モジュールdocstring参照）。

    `benchmark/panel/references/linearmodels_ref.py`の`_build_panel_index`の
    time指定ありブランチと同じロジック（辞書順=時系列順のカテゴリコード変換）。
    """
    pdf = df.to_pandas()
    time_categories = sorted(pdf[_TIME_COL].unique())
    pdf[_TIME_COL] = pd.Categorical(
        pdf[_TIME_COL], categories=time_categories, ordered=True
    ).codes
    return pdf.set_index([_ENTITY_COL, _TIME_COL])


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import FE, FEOptions

    two_way = ctx.method == "two_way"
    if ctx.cov_type == "classical":
        options = FEOptions(
            cov_type="classical", time=_TIME_COL if two_way else None
        )
    elif ctx.cov_type == "hac":
        options = FEOptions(
            cov_type="hac",
            time=_TIME_COL if two_way else None,
            time_col=_TIME_COL,
            dk_bandwidth=_DK_BANDWIDTH,
        )
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return FE(
        ctx.df, y=ctx.y_col, x=ctx.x_cols, entity=_ENTITY_COL, options=options
    ).fit()


def _fit_once_linearmodels(ctx: FitContext):
    from linearmodels.panel import PanelOLS

    two_way = ctx.method == "two_way"
    mod = PanelOLS(
        ctx.pandas_df[ctx.y_col],
        ctx.pandas_df[ctx.x_cols],
        entity_effects=True,
        time_effects=two_way,
    )
    if ctx.cov_type == "classical":
        lm_cov_type, cov_config = "unadjusted", {"debiased": True}
    elif ctx.cov_type == "hac":
        lm_cov_type, cov_config = (
            "kernel",
            {
                "debiased": True,
                "kernel": "bartlett",
                "bandwidth": _DK_BANDWIDTH,
            },
        )
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")

    res = mod.fit(cov_type=lm_cov_type, **cov_config)
    # engineと計測範囲を揃えるため、遅延評価プロパティを明示的に確定させる
    # （モジュール docstring「計測範囲の対称性」参照）。
    _ = (
        res.params,
        res.std_errors,
        res.tstats,
        res.pvalues,
        res.rsquared_within,
        res.rsquared_between,
        res.rsquared_overall,
        res.f_statistic_robust.stat,
    )
    return res


def _fit_once(ctx: FitContext):
    if ctx.library == "engine":
        return _fit_once_engine(ctx)
    if ctx.library == "linearmodels":
        return _fit_once_linearmodels(ctx)
    raise ValueError(f"unknown library: {ctx.library!r}")


FE_ADAPTER = PerfAdapter(
    method="fe",
    module="performance.compare_fe",
    libraries=("engine", "linearmodels"),
    cov_types=("classical", "hac"),
    reference_versions=lambda: {
        "linearmodels_version": linearmodels.__version__
    },
    build_dataframe=_build_dataframe,
    build_pandas_df=_build_pandas_df,
    fit_once=_fit_once,
    default_method="one_way",
    extra_methods=("two_way",),
    k_sweep_cov_types=("classical",),
)


if __name__ == "__main__":
    run_cli(FE_ADAPTER, doc=__doc__)
