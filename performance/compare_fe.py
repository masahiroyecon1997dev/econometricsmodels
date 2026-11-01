"""FEの実行時間・メモリ使用量を linearmodels と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`FE(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・releaseビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは FE 固有のアダプタ（データセット生成・ライブラリ別 `fit_once`）のみを
定義する。

## リファレンス実装

FE/RE共通のPython主リファレンス（`docs/spec/panel-common.md`5.1節・
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

## 軸と cov_type の範囲

n 軸（10,000 / 100,000 / 1,000,000、k=5）と k 軸（n=10,000）は既定の標準誤差
（classical）だけを回し、標準誤差の種類は **cov 軸**（n=100,000・k=5）で全て回す:
`classical`・`hc1`・`hc2`・`hc3`・`cluster`・`dk`。`hc2`/`hc3` は linearmodels
（`PanelOLS`）に対応が無いため engine 単独で測る。`hc0` は FE が対応しない。

`cluster` は entity クラスター（G = N = n / T、データは entity 順）の1ケース
だけで、`cluster_cases()` の G・レイアウトの 2×2 は使わない。パネルの
クラスターは entity で決まり、G を変えると N×T の構成が変わってしまうため。
linearmodels 側は `cluster_entity=True`。

DKのバンド幅は時点数`T`ベース
（`engine::panel::fe::resolve_dk_bandwidth`）であり、ハーネスの`ctx.hac_lags`は
総観測数`n`ベース（Newey-West用）で基準が異なるため使わない——
`hac_auto_lag(_N_PERIODS_FIXED)`をモジュール定数として別途計算し、engine・
linearmodels双方のバンド幅指定に使う。

## 2-way固定効果の計測（method軸の流用）

2-way FE（entity+time）はIVのgmm・Logit/Probitのbfgs/lbfgsと同じ`extra_methods`
仕組みを流用し、代表点1つ（既定の標準誤差・k=n_sweep_fixed_k・
n=n_sweep[-1]）だけ追加計測する（`default_method="one_way"`,
`extra_methods=("two_way",)`。ユーザー確認済み・2026-09-21）。`FitContext.method`
の意味が他手法の推定アルゴリズム選択（newton等）からFEの固定効果構造選択に
変わる点で既存パターンからの意図的な逸脱だが、「cov_type・n・k軸を共有し代表点
1つだけ追加計測する」という構造自体はIV/Logit/Probitと同型。

## dk は k=5 < T=6 の境界に依存する

k軸スイープの`k=20`（`k_sweep=(5, 20)`は全手法共通の既定値）は
`T=6`（`_N_PERIODS_FIXED`）に対して次元過多で、`cov_type="dk"`だと engineは
`ValidationError`（`PanelError::InsufficientDkPeriodsForInference`）で`fit()`を
拒否する——DKの`S`行列は時点ごとのスコアの外積の和でrankが`T-1`以下のため、
F検定の`k×k`部分行列が`k >= T`で構造的に特異になる。REもハウスマン検定（`X̃`の
`k`個の同時Wald検定）が同じ制約を受ける（`compare_re.py`）。k軸は既定の
classicalだけを回すので影響しないが、cov軸の`dk`は`k=5 < T=6`の境界ちょうどに
依存しており、`n_sweep_fixed_k`を6以上にするか`_N_PERIODS_FIXED`を5以下にすると
cov軸のdkが同じ`ValidationError`で失敗する。

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

## 非対称な点: cluster と dk の小標本補正

FE の `cluster` と `dk` は、engine が fixest/Stata 型の小標本補正と自由度を
意図的に使う（cluster は `G/(G-1)·(n-1)/(n-K)` と t(G-1)、dk は
`T/(T-1)·(n-1)/(n-K)` と t(T-1)）ため、linearmodels（`df_resid`）と標準誤差が
一致しない（n=3,000 で cluster は 1e-3 程度、dk は `sqrt(T/(T-1))` ≒ 9.5%）。
差は補正のスカラーで計算量は同等なので両ライブラリで測るが、数値の一致は
前提にしない。`classical` と `hc1` は機械精度で一致する。

使用例（リポジトリルートから）:
    python -m performance.compare_fe \\
        --output docs/performance/results/fe.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_fe \\
        --worker --library engine --cov-case dk --n 10000 --k 5 --method one_way
"""

from __future__ import annotations

import linearmodels
import pandas as pd
import polars as pl

from benchmark.common import hac_auto_lag
from benchmark.panel.datasets import generate_fe_dataset
from performance._perf_harness import (
    CovCase,
    FitContext,
    PerfAdapter,
    run_cli,
)

_ENTITY_COL = "entity"
_TIME_COL = "time"

# T（時点数）を固定しNをスイープする（モジュールdocstring「N×T分解」参照）。
# `benchmark/panel/datasets.py`のbaselineシナリオの既定値と同値。
_N_PERIODS_FIXED = 6

# DKバンド幅は時点数Tベース（ハーネスの`ctx.hac_lags`は総観測数nベースで基準が
# 異なるため使わない、モジュールdocstring「cov_typeの範囲」参照）。
_DK_BANDWIDTH = hac_auto_lag(_N_PERIODS_FIXED)


_ENGINE_COV_TYPES = ("classical", "hc1", "hc2", "hc3", "cluster")

_COV_CASES = (
    CovCase("classical", "classical"),
    CovCase("hc1", "hc1"),
    # linearmodels に対応する HC2/HC3 が無いため engine 単独。
    CovCase("hc2", "hc2", libraries=("engine",)),
    CovCase("hc3", "hc3", libraries=("engine",)),
    # entity クラスター（モジュール docstring 参照）。クラスター列はハーネスが
    # 作らず、engine は entity 列を既定で使う。
    CovCase("cluster", "cluster"),
    CovCase("dk", "dk"),
)


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
    effects = [_ENTITY_COL, _TIME_COL] if two_way else [_ENTITY_COL]
    kwargs: dict = {"cov_type": ctx.cov_type}
    if ctx.cov_type == "dk":
        kwargs.update(dk_time=_TIME_COL, dk_bandwidth=_DK_BANDWIDTH)
    elif ctx.cov_type not in _ENGINE_COV_TYPES:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    options = FEOptions(**kwargs)
    return FE(
        ctx.df, y=ctx.y_col, x=ctx.x_cols, effects=effects, options=options
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
    elif ctx.cov_type == "hc1":
        lm_cov_type, cov_config = "robust", {"debiased": True}
    elif ctx.cov_type == "cluster":
        # engine の既定（entity クラスター）に対応する。
        lm_cov_type, cov_config = (
            "clustered",
            {"debiased": True, "cluster_entity": True},
        )
    elif ctx.cov_type == "dk":
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
    cov_cases=_COV_CASES,
    reference_versions=lambda: {
        "linearmodels_version": linearmodels.__version__
    },
    build_dataframe=_build_dataframe,
    build_pandas_df=_build_pandas_df,
    fit_once=_fit_once,
    default_method="one_way",
    extra_methods=("two_way",),
)


# cov 軸の dk は k < T（時点数）が前提（モジュール docstring 参照）。設定変更で
# 崩れたら、計測の途中で engine の ValidationError になる前にここで気づく。
if FE_ADAPTER.n_sweep_fixed_k >= _N_PERIODS_FIXED:
    raise ValueError("cov-sweep dk needs n_sweep_fixed_k < _N_PERIODS_FIXED")


if __name__ == "__main__":
    run_cli(FE_ADAPTER, doc=__doc__)
