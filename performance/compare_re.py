"""REの実行時間・メモリ使用量を linearmodels と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`RE(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス・N×T分解・MultiIndex構築の計測外化については`performance/
compare_fe.py`のモジュールdocstringを参照（本ファイルはFEと共通の設計方針を
踏襲し、差分のみ以下に記す）。

## リファレンス実装

FE/RE共通のPython主リファレンス`linearmodels.panel.RandomEffects`と比較する
（`benchmark/panel/references/linearmodels_ref.py`と同じ）。REはFE用の合成
データセットをそのまま再利用する既存方針（同ファイルのモジュールdocstring
「RE専用の合成データセット・凍結コードは追加していない」）に倣い、
`benchmark.panel.datasets.generate_fe_dataset`を直接使う。

## 軸と cov_type の範囲

FEと同じ構成（`compare_fe.py`「軸と cov_type の範囲」参照）: n 軸・k 軸は既定の
標準誤差（classical）だけを回し、cov 軸（n=100,000・k=5）で`classical`・`hc1`・
`hc2`・`hc3`・`cluster`（entity クラスター）・`dk`を全て回す。`hc2`/`hc3` は
linearmodels（`RandomEffects`）に対応が無いため engine 単独で測る。

## `RandomEffects`の切片（`build_pandas_df`での定数列追加）

`linearmodels.RandomEffects`は明示的な定数列が無いと切片を推定しないため、
`_build_pandas_df`でMultiIndex構築に加え`pdf["const"] = 1.0`を追加する
（`benchmark/panel/references/linearmodels_ref.py`の`run_re()`と同じ理由）。
この定数列追加も実務上は一度きりのデータ準備であり、MultiIndex構築と同じ理由で
計測区間の外に置く。

## `dk_time`の扱い

`REOptions.dk_time`は`cov_type="dk"`のHAC時点列専用（必須）で、ハウスマン検定（常に1-way
比較）には影響しない。`_fit_once_engine`は`cov_type="dk"`のときのみ
`dk_time=_TIME_COL`を渡す（`dk`以外で渡すと`ValidationError`）。

2-way軸は無い（RE自体がv1で2-wayをスコープ外にしているため、`extra_methods=()`）。

## dk は k=5 < T=6 の境界に依存する

FE（`compare_fe.py`の同名節）と同じ理由。RE本体の標準誤差は`dk`でも`k=20`・
`T=6`で計算できるが、`fit()`が常に計算するハウスマン検定は`X̃`の`k`個を同時Wald
検定するため、DK共分散のrank（`T-1`以下）が`k`に足りず、engineは
`ValidationError`（`PanelError::InsufficientDkPeriodsForInference`）で`fit()`を
拒否する。k軸は既定のclassicalだけを回すので影響しないが、cov軸のdkは
`k=5 < T=6`の境界ちょうどに依存している。

## 計測範囲の対称性

`engine`（`ReEstimator::fit`）は係数・標準誤差と同じ`.fit()`の中でパネル固有R²・
F統計量まで常に一括計算する。一方linearmodelsの`PanelResults`は遅延評価
プロパティのため、`_fit_once_linearmodels`は`.fit()`直後に明示アクセスして
確定させる。F統計量は`f_statistic`（cov_type非依存のhomoskedastic固定）を使う
——FEの`f_statistic_robust`とは異なり、`engine::panel::re::ReEstimator`の
F統計量自体が`cov_type`に連動しない独自定義のため（`benchmark/panel/references/
linearmodels_ref.py`の`run_re()`と同じ理由）。`aic`/`bic`はlinearmodelsが
提供しないため対称性を取る対象に含めない。

## 非対称な点: ハウスマン検定

engine の `RE.fit()` は常にハウスマン検定（内部で FE の within 推定と補助回帰を
実行）まで計算するが、`linearmodels.RandomEffects` には対応する処理が無く、本
スクリプトの `_fit_once_linearmodels` もそこを触らない（遅延評価ではなく、そもそも
同等の計算が提供されない）。そのため RE の速度比は engine に不利な方向（過小評価）に
偏る。揃えられない非対称なので、結果の読み方として `docs/performance/re.md`・
`docs/guide/performance.md` にも記す。

## 非対称な点: cluster と dk の小標本補正

RE の `cluster` と `dk` は、engine が fixest/Stata 型の小標本補正と自由度を
意図的に使う（cluster は `G/(G-1)·(n-1)/(n-K)` と t(G-1)、dk は
`T/(T-1)·(n-1)/(n-K)` と t(T-1)）ため、linearmodels（`df_resid`）と標準誤差が
一致しない（n=3,000 で cluster は 1e-3 程度、dk は `sqrt(T/(T-1))` ≒ 9.5%）。
差は補正のスカラーで計算量は同等なので両ライブラリで測るが、数値の一致は
前提にしない。`classical` と `hc1` は機械精度で一致する。

使用例（リポジトリルートから）:
    python -m performance.compare_re \\
        --output docs/performance/results/re.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_re \\
        --worker --library engine --cov-case dk --n 10000 --k 5
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

# T（時点数）を固定しNをスイープする（`compare_fe.py`と同じ方針・同じ値）。
_N_PERIODS_FIXED = 6

# DKバンド幅は時点数Tベース（`compare_fe.py`と同じ理由）。
_DK_BANDWIDTH = hac_auto_lag(_N_PERIODS_FIXED)


_ENGINE_COV_TYPES = ("classical", "hc1", "hc2", "hc3", "cluster")

_COV_CASES = (
    CovCase("classical", "classical"),
    CovCase("hc1", "hc1"),
    # linearmodels に対応する HC2/HC3 が無いため engine 単独。
    CovCase("hc2", "hc2", libraries=("engine",)),
    CovCase("hc3", "hc3", libraries=("engine",)),
    # entity クラスター（`compare_fe.py` docstring 参照）。
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
    """MultiIndex(entity, time)構築＋RE用切片列の追加（計測区間の外、モジュール
    docstring「`RandomEffects`の切片」参照）。
    """
    pdf = df.to_pandas()
    time_categories = sorted(pdf[_TIME_COL].unique())
    pdf[_TIME_COL] = pd.Categorical(
        pdf[_TIME_COL], categories=time_categories, ordered=True
    ).codes
    pdf["const"] = 1.0
    return pdf.set_index([_ENTITY_COL, _TIME_COL])


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import RE, REOptions

    if ctx.cov_type == "dk":
        options = REOptions(
            cov_type="dk", dk_time=_TIME_COL, dk_bandwidth=_DK_BANDWIDTH
        )
    elif ctx.cov_type in _ENGINE_COV_TYPES:
        options = REOptions(cov_type=ctx.cov_type)
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return RE(
        ctx.df,
        y=ctx.y_col,
        x=ctx.x_cols,
        effects=[_ENTITY_COL],
        options=options,
    ).fit()


def _fit_once_linearmodels(ctx: FitContext):
    from linearmodels.panel import RandomEffects

    mod = RandomEffects(
        ctx.pandas_df[ctx.y_col], ctx.pandas_df[["const", *ctx.x_cols]]
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
        res.f_statistic.stat,
    )
    return res


def _fit_once(ctx: FitContext):
    if ctx.library == "engine":
        return _fit_once_engine(ctx)
    if ctx.library == "linearmodels":
        return _fit_once_linearmodels(ctx)
    raise ValueError(f"unknown library: {ctx.library!r}")


RE_ADAPTER = PerfAdapter(
    method="re",
    module="performance.compare_re",
    libraries=("engine", "linearmodels"),
    cov_cases=_COV_CASES,
    reference_versions=lambda: {
        "linearmodels_version": linearmodels.__version__
    },
    build_dataframe=_build_dataframe,
    build_pandas_df=_build_pandas_df,
    fit_once=_fit_once,
)


# cov 軸の dk は k < T（時点数）が前提（モジュール docstring 参照）。設定変更で
# 崩れたら、計測の途中で engine の ValidationError になる前にここで気づく。
if RE_ADAPTER.n_sweep_fixed_k >= _N_PERIODS_FIXED:
    raise ValueError("cov-sweep dk needs n_sweep_fixed_k < _N_PERIODS_FIXED")


if __name__ == "__main__":
    run_cli(RE_ADAPTER, doc=__doc__)
