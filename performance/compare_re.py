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

## cov_type の範囲

FEと同じプロセスで実測選定した（`.claude/rules/testing-policy.md`「パフォーマンス
比較（ベンチマーク）の方法論」）。n_entities=16,666・n_periods=6・k=5で
`classical`/`hc1`/`hc2`/`hc3`/`cluster`/`hac`を実測した結果、`hac`が最重量
（中央値318.5ms、classicalの212.0msに対し+50%）だったため、FEと同じく
`classical`と`hac`を採用する。

## `RandomEffects`の切片（`build_pandas_df`での定数列追加）

`linearmodels.RandomEffects`は明示的な定数列が無いと切片を推定しないため、
`_build_pandas_df`でMultiIndex構築に加え`pdf["const"] = 1.0`を追加する
（`benchmark/panel/references/linearmodels_ref.py`の`run_re()`と同じ理由）。
この定数列追加も実務上は一度きりのデータ準備であり、MultiIndex構築と同じ理由で
計測区間の外に置く。

## 既知の限界: `cov_type`間でハウスマン内部FEの構造も変わる

`REOptions.time`は「HACの時系列順序」と「ハウスマン検定用の内部FE呼び出しの
1-way/2-way選択（`Some`なら2-way、`None`なら1-way）」を兼ねる
（`engine_pybind/src/panel/re.rs`モジュールdoc参照）。そのため本スクリプトの
`_fit_once_engine`は`cov_type="hac"`のときのみ`time=_TIME_COL`を渡すことになり、
`classical`と`hac`の計測差には「cov_type自体の計算コスト差」に加え「内部
ハウスマン用FEが1-way→2-wayに変わることによる追加コスト」が混入する（RE自身の
設計上不可避な交絡で、回避策は無い。詳細は`docs/performance/re.md`「既知の
限界」）。

2-way軸は無い（RE自体がv1で2-wayをスコープ外にしているため、`extra_methods=()`）。

## 計測範囲の対称性

`engine`（`ReEstimator::fit`）は係数・標準誤差と同じ`.fit()`の中でパネル固有R²・
F統計量まで常に一括計算する。一方linearmodelsの`PanelResults`は遅延評価
プロパティのため、`_fit_once_linearmodels`は`.fit()`直後に明示アクセスして
確定させる。F統計量は`f_statistic`（cov_type非依存のhomoskedastic固定）を使う
——FEの`f_statistic_robust`とは異なり、`engine::panel::re::ReEstimator`の
F統計量自体が`cov_type`に連動しない独自定義のため（`benchmark/panel/references/
linearmodels_ref.py`の`run_re()`と同じ理由）。`aic`/`bic`はlinearmodelsが
提供しないため対称性を取る対象に含めない。

使用例（リポジトリルートから）:
    python -m performance.compare_re \\
        --output docs/performance/results/re.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_re \\
        --worker --library engine --cov-type hac --n 10000 --k 5
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

# T（時点数）を固定しNをスイープする（`compare_fe.py`と同じ方針・同じ値）。
_N_PERIODS_FIXED = 6

# DKバンド幅は時点数Tベース（`compare_fe.py`と同じ理由）。
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

    if ctx.cov_type == "classical":
        options = REOptions(cov_type="classical")
    elif ctx.cov_type == "hac":
        options = REOptions(
            cov_type="hac", time=_TIME_COL, dk_bandwidth=_DK_BANDWIDTH
        )
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return RE(
        ctx.df, y=ctx.y_col, x=ctx.x_cols, entity=_ENTITY_COL, options=options
    ).fit()


def _fit_once_linearmodels(ctx: FitContext):
    from linearmodels.panel import RandomEffects

    mod = RandomEffects(
        ctx.pandas_df[ctx.y_col], ctx.pandas_df[["const", *ctx.x_cols]]
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
    cov_types=("classical", "hac"),
    reference_versions=lambda: {
        "linearmodels_version": linearmodels.__version__
    },
    build_dataframe=_build_dataframe,
    build_pandas_df=_build_pandas_df,
    fit_once=_fit_once,
)


if __name__ == "__main__":
    run_cli(RE_ADAPTER, doc=__doc__)
