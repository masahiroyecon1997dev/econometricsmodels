"""FEIV（固定効果つきの2SLS）の実行時間・ピークRSS を計測するベンチマークスクリプト（engine 単独）。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`FEIV(...).fit()` 全体（Python API 呼び出し、Arrow 変換・PyO3 オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・release ビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは FEIV 固有のアダプタ（データセット生成・`fit_once`）のみを定義する。

## engine 単独で計測する（比較対象を置かない）

Tobit と同じく、他ライブラリとの相対比較ではなく、engine の絶対値とスケーリングの推移を
観察する。

- `linearmodels` に固定効果つきのIVは無い（`PanelOLS`・`AbsorbingLS` は内生変数を取れない）。
  `IV2SLS` にダミー変数を入れる形（LSDV）は、固定効果を除く反復の代わりに N 個のダミーの
  逆行列を作る別のアルゴリズムで、n が大きいと計測できない。pandas の groupby で個体内平均を
  引いてから `IV2SLS` に渡す合成は、計測の大部分が pandas の変換になり、不均衡な2-way・
  3次元では参照側に交互射影を手書きする必要も出て、比較が恣意的になる。
- pyfixest は性能比較からも外した（`docs/performance/ols.md`）。
- fixest（R）は共通ハーネスのインプロセス計測に乗らない。
- 正確性は `tests/panel/test_fe_iv_*.py`（fixest・linearmodels・plm・ivDiag）が担う。

## 計測範囲に診断が含まれる

`FEIV.fit()` は係数・標準誤差に加えて、Sargan（過剰識別のとき）・Wu-Hausman・弱操作変数の
部分F・パネルR²・全体のF検定を**常に一括計算する**。計測した時間はこれらを含む（比較対象が
無いので対称性の調整は要らない）。Cragg-Donald・Kleibergen-Paap・有効Fは事後診断のメソッドで、
`fit()` では計算しないため計測に含まない。

## N×T 分解と `k` の定義

`compare_fe.py` と同じく、**T（時点数）を 6 に固定し N をスイープする**
（`n_entities = n // 6`。実際の観測数は要求した `n` と若干ずれる）。

`k` は**傾き係数の総数**（`x_exog` + `x_endog`）で、`x_exog = k − 内生変数の数` にする。
FE の `k` と同じ規模で比べられ、DK が要求する「時点数 T > 傾き係数の数」
（`T=6 > k=5`）が cov 軸で成り立つ。IV 本体（`compare_iv.py`）は `k = x_exog の数` なので
定義が違う。k 軸の `k=20` は classical だけなので DK の制約に触れない。

## 軸

- **n 軸**（k=5 固定、classical）: n=10,000 / 100,000 / 1,000,000。
- **k 軸**（n=10,000 固定、classical）: k=5・20。
- **cov 軸**（n=100,000・k=5、1-way）: `classical`・`hc1`・`hc2`・`hc3`・`cluster`（個体）・`dk`
  （バンド幅は engine の既定式）。比較対象が無いので全て engine のみ。
- **method 軸**（固定効果の構造と、内生変数・操作変数の数。n=1,000,000・k=5・classical の
  代表点1つずつ。`compare_fe.py` と同じく `extra_methods` の仕組みを流用する）:

  | method | `effects` | 内生変数 / 操作変数 | データ |
  |---|---|---|---|
  | `one_way`（既定） | entity | 1 / 2 | 均衡（T=6） |
  | `two_way` | entity, time | 1 / 2 | 均衡 |
  | `two_way_unbalanced` | entity, time | 1 / 2 | 均衡の各行を確率0.7で残す（個体は最低2行） |
  | `three_way` | entity, time, c | 1 / 2 | 均衡の各行に3つ目の効果`c`（50水準）を割り振る |
  | `endog3` | entity | 3 / 5 | 均衡 |
  | `inst6` | entity | 1 / 6 | 均衡 |

  3次元の HC2/HC3 は cov 軸に `hc2_three_way`・`hc3_three_way` として足す（n=100,000・k=5）。

## 繰り返しの within 変換の見方

FEIV は `y`・`x_exog`・`x_endog`・操作変数の全列に同じ within 変換をかける。不均衡な2-way・
3次元では交互射影の反復が支配的な費用になりうる。変換だけを取り出して測る公開 API は無いので、
単独の計測は置かない。代わりに、変換する列の数を変える軸（k 軸・`endog3`・`inst6`）と、
交互射影が必要な構造（`two_way_unbalanced`・`three_way`。均衡な `two_way` は閉形式）の
`one_way` に対する比から、変換の費用を読む。

## ピークRSS

構造式と第一段階が変換済みの列・残差・吸収した効果を保持するため、n が大きいときの
ピークRSSは FE より大きい。ハーネスが記録するので、大きさの推移を観察する。

## GMM

FE-GMM は未実装。実装したとき、`extra_methods` に推定量（2段階GMM・反復GMM）を足す。

使用例（リポジトリルートから）:
    python -m performance.compare_fe_iv \\
        --output docs/performance/results/fe_iv.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_fe_iv \\
        --worker --library engine --cov-case dk --n 10000 --k 5 --method one_way
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import polars as pl

from benchmark.panel.fe_iv_datasets import generate_fe_iv_perf_dataset
from performance._perf_harness import (
    CovCase,
    FitContext,
    PerfAdapter,
    run_cli,
)

if TYPE_CHECKING:
    from econometricsmodels import FEIVResults

_ENTITY_COL = "entity"
_TIME_COL = "time"
_THIRD_COL = "c"

# T（時点数）を固定しNをスイープする（モジュールdocstring「N×T分解」参照）。
_N_PERIODS_FIXED = 6

# 不均衡パネルは、各行を確率`_KEEP_PROB`で残して作る（個体は最低`_MIN_ROWS`行）。
# データ生成の乱数列と重ならないよう、間引きと3つ目の効果には`seed`から派生した別の
# 乱数列を使う（`compare_fe.py`と同じ）。
_UNBALANCED_STREAM = 1
_THIRD_STREAM = 2
_KEEP_PROB = 0.7
_MIN_ROWS = 2

# 3つ目の効果`c`の水準数。HC2/HC3の密な行列は最大の次元（個体）を除いた水準の合計
# （時点6＋`c`50）で作るため、3,000水準の上限に収まる。
_N_THIRD_LEVELS = 50
_THIRD_EFFECT_SD = 1.0


class _Structure(NamedTuple):
    """method（固定効果の構造・内生変数と操作変数の数）の定義。"""

    effects: tuple[str, ...]
    k_endog: int
    k_instruments: int


_STRUCTURES = {
    "one_way": _Structure((_ENTITY_COL,), 1, 2),
    "two_way": _Structure((_ENTITY_COL, _TIME_COL), 1, 2),
    "two_way_unbalanced": _Structure((_ENTITY_COL, _TIME_COL), 1, 2),
    "three_way": _Structure((_ENTITY_COL, _TIME_COL, _THIRD_COL), 1, 2),
    "endog3": _Structure((_ENTITY_COL,), 3, 5),
    "inst6": _Structure((_ENTITY_COL,), 1, 6),
}

_ENGINE_COV_TYPES = ("classical", "hc1", "hc2", "hc3", "cluster")

_COV_CASES = (
    CovCase("classical", "classical"),
    CovCase("hc1", "hc1"),
    CovCase("hc2", "hc2"),
    CovCase("hc3", "hc3"),
    # 個体クラスター（クラスター列はハーネスが作らず、engine は`effects`の先頭を既定で使う）。
    CovCase("cluster", "cluster"),
    CovCase("dk", "dk"),
    # 3次元の固定効果のHC2/HC3（モジュールdocstring「軸」参照）。
    CovCase("hc2_three_way", "hc2", method="three_way"),
    CovCase("hc3_three_way", "hc3", method="three_way"),
)


def _build(structure: _Structure, n: int, k: int, seed: int) -> pl.DataFrame:
    k_exog = k - structure.k_endog
    if k_exog < 0:
        raise ValueError(
            f"k={k} is smaller than the number of endogenous variables "
            f"({structure.k_endog})"
        )
    return generate_fe_iv_perf_dataset(
        n_entities=max(1, n // _N_PERIODS_FIXED),
        n_periods=_N_PERIODS_FIXED,
        k_exog=k_exog,
        k_endog=structure.k_endog,
        k_instruments=structure.k_instruments,
        seed=seed,
    )


def _build_dataframe(n: int, k: int, seed: int) -> pl.DataFrame:
    return _build(_STRUCTURES["one_way"], n, k, seed)


def _build_unbalanced_dataframe(n: int, k: int, seed: int) -> pl.DataFrame:
    """均衡パネルの行を無作為に間引く（個体は最低`_MIN_ROWS`行を残す）。"""
    df = _build(_STRUCTURES["two_way_unbalanced"], n, k, seed)
    rng = np.random.default_rng([seed, _UNBALANCED_STREAM])
    return (
        df.with_columns(pl.Series("keep", rng.random(df.height) < _KEEP_PROB))
        .filter(
            pl.col("keep")
            | (pl.col("keep").sum().over(_ENTITY_COL) < _MIN_ROWS)
            & (pl.int_range(pl.len()).over(_ENTITY_COL) < _MIN_ROWS)
        )
        .drop("keep")
    )


def _build_three_way_dataframe(n: int, k: int, seed: int) -> pl.DataFrame:
    """均衡パネルの各行に3つ目の効果`c`を無作為に割り振り、`y`へ`c`の効果を足す。"""
    df = _build(_STRUCTURES["three_way"], n, k, seed)
    rng = np.random.default_rng([seed, _THIRD_STREAM])
    levels = rng.integers(0, _N_THIRD_LEVELS, df.height)
    effects = rng.normal(0.0, _THIRD_EFFECT_SD, _N_THIRD_LEVELS)
    return df.with_columns(
        pl.Series(_THIRD_COL, levels),
        (pl.col("y") + pl.Series(effects[levels])).alias("y"),
    )


def _build_endog3_dataframe(n: int, k: int, seed: int) -> pl.DataFrame:
    return _build(_STRUCTURES["endog3"], n, k, seed)


def _build_inst6_dataframe(n: int, k: int, seed: int) -> pl.DataFrame:
    return _build(_STRUCTURES["inst6"], n, k, seed)


def _fit_once_engine(ctx: FitContext) -> FEIVResults:
    from econometricsmodels import FEIV, FEIVOptions

    structure = _STRUCTURES[ctx.method]
    kwargs: dict = {"cov_type": ctx.cov_type}
    if ctx.cov_type == "dk":
        kwargs["dk_time"] = _TIME_COL
    elif ctx.cov_type not in _ENGINE_COV_TYPES:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    # `ctx.x_cols`は`x1..xk`（k=傾き係数の総数）。内生変数の分を除いたものが外生変数。
    x_exog = ctx.x_cols[: len(ctx.x_cols) - structure.k_endog]
    return FEIV(
        ctx.df,
        y=ctx.y_col,
        x_exog=x_exog,
        x_endog=[f"endog{j + 1}" for j in range(structure.k_endog)],
        instruments=[f"z{j + 1}" for j in range(structure.k_instruments)],
        effects=list(structure.effects),
        options=FEIVOptions(**kwargs),
    ).fit()


def _fit_once(ctx: FitContext) -> FEIVResults:
    if ctx.library == "engine":
        return _fit_once_engine(ctx)
    raise ValueError(f"unknown library: {ctx.library!r}")


FE_IV_ADAPTER = PerfAdapter(
    method="fe_iv",
    module="performance.compare_fe_iv",
    # 比較対象が無いため engine 単独（モジュールdocstring「engine 単独で計測する」参照）。
    libraries=("engine",),
    cov_cases=_COV_CASES,
    # engine は git ハッシュで足りるため記録するリファレンス版は無い。
    reference_versions=dict,
    build_dataframe=_build_dataframe,
    fit_once=_fit_once,
    default_method="one_way",
    extra_methods=(
        "two_way",
        "two_way_unbalanced",
        "three_way",
        "endog3",
        "inst6",
    ),
    method_dataframes={
        "two_way_unbalanced": _build_unbalanced_dataframe,
        "three_way": _build_three_way_dataframe,
        "endog3": _build_endog3_dataframe,
        "inst6": _build_inst6_dataframe,
    },
)


# cov 軸の dk は傾き係数の総数 k < T（時点数）が前提（モジュールdocstring参照）。設定変更で
# 崩れたら、計測の途中で engine の ValidationError になる前にここで気づく。
if FE_IV_ADAPTER.n_sweep_fixed_k >= _N_PERIODS_FIXED:
    raise ValueError("cov-sweep dk needs n_sweep_fixed_k < _N_PERIODS_FIXED")


if __name__ == "__main__":
    run_cli(FE_IV_ADAPTER, doc=__doc__)
