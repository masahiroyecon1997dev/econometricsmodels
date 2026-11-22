"""実データのクラスターロバスト推定（OLS・WLS）を、有理数の厳密値と比べるための共通部品。

`test_ols_reference.py`と`test_wls_reference.py`の「厳密値（有理数）との照合」が使う。
列のスケールが大きく違い、互いに高相関な実データ（`expersq`・`tenursq`は他の説明変数より桁が
大きく、`exper`の多項式の各項は互いに高相関）で、クラスターロバストの共分散を
`V = R⁻¹M̃R⁻ᵀ`（直交座標）で求める経路が精度を保つことを確かめる。`(X'X)⁻¹`を明示的に作って
サンドイッチを組む経路ではずれる設計（`card_poly`）を含むので、その経路の再発を防ぐ回帰ガードに
なる。厳密値は`_exact_wald.exact_inference`（`fractions`）で求める。

WLSの厳密値は、重みを`w = s²`（`s`は整数）にして、`s`倍した変数のOLSに置き換えて求める
（`√w = s`が厳密に表せる重みだけが扱える）。重み自体は任意の値でよいので、`s = 1 + ダミー変数`にする。
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from _exact_wald import ExactInference, exact_inference

from benchmark.iv.datasets import with_card_region

REGION = "region"
SQRT_WEIGHT = "sqrt_weight"
WEIGHT = "weight"


@dataclass(frozen=True)
class ExactClusterCase:
    """実データ1件の設定。`weight_dummy`は、WLSの`√w = 1 + weight_dummy`に使う0/1の列。"""

    dataset: str
    y: str
    x: list[str]
    weight_dummy: str
    # 列のべき乗で作る説明変数（新しい列名 → (元の列, 指数)）。
    powers: dict[str, tuple[str, int]]
    # クラスター数G。傾きの数qより大きくないと、ロバストWaldが構造的に特異になる。
    n_clusters: int

    def __post_init__(self) -> None:
        assert self.n_clusters > len(self.x), "G > q が必要"


EXACT_CLUSTER_CASES: dict[str, ExactClusterCase] = {
    # 9地域・傾き6つ。`expersq`は`black`・`smsa`・`south`（0/1）の約1e3倍の尺度。
    "card": ExactClusterCase(
        dataset="card",
        y="lwage",
        x=["educ", "exper", "expersq", "black", "smsa", "south"],
        weight_dummy="smsa",
        powers={},
        n_clusters=9,
    ),
    # 9地域・傾き8つ。経験年数の4次多項式（`exper`〜`exper4`は桁が約1e4まで開き、互いに高相関）。
    # `X'X`を明示的に反転してサンドイッチを組む経路では、この設計で全体Fが相対1e-8〜1e-7ずれる
    # （同じ式を倍精度のnumpyで組んだ実測。`card`・`wage1`ではその経路でも2e-12以下で、
    # 経路の差が見えない）。
    "card_poly": ExactClusterCase(
        dataset="card",
        y="lwage",
        x=[
            "educ",
            "exper",
            "expersq",
            "expercu",
            "exper4",
            "black",
            "smsa",
            "south",
        ],
        weight_dummy="smsa",
        powers={"expercu": ("exper", 3), "exper4": ("exper", 4)},
        n_clusters=9,
    ),
    # 4地域・傾き3つ。`expersq`・`tenursq`は`educ`の数十〜数百倍。
    "wage1": ExactClusterCase(
        dataset="wage1",
        y="lwage",
        x=["educ", "expersq", "tenursq"],
        weight_dummy="female",
        powers={},
        n_clusters=4,
    ),
}


def load_exact_cluster_case(
    load_wooldridge, name: str
) -> tuple[pl.DataFrame, ExactClusterCase]:
    """ケースのデータを返す。地域列`region`・WLS用の`sqrt_weight`（整数）・`weight`
    （その平方）・べき乗で作った列を足してある。
    """
    case = EXACT_CLUSTER_CASES[name]
    df = load_wooldridge(case.dataset)
    if case.dataset == "card":
        df = with_card_region(df)
    else:
        # wage1: 地域ダミー（基準カテゴリはnortheast）から4グループの実カテゴリ列を作る。
        df = df.with_columns(
            pl.when(pl.col("northcen") == 1)
            .then(pl.lit("northcen"))
            .when(pl.col("south") == 1)
            .then(pl.lit("south"))
            .when(pl.col("west") == 1)
            .then(pl.lit("west"))
            .otherwise(pl.lit("northeast"))
            .alias(REGION)
        )
    df = df.with_columns(
        *[
            (pl.col(src) ** power).alias(new)
            for new, (src, power) in case.powers.items()
        ],
        (1 + pl.col(case.weight_dummy)).cast(pl.Int64).alias(SQRT_WEIGHT),
    ).with_columns((pl.col(SQRT_WEIGHT) ** 2).alias(WEIGHT))
    return df, case


def exact_cluster_inference(
    df: pl.DataFrame, case: ExactClusterCase, *, weighted: bool
) -> tuple[ExactInference, float]:
    """係数・標準誤差・全体のF統計量（`W / q`）の厳密値。`weighted=True`はWLS
    （`√w`倍した変数のOLS）。
    """
    rows = df.select([case.y, *case.x, SQRT_WEIGHT, REGION]).rows()
    x = [[1.0, *(float(v) for v in row[1 : 1 + len(case.x)])] for row in rows]
    y = [float(row[0]) for row in rows]
    groups = [row[-1] for row in rows]
    if weighted:
        scale = [int(row[-2]) for row in rows]
        x = [[s * v for v in xi] for s, xi in zip(scale, x)]
        y = [s * v for s, v in zip(scale, y)]
    exact = exact_inference(x, y, cov_type="cluster", groups=groups)
    return exact, exact.wald / len(case.x)
