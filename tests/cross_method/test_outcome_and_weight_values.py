"""被説明変数の値と重みの不正値のエラーが、列を正しく特定することを確かめる。

`OUTCOME_NOT_BINARY`・`OUTCOME_OUTSIDE_CENSORING_BOUNDS`・`OUTCOME_CONSTANT`・
`OUTCOME_CONSTANT_WITHIN_ENTITY`・`WEIGHT_NOT_POSITIVE`は、
`column`（データの列名）と`argument`（その列を渡した引数名）を別のキーで持つ。
列名が引数名と同じ文字列（`"y"`・`"weight"`）だと取り違えを検出できないので、
`make_frame`の別名の列（`y01`・`yc`・`w`）を使う。重みの非有限値は、`test_numeric_column_values.py`
の`FIT_CASES`が扱わないので、ここで`null`・`NaN`・`±inf`を確かめる。
"""

from __future__ import annotations

import polars as pl
import pytest
from _dtype_helpers import make_frame
from _errors import approx_param, raises_error
from econometricsmodels import (
    FE,
    FEIV,
    IV,
    OLS,
    RE,
    WLS,
    ErrorCode,
    Logit,
    Probit,
    Tobit,
    TobitOptions,
)

# 壊す行。
ROW = 1


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return make_frame()


def _set(
    frame: pl.DataFrame, column: str, value: float | None
) -> pl.DataFrame:
    """`column`の`ROW`行目だけを`value`（`None`ならnull）に置き換える。"""
    return frame.with_columns(
        frame[column].cast(pl.Float64).scatter(ROW, value).alias(column)
    )


def _wls(data: pl.DataFrame) -> None:
    WLS(data, "y", ["x1", "x2"], "w").fit()


# ── WEIGHT_NOT_POSITIVE ──────────────────────────────────────────────


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_weight_names_the_weight_column(
    frame: pl.DataFrame, bad: float
) -> None:
    with raises_error(
        ErrorCode.WEIGHT_NOT_POSITIVE,
        column="w",
        argument="weight",
        data_argument="data",
        row=ROW,
        value=approx_param(bad),
    ):
        _wls(_set(frame, "w", bad))


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf")],
    ids=["NaN", "inf", "-inf"],
)
def test_non_finite_weight_is_a_non_finite_column_error(
    frame: pl.DataFrame, bad: float
) -> None:
    """非有限の重みは、重みの検査（0以下）より先に列の非有限値として報告する。"""
    with raises_error(
        ErrorCode.COLUMN_NON_FINITE,
        column="w",
        argument="weight",
        data_argument="data",
        row=ROW,
        count=1,
        value=approx_param(bad),
    ):
        _wls(_set(frame, "w", bad))


def test_null_weight_is_a_null_column_error(frame: pl.DataFrame) -> None:
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL,
        column="w",
        argument="weight",
        data_argument="data",
        count=1,
    ):
        _wls(_set(frame, "w", None))


# ── OUTCOME_NOT_BINARY ───────────────────────────────────────────────


@pytest.mark.parametrize("estimator", [Logit, Probit])
@pytest.mark.parametrize("bad", [0.5, 2.0, -1.0])
def test_non_binary_outcome_names_the_outcome_column(
    frame: pl.DataFrame, estimator, bad: float
) -> None:
    with raises_error(
        ErrorCode.OUTCOME_NOT_BINARY,
        column="y01",
        argument="y",
        data_argument="data",
        row=ROW,
        value=approx_param(bad),
    ):
        estimator(_set(frame, "y01", bad), "y01", ["x1", "x2"]).fit()


# ── OUTCOME_OUTSIDE_CENSORING_BOUNDS ─────────────────────────────────


def test_outcome_below_the_lower_bound_names_the_outcome_column(
    frame: pl.DataFrame,
) -> None:
    with raises_error(
        ErrorCode.OUTCOME_OUTSIDE_CENSORING_BOUNDS,
        column="yc",
        argument="y",
        data_argument="data",
        row=ROW,
        value=approx_param(-1.0),
        lower=approx_param(0.0),
        upper=None,
    ):
        Tobit(
            _set(frame, "yc", -1.0),
            "yc",
            ["x1", "x2"],
            options=TobitOptions(lower=0.0),
        ).fit()


def test_outcome_above_the_upper_bound_names_the_outcome_column(
    frame: pl.DataFrame,
) -> None:
    with raises_error(
        ErrorCode.OUTCOME_OUTSIDE_CENSORING_BOUNDS,
        column="yc",
        argument="y",
        data_argument="data",
        row=ROW,
        value=approx_param(11.0),
        lower=None,
        upper=approx_param(10.0),
    ):
        Tobit(
            _set(frame, "yc", 11.0),
            "yc",
            ["x1", "x2"],
            options=TobitOptions(lower=None, upper=10.0),
        ).fit()


# ── OUTCOME_CONSTANT ─────────────────────────────────────────────────

# 手法 → (`y`の列名、定数にした`y`で`fit()`する関数)。列名は引数名`y`と違う名前にする。
_X = ["x1", "x2"]
CONSTANT_OUTCOME_FITS = {
    "OLS": ("yc", lambda d: OLS(d, "yc", _X).fit()),
    "WLS": ("yc", lambda d: WLS(d, "yc", _X, "w").fit()),
    "Logit": ("y01", lambda d: Logit(d, "y01", _X).fit()),
    "Probit": ("y01", lambda d: Probit(d, "y01", _X).fit()),
    "Tobit": (
        "yc",
        lambda d: Tobit(d, "yc", _X, options=TobitOptions(lower=0.0)).fit(),
    ),
    "IV": ("yc", lambda d: IV(d, "yc", ["x2"], ["x1"], ["z", "z2"]).fit()),
    "FE": ("yc", lambda d: FE(d, "yc", _X, ["entity"]).fit()),
    "FEIV": (
        "yc",
        lambda d: FEIV(d, "yc", ["x2"], ["x1"], ["z", "z2"], ["entity"]).fit(),
    ),
    "RE": ("yc", lambda d: RE(d, "yc", _X, ["entity"]).fit()),
}


@pytest.mark.parametrize("method", list(CONSTANT_OUTCOME_FITS))
def test_constant_outcome_names_the_outcome_column(
    frame: pl.DataFrame, method: str
) -> None:
    column, fit = CONSTANT_OUTCOME_FITS[method]
    data = frame.with_columns(
        pl.lit(1.0 if column == "y01" else 3.0).alias(column)
    )
    with raises_error(
        ErrorCode.OUTCOME_CONSTANT,
        column=column,
        argument="y",
        data_argument="data",
    ):
        fit(data)


def test_constant_endogenous_variable_names_x_endog_in_the_first_stage(
    frame: pl.DataFrame,
) -> None:
    """第一段階の被説明変数は内生変数なので、`argument`は`y`ではなく`x_endog`。"""
    data = frame.with_columns(pl.lit(2.0).alias("x1"))
    with raises_error(
        ErrorCode.OUTCOME_CONSTANT,
        stage="FIRST_STAGE",
        endog_name="x1",
        column="x1",
        argument="x_endog",
        data_argument="data",
    ):
        IV(data, "y", ["x2"], ["x1"], ["z", "z2"]).fit()


# ── OUTCOME_CONSTANT_WITHIN_ENTITY ───────────────────────────────────

WITHIN_ENTITY_FITS = {
    "FE": lambda d: FE(d, "yc", _X, ["entity"]).fit(),
    "FEIV": lambda d: FEIV(
        d, "yc", ["x2"], ["x1"], ["z", "z2"], ["entity"]
    ).fit(),
    "RE": lambda d: RE(d, "yc", _X, ["entity"]).fit(),
}


@pytest.mark.parametrize("method", list(WITHIN_ENTITY_FITS))
def test_outcome_constant_within_entity_names_the_outcome_column(
    frame: pl.DataFrame, method: str
) -> None:
    """個体内で定数の`y`（個体ごとに値は違う）は、固定効果を除くと分散が残らない。"""
    data = frame.with_columns(pl.col("entity").cast(pl.Float64).alias("yc"))
    with raises_error(
        ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY,
        column="yc",
        argument="y",
        data_argument="data",
    ):
        WITHIN_ENTITY_FITS[method](data)
