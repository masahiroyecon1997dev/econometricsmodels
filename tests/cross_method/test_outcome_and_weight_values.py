"""被説明変数の値と重みの不正値のエラーが、列を正しく特定することを確かめる。

`OUTCOME_NOT_BINARY`・`OUTCOME_OUTSIDE_CENSORING_BOUNDS`・`WEIGHT_NOT_POSITIVE`は、
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
