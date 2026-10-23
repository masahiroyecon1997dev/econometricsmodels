"""列エラーの`params`のうち、どの引数の列か(`argument`)と許可dtypeの一覧(`allowed`)。

`argument`はアプリが利用者に「どこでエラーが出たか」を示すための値で、呼び出し側が
正しい引数名と列の役割を抽出関数に渡していないと誤る。ここでは、他のテストが個別には
固定していない組み合わせ(IVの`x_exog`/`x_endog`、役割ごとの`allowed`)を確かめる。
"""

from __future__ import annotations

import polars as pl
import pytest
from _dtype_helpers import make_frame
from _errors import raises_error
from econometricsmodels import (
    FE,
    IV,
    OLS,
    ErrorCode,
    FEOptions,
    OLSOptions,
)

NUMERIC_ALLOWED = ["integer", "float", "Boolean", "Decimal"]
TIME_ORDER_ALLOWED = ["integer", "float", "Decimal", "Date", "Datetime"]
IDENTITY_ALLOWED = [
    "integer",
    "float",
    "String",
    "Categorical",
    "Enum",
    "Boolean",
    "Date",
    "Datetime",
]
TIME_ALLOWED = [
    "integer",
    "float",
    "String",
    "Categorical",
    "Enum",
    "Date",
    "Datetime",
]


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return make_frame()


def _iv(data: pl.DataFrame) -> None:
    IV(data, "y", ["x2"], ["x1"], ["z", "z2"]).fit()


@pytest.mark.parametrize(
    ("column", "argument"),
    [("x2", "x_exog"), ("x1", "x_endog"), ("z", "instruments"), ("y", "y")],
)
def test_iv_column_errors_name_the_role_of_the_column(
    frame: pl.DataFrame, column: str, argument: str
) -> None:
    broken = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL,
        column=column,
        argument=argument,
        data_argument="data",
    ):
        _iv(broken)


@pytest.mark.parametrize(
    ("column", "argument"),
    [("x2", "x_exog"), ("x1", "x_endog"), ("z2", "instruments")],
)
def test_iv_missing_column_names_the_role_of_the_column(
    frame: pl.DataFrame, column: str, argument: str
) -> None:
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column=column,
        argument=argument,
        data_argument="data",
    ):
        _iv(frame.drop(column))


def test_numeric_role_allows_numbers_and_booleans(frame: pl.DataFrame) -> None:
    broken = frame.with_columns(pl.lit("a").alias("y"))
    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column="y",
        argument="y",
        dtype="String",
        allowed=NUMERIC_ALLOWED,
    ):
        OLS(broken, "y", ["x1"]).fit()


def test_hac_time_role_allows_orderable_dtypes(frame: pl.DataFrame) -> None:
    broken = frame.with_columns(pl.col("t").cast(pl.String).alias("t"))
    options = OLSOptions(cov_type="hac", hac_time="t")
    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column="t",
        argument="hac_time",
        dtype="String",
        allowed=TIME_ORDER_ALLOWED,
    ):
        OLS(broken, "y", ["x1"], options).fit()


def test_group_identity_role_rejects_time_of_day(frame: pl.DataFrame) -> None:
    broken = frame.with_columns(pl.lit(None).cast(pl.Time).alias("cluster"))
    # `Time`型はキーとして使えないdtype(全値が欠損でもdtypeの検査が先に起きる)。
    options = OLSOptions(cov_type="cluster", cluster="cluster")
    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column="cluster",
        argument="cluster",
        dtype="Time",
        allowed=IDENTITY_ALLOWED,
    ):
        OLS(broken, "y", ["x1"], options).fit()


def test_time_period_role_does_not_allow_booleans(frame: pl.DataFrame) -> None:
    broken = frame.with_columns((pl.col("t") > 2).alias("t"))
    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column="t",
        argument="time",
        dtype="Boolean",
        allowed=TIME_ALLOWED,
    ):
        FE(broken, "y", ["x1"], "entity", FEOptions(time="t")).fit()


def test_dk_time_role_is_named_dk_time(frame: pl.DataFrame) -> None:
    broken = frame.with_columns((pl.col("t") > 2).alias("t"))
    options = FEOptions(cov_type="dk", dk_time="t", dk_bandwidth=1)
    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column="t",
        argument="dk_time",
        allowed=TIME_ALLOWED,
    ):
        FE(broken, "y", ["x1"], "entity", options).fit()
