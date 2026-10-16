"""全手法共通の、`data`・`new_data`の型と、新しいデータの列のdtypeのテスト。

`data`の抽出は手法ごとに別の呼び出し箇所を持ち、`predict()`/`augment()`の`new_data`も
OLS/WLS/Logit/Probit/Tobitがそれぞれ持つため、全手法・全経路で確認する。
"""

from __future__ import annotations

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
from _dtype_helpers import (
    UNUSED_COLUMN_DTYPES,
    N,
    dtype_id,
    make_frame,
    unused_column,
)
from _error_messages import escaped
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    Logit,
    Probit,
    Tobit,
    ValidationError,
)


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return make_frame()


# ── data ────────────────────────────────────────────────────────────

# `data`を差し替えて`fit()`する関数（手法ごと）。
FITS = {
    "OLS": lambda d: OLS(d, "y", ["x1"]).fit(),
    "WLS": lambda d: WLS(d, "y", ["x1"], "w").fit(),
    "Logit": lambda d: Logit(d, "y01", ["x1"]).fit(),
    "Probit": lambda d: Probit(d, "y01", ["x1"]).fit(),
    "Tobit": lambda d: Tobit(d, "yc", ["x1"]).fit(),
    "IV": lambda d: IV(d, "y", ["x2"], ["x1"], ["z", "z2"]).fit(),
    "FE": lambda d: FE(d, "y", ["x1"], "entity").fit(),
    "RE": lambda d: RE(d, "y", ["x1"], "entity").fit(),
}


@pytest.mark.parametrize("method", list(FITS))
def test_lazyframe_data_raises_with_collect_hint_for_every_method(
    frame, method
):
    """`LazyFrame`は全手法で、内部実装が漏れたメッセージではなく`.collect()`の案内付きの
    `ValidationError`になる。
    """
    lazy = frame.lazy()

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME_LAZY,
            param_name="data",
            type_name=msgs.fully_qualified_type_name(lazy),
        ),
    ):
        FITS[method](lazy)


@pytest.mark.parametrize("method", list(FITS))
def test_series_data_raises_for_every_method(frame, method):
    series = frame["y"]

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="data",
            type_name=msgs.fully_qualified_type_name(series),
        ),
    ):
        FITS[method](series)


@pytest.mark.parametrize("method", list(FITS))
def test_dataframe_subclass_is_accepted(frame, method):
    """`polars.DataFrame`のサブクラスも本物のDataFrameとして扱い、同じ結果になる。"""

    class MyFrame(pl.DataFrame):
        pass

    subclassed = MyFrame(frame)

    assert FITS[method](subclassed).params == FITS[method](frame).params


# ── new_data（predict / augment） ───────────────────────────────────

# `predict()`/`augment()`を持つ手法。`new_data`には`x1`・`x2`を持たせる。
NEW_DATA_METHODS = {
    "OLS": lambda d: OLS(d, "y", ["x1", "x2"]).fit(),
    "WLS": lambda d: WLS(d, "y", ["x1", "x2"], "w").fit(),
    "Logit": lambda d: Logit(d, "y01", ["x1", "x2"]).fit(),
    "Probit": lambda d: Probit(d, "y01", ["x1", "x2"]).fit(),
    "Tobit": lambda d: Tobit(d, "yc", ["x1", "x2"]).fit(),
}
PATHS = ["predict", "augment"]


def _new_data() -> pl.DataFrame:
    return pl.DataFrame({"x1": [0.5, -0.2, 1.0], "x2": [1.0, 0.0, -0.5]})


def _call(result, path: str, new_data):
    return getattr(result, path)(new_data)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_lazyframe_new_data_raises_with_collect_hint(frame, method, path):
    result = NEW_DATA_METHODS[method](frame)
    lazy = _new_data().lazy()

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME_LAZY,
            param_name="new_data",
            type_name=msgs.fully_qualified_type_name(lazy),
        ),
    ):
        _call(result, path, lazy)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_series_new_data_raises(frame, method, path):
    result = NEW_DATA_METHODS[method](frame)
    series = pl.Series("x1", [0.5, -0.2])

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="new_data",
            type_name=msgs.fully_qualified_type_name(series),
        ),
    ):
        _call(result, path, series)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.UInt8, pl.Float32, pl.Float16, pl.Decimal(18, 0)],
    ids=str,
)
def test_new_data_accepts_numeric_dtypes(frame, method, path, dtype):
    """`new_data`の`x`が整数・浮動小数・Decimalでも、`Float64`と同じ結果になる。"""
    result = NEW_DATA_METHODS[method](frame)
    integers = pl.DataFrame({"x1": [1, 2, 3], "x2": [0, 1, 0]})
    converted = integers.with_columns(pl.all().cast(dtype))

    got = _call(result, path, converted)
    expected = _call(
        result, path, integers.with_columns(pl.all().cast(pl.Float64))
    )

    if isinstance(got, pl.DataFrame):
        assert got.equals(expected)
    else:
        assert list(got) == pytest.approx(list(expected), rel=1e-12)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
@pytest.mark.parametrize("dtype", [pl.Boolean, pl.UInt8], ids=str)
def test_new_data_accepts_boolean_and_dummy_columns(
    frame, method, path, dtype
):
    result = NEW_DATA_METHODS[method](frame)
    new = pl.DataFrame({"x1": [1, 0, 1], "x2": [0, 1, 1]}).with_columns(
        pl.all().cast(dtype)
    )
    as_float = new.with_columns(pl.all().cast(pl.Float64))

    got = _call(result, path, new)
    expected = _call(result, path, as_float)

    if isinstance(got, pl.DataFrame):
        assert got.equals(expected)
    else:
        assert list(got) == pytest.approx(list(expected), rel=1e-12)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
@pytest.mark.parametrize(
    "dtype",
    [pl.Date, pl.Datetime, pl.Struct({"a": pl.Int64}), pl.Categorical],
    ids=str,
)
def test_new_data_rejects_non_numeric_dtype(frame, method, path, dtype):
    result = NEW_DATA_METHODS[method](frame)
    new = _new_data().with_columns(unused_column(dtype).head(3).alias("x1"))
    label = {
        "Date": "Date",
        "Datetime": "Datetime",
        "Categorical": "Categorical",
    }.get(str(dtype), "Struct")

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="x1", dtype=label
        ),
    ):
        _call(result, path, new)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
@pytest.mark.parametrize("dtype", UNUSED_COLUMN_DTYPES, ids=dtype_id)
def test_new_data_with_unused_columns_of_any_dtype(frame, method, path, dtype):
    """`new_data`に推定に使わない任意のdtypeの列があっても、`predict()`は無視し、
    `augment()`はその列を結果のDataFrameにそのまま引き継ぐ（Pythonへ返す変換も通る）。
    """
    result = NEW_DATA_METHODS[method](frame)
    extra = unused_column(dtype).head(3).alias("extra")
    new = _new_data().with_columns(extra)

    got = _call(result, path, new)
    expected = _call(result, path, _new_data())

    if path == "predict":
        assert list(got) == pytest.approx(list(expected), rel=1e-12)
    else:
        assert got.columns[: len(_new_data().columns)] == _new_data().columns
        assert "extra" in got.columns
        assert got["extra"].dtype == new["extra"].dtype
        assert got.drop("extra").equals(expected)


@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_augment_of_the_estimation_data_keeps_unused_columns(frame, method):
    """推定に使ったデータそのままの`augment()`も、推定に使わない列のdtypeを保つ。"""
    result = NEW_DATA_METHODS[method](frame)
    rng = np.random.default_rng(0)
    extra = pl.Series("extra", rng.integers(0, 100, N), dtype=pl.Int64)

    augmented = result.augment(frame.with_columns(extra.cast(pl.Int8)))

    assert augmented["extra"].dtype == pl.Int8
