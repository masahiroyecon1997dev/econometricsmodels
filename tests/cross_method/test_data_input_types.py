"""全手法共通の、`data`・`new_data`の型と、新しいデータの列のdtypeのテスト。

`data`の抽出は手法ごとに別の呼び出し箇所を持ち、`predict()`/`augment()`の`new_data`も
OLS/WLS/Logit/Probit/Tobitがそれぞれ持つため、全手法・全経路で確認する。
"""

from __future__ import annotations

import _error_expectations as msgs
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
from _error_expectations import escaped
from _errors import raises_error
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    ErrorCode,
    FEOptions,
    Logit,
    LogitOptions,
    OLSOptions,
    Probit,
    REOptions,
    Tobit,
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
    `TypeError`になる。
    """
    lazy = frame.lazy()

    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME_LAZY,
            param_name="data",
        ),
    ):
        FITS[method](lazy)


@pytest.mark.parametrize("method", list(FITS))
def test_series_data_raises_for_every_method(frame, method):
    series = frame["y"]

    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="data",
            type_name=msgs.fully_qualified_type_name(series),
        ),
    ):
        FITS[method](series)


@pytest.mark.parametrize("method", list(FITS))
def test_pandas_data_raises_with_conversion_hint_for_every_method(
    frame, method
):
    bad = frame.to_pandas()

    with pytest.raises(
        TypeError,
        match=escaped(msgs.NOT_A_POLARS_DATAFRAME_PANDAS, param_name="data"),
    ):
        FITS[method](bad)


def test_pandas_dataframe_subclass_gets_the_conversion_hint(frame):
    import pandas as pd

    class MyPandas(pd.DataFrame):
        pass

    with pytest.raises(
        TypeError,
        match=escaped(msgs.NOT_A_POLARS_DATAFRAME_PANDAS, param_name="data"),
    ):
        OLS(MyPandas(frame.to_pandas()), "y", ["x1"]).fit()


@pytest.mark.parametrize("method", list(FITS))
@pytest.mark.parametrize(
    "bad",
    [None, {"y": [1.0, 2.0]}, [1.0, 2.0], np.array([1.0, 2.0]), 3.5],
    ids=["none", "dict", "list", "ndarray", "float"],
)
def test_other_non_polars_data_raises_for_every_method(method, bad):
    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="data",
            type_name=msgs.fully_qualified_type_name(bad),
        ),
    ):
        FITS[method](bad)


def test_data_type_is_checked_before_the_column_arguments(frame):
    """`data`が`polars.DataFrame`でなければ、`y`の型の誤りより先にそれを報告する。"""
    with pytest.raises(TypeError, match="data must be a polars.DataFrame"):
        OLS(frame.lazy(), 1, ["x1"]).fit()


class _UnreadableFrame(pl.DataFrame):
    """本物の`polars.DataFrame`だが、列の読み取りに失敗する（内部の不整合を模す）。"""

    def get_column(self, name):
        raise RuntimeError("boom")


@pytest.mark.parametrize("method", list(FITS))
def test_unreadable_real_dataframe_data_is_an_internal_error(frame, method):
    """本物の`polars.DataFrame`の読み取り失敗は、利用者の入力の誤りではなく
    `INTERNAL_ERROR`になる（`TypeError`ではない）。
    """
    broken = _UnreadableFrame(frame)

    with raises_error(ErrorCode.INTERNAL_ERROR) as info:
        FITS[method](broken)

    assert "data" in info.value.params["detail"]


@pytest.mark.parametrize("method", list(FITS))
def test_unused_uint128_column_does_not_affect_the_fit(frame, method):
    """推定に指名されない列は読まないので、取り込めないdtype（`UInt128`）の列があっても
    `fit()`は成功し、結果は同じになる。"""
    extra = pl.Series("extra", np.arange(N) % 100, dtype=pl.Int64).cast(
        pl.UInt128
    )

    got = FITS[method](frame.with_columns(extra))

    assert got.params == FITS[method](frame).params


@pytest.mark.parametrize("method", list(FITS))
def test_uint128_column_is_rejected_where_it_is_named(frame, method):
    """指名された列が`UInt128`なら、取り込む前にdtype名入りの`ValidationError`にする
    （Rustのpanicにしない）。`y`の列を置き換えて確かめる。"""
    y = {"Logit": "y01", "Probit": "y01", "Tobit": "yc"}.get(method, "y")
    bad = frame.with_columns(pl.col(y).round(0).abs().cast(pl.UInt128))

    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column=y,
        dtype="UInt128",
        argument="y",
        data_argument="data",
    ):
        FITS[method](bad)


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
        TypeError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME_LAZY,
            param_name="new_data",
        ),
    ):
        _call(result, path, lazy)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_series_new_data_raises(frame, method, path):
    result = NEW_DATA_METHODS[method](frame)
    series = pl.Series("x1", [0.5, -0.2])

    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="new_data",
            type_name=msgs.fully_qualified_type_name(series),
        ),
    ):
        _call(result, path, series)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_pandas_new_data_raises_with_conversion_hint(frame, method, path):
    result = NEW_DATA_METHODS[method](frame)
    bad = _new_data().to_pandas()

    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME_PANDAS, param_name="new_data"
        ),
    ):
        _call(result, path, bad)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_unreadable_real_dataframe_new_data_is_an_internal_error(
    frame, method, path
):
    result = NEW_DATA_METHODS[method](frame)

    with raises_error(ErrorCode.INTERNAL_ERROR) as info:
        _call(result, path, _UnreadableFrame(_new_data()))

    assert "new_data" in info.value.params["detail"]


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
    [
        pl.Date,
        pl.Datetime,
        pl.Struct({"a": pl.Int64}),
        pl.Categorical,
        pl.UInt128,
        pl.List(pl.UInt128),
    ],
    ids=str,
)
def test_new_data_rejects_non_numeric_dtype(frame, method, path, dtype):
    result = NEW_DATA_METHODS[method](frame)
    new = _new_data().with_columns(unused_column(dtype).head(3).alias("x1"))
    label = {
        "Date": "Date",
        "Datetime": "Datetime",
        "Categorical": "Categorical",
        "UInt128": "UInt128",
        "List(UInt128)": "List",
    }.get(str(dtype), "Struct")

    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column="x1",
        dtype=label,
        argument="x",
        data_argument="new_data",
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
def test_augment_keeps_an_uint128_column_that_the_model_does_not_read(
    frame, method
):
    """`UInt128`の列は推定にも`predict()`にも使わず、`augment()`の出力に元のまま残る
    （学習データそのままの`augment()`と、`new_data`を渡した`augment()`の両方）。"""
    extra = pl.Series("extra", np.arange(N) % 100, dtype=pl.Int64).cast(
        pl.UInt128
    )
    data = frame.with_columns(extra)
    result = NEW_DATA_METHODS[method](data)

    from_training = result.augment()
    from_new = result.augment(_new_data().with_columns(extra.head(3)))

    assert from_training["extra"].dtype == pl.UInt128
    assert from_training["extra"].equals(data["extra"])
    assert from_new["extra"].dtype == pl.UInt128
    assert from_new.height == 3


@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_augment_of_the_estimation_data_keeps_unused_columns(frame, method):
    """推定に使ったデータそのままの`augment()`も、推定に使わない列のdtypeを保つ。"""
    result = NEW_DATA_METHODS[method](frame)
    rng = np.random.default_rng(0)
    extra = pl.Series("extra", rng.integers(0, 100, N), dtype=pl.Int64)

    augmented = result.augment(frame.with_columns(extra.cast(pl.Int8)))

    assert augmented["extra"].dtype == pl.Int8


# ── 結果が保持する元のデータ（augment・診断検定）─────────────────────


def _with_uint128(frame: pl.DataFrame) -> pl.DataFrame:
    """`frame`に、読まれない`UInt128`列`u`と、重複しない時間順の列`tt`を足す。"""
    extra = pl.Series("u", np.arange(N) % 100, dtype=pl.Int64).cast(pl.UInt128)
    return frame.with_columns(extra, pl.Series("tt", np.arange(N)))


def _with_tt(frame: pl.DataFrame) -> pl.DataFrame:
    return _with_uint128(frame).drop("u")


@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_augment_appends_the_prediction_after_the_original_columns(
    frame, method
):
    """予測値の列は元の列の後ろに付き、値は`predict()`と一致し、他の列は元のまま。"""
    data = _with_uint128(frame)
    result = NEW_DATA_METHODS[method](data)

    augmented = result.augment()

    added = [c for c in augmented.columns if c not in data.columns]
    assert augmented.columns == [*data.columns, *added]
    assert len(added) == 1
    assert augmented[added[0]].to_list() == pytest.approx(
        result.predict(), rel=1e-12
    )
    assert augmented.select(data.columns).equals(data)


def test_tobit_augment_names_the_column_after_each_target(frame):
    data = _with_uint128(frame)
    result = NEW_DATA_METHODS["Tobit"](data)

    for target in ["expected_latent", "prob_uncensored"]:
        augmented = result.augment(target=target)
        assert augmented.columns[-1] == f"predicted_{target}"
        assert augmented["u"].dtype == pl.UInt128


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_new_data_with_zero_rows(frame, method, path):
    result = NEW_DATA_METHODS[method](frame)
    empty = pl.DataFrame(
        {
            "x1": pl.Series([], dtype=pl.Float64),
            "x2": pl.Series([], dtype=pl.Float64),
        }
    )

    got = _call(result, path, empty)

    assert len(got) == 0 if path == "predict" else got.height == 0


@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_augment_collision_with_an_unreadable_column(frame, method):
    """`predicted`系の列が既にあれば、その列が読めないdtypeでも黙って上書きしない。"""
    result = NEW_DATA_METHODS[method](frame)
    name = result.augment().columns[-1]
    colliding = frame.with_columns(
        pl.Series(name, np.arange(N) % 100, dtype=pl.Int64).cast(pl.UInt128)
    )
    refit = NEW_DATA_METHODS[method](colliding)

    with raises_error(ErrorCode.AUGMENT_COLUMN_EXISTS, column=name):
        refit.augment()


@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_a_result_keeps_working_after_the_data_variable_is_deleted(
    frame, method
):
    import gc

    data = _with_uint128(frame)
    result = NEW_DATA_METHODS[method](data)
    expected = result.augment()

    del data
    gc.collect()

    assert result.augment().equals(expected)


@pytest.mark.parametrize("method", list(NEW_DATA_METHODS))
def test_a_result_is_not_changed_by_editing_the_data_in_place(frame, method):
    """結果は`fit()`時点のデータを持つ。後から元のデータをその場で書き換えても
    （列の差し替え・削除・行の追加）、`augment()`は変わらない。"""
    data = _with_uint128(frame)
    result = NEW_DATA_METHODS[method](data)
    expected = result.augment()

    data.replace_column(
        data.get_column_index("x1"), pl.Series("x1", [0.0] * N)
    )
    data.drop_in_place("x2")
    data.vstack(data, in_place=True)

    assert result.augment().equals(expected)


def test_ols_diagnostics_are_not_changed_by_editing_the_data_in_place(frame):
    data = _with_uint128(frame)
    result = OLS(data, "y", ["x1", "x2"]).fit()
    white = result.white_test().statistic
    breusch_pagan = result.breusch_pagan_test(["x2"]).statistic

    data.replace_column(
        data.get_column_index("x2"), pl.Series("x2", [1.0] * N)
    )
    data.drop_in_place("x1")

    assert result.white_test().statistic == white
    assert result.breusch_pagan_test(["x2"]).statistic == breusch_pagan


# ── OLSの診断検定と取り込めないdtype ─────────────────────────────────


def test_ols_diagnostics_ignore_an_unused_uint128_column(frame):
    data = _with_uint128(frame)
    plain = OLS(_with_tt(frame), "y", ["x1", "x2"]).fit()
    result = OLS(data, "y", ["x1", "x2"]).fit()

    assert result.white_test().statistic == plain.white_test().statistic
    assert (
        result.breusch_pagan_test(["z"]).statistic
        == plain.breusch_pagan_test(["z"]).statistic
    )
    assert (
        result.breusch_godfrey_test("tt", 2).statistic
        == plain.breusch_godfrey_test("tt", 2).statistic
    )


def test_breusch_pagan_variables_naming_an_uint128_column_are_rejected(frame):
    result = OLS(_with_uint128(frame), "y", ["x1", "x2"]).fit()

    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column="u",
        dtype="UInt128",
        argument="variables",
        data_argument="data",
    ):
        result.breusch_pagan_test(["z", "u"])


def test_breusch_godfrey_time_naming_an_uint128_column_is_rejected(frame):
    result = OLS(_with_uint128(frame), "y", ["x1", "x2"]).fit()

    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column="u",
        dtype="UInt128",
        argument="time",
        data_argument="data",
    ):
        result.breusch_godfrey_test("u", 2)


# ── 指名される列の組み立て（オプションの列と取り込めない列の共存）───────


def test_option_columns_are_read_when_an_unreadable_column_is_present(frame):
    """`cluster`・`hac_time`・`time`・`dk_time`が指名されると読まれ、無関係の`UInt128`列は
    読まれない（結果は`UInt128`列が無いときと同じ）。"""
    data = _with_uint128(frame)
    cases = {
        "ols-cluster": lambda d: OLS(
            d,
            "y",
            ["x1"],
            options=OLSOptions(cov_type="cluster", cluster="cluster"),
        ).fit(),
        "ols-hac": lambda d: OLS(
            d, "y", ["x1"], options=OLSOptions(cov_type="hac", hac_time="tt")
        ).fit(),
        "fe-dk": lambda d: FE(
            d,
            "y",
            ["x1"],
            "entity",
            options=FEOptions(cov_type="dk", dk_time="t"),
        ).fit(),
        "re-twoway": lambda d: RE(
            d, "y", ["x1"], "entity", options=REOptions(time="t")
        ).fit(),
        "logit-cluster": lambda d: Logit(
            d,
            "y01",
            ["x1"],
            options=LogitOptions(cov_type="cluster", cluster="cluster"),
        ).fit(),
    }
    for label, fit in cases.items():
        got, expected = fit(data), fit(_with_tt(frame))
        assert got.std_errors == expected.std_errors, label
        assert got.n_obs == expected.n_obs, label
