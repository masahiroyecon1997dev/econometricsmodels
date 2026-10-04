"""全手法共通の入力列のdtypeのテスト。

polarsの数値dtype（小さい整数型・`Float16`・`Int128`・`Decimal`）が、推定に使わない列に
含まれていても`fit()`を壊さないこと、`x`・`y`として`Float64`列と同じ結果で使えることを
確認する。数値として使えないdtype（文字列・日付・時刻等）は`ValidationError`になる。dtypeごとの許可・拒否の方針は`docs/guide/accepted-data.md`が正本。
"""

from __future__ import annotations

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
from _error_messages import escaped
from econometricsmodels import (
    FE,
    IV,
    OLS,
    WLS,
    IVOptions,
    Logit,
    OLSOptions,
    ValidationError,
)

N = 60

# 整数値の列として作ってから各dtypeへキャストする。値は0..99で、Int8/UInt8の範囲
# （最大127）とFloat16で正確に表せる範囲（整数は2048まで）に収まる。
NUMERIC_DTYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float16,
    pl.Float32,
    pl.Float64,
    pl.Decimal(18, 0),
]

# 推定に使わない列に置いても`fit()`を壊してはいけないdtype。
UNUSED_COLUMN_DTYPES = [
    *NUMERIC_DTYPES,
    pl.Boolean,
    pl.String,
    pl.Categorical,
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
    pl.Binary,
    pl.List(pl.Int64),
    pl.Array(pl.Int64, 2),
    pl.Struct({"a": pl.Int64}),
    pl.Null,
]


def _dtype_id(dtype: pl.DataType) -> str:
    return str(dtype)


def _unused_column(dtype: pl.DataType) -> pl.Series:
    """推定に使わない列（`n`行）をdtypeごとに作る。"""
    if dtype == pl.Null:
        return pl.Series("unused", [None] * N, dtype=pl.Null)
    if dtype == pl.Boolean:
        return pl.Series("unused", [True, False] * (N // 2))
    if dtype == pl.String:
        return pl.Series("unused", ["a", "b"] * (N // 2))
    if dtype == pl.Categorical:
        return pl.Series("unused", ["a", "b"] * (N // 2), dtype=pl.Categorical)
    if dtype == pl.Date:
        return pl.Series("unused", [0] * N, dtype=pl.Int32).cast(pl.Date)
    if dtype == pl.Datetime:
        return pl.Series("unused", [0] * N, dtype=pl.Int64).cast(pl.Datetime)
    if dtype == pl.Duration:
        return pl.Series("unused", [0] * N, dtype=pl.Int64).cast(pl.Duration)
    if dtype == pl.Time:
        return pl.Series("unused", [0] * N, dtype=pl.Int64).cast(pl.Time)
    if dtype == pl.Binary:
        return pl.Series("unused", [b"a", b"b"] * (N // 2))
    if dtype == pl.List(pl.Int64):
        return pl.Series("unused", [[1, 2]] * N)
    if dtype == pl.Array(pl.Int64, 2):
        return pl.Series("unused", [[1, 2]] * N, dtype=dtype)
    if dtype == pl.Struct({"a": pl.Int64}):
        return pl.Series("unused", [{"a": 1}] * N)
    return pl.Series("unused", np.arange(N) % 100, dtype=pl.Int64).cast(dtype)


@pytest.fixture(scope="module")
def base() -> pl.DataFrame:
    """整数値の`x1`・`x2`と、それに線形に依存する`y`（Float64）。"""
    rng = np.random.default_rng(7)
    x1 = rng.integers(0, 100, N)
    x2 = rng.integers(0, 100, N)
    y = 1.0 + 0.5 * x1 - 0.2 * x2 + rng.normal(0.0, 1.0, N)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.mark.parametrize("dtype", UNUSED_COLUMN_DTYPES, ids=_dtype_id)
def test_unused_column_of_any_dtype_does_not_break_fit(base, dtype):
    """推定に使わない列がどのdtypeでも、`fit()`は未使用列を無視して成功する。

    以前は`Int8`/`Int16`/`UInt8`/`UInt16`/`Float16`/`Array`/`Struct`の列が
    あるだけで`DataFrame`の読み込みに失敗し、`Decimal`/`Int128`はRustのpanicに
    なった。
    """
    df = base.with_columns(_unused_column(dtype))

    result = OLS(df, y="y", x=["x1", "x2"]).fit()

    expected = OLS(base, y="y", x=["x1", "x2"]).fit()
    assert result.params == expected.params


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=_dtype_id)
def test_numeric_dtype_as_x_matches_float64(base, dtype):
    """どの数値dtypeを`x`に使っても、`Float64`列と同じ推定結果になる。"""
    df = base.with_columns(pl.col("x1").cast(dtype), pl.col("x2").cast(dtype))

    result = OLS(df, y="y", x=["x1", "x2"]).fit()

    expected = OLS(base, y="y", x=["x1", "x2"]).fit()
    for name, value in expected.params.items():
        assert result.params[name] == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=_dtype_id)
def test_numeric_dtype_as_y_matches_float64(base, dtype):
    """どの数値dtypeを`y`に使っても、`Float64`列と同じ推定結果になる。"""
    y_int = pl.Series("y", (np.arange(N) * 7 + base["x1"].to_numpy()) % 100)
    df = base.with_columns(y_int.cast(dtype))
    df_float = base.with_columns(y_int.cast(pl.Float64))

    result = OLS(df, y="y", x=["x1", "x2"]).fit()

    expected = OLS(df_float, y="y", x=["x1", "x2"]).fit()
    for name, value in expected.params.items():
        assert result.params[name] == pytest.approx(value, rel=1e-12)


def test_to_dummies_columns_are_usable_as_regressors():
    """`DataFrame.to_dummies()`が返す`UInt8`のダミー変数をそのまま`x`に使える。"""
    rng = np.random.default_rng(11)
    group = rng.choice(["a", "b", "c"], N)
    df = pl.DataFrame(
        {"y": rng.normal(0.0, 1.0, N), "x1": rng.normal(0.0, 1.0, N)}
    ).with_columns(pl.Series("g", group))
    dummies = df.to_dummies(columns=["g"])
    assert dummies["g_a"].dtype == pl.UInt8

    result = OLS(dummies, y="y", x=["x1", "g_b", "g_c"]).fit()

    manual = df.with_columns(
        (pl.col("g") == "b").cast(pl.Float64).alias("g_b"),
        (pl.col("g") == "c").cast(pl.Float64).alias("g_c"),
    )
    expected = OLS(manual, y="y", x=["x1", "g_b", "g_c"]).fit()
    assert result.params == expected.params


def test_small_integer_entity_column_is_usable_as_group_key(base):
    """`Int8`のentity列でも固定効果推定ができる（グループキーとして文字列化される）。"""
    df = base.with_columns(
        (pl.int_range(pl.len()) // 6).cast(pl.Int8).alias("entity")
    )

    result = FE(df, y="y", x=["x1", "x2"], entity="entity").fit()

    as_int64 = df.with_columns(pl.col("entity").cast(pl.Int64))
    expected = FE(as_int64, y="y", x=["x1", "x2"], entity="entity").fit()
    assert result.params == expected.params


def test_boolean_y_is_usable_for_logit():
    """`Boolean`の`y`（`True`=1）をLogitの二値アウトカムに使える。"""
    rng = np.random.default_rng(5)
    x1 = rng.normal(0.0, 1.0, 200)
    y = rng.random(200) < 1.0 / (1.0 + np.exp(-(0.3 + 1.2 * x1)))
    df = pl.DataFrame({"y": y, "x1": x1})
    assert df["y"].dtype == pl.Boolean

    result = Logit(df, y="y", x=["x1"]).fit()

    expected = Logit(
        df.with_columns(pl.col("y").cast(pl.Float64)), y="y", x=["x1"]
    ).fit()
    assert result.params == expected.params


# 数値として使えないdtypeと、メッセージに出るdtype名。
NON_NUMERIC_DTYPES = [
    (pl.String, "String"),
    (pl.Categorical, "Categorical"),
    (pl.Enum(["a", "b"]), "Enum"),
    (pl.Date, "Date"),
    (pl.Datetime, "Datetime"),
    (pl.Duration, "Duration"),
    (pl.Time, "Time"),
    (pl.Binary, "Binary"),
    (pl.List(pl.Int64), "List"),
    (pl.Array(pl.Int64, 2), "Array"),
    (pl.Struct({"a": pl.Int64}), "Struct"),
]


def _non_numeric_column(name: str, dtype: pl.DataType) -> pl.Series:
    """数値として使えないdtypeの列（`n`行）を作る。"""
    if dtype == pl.Enum(["a", "b"]):
        return pl.Series(name, ["a", "b"] * (N // 2), dtype=dtype)
    return _unused_column(dtype).alias(name)


@pytest.mark.parametrize(
    ("dtype", "label"),
    NON_NUMERIC_DTYPES,
    ids=[x[1] for x in NON_NUMERIC_DTYPES],
)
@pytest.mark.parametrize("role", ["x", "y"])
def test_non_numeric_dtype_raises_for_numeric_role(base, dtype, label, role):
    """数値として使えないdtypeを`x`・`y`に使うと、dtype名入りの`ValidationError`。"""
    df = base.with_columns(
        _non_numeric_column(role if role == "y" else "x1", dtype)
    )

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE,
            name="y" if role == "y" else "x1",
            dtype=label,
        ),
    ):
        OLS(df, y="y", x=["x1", "x2"]).fit()


def test_numeric_looking_strings_are_rejected(base):
    """`"1.0"`のような数値に見える文字列も、黙って数値化せず拒否する。"""
    df = base.with_columns(
        pl.col("x1").cast(pl.Float64).cast(pl.String).alias("x1")
    )

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="x1", dtype="String"
        ),
    ):
        OLS(df, y="y", x=["x1", "x2"]).fit()


def test_string_y_is_rejected_for_logit(base):
    """Logitの`y`に`"0"`/`"1"`の文字列を渡しても拒否する。"""
    df = pl.DataFrame(
        {"y": ["0", "1"] * (N // 2), "x1": base["x1"].cast(pl.Float64)}
    )

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="y", dtype="String"
        ),
    ):
        Logit(df, y="y", x=["x1"]).fit()


def test_non_numeric_weight_is_rejected(base):
    """WLSの重み列が文字列のときも拒否する。"""
    df = base.with_columns(pl.lit("1").alias("w"))

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="w", dtype="String"
        ),
    ):
        WLS(df, y="y", x=["x1"], weight="w").fit()


def test_non_numeric_instrument_is_rejected(base):
    """IVの操作変数が日付型のときも拒否する。"""
    df = base.with_columns(
        pl.Series("z", [0] * N, dtype=pl.Int32).cast(pl.Date)
    )

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="z", dtype="Date"
        ),
    ):
        IV(df, y="y", x_exog=["x2"], x_endog=["x1"], instruments=["z"]).fit()


@pytest.mark.parametrize("dtype", [pl.Date, pl.Datetime], ids=str)
def test_hac_time_accepts_date_and_datetime_in_the_same_order(base, dtype):
    """`hac_time`は順序だけに使われるため、`Date`/`Datetime`でも整数の時点と同じ結果になる。"""
    rng = np.random.default_rng(3)
    order = rng.permutation(N)
    base = base.with_columns(pl.Series("t_int", order, dtype=pl.Int32))
    options = OLSOptions(cov_type="hac", hac_lags=2, hac_time="t")

    as_int = base.with_columns(pl.col("t_int").alias("t"))
    as_date = base.with_columns(
        pl.col("t_int").cast(pl.Int64).cast(dtype).alias("t")
    )

    expected = OLS(as_int, y="y", x=["x1"], options=options).fit()
    result = OLS(as_date, y="y", x=["x1"], options=options).fit()

    assert result.std_errors == expected.std_errors


def test_hac_time_accepts_date_for_iv(base):
    """IVの`hac_time`（GMMとは別の抽出経路）でも`Date`を受け付ける。"""
    rng = np.random.default_rng(4)
    z = rng.integers(0, 100, N)
    base = base.with_columns(
        pl.Series("z", z),
        pl.Series("t_int", rng.permutation(N), dtype=pl.Int32),
    )
    options = IVOptions(cov_type="hac", hac_lags=2, hac_time="t")

    def fit(df):
        return IV(
            df,
            y="y",
            x_exog=["x2"],
            x_endog=["x1"],
            instruments=["z"],
            options=options,
        ).fit()

    as_int = base.with_columns(pl.col("t_int").alias("t"))
    as_date = base.with_columns(pl.col("t_int").cast(pl.Date).alias("t"))

    assert fit(as_date).std_errors == fit(as_int).std_errors


@pytest.mark.parametrize(
    ("dtype", "label"),
    [
        (pl.String, "String"),
        (pl.Time, "Time"),
        (pl.Duration, "Duration"),
        (pl.Categorical, "Categorical"),
    ],
    ids=["String", "Time", "Duration", "Categorical"],
)
def test_hac_time_rejects_non_orderable_dtypes(base, dtype, label):
    """`hac_time`に文字列・時刻・期間・カテゴリを使うと、時点列用のメッセージで拒否する。"""
    df = base.with_columns(_non_numeric_column("t", dtype))
    options = OLSOptions(cov_type="hac", hac_lags=2, hac_time="t")

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_ORDER_DTYPE, name="t", dtype=label
        ),
    ):
        OLS(df, y="y", x=["x1"], options=options).fit()


def test_all_null_column_reports_missing_values(base):
    """全値が欠損の列（`Null`型）は、dtypeではなく欠損値として報告する。"""
    df = base.with_columns(pl.Series("x1", [None] * N, dtype=pl.Null))

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="x1", count=N),
    ):
        OLS(df, y="y", x=["x1"]).fit()
