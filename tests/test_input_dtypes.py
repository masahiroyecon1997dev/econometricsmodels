"""全手法共通の入力列のdtypeのテスト。

polarsの数値dtype（小さい整数型・`Float16`・`Int128`・`Decimal`）が、推定に使わない列に
含まれていても`fit()`を壊さないこと、`x`・`y`として`Float64`列と同じ結果で使えることを
確認する。数値として使えないdtype（文字列・日付・時刻等）は`ValidationError`になる。
dtypeごとの許可・拒否の方針は`docs/guide/accepted-data.md`が正本。
"""

from __future__ import annotations

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
from _dtype_helpers import (
    NON_NUMERIC_DTYPES,
    NUMERIC_DTYPES,
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
    IVOptions,
    Logit,
    OLSOptions,
    Probit,
    Tobit,
    ValidationError,
    WLSOptions,
)


@pytest.fixture(scope="module")
def base() -> pl.DataFrame:
    """整数値の`x1`・`x2`と、それに線形に依存する`y`（Float64）。"""
    rng = np.random.default_rng(7)
    x1 = rng.integers(0, 100, N)
    x2 = rng.integers(0, 100, N)
    y = 1.0 + 0.5 * x1 - 0.2 * x2 + rng.normal(0.0, 1.0, N)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.mark.parametrize("dtype", UNUSED_COLUMN_DTYPES, ids=dtype_id)
def test_unused_column_of_any_dtype_does_not_break_fit(base, dtype):
    """推定に使わない列がどのdtypeでも、`fit()`は未使用列を無視して成功する。

    以前は`Int8`/`Int16`/`UInt8`/`UInt16`/`Float16`/`Array`/`Struct`の列が
    あるだけで`DataFrame`の読み込みに失敗し、`Decimal`/`Int128`はRustのpanicに
    なった。
    """
    df = base.with_columns(unused_column(dtype))

    result = OLS(df, y="y", x=["x1", "x2"]).fit()

    expected = OLS(base, y="y", x=["x1", "x2"]).fit()
    assert result.params == expected.params


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=dtype_id)
def test_numeric_dtype_as_x_matches_float64(base, dtype):
    """どの数値dtypeを`x`に使っても、`Float64`列と同じ推定結果になる。"""
    df = base.with_columns(pl.col("x1").cast(dtype), pl.col("x2").cast(dtype))

    result = OLS(df, y="y", x=["x1", "x2"]).fit()

    expected = OLS(base, y="y", x=["x1", "x2"]).fit()
    for name, value in expected.params.items():
        assert result.params[name] == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=dtype_id)
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


def _non_numeric_column(name: str, dtype: pl.DataType) -> pl.Series:
    """数値として使えないdtypeの列（`n`行）を作る。"""
    if dtype == pl.Enum(["a", "b"]):
        return pl.Series(name, ["a", "b"] * (N // 2), dtype=dtype)
    return unused_column(dtype).alias(name)


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


# ── 数値として使う全ロール × dtype ──────────────────────────────────
#
# 列の抽出は手法・ロールごとに別の呼び出し箇所を持つため、`weight`・IVの各変数・
# FE/REの`y`/`x`・MLEの`y`についても、許可と拒否の両方を確認する。

# ロール名 → `fit()`（列名`col`をそのロールに使う）。
ROLE_FITS = {
    "wls.weight": lambda d: WLS(d, "y", ["x1"], "col").fit(),
    "iv.x_exog": lambda d: IV(d, "y", ["col"], ["x1"], ["z", "z2"]).fit(),
    "iv.x_endog": lambda d: IV(d, "y", ["x2"], ["col"], ["z", "z2"]).fit(),
    "iv.instruments": lambda d: IV(d, "y", ["x2"], ["x1"], ["z", "col"]).fit(),
    "fe.y": lambda d: FE(d, "col", ["x1"], "entity").fit(),
    "fe.x": lambda d: FE(d, "y", ["col"], "entity").fit(),
    "re.y": lambda d: RE(d, "col", ["x1"], "entity").fit(),
    "re.x": lambda d: RE(d, "y", ["col"], "entity").fit(),
    "logit.y": lambda d: Logit(d, "col", ["x1"]).fit(),
    "probit.y": lambda d: Probit(d, "col", ["x1"]).fit(),
    "tobit.y": lambda d: Tobit(d, "col", ["x1"]).fit(),
    "ols.x": lambda d: OLS(d, "y", ["col"]).fit(),
}

BINARY_Y_ROLES = {"logit.y", "probit.y"}


@pytest.fixture(scope="module")
def roles_frame() -> pl.DataFrame:
    return make_frame()


def _role_column(role: str, dtype: pl.DataType) -> pl.Series:
    """ロールに合う値の列（`N`行）を`dtype`で作る。

    二値の`y`は0/1、重みは正の整数、Tobitの`y`は0以上、その他は整数値の乱数。
    """
    rng = np.random.default_rng(13)
    if role in BINARY_Y_ROLES:
        values = np.arange(N) % 2
        values[:4] = [0, 1, 1, 0]
        # 説明変数と無関係に交互にすると分離しないよう、少し崩す
        values = np.where(rng.random(N) < 0.5, values, 1 - values)
    elif role == "wls.weight":
        values = rng.integers(1, 100, N)
    else:
        values = rng.integers(0, 100, N)
    return pl.Series("col", values, dtype=pl.Int64).cast(dtype)


@pytest.mark.parametrize("role", list(ROLE_FITS))
@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.UInt8,
        pl.UInt64,
        pl.Int128,
        pl.Float16,
        pl.Float32,
        pl.Decimal(18, 0),
    ],
    ids=str,
)
def test_numeric_dtypes_are_accepted_in_every_numeric_role(
    roles_frame, role, dtype
):
    """整数・浮動小数・Decimalの列は、全ての数値ロールで`Float64`列と同じ結果になる。"""
    df = roles_frame.with_columns(_role_column(role, dtype))
    as_float = roles_frame.with_columns(_role_column(role, pl.Float64))

    result = ROLE_FITS[role](df)
    expected = ROLE_FITS[role](as_float)

    assert result.n_obs == N
    for name, value in expected.params.items():
        assert result.params[name] == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("role", [r for r in ROLE_FITS if r != "wls.weight"])
def test_boolean_is_accepted_in_every_numeric_role_except_weight(
    roles_frame, role
):
    """`Boolean`は`True`=1の数値として使える。重みは0を許さないため別のテストで扱う。"""
    as_int = _role_column(role, pl.Int64) % 2
    df = roles_frame.with_columns((as_int == 1).alias("col"))
    as_float = roles_frame.with_columns(as_int.cast(pl.Float64).alias("col"))

    result = ROLE_FITS[role](df)
    expected = ROLE_FITS[role](as_float)

    for name, value in expected.params.items():
        assert result.params[name] == pytest.approx(value, rel=1e-12)


def test_boolean_weight_is_accepted_when_all_true_and_rejected_with_false(
    roles_frame,
):
    """`Boolean`の重みは、全て`True`（=1）なら通り、`False`（=0）を含むと重みの検証で拒否する。"""
    all_true = roles_frame.with_columns(pl.lit(True).alias("col"))
    assert ROLE_FITS["wls.weight"](all_true).n_obs == N

    with_false = roles_frame.with_columns(
        (pl.int_range(pl.len()) > 0).alias("col")
    )
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.NON_POSITIVE_WEIGHT, row=0, weight="0"),
    ):
        ROLE_FITS["wls.weight"](with_false)


@pytest.mark.parametrize("role", list(ROLE_FITS))
@pytest.mark.parametrize(
    ("dtype", "label"),
    NON_NUMERIC_DTYPES,
    ids=[x[1] for x in NON_NUMERIC_DTYPES],
)
def test_non_numeric_dtypes_are_rejected_in_every_numeric_role(
    roles_frame, role, dtype, label
):
    """数値として使えないdtypeは、全ての数値ロールで列名とdtype名入りの`ValidationError`。"""
    df = roles_frame.with_columns(unused_column(dtype).alias("col"))

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="col", dtype=label
        ),
    ):
        ROLE_FITS[role](df)


@pytest.mark.parametrize("role", list(ROLE_FITS))
def test_all_null_column_reports_missing_values_in_every_numeric_role(
    roles_frame, role
):
    """全値が欠損の列（`Null`型）は、どのロールでも欠損値として報告する。"""
    df = roles_frame.with_columns(pl.Series("col", [None] * N, dtype=pl.Null))

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="col", count=N),
    ):
        ROLE_FITS[role](df)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float16], ids=str)
@pytest.mark.parametrize(
    ("bad", "shown"), [(float("nan"), "NaN"), (float("inf"), "inf")]
)
def test_non_finite_values_are_rejected_for_narrow_floats(
    roles_frame, dtype, bad, shown
):
    """幅の狭い浮動小数の列のNaN・無限大も、`Float64`と同じメッセージで拒否する。"""
    df = roles_frame.with_columns(
        pl.Series("col", np.arange(N) % 100, dtype=pl.Int64)
        .cast(dtype)
        .scatter(3, bad)
    )

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="col", value=shown, row=3
        ),
    ):
        ROLE_FITS["ols.x"](df)


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Decimal(18, 0)], ids=str)
def test_null_in_integer_and_decimal_columns_is_rejected(roles_frame, dtype):
    values = [None if i == 5 else i % 100 for i in range(N)]
    df = roles_frame.with_columns(
        pl.Series("col", values, dtype=pl.Int64).cast(dtype)
    )

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="col", count=1),
    ):
        ROLE_FITS["ols.x"](df)


def test_decimal_with_scale_keeps_its_value(roles_frame):
    """小数部のあるDecimalが、桁をずらさずそのまま数値として使われる。"""
    values = np.round(np.random.default_rng(3).normal(size=N), 2)
    as_decimal = pl.Series("col", values).cast(pl.Decimal(18, 2))
    as_float = pl.Series("col", values)

    result = ROLE_FITS["ols.x"](roles_frame.with_columns(as_decimal))
    expected = ROLE_FITS["ols.x"](roles_frame.with_columns(as_float))

    assert result.params["col"] == pytest.approx(
        expected.params["col"], rel=1e-12
    )


# ── hac_time（順序だけに使う数値列） ────────────────────────────────

HAC_FITS = {
    "OLS": lambda d: OLS(
        d, "y", ["x1"], OLSOptions(cov_type="hac", hac_lags=2, hac_time="t")
    ).fit(),
    "WLS": lambda d: WLS(
        d,
        "y",
        ["x1"],
        "w",
        WLSOptions(cov_type="hac", hac_lags=2, hac_time="t"),
    ).fit(),
    "IV": lambda d: IV(
        d,
        "y",
        ["x2"],
        ["x1"],
        ["z", "z2"],
        IVOptions(cov_type="hac", hac_lags=2, hac_time="t"),
    ).fit(),
}


@pytest.fixture(scope="module")
def shuffled_time_frame() -> pl.DataFrame:
    """行順と時点の順序が異なるデータ（`t`が0..59の並べ替え）。"""
    frame = make_frame()
    order = np.random.default_rng(3).permutation(N)
    return frame.with_columns(pl.Series("t", order, dtype=pl.Int64))


@pytest.mark.parametrize("method", list(HAC_FITS))
@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int16,
        pl.UInt8,
        pl.Float32,
        pl.Float64,
        pl.Decimal(18, 0),
        pl.Date,
        pl.Datetime,
        pl.Datetime("ns"),
        pl.Datetime("us", "UTC"),
    ],
    ids=str,
)
def test_hac_time_accepts_orderable_dtypes_with_the_same_order(
    shuffled_time_frame, method, dtype
):
    """`hac_time`は順序だけに使われるため、整数・浮動小数・Decimal・Date・Datetime
    （単位・タイムゾーン違いを含む）でも、整数の時点と同じ結果になる。
    """
    # UInt8は0..255にしか収まらないため、時点を0..199へ折り返した並べ替えで比較する
    # （どのdtypeでも比較対象の整数の時点と同じ並びになる）。
    base_time = (pl.col("t") % 200).cast(pl.Int64)
    frame = shuffled_time_frame.with_columns(base_time.alias("t"))
    expected = HAC_FITS[method](frame)
    converted = frame.with_columns(pl.col("t").cast(dtype))

    result = HAC_FITS[method](converted)

    assert result.std_errors == expected.std_errors


@pytest.mark.parametrize("method", list(HAC_FITS))
@pytest.mark.parametrize(
    "dtype",
    [pl.Int64, pl.Datetime("ns"), pl.Datetime("ns", "UTC")],
    ids=str,
)
def test_hac_time_keeps_order_of_values_that_collapse_in_float64(
    shuffled_time_frame, method, dtype
):
    """2^53を超える整数・ナノ秒の`Datetime`で、f64に変換すると同値に潰れる
    ほど近い時点でも、元のdtypeで順序づけるので重複とは見なされず、
    小さな整数の時点と同じ結果になる。
    """
    base = 1_700_000_000_000_000_000
    assert float(base + 1) == float(base + 2)  # 前提: f64では区別できない
    expected = HAC_FITS[method](shuffled_time_frame)
    shifted = shuffled_time_frame.with_columns(
        (pl.col("t") + base).cast(pl.Int64).cast(dtype).alias("t")
    )

    result = HAC_FITS[method](shifted)

    assert result.std_errors == expected.std_errors


@pytest.mark.parametrize("method", list(HAC_FITS))
@pytest.mark.parametrize(
    ("dtype", "label"),
    [
        (pl.String, "String"),
        (pl.Categorical, "Categorical"),
        (pl.Enum(["a", "b"]), "Enum"),
        (pl.Boolean, "Boolean"),
        (pl.Time, "Time"),
        (pl.Duration, "Duration"),
        (pl.Binary, "Binary"),
        (pl.List(pl.Int64), "List"),
        (pl.Array(pl.Int64, 2), "Array"),
        (pl.Struct({"a": pl.Int64}), "Struct"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_hac_time_rejects_non_orderable_dtypes_for_every_method(
    roles_frame, method, dtype, label
):
    df = roles_frame.with_columns(unused_column(dtype).alias("t"))

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_ORDER_DTYPE, name="t", dtype=label
        ),
    ):
        HAC_FITS[method](df)
