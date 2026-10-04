"""全手法共通の、キー列（`entity`・`cluster`・`time`・`dk_time`）のdtypeと値の検証のテスト。

キー列の抽出は手法ごとに別の呼び出し箇所を持つため（OLS/WLS・Logit/Probit/Tobit・IV・
FE・RE）、すべての手法と役割の組み合わせで確認する。許可するdtypeは役割ごとに異なる
（同一性だけのキー: `entity`・`cluster`、時点のキー: `time`・`dk_time`）。
"""

from __future__ import annotations

import datetime as dt

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
from _dtype_helpers import make_frame, unused_column
from _error_messages import escaped
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    FEOptions,
    IVOptions,
    Logit,
    LogitOptions,
    OLSOptions,
    Probit,
    ProbitOptions,
    REOptions,
    Tobit,
    TobitOptions,
    ValidationError,
    WLSOptions,
)


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return make_frame()


# ── 役割ごとの`fit()`（キー列は常に列名`cluster`・`entity`・`t`） ──────────

CLUSTER = {"cov_type": "cluster", "cluster": "cluster"}

CLUSTER_FITS = {
    "OLS": lambda d: OLS(d, "y", ["x1"], OLSOptions(**CLUSTER)).fit(),
    "WLS": lambda d: WLS(d, "y", ["x1"], "w", WLSOptions(**CLUSTER)).fit(),
    "Logit": lambda d: Logit(d, "y01", ["x1"], LogitOptions(**CLUSTER)).fit(),
    "Probit": lambda d: Probit(
        d, "y01", ["x1"], ProbitOptions(**CLUSTER)
    ).fit(),
    "Tobit": lambda d: Tobit(d, "yc", ["x1"], TobitOptions(**CLUSTER)).fit(),
    "IV": lambda d: IV(
        d,
        "y",
        ["x2"],
        ["x1"],
        ["z", "z2"],
        IVOptions(**CLUSTER),
    ).fit(),
    "FE": lambda d: FE(d, "y", ["x1"], "entity", FEOptions(**CLUSTER)).fit(),
    "RE": lambda d: RE(d, "y", ["x1"], "entity", REOptions(**CLUSTER)).fit(),
}

ENTITY_FITS = {
    "FE": lambda d: FE(d, "y", ["x1"], "entity").fit(),
    "RE": lambda d: RE(d, "y", ["x1"], "entity").fit(),
}

TIME_FITS = {
    "FE.time": lambda d: FE(
        d, "y", ["x1"], "entity", FEOptions(time="t")
    ).fit(),
    "FE.dk_time": lambda d: FE(
        d, "y", ["x1"], "entity", FEOptions(cov_type="dk", dk_time="t")
    ).fit(),
    "RE.time": lambda d: RE(
        d, "y", ["x1"], "entity", REOptions(cov_type="dk", time="t")
    ).fit(),
}

# 同一性だけのキーに使えないdtype / 時点のキーに使えないdtypeと、メッセージ中の呼び名。
UNSUPPORTED_IDENTITY_DTYPES = [
    (pl.Duration, "Duration"),
    (pl.Time, "Time"),
    (pl.Decimal(18, 0), "Decimal"),
    (pl.Binary, "Binary"),
    (pl.List(pl.Int64), "List"),
    (pl.Array(pl.Int64, 2), "Array"),
    (pl.Struct({"a": pl.Int64}), "Struct"),
]
UNSUPPORTED_TIME_DTYPES = [
    (pl.Boolean, "Boolean"),
    *UNSUPPORTED_IDENTITY_DTYPES,
]


def _with_key(frame, column, dtype):
    """`column`を、グループ構造を保ったまま`dtype`のキー列に置き換える。"""
    base = pl.col(column)
    if dtype == pl.String:
        expr = base.cast(pl.String)
    elif dtype == pl.Categorical:
        expr = base.cast(pl.String).cast(pl.Categorical)
    elif dtype == pl.Enum([str(i) for i in range(10)]):
        expr = base.cast(pl.String).cast(dtype)
    elif dtype == pl.Date:
        expr = base.cast(pl.Int32).cast(pl.Date)
    elif dtype == pl.Datetime or isinstance(dtype, pl.Datetime):
        expr = base.cast(pl.Int64).cast(dtype)
    else:
        expr = base.cast(dtype)
    return frame.with_columns(expr.alias(column))


def _assert_same_result(result, expected):
    assert result.params == expected.params
    assert result.std_errors == expected.std_errors


# ── 同一性だけのキー: entity・cluster ───────────────────────────────

IDENTITY_OK = [
    pl.Int8,
    pl.UInt16,
    pl.Int64,
    pl.Float64,
    pl.Float32,
    pl.String,
    pl.Categorical,
    pl.Enum([str(i) for i in range(10)]),
    pl.Date,
    pl.Datetime,
    pl.Datetime("ns"),
]


@pytest.mark.parametrize("method", list(CLUSTER_FITS))
@pytest.mark.parametrize("dtype", IDENTITY_OK, ids=str)
def test_supported_cluster_dtypes_give_the_same_result(frame, method, dtype):
    """整数・浮動小数・文字列・カテゴリ・Dateのクラスター列は、同じグループなら全手法で
    整数の列と同じ結果になる。
    """
    fit = CLUSTER_FITS[method]

    _assert_same_result(fit(_with_key(frame, "cluster", dtype)), fit(frame))


@pytest.mark.parametrize("method", list(ENTITY_FITS))
@pytest.mark.parametrize("dtype", IDENTITY_OK, ids=str)
def test_supported_entity_dtypes_give_the_same_result(frame, method, dtype):
    fit = ENTITY_FITS[method]

    _assert_same_result(fit(_with_key(frame, "entity", dtype)), fit(frame))


def test_boolean_cluster_matches_the_same_grouping_as_strings(frame):
    """`Boolean`は2グループのクラスター列として使える（文字列の同じ分け方と同じ結果）。"""
    keyed = frame.with_columns((pl.col("cluster") % 2 == 0).alias("cluster"))
    as_text = keyed.with_columns(pl.col("cluster").cast(pl.String))

    _assert_same_result(
        CLUSTER_FITS["OLS"](keyed), CLUSTER_FITS["OLS"](as_text)
    )


@pytest.mark.parametrize("method", list(CLUSTER_FITS))
@pytest.mark.parametrize(
    ("dtype", "label"),
    UNSUPPORTED_IDENTITY_DTYPES,
    ids=[x[1] for x in UNSUPPORTED_IDENTITY_DTYPES],
)
def test_unsupported_dtype_as_cluster_raises(frame, method, dtype, label):
    """`cluster`に使えないdtypeを渡すと、全手法でdtype名入りの`ValidationError`。"""
    df = frame.with_columns(unused_column(dtype).alias("cluster"))

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_IDENTITY_DTYPE,
            name="cluster",
            dtype=label,
        ),
    ):
        CLUSTER_FITS[method](df)


@pytest.mark.parametrize("method", list(ENTITY_FITS))
@pytest.mark.parametrize(
    ("dtype", "label"),
    UNSUPPORTED_IDENTITY_DTYPES,
    ids=[x[1] for x in UNSUPPORTED_IDENTITY_DTYPES],
)
def test_unsupported_dtype_as_entity_raises(frame, method, dtype, label):
    df = frame.with_columns(unused_column(dtype).alias("entity"))

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_IDENTITY_DTYPE,
            name="entity",
            dtype=label,
        ),
    ):
        ENTITY_FITS[method](df)


# ── 時点のキー: time・dk_time ───────────────────────────────────────

# 時点は0..5の1桁のため、文字列としての順序と時点の順序が一致する。
TIME_OK = [
    pl.Int8,
    pl.Int64,
    pl.Float64,
    pl.String,
    pl.Categorical,
    pl.Date,
    pl.Datetime,
    pl.Datetime("ns"),
]


@pytest.mark.parametrize("method", list(TIME_FITS))
@pytest.mark.parametrize("dtype", TIME_OK, ids=str)
def test_supported_time_dtypes_give_the_same_result(frame, method, dtype):
    fit = TIME_FITS[method]

    _assert_same_result(fit(_with_key(frame, "t", dtype)), fit(frame))


@pytest.mark.parametrize("method", list(TIME_FITS))
@pytest.mark.parametrize(
    ("dtype", "label"),
    UNSUPPORTED_TIME_DTYPES,
    ids=[x[1] for x in UNSUPPORTED_TIME_DTYPES],
)
def test_unsupported_dtype_as_time_raises(frame, method, dtype, label):
    """`time`・`dk_time`に使えないdtypeを渡すと、時点列用のメッセージで拒否する。"""
    df = frame.with_columns(unused_column(dtype).alias("t"))

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_TIME_DTYPE, name="t", dtype=label
        ),
    ):
        TIME_FITS[method](df)


# ── 浮動小数のキーのNaN・無限大 ─────────────────────────────────────

NON_FINITE = [
    (float("nan"), "NaN"),
    (float("inf"), "inf"),
    (float("-inf"), "-inf"),
]


def _scatter(frame, column, value):
    return frame.with_columns(
        frame[column].cast(pl.Float64).scatter(2, value).alias(column)
    )


@pytest.mark.parametrize(
    ("bad", "shown"), NON_FINITE, ids=[x[1] for x in NON_FINITE]
)
@pytest.mark.parametrize(
    ("column", "fits"),
    [
        ("cluster", CLUSTER_FITS),
        ("entity", ENTITY_FITS),
        ("t", TIME_FITS),
    ],
    ids=["cluster", "entity", "time"],
)
def test_non_finite_float_key_raises_for_every_method(
    frame, column, fits, bad, shown
):
    """浮動小数のキー列のNaN・無限大は、全手法・全役割で数値列と同じメッセージで拒否する。"""
    df = _scatter(frame, column, bad)

    for fit in fits.values():
        with pytest.raises(
            ValidationError,
            match=escaped(
                msgs.COLUMN_HAS_NON_FINITE_VALUE,
                name=column,
                value=shown,
                row=2,
            ),
        ):
            fit(df)


@pytest.mark.parametrize(
    ("column", "fits"),
    [
        ("cluster", CLUSTER_FITS),
        ("entity", ENTITY_FITS),
        ("t", TIME_FITS),
    ],
    ids=["cluster", "entity", "time"],
)
def test_null_key_raises_for_every_method(frame, column, fits):
    """キー列のnullも、全手法・全役割で拒否する。"""
    df = frame.with_columns(
        frame[column].cast(pl.Int64).scatter(2, None).alias(column)
    )

    for fit in fits.values():
        with pytest.raises(
            ValidationError,
            match=escaped(
                msgs.GROUP_KEY_COLUMN_HAS_MISSING_VALUES, name=column
            ),
        ):
            fit(df)


# ── 時点の順序: 桁が上がる・月や年をまたぐ時点でも、時系列順になる ───────────


def _long_panel() -> pl.DataFrame:
    """8エンティティ×15期。時点の共通ショックが強い系列相関を持ち、DKの重みが効く。"""
    rng = np.random.default_rng(5)
    n_entities, n_periods = 8, 15
    common = np.zeros(n_periods)
    for k in range(1, n_periods):
        common[k] = 0.9 * common[k - 1] + 0.4 * rng.normal()
    rows = [(e, t) for e in range(n_entities) for t in range(n_periods)]
    x1 = rng.normal(size=len(rows))
    u = np.array([common[t] * (1 + 0.3 * rng.normal()) for _, t in rows])
    return pl.DataFrame(
        {
            "entity": [e for e, _ in rows],
            "period": [t for _, t in rows],
            "x1": x1,
            "y": 0.5 * x1 + u + 0.3 * x1 * u,
        }
    )


def _dk_se(df: pl.DataFrame, column: str) -> float:
    options = FEOptions(cov_type="dk", dk_time=column, dk_bandwidth=3)
    return FE(df, "y", ["x1"], "entity", options).fit().std_errors["x1"]


def _month_start(period: int) -> dt.date:
    """2019年9月から`period`か月後の月初（年をまたぐ）。"""
    month0 = 8 + period
    return dt.date(2019 + month0 // 12, month0 % 12 + 1, 1)


def test_date_and_datetime_time_orders_follow_chronology():
    """月・年をまたぐ`Date`や、単位・小数秒の違う`Datetime`でも、DKの標準誤差は
    ゼロ埋め文字列の時点（辞書順＝時系列順）と一致する。
    """
    panel = _long_panel()
    padded = panel.with_columns(
        pl.col("period").cast(pl.String).str.zfill(3).alias("t_padded")
    )
    expected = _dk_se(padded, "t_padded")

    dates = [_month_start(p) for p in range(15)]
    as_date = panel.with_columns(
        pl.col("period").replace_strict(range(15), dates).alias("t_date")
    )
    assert _dk_se(as_date, "t_date") == pytest.approx(expected, rel=1e-12)

    for unit in ("us", "ns"):
        as_datetime = as_date.with_columns(
            pl.col("t_date").cast(pl.Datetime(unit)).alias("t_dt")
        )
        assert _dk_se(as_datetime, "t_dt") == pytest.approx(
            expected, rel=1e-12
        )

    fractional = as_date.with_columns(
        (
            pl.col("t_date").cast(pl.Datetime("us"))
            + pl.duration(seconds=3, microseconds=250000)
        ).alias("t_frac")
    )
    assert _dk_se(fractional, "t_frac") == pytest.approx(expected, rel=1e-12)


def test_row_order_does_not_change_the_dk_result():
    """行の並びを入れ替えても、時点の順序はキーの値で決まり結果は変わらない。"""
    panel = _long_panel()
    padded = panel.with_columns(
        pl.col("period").cast(pl.String).str.zfill(3).alias("t_padded")
    )

    shuffled = padded.sample(fraction=1.0, shuffle=True, seed=1)

    assert _dk_se(shuffled, "t_padded") == pytest.approx(
        _dk_se(padded, "t_padded"), rel=1e-12
    )


# ── タイムゾーン付きのDatetime ──────────────────────────────────────


def _aware(frame, column):
    return frame.with_columns(
        pl.col(column).cast(pl.Int64).cast(pl.Datetime("us", "UTC"))
    )


@pytest.mark.parametrize("method", list(CLUSTER_FITS))
def test_datetime_with_time_zone_is_rejected_as_cluster(frame, method):
    """タイムゾーン付きの`Datetime`は、対処法を示すメッセージで拒否する。"""
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_KEY_WITH_TIME_ZONE,
            name="cluster",
            time_zone="UTC",
            role="group identifier",
        ),
    ):
        CLUSTER_FITS[method](_aware(frame, "cluster"))


@pytest.mark.parametrize("method", list(ENTITY_FITS))
def test_datetime_with_time_zone_is_rejected_as_entity(frame, method):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_KEY_WITH_TIME_ZONE,
            name="entity",
            time_zone="UTC",
            role="group identifier",
        ),
    ):
        ENTITY_FITS[method](_aware(frame, "entity"))


@pytest.mark.parametrize("method", list(TIME_FITS))
def test_datetime_with_time_zone_is_rejected_as_time(frame, method):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_KEY_WITH_TIME_ZONE,
            name="t",
            time_zone="UTC",
            role="time",
        ),
    ):
        TIME_FITS[method](_aware(frame, "t"))


def test_removing_the_time_zone_makes_the_column_usable(frame):
    """メッセージが示す対処（`replace_time_zone(None)`）で、同じ結果が得られる。"""
    expected = TIME_FITS["FE.dk_time"](frame)
    aware = _aware(frame, "t")

    naive = aware.with_columns(pl.col("t").dt.replace_time_zone(None))

    _assert_same_result(TIME_FITS["FE.dk_time"](naive), expected)
