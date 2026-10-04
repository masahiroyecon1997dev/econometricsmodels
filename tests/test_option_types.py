"""全手法共通の、推定オプションの数値フィールドの型・値の検証のテスト。

- 型の誤り（`bool`・文字列・`None`・整数に`float`）は`TypeError`
  （コンストラクタ引数と属性代入の両方）。`bool`は`int`のサブクラスだが、数値として渡す
  意図はないため拒否する。
- 値の誤り（範囲外・NaN・巨大な値）は`fit()`時の`ValidationError`。`i64`に収まらない
  巨大な整数は`OverflowError`ではなく、`i64`の端に丸められて範囲検査に到達する。
"""

from __future__ import annotations

import re

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
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

N = 60
I64_MAX = 2**63 - 1
I64_MIN = -(2**63)


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(2)
    x1 = rng.normal(0.0, 1.0, N)
    x2 = rng.normal(0.0, 1.0, N)
    y = 1.0 + x1 - 0.5 * x2 + rng.normal(0.0, 1.0, N)
    return pl.DataFrame(
        {
            "y": y,
            "y01": (y > np.median(y)).astype(np.float64),
            "yc": np.clip(y, 0.0, None),
            "x1": x1,
            "x2": x2,
            "z": x1 + rng.normal(0.0, 1.0, N),
            "z2": x1 + rng.normal(0.0, 1.0, N),
            "w": np.ones(N),
            "entity": np.arange(N) // 6,
            "t": np.arange(N) % 6,
        }
    )


# (オプションクラス, フィールド名, 種類)。種類は`int`・`float`。`Option`のフィールド
# （`None`を許す）かどうかは`NULLABLE`で表す。
FIELDS = [
    (OLSOptions, "confidence_level", "float"),
    (OLSOptions, "hac_lags", "int"),
    (WLSOptions, "confidence_level", "float"),
    (WLSOptions, "hac_lags", "int"),
    (LogitOptions, "confidence_level", "float"),
    (LogitOptions, "max_iter", "int"),
    (LogitOptions, "tol", "float"),
    (ProbitOptions, "confidence_level", "float"),
    (ProbitOptions, "max_iter", "int"),
    (ProbitOptions, "tol", "float"),
    (TobitOptions, "confidence_level", "float"),
    (TobitOptions, "max_iter", "int"),
    (TobitOptions, "tol", "float"),
    (TobitOptions, "lower", "float"),
    (TobitOptions, "upper", "float"),
    (IVOptions, "confidence_level", "float"),
    (IVOptions, "hac_lags", "int"),
    (IVOptions, "gmm_max_iter", "int"),
    (IVOptions, "gmm_tol", "float"),
    (FEOptions, "confidence_level", "float"),
    (FEOptions, "dk_bandwidth", "int"),
    (REOptions, "confidence_level", "float"),
    (REOptions, "dk_bandwidth", "int"),
]

# `None`を渡せる（既定が`None`、または`None`が「自動」を表す）フィールド。
NULLABLE = {
    "hac_lags",
    "gmm_max_iter",
    "gmm_tol",
    "dk_bandwidth",
    "lower",
    "upper",
    "tol",  # コンストラクタ引数のみ。`tol`のsetterは実数を要求する。
}

FIELD_IDS = [f"{cls.__name__}.{name}" for cls, name, _ in FIELDS]


def _expected(name: str, kind: str, got: str) -> str:
    article = "an int" if kind == "int" else "a real number"
    return re.escape(f"'{name}' must be {article}, got {got}")


@pytest.mark.parametrize(("cls", "name", "kind"), FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("value", [True, False], ids=["True", "False"])
def test_bool_is_rejected_with_type_error(cls, name, kind, value):
    """`bool`は数値として受け付けない（コンストラクタ・属性代入とも`TypeError`）。"""
    with pytest.raises(TypeError, match=_expected(name, kind, "bool")):
        cls(**{name: value})

    options = cls()
    with pytest.raises(TypeError, match=_expected(name, kind, "bool")):
        setattr(options, name, value)


@pytest.mark.parametrize(("cls", "name", "kind"), FIELDS, ids=FIELD_IDS)
def test_non_numeric_value_is_rejected_with_type_error(cls, name, kind):
    """文字列は`TypeError`（コンストラクタ・属性代入とも）。"""
    with pytest.raises(TypeError, match=_expected(name, kind, "str")):
        cls(**{name: "1"})

    options = cls()
    with pytest.raises(TypeError, match=_expected(name, kind, "str")):
        setattr(options, name, "1")


@pytest.mark.parametrize(
    ("cls", "name"),
    [(c, n) for c, n, kind in FIELDS if kind == "int"],
    ids=[i for i, (_, _, kind) in zip(FIELD_IDS, FIELDS) if kind == "int"],
)
def test_float_is_rejected_for_int_fields(cls, name):
    """整数のフィールドに`float`（`1.5`）を渡すと`TypeError`。"""
    with pytest.raises(TypeError, match=_expected(name, "int", "float")):
        cls(**{name: 1.5})


@pytest.mark.parametrize(
    ("cls", "name"),
    [
        (c, n)
        for c, n, kind in FIELDS
        if n not in NULLABLE and kind in ("int", "float")
    ],
    ids=[
        i
        for i, (_, n, kind) in zip(FIELD_IDS, FIELDS)
        if n not in NULLABLE and kind in ("int", "float")
    ],
)
def test_none_is_rejected_for_required_fields(cls, name):
    """`None`を許さないフィールド（`confidence_level`・`max_iter`）に`None`を渡すと`TypeError`。"""
    kind = next(k for c, n, k in FIELDS if c is cls and n == name)
    with pytest.raises(TypeError, match=_expected(name, kind, "NoneType")):
        cls(**{name: None})


@pytest.mark.parametrize(
    ("cls", "name"),
    [(c, n) for c, n, _ in FIELDS if n in NULLABLE and n != "tol"],
    ids=[
        i
        for i, (_, n, _) in zip(FIELD_IDS, FIELDS)
        if n in NULLABLE and n != "tol"
    ],
)
def test_none_is_accepted_for_nullable_fields(cls, name):
    """`None`を許すフィールドは`None`を受け付け、そのまま保持する。"""
    options = cls(**{name: None})

    assert getattr(options, name) is None


def test_valid_values_round_trip():
    """正しい値（整数・実数）は従来どおり受け付け、属性代入でも反映される。"""
    options = OLSOptions(
        cov_type="hac",
        confidence_level=0.9,
        hac_lags=3,
        include_intercept=True,
    )
    assert options.hac_lags == 3
    assert options.confidence_level == 0.9

    options.hac_lags = 5
    options.confidence_level = 1 - 0.05  # 実数の計算結果
    assert options.hac_lags == 5

    # 整数を実数のフィールドに渡してよい（`tol=1`は`1.0`）。
    assert LogitOptions(tol=1).tol == 1.0


# ── 属性代入（setter）──────────────────────────────────────────────
#
# setterはフィールドごとの手書き実装のため、コンストラクタとは別に全フィールドを検証する。


def _valid_value(kind: str):
    return 7 if kind == "int" else 0.25


@pytest.mark.parametrize(("cls", "name", "kind"), FIELDS, ids=FIELD_IDS)
def test_setter_stores_the_value_in_its_own_field(cls, name, kind):
    """正しい値を代入すると、同じフィールドにその値が入る（別のフィールドへの
    代入や代入忘れを検出する）。他のフィールドは変わらない。
    """
    options = cls()
    before = {n: getattr(options, n) for c, n, _ in FIELDS if c is cls}

    setattr(options, name, _valid_value(kind))

    assert getattr(options, name) == _valid_value(kind)
    for other, value in before.items():
        if other != name:
            assert getattr(options, other) == value


@pytest.mark.parametrize(
    ("cls", "name"),
    [(c, n) for c, n, _ in FIELDS if n in NULLABLE and n != "tol"],
    ids=[
        i
        for i, (_, n, _) in zip(FIELD_IDS, FIELDS)
        if n in NULLABLE and n != "tol"
    ],
)
def test_setter_accepts_none_for_nullable_fields(cls, name):
    """`None`を許すフィールドは、代入でも`None`に戻せる。"""
    options = cls(
        **{
            name: 3
            if "iter" in name or "lags" in name or "band" in name
            else 0.5
        }
    )
    assert getattr(options, name) is not None

    setattr(options, name, None)

    assert getattr(options, name) is None


@pytest.mark.parametrize(
    ("cls", "name", "kind"),
    [(c, n, k) for c, n, k in FIELDS if n not in NULLABLE or n == "tol"],
    ids=[
        i
        for i, (_, n, _) in zip(FIELD_IDS, FIELDS)
        if n not in NULLABLE or n == "tol"
    ],
)
def test_setter_rejects_none_for_required_fields(cls, name, kind):
    """`None`を許さないフィールド（`tol`のsetterを含む）への`None`の代入は`TypeError`。"""
    with pytest.raises(TypeError, match=_expected(name, kind, "NoneType")):
        setattr(cls(), name, None)


@pytest.mark.parametrize(
    ("cls", "name"),
    [(c, n) for c, n, kind in FIELDS if kind == "int"],
    ids=[i for i, (_, _, kind) in zip(FIELD_IDS, FIELDS) if kind == "int"],
)
def test_setter_rejects_float_for_int_fields(cls, name):
    with pytest.raises(TypeError, match=_expected(name, "int", "float")):
        setattr(cls(), name, 1.5)


@pytest.mark.parametrize(("cls", "name", "kind"), FIELDS, ids=FIELD_IDS)
def test_setter_saturates_huge_numbers(cls, name, kind):
    """`i64`・`f64`に収まらない巨大な値は、代入でも`OverflowError`にならず端に丸められる。"""
    options = cls()
    if kind == "int":
        setattr(options, name, 2**70)
        assert getattr(options, name) == I64_MAX
        setattr(options, name, -(2**70))
        assert getattr(options, name) == I64_MIN
    else:
        setattr(options, name, 10**400)
        assert getattr(options, name) == float("inf")
        setattr(options, name, -(10**400))
        assert getattr(options, name) == float("-inf")


# ── 値の誤り: fit()時のValidationError ──────────────────────────────


def _fit(options, df):
    """オプションのクラスに対応する手法で`fit()`する。"""
    cls = type(options)
    if cls is OLSOptions:
        return OLS(df, y="y", x=["x1"], options=options).fit()
    if cls is WLSOptions:
        return WLS(df, y="y", x=["x1"], weight="w", options=options).fit()
    if cls is LogitOptions:
        return Logit(df, y="y01", x=["x1"], options=options).fit()
    if cls is ProbitOptions:
        return Probit(df, y="y01", x=["x1"], options=options).fit()
    if cls is TobitOptions:
        return Tobit(df, y="yc", x=["x1"], options=options).fit()
    if cls is IVOptions:
        return IV(
            df,
            y="y",
            x_exog=["x2"],
            x_endog=["x1"],
            instruments=["z", "z2"],
            options=options,
        ).fit()
    if cls is FEOptions:
        return FE(df, y="y", x=["x1"], entity="entity", options=options).fit()
    return RE(df, y="y", x=["x1"], entity="entity", options=options).fit()


ALL_OPTION_CLASSES = [
    OLSOptions,
    WLSOptions,
    LogitOptions,
    ProbitOptions,
    TobitOptions,
    IVOptions,
    FEOptions,
    REOptions,
]

# 値を保ったままオプションを作る方法の違い: コンストラクタか属性代入か。
SETTERS = ["constructor", "setter"]


def _with_value(cls, name, value, how, **kwargs):
    if how == "constructor":
        return cls(**{name: value}, **kwargs)
    options = cls(**kwargs)
    setattr(options, name, value)
    return options


@pytest.mark.parametrize("how", SETTERS)
@pytest.mark.parametrize("huge", [2**70, -(2**70)], ids=["+2^70", "-2^70"])
@pytest.mark.parametrize(
    "cls", [OLSOptions, WLSOptions, IVOptions], ids=["OLS", "WLS", "IV"]
)
def test_huge_hac_lags_is_validation_error_not_overflow(df, cls, huge, how):
    """`hac_lags`が`i64`に収まらなくても`OverflowError`ではなく、範囲検査の
    `ValidationError`（メッセージには丸めた`i64`の端の値が出る）。
    """
    options = _with_value(cls, "hac_lags", huge, how, cov_type="hac")

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_HAC_LAGS,
            hac_lags=I64_MAX if huge > 0 else I64_MIN,
            n=N,
        ),
    ):
        _fit(options, df)


@pytest.mark.parametrize("how", SETTERS)
@pytest.mark.parametrize("huge", [2**70, -(2**70)], ids=["+2^70", "-2^70"])
@pytest.mark.parametrize("cls", [FEOptions, REOptions], ids=["FE", "RE"])
def test_huge_dk_bandwidth_is_validation_error(df, cls, huge, how):
    kwargs = (
        {"cov_type": "dk", "dk_time": "t"}
        if cls is FEOptions
        else {"cov_type": "dk", "time": "t"}
    )
    options = _with_value(cls, "dk_bandwidth", huge, how, **kwargs)

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_DK_BANDWIDTH,
            bandwidth=I64_MAX if huge > 0 else I64_MIN,
            t=6,
        ),
    ):
        _fit(options, df)


@pytest.mark.parametrize("how", SETTERS)
@pytest.mark.parametrize(
    "cls",
    [LogitOptions, ProbitOptions, TobitOptions],
    ids=["Logit", "Probit", "Tobit"],
)
class TestMleOptionValues:
    def test_negative_huge_max_iter_is_validation_error(self, df, cls, how):
        options = _with_value(cls, "max_iter", -(2**70), how)

        with pytest.raises(
            ValidationError,
            match=escaped(msgs.INVALID_MAX_ITER, max_iter=I64_MIN),
        ):
            _fit(options, df)

    def test_huge_max_iter_is_accepted_as_unbounded(self, df, cls, how):
        """大きな正の`max_iter`は有効な値（実質無制限）で、収束すれば通常どおり返る。"""
        options = _with_value(cls, "max_iter", 2**70, how)

        assert _fit(options, df).converged

    @pytest.mark.parametrize(
        ("tol", "shown"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "inf"),
            (float("-inf"), "-inf"),
            (0.0, "0"),
            (-1.0, "-1"),
            (10**400, "inf"),
        ],
        ids=["nan", "inf", "-inf", "zero", "negative", "huge_int"],
    )
    def test_invalid_tol_is_validation_error(self, df, cls, how, tol, shown):
        """`tol`がNaN・無限大・0以下・巨大な値のときは`ValidationError`（NaNは
        以前は収束失敗の`ComputationError`になっていた）。
        """
        options = _with_value(cls, "tol", tol, how)

        with pytest.raises(
            ValidationError, match=escaped(msgs.INVALID_TOL, tol=shown)
        ):
            _fit(options, df)


@pytest.mark.parametrize("how", SETTERS)
@pytest.mark.parametrize(
    "confidence_level",
    [float("nan"), float("inf"), 2**70, -(2**70), 10**400, 0, 1],
    ids=["nan", "inf", "2^70", "-2^70", "huge_int", "zero", "one"],
)
@pytest.mark.parametrize(
    "cls", ALL_OPTION_CLASSES, ids=[c.__name__ for c in ALL_OPTION_CLASSES]
)
def test_invalid_confidence_level_is_validation_error_for_every_method(
    df, cls, confidence_level, how
):
    """`confidence_level`の不正な値は、全手法で範囲検査の`ValidationError`になる
    （検査は手法ごとに別の箇所にあり、IV・FE・REはエラーを別の文言で包みうる）。
    """
    options = _with_value(cls, "confidence_level", confidence_level, how)

    with pytest.raises(
        ValidationError,
        match=re.escape("confidence_level must be in the range (0, 1)"),
    ):
        _fit(options, df)


@pytest.mark.parametrize("how", SETTERS)
@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), 10**400],
    ids=["nan", "inf", "huge_int"],
)
@pytest.mark.parametrize("field", ["lower", "upper"])
def test_invalid_tobit_bound_is_validation_error(df, field, bad, how):
    """Tobitの打ち切り境界がNaN・無限大・巨大な値のときは`ValidationError`。"""
    options = _with_value(TobitOptions, field, bad, how)

    with pytest.raises(
        ValidationError, match=re.escape("invalid censoring bounds")
    ):
        _fit(options, df)


@pytest.mark.parametrize("how", SETTERS)
@pytest.mark.parametrize("huge", [2**70, -(2**70)], ids=["+2^70", "-2^70"])
def test_huge_gmm_max_iter(df, huge, how):
    """`gmm_max_iter`が`i64`に収まらなくても`OverflowError`にならない。負の巨大な値は
    下限（3以上）の検査で`ValidationError`、正の巨大な値は有効な値として通る。
    """
    options = _with_value(
        IVOptions,
        "gmm_max_iter",
        huge,
        how,
        estimator="gmm",
        gmm_type="iterated",
        raise_on_non_convergence=False,
    )

    if huge < 0:
        with pytest.raises(
            ValidationError,
            match=escaped(msgs.INVALID_GMM_MAX_ITER, max_iter=I64_MIN),
        ):
            _fit(options, df)
    else:
        assert _fit(options, df).n_obs == N


@pytest.mark.parametrize("how", SETTERS)
@pytest.mark.parametrize(
    ("tol", "shown"),
    [(float("nan"), "NaN"), (float("inf"), "inf"), (-1.0, "-1")],
    ids=["nan", "inf", "negative"],
)
def test_invalid_gmm_tol_is_validation_error(df, tol, shown, how):
    options = _with_value(
        IVOptions,
        "gmm_tol",
        tol,
        how,
        estimator="gmm",
        gmm_type="iterated",
    )

    with pytest.raises(
        ValidationError, match=escaped(msgs.INVALID_GMM_TOL, gmm_tol=shown)
    ):
        _fit(options, df)


# ── NumPyのスカラー ────────────────────────────────────────────────


@pytest.mark.parametrize(("cls", "name", "kind"), FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("value", [np.True_, np.False_], ids=["True", "False"])
def test_numpy_bool_is_rejected_like_bool(cls, name, kind, value):
    """`numpy.bool_`も`bool`と同じく`TypeError`（実数のoptionに`__float__`経由で
    通り抜け、`tol=np.True_`が1.0になっていた）。コンストラクタ・属性代入とも。
    """
    article = "an int" if kind == "int" else "a real number"
    pattern = re.escape(f"'{name}' must be {article}, got numpy.bool")
    with pytest.raises(TypeError, match=pattern):
        cls(**{name: value})

    with pytest.raises(TypeError, match=pattern):
        setattr(cls(), name, value)


def test_numpy_numbers_are_accepted():
    """NumPyの整数・実数のスカラーは通常の数値として受け付ける。"""
    options = OLSOptions(
        cov_type="hac", hac_lags=np.int64(3), confidence_level=np.float64(0.9)
    )

    assert options.hac_lags == 3
    assert options.confidence_level == 0.9
    assert LogitOptions(tol=np.float32(1e-3)).tol == pytest.approx(1e-3)


@pytest.mark.parametrize("how", SETTERS)
def test_numpy_uint64_beyond_i64_is_validation_error(df, how):
    """`i64`を超える`numpy.uint64`も`OverflowError`ではなく`ValidationError`。"""
    options = _with_value(
        OLSOptions, "hac_lags", np.uint64(2**63 + 5), how, cov_type="hac"
    )

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INVALID_HAC_LAGS, hac_lags=I64_MAX, n=N),
    ):
        _fit(options, df)


# ── 文字列のoption ─────────────────────────────────────────────────
#
# 列名（`cluster`・`hac_time`・`time`・`dk_time`）は`fit()`の列名引数と同じ形式、
# それ以外（`cov_type`・`solver`等の選択肢）は`must be a str`のメッセージになる。

# (オプションクラス, フィールド名, 種類, `None`を許すか)。種類は`column`・`text`。
STRING_FIELDS = [
    (OLSOptions, "cov_type", "text", False),
    (OLSOptions, "cluster", "column", True),
    (OLSOptions, "hac_time", "column", True),
    (WLSOptions, "cov_type", "text", False),
    (WLSOptions, "cluster", "column", True),
    (WLSOptions, "hac_time", "column", True),
    (LogitOptions, "cov_type", "text", False),
    (LogitOptions, "cluster", "column", True),
    (LogitOptions, "solver", "text", False),
    (ProbitOptions, "cov_type", "text", False),
    (ProbitOptions, "cluster", "column", True),
    (ProbitOptions, "solver", "text", False),
    (TobitOptions, "cov_type", "text", False),
    (TobitOptions, "cluster", "column", True),
    (TobitOptions, "solver", "text", False),
    (IVOptions, "estimator", "text", False),
    (IVOptions, "cov_type", "text", False),
    (IVOptions, "cluster", "column", True),
    (IVOptions, "hac_time", "column", True),
    (IVOptions, "gmm_weight_type", "text", True),
    (IVOptions, "gmm_type", "text", True),
    (FEOptions, "cov_type", "text", False),
    (FEOptions, "time", "column", True),
    (FEOptions, "cluster", "column", True),
    (FEOptions, "dk_time", "column", True),
    (REOptions, "cov_type", "text", False),
    (REOptions, "time", "column", True),
    (REOptions, "cluster", "column", True),
]
STRING_IDS = [f"{c.__name__}.{n}" for c, n, _, _ in STRING_FIELDS]


def _string_message(name: str, kind: str, got: str) -> str:
    noun = "a str column name" if kind == "column" else "a str"
    return re.escape(f"'{name}' must be {noun}, got {got}")


@pytest.mark.parametrize(
    ("cls", "name", "kind", "nullable"), STRING_FIELDS, ids=STRING_IDS
)
@pytest.mark.parametrize(
    ("value", "got"),
    [(1, "int"), (["a"], "list"), (b"a", "bytes"), (1.5, "float")],
    ids=["int", "list", "bytes", "float"],
)
def test_string_option_rejects_non_str_with_type_error(
    cls, name, kind, nullable, value, got
):
    """文字列のoptionに`str`以外を渡すと、引数名と実際の型を含む`TypeError`
    （コンストラクタ・属性代入とも）。
    """
    pattern = _string_message(name, kind, got)
    with pytest.raises(TypeError, match=pattern):
        cls(**{name: value})

    with pytest.raises(TypeError, match=pattern):
        setattr(cls(), name, value)


@pytest.mark.parametrize(
    ("cls", "name", "kind", "nullable"), STRING_FIELDS, ids=STRING_IDS
)
def test_string_option_none_handling(cls, name, kind, nullable):
    """`None`は、`None`を許すoptionだけが受け付け、許さないoptionは`TypeError`。"""
    if nullable:
        assert getattr(cls(**{name: None}), name) is None
        options = cls(**{name: "abc"})
        setattr(options, name, None)
        assert getattr(options, name) is None
    else:
        pattern = _string_message(name, kind, "NoneType")
        with pytest.raises(TypeError, match=pattern):
            cls(**{name: None})
        with pytest.raises(TypeError, match=pattern):
            setattr(cls(), name, None)


@pytest.mark.parametrize(
    ("cls", "name", "kind", "nullable"), STRING_FIELDS, ids=STRING_IDS
)
def test_string_option_stores_the_value_in_its_own_field(
    cls, name, kind, nullable
):
    """正しい文字列は、コンストラクタでも属性代入でも同じフィールドに入り、
    他のフィールドは変わらない。
    """
    others = {n: getattr(cls(), n) for c, n, _, _ in STRING_FIELDS if c is cls}

    constructed = cls(**{name: "value_abc"})
    assigned = cls()
    setattr(assigned, name, "value_abc")

    for options in (constructed, assigned):
        assert getattr(options, name) == "value_abc"
        for other, default in others.items():
            if other != name:
                assert getattr(options, other) == default
