"""全手法共通の、推定オプションの数値フィールドの型・値の検証のテスト。

- 型の誤り（`bool`・文字列・`None`・整数に`float`）は`TypeError`
  （コンストラクタ引数と属性代入の両方）。`bool`は`int`のサブクラスだが、数値として渡す
  意図はないため拒否する。
- 値の誤り（範囲外・NaN・巨大な値）は`fit()`時の`ValidationError`。`i64`に収まらない
  巨大な整数は`OverflowError`ではなく、`i64`の端に丸められて範囲検査に到達する。
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest
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


# ── 値の誤り: fit()時のValidationError ──────────────────────────────


@pytest.mark.parametrize("huge", [2**70, -(2**70)], ids=["+2^70", "-2^70"])
def test_huge_hac_lags_is_validation_error_not_overflow(df, huge):
    """`hac_lags`が`i64`に収まらなくても`OverflowError`ではなく`ValidationError`。"""
    ols = OLSOptions(cov_type="hac", hac_lags=huge)
    assert ols.hac_lags in (I64_MAX, I64_MIN)
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1"], options=ols).fit()

    wls = WLSOptions(cov_type="hac", hac_lags=huge)
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1"], weight="w", options=wls).fit()

    iv = IVOptions(cov_type="hac", hac_lags=huge)
    with pytest.raises(ValidationError):
        IV(
            df,
            y="y",
            x_exog=["x2"],
            x_endog=["x1"],
            instruments=["z"],
            options=iv,
        ).fit()


@pytest.mark.parametrize("huge", [2**70, -(2**70)], ids=["+2^70", "-2^70"])
@pytest.mark.parametrize("cls", [FEOptions, REOptions], ids=["FE", "RE"])
def test_huge_dk_bandwidth_is_validation_error(df, cls, huge):
    model = FE if cls is FEOptions else RE
    options = (
        cls(cov_type="dk", dk_time="t", dk_bandwidth=huge)
        if (cls is FEOptions)
        else cls(cov_type="dk", time="t", dk_bandwidth=huge)
    )

    with pytest.raises(ValidationError):
        model(df, y="y", x=["x1"], entity="entity", options=options).fit()


@pytest.mark.parametrize(
    ("model", "options_cls", "y"),
    [
        (Logit, LogitOptions, "y01"),
        (Probit, ProbitOptions, "y01"),
        (Tobit, TobitOptions, "yc"),
    ],
    ids=["Logit", "Probit", "Tobit"],
)
class TestMleOptionValues:
    def test_negative_huge_max_iter_is_validation_error(
        self, df, model, options_cls, y
    ):
        options = options_cls(max_iter=-(2**70))

        with pytest.raises(ValidationError):
            model(df, y=y, x=["x1"], options=options).fit()

    def test_huge_max_iter_is_accepted_as_unbounded(
        self, df, model, options_cls, y
    ):
        """大きな正の`max_iter`は有効な値（実質無制限）で、収束すれば通常どおり返る。"""
        options = options_cls(max_iter=2**70)

        result = model(df, y=y, x=["x1"], options=options).fit()

        assert result.converged

    @pytest.mark.parametrize(
        "tol",
        [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 10**400],
        ids=["nan", "inf", "-inf", "zero", "negative", "huge_int"],
    )
    def test_invalid_tol_is_validation_error(
        self, df, model, options_cls, y, tol
    ):
        """`tol`がNaN・無限大・0以下・巨大な値のときは`ValidationError`（NaNは
        以前は収束失敗の`ComputationError`になっていた）。"""
        options = options_cls(tol=tol)

        with pytest.raises(ValidationError, match="tol must be a positive"):
            model(df, y=y, x=["x1"], options=options).fit()


@pytest.mark.parametrize(
    "confidence_level",
    [float("nan"), float("inf"), 2**70, -(2**70), 10**400, 0, 1],
    ids=["nan", "inf", "2^70", "-2^70", "huge_int", "zero", "one"],
)
def test_invalid_confidence_level_is_validation_error(df, confidence_level):
    options = OLSOptions(confidence_level=confidence_level)

    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1"], options=options).fit()


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf")], ids=["nan", "inf"]
)
@pytest.mark.parametrize("field", ["lower", "upper"])
def test_invalid_tobit_bound_is_validation_error(df, field, bad):
    """Tobitの打ち切り境界がNaN・無限大のときは`ValidationError`。"""
    options = TobitOptions(**{field: bad})

    with pytest.raises(ValidationError):
        Tobit(df, y="yc", x=["x1"], options=options).fit()
