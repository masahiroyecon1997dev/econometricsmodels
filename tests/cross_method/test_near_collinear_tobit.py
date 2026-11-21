"""Tobitの、ほぼ共線な傾きの受け入れ境界と、全体Wald統計量・係数・標準誤差の精度。

Tobitは標準化した設計行列を`X̃ = QR`に分解し、`Q`の座標`b = Rβ̃`で最適化する
（`engine/src/nonlinear/tobit/coordinates.rs`）。`Q`の座標では`-H`とスコアの条件数が共線性に依らず、
`X`の座標で`H`を組む丸め（`cond(X)²`）が無いので、受け入れ範囲も精度もOLS系と同じ水準になる。
受け入れ境界は他の手法と同じ（列を単位ノルムに揃えた設計の条件数`5e7`を超えると
`MATRIX_NEAR_SINGULAR`の`DESIGN_GRAM`、`1e12`を超えるか完全な共線なら列名つきの
`DESIGN_MATRIX_SINGULAR`。`test_collinearity_boundary.py`）。`cov_type`によらず同じ。

同じ`H`・`S`を入力にした有理数の厳密値との照合は、`engine`の単体テスト
（`overall_wald_in_q_coordinates_matches_exact_rational_values`）が持つ。ここでは、モデルを丸めなしに
再パラメータ化した設計との一致で、`fit()`全体の精度を確かめる。
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _errors import raises_error, raises_one_of_errors
from _tolerances import TOLERANCES
from econometricsmodels import ErrorCode, Tobit, TobitOptions

N = 600
COV_TYPES = ["classical", "opg", "hc0", "hc1", "cluster"]

assert_invariant = partial(
    assert_close, **TOLERANCES["near_collinear_tobit_reparametrization"]
)
assert_slope = partial(
    assert_close, **TOLERANCES["near_collinear_tobit_slope"]
)
assert_unit = partial(assert_close, **TOLERANCES["near_collinear_tobit_unit"])

# 列を揃えた設計の条件数は`eps`にほぼ反比例し、上限`5e7`の境界は`eps ≈ 3.7e-8`
# （`test_collinearity_boundary.py`のOLSなどと同じ位置）。`5e-8`は内側（約1.35倍の余裕）、
# `3e-8`は外側（約1.2倍）。
_INSIDE = [1e-5, 1e-6, 1e-7]
_NEAR_INSIDE = 5e-8
_NEAR_OUTSIDE = 3e-8
_OUTSIDE = [1e-9, 1e-10]
# 条件数が約1e12を超える範囲（完全な共線として扱う）。
_FAR_BEYOND = [1e-12, 1e-13, 1e-14]


def _data(
    eps: float, scale: float = 1.0, scale_b: float | None = None
) -> pl.DataFrame:
    """`x2 = x1 + eps·ノイズ`。左打ち切り（0）が約3割。`x1`は`scale`倍、`x2`は`scale_b`倍
    （省略時は`scale`倍）して単位を変える。"""
    rng = np.random.default_rng(4)
    x1 = rng.normal(size=N)
    x2 = x1 + eps * rng.normal(size=N)
    y = np.maximum(0.3 + x1 + 0.5 * x2 + rng.normal(size=N), 0.0)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1 * scale,
            "x2": x2 * (scale if scale_b is None else scale_b),
            "g": (np.arange(N) % 20).astype(str),
        }
    )


def _fit(
    df: pl.DataFrame,
    cov_type: str,
    solver: str = "newton",
    include_intercept: bool = True,
) -> object:
    options = TobitOptions(
        lower=0.0,
        include_intercept=include_intercept,
        cov_type=cov_type,
        cluster="g" if cov_type == "cluster" else None,
        solver=solver,
    )
    return Tobit(df, "y", ["x1", "x2"], options=options).fit()


def _reparametrize(df: pl.DataFrame, eps: float) -> pl.DataFrame:
    """`x2`を`(x2 - x1) / 2^k`に置き換える（`2^k`は`eps`に最も近い2の冪）。"""
    scale = 2.0 ** round(math.log2(eps))
    return df.with_columns(((pl.col("x2") - pl.col("x1")) / scale).alias("x2"))


def _scale(eps: float) -> float:
    return 2.0 ** round(math.log2(eps))


def _statistics(res: object) -> dict[str, float]:
    """傾き2列の置き換えで変わらない量（傾きの係数と標準誤差は変わるので含めない）。"""
    return {
        "wald": res.wald_statistic,
        "sigma": res.params["sigma"],
        "sigma_se": res.std_errors["sigma"],
    }


# ── 受け入れ境界 ──────────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("eps", [*_INSIDE, _NEAR_INSIDE])
@pytest.mark.parametrize("scale", [1.0, 1e6, 1e-6])
def test_a_design_inside_the_limit_is_estimated_in_any_unit(
    cov_type, eps, scale
):
    res = _fit(_data(eps, scale), cov_type)
    assert np.isfinite(res.wald_statistic)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("eps", [_NEAR_OUTSIDE, *_OUTSIDE])
@pytest.mark.parametrize("scale", [1.0, 1e6, 1e-6])
def test_a_design_outside_the_limit_raises_near_singular_design_gram(
    cov_type, eps, scale
):
    """境界の近く（上限の約1.2倍）から遠くまで、同じ`DESIGN_GRAM`で止まる（単位に依らない。
    列を単位ノルムに揃えて判定するため）。`cov_type`にもよらない。"""
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="DESIGN_GRAM"):
        _fit(_data(eps, scale), cov_type)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("eps", _FAR_BEYOND)
def test_a_design_far_beyond_the_limit_raises_a_design_error_not_an_internal_one(
    cov_type, eps
):
    """条件数が約1e12を超えると、完全な共線として`DESIGN_MATRIX_SINGULAR`（列名つき）になる
    （境の近くは`DESIGN_GRAM`でもよい。`INTERNAL_ERROR`にはならない）。"""
    with raises_one_of_errors(
        {
            ErrorCode.MATRIX_NEAR_SINGULAR: {"matrix": "DESIGN_GRAM"},
            ErrorCode.DESIGN_MATRIX_SINGULAR: {},
        }
    ):
        _fit(_data(eps), cov_type)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize(
    ("scale", "scale_b"), [(1e6, 1.0), (1.0, 1e-6), (1e12, 1.0)]
)
def test_the_limit_does_not_move_when_only_one_column_is_rescaled(
    cov_type, scale, scale_b
):
    _fit(_data(_NEAR_INSIDE, scale, scale_b), cov_type)
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="DESIGN_GRAM"):
        _fit(_data(_NEAR_OUTSIDE, scale, scale_b), cov_type)


def test_exact_collinearity_is_reported_with_the_columns():
    df = _data(1e-5).with_columns((2.0 * pl.col("x1")).alias("x2"))
    with raises_error(ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["x1", "x2"]):
        _fit(df, "classical")


# ── 精度 ────────────────────────────────────────────────────────────────


def _compare_with_reparametrization(
    eps: float, cov_type: str, solver: str, include_intercept: bool
) -> None:
    df = _data(eps)
    if not include_intercept:
        df = df.with_columns((pl.col("y") + 2.0).alias("y"))
    got = _fit(df, cov_type, solver, include_intercept)
    expected = _fit(
        _reparametrize(df, eps), cov_type, solver, include_intercept
    )
    label = f"{cov_type} {solver} eps={eps:g}"
    stats_got, stats_expected = _statistics(got), _statistics(expected)
    for name in stats_got:
        assert_invariant(
            stats_got[name], stats_expected[name], f"{label} {name}"
        )
    # 第2の傾きは、置き換えた設計では`2^k`倍の係数・標準誤差になる。
    scale = _scale(eps)
    assert_slope(
        got.params["x2"], expected.params["x2"] / scale, f"{label} coef x2"
    )
    assert_slope(
        got.std_errors["x2"],
        expected.std_errors["x2"] / scale,
        f"{label} se x2",
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("eps", [*_INSIDE, _NEAR_INSIDE])
def test_the_overall_wald_and_slopes_are_invariant_to_reparametrization(
    cov_type, eps
):
    _compare_with_reparametrization(eps, cov_type, "newton", True)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("eps", [1e-5, 1e-7])
def test_the_overall_wald_is_invariant_without_an_intercept(cov_type, eps):
    """切片なし（検定しない先頭の列が無い）でも、`Q`の座標と検定の範囲が合っている。"""
    _compare_with_reparametrization(eps, cov_type, "newton", False)


@pytest.mark.parametrize("solver", ["bfgs", "lbfgs"])
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
@pytest.mark.parametrize("eps", [1e-5, 1e-7])
def test_the_overall_wald_is_invariant_with_every_solver(
    cov_type, solver, eps
):
    """Newton以外のソルバー（収束点が`tol`の分だけずれる）でも、同じ精度で再パラメータ化に
    依らない。"""
    _compare_with_reparametrization(eps, cov_type, solver, True)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("eps", [1e-5, 1e-7])
@pytest.mark.parametrize(
    ("scale", "scale_b"),
    [(1e6, 1e6), (1e-6, 1e-6), (1e6, 1.0), (1.0, 1e-6), (1e12, 1.0)],
)
def test_the_overall_wald_does_not_depend_on_the_unit_of_a_column(
    cov_type, eps, scale, scale_b
):
    expected = _fit(_data(eps), cov_type).wald_statistic
    got = _fit(_data(eps, scale, scale_b), cov_type).wald_statistic
    assert_unit(got, expected, f"{cov_type} scale=({scale}, {scale_b})")
