"""Tobitの、ほぼ共線な傾きの受け入れ境界と、全体Wald統計量の精度。

Tobitは`-H`（対数尤度のHessian。`X'WX`の形で条件数が設計行列の2乗になる）を、対角で揃えた
条件数の平方根が`1e6`以内のときだけ受け入れ（`engine/src/shared/orthonormal.rs`の
`MAX_INFORMATION_CONDITION`）、超えれば`MATRIX_NEAR_SINGULAR`（`matrix="HESSIAN"`）で止める。
他の手法の設計行列の上限`5e7`にそろえないのは、`-H`の各要素の丸めで統計量の誤差が条件数に
比例して増え、`5e7`の近くでは桁違いに外れても分解が失敗しないため（`eps = 1e-8`で、Waldが
`735`のところ`764`になった）。`cov_type="opg"`は`-H`を使わず、スコア行列の列を揃えた条件数
（上限`5e7`）で判定する（`matrix="OPG"`）。

統計量は直交座標（`-H = R'R`の`R`で`b̃ = Rθ`・`V = R⁻¹M̃R⁻ᵀ`）で計算するので、`-H⁻¹`と
スコアの外積和を明示的に作ってかけ合わせる従来の式で失っていた桁（`hc0`・`hc1`・cluster）が
戻る。同じ`H`・`S`を入力にした有理数の厳密値との照合は、`engine`の単体テスト
（`orthogonal_wald_matches_exact_rational_values_for_nearly_collinear_information`）が持つ。
ここでは、モデルを丸めなしに再パラメータ化した設計との一致で、`fit()`全体の精度を確かめる。
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
# `-H`を使う`cov_type`（`opg`は別の境界）。
HESSIAN_COV_TYPES = ["classical", "hc0", "hc1", "cluster"]

assert_invariant = partial(
    assert_close, **TOLERANCES["near_collinear_tobit_reparametrization"]
)
assert_edge = partial(assert_close, **TOLERANCES["near_collinear_tobit_edge"])
assert_unit = partial(assert_close, **TOLERANCES["near_collinear_tobit_unit"])
assert_solver = partial(
    assert_close, **TOLERANCES["near_collinear_tobit_solver"]
)

# `-H`の列を揃えた条件数の平方根は`eps`にほぼ反比例し、`eps = 1e-6`で約`1.8e6`。上限`1e6`の
# 境界は`eps ≈ 1.8e-6`で、`2.5e-6`（約`7e5`）は内側、`1.2e-6`（約`1.5e6`）は外側。
_INSIDE = [1e-4, 1e-5]
_NEAR_INSIDE = 2.5e-6
_NEAR_OUTSIDE = 1.2e-6
_OUTSIDE = [1e-7, 1e-8, 1e-10]


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


@pytest.mark.parametrize("cov_type", HESSIAN_COV_TYPES)
@pytest.mark.parametrize("eps", [_NEAR_OUTSIDE, *_OUTSIDE])
@pytest.mark.parametrize("scale", [1.0, 1e6, 1e-6])
def test_a_design_outside_the_limit_raises_near_singular_hessian(
    cov_type, eps, scale
):
    """境界の近く（上限の1.5倍）から遠くまで、同じ`HESSIAN`で止まる（単位に依らない。
    列を対角で揃えてから判定するため）。収束が先に止まることはない。"""
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="HESSIAN"):
        _fit(_data(eps, scale), cov_type)


def test_a_design_far_beyond_the_limit_is_reported_as_singular():
    """条件数が`1e12`を超える（完全な共線）設計は、`HESSIAN`か、列名つきの
    `DESIGN_MATRIX_SINGULAR`で止まる（内部エラーにならない）。"""
    with raises_one_of_errors(
        {
            ErrorCode.MATRIX_NEAR_SINGULAR: {"matrix": "HESSIAN"},
            ErrorCode.DESIGN_MATRIX_SINGULAR: {},
        }
    ):
        _fit(_data(1e-13), "classical")


@pytest.mark.parametrize("eps", [1e-6, 1e-7, 5e-8])
def test_opg_has_its_own_wider_limit_through_the_score_matrix(eps):
    """`opg`は`-H`を使わないので、スコア行列の条件数（約`1/eps`）の上限`5e7`で判定する
    （`eps ≈ 4e-8`まで）。"""
    assert np.isfinite(_fit(_data(eps), "opg").wald_statistic)


@pytest.mark.parametrize("eps", [3e-8, 1e-8, 1e-10])
def test_opg_outside_its_limit_raises_near_singular_opg(eps):
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="OPG"):
        _fit(_data(eps), "opg")


@pytest.mark.parametrize("cov_type", HESSIAN_COV_TYPES)
@pytest.mark.parametrize(
    ("scale", "scale_b"), [(1e6, 1.0), (1.0, 1e-6), (1e12, 1.0)]
)
def test_the_limit_does_not_move_when_only_one_column_is_rescaled(
    cov_type, scale, scale_b
):
    _fit(_data(_NEAR_INSIDE, scale, scale_b), cov_type)
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="HESSIAN"):
        _fit(_data(_NEAR_OUTSIDE, scale, scale_b), cov_type)


# ── 精度 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("eps", _INSIDE)
def test_the_overall_wald_is_invariant_to_reparametrization(cov_type, eps):
    df = _data(eps)
    got = _statistics(_fit(df, cov_type))
    expected = _statistics(_fit(_reparametrize(df, eps), cov_type))
    for name in got:
        assert_invariant(got[name], expected[name], f"{cov_type} {name}")


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_the_overall_wald_is_invariant_to_reparametrization_at_the_limit(
    cov_type,
):
    df = _data(_NEAR_INSIDE)
    got = _statistics(_fit(df, cov_type))
    expected = _statistics(_fit(_reparametrize(df, _NEAR_INSIDE), cov_type))
    for name in got:
        assert_edge(got[name], expected[name], f"{cov_type} {name}")


@pytest.mark.parametrize("solver", ["bfgs", "lbfgs"])
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
def test_the_overall_wald_is_invariant_with_every_solver(cov_type, solver):
    """Newton以外のソルバー（収束点が`tol`の分だけずれる）でも、受け入れ範囲の内側の統計量は
    再パラメータ化に依らない。"""
    df = _data(1e-4)
    got = _fit(df, cov_type, solver).wald_statistic
    expected = _fit(_reparametrize(df, 1e-4), cov_type, solver).wald_statistic
    assert_solver(got, expected, f"{cov_type} {solver}")


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize(
    ("scale", "scale_b"),
    [(1e6, 1e6), (1e-6, 1e-6), (1e6, 1.0), (1.0, 1e-6), (1e12, 1.0)],
)
def test_the_overall_wald_does_not_depend_on_the_unit_of_a_column(
    cov_type, scale, scale_b
):
    expected = _fit(_data(1e-5), cov_type).wald_statistic
    got = _fit(_data(1e-5, scale, scale_b), cov_type).wald_statistic
    assert_unit(got, expected, f"{cov_type} scale=({scale}, {scale_b})")


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("eps", _INSIDE)
def test_the_overall_wald_is_invariant_without_an_intercept(cov_type, eps):
    """切片なし（検定しない先頭の列が`logσ`だけ）でも、直交座標の並びと検定の範囲が合っている。
    Waldが小さい（約47）ので相対誤差が大きめ（`eps = 1e-5`のclusterで6e-7）で、緩い許容誤差を使う。"""
    df = _data(eps).with_columns((pl.col("y") + 2.0).alias("y"))
    got = _statistics(_fit(df, cov_type, include_intercept=False))
    expected = _statistics(
        _fit(_reparametrize(df, eps), cov_type, include_intercept=False)
    )
    for name in got:
        assert_edge(got[name], expected[name], f"{cov_type} {name}")
