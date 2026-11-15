"""ほぼ共線な設計の受け入れ境界が、OLS・WLS・IV・FE・RE・FE-IVで同じであること。

設計行列は、列を単位ノルムに揃えた条件数が約`5e7`を超えると`MATRIX_NEAR_SINGULAR`
（`matrix="DESIGN_GRAM"`）で止める（QRなら推定できても、残差が非ゼロのとき係数が静かに大きく
ずれるため。`engine/src/shared/orthonormal.rs`の`MAX_EQUILIBRATED_CONDITION`）。
境界は手法に依らず、列の単位にも依らない。条件数が約`1e12`を超えると、QRが崩れて
`DESIGN_MATRIX_SINGULAR`（列名つき）になることがある（どちらも誤りではなく、内部エラーに
ならないことを固定する）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from _errors import raises_error, raises_one_of_errors
from econometricsmodels import (
    FE,
    FEIV,
    IV,
    OLS,
    RE,
    WLS,
    ErrorCode,
)

if TYPE_CHECKING:
    from collections.abc import Callable

N_ENTITIES = 30
N_PERIODS = 10
N = N_ENTITIES * N_PERIODS


def _data(eps: float, scale: float = 1.0) -> pl.DataFrame:
    """`b = a + eps·ノイズ`（`a`と`b`がほぼ共線）。`a`・`b`は`scale`倍して単位を変える。"""
    rng = np.random.default_rng(7)
    a = rng.normal(size=N)
    z = rng.normal(size=(N, 3))
    d = z @ np.array([1.0, 0.4, 0.2]) + rng.normal(size=N)
    b = a + eps * rng.normal(size=N)
    y = 0.5 * a + 0.4 * b + 0.7 * d + rng.normal(size=N)
    return pl.DataFrame(
        {
            "y": y,
            "a": a * scale,
            "b": b * scale,
            "d": d,
            "z1": z[:, 0],
            "z2": z[:, 1],
            "z3": z[:, 2],
            "w": rng.uniform(0.5, 2.0, size=N),
            "entity": np.repeat(np.arange(N_ENTITIES), N_PERIODS),
        }
    )


_FITTERS: dict[str, Callable[[pl.DataFrame], object]] = {
    "ols": lambda df: OLS(df, "y", ["a", "b", "d"]).fit(),
    "wls": lambda df: WLS(df, "y", ["a", "b", "d"], "w").fit(),
    "iv": lambda df: IV(df, "y", ["a", "b"], ["d"], ["z1", "z2", "z3"]).fit(),
    "fe": lambda df: FE(df, "y", ["a", "b", "d"], effects=["entity"]).fit(),
    "re": lambda df: RE(df, "y", ["a", "b", "d"], effects=["entity"]).fit(),
    "fe_iv": lambda df: FEIV(
        df,
        y="y",
        x_exog=["a", "b"],
        x_endog=["d"],
        instruments=["z1", "z2", "z3"],
        effects=["entity"],
    ).fit(),
}

# 列を揃えた条件数は`eps`にほぼ反比例し、`eps = 1e-7`で約2e7、`1e-8`で約2e8（上限5e7との間に
# 2.5倍以上の余裕がある）。
_INSIDE = [1e-5, 1e-6, 1e-7]
_OUTSIDE = [1e-9, 1e-10, 1e-11]
# 条件数が約1e12を超える範囲（列の順序によって、ランク落ちか悪条件かが変わる）。
_FAR_BEYOND = [1e-12, 1e-13, 1e-14]


@pytest.mark.parametrize("method", list(_FITTERS))
@pytest.mark.parametrize("eps", _INSIDE)
def test_a_design_inside_the_limit_is_estimated_by_every_method(method, eps):
    _FITTERS[method](_data(eps))


@pytest.mark.parametrize("method", list(_FITTERS))
@pytest.mark.parametrize("eps", _OUTSIDE)
def test_a_design_outside_the_limit_raises_near_singular_in_every_method(
    method, eps
):
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="DESIGN_GRAM"):
        _FITTERS[method](_data(eps))


@pytest.mark.parametrize("method", list(_FITTERS))
@pytest.mark.parametrize("eps", _FAR_BEYOND)
def test_a_design_far_beyond_the_limit_raises_a_design_error_not_an_internal_one(
    method, eps
):
    """条件数が約1e12を超えると、QRが崩れて`DESIGN_MATRIX_SINGULAR`になることがある。
    どちらの場合も、`INTERNAL_ERROR`（弱操作変数の部分F検定の制限モデルの失敗など）には
    ならない。"""
    with raises_one_of_errors(
        {
            ErrorCode.MATRIX_NEAR_SINGULAR: {"matrix": "DESIGN_GRAM"},
            ErrorCode.DESIGN_MATRIX_SINGULAR: {},
        }
    ):
        _FITTERS[method](_data(eps))


@pytest.mark.parametrize("method", list(_FITTERS))
@pytest.mark.parametrize("scale", [1e12, 1e6, 1e-6, 1e-12])
def test_the_limit_does_not_move_with_the_unit_of_the_correlated_columns(
    method, scale
):
    """共線な2列の単位を`1e±6`倍・`1e±12`倍しても、受け入れる設計は変わらず、止める設計は
    `DESIGN_GRAM`で止まる。ランク判定も条件数のガードも、列を単位ノルムに揃えて行うため
    （生の条件数なら単位の比だけ変わり、生の`R`の相対閾値でランク判定すると、列を小さくした
    ときに先に`DESIGN_MATRIX_SINGULAR`で弾かれる）。"""
    _FITTERS[method](_data(_INSIDE[-1], scale))
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="DESIGN_GRAM"):
        _FITTERS[method](_data(_OUTSIDE[0], scale))
