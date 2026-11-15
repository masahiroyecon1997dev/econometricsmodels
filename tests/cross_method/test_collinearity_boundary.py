"""ほぼ共線な設計の受け入れ境界が、OLS・WLS・IV・FE・RE・FE-IVで同じであること。

設計行列は、列を単位ノルムに揃えた条件数が約`5e7`を超えると`MATRIX_NEAR_SINGULAR`
（`matrix="DESIGN_GRAM"`）で止める（QRなら推定できても、残差が非ゼロのとき係数が静かに大きく
ずれるため。`engine/src/shared/orthonormal.rs`の`MAX_EQUILIBRATED_CONDITION`）。
境界は手法に依らず、列の単位にも依らない。条件数が約`1e12`を超えると、QRが崩れて
`DESIGN_MATRIX_SINGULAR`（列名つき）になることがある（どちらも誤りではなく、内部エラーに
ならないことを固定する）。
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _errors import raises_error, raises_one_of_errors
from _tolerances import TOLERANCES
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

assert_invariant = partial(
    assert_close, **TOLERANCES["near_collinear_reparametrization"]
)

N_ENTITIES = 30
N_PERIODS = 10
N = N_ENTITIES * N_PERIODS


def _data(
    eps: float, scale: float = 1.0, scale_b: float | None = None
) -> pl.DataFrame:
    """`b = a + eps·ノイズ`（`a`と`b`がほぼ共線）。`a`は`scale`倍、`b`は`scale_b`倍（省略時は
    `scale`倍）して単位を変える。"""
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
            "b": b * (scale if scale_b is None else scale_b),
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
# 境界の近く: 列を揃えた条件数は、`eps = 5e-8`で約4e7（上限5e7の内側、約1.2倍の余裕）、
# `eps = 3e-8`で約9e7（外側、約1.8倍）。手法ごとの前処理（WLSの重み、FEのwithin変換、REの`θ`、
# IVの`X̂`）が境界を桁単位でずらさないことを、この幅で固定する。FEとREは設計行列そのものでなく
# within・準偏差化した後の設計で判定されるので、境界が少しずれる。このデータでの境界は
# OLS・WLS・IV・FE-IVが約`3.8e-8`、FEが約`3.5e-8`、REが約`4.6e-8`（REは内側の`5e-8`との余裕が
# 約8%、他は約30%）。条件数の計算誤差は約1e-9で、他のOSでも合否は変わらない。
_NEAR_INSIDE = 5e-8
_NEAR_OUTSIDE = 3e-8
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


@pytest.mark.parametrize("method", list(_FITTERS))
def test_the_limit_is_the_same_close_to_the_boundary_in_every_method(method):
    """境界の近く（上限の0.8倍と1.8倍の条件数）でも、全手法が同じ側に分かれる。"""
    _FITTERS[method](_data(_NEAR_INSIDE))
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="DESIGN_GRAM"):
        _FITTERS[method](_data(_NEAR_OUTSIDE))


@pytest.mark.parametrize("method", list(_FITTERS))
@pytest.mark.parametrize(
    ("scale", "scale_b"),
    [(1e12, 1.0), (1.0, 1e12), (1e-12, 1.0), (1.0, 1e-12), (1e6, 1e-6)],
)
def test_the_limit_does_not_move_when_only_one_correlated_column_is_rescaled(
    method, scale, scale_b
):
    """共線な対の片方だけの単位を変えても（比が`1e±12`）、境界は動かない。"""
    _FITTERS[method](_data(_INSIDE[-1], scale, scale_b))
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR, matrix="DESIGN_GRAM"):
        _FITTERS[method](_data(_OUTSIDE[0], scale, scale_b))


def _overall_statistic(res: object) -> float:
    """全体のF統計量（OLS・WLS・FE・RE）か、Wald統計量（IV・FE-IV）。"""
    if hasattr(res, "f_statistic"):
        return res.f_statistic
    return res.wald_statistic * res.wald_df_num


@pytest.mark.parametrize("method", list(_FITTERS))
@pytest.mark.parametrize(
    ("scale", "scale_b"), [(1e6, 1e6), (1e-6, 1e-6), (1e6, 1.0), (1.0, 1e-6)]
)
def test_the_overall_statistic_does_not_depend_on_the_unit_of_a_column(
    method, scale, scale_b
):
    """列の単位を変えても、全体のF・Wald統計量は変わらない（2の冪でない倍率なので丸めは
    入るが、受け入れ範囲の内側（`eps = 1e-5`）では桁落ちは小さい）。"""
    expected = _overall_statistic(_FITTERS[method](_data(1e-5)))
    got = _overall_statistic(_FITTERS[method](_data(1e-5, scale, scale_b)))
    assert_invariant(got, expected, f"{method} scale=({scale}, {scale_b})")


# ── 操作変数側のほぼ共線（IV・FE-IV） ─────────────────────────────────
#
# 第一段階の設計行列`[x_exog, 操作変数]`と、第二段階の`X̂`（第一段階の当てはめ値）にも、
# 同じ限界を適用する。どの段階で止まったかは`stage`で分かる。


def _instrument_data(eps: float, which: str) -> pl.DataFrame:
    """`which`は、`z2 ≈ z1`（`"zz"`）、`z3 ≈ a`（外生変数との共線、`"za"`）、内生変数`d2 ≈ d1`
    （第一段階の設計は良条件だが`X̂`がほぼ共線、`"dd"`）。"""
    rng = np.random.default_rng(7)
    a = rng.normal(size=N)
    z = rng.normal(size=(N, 3))
    if which == "zz":
        z[:, 1] = z[:, 0] + eps * rng.normal(size=N)
    elif which == "za":
        z[:, 2] = a + eps * rng.normal(size=N)
    d1 = z @ np.array([1.0, 0.4, 0.2]) + rng.normal(size=N)
    d2 = d1 + eps * rng.normal(size=N) if which == "dd" else d1
    y = 0.5 * a + 0.7 * d1 + 0.3 * d2 + rng.normal(size=N)
    frame = {
        "y": y,
        "a": a,
        "d1": d1,
        "z1": z[:, 0],
        "z2": z[:, 1],
        "z3": z[:, 2],
        "entity": np.repeat(np.arange(N_ENTITIES), N_PERIODS),
    }
    if which == "dd":
        frame["d2"] = d2
    return pl.DataFrame(frame)


def _fit_iv_side(method: str, df: pl.DataFrame, endog: list[str]) -> object:
    if method == "iv":
        return IV(df, "y", ["a"], endog, ["z1", "z2", "z3"]).fit()
    return FEIV(
        df,
        y="y",
        x_exog=["a"],
        x_endog=endog,
        instruments=["z1", "z2", "z3"],
        effects=["entity"],
    ).fit()


@pytest.mark.parametrize("method", ["iv", "fe_iv"])
@pytest.mark.parametrize("which", ["zz", "za"])
@pytest.mark.parametrize("eps", _INSIDE)
def test_almost_collinear_instruments_inside_the_limit_are_estimated(
    method, which, eps
):
    _fit_iv_side(method, _instrument_data(eps, which), ["d1"])


@pytest.mark.parametrize("method", ["iv", "fe_iv"])
@pytest.mark.parametrize("which", ["zz", "za"])
@pytest.mark.parametrize("eps", _OUTSIDE)
def test_almost_collinear_instruments_outside_the_limit_stop_in_the_first_stage(
    method, which, eps
):
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR,
        matrix="DESIGN_GRAM",
        stage="FIRST_STAGE",
    ):
        _fit_iv_side(method, _instrument_data(eps, which), ["d1"])


@pytest.mark.parametrize("method", ["iv", "fe_iv"])
@pytest.mark.parametrize("which", ["zz", "za"])
def test_instruments_far_beyond_the_limit_raise_a_first_stage_design_error(
    method, which
):
    with raises_one_of_errors(
        {
            ErrorCode.MATRIX_NEAR_SINGULAR: {
                "matrix": "DESIGN_GRAM",
                "stage": "FIRST_STAGE",
            },
            ErrorCode.DESIGN_MATRIX_SINGULAR: {"stage": "FIRST_STAGE"},
        }
    ):
        _fit_iv_side(method, _instrument_data(1e-13, which), ["d1"])


@pytest.mark.parametrize("method", ["iv", "fe_iv"])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_almost_collinear_fitted_endogenous_regressors_are_estimated(
    method, eps
):
    _fit_iv_side(method, _instrument_data(eps, "dd"), ["d1", "d2"])


@pytest.mark.parametrize("method", ["iv", "fe_iv"])
@pytest.mark.parametrize("eps", [1e-8, 1e-10, 1e-11])
def test_almost_collinear_fitted_endogenous_regressors_stop_in_the_second_stage(
    method, eps
):
    """第一段階の設計行列は良条件で、第二段階の`X̂`だけがほぼ共線のとき、第二段階で止まる
    （当てはめ値の差は、内生変数の差のうち操作変数で説明される部分だけなので、`X`より共線が強い）。"""
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR,
        matrix="DESIGN_GRAM",
        stage="SECOND_STAGE",
    ):
        _fit_iv_side(method, _instrument_data(eps, "dd"), ["d1", "d2"])
