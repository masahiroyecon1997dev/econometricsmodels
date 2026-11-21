"""FE-GMM（`FEIV(estimator="gmm")`）の`fixed_effects()`の照合。

固定効果自体は、構造式の`y`・実際の`X`・係数`β̂`から`α_i = ȳ_i − x̄_i'β̂`で復元する（FE-2SLSと同じ復元）。
GMMでは係数だけが違う。次を確かめる。

- 係数がFE-2SLSと同じになるGMM（`one_step`と`classical`重みの2段階。`reduction_cases()`）は、FE-2SLSの
  `fixed_effects()`と一致する。1-way・2-way・不均衡な2-way・3次元（基準化を含む）の全ケース
- 係数がFE-2SLSと違うGMM（`robust`重みの2段階）では、GMMの係数で復元する。1-wayは個体ごとの平均の
  手計算、2-wayは復元した効果が構造式の残差の正規方程式（各水準で残差から効果の和を引いた和が0）を満たす
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _fe_iv_gmm_helpers import fit_case as fit_gmm_case
from _fe_iv_helpers import fit_case as fit_2sls_case
from _fe_iv_helpers import overidentified_panel
from _tolerances import TOLERANCES
from econometricsmodels import FEIV, FEIVOptions

from benchmark.panel.fe_iv_gmm_cases import reduction_cases

_TOL = TOLERANCES["fe_iv_gmm_reduction"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

CASES = reduction_cases()
N_ENTITIES = 30


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_gmm_fixed_effects_equal_those_of_fe_2sls_for_equal_coefficients(
    case,
) -> None:
    gmm = fit_gmm_case(case).fixed_effects()
    two_sls = fit_2sls_case(case.base).fixed_effects()
    assert list(gmm) == list(two_sls), case.label
    for column, effects in two_sls.items():
        assert_dict_close(
            gmm[column],
            effects,
            f"{case.label}/{column}",
            rtol=RTOL,
            atol=ATOL,
        )


def _fit(data: pl.DataFrame, effects: list[str], estimator: str):
    options = (
        FEIVOptions(
            estimator="gmm", gmm_type="two_step", gmm_weight_type="robust"
        )
        if estimator == "gmm"
        else FEIVOptions()
    )
    return FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=["w"],
        instruments=["z1", "z2", "z3"],
        effects=effects,
        options=options,
    ).fit()


def _structural_residual(data: pl.DataFrame, params: dict[str, float]):
    return (
        data["y"].to_numpy()
        - params["x1"] * data["x1"].to_numpy()
        - params["w"] * data["w"].to_numpy()
    )


def test_one_way_effects_use_the_gmm_coefficients() -> None:
    data = overidentified_panel()
    gmm = _fit(data, ["entity"], "gmm")
    two_sls = _fit(data, ["entity"], "2sls")
    # この入力では、ロバスト重みの係数が2SLSと違う（違わなければ、GMMの係数で復元したことを確かめられない）。
    assert abs(gmm.params["w"] - two_sls.params["w"]) > 1e-4

    residual = _structural_residual(data, gmm.params)
    labels = data["entity"].to_numpy()
    effects = gmm.fixed_effects()["entity"]
    assert len(effects) == N_ENTITIES
    for group in np.unique(labels):
        assert_close(
            effects[str(group)],
            float(residual[labels == group].mean()),
            f"entity/{group}",
            rtol=RTOL,
            atol=ATOL,
        )
    # 2SLSの係数で復元した効果とは違う。
    other = two_sls.fixed_effects()["entity"]
    assert max(abs(effects[g] - other[g]) for g in effects) > 1e-4


def test_two_way_effects_solve_the_normal_equations_of_the_gmm_residual() -> (
    None
):
    data = overidentified_panel()
    gmm = _fit(data, ["entity", "time"], "gmm")
    residual = _structural_residual(data, gmm.params)
    fixed = gmm.fixed_effects()
    assert list(fixed) == ["entity", "time"]
    labels = {c: data[c].to_numpy() for c in ("entity", "time")}
    rest = residual - sum(
        np.array([fixed[c][str(g)] for g in labels[c]]) for c in labels
    )
    for column, values in labels.items():
        for group in np.unique(values):
            assert abs(rest[values == group].sum()) <= 1e-9, (column, group)
