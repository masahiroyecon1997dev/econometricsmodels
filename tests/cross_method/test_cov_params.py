"""全手法共通の`cov_params`（係数の分散共分散行列）のテスト。

形（名前→名前→値、`param_names`の順）、対称性、対角成分の平方根が`std_errors`と一致すること
を、手法と`cov_type`の組み合わせごとに確かめる。数値そのものの照合は各手法の
reference/crosscheckテストが担う。
"""

from __future__ import annotations

import math

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
    WLSOptions,
)

X = ["x1", "x2"]
IV_ARGS = {
    "y": "y",
    "x_exog": ["x1"],
    "x_endog": ["endog"],
    "instruments": ["z1", "z2"],
}


def _ols(df, **options):
    return OLS(df, y="y", x=X, options=OLSOptions(**options)).fit()


def _wls(df, **options):
    return WLS(df, y="y", x=X, weight="w", options=WLSOptions(**options)).fit()


def _logit(df, **options):
    return Logit(df, y="y_bin", x=X, options=LogitOptions(**options)).fit()


def _probit(df, **options):
    return Probit(df, y="y_bin", x=X, options=ProbitOptions(**options)).fit()


def _tobit(df, **options):
    options = {"lower": 0.5, **options}
    return Tobit(df, y="y_cens", x=X, options=TobitOptions(**options)).fit()


def _iv(df, **options):
    return IV(df, options=IVOptions(**options), **IV_ARGS).fit()


def _fe(df, **options):
    return FE(
        df, y="y", x=X, entity="entity", options=FEOptions(**options)
    ).fit()


def _re(df, **options):
    return RE(
        df, y="y", x=X, entity="entity", options=REOptions(**options)
    ).fit()


CASES = {
    "ols-classical": (_ols, {}),
    "ols-hc1": (_ols, {"cov_type": "hc1"}),
    "ols-hac": (_ols, {"cov_type": "hac", "hac_time": "row"}),
    "ols-cluster": (_ols, {"cov_type": "cluster", "cluster": "entity"}),
    "ols-no-intercept": (_ols, {"include_intercept": False}),
    "wls-classical": (_wls, {}),
    "wls-cluster": (_wls, {"cov_type": "cluster", "cluster": "entity"}),
    "logit-classical": (_logit, {}),
    "logit-cluster": (_logit, {"cov_type": "cluster", "cluster": "entity"}),
    "probit-classical": (_probit, {}),
    "probit-cluster": (_probit, {"cov_type": "cluster", "cluster": "entity"}),
    "tobit-classical": (_tobit, {}),
    "tobit-cluster": (_tobit, {"cov_type": "cluster", "cluster": "entity"}),
    "2sls-classical": (_iv, {}),
    "2sls-cluster": (_iv, {"cov_type": "cluster", "cluster": "entity"}),
    "gmm-classical": (_iv, {"estimator": "gmm"}),
    "gmm-hc1": (_iv, {"estimator": "gmm", "cov_type": "hc1"}),
    "fe-cluster": (_fe, {}),
    "fe-classical": (_fe, {"cov_type": "classical"}),
    "fe-hc2": (_fe, {"cov_type": "hc2"}),
    "fe-dk": (_fe, {"cov_type": "dk", "dk_time": "time"}),
    "fe-two-way": (_fe, {"time": "time"}),
    "re-cluster": (_re, {}),
    "re-classical": (_re, {"cov_type": "classical"}),
    "re-dk": (_re, {"cov_type": "dk", "dk_time": "time"}),
    "re-two-way": (_re, {"time": "time"}),
}


@pytest.fixture(scope="module", params=list(CASES))
def res(request, panel_df):
    fit, options = CASES[request.param]
    # HACは重複のない順序列を要る
    return fit(panel_df.with_row_index("row"), **options)


def test_cov_params_is_keyed_by_param_names_on_both_axes(res):
    assert list(res.cov_params) == res.param_names
    for row in res.cov_params.values():
        assert list(row) == res.param_names


def test_cov_params_is_symmetric(res):
    cov = res.cov_params
    for a in res.param_names:
        for b in res.param_names:
            assert cov[a][b] == pytest.approx(cov[b][a], rel=1e-9, abs=1e-14)


def test_cov_params_diagonal_is_squared_std_errors(res):
    for name, se in res.std_errors.items():
        assert math.sqrt(res.cov_params[name][name]) == pytest.approx(
            se, rel=1e-10
        )


def test_cov_params_gives_standard_error_of_a_linear_combination(panel_df):
    """`b1 + b2`の標準誤差を`cov_params`から組み立てられる（利用者の主な用途）。

    `y = a + (b1 + b2) x1 + b2 (x2 - x1)`と書き直した回帰では、`x1`の係数が`b1 + b2`なので、
    その標準誤差が`cov_params`から組み立てた値と一致する。
    """
    res = _ols(panel_df)
    cov = res.cov_params
    var = cov["x1"]["x1"] + cov["x2"]["x2"] + 2.0 * cov["x1"]["x2"]
    reparametrized = OLS(
        panel_df.with_columns(diff=panel_df["x2"] - panel_df["x1"]),
        y="y",
        x=["x1", "diff"],
    ).fit()
    assert reparametrized.params["x1"] == pytest.approx(
        res.params["x1"] + res.params["x2"], rel=1e-9
    )
    assert math.sqrt(var) == pytest.approx(
        reparametrized.std_errors["x1"], rel=1e-9
    )


def test_tobit_cov_params_includes_sigma(panel_df):
    res = _tobit(panel_df)
    assert res.param_names[-1] == "sigma"
    assert list(res.cov_params["sigma"]) == res.param_names
