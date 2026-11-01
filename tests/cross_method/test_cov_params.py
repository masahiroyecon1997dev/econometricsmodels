"""全手法共通の`cov_params`（係数の分散共分散行列）のテスト。

形（名前→名前→値、`param_names`の順）、対称性、対角成分の平方根が`std_errors`と一致すること
を、手法と`cov_type`の組み合わせごとに確かめる。組み合わせは、各手法が受け付ける`cov_type`の
全体（`COV_TYPES_BY_METHOD`。パッケージの受け付ける値と食い違うと失敗する）。数値そのものの照合は
各手法のreference/crosscheckテストが担う。
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
    ValidationError,
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


def _fe(df, *, two_way=False, **options):
    effects = ["entity", "time"] if two_way else ["entity"]
    return FE(
        df, y="y", x=X, effects=effects, options=FEOptions(**options)
    ).fit()


def _re(df, *, two_way=False, **options):
    effects = ["entity", "time"] if two_way else ["entity"]
    return RE(
        df, y="y", x=X, effects=effects, options=REOptions(**options)
    ).fit()


# 手法（と、結果の形が変わるオプション）ごとの、受け付ける`cov_type`の全体。
# `test_cov_types_cover_every_value_the_option_accepts`が、パッケージが受け付ける値と
# 食い違うと失敗する（`cov_type`を足したら、ここにも足す）。
COV_TYPES_BY_METHOD = {
    "ols": ("classical", "hc0", "hc1", "hc2", "hc3", "hac", "cluster"),
    "ols-no-intercept": (
        "classical",
        "hc0",
        "hc1",
        "hc2",
        "hc3",
        "hac",
        "cluster",
    ),
    "wls": ("classical", "hc0", "hc1", "hc2", "hc3", "hac", "cluster"),
    "logit": ("classical", "opg", "hc0", "hc1", "cluster"),
    "probit": ("classical", "opg", "hc0", "hc1", "cluster"),
    "tobit": ("classical", "opg", "hc0", "hc1", "cluster"),
    "2sls": ("classical", "hc0", "hc1", "hc2", "hc3", "hac", "cluster"),
    "gmm": ("classical", "hc0", "hc1", "hc2", "hc3", "hac", "cluster"),
    "fe": ("classical", "hc1", "hc2", "hc3", "cluster", "dk"),
    "fe-two-way": ("classical", "hc1", "hc2", "hc3", "cluster", "dk"),
    "re": ("classical", "hc1", "hc2", "hc3", "cluster", "dk"),
    "re-two-way": ("classical", "hc1", "hc2", "hc3", "cluster", "dk"),
}

# 手法ごとの、`cov_type`以外の固定のオプション。
METHOD_OPTIONS = {
    "ols": (_ols, {}),
    "ols-no-intercept": (_ols, {"include_intercept": False}),
    "wls": (_wls, {}),
    "logit": (_logit, {}),
    "probit": (_probit, {}),
    "tobit": (_tobit, {}),
    "2sls": (_iv, {}),
    "gmm": (_iv, {"estimator": "gmm"}),
    "fe": (_fe, {}),
    "fe-two-way": (_fe, {"two_way": True}),
    "re": (_re, {}),
    "re-two-way": (_re, {"two_way": True}),
}

# `cov_type`が列を要求するときに足すオプション（HACは重複のない順序列、DKは時点列）。
COV_TYPE_OPTIONS = {
    "hac": {"hac_time": "row"},
    "cluster": {"cluster": "entity"},
    "dk": {"dk_time": "time"},
}

CASES = {
    f"{method}-{cov_type}": (
        METHOD_OPTIONS[method][0],
        {
            **METHOD_OPTIONS[method][1],
            "cov_type": cov_type,
            **COV_TYPE_OPTIONS.get(cov_type, {}),
        },
    )
    for method, cov_types in COV_TYPES_BY_METHOD.items()
    for cov_type in cov_types
}


@pytest.fixture(scope="module", params=list(CASES))
def res(request, panel_df):
    fit, options = CASES[request.param]
    # HACは重複のない順序列を要る
    return fit(panel_df.with_row_index("row"), **options)


@pytest.mark.parametrize("method", list(COV_TYPES_BY_METHOD))
def test_cov_types_cover_every_value_the_option_accepts(method, panel_df):
    """表の`cov_type`が、パッケージが受け付ける値の全体と一致する。

    未知の値を渡したときのエラーが、受け付ける値の一覧を`allowed`に持つ。
    """
    fit, options = METHOD_OPTIONS[method]
    with pytest.raises(ValidationError) as caught:
        fit(
            panel_df.with_row_index("row"),
            **{**options, "cov_type": "not-a-cov-type"},
        )
    assert sorted(caught.value.params["allowed"]) == sorted(
        COV_TYPES_BY_METHOD[method]
    ), f"{method}: update COV_TYPES_BY_METHOD"


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
