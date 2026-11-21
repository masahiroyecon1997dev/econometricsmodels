"""`FEIV`・`FEIVResults`（Pythonのラッパー）の公開面のテスト。

数値の正しさは`test_fe_iv_gmm_binding.py`が`_lib`を直接呼んで照合する。ここでは、ラッパーが
`_lib`の結果を公開の形（名前をキーにした辞書・`CStatisticResult`）にそろえること、
オプションの既定値、結果の型、持たないものを確かめる。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _assertions import assert_option_case_insensitive
from _helpers import CASE_SPELLINGS
from econometricsmodels import (
    FEIV,
    CStatisticResult,
    FEIVOptions,
    FEIVResults,
    _lib,
)

N_ENTITIES = 12
N_PERIODS = 5


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    n = N_ENTITIES * N_PERIODS
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    time = np.tile(np.arange(N_PERIODS), N_ENTITIES)
    alpha = rng.normal(size=N_ENTITIES)[entity]
    z1, z2, z3, x1, u = (rng.normal(size=n) for _ in range(5))
    a = 0.8 * z1 + 0.4 * z2 + 0.5 * u + alpha + rng.normal(size=n)
    b = 0.3 * z1 - 0.7 * z3 + 0.5 * u + rng.normal(size=n)
    y = a + 0.5 * b + x1 + u + alpha
    return pl.DataFrame(
        {
            "y": y,
            "a": a,
            "b": b,
            "x1": x1,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "entity": entity,
            "time": time,
        }
    )


def _fit(
    data: pl.DataFrame,
    x_endog: list[str],
    instruments: list[str] | None = None,
    **options,
) -> FEIVResults:
    return FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=x_endog,
        instruments=instruments or ["z1", "z2", "z3"],
        effects=["entity"],
        options=FEIVOptions(estimator="gmm", **options),
    ).fit()


# ── 成功パス・結果型 ─────────────────────────────────────────────────────


def test_fit_succeeds_and_returns_fe_iv_gmm_results(data):
    result = _fit(data, ["a", "b"])
    assert isinstance(result, FEIVResults)
    assert result.param_names == ["x1", "a", "b"]
    assert result.n_obs == N_ENTITIES * N_PERIODS
    assert result.n_groups == {"entity": N_ENTITIES}
    assert result.dep_var_name == "y"


def test_default_options_use_two_step_with_the_cluster_weight(data):
    result = FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=["a"],
        instruments=["z1", "z2"],
        effects=["entity"],
        options=FEIVOptions(estimator="gmm"),
    ).fit()
    assert result.cov_type == "cluster"
    assert result.gmm_type == "two_step"
    assert result.gmm_weight_type == "cluster"
    assert result.n_iter == 2
    assert result.converged is True
    assert result.stat_dist == "normal"
    assert result.stat_df is None
    assert result.wald_dist == "chi2"
    assert result.wald_df_denom is None


# ── API構造 ──────────────────────────────────────────────────────────────


def test_params_table_structure(data):
    result = _fit(data, ["a"])
    rows = result.params_table()
    assert [row["param_name"] for row in rows] == result.param_names
    for row in rows:
        name = row["param_name"]
        assert row["param"] == result.params[name]
        assert row["std_error"] == result.std_errors[name]
        assert row["test_stat"] == result.test_stats[name]
        assert row["p_value"] == result.p_values[name]
        assert (row["conf_lower"], row["conf_upper"]) == result.conf_int[name]


def test_params_std_errors_test_stats_p_values_share_keys(data):
    result = _fit(data, ["a", "b"])
    keys = list(result.params)
    assert keys == result.param_names
    for table in (
        result.std_errors,
        result.test_stats,
        result.p_values,
        result.conf_int,
        result.cov_params,
    ):
        assert list(table) == keys


def test_overid_test_is_hansen_j(data):
    result = _fit(data, ["a"], instruments=["z1", "z2", "z3"])
    assert result.overid_df == 2
    assert result.overid_statistic is not None
    assert 0.0 <= result.overid_p_value <= 1.0


def test_just_identified_model_has_no_overid_test(data):
    result = _fit(data, ["a"], instruments=["z1"])
    assert result.overid_statistic is None
    assert result.overid_p_value is None
    assert result.overid_df is None


def test_one_step_reports_no_weight_type(data):
    result = _fit(data, ["a"], gmm_type="one_step")
    assert result.gmm_type == "one_step"
    assert result.gmm_weight_type is None
    assert result.n_iter == 1


def test_iterated_reports_iterations_and_convergence(data):
    result = _fit(data, ["a"], gmm_type="iterated")
    assert result.gmm_type == "iterated"
    assert result.n_iter >= 2
    assert result.converged is True


def test_x_exog_may_be_empty(data):
    result = FEIV(
        data,
        y="y",
        x_exog=[],
        x_endog=["a"],
        instruments=["z1", "z2"],
        effects=["entity"],
        options=FEIVOptions(estimator="gmm"),
    ).fit()
    assert result.param_names == ["a"]


def test_r_squared_properties_are_floats(data):
    result = _fit(data, ["a"])
    for value in (
        result.r_squared_within,
        result.r_squared_between,
        result.r_squared_overall,
    ):
        assert isinstance(value, float)


def test_gmm_provides_the_first_stage_and_its_diagnostics(data):
    """第一段階・弱操作変数診断・Wu-Hausmanは第一段階だけに依存するので、GMMでも持つ
    （どの項目も`None`にならない）。"""
    result = _fit(data, ["a"])
    assert list(result.first_stage()) == ["a"]
    assert list(result.weak_instrument_f_statistics) == ["a"]
    for name in (
        "weak_instrument_f_df_num",
        "weak_instrument_f_df_denom",
        "wu_hausman_statistic",
        "wu_hausman_p_value",
        "wu_hausman_df_num",
        "wu_hausman_df_denom",
    ):
        assert getattr(result, name) is not None, name
    assert result.cragg_donald().statistic > 0.0
    assert result.kleibergen_paap().rk_wald_f is not None
    assert result.effective_f().statistic > 0.0


def test_gmm_provides_the_fixed_effects(data):
    """固定効果の復元もGMMで使える（GMMの係数で復元する）。"""
    result = _fit(data, ["a"])
    effects = result.fixed_effects()
    assert list(effects) == ["entity"]
    assert len(effects["entity"]) == data["entity"].n_unique()


def test_effective_f_checks_its_argument_for_gmm_too(data):
    """GMMの結果でも、`alpha`の型の誤りは`TypeError`になる。"""
    result = _fit(data, ["a"])
    with pytest.raises(TypeError):
        result.effective_f(alpha="0.05")  # type: ignore[arg-type]


def test_c_stat_is_available_for_two_stage_least_squares(data):
    """2SLSの`c_stat()`も使える（重みは`cov_type`の系統から決まる。数値の照合は
    `test_fe_iv_c_statistic_2sls.py`）。"""
    result = FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=["a"],
        instruments=["z1", "z2"],
        effects=["entity"],
    ).fit()
    assert result.c_stat(variables=["a"]).df == 1


# ── C統計量 ──────────────────────────────────────────────────────────────


def test_c_stat_wraps_the_engine_output(data):
    result = _fit(data, ["a", "b"])
    raw = result._raw.c_stat(["a"], ["z3"])
    c = result.c_stat(variables=["a"], instruments=["z3"])
    assert isinstance(c, CStatisticResult)
    assert c.statistic == raw.statistic
    assert c.p_value == raw.p_value
    assert c.df == raw.df == 2
    assert c.df_denom is None
    assert c.distribution == "chi2"
    assert c.variables == ["a"]
    assert c.instruments == ["z3"]
    assert c.statistic == pytest.approx(
        c.j_with_tested - c.j_without_tested, rel=1e-12
    )


def test_c_stat_takes_its_arguments_by_keyword(data):
    result = _fit(data, ["a"])
    with pytest.raises(TypeError, match="positional"):
        result.c_stat(["a"])  # type: ignore[misc]


def test_c_stat_reports_variables_in_the_order_of_x_endog(data):
    result = _fit(data, ["a", "b"], instruments=["z1", "z2", "z3"])
    c = result.c_stat(variables=["b", "a"])
    assert c.variables == ["a", "b"]


# ── オプション ───────────────────────────────────────────────────────────


def test_options_is_reexported_from_the_engine():
    assert FEIVOptions is _lib.FEIVOptions


def test_options_are_not_read_before_fit(data):
    """コンストラクタは検証しない。`fit()`で初めて`ValidationError`になる。"""
    estimator = FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=["a"],
        instruments=["z1", "z2"],
        effects=["entity"],
        options=FEIVOptions(estimator="gmm", cov_type="hc0"),
    )
    with pytest.raises(ValueError):
        estimator.fit()


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize(
    "cov_type", ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
)
def test_cov_type_is_case_insensitive(data, cov_type, spelling):
    """全`cov_type`で、`res.cov_type`が正規形（小文字）で返り、どの大文字小文字の綴りでも正規形と
    同一の推定結果になる。`hc0`はFEIVでは受け付けない（`test_fe_iv_gmm_validation.py`）。`dk`は
    `dk_time`が無いと`ValidationError`になるため`dk_time="time"`を渡す。"""
    kwargs = {"dk_time": "time"} if cov_type == "dk" else {}

    def fit(label: str) -> FEIVResults:
        return _fit(data, ["a", "b"], cov_type=label, **kwargs)

    assert_option_case_insensitive(fit, "cov_type", cov_type, spelling)


def test_confidence_level_changes_the_interval_width(data):
    narrow = _fit(data, ["a"], confidence_level=0.8)
    wide = _fit(data, ["a"], confidence_level=0.99)
    lower_n, upper_n = narrow.conf_int["a"]
    lower_w, upper_w = wide.conf_int["a"]
    assert upper_w - lower_w > upper_n - lower_n


def test_dk_reports_the_bandwidth_used(data):
    result = _fit(data, ["a"], cov_type="dk", dk_time="time")
    assert result.dk_bandwidth_used is not None
    assert _fit(data, ["a"]).dk_bandwidth_used is None


# ── repr ─────────────────────────────────────────────────────────────────


def test_repr_has_header_and_one_row_per_coefficient(data):
    text = repr(_fit(data, ["a", "b"]))
    lines = text.splitlines()
    assert lines[0].startswith("FEIVResults(dep_var='y'")
    assert len(lines) == 2 + 3
