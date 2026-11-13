"""`FEIV`・`FEIVResults`（Pythonのラッパー）の公開面のテスト。

数値の正しさは`test_fe_iv_binding.py`が`_lib`を直接呼んで照合する。ここでは、ラッパーが
`_lib`の結果を公開の形（`x_endog`の順の辞書・`FEResults`・診断のdataclass）にそろえること、
オプションの既定値、結果の型を確かめる。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _assertions import assert_option_case_insensitive
from _helpers import CASE_SPELLINGS
from econometricsmodels import (
    FEIV,
    CraggDonaldResult,
    EffectiveFResult,
    FEIVOptions,
    FEIVResults,
    FEResults,
    KleibergenPaapResult,
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


def _fit(data: pl.DataFrame, x_endog: list[str], **options) -> FEIVResults:
    return FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=x_endog,
        instruments=["z1", "z2", "z3"],
        effects=["entity"],
        options=FEIVOptions(**options),
    ).fit()


# ── 成功パス・結果型 ─────────────────────────────────────────────────────


def test_fit_succeeds_and_returns_fe_iv_results(data):
    result = _fit(data, ["a", "b"])
    assert isinstance(result, FEIVResults)
    assert result.param_names == ["x1", "a", "b"]
    assert result.n_obs == N_ENTITIES * N_PERIODS
    assert result.n_groups == {"entity": N_ENTITIES}
    assert result.dep_var_name == "y"


def test_default_options_use_cluster_on_the_first_effect(data):
    result = FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=["a"],
        instruments=["z1", "z2"],
        effects=["entity"],
    ).fit()
    assert result.cov_type == "cluster"
    assert result.stat_dist == "t"
    assert result.stat_df == N_ENTITIES - 1
    assert result.wald_dist == "f"
    assert result.wald_df_denom == N_ENTITIES - 1


# ── API構造 ──────────────────────────────────────────────────────────────


def test_params_table_structure(data):
    result = _fit(data, ["a"])
    rows = result.params_table()
    assert [row["param_name"] for row in rows] == result.param_names
    assert set(rows[0]) == {
        "param_name",
        "param",
        "std_error",
        "test_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }


def test_params_std_errors_test_stats_p_values_share_keys(data):
    result = _fit(data, ["a", "b"])
    keys = result.param_names
    for table in (
        result.params,
        result.std_errors,
        result.test_stats,
        result.p_values,
        result.conf_int,
        result.cov_params,
    ):
        assert list(table) == keys


@pytest.mark.parametrize("x_endog", [["a", "b"], ["b", "a"]])
def test_per_endogenous_results_follow_the_order_of_x_endog(data, x_endog):
    """`weak_instrument_f_statistics`と`first_stage()`は`x_endog`の順に並ぶ
    （`_lib`の辞書は順序を持たない）。"""
    result = _fit(data, x_endog)
    assert list(result.weak_instrument_f_statistics) == x_endog
    assert list(result.first_stage()) == x_endog


def test_weak_instrument_f_statistics_match_the_engine(data):
    result = _fit(data, ["a", "b"])
    assert result.weak_instrument_f_statistics == (
        result._raw.weak_instrument_f_statistics
    )
    assert result.weak_instrument_f_df_num == 3
    assert result.weak_instrument_f_df_denom is not None


def test_first_stage_returns_fe_results(data):
    result = _fit(data, ["a", "b"])
    stages = result.first_stage()
    for name, stage in stages.items():
        assert isinstance(stage, FEResults)
        assert stage.dep_var_name == name
        assert stage.param_names == ["x1", "z1", "z2", "z3"]
        assert stage.n_groups == {"entity": N_ENTITIES}
        assert stage.cov_type == result.cov_type
        # 第一段階では全体のF検定を計算しない。
        assert stage.f_statistic is None
        assert list(stage.fixed_effects()) == ["entity"]


def test_overid_and_wu_hausman_are_exposed(data):
    result = _fit(data, ["a"])
    assert result.overid_df == 2
    assert result.overid_statistic is not None
    assert result.overid_p_value is not None
    assert result.wu_hausman_df_num == 1
    assert result.wu_hausman_statistic is not None


def test_just_identified_model_has_no_overid_test(data):
    result = FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=["a"],
        instruments=["z1"],
        effects=["entity"],
    ).fit()
    assert result.overid_statistic is None
    assert result.overid_p_value is None
    assert result.overid_df is None


def test_x_exog_may_be_empty(data):
    result = FEIV(
        data,
        y="y",
        x_exog=[],
        x_endog=["a"],
        instruments=["z1", "z2"],
        effects=["entity"],
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


# ── 固定効果 ─────────────────────────────────────────────────────────────


def test_fixed_effects_are_keyed_by_the_effects_columns(data):
    one_way = _fit(data, ["a"]).fixed_effects()
    assert list(one_way) == ["entity"]
    assert len(one_way["entity"]) == N_ENTITIES
    two_way = FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=["a"],
        instruments=["z1", "z2"],
        effects=["entity", "time"],
    ).fit()
    assert list(two_way.fixed_effects()) == ["entity", "time"]
    assert two_way.n_groups == {"entity": N_ENTITIES, "time": N_PERIODS}


# ── 診断 ─────────────────────────────────────────────────────────────────


def test_cragg_donald_wraps_the_engine_output(data):
    result = _fit(data, ["a", "b"])
    raw = result._raw.cragg_donald()
    wrapped = result.cragg_donald()
    assert isinstance(wrapped, CraggDonaldResult)
    assert wrapped.statistic == raw.statistic
    assert wrapped.n_endog == 2
    assert wrapped.n_excluded == 3
    assert wrapped.tsls_size_critical_values == (
        None
        if raw.tsls_size is None
        else {level: value for level, value in raw.tsls_size}
    )


def test_cragg_donald_equals_the_partial_f_for_one_endogenous_variable(data):
    result = _fit(data, ["a"])
    assert result.cragg_donald().statistic == pytest.approx(
        result.weak_instrument_f_statistics["a"], rel=1e-8
    )


def test_kleibergen_paap_wraps_the_engine_output(data):
    result = _fit(data, ["a", "b"])
    wrapped = result.kleibergen_paap()
    raw = result._raw.kleibergen_paap()
    assert isinstance(wrapped, KleibergenPaapResult)
    assert wrapped.rk_lm_statistic == raw.rk_lm_statistic
    assert wrapped.rk_wald_f == raw.rk_wald_f
    assert wrapped.n_endog == 2


def test_kleibergen_paap_follows_cov_type(data):
    classical = _fit(data, ["a"], cov_type="classical")
    assert classical.kleibergen_paap().rk_wald_f == pytest.approx(
        classical.cragg_donald().statistic, rel=1e-8
    )


def test_effective_f_wraps_the_engine_output_and_takes_alpha_by_keyword(data):
    result = _fit(data, ["a"])
    wrapped = result.effective_f(alpha=0.1)
    assert isinstance(wrapped, EffectiveFResult)
    assert wrapped.alpha == 0.1
    assert wrapped.statistic == result._raw.effective_f(0.1).statistic
    with pytest.raises(TypeError, match="positional"):
        result.effective_f(0.1)  # type: ignore[misc]


def test_effective_f_needs_one_endogenous_variable(data):
    with pytest.raises(ValueError, match="endogenous"):
        _fit(data, ["a", "b"]).effective_f()


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
        options=FEIVOptions(cov_type="hc0"),
    )
    with pytest.raises(ValueError):
        estimator.fit()


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize(
    "cov_type", ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
)
def test_cov_type_is_case_insensitive(data, cov_type, spelling):
    """全`cov_type`で、`res.cov_type`が正規形（小文字）で返り、どの大文字小文字の綴りでも正規形と
    同一の推定結果になる。`hc0`はFEIVでは受け付けない（`test_fe_iv_validation.py`）。`dk`は
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
