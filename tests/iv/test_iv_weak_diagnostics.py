"""IVの弱操作変数診断（`cragg_donald()`・`kleibergen_paap()`・`effective_f()`）。

いずれも`fit()`では計算しない事後診断のメソッドで、結果が保持する元のデータから
`fit()`と同じ手順で入力を作り直して計算する。参照は次のとおり。

- Cragg-Donald: linearmodelsの`first_stage.cragg_donald_f`（等分散の統計量）。内生変数が1個の
  ときは部分F統計量（`weak_instrument_f_statistics`）にも一致する。
- Kleibergen-Paap: linearmodelsの`first_stage.kleibergen_paap`（rk LM）と`kleibergen_paap_f`
  （rk Wald F）。**linearmodelsの`robust`のFは本実装の`hc1`、`robust`のLMは`hc0`に対応する**
  （Fは自由度の補正を定義に含める）。
- 有効F: `benchmark/iv/references/mop_oracle.py`（論文の式だけから書いた独立実装）。
"""

from __future__ import annotations

import copy
import json
import pickle
from functools import partial

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _constants import DATA_DIR
from _errors import raises_error
from _helpers import with_cluster_groups
from _tolerances import TOLERANCES
from econometricsmodels import (
    IV,
    CraggDonaldResult,
    EffectiveFResult,
    IVOptions,
    KleibergenPaapResult,
)
from linearmodels.iv import IV2SLS

from benchmark.iv.references import mop_oracle as mop

_REF = TOLERANCES["iv_weak_diagnostics"]
_close = partial(assert_close, rtol=_REF["rtol"], atol=0.0)

_X_EXOG = ["x1"]
_X_ENDOG = ["endog1"]
_INSTRUMENTS = ["z1", "z2"]
_CLUSTER = "cluster_group"
_N_CLUSTERS = 10


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return with_cluster_groups(
        pl.read_csv(DATA_DIR / "iv_baseline.csv"), _N_CLUSTERS, _CLUSTER
    )


@pytest.fixture(scope="module")
def multi_endog_data() -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")


def _fit(
    df: pl.DataFrame,
    *,
    cov_type: str = "classical",
    x_endog: list[str] = _X_ENDOG,
    instruments: list[str] = _INSTRUMENTS,
    estimator: str = "2sls",
    **extra,
):
    if cov_type == "cluster":
        extra["cluster"] = _CLUSTER
    return IV(
        df,
        y="y",
        x_exog=_X_EXOG,
        x_endog=x_endog,
        instruments=instruments,
        options=IVOptions(cov_type=cov_type, estimator=estimator, **extra),
    ).fit()


def _linearmodels(
    df: pl.DataFrame,
    cov_type: str,
    x_endog: list[str] = _X_ENDOG,
    instruments: list[str] = _INSTRUMENTS,
):
    pdf = df.to_pandas()
    pdf["const"] = 1.0
    kwargs = {}
    if cov_type == "robust":
        kwargs = {"cov_type": "robust"}
    elif cov_type == "clustered":
        kwargs = {"cov_type": "clustered", "clusters": pdf[_CLUSTER]}
    else:
        kwargs = {"cov_type": "unadjusted"}
    model = IV2SLS(
        pdf["y"],
        pdf[["const", *_X_EXOG]],
        pdf[x_endog],
        pdf[instruments],
    )
    return model.fit(**kwargs).first_stage


# ── Cragg-Donald ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_cragg_donald_matches_linearmodels(data, estimator) -> None:
    result = _fit(data, estimator=estimator).cragg_donald()
    expected = _linearmodels(data, "unadjusted").cragg_donald_f
    assert isinstance(result, CraggDonaldResult)
    _close(result.statistic, float(expected), "cragg_donald statistic")
    assert (result.n_endog, result.n_excluded) == (1, 2)


def test_cragg_donald_equals_the_partial_f_with_one_endogenous_variable(
    data,
) -> None:
    res = _fit(data)
    _close(
        res.cragg_donald().statistic,
        res.weak_instrument_f_statistics["endog1"],
        "partial F",
    )


def test_cragg_donald_with_two_endogenous_variables(multi_endog_data) -> None:
    kwargs = {
        "x_endog": ["endog1", "endog2"],
        "instruments": ["z1", "z2", "z3"],
    }
    result = _fit(multi_endog_data, **kwargs).cragg_donald()
    expected = _linearmodels(multi_endog_data, "unadjusted", **kwargs)
    _close(result.statistic, float(expected.cragg_donald_f), "joint F")
    assert (result.n_endog, result.n_excluded) == (2, 3)


def test_cragg_donald_does_not_depend_on_cov_type(data) -> None:
    classical = _fit(data).cragg_donald()
    for cov_type in ("hc1", "cluster"):
        other = _fit(data, cov_type=cov_type).cragg_donald()
        assert other == classical


def test_cragg_donald_critical_values_are_keyed_by_level(
    multi_endog_data,
) -> None:
    # One endogenous variable and three instruments: both tables apply.
    result = _fit(
        multi_endog_data, instruments=["z1", "z2", "z3"]
    ).cragg_donald()
    assert result.tsls_bias_critical_values is not None
    assert list(result.tsls_bias_critical_values) == [0.05, 0.10, 0.20, 0.30]
    assert result.tsls_size_critical_values is not None
    assert list(result.tsls_size_critical_values) == [0.10, 0.15, 0.20, 0.25]
    assert result.to_dict()["statistic"] == result.statistic


def test_cragg_donald_matches_the_published_stock_yogo_values(data) -> None:
    # Stock and Yogo (2005), Table 5.2: one endogenous variable, two
    # instruments, 2SLS size distortion of 10%: 19.93. The relative bias
    # table starts at three instruments.
    result = _fit(data).cragg_donald()
    assert result.tsls_size_critical_values is not None
    assert result.tsls_size_critical_values[0.10] == pytest.approx(
        19.93, abs=0.005
    )
    assert result.tsls_bias_critical_values is None


def test_cragg_donald_has_no_critical_values_outside_the_tables(
    multi_endog_data,
) -> None:
    # Two endogenous variables and two instruments: the bias table needs at
    # least `n_endog + 2` excluded instruments.
    result = _fit(
        multi_endog_data,
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2"],
    ).cragg_donald()
    assert result.tsls_bias_critical_values is None
    assert result.tsls_size_critical_values is not None


# ── Kleibergen-Paap ──────────────────────────────────────────────────────


def test_kleibergen_paap_classical_matches_linearmodels(data) -> None:
    result = _fit(data).kleibergen_paap()
    expected = _linearmodels(data, "unadjusted")
    assert isinstance(result, KleibergenPaapResult)
    _close(
        result.rk_lm_statistic,
        float(expected.kleibergen_paap.stat),
        "rk LM",
    )
    _close(
        result.rk_lm_p_value, float(expected.kleibergen_paap.pval), "p-value"
    )
    _close(result.rk_wald_f, float(expected.kleibergen_paap_f), "rk Wald F")
    assert result.rk_lm_df == 2
    # With a classical covariance the rk Wald F is the Cragg-Donald F.
    _close(
        result.rk_wald_f,
        float(expected.cragg_donald_f),
        "rk Wald F = Cragg-Donald F",
    )


def test_kleibergen_paap_robust_matches_linearmodels(data) -> None:
    # linearmodels' robust LM is HC0 and its robust F is HC1.
    expected = _linearmodels(data, "robust")
    lm = _fit(data, cov_type="hc0").kleibergen_paap()
    f = _fit(data, cov_type="hc1").kleibergen_paap()
    _close(lm.rk_lm_statistic, float(expected.kleibergen_paap.stat), "LM")
    _close(f.rk_wald_f, float(expected.kleibergen_paap_f), "F")


def test_kleibergen_paap_cluster_matches_linearmodels(data) -> None:
    result = _fit(data, cov_type="cluster").kleibergen_paap()
    expected = _linearmodels(data, "clustered")
    _close(result.rk_lm_statistic, float(expected.kleibergen_paap.stat), "LM")
    _close(result.rk_wald_f, float(expected.kleibergen_paap_f), "F")


def test_kleibergen_paap_uses_the_cov_type_of_the_fit(data) -> None:
    classical = _fit(data).kleibergen_paap()
    robust = _fit(data, cov_type="hc1").kleibergen_paap()
    assert robust.rk_wald_f != pytest.approx(classical.rk_wald_f, rel=1e-3)


def test_kleibergen_paap_with_two_endogenous_variables(
    multi_endog_data,
) -> None:
    kwargs = {
        "x_endog": ["endog1", "endog2"],
        "instruments": ["z1", "z2", "z3"],
    }
    result = _fit(multi_endog_data, **kwargs).kleibergen_paap()
    expected = _linearmodels(multi_endog_data, "unadjusted", **kwargs)
    _close(result.rk_lm_statistic, float(expected.kleibergen_paap.stat), "LM")
    assert result.rk_lm_df == 2


def test_kleibergen_paap_one_statistic_can_be_undefined(data) -> None:
    # Two clusters: the LM statistic is defined but the Wald F needs one more.
    df = with_cluster_groups(data, 4, _CLUSTER)
    res = IV(
        df,
        y="y",
        x_exog=[],
        x_endog=_X_ENDOG,
        instruments=_INSTRUMENTS,
        options=IVOptions(cov_type="cluster", cluster=_CLUSTER),
    ).fit()
    result = res.kleibergen_paap()
    assert result.rk_lm_statistic is not None
    assert (result.rk_lm_df is None) == (result.rk_lm_p_value is None)


# ── 有効F ────────────────────────────────────────────────────────────────

_MOP_COV = {
    "classical": "classical",
    "hc0": "hc0",
    "hc1": "hc1",
    "hc2": "hc2",
    "hc3": "hc3",
    "cluster": "cluster",
}


def _oracle(df: pl.DataFrame, cov: str, alpha: float = 0.05):
    x = np.column_stack([np.ones(df.height), df["x1"].to_numpy()])
    rf = mop.reduced_form(
        df["y"].to_numpy(),
        df["endog1"].to_numpy(),
        x,
        df.select(_INSTRUMENTS).to_numpy(),
        _MOP_COV[cov],
        groups=df[_CLUSTER].to_numpy() if cov == "cluster" else None,
    )
    return mop.mop_effective_f(rf, alpha=alpha)


@pytest.mark.parametrize("cov", list(_MOP_COV))
def test_effective_f_matches_the_oracle(data, cov) -> None:
    result = _fit(data, cov_type=cov).effective_f()
    expected = _oracle(data, cov)
    assert isinstance(result, EffectiveFResult)
    _close(result.statistic, expected.f_eff, "effective F")
    assert result.n_excluded == 2
    assert result.alpha == 0.05
    for tau, simple in expected.simple.items():
        _close(
            result.simplified_critical_values[float(tau)],
            simple.value,
            f"simplified critical value tau={float(tau)}",
        )
    assert result.critical_values is not None
    assert result.bias_bound is not None
    assert_close(
        result.bias_bound,
        expected.b_tsls,
        rtol=_REF["rtol_b_tsls"],
        atol=0.0,
        label="B_TSLS",
    )
    for tau, general in expected.generalized.items():
        assert_close(
            result.critical_values[float(tau)],
            general.value,
            rtol=_REF["rtol_b_tsls"],
            atol=0.0,
            label=f"generalized critical value tau={float(tau)}",
        )


def test_effective_f_alpha_changes_the_critical_values(data) -> None:
    res = _fit(data, cov_type="hc1")
    strict = res.effective_f(alpha=0.01)
    loose = res.effective_f(alpha=0.10)
    assert strict.alpha == 0.01
    expected = _oracle(data, "hc1", alpha=0.01)
    for tau, simple in expected.simple.items():
        _close(
            strict.simplified_critical_values[float(tau)],
            simple.value,
            f"alpha=0.01 tau={float(tau)}",
        )
    assert (
        strict.simplified_critical_values[0.10]
        > loose.simplified_critical_values[0.10]
    )


def test_effective_f_equals_cragg_donald_with_a_classical_covariance(
    data,
) -> None:
    res = _fit(data)
    _close(
        res.effective_f().statistic,
        res.cragg_donald().statistic,
        "effective F = Cragg-Donald F",
    )


# ── 共通 ─────────────────────────────────────────────────────────────────


def test_diagnostics_are_available_for_gmm(data) -> None:
    res = _fit(data, estimator="gmm", cov_type="hc1")
    _close(
        res.kleibergen_paap().rk_wald_f,
        _fit(data, cov_type="hc1").kleibergen_paap().rk_wald_f,
        "gmm vs 2sls",
    )


def test_diagnostics_survive_pickling(data) -> None:
    res = _fit(data, cov_type="cluster")
    restored = pickle.loads(pickle.dumps(res))
    assert restored.cragg_donald() == res.cragg_donald()
    assert restored.kleibergen_paap() == res.kleibergen_paap()
    assert restored.effective_f() == res.effective_f()


def test_diagnostics_do_not_modify_the_result(data) -> None:
    res = _fit(data, cov_type="hc1")
    before = res.params
    res.cragg_donald()
    res.kleibergen_paap()
    res.effective_f()
    assert res.params == before


def test_diagnostics_to_dict_is_json_ready(data) -> None:
    res = _fit(data, cov_type="hc1")
    for result in (
        res.cragg_donald(),
        res.kleibergen_paap(),
        res.effective_f(),
    ):
        assert set(result.to_dict()) == set(result.__dataclass_fields__)
        json.dumps(result.to_dict())


_SETTINGS = {
    "hc2": {"cov_type": "hc2"},
    "hc3": {"cov_type": "hc3"},
    "hac": {"cov_type": "hac", "hac_time": "t", "hac_lags": 3},
    "no_intercept": {"cov_type": "hc1", "include_intercept": False},
    "gmm_cluster_weight": {
        "estimator": "gmm",
        "cov_type": "cluster",
        "gmm_weight_type": "cluster",
    },
    "gmm_hac_weight": {
        "estimator": "gmm",
        "cov_type": "hac",
        "hac_time": "t",
        "gmm_weight_type": "hac",
    },
}


@pytest.mark.parametrize("setting", list(_SETTINGS))
def test_diagnostics_run_under_every_covariance_setting(data, setting) -> None:
    df = data.with_columns(pl.int_range(pl.len()).alias("t"))
    res = _fit(df, **_SETTINGS[setting])
    cd, kp, eff = res.cragg_donald(), res.kleibergen_paap(), res.effective_f()
    assert np.isfinite(cd.statistic)
    assert kp.rk_lm_statistic is not None and np.isfinite(kp.rk_lm_statistic)
    assert np.isfinite(eff.statistic)
    # `copy` keeps the retained data and options.
    assert copy.deepcopy(res).kleibergen_paap() == kp
    assert copy.copy(res).effective_f() == eff


# ── ValidationError ──────────────────────────────────────────────────────


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_effective_f_rejects_alpha_outside_the_open_unit_interval(
    data, alpha
) -> None:
    res = _fit(data)
    with raises_error("ALPHA_OUT_OF_RANGE"):
        res.effective_f(alpha=alpha)


def test_effective_f_needs_one_endogenous_variable(multi_endog_data) -> None:
    res = _fit(
        multi_endog_data,
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2", "z3"],
    )
    with raises_error("EFFECTIVE_F_ENDOGENOUS_COUNT", n_endog=2):
        res.effective_f()


@pytest.mark.parametrize("alpha", ["0.05", None, True])
def test_effective_f_alpha_must_be_a_real_number(data, alpha) -> None:
    with pytest.raises(TypeError, match="alpha must be a real number"):
        _fit(data).effective_f(alpha=alpha)


def test_effective_f_alpha_is_keyword_only(data) -> None:
    with pytest.raises(TypeError):
        _fit(data).effective_f(0.05)  # type: ignore[misc]


# ── ComputationError ─────────────────────────────────────────────────────


def test_a_singular_score_covariance_is_undefined() -> None:
    # Five instruments and three clusters: the covariance of the scores has
    # rank at most two, so the effective F statistic is not defined.
    rng = np.random.default_rng(3)
    n = 60
    z = rng.normal(size=(n, 5))
    endog = z @ np.array([0.5, 0.4, 0.3, 0.2, 0.1]) + rng.normal(size=n)
    df = pl.DataFrame(
        {
            "y": endog + rng.normal(size=n),
            "endog1": endog,
            **{f"z{i + 1}": z[:, i] for i in range(5)},
            _CLUSTER: np.arange(n) % 3,
        }
    )
    res = IV(
        df,
        y="y",
        x_exog=[],
        x_endog=_X_ENDOG,
        instruments=[f"z{i + 1}" for i in range(5)],
        options=IVOptions(cov_type="cluster", cluster=_CLUSTER),
    ).fit()
    with raises_error(
        "WEAK_INSTRUMENT_STATISTIC_UNDEFINED", statistic="effective F"
    ):
        res.effective_f()
    with raises_error(
        "WEAK_INSTRUMENT_STATISTIC_UNDEFINED", statistic="Kleibergen-Paap"
    ):
        res.kleibergen_paap()
    # The homoskedastic Cragg-Donald statistic does not use the clusters.
    assert res.cragg_donald().statistic > 0
