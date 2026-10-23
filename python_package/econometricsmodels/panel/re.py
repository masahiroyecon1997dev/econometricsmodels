"""Python wrapper for RE (random effects panel regression).

A thin wrapper around `econometricsmodels._lib.fit_re` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — `entity` as a bare
column-name argument (`time` goes in `REOptions`), `x` as a list, an
options object for estimation settings (CLAUDE.md section 2,
`.claude/rules/python-style.md` "設計方針との整合性",
`docs/spec/panel-common.md` section 1).

`REOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; same policy as `OLSOptions`/`IVOptions`/`FEOptions`).

Unlike FE, RE has no separate "additional result" method: the Hausman
test comparing RE against the equivalent FE specification is computed
automatically inside `fit()` and exposed directly as properties on
`REResults` (`docs/spec/panel-common.md` section 2.4).

`summary()` is not implemented (structured-data-only output policy; see
the `OLSResults`/`FEResults` precedent).
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import REOptions
from .._results_utils import (
    by_param,
    conf_int_by_param,
    params_table_rows,
)

__all__ = ["RE", "REOptions", "REResults"]


class RE:
    """Random effects (GLS) panel regression estimator.

    The variance components are estimated by Swamy-Arora by default;
    `REOptions.variance_method` selects Wallace-Hussain or Amemiya
    instead.

    Unlike FE, RE has an intercept. By default it estimates one-way
    (entity) random effects. Setting `REOptions.time` requests two-way
    (entity + time) random effects, which requires a balanced panel
    (`ValidationError` otherwise). The Hausman test and the panel R²
    (within / between / overall) are available for both one-way and
    two-way effects; see `REResults`.

    Args:
        data: A polars DataFrame containing the dependent variable,
            independent variables, entity identifier, and (if
            specified) time/cluster columns.
        y: Column name of the dependent variable.
        x: List of column names of the independent variables. Must
            contain at least one column name. A time-invariant
            regressor (constant over time within every entity) is
            allowed, unlike in FE.
        entity: Column name of the entity (individual/panel unit)
            identifier.
        options: Estimation options. Defaults to `REOptions()`
            (`cov_type="cluster"` on `entity`, confidence_level=0.95)
            when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import RE
        >>> df = pl.DataFrame(
        ...     {
        ...         "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ...         "x1": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
        ...         "id": ["a", "a", "b", "b", "c", "c"],
        ...     }
        ... )
        >>> result = RE(df, y="y", x=["x1"], entity="id").fit()
        >>> result.params["x1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        entity: str,
        options: REOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._entity = entity
        self._options = options if options is not None else REOptions()

    def fit(self) -> REResults:
        """Estimate the RE model.

        Returns:
            The estimation results.

        Raises:
            TypeError: An argument has the wrong type (for example
                `x` is a string instead of a list of column names). A
                builtin exception, not a `ValidationError`.
            ValidationError: The input or options are invalid (`x` is
                empty, a column is missing, contains missing values or
                NaN/infinity, `y`/`x`/`entity`/`time` overlap, `dk_time`
                equal to `y` or `entity`, a two-way model (`time` set) on
                an unbalanced panel or with a singleton time period,
                insufficient observations, `confidence_level` out of
                range, an unknown `cov_type` (or `cov_type="hc0"`,
                unsupported for RE), `dk_time` specified with a
                `cov_type` other than `"dk"`, a singleton entity group
                (raised by RE's internal FE regression, which is built
                for every `variance_method`), a `cov_type="dk"` request
                with `dk_time` unset, `cov_type="dk"` with no more
                unique time periods than regressors (the
                Driscoll-Kraay covariance has rank at most T-1, so the
                Hausman test cannot be computed),
                `cov_type="cluster"` with no more clusters than the
                identified slopes of the Hausman auxiliary regression,
                an unknown `variance_method`, or
                `variance_method="amemiya"` with a
                regressor that has no within variation (time-invariant;
                for two-way effects, varying in at most one
                dimension)). A subclass of `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. a singular quasi-demeaned design
                matrix, in particular a time-invariant regressor that
                is constant across entities or collinear with another
                time-invariant regressor, or a regressor together with
                a multiple of itself or of its entity mean; the
                between and Hausman auxiliary regressions drop such
                aliased columns, the final regression does not, so
                the failure comes from the final regression; a
                singular pooled first-stage
                regression with `variance_method="wallace_hussain"`, or
                a non-positive estimate of σ_ε² with `"wallace_hussain"`
                or `"amemiya"`). A subclass of `RuntimeError`.
        """
        raw = _lib.fit_re(
            self._data, self._y, self._x, self._entity, self._options
        )
        return REResults(raw)


class REResults:
    """RE estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `params_table()` for a row-oriented listing.
    `param_names[0]` is always `"const"` (unlike FE, RE has an
    intercept; see `engine::panel::re`'s module docstring on the Rust
    side for why the quasi-demeaned constant column is named this
    way).

    The Hausman test (`hausman_statistic`/`hausman_p_value`/
    `hausman_df`) is computed automatically inside `fit()` and exposed
    directly as properties here, rather than as a separate method like
    FE's `fixed_effects()` (`panel-common.md` section 2.4). It is the
    regression-based (auxiliary regression) version (Wooldridge 2010,
    section 10.7.3): the quasi-demeaned `y` is regressed on a constant,
    the quasi-demeaned regressors and the within-transformed
    regressors, and the within-transformed coefficients are tested
    jointly. The comparison is against fixed effects with the same
    structure as RE itself (entity for one-way, entity and time for
    two-way), regardless of `REOptions.dk_time`. It matches
    `plm::phtest(method = "aux")` (`effect = "individual"` or
    `"twoways"`) when every regressor varies in both dimensions;
    otherwise `plm` tests the wrong coefficients. The Wald test's
    covariance follows `cov_type` (the default `"cluster"` gives the
    cluster-robust Hausman test; `"classical"` gives the classical
    version). The auxiliary regression uses OLS-style small-sample
    corrections, which differ from the RE standard errors'.

    `hausman_df` is the number of tested coefficients: the regressors
    that still vary after the within transformation (entity means
    removed for one-way, entity and time means removed for two-way, so
    for two-way a regressor that varies in only one dimension is not
    tested) and whose coefficient is identified. In a one-way model on a
    balanced panel, a regressor that varies only over time (a year
    trend) stays in the model but its within-transformed copy is
    collinear with the constant, so it is left out of the tested set.
    So is the within-transformed copy of a regressor whose entity mean
    (or, for two-way effects, time mean) is also a regressor, as in a
    correlated-random-effects (Mundlak) specification: it is a linear
    combination of the constant, the quasi-demeaned regressors and the
    earlier within-transformed copies, and, as in R's `lm`, the later
    aliased column is dropped (relative tolerance `1e-7`). Only
    within-transformed copies are dropped; a perfectly collinear
    design still raises `ComputationError`. A time-invariant regressor
    (constant over time within every entity) is allowed in RE and is
    never tested. All three values are `None` only when no coefficient
    can be tested. If the auxiliary regression
    cannot be computed (rank-deficient design, or a structurally
    singular robust covariance such as too few clusters or periods for
    the auxiliary slopes), `fit()` raises
    `ValidationError`/`ComputationError`. A failure in RE's own internal
    FE call (used for the Hausman test, the within R² and the
    Swamy-Arora and Amemiya variance components) makes `fit()` itself
    raise instead (e.g. a singleton entity); see `RE.fit()`'s
    docstring.

    The estimated variance components (`sigma2_eps`, `sigma2_u`, and
    `sigma2_lambda` for two-way effects) and the weights of the
    quasi-demeaning that follow from them (`theta` for one-way,
    `theta_entity`/`theta_time`/`theta_total` for two-way) are exposed as
    properties; the ones that do not apply to the model are `None`.
    `variance_method` tells which estimator produced them.

    Args:
        raw: The estimation result object returned by `_lib.fit_re`
            (`_lib.REResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `RE.fit()`.
    """

    def __init__(self, raw: _lib.REResult) -> None:
        self._raw = raw
        self._theta: dict[str, float] | None = None

    @property
    def param_names(self) -> list[str]:
        """List of coefficient names (`param_names[0]` is always
        `"const"`)."""
        return self._raw.param_names

    @property
    def params(self) -> dict[str, float]:
        """Coefficient name to coefficient value."""
        return by_param(self._raw.param_names, self._raw.params)

    @property
    def std_errors(self) -> dict[str, float]:
        """Coefficient name to standard error."""
        return by_param(self._raw.param_names, self._raw.std_errors)

    @property
    def test_stats(self) -> dict[str, float]:
        """Coefficient name to test statistic (t-statistic; see
        `stat_dist`)."""
        return by_param(self._raw.param_names, self._raw.test_stats)

    @property
    def stat_dist(self) -> Literal["t", "normal"]:
        """Distribution of `test_stats`: `"t"` (t-statistics) or
        `"normal"` (z-statistics)."""
        return self._raw.stat_dist

    @property
    def stat_df(self) -> int | None:
        """Degrees of freedom of the t distribution behind `test_stats`,
        or `None` when `stat_dist` is `"normal"`. May differ from
        `df_resid` (cluster-robust inference uses `G - 1`, Driscoll-Kraay
        `T - 1`), so use this to recompute p-values from `test_stats`."""
        return self._raw.stat_df

    @property
    def p_values(self) -> dict[str, float]:
        """Coefficient name to two-sided p-value."""
        return by_param(self._raw.param_names, self._raw.p_values)

    @property
    def conf_int(self) -> dict[str, tuple[float, float]]:
        """Coefficient name to confidence interval `(lower, upper)`."""
        return conf_int_by_param(self._raw)

    @property
    def residuals(self) -> list[float]:
        """Quasi-demeaned residuals (in observation order)."""
        return self._raw.residuals

    @property
    def dep_var_name(self) -> str:
        """Column name of the dependent variable."""
        return self._raw.dep_var_name

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._raw.n_obs

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom (`n - k`, unlike FE's
        `n - n_entities - k`; RE is a GLS transform that does not
        consume entity-dummy degrees of freedom)."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (`k`, including the intercept)."""
        return self._raw.df_model

    @property
    def n_entities(self) -> int:
        """Number of panel entities."""
        return self._raw.n_entities

    @property
    def n_periods(self) -> int | None:
        """Number of unique time periods (two-way effects only; `None`
        for one-way, also when only `REOptions.dk_time` is given).
        Two-way requires a balanced panel, so this equals the number of
        observations per entity."""
        return self._raw.n_periods

    @property
    def variance_method(self) -> str:
        """Estimator of the variance components actually used (the
        normalized value of `REOptions.variance_method`):
        `"swamy_arora"`, `"wallace_hussain"` or `"amemiya"`. `sigma2_eps`,
        `sigma2_u`, `sigma2_lambda` and the weights `theta`,
        `theta_entity`, `theta_time` and `theta_total` are the values of
        this estimator."""
        return self._raw.variance_method

    @property
    def sigma2_eps(self) -> float:
        """Estimated variance of the idiosyncratic error, σ_ε².
        Swamy-Arora: the residual variance of the internal
        fixed-effects regression with the same effect structure as the
        model (entity for one-way, entity and time for two-way); its
        degrees of freedom exclude the regressors without within
        variation (time-invariant ones), as in `plm`, whereas
        `linearmodels` counts them and so differs. Wallace-Hussain and
        Amemiya: the within sum of squares of the first-stage residuals
        divided by `n - N` (`n - N - T + 1` for two-way effects; the
        number of regressors is not subtracted) on a balanced panel,
        and the trace-term form of `plm` on an unbalanced one-way
        panel. A one-way panel is balanced here when every entity has
        the same number of observations, and a two-way panel always is.
        The two forms give different values even on the same balanced
        panel, so the values are not continuous when a row is
        dropped."""
        return self._raw.sigma2_eps

    @property
    def sigma2_u(self) -> float:
        """Estimated variance of the entity effect, σ_u² (a negative
        estimate is raised to 0). Swamy-Arora on an unbalanced panel is
        the `linearmodels` estimator (harmonic mean of the entity
        sizes), which differs from `plm`'s by a few percent up to about
        10 %; on a balanced panel the two agree. Wallace-Hussain and
        Amemiya follow `plm` on balanced and unbalanced panels."""
        return self._raw.sigma2_u

    @property
    def sigma2_lambda(self) -> float | None:
        """Estimated variance of the time effect, σ_λ² (two-way effects
        only; a negative estimate is raised to 0). `None` for one-way."""
        return self._raw.sigma2_lambda

    @property
    def theta(self) -> dict[str, float] | None:
        """One-way quasi-demeaning weight of each entity,
        `θ_i = 1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`, where `T_i` is the
        number of observations of entity `i`: the data are transformed
        as `x - θ_i · mean_i(x)`. θ_i is 0 when σ_u² is 0 (pooled OLS)
        and it differs across entities only on an unbalanced panel.

        The keys are the entity ids as strings, in lexicographic order
        (so `"10"` precedes `"2"` for integer ids, and `res.theta[1]`
        raises `KeyError`; use `res.theta["1"]`). The same dictionary is
        returned on every access: do not modify it. `None` for two-way
        effects (see `theta_entity`, `theta_time` and `theta_total`)."""
        # 個体数に比例する辞書のため、最初の参照で一度だけ作る。
        if self._theta is None and self._raw.theta is not None:
            self._theta = self._raw.theta
        return self._theta

    @property
    def theta_entity(self) -> float | None:
        """Two-way weight θ₁ of the entity means in the GLS
        transformation `x - θ₁·x̄ᵢ - θ₂·x̄ₜ + θ₃·x̄`:
        `θ₁ = 1 - σ_ε/σ₁` with `σ₁² = σ_ε² + T·σ_u²` (`T` periods).
        It is 0 when σ_u² is 0. `None` for one-way."""
        return self._raw.theta_entity

    @property
    def theta_time(self) -> float | None:
        """Two-way weight θ₂ of the time means: `θ₂ = 1 - σ_ε/σ₂` with
        `σ₂² = σ_ε² + N·σ_λ²` (`N` entities). It is 0 when σ_λ² is 0.
        `None` for one-way."""
        return self._raw.theta_time

    @property
    def theta_total(self) -> float | None:
        """Two-way weight θ₃ of the overall mean:
        `θ₃ = θ₁ + θ₂ - 1 + σ_ε/σ₄` with
        `σ₄² = σ_ε² + T·σ_u² + N·σ_λ²`. It is not the product θ₁·θ₂, and
        it is exactly 0 when σ_u² or σ_λ² is 0. `None` for one-way."""
        return self._raw.theta_total

    @property
    def cov_type(self) -> str:
        """Standard error type actually used (normalized to lowercase)."""
        return self._raw.cov_type

    @property
    def dk_bandwidth_used(self) -> int | None:
        """Driscoll-Kraay bandwidth actually used: the explicit
        `dk_bandwidth` if given, otherwise the value chosen automatically,
        `floor(4 * (t / 100) ** (2 / 9))` where `t` is the number of
        unique time periods in `REOptions.dk_time`. `None` unless
        `cov_type="dk"`."""
        return self._raw.dk_bandwidth_used

    @property
    def f_statistic(self) -> float | None:
        """F-statistic for the joint significance of the slope
        coefficients, excluding the intercept (classical F-test when
        `cov_type="classical"`, a robust Wald test otherwise; the
        degrees of freedom follow `cov_type`, see `f_df_denom`). `None`
        when there are no slope coefficients (`df_model == 1`)."""
        return self._raw.f_statistic

    @property
    def f_p_value(self) -> float | None:
        """P-value of the F-statistic (`None` when `f_statistic` is
        `None`)."""
        return self._raw.f_p_value

    @property
    def f_df_num(self) -> int | None:
        """Numerator degrees of freedom of `f_statistic` (`None` when
        `f_statistic` is `None`)."""
        return self._raw.f_df_num

    @property
    def f_df_denom(self) -> int | None:
        """Denominator degrees of freedom of `f_statistic` (`None` when
        `f_statistic` is `None`). Follows `cov_type` like the t-tests (`G - 1` for
        `cov_type="cluster"`, `T - 1` for `"dk"`, `df_resid` otherwise),
        so it equals `stat_df` whenever `f_statistic` is not `None`. Use
        this to recompute the p-value from `f_statistic`."""
        return self._raw.f_df_denom

    @property
    def log_likelihood(self) -> float:
        """Log-likelihood."""
        return self._raw.log_likelihood

    @property
    def aic(self) -> float:
        """Akaike Information Criterion (AIC)."""
        return self._raw.aic

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion (BIC)."""
        return self._raw.bic

    @property
    def r_squared_within(self) -> float:
        """Within R² (based on the demeaned variables, θ=1; unrelated to
        RE's own quasi-demeaning). The variables are
        demeaned by entity, and for two-way effects by entity and time.
        Only the coefficients of regressors that still vary after
        demeaning enter the fitted values, so it is `0.0` when none
        does (for example every regressor is time-invariant)."""
        return self._raw.r_squared_within

    @property
    def r_squared_between(self) -> float:
        """Between R² (based on entity-mean variables, with the
        intercept; the same definition for one-way and two-way
        effects)."""
        return self._raw.r_squared_between

    @property
    def r_squared_overall(self) -> float:
        """Overall R² (based on the untransformed variables, with the
        intercept; the same definition for one-way and two-way
        effects)."""
        return self._raw.r_squared_overall

    @property
    def hausman_statistic(self) -> float | None:
        """Regression-based Hausman test statistic (chi-squared
        version, `k × F` of the Wald test that the `k` within-transformed
        slope regressors (`k` is `hausman_df`) are jointly
        zero in the auxiliary regression of the quasi-demeaned `y` on
        the quasi-demeaned regressors and the within-transformed
        regressors). Always non-negative. The comparison is against FE
        with the same structure as RE (entity, or entity and time for
        two-way effects). The Wald test's covariance follows
        `cov_type` (default `"cluster"` gives the cluster-robust
        Hausman test; `"classical"` gives the classical version). The
        auxiliary regression uses OLS-style small-sample corrections,
        unlike the RE standard errors. For unbalanced panels the
        auxiliary regression's constant is left untransformed, as in
        `plm::phtest`. `None` when no coefficient can be tested (no
        regressor varies after the within transformation, or the
        remaining ones are aliased, for example because the entity
        mean of every time-varying regressor is also a regressor); see
        the class docstring. If
        the auxiliary regression cannot be computed, `fit()` raises
        instead."""
        return self._raw.hausman_statistic

    @property
    def hausman_p_value(self) -> float | None:
        """P-value of `hausman_statistic` (upper-tail chi-squared
        probability). `None` under the same conditions as
        `hausman_statistic`."""
        return self._raw.hausman_p_value

    @property
    def hausman_df(self) -> int | None:
        """Degrees of freedom of the Hausman test (number of compared
        slope coefficients: the regressors that still vary after the
        within transformation and can be identified, which is
        `df_model - 1` when every regressor does). A
        within-transformed copy is not counted when it is aliased: its
        quasi-demeaned counterpart differs from it by a constant, or
        the regressor's entity (or, for two-way effects, time) mean is
        also a regressor (correlated random effects), so `hausman_df`
        is then smaller than the number of time-varying regressors.
        `None` under the same conditions as `hausman_statistic`."""
        return self._raw.hausman_df

    def params_table(self) -> list[dict[str, float | str]]:
        """Row-oriented summary table of the coefficients.

        Shaped to be usable almost as-is in a REST API response (same
        policy as `OLSResults.params_table()`). Returned as `list[dict]`
        rather than a polars DataFrame, per the project's policy of
        not using DataFrames for the parameter table itself.

        Returns:
            A list of dictionaries, one per coefficient. Keys are
            `param_name`, `param`, `std_error`, `test_stat`, `p_value`,
            `conf_lower`, `conf_upper`.
        """
        return params_table_rows(self._raw)
