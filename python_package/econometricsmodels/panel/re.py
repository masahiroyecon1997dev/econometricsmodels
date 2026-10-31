"""Python wrapper for RE (random effects panel regression).

A thin wrapper around `econometricsmodels._lib.fit_re` (the Rust
implementation). Validation and estimation happen on the Rust side; this
module only provides the Python-facing API shape for polars DataFrames:
`effects` and `x` as lists of column names, and an options object for
the estimation settings.

`REOptions` is re-exported as-is from `_lib`. Unlike FE, the Hausman
test against the equivalent FE specification is computed by `fit()` and
exposed as properties of `REResults`. There is no `summary()`: results
are returned as structured data.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import REOptions
from .._results_utils import (
    by_param,
    conf_int_by_param,
    cov_params_by_param,
    params_table_rows,
)

__all__ = ["RE", "REOptions", "REResults"]


class RE:
    """Random effects (GLS) panel regression estimator.

    The variance components are estimated by Swamy-Arora by default;
    `REOptions.variance_method` selects Wallace-Hussain or Amemiya
    instead.

    Unlike FE, RE has an intercept. `effects` with one column
    estimates one-way (entity) random effects; with two columns, the
    entity and then the time period, it estimates two-way random
    effects, which requires a balanced panel (`ValidationError`
    otherwise). More than two columns are not supported. The order of
    the two columns matters: the first is the entity dimension
    (`sigma2_u`, `theta_entity`) and the second the time dimension
    (`sigma2_lambda`, `theta_time`). The first column is also the
    default clustering column, so swapping the columns changes the
    default standard errors and the Hausman test. The Hausman test and
    the panel R²
    (within / between / overall) are available for both one-way and
    two-way effects; see `REResults`.

    Args:
        data: A polars DataFrame containing the dependent variable,
            independent variables, the `effects` columns, and (if
            specified) the `cluster` and `dk_time` columns of the
            options.
        y: Column name of the dependent variable.
        x: List of column names of the independent variables. Must
            contain at least one column name. A time-invariant
            regressor (constant over time within every entity) is
            allowed, unlike in FE.
        effects: List of one or two column names that define the random
            effects. The first is the entity (individual/panel unit)
            identifier; the optional second is the time period,
            which makes the model two-way (balanced panels only). The
            order matters: the first column gives `sigma2_u` and
            `theta_entity`, the second `sigma2_lambda` and `theta_time`
            (see above), and the first is also the default cluster
            key.
        options: Estimation options. Defaults to `REOptions()`
            (`cov_type="cluster"` on the first column of `effects`,
            confidence_level=0.95) when omitted.

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
        >>> result = RE(df, y="y", x=["x1"], effects=["id"]).fit()
        >>> round(result.params["x1"], 3)
        0.315

        Two-way effects take a second column, the time period:

        >>> panel = pl.DataFrame(
        ...     {
        ...         "y": [1.2, 2.1, 2.9, 2.0, 3.2, 3.9]
        ...         + [0.5, 1.7, 2.2, 3.1, 3.8, 5.2],
        ...         "x1": [1.0, 2.0, 3.0, 1.5, 2.5, 3.0]
        ...         + [0.5, 1.0, 2.0, 2.0, 3.0, 4.0],
        ...         "id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
        ...         "t": [1, 2, 3] * 4,
        ...     }
        ... )
        >>> two_way = RE(
        ...     panel, y="y", x=["x1"], effects=["id", "t"]
        ... ).fit()
        >>> two_way.n_groups
        {'id': 4, 't': 3}
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        effects: list[str],
        *,
        options: REOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._effects = effects
        self._options = options if options is not None else REOptions()

    def fit(self) -> REResults:
        """Estimate the RE model.

        Returns:
            The estimation results.

        Raises:
            TypeError: An argument has the wrong type (for example
                `x` or `effects` is a string instead of a list of
                column names). A builtin exception, not a
                `ValidationError`.
            ValidationError: The input or options are invalid. A
                subclass of `ValueError`. For example:

                - `x` is empty, a column is missing, or a column
                  contains missing values, NaN or infinity.
                - `effects` is empty or has more than two columns, or
                  `y`, `x` and `effects` overlap, or `dk_time` equals
                  `y` or the first column of `effects`.
                - There are insufficient observations, or
                  `confidence_level` is out of range.
                - `cov_type` is unknown or `"hc0"` (not supported for
                  RE), or `variance_method` is unknown.
                - `cluster`, `dk_time` or `dk_bandwidth` is given with
                  a `cov_type` that does not use it, or
                  `cov_type="dk"` is requested without `dk_time`.
                - A two-way model (two columns in `effects`) is fitted
                  on an unbalanced panel.
                - A group of a column in `effects` has a single
                  observation (an entity or, in a two-way model, a
                  time period).
                - `cov_type="dk"` with no more unique `dk_time` periods
                  than regressors: the Driscoll-Kraay covariance has
                  rank at most `T - 1` (`T` periods), so the Hausman
                  test cannot be computed.
                - `cov_type="cluster"` with no more clusters than the
                  slope coefficients compared in the Hausman test.
                - The outcome is constant within every entity.
                - `variance_method="amemiya"` with a regressor that has
                  no within variation (time-invariant; for two-way
                  effects, varying in at most one dimension).
            ComputationError: A problem was detected during
                computation. A subclass of `RuntimeError`. For example:

                - The transformed design matrix is singular: a
                  time-invariant regressor that is constant across
                  entities or collinear with another time-invariant
                  regressor, or a regressor together with a multiple of
                  itself or of its entity mean.
                - `variance_method="wallace_hussain"` with a singular
                  pooled regression.
                - `variance_method="wallace_hussain"` or `"amemiya"`
                  gives a non-positive estimate of σ_ε².
                - The within regression that estimates σ_ε² fits
                  perfectly (residuals zero up to rounding).
        """
        raw = _lib.fit_re(
            self._data, self._y, self._x, self._effects, self._options
        )
        return REResults(raw)


class REResults:
    """RE estimation results.

    Coefficient-level properties (`params`, `std_errors`, etc.) are
    dictionaries keyed by coefficient name, so one coefficient is
    `result.params["x1"]`. Use `params_table()` for a row-oriented
    listing (a list of dicts), for example for a JSON response.
    `param_names[0]` is always `"const"` (unlike FE, RE has an
    intercept).

    Fields by purpose:

    - Coefficients: `params`, `std_errors`, `test_stats`, `p_values`,
      `conf_int`; the names in order: `param_names`.
    - Covariance of the coefficients: `cov_params`.
    - Which distribution the tests use: `stat_dist`, `stat_df`.
    - Overall F test: `f_statistic`, `f_p_value`, `f_df_num`,
      `f_df_denom`.
    - Hausman test against FE (computed by `fit()`):
      `hausman_statistic`, `hausman_p_value`, `hausman_df`.
    - Variance components: `sigma2_eps`, `sigma2_u`, `sigma2_lambda`
      (two-way only), and `variance_method`.
    - Quasi-demeaning weights: `theta` (one-way), `theta_entity`,
      `theta_time`, `theta_total` (two-way). The ones that do not
      apply are `None`.
    - Fit: `r_squared_within`, `r_squared_between`,
      `r_squared_overall`, `log_likelihood`, `aic`, `bic`.
    - Panel size: `n_obs`, `n_groups`.
    - Model size: `df_model`, `df_resid`, `dep_var_name`.
    - Settings echoed back: `cov_type`, `dk_bandwidth_used`.
    - Residuals: `residuals` (quasi-demeaned).
    - Methods: `params_table()`.

    The Hausman test is the regression-based (auxiliary regression)
    version; see the docstring of `hausman_statistic` and the
    definitions in the documentation.

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
    def cov_params(self) -> dict[str, dict[str, float]]:
        """Covariance matrix of the coefficients, as a dict of dicts by name.

        Computed with the `cov_type` of the options, so `std_errors[a]` is
        the square root of `cov_params[a][a]`. The covariance of `a` and `b`
        is `cov_params["a"]["b"]`, and the keys are in `param_names` order.

        The matrix is symmetric up to rounding error (a relative difference
        of 1e-16 to 1e-15 between `cov_params["a"]["b"]` and
        `cov_params["b"]["a"]`), so compare entries with a tolerance rather
        than with `==`.
        """
        return cov_params_by_param(self._raw.param_names, self._raw.cov_params)

    @property
    def test_stats(self) -> dict[str, float]:
        """Coefficient name to test statistic (t-statistic; see
        `stat_dist`)."""
        return by_param(self._raw.param_names, self._raw.test_stats)

    @property
    def stat_dist(self) -> Literal["t"]:
        """Distribution of `test_stats`. Always `"t"`: t-statistics are
        used for every `cov_type`."""
        return self._raw.stat_dist

    @property
    def stat_df(self) -> int:
        """Degrees of freedom of the t distribution behind `test_stats`.
        May differ from `df_resid` (cluster-robust inference uses
        `G - 1`, Driscoll-Kraay the number of `dk_time` periods minus
        1), so use this to recompute p-values from `test_stats`."""
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
        """Residual degrees of freedom (`n - df_model`, where `df_model`
        counts the intercept; unlike FE's `n - G1 - k` (one-way;
        `n - G1 - G2 + 1 - k` for two-way), with `G1` and `G2` the
        group counts of the first and second `effects` column, since
        RE is a GLS transform that does not consume entity-dummy
        degrees of freedom)."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Number of estimated coefficients, including the intercept (the
        number of slope coefficients plus 1). Unlike OLS's `df_model`,
        it does not exclude the intercept."""
        return self._raw.df_model

    @property
    def n_groups(self) -> dict[str, int]:
        """Number of groups (unique values) of each column in
        `effects`, keyed by column name in the order given: one entry
        for a one-way model, two for a two-way model. Two-way requires
        a balanced panel, so the count of the second column equals the
        number of observations per group of the first. For example
        `{"id": 3, "t": 2}`."""
        return self._raw.n_groups

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
        """Estimated variance of the effect of the first column of
        `effects` (the entity), σ_u² (a negative
        estimate is raised to 0). Swamy-Arora on an unbalanced panel is
        the `linearmodels` estimator (harmonic mean of the entity
        sizes), which differs from `plm`'s; the difference depends on
        how unbalanced the panel is (2.6 % in one test panel), and on
        a balanced panel the two agree. Wallace-Hussain and Amemiya
        follow `plm` on balanced and unbalanced panels."""
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
        `θ₁ = 1 - σ_ε/σ₁` with `σ₁² = σ_ε² + T·σ_u²` (`T` is the group
        count of the second column of `effects`, the periods).
        It is 0 when σ_u² is 0. `None` for one-way."""
        return self._raw.theta_entity

    @property
    def theta_time(self) -> float | None:
        """Two-way weight θ₂ of the time means: `θ₂ = 1 - σ_ε/σ₂` with
        `σ₂² = σ_ε² + N·σ_λ²` (`N` is the group count of the first
        column of `effects`, the entities). It is 0 when σ_λ² is 0.
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
        `cov_type="cluster"`, the number of `dk_time` periods minus 1 for
        `"dk"`, `df_resid` otherwise),
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
        version, `hausman_df × F` of the Wald test that the
        `hausman_df` within-transformed slope regressors are jointly
        zero in the auxiliary regression of the quasi-demeaned `y` on
        the quasi-demeaned regressors and the within-transformed
        regressors). Always non-negative. The comparison is against FE
        with the same structure as RE (entity, or entity and time for
        two-way effects). The Wald test's covariance follows
        `cov_type` (default `"cluster"` gives the cluster-robust
        Hausman test; `"classical"` gives the classical version). The
        auxiliary regression uses the same small-sample corrections as
        the RE standard errors. For unbalanced panels the
        auxiliary regression's constant is left untransformed, as in
        `plm::phtest`. `None` when no coefficient can be tested (no
        regressor varies after the within transformation, or the
        remaining ones are aliased, for example because the entity
        mean of every time-varying regressor is also a regressor). If
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
