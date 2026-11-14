"""Python wrapper for FEGMM (fixed effects with generalized method of moments).

A thin wrapper around `econometricsmodels._lib.fit_fe_gmm` (the Rust
implementation). Validation and estimation happen on the Rust side; this
module only provides the Python-facing API shape for polars DataFrames:
lists of column names for `x_exog`, `x_endog`, `instruments` and
`effects`, and an options object for the estimation settings.

`FEGMMOptions` is re-exported as-is from `_lib`. The C statistic uses the
same result type as `IVResults`. There is no `summary()`: results are
returned as structured data.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import FEGMMOptions
from .._results_utils import (
    by_param,
    c_statistic_result,
    conf_int_by_param,
    cov_params_by_param,
    params_table_rows,
    results_repr,
)
from ..diagnostics import CStatisticResult

__all__ = ["FEGMM", "FEGMMOptions", "FEGMMResults"]


class FEGMM:
    """Fixed effects generalized method of moments (FE-GMM) estimator.

    Removes the fixed effects by the within transformation and then
    applies GMM to the transformed data, with the endogenous variables
    instrumented by `instruments`. The default is two-step GMM weighted
    by the cluster-robust covariance of the moment conditions, which is
    efficient under arbitrary correlation within the clusters (for
    example over time within an entity). The `effects` argument works as
    in `FE`: one column gives one-way effects, two columns give two-way
    effects, and three to five columns give higher-dimensional effects.

    Args:
        data: A polars DataFrame containing the dependent variable, the
            exogenous and endogenous independent variables, the
            instruments, the `effects` columns, and (if specified) the
            `cluster` and `dk_time` columns of the options.
        y: Column name of the dependent variable.
        x_exog: List of column names of the exogenous independent
            variables. May be empty.
        x_endog: List of column names of the endogenous independent
            variables. Must contain at least one column name.
        instruments: List of column names of the excluded instruments.
            Must contain at least as many columns as `x_endog`; with
            more, the model is overidentified and the Hansen J test is
            reported. No column may appear in more than one of `y`,
            `x_exog`, `x_endog`, `instruments` and `effects`.
        effects: List of one to five column names that define the fixed
            effects, with the same meaning and order as in `FE`: the
            first is the entity identifier (and the default cluster
            key), the optional second is the time period, and the
            optional third to fifth are further grouping columns.
        options: Estimation options. Defaults to `FEGMMOptions()`
            (two-step GMM with the cluster weight, `cov_type="cluster"`
            on the first column of `effects`, `confidence_level=0.95`)
            when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import FEGMM
        >>> df = pl.DataFrame(
        ...     {
        ...         "y": [1.0, 2.1, 2.9, 1.5, 2.2, 3.8, 0.9, 2.4, 3.3]
        ...         + [1.8, 2.6, 4.1],
        ...         "endog": [1.0, 2.2, 3.1, 1.2, 2.0, 3.9, 0.8, 2.1, 3.5]
        ...         + [1.9, 2.5, 3.8],
        ...         "z1": [0.9, 2.5, 2.7, 1.6, 1.8, 4.0, 1.1, 2.3, 3.0]
        ...         + [1.7, 2.9, 4.2],
        ...         "z2": [1.2, 2.0, 3.4, 1.1, 2.6, 3.5, 1.0, 2.9, 3.1]
        ...         + [2.2, 2.3, 3.6],
        ...         "id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
        ...     }
        ... )
        >>> result = FEGMM(
        ...     df,
        ...     y="y",
        ...     x_exog=[],
        ...     x_endog=["endog"],
        ...     instruments=["z1", "z2"],
        ...     effects=["id"],
        ... ).fit()
        >>> list(result.params)
        ['endog']
        >>> result.gmm_type
        'two_step'
        >>> result.overid_df
        1
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x_exog: list[str],
        x_endog: list[str],
        instruments: list[str],
        effects: list[str],
        *,
        options: FEGMMOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x_exog = x_exog
        self._x_endog = x_endog
        self._instruments = instruments
        self._effects = effects
        self._options = options if options is not None else FEGMMOptions()

    def fit(self) -> FEGMMResults:
        """Estimate the FEGMM model.

        Returns:
            The estimation results.

        Raises:
            TypeError: An argument has the wrong type (for example
                `x_endog` or `effects` is a string instead of a list of
                column names). A builtin exception, not a
                `ValidationError`.
            ValidationError: The input or options are invalid. A
                subclass of `ValueError`. For example:

                - A column is missing, or a column contains missing
                  values, NaN or infinity.
                - `x_endog` or `instruments` is empty, or there are
                  fewer `instruments` than `x_endog` columns (the model
                  is not identified).
                - `effects` is empty or has more than five columns, or
                  `y`, `x_exog`, `x_endog`, `instruments` and `effects`
                  overlap, or `dk_time` equals `y` or the first column
                  of `effects`.
                - There are too few degrees of freedom (too few
                  observations for the fixed effects and coefficients), or
                  `confidence_level` is out of range.
                - `cov_type` is unknown or `"hc0"` (not supported for
                  FEGMM), or `gmm_type` or `gmm_weight_type` is unknown
                  (there is no `"hac"` weight).
                - `cluster`, `dk_time`, `dk_bandwidth`,
                  `gmm_weight_type`, `gmm_max_iter`, `gmm_tol` or
                  `raise_on_non_convergence` is given with a `cov_type`
                  or `gmm_type` that does not use it, or
                  `cov_type="dk"` is requested without `dk_time`.
                - A group of a column in `effects` has a single
                  observation.
                - The cluster weight is used with fewer clusters than the
                  moment conditions (the columns of `x_exog` and
                  `instruments`).

            ComputationError: A problem was detected during
                computation (e.g. a singular within-transformed design
                matrix, a perfect fit of the structural equation,
                alternating projections that did not converge within
                `max_iter`, or `gmm_type="iterated"` that did not
                converge within `gmm_max_iter` while
                `raise_on_non_convergence` is true). A subclass of
                `RuntimeError`.
        """
        raw = _lib.fit_fe_gmm(
            self._data,
            self._y,
            self._x_exog,
            self._x_endog,
            self._instruments,
            self._effects,
            self._options,
        )
        return FEGMMResults(raw)


class FEGMMResults:
    """FEGMM estimation results.

    Coefficient-level properties (`params`, `std_errors`, etc.) are
    dictionaries keyed by coefficient name, so one coefficient is
    `result.params["endog"]`. Use `params_table()` for a row-oriented
    listing (a list of dicts), for example for a JSON response.

    Printing a result shows a header line (the dependent variable,
    `n_obs` and `cov_type`) and a short table of the coefficients and
    their standard errors. Its layout may change between versions, so
    read the fields listed below or use `params_table()` rather than
    parsing it.

    Fields by purpose:

    - Coefficients (no intercept): `params`, `std_errors`,
      `test_stats`, `p_values`, `conf_int`; the names in order:
      `param_names`.
    - Covariance of the coefficients: `cov_params`.
    - Which distribution the tests use: `stat_dist`, `stat_df`.
    - Overall Wald test: `wald_statistic`, `wald_p_value`,
      `wald_dist`, `wald_df_num`, `wald_df_denom`.
    - Overidentification (Hansen J): `overid_statistic`,
      `overid_p_value`, `overid_df`.
    - Estimation method actually used: `gmm_type`, `gmm_weight_type`,
      `n_iter`, `converged`.
    - Fit: `r_squared_within`, `r_squared_between`,
      `r_squared_overall`.
    - Panel size: `n_obs`, `n_groups`.
    - Model size: `df_model`, `df_resid`.
    - Settings echoed back: `dep_var_name`, `cov_type`,
      `dk_bandwidth_used`.
    - Residuals: `residuals` (structural residuals of the
      within-transformed data).
    - Methods: `params_table()`.
    - Diagnostic (a method, computed when called): `c_stat()`.

    Note:
        Users normally do not construct this directly; it is returned
        by `FEGMM.fit()`.
    """

    def __init__(self, raw: _lib.FEGMMResult) -> None:
        self._raw = raw

    def __repr__(self) -> str:
        return results_repr(type(self).__name__, self._raw)

    @property
    def param_names(self) -> list[str]:
        """List of coefficient names (no intercept: within-transformed
        variables have no constant term)."""
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
        """Coefficient name to test statistic (z-statistic; see
        `stat_dist`)."""
        return by_param(self._raw.param_names, self._raw.test_stats)

    @property
    def stat_dist(self) -> Literal["normal"]:
        """Distribution of `test_stats`. Always `"normal"` (standard
        normal) for GMM."""
        return self._raw.stat_dist

    @property
    def stat_df(self) -> int | None:
        """Degrees of freedom of the distribution behind `test_stats`.
        Always `None`: the standard normal distribution has none."""
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
        """Structural residuals, in observation order: the
        within-transformed `y` minus the within-transformed `X` times
        `β̂`, with the actual endogenous variables (not their
        first-stage fitted values)."""
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
        """Residual degrees of freedom: `n - df_model`, where `df_model`
        is the model degrees of freedom below. The tests use the normal
        and chi-squared distributions, so it does not enter the p-values;
        it is the divisor of the degrees-of-freedom corrections of the
        classical covariance."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom: the `k` slope coefficients plus the
        absorbed fixed effects (`k + sum(G) - (number of columns - 1)`,
        where the `G` are the group counts in `n_groups`); the number of
        slope coefficients is `len(param_names)`."""
        return self._raw.df_model

    @property
    def n_groups(self) -> dict[str, int]:
        """Number of groups (unique values) of each column in
        `effects`, keyed by column name in the order given: one entry
        for a one-way model, two for a two-way model, and so on. For
        example `{"id": 3, "t": 2}`."""
        return self._raw.n_groups

    @property
    def cov_type(self) -> str:
        """Standard error type actually used (normalized to lowercase)."""
        return self._raw.cov_type

    @property
    def dk_bandwidth_used(self) -> int | None:
        """Driscoll-Kraay bandwidth actually used: the explicit
        `dk_bandwidth` if given, otherwise the value chosen automatically,
        `floor(4 * (t / 100) ** (2 / 9))` where `t` is the number of
        unique time periods in `FEGMMOptions.dk_time`. `None` unless
        `cov_type="dk"`."""
        return self._raw.dk_bandwidth_used

    @property
    def gmm_type(self) -> Literal["one_step", "two_step", "iterated"]:
        """GMM estimation type actually used (normalized to lowercase):
        `"one_step"`, `"two_step"` or `"iterated"`."""
        return self._raw.gmm_type

    @property
    def gmm_weight_type(
        self,
    ) -> Literal["classical", "robust", "cluster"] | None:
        """Weight matrix actually used for the point estimate (normalized
        to lowercase): `"classical"`, `"robust"` or `"cluster"`. `None`
        for `gmm_type="one_step"`, which does not use a weight type."""
        return self._raw.gmm_weight_type

    @property
    def n_iter(self) -> int:
        """Number of GMM estimations actually run, counting the initial
        estimate: 1 for `"one_step"`, 2 for `"two_step"`, at most
        `gmm_max_iter` for `"iterated"`."""
        return self._raw.n_iter

    @property
    def converged(self) -> bool:
        """Whether the iteration converged. Only meaningful for
        `gmm_type="iterated"`; always `True` for `"one_step"` and
        `"two_step"`, which never check convergence."""
        return self._raw.converged

    @property
    def wald_statistic(self) -> float:
        """Wald statistic for all slope coefficients being zero (not
        divided by the number of slope coefficients). It follows a
        chi-squared distribution (`wald_dist`) and uses the covariance of
        the coefficients, so it follows `cov_type`."""
        return self._raw.wald_statistic

    @property
    def wald_p_value(self) -> float:
        """P-value of `wald_statistic`."""
        return self._raw.wald_p_value

    @property
    def wald_dist(self) -> Literal["chi2"]:
        """Distribution of `wald_statistic`. Always `"chi2"`
        (chi-squared) for GMM."""
        return self._raw.wald_dist

    @property
    def wald_df_num(self) -> int | None:
        """Degrees of freedom of the chi-squared distribution of
        `wald_statistic` (the number of slope coefficients)."""
        return self._raw.wald_df_num

    @property
    def wald_df_denom(self) -> int | None:
        """Denominator degrees of freedom of `wald_statistic`. Always
        `None`: the chi-squared distribution has none."""
        return self._raw.wald_df_denom

    @property
    def r_squared_within(self) -> float:
        """Within R² (based on the within-transformed variables, from the
        structural residuals). Can be negative."""
        return self._raw.r_squared_within

    @property
    def r_squared_between(self) -> float:
        """Between R² (based on entity-mean variables, from the structural
        residuals). Can be negative."""
        return self._raw.r_squared_between

    @property
    def r_squared_overall(self) -> float:
        """Overall R² (based on the untransformed variables, from the
        structural residuals). Can be negative."""
        return self._raw.r_squared_overall

    @property
    def overid_statistic(self) -> float | None:
        """Hansen J overidentification test statistic, computed from the
        within-transformed data at the estimate.

        The covariance of the moment conditions follows the weight
        actually used (`gmm_weight_type`; the classical weight for
        `gmm_type="one_step"`). With the classical weight the error
        variance is `e'e / (n - n_absorbed)` (`n_absorbed` is the number
        of absorbed fixed-effect parameters); the robust and cluster
        weights are not adjusted. The cluster weight needs no adjustment
        when the clusters contain the entities (the entities themselves,
        or a coarser column), but not when they cross the entities (for
        example the time periods), and the robust weight makes the test
        over-reject on a short panel.

        `None` when just-identified (`len(instruments) ==
        len(x_endog)`, degrees of freedom 0).
        """
        return self._raw.overid_statistic

    @property
    def overid_p_value(self) -> float | None:
        """P-value of the overidentification test.

        Same conditions as `overid_statistic` for when this is `None`.
        """
        return self._raw.overid_p_value

    @property
    def overid_df(self) -> int | None:
        """Degrees of freedom of the chi-squared overidentification test
        (`len(instruments) - len(x_endog)`). `None` under the same
        conditions as `overid_statistic`."""
        return self._raw.overid_df

    def params_table(self) -> list[dict[str, float | str]]:
        """Row-oriented summary table of the coefficients.

        Shaped to be usable almost as-is in a REST API response (same
        policy as `OLSResults.params_table()`). Returned as `list[dict]`
        rather than a polars DataFrame.

        Returns:
            A list of dictionaries, one per coefficient. Keys are
            `param_name`, `param`, `std_error`, `test_stat`, `p_value`,
            `conf_lower`, `conf_upper`.
        """
        return params_table_rows(self._raw)

    def c_stat(
        self,
        *,
        variables: list[str] | None = None,
        instruments: list[str] | None = None,
    ) -> CStatisticResult:
        """C statistic (difference-in-Hansen test).

        Tests whether endogenous variables in `variables` are in fact
        exogenous, and whether excluded instruments in `instruments` are
        uncorrelated with the error. The model is estimated by two-step GMM
        on the within-transformed data with and without the tested moment
        conditions, and the statistic is the difference of the two Hansen J
        statistics, which is chi-squared distributed under the null that
        the tested moment conditions are valid. The test is conditional:
        the remaining instruments are taken as valid. Both J statistics use
        the same estimate of the covariance of the moment conditions, so
        the statistic is never negative. Pass `variables`, `instruments`
        or both.

        The covariance of the moment conditions follows `gmm_weight_type`
        (grouped by `cluster` for the cluster weight) when `gmm_type` is
        `"two_step"` or `"iterated"`,
        and is the classical one for `gmm_type="one_step"`, which has no
        weight matrix. It does not depend on `cov_type`, `gmm_max_iter` or
        `gmm_tol`, and a model fitted with `gmm_type="one_step"` or
        `"iterated"` gets the two-step statistic. As for the Hansen J
        statistic, only the classical weight is adjusted by `n - n_absorbed`
        for the absorbed fixed effects. The cluster weight needs no
        adjustment when the clusters contain the entities, but not when
        they cross the entities (for example the time periods), and with a
        robust weight on a short panel the statistic over-rejects.

        This is a post-estimation diagnostic: it is never computed by
        `fit()`. It re-reads the variables from the data passed to
        `fit()` (keep that data unchanged) and estimates the model again
        on every call.

        Args:
            variables: Names from `x_endog` whose exogeneity is tested.
                Defaults to `None` (no endogenous variable is tested). The
                names are reported in the order of `x_endog`, not the order
                given.
            instruments: Names from the excluded `instruments` whose
                orthogonality is tested. Defaults to `None` (no instrument
                is tested). At least as many instruments as endogenous
                variables must remain.

        Returns:
            A `CStatisticResult` with the statistic, p-value, degrees of
            freedom and the two J statistics.

        Raises:
            TypeError: `variables` or `instruments` is not a list of str.
            ValidationError: Neither argument is given, a list is empty or
                repeats a name, a name in `variables` is not in `x_endog`,
                a name in `instruments` is not among the excluded
                instruments, or so many instruments are tested that fewer
                excluded instruments than endogenous variables remain.
            ComputationError: The statistic is undefined for these data, for
                example because the instruments are collinear, or because
                there are fewer clusters than the columns of the full
                instrument set (the exogenous regressors, the tested
                endogenous variables and all excluded instruments). Unlike
                `fit()`, which rejects too few clusters for the cluster
                weight with a `ValidationError`, the diagnostic can only
                detect this while computing.
        """
        return c_statistic_result(self._raw.c_stat(variables, instruments))
