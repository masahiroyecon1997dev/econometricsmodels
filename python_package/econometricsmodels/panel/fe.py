"""Python wrapper for FE (fixed effects panel regression).

A thin wrapper around `econometricsmodels._lib.fit_fe` (the Rust
implementation). Validation and estimation happen on the Rust side; this
module only provides the Python-facing API shape for polars DataFrames:
`effects` and `x` as lists of column names, and an options object for
the estimation settings.

`FEOptions` is re-exported as-is from `_lib`. The fixed effects
themselves (`α_i`, `γ_t`) come from the separate method
`FEResults.fixed_effects()`. There is no `summary()`: results are
returned as structured data.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import FEOptions
from .._results_utils import (
    by_param,
    conf_int_by_param,
    cov_params_by_param,
    params_table_rows,
)

__all__ = ["FE", "FEOptions", "FEResults"]


class FE:
    """Fixed effects (within) panel regression estimator.

    Supports one-way and two-way fixed effects, selected by the number
    of columns in `effects`: one column (the entity identifier) for
    one-way effects, two columns (the entity identifier, then the time
    identifier) for two-way effects.

    Args:
        data: A polars DataFrame containing the dependent variable,
            independent variables, the `effects` columns, and (if
            specified) the `cluster` and `dk_time` columns of the
            options.
        y: Column name of the dependent variable.
        x: List of column names of the independent variables. Must
            contain at least one column name.
        effects: List of one or two column names that define the fixed
            effects. The first is the entity (individual/panel unit)
            identifier; the optional second is the time identifier,
            which makes the model two-way (balanced panels only). The
            order matters: the first column is the default cluster key,
            and the first period of the second column (in time order)
            is the reference period of `fixed_effects()`. Swapping the
            columns leaves the coefficients unchanged.
        options: Estimation options. Defaults to `FEOptions()`
            (`cov_type="cluster"` on the first column of `effects`,
            confidence_level=0.95) when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import FE
        >>> df = pl.DataFrame(
        ...     {
        ...         "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ...         "x1": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
        ...         "id": ["a", "a", "b", "b", "c", "c"],
        ...     }
        ... )
        >>> result = FE(df, y="y", x=["x1"], effects=["id"]).fit()
        >>> round(result.params["x1"], 3)
        0.31

        Two-way effects take a second column, the time identifier:

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
        >>> two_way = FE(
        ...     panel, y="y", x=["x1"], effects=["id", "t"]
        ... ).fit()
        >>> two_way.n_groups
        {'id': 4, 't': 3}
        >>> list(two_way.fixed_effects())
        ['id', 't']
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        effects: list[str],
        *,
        options: FEOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._effects = effects
        self._options = options if options is not None else FEOptions()

    def fit(self) -> FEResults:
        """Estimate the FE model.

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
                  FE).
                - A two-way model (two columns in `effects`) is fitted
                  on an unbalanced panel.
                - A group of a column in `effects` has a single
                  observation.
                - An explanatory variable has zero variance after the
                  within transformation, or the outcome has no variation
                  left after the fixed effects are removed.
                - `cov_type="dk"` without `dk_time` (the time periods
                  are never taken from `effects`).
                - `cov_type="dk"` with no more unique `dk_time` periods
                  than regressors: the Driscoll-Kraay covariance has
                  rank at most `T - 1` (`T` periods), so the F-test of
                  the slopes cannot be computed.
                - `cov_type="dk"` with 2 `dk_time` periods, or
                  `cov_type="cluster"` with 2 clusters, where every
                  entity (with two-way effects, every time period) is
                  observed exactly once in each: the covariance is then
                  identically zero.

            ComputationError: A problem was detected during
                computation (e.g. a singular within-transformed design
                matrix, or a perfect fit of the within regression:
                residuals zero up to rounding). A subclass of
                `RuntimeError`.
        """
        raw = _lib.fit_fe(
            self._data, self._y, self._x, self._effects, self._options
        )
        return FEResults(raw)


class FEResults:
    """FE estimation results.

    Coefficient-level properties (`params`, `std_errors`, etc.) are
    dictionaries keyed by coefficient name, so one coefficient is
    `result.params["x1"]`. Use `params_table()` for a row-oriented
    listing (a list of dicts), for example for a JSON response.

    Fields by purpose:

    - Coefficients (no intercept): `params`, `std_errors`,
      `test_stats`, `p_values`, `conf_int`; the names in order:
      `param_names`.
    - Covariance of the coefficients: `cov_params`.
    - Which distribution the tests use: `stat_dist`, `stat_df`.
    - Overall F test: `f_statistic`, `f_p_value`, `f_df_num`,
      `f_df_denom`.
    - Fit: `r_squared_within`, `r_squared_between`,
      `r_squared_overall`, `log_likelihood`, `aic`, `bic`.
    - Panel size: `n_obs`, `n_groups`.
    - Model size: `df_model`, `df_resid`, `dep_var_name`.
    - Settings echoed back: `cov_type`, `dk_bandwidth_used`.
    - Residuals: `residuals` (within-transformed).
    - Methods: `fixed_effects()` (the estimated effect of every group,
      for each effects column of the model),
      `params_table()`.

    Note:
        Users normally do not construct this directly; it is returned
        by `FE.fit()`.
    """

    def __init__(self, raw: _lib.FEResult) -> None:
        self._raw = raw

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
        """Within-transformed residuals (in observation order)."""
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
        """Residual degrees of freedom (`n - G1 - k` for one-way,
        `n - G1 - G2 + 1 - k` for two-way, where `G1` and `G2` are the
        group counts of the first and second column of `effects`, as in
        `n_groups`)."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom: the `k` slope coefficients plus the
        absorbed fixed effects (`k + G1` for one-way, `k + G1 + G2 - 1`
        for two-way, where `G1` and `G2` are the group counts of the
        first and second column of `effects`, as in `n_groups`). It is
        used for the degrees of freedom of the inference (`df_resid = n
        - df_model`) and for the information criteria; the number of
        slope coefficients is `len(param_names)`."""
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
    def cov_type(self) -> str:
        """Standard error type actually used (normalized to lowercase)."""
        return self._raw.cov_type

    @property
    def dk_bandwidth_used(self) -> int | None:
        """Driscoll-Kraay bandwidth actually used: the explicit
        `dk_bandwidth` if given, otherwise the value chosen automatically,
        `floor(4 * (t / 100) ** (2 / 9))` where `t` is the number of
        unique time periods in `FEOptions.dk_time`. `None` unless
        `cov_type="dk"`."""
        return self._raw.dk_bandwidth_used

    @property
    def f_statistic(self) -> float | None:
        """F-statistic for the joint significance of the slope
        coefficients (classical F-test when `cov_type="classical"`, a
        robust Wald test otherwise). `None` when there are no slope
        coefficients."""
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
        """Within R² (based on the within-transformed variables)."""
        return self._raw.r_squared_within

    @property
    def r_squared_between(self) -> float:
        """Between R² (based on entity-mean variables)."""
        return self._raw.r_squared_between

    @property
    def r_squared_overall(self) -> float:
        """Overall R² (based on the untransformed variables)."""
        return self._raw.r_squared_overall

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

    def fixed_effects(self) -> dict[str, dict[str, float]]:
        """The fixed effects themselves (`α_i` for the first column of
        `effects`, `γ_t` for the second), recovered post-hoc from the
        fitted coefficients.

        Always a `dict[str, dict[str, float]]` keyed by the column
        names in `effects`, then by group (the value of the column as a
        string): one entry for a one-way model, two for a two-way
        model. `α_i`/`γ_t` are not individually identified in the
        two-way case, so `γ_t` of the first period in time order (the
        order of the values of the second column) is fixed to 0 and
        `α_i` absorbs the overall level. The dictionary of the second
        column lists the periods in time order.
        `fixest::fixef()` instead uses the first time value in
        observation order as the reference, so the two match
        numerically only when both choose the same reference period.

        Returns:
            The fixed effects, shaped as described above. The group key
            is always a string, even for an integer column, for example
            `result.fixed_effects()["id"]["a"]`.
        """
        return self._raw.fixed_effects()
