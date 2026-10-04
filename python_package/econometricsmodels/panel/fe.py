"""Python wrapper for FE (fixed effects panel regression).

A thin wrapper around `econometricsmodels._lib.fit_fe` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — `entity`/`time` as bare
column-name arguments, `x` as a list, an options object for estimation
settings (CLAUDE.md section 2, `.claude/rules/python-style.md`
"設計方針との整合性", `docs/spec/panel-common.md` section 1).

`FEOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; same policy as `OLSOptions`/`IVOptions`).

`fixed_effects()` (the fixed effects themselves, `α_i`/`γ_t`) is
provided as a separate method rather than a field on `FEResults`
(`docs/spec/fe-spec.md` section 3.5, the same
"additional results are a separate method" policy as IV's
`first_stage()`).

`summary()` is not implemented (structured-data-only output policy; see
the `OLSResults`/`IVResults` precedent).
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import FEOptions

__all__ = ["FE", "FEOptions", "FEResults"]


class FE:
    """Fixed effects (within) panel regression estimator.

    Supports one-way (entity) and two-way (entity + time) fixed
    effects, selected via `FEOptions.time` (`docs/spec/fe-spec.md`
    section 1).

    Args:
        data: A polars DataFrame containing the dependent variable,
            independent variables, entity identifier, and (if
            specified) time/cluster/HAC time columns.
        y: Column name of the dependent variable.
        x: List of column names of the independent variables. Must
            contain at least one column name.
        entity: Column name of the entity (individual/panel unit)
            identifier.
        options: Estimation options. Defaults to `FEOptions()`
            (`cov_type="cluster"` on `entity`, one-way,
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
        >>> result = FE(df, y="y", x=["x1"], entity="id").fit()
        >>> result.params["x1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        entity: str,
        options: FEOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._entity = entity
        self._options = options if options is not None else FEOptions()

    def fit(self) -> FEResults:
        """Estimate the FE model.

        Returns:
            The estimation results.

        Raises:
            TypeError: An argument has the wrong type (for example
                `x` is a string instead of a list of column names). A
                builtin exception, not a `ValidationError`.
            ValidationError: The input or options are invalid (`x` is
                empty, a column is missing, contains missing values or
                NaN/infinity, `y`/`x`/`entity`/`time` overlap,
                insufficient observations, `confidence_level` out of
                range, an unknown `cov_type` (or `cov_type="hc0"`,
                unsupported for FE), an unbalanced panel with two-way
                effects, a singleton entity/time group, or an
                explanatory variable with zero variance after the
                within transformation, `cov_type="dk"` with no more
                unique time periods than regressors: the
                Driscoll-Kraay covariance has rank at most T-1, so the
                slope F-test cannot be computed, or `cov_type="dk"` with
                2 time periods / `cov_type="cluster"` with 2 clusters
                where every entity (or, with two-way effects, every
                time period) is observed exactly once in each: the
                covariance is then identically zero). A subclass of
                `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. a singular within-transformed design
                matrix). A subclass of `RuntimeError`.
        """
        raw = _lib.fit_fe(
            self._data, self._y, self._x, self._entity, self._options
        )
        return FEResults(raw)


class FEResults:
    """FE estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing.

    `fixed_effects()` (the fixed effects themselves) is provided as a
    separate method rather than a field on this class
    (`docs/spec/fe-spec.md` section 3.5).

    Args:
        raw: The estimation result object returned by `_lib.fit_fe`
            (`_lib.FEResult`).

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
        return dict(zip(self._raw.param_names, self._raw.params))

    @property
    def std_errors(self) -> dict[str, float]:
        """Coefficient name to standard error."""
        return dict(zip(self._raw.param_names, self._raw.std_errors))

    @property
    def test_stats(self) -> dict[str, float]:
        """Coefficient name to test statistic (t-statistic; see
        `stat_dist`)."""
        return dict(zip(self._raw.param_names, self._raw.test_stats))

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
        return dict(zip(self._raw.param_names, self._raw.p_values))

    @property
    def conf_int(self) -> dict[str, tuple[float, float]]:
        """Coefficient name to confidence interval `(lower, upper)`."""
        return {
            name: (lower, upper)
            for name, lower, upper in zip(
                self._raw.param_names,
                self._raw.conf_lower,
                self._raw.conf_upper,
            )
        }

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
        """Residual degrees of freedom (`n - n_entities - k` for
        one-way, `n - n_entities - n_periods + 1 - k` for two-way)."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (`k`, the number of slope
        coefficients)."""
        return self._raw.df_model

    @property
    def n_entities(self) -> int:
        """Number of panel entities."""
        return self._raw.n_entities

    @property
    def n_periods(self) -> int | None:
        """Number of unique time periods (two-way effects only; `None`
        for one-way). Two-way requires a balanced panel, so this equals
        the number of observations per entity."""
        return self._raw.n_periods

    @property
    def cov_type(self) -> str:
        """Standard error type actually used (normalized to lowercase)."""
        return self._raw.cov_type

    @property
    def f_statistic(self) -> float:
        """F-statistic for the joint significance of the slope
        coefficients (classical F-test when `cov_type="classical"`, a
        robust Wald test otherwise)."""
        return self._raw.f_statistic

    @property
    def f_p_value(self) -> float:
        """P-value of the F-statistic."""
        return self._raw.f_p_value

    @property
    def f_df_num(self) -> int | None:
        """Numerator degrees of freedom of `f_statistic` (`None` when it
        is NaN, i.e. there are no slope coefficients)."""
        return self._raw.f_df_num

    @property
    def f_df_denom(self) -> int | None:
        """Denominator degrees of freedom of `f_statistic` (`None` when it
        is NaN). Follows `cov_type` like the t-tests (`G - 1` for
        `cov_type="cluster"`, `T - 1` for `"dk"`, `df_resid` otherwise),
        so it equals `stat_df` whenever `f_statistic` is not NaN. Use this
        to recompute the p-value from `f_statistic`."""
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

    def coef_table(self) -> list[dict[str, float | str]]:
        """Row-oriented summary table of the coefficients.

        Shaped to be usable almost as-is in a REST API response (same
        policy as `OLSResults.coef_table()`). Returned as `list[dict]`
        rather than a polars DataFrame, per the project's policy of
        not using DataFrames for the coefficient table itself.

        Returns:
            A list of dictionaries, one per coefficient. Keys are
            `param`, `coef`, `std_err`, `test_stat`, `p_value`,
            `conf_lower`, `conf_upper`.
        """
        return [
            {
                "param": name,
                "coef": coef,
                "std_err": se,
                "test_stat": t,
                "p_value": p,
                "conf_lower": lower,
                "conf_upper": upper,
            }
            for name, coef, se, t, p, lower, upper in zip(
                self._raw.param_names,
                self._raw.params,
                self._raw.std_errors,
                self._raw.test_stats,
                self._raw.p_values,
                self._raw.conf_lower,
                self._raw.conf_upper,
            )
        ]

    def fixed_effects(
        self,
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """The fixed effects themselves (`α_i` for entity, `γ_t` for
        time), recovered post-hoc from the fitted coefficients.

        One-way: `dict[str, float]` keyed by entity id. Two-way:
        `dict[str, dict[str, float]]` with top-level keys `"entity"`/
        `"time"`. See `docs/spec/fe-spec.md`
        section 3.5 and `_lib.FEResult.fixed_effects`'s docstring for
        the exact formula, including the two-way normalization
        convention. `α_i`/`γ_t` are not individually identified in the
        two-way case, so `γ_t` of the lexicographically smallest time
        value is fixed to 0 and `α_i` absorbs the overall level.
        `fixest::fixef()` instead uses the first time value in
        observation order as the reference, so the two match
        numerically only when both choose the same reference period.

        Returns:
            The fixed effects, shaped as described above.
        """
        return self._raw.fixed_effects()
