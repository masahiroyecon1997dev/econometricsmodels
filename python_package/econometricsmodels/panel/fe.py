"""Python wrapper for FE (fixed effects panel regression).

A thin wrapper around `econometricsmodels._lib.fit_fe` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — `entity`/`time` as bare
column-name arguments, `x` as a list, an options object for estimation
settings (CLAUDE.md section 2, `.claude/rules/python-style.md`
"設計方針との整合性", `docs/planning/specs/panel-api-design.md` section 1).

`FeOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; same policy as `OLSOptions`/`IvOptions`).

`fixed_effects()` (the fixed effects themselves, `α_i`/`γ_t`) is
provided as a separate method rather than a field on `FeResults`
(`docs/planning/specs/panel-api-design.md` section 6.6, the same
"additional results are a separate method" policy as IV's
`first_stage()`).

`summary()` is not implemented (structured-data-only output policy; see
the `OlsResults`/`IvResults` precedent).
"""

from __future__ import annotations

import polars as pl

from .. import _lib
from .._lib import FeOptions

__all__ = ["FE", "FeOptions", "FeResults"]


class FE:
    """Fixed effects (within) panel regression estimator.

    Supports one-way (entity) and two-way (entity + time) fixed
    effects, selected via `FeOptions.time` (`docs/planning/specs/
    panel-api-design.md` section 6.2).

    Args:
        data: A polars DataFrame containing the dependent variable,
            independent variables, entity identifier, and (if
            specified) time/cluster/HAC time columns.
        y: Column name of the dependent variable.
        x: List of column names of the independent variables. Must
            contain at least one column name.
        entity: Column name of the entity (individual/panel unit)
            identifier.
        options: Estimation options. Defaults to `FeOptions()`
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
        options: FeOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._entity = entity
        self._options = options if options is not None else FeOptions()

    def fit(self) -> FeResults:
        """Estimate the FE model.

        Returns:
            The estimation results.

        Raises:
            ValidationError: The input or options are invalid (`x` is
                empty, a column is missing, contains missing values or
                NaN/infinity, `y`/`x`/`entity`/`time` overlap,
                insufficient observations, `confidence_level` out of
                range, an unknown `cov_type` (or `cov_type="hc0"`,
                unsupported for FE), an unbalanced panel with two-way
                effects, a singleton entity/time group, or an
                explanatory variable with zero variance after the
                within transformation). A subclass of `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. a singular within-transformed design
                matrix). A subclass of `RuntimeError`.
        """
        raw = _lib.fit_fe(
            self._data, self._y, self._x, self._entity, self._options
        )
        return FeResults(raw)


class FeResults:
    """FE estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing.

    `fixed_effects()` (the fixed effects themselves) is provided as a
    separate method rather than a field on this class
    (`docs/planning/specs/panel-api-design.md` section 6.6).

    Args:
        raw: The estimation result object returned by `_lib.fit_fe`
            (`_lib.FeResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `FE.fit()`.
    """

    def __init__(self, raw: _lib.FeResult) -> None:
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
    def t_stats(self) -> dict[str, float]:
        """Coefficient name to t-statistic."""
        return dict(zip(self._raw.param_names, self._raw.t_stats))

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
        policy as `OlsResults.coef_table()`). Returned as `list[dict]`
        rather than a polars DataFrame, per the project's policy of
        not using DataFrames for the coefficient table itself.

        Returns:
            A list of dictionaries, one per coefficient. Keys are
            `param`, `coef`, `std_err`, `t_stat`, `p_value`,
            `conf_lower`, `conf_upper`.
        """
        return [
            {
                "param": name,
                "coef": coef,
                "std_err": se,
                "t_stat": t,
                "p_value": p,
                "conf_lower": lower,
                "conf_upper": upper,
            }
            for name, coef, se, t, p, lower, upper in zip(
                self._raw.param_names,
                self._raw.params,
                self._raw.std_errors,
                self._raw.t_stats,
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
        `"time"`. See `docs/planning/specs/panel-api-design.md`
        section 6.6 and `_lib.FeResult.fixed_effects`'s docstring for
        the exact formula, including the two-way normalization
        convention (which does not always numerically match
        `fixest::fixef()`).

        Returns:
            The fixed effects, shaped as described above.
        """
        return self._raw.fixed_effects()
