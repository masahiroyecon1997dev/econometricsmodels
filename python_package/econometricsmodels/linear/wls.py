"""Python wrapper for WLS (Weighted Least Squares).

A thin wrapper around `econometricsmodels._lib.fit_wls` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — a list of column names
for `x`, an options object for estimation settings (CLAUDE.md section 2,
`.claude/rules/python-style.md` "設計方針との整合性").

`weight`, like `y`, is a top-level argument referring to a column name
in `data` (see `docs/spec/wls-spec.md`, "API引数").
Since the estimation options WLS needs are identical to OLS's, no
separate options class is introduced; WLS reuses `OLSOptions` as-is
(see section 3).
"""

from __future__ import annotations

import polars as pl

from .. import _lib
from .._lib import OLSOptions

__all__ = ["WLS", "WlsResults"]


class WLS:
    """Weighted Least Squares regression estimator.

    Args:
        data: A polars DataFrame containing the dependent variable,
            independent variables, and weight column.
        y: Column name of the dependent variable.
        x: List of column names of the independent variables.
        weight: Column name of the weight column. Treated as an
            analytic weight (proportional to the inverse of the
            variance; no normalization required). Non-positive values
            raise `ValidationError`.
        options: Estimation options. Uses the same `OLSOptions` as
            `OLS`. Defaults to `OLSOptions()` (classical, with
            intercept, confidence_level=0.95) when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import WLS
        >>> df = pl.DataFrame(
        ...     {"y": [1.0, 2.0], "x1": [1.0, 2.0], "w": [1.0, 2.0]}
        ... )
        >>> result = WLS(df, y="y", x=["x1"], weight="w").fit()
        >>> result.params["x1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        weight: str,
        options: OLSOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._weight = weight
        self._options = options if options is not None else OLSOptions()

    def fit(self) -> WlsResults:
        """Estimate the WLS model.

        Returns:
            The estimation results.

        Raises:
            ValidationError: The input or options are invalid (a
                column is missing, contains missing values or
                NaN/infinity, a weight is non-positive, `weight`
                duplicates `y`/`x`, insufficient observations,
                `confidence_level` out of range, etc.). A subclass of
                `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. a singular design matrix). A
                subclass of `RuntimeError`.
        """
        raw = _lib.fit_wls(
            self._data, self._y, self._x, self._weight, self._options
        )
        return WlsResults(raw)


class WlsResults:
    """WLS estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing
    (same shape as `OlsResults`; see
    `docs/spec/ols-spec.md`, "結果構造体").

    Args:
        raw: The estimation result object returned by `_lib.fit_wls`
            (`_lib.WLSResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `WLS.fit()`. `residuals` are on the original (unweighted)
        scale, `y_i - x_i'β̂`, which differs from the weighted
        residuals used in the standard error computation (see
        `docs/spec/wls-spec.md`, "結果構造体").
    """

    def __init__(self, raw: _lib.WLSResult) -> None:
        self._raw = raw

    @property
    def param_names(self) -> list[str]:
        """List of coefficient names (`"const"` first when `include_intercept=True`)."""
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
        """Residuals on the original (unweighted) scale (observation order, `y - Xβ̂`)."""
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
    def cov_type(self) -> str:
        """Standard error type actually used (normalized to lowercase)."""
        return self._raw.cov_type

    @property
    def r_squared(self) -> float:
        """Coefficient of determination (R²)."""
        return self._raw.r_squared

    @property
    def r_squared_adj(self) -> float:
        """Degrees-of-freedom-adjusted R²."""
        return self._raw.r_squared_adj

    @property
    def f_statistic(self) -> float:
        """F-statistic."""
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

    def coef_table(self) -> list[dict[str, float | str]]:
        """Row-oriented summary table of the coefficients.

        Shaped to be usable almost as-is in a REST API response (see
        `docs/spec/ols-spec.md`, "結果構造体"). Returned
        as `list[dict]` rather than a polars DataFrame, per the
        project's policy of not using DataFrames for the coefficient
        table itself.

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
