"""Python wrapper for OLS (Ordinary Least Squares).

A thin wrapper around `econometricsmodels._lib.fit_ols` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — a list of column names
for `x`, an options object for estimation settings (CLAUDE.md section 2,
`.claude/rules/python-style.md` "設計方針との整合性").

`OLSOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; see `docs/planning/specs/ols-api-design.md` section 3).
"""

from __future__ import annotations

import polars as pl

from .. import _lib
from .._lib import OLSOptions

__all__ = ["OLS", "OLSOptions", "OlsResults"]


class OLS:
    """Ordinary Least Squares regression estimator.

    Args:
        data: A polars DataFrame containing the dependent and
            independent variables.
        y: Column name of the dependent variable.
        x: List of column names of the independent variables.
        options: Estimation options. Defaults to `OLSOptions()`
            (classical, with intercept, confidence_level=0.95) when
            omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import OLS
        >>> df = pl.DataFrame({"y": [1.0, 2.0], "x1": [1.0, 2.0]})
        >>> result = OLS(df, y="y", x=["x1"]).fit()
        >>> result.params["x1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        options: OLSOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._options = options if options is not None else OLSOptions()

    def fit(self) -> OlsResults:
        """Estimate the OLS model.

        Returns:
            The estimation results.

        Raises:
            ValidationError: The input or options are invalid (a
                column is missing, contains missing values or
                NaN/infinity, insufficient observations,
                `confidence_level` out of range, etc.). A subclass of
                `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. a singular design matrix). A
                subclass of `RuntimeError`.
        """
        raw = _lib.fit_ols(self._data, self._y, self._x, self._options)
        return OlsResults(raw)


class OlsResults:
    """OLS estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing
    (see `docs/planning/specs/ols-api-design.md` section 5).

    Args:
        raw: The estimation result object returned by `_lib.fit_ols`
            (`_lib.OLSResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `OLS.fit()`.
    """

    def __init__(self, raw: _lib.OLSResult) -> None:
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
        """Residuals (in observation order, `y - Xβ̂`)."""
        return self._raw.residuals

    @property
    def dep_var_name(self) -> str:
        """Column name of the dependent variable."""
        return self._raw.dep_var_name

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return self._raw.nobs

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
        `docs/planning/specs/ols-api-design.md` section 5). Returned
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

    def predict(
        self, new_data: pl.DataFrame | None = None
    ) -> list[dict[str, float]]:
        """Predicted values.

        Unified into a single method rather than a separate
        `fitted_values` property, to match the naming used by Logit's
        `predict()` (`docs/planning/specs/ols-api-design.md` section 7).

        Args:
            new_data: New data to predict on. Must contain columns with
                the same names as the `x` columns passed at fit time
                (matched by name; column order does not matter). If
                `include_intercept=True` was used at fit time, the
                constant column is added automatically and must not be
                included here. If `None` (default), returns the fitted
                values for the training data used in `fit()`.

        Returns:
            Row-oriented predictions, one dict per observation. Each
            dict currently has a single key, `"fitted"`.

        Raises:
            ValidationError: `new_data` is missing a required `x`
                column, or a column contains missing/NaN/infinite
                values.
        """
        raw = self._raw.predict(new_data)
        return [{"fitted": value} for value in raw]
