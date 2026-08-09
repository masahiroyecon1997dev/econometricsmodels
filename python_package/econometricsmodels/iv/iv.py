"""Python wrapper for IV (instrumental variables: 2SLS/GMM).

A thin wrapper around `econometricsmodels._lib.fit_iv` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — lists of column names
for `x_exog`/`x_endog`/`instruments`, an options object for estimation
settings (CLAUDE.md section 2, `.claude/rules/python-style.md`
"設計方針との整合性").

`IvOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; same policy as `OLSOptions`/`LogitOptions`, see
`docs/spec/ols-spec.md`, "API引数"). `IvOptions.method` selects `"2sls"`
(the only method currently implemented; `"gmm"` raises
`ValidationError`) — a single `IV`/`IvResults` pair serves both methods
(`docs/planning/specs/iv-api-design.md` section 1.2).

`summary()` is not implemented (structured-data-only output policy; see
the `OlsResults`/`LogitResults` precedent).
"""

from __future__ import annotations

import polars as pl

from .. import _lib
from .._lib import IvOptions
from ..linear.ols import OlsResults

__all__ = ["IV", "IvOptions", "IvResults"]


class IV:
    """Instrumental variables estimator (2SLS/GMM).

    Args:
        data: A polars DataFrame containing the dependent variable,
            exogenous/endogenous independent variables, and instrument
            columns.
        y: Column name of the dependent variable.
        x_exog: List of column names of the exogenous independent
            variables.
        x_endog: List of column names of the endogenous independent
            variables.
        instruments: List of column names of the excluded instruments
            (must not overlap `x_exog`; see
            `docs/planning/specs/iv-api-design.md` section 1.1.1).
        options: Estimation options. Defaults to `IvOptions()`
            (`method="2sls"`, classical, with intercept,
            confidence_level=0.95) when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import IV
        >>> df = pl.DataFrame(
        ...     {
        ...         "y": [1.0, 2.0, 3.0, 4.0],
        ...         "endog1": [2.0, 1.0, 4.0, 3.0],
        ...         "z1": [1.0, 3.0, 2.0, 4.0],
        ...     }
        ... )
        >>> result = IV(
        ...     df, y="y", x_exog=[], x_endog=["endog1"], instruments=["z1"]
        ... ).fit()
        >>> result.params["endog1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x_exog: list[str],
        x_endog: list[str],
        instruments: list[str],
        options: IvOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x_exog = x_exog
        self._x_endog = x_endog
        self._instruments = instruments
        self._options = options if options is not None else IvOptions()

    def fit(self) -> IvResults:
        """Estimate the IV model.

        Returns:
            The estimation results.

        Raises:
            ValidationError: The input or options are invalid (a
                column is missing, contains missing values or
                NaN/infinity, `y`/`x_exog`/`x_endog`/`instruments`
                overlap, insufficient observations,
                `confidence_level` out of range, an unknown
                `cov_type` string, `method="gmm"` (not yet
                implemented), or too few instruments for
                identification). A subclass of `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. a singular first- or second-stage
                design matrix). A subclass of `RuntimeError`.
        """
        raw = _lib.fit_iv(
            self._data,
            self._y,
            self._x_exog,
            self._x_endog,
            self._instruments,
            self._options,
        )
        return IvResults(raw)


class IvResults:
    """IV estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing.

    `first_stage()` (per-endogenous-variable first-stage regression
    results) is provided as a separate method rather than a field on
    this class (`docs/planning/specs/iv-api-design.md` section 2.2).

    Args:
        raw: The estimation result object returned by `_lib.fit_iv`
            (`_lib.IvResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `IV.fit()`.
    """

    def __init__(self, raw: _lib.IvResult) -> None:
        self._raw = raw

    @property
    def param_names(self) -> list[str]:
        """List of coefficient names (`"const"` first when
        `include_intercept=True`)."""
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
    def stats(self) -> dict[str, float]:
        """Coefficient name to test statistic.

        t-statistic for `method="2sls"`, z-statistic for
        `method="gmm"` — named generically (not `t_stats`/`z_stats`)
        because `IvResults` is shared by both methods (mirrors the
        `_lib.IvResult.stats` naming, `docs/planning/specs/
        iv-api-design.md` section 2.1).
        """
        return dict(zip(self._raw.param_names, self._raw.stats))

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
        """Structural residuals (in observation order, `y - Xβ̂`, using
        the actual endogenous variables rather than their first-stage
        fitted values)."""
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
        """Residual degrees of freedom (`n - k`)."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (`k` minus 1 if
        `include_intercept=True`)."""
        return self._raw.df_model

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
        """F-statistic (classical F-test when `cov_type="classical"`,
        a robust Wald test otherwise)."""
        return self._raw.f_statistic

    @property
    def f_p_value(self) -> float:
        """P-value of the F-statistic."""
        return self._raw.f_p_value

    @property
    def weak_instrument_f_statistics(self) -> dict[str, float]:
        """Weak-instrument diagnostic: partial F-statistic for each
        endogenous variable, keyed by variable name.

        Tests the excluded instruments' joint significance after
        partialling out `x_exog`, always under the classical
        (homoskedastic) formula regardless of `cov_type`. Not the
        same as the plain F-statistic of the corresponding regression
        in `first_stage()`, which includes `x_exog`'s contribution
        too. Empty when `x_endog=[]`. `method="gmm"` is not yet
        implemented and raises `ValidationError` before this result
        is ever returned; see
        `docs/planning/specs/iv-api-design.md` section 6.4.
        """
        return self._raw.weak_instrument_f_statistics

    @property
    def overid_statistic(self) -> float | None:
        """Overidentification test statistic (Sargan/Hansen J).

        Not yet computed (placeholder `None`); see
        `docs/planning/specs/iv-api-design.md` section 6.5.
        """
        return self._raw.overid_statistic

    @property
    def overid_p_value(self) -> float | None:
        """P-value of the overidentification test.

        Not yet computed (placeholder `None`).
        """
        return self._raw.overid_p_value

    @property
    def wu_hausman_statistic(self) -> float | None:
        """Wu-Hausman endogeneity test statistic.

        Not yet computed (placeholder `None`); see
        `docs/planning/specs/iv-api-design.md` section 6.6.
        """
        return self._raw.wu_hausman_statistic

    @property
    def wu_hausman_p_value(self) -> float | None:
        """P-value of the Wu-Hausman test.

        Not yet computed (placeholder `None`).
        """
        return self._raw.wu_hausman_p_value

    def coef_table(self) -> list[dict[str, float | str]]:
        """Row-oriented summary table of the coefficients.

        Shaped to be usable almost as-is in a REST API response (same
        policy as `OlsResults.coef_table()`). Returned as `list[dict]`
        rather than a polars DataFrame, per the project's policy of
        not using DataFrames for the coefficient table itself.

        Returns:
            A list of dictionaries, one per coefficient. Keys are
            `param`, `coef`, `std_err`, `stat` (see `stats` property
            for why this is not `t_stat`/`z_stat`), `p_value`,
            `conf_lower`, `conf_upper`.
        """
        return [
            {
                "param": name,
                "coef": coef,
                "std_err": se,
                "stat": stat,
                "p_value": p,
                "conf_lower": lower,
                "conf_upper": upper,
            }
            for name, coef, se, stat, p, lower, upper in zip(
                self._raw.param_names,
                self._raw.params,
                self._raw.std_errors,
                self._raw.stats,
                self._raw.p_values,
                self._raw.conf_lower,
                self._raw.conf_upper,
            )
        ]

    def first_stage(self) -> dict[str, OlsResults]:
        """Per-endogenous-variable first-stage regression results.

        Each first-stage regression is `x_endog[i] ~ x_exog +
        instruments`, estimated by plain OLS (`docs/planning/specs/
        iv-api-design.md` section 2.2). Returns the existing
        `OlsResults` type rather than a new IV-specific type — the
        first stage is a genuine, valid OLS regression in its own
        right. Its `f_statistic`/`f_p_value` include `x_exog`'s
        contribution and are **not** the weak-instrument partial
        F-statistic (`weak_instrument_f_statistics`).

        Returns:
            A dictionary keyed by endogenous variable name (matching
            `x_endog`), values are the first-stage `OlsResults`.
        """
        return {
            name: OlsResults(raw)
            for name, raw in self._raw.first_stage().items()
        }
