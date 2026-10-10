"""Python wrapper for OLS (Ordinary Least Squares).

A thin wrapper around `econometricsmodels._lib.fit_ols` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — a list of column names
for `x`, an options object for estimation settings (CLAUDE.md section 2,
`.claude/rules/python-style.md` "設計方針との整合性").

`OLSOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; see `docs/spec/ols-spec.md`, "API引数").
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import OLSOptions
from ..diagnostics import BreuschGodfreyTestResult, WhiteTestResult

__all__ = ["OLS", "OLSOptions", "OLSResults"]


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

    def fit(self) -> OLSResults:
        """Estimate the OLS model.

        Returns:
            The estimation results.

        Raises:
            TypeError: An argument has the wrong type (for example
                `x` is a string instead of a list of column names). A
                builtin exception, not a `ValidationError`.
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
        return OLSResults(raw)


class OLSResults:
    """OLS estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing
    (see `docs/spec/ols-spec.md`, "結果構造体").

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
        `df_resid` (e.g. cluster-robust inference uses `G - 1`), so use
        this to recompute p-values from `test_stats`."""
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
        """Residuals (in observation order, `y - Xβ̂`)."""
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
    def hac_lags_used(self) -> int | None:
        """Number of HAC (Newey-West) lags actually used: the explicit
        `hac_lags` if given, otherwise the value chosen automatically,
        `floor(4 * (n / 100) ** (2 / 9))`. `None` unless
        `cov_type="hac"`."""
        return self._raw.hac_lags_used

    @property
    def r_squared(self) -> float:
        """Coefficient of determination (R²)."""
        return self._raw.r_squared

    @property
    def adj_r_squared(self) -> float:
        """Degrees-of-freedom-adjusted R²."""
        return self._raw.adj_r_squared

    @property
    def f_statistic(self) -> float:
        """F-statistic."""
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
        is NaN). May differ from `df_resid`, so use this to recompute the
        p-value from `f_statistic`."""
        return self._raw.f_df_denom

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom (`n - k`)."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (number of slope coefficients,
        excluding the intercept)."""
        return self._raw.df_model

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

    def predict(self, new_data: pl.DataFrame | None = None) -> list[float]:
        """Predicted values.

        Unified into a single method rather than a separate
        `fitted_values` property, to match the naming used by Logit's
        `predict()` (`docs/spec/ols-spec.md`, "predict()").

        Args:
            new_data: New data to predict on. Must contain columns with
                the same names as the `x` columns passed at fit time
                (matched by name; column order does not matter). If
                `include_intercept=True` was used at fit time, the
                constant column is added automatically and must not be
                included here. If `None` (default), returns the fitted
                values for the training data used in `fit()`.

        Returns:
            Predicted values, one per observation, in row order (like
            `residuals`).

        Raises:
            ValidationError: `new_data` is missing a required `x`
                column, or a column contains missing/NaN/infinite
                values.
        """
        return self._raw.predict(new_data)

    def augment(self, new_data: pl.DataFrame | None = None) -> pl.DataFrame:
        """Source data with the predicted values appended as a column.

        Same `new_data` semantics as `predict()`, but returns a polars
        DataFrame (the training data, or `new_data` when given, plus a
        new `"predicted"` column) instead of a plain list. This
        is the one exception to the project's policy of not returning
        DataFrames (`docs/spec/ols-spec.md`, "augment()"): it exists
        specifically to attach predictions back to their source rows.

        Args:
            new_data: Same as `predict()`. If `None` (default), returns
                the training data used in `fit()` with the predicted
                values appended.

        Returns:
            A polars DataFrame: the source data's columns plus
            `"predicted"`, in the same row order as the source.

        Raises:
            ValidationError: Same as `predict()`, or the source data
                already has a column named `"predicted"` (which would
                otherwise be silently overwritten). Also raised for
                `new_data=None` when this result has no retained
                training data (currently only possible for the
                `OLSResults` returned by `IVResult.first_stage()`,
                which has no single source DataFrame to attach a
                column to; calling `augment(new_data)` with an
                explicit `new_data` still works normally in that case).
        """
        return self._raw.augment(new_data)

    def white_test(
        self, statistic: Literal["lm", "f"] = "lm"
    ) -> WhiteTestResult:
        """White test for heteroskedasticity.

        Regresses the squared residuals on the independent variables,
        their squares and their pairwise products, and tests that all
        slopes are zero. The auxiliary regression always includes a
        constant, even when the model was fitted with
        `include_intercept=False`. Terms that are constant or numerically
        identical to an earlier term (for example the square of a 0/1
        dummy) are dropped, and the degrees of freedom count the terms
        that remain; `WhiteTestResult.aux_terms` and `dropped_terms`
        report them. The test assumes homoskedastic errors under the
        null and does not depend on `cov_type`.

        This is a post-estimation diagnostic: it is never computed by
        `fit()`. It re-reads the independent variables from the data
        passed to `fit()`.

        Args:
            statistic: `"lm"` (default) for the LM version
                `n * R²` (chi-squared distribution), or `"f"` for the
                F version of the same auxiliary regression.

        Returns:
            The test result, a `WhiteTestResult`: `distribution` is
            `"chi2"` and `df_denom` is `None` for the LM version, `"f"`
            and an integer for the F version. It also lists the terms of
            the auxiliary regression (`aux_terms`, always starting with
            the constant) and the dropped ones (`dropped_terms`).

        Raises:
            ValidationError: `statistic` is not `"lm"` or `"f"`; there are
                too few observations for the auxiliary regression; or
                this result has no retained training data (currently only
                the `OLSResults` returned by `IVResult.first_stage()`).
            ComputationError: The auxiliary design matrix is still
                singular after dropping terms, or the auxiliary
                regression's R² is undefined (for example the squared
                residuals are constant).
        """
        raw = self._raw.white_test(statistic)
        return WhiteTestResult(
            statistic=raw.statistic,
            p_value=raw.p_value,
            df=raw.df,
            df_denom=raw.df_denom,
            distribution=raw.distribution,
            aux_terms=raw.aux_terms,
            dropped_terms=raw.dropped_terms,
        )

    def breusch_godfrey_test(
        self,
        time: str,
        nlags: int,
        statistic: Literal["lm", "f"] = "lm",
    ) -> BreuschGodfreyTestResult:
        """Breusch-Godfrey test for serial correlation of the errors.

        Regresses the residuals on the model's independent variables and
        on their own lags 1 to `nlags`, and tests that the lag
        coefficients are all zero. The lags are taken in the order of the
        `time` column, and the lags before the first observation are
        filled with zero (the convention of R's `lmtest::bgtest`,
        statsmodels and Stata). The auxiliary regression uses the model's
        regressors as they are: no constant is added to a model fitted
        with `include_intercept=False` (the definition of R and Greene;
        statsmodels adds one in that case, so its value differs). The test
        does not depend on `cov_type`.

        This is a post-estimation diagnostic: it is never computed by
        `fit()`. It re-reads the independent variables and the `time`
        column from the data passed to `fit()`.

        The row order is never assumed to be the time order, so there is
        no default for `time`. Only the order of the `time` values is
        used: gaps between periods are not checked, and the lags follow
        the sorted order. A cross-sectional data set has no meaningful
        order, and the test is not meaningful for it.

        Args:
            time: Name of a column of the data passed to `fit()` that
                gives the time order (integers, floats, `Decimal`,
                `Date` or `Datetime`; the values must be distinct, as for
                `OLSOptions.hac_time`).
            nlags: Number of lags of the residuals, at least 1. There is
                no default: pick the order that suits your data.
            statistic: `"lm"` (default) for the LM version `n * R^2`
                (chi-squared distribution), or `"f"` for the F version
                (case-insensitive).

        Returns:
            The test result, a `BreuschGodfreyTestResult`: `distribution`
            is `"chi2"` and `df_denom` is `None` for the LM version, `"f"`
            and an integer for the F version.

        Raises:
            TypeError: `time` or `statistic` is not a `str`, or `nlags`
                is not an `int` (a `bool` or `float` is rejected). A
                builtin exception, not a `ValidationError`.
            ValidationError: `statistic` is not `"lm"` or `"f"`;
                `nlags < 1`; there are too few observations for the
                auxiliary regression (`n <= k + nlags`); `time` does not
                exist, has an unsupported dtype, or contains missing,
                non-finite or duplicate values; or this result has no
                retained training data (currently only the `OLSResults`
                returned by `IVResult.first_stage()`).
            ComputationError: The auxiliary design matrix is singular, or
                the residuals are all zero or fitted exactly by the
                auxiliary regression.
        """
        raw = self._raw.breusch_godfrey_test(time, nlags, statistic)
        return BreuschGodfreyTestResult(
            statistic=raw.statistic,
            p_value=raw.p_value,
            df=raw.df,
            df_denom=raw.df_denom,
            distribution=raw.distribution,
            nlags=raw.nlags,
        )
