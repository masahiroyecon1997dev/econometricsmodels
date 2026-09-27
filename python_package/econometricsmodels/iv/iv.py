"""Python wrapper for IV (instrumental variables: 2SLS/GMM).

A thin wrapper around `econometricsmodels._lib.fit_iv` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — lists of column names
for `x_exog`/`x_endog`/`instruments`, an options object for estimation
settings (CLAUDE.md section 2, `.claude/rules/python-style.md`
"設計方針との整合性").

`IVOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; same policy as `OLSOptions`/`LogitOptions`, see
`docs/spec/ols-spec.md`, "API引数"). `IVOptions.estimator` selects
`"2sls"` (default) or `"gmm"` — a single `IV`/`IVResults` pair serves
both methods (`docs/spec/iv-spec.md` section 1.2).

`summary()` is not implemented (structured-data-only output policy; see
the `OLSResults`/`LogitResults` precedent).
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import IVOptions
from ..linear.ols import OLSResults

__all__ = ["IV", "IVOptions", "IVResults"]


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
            `docs/spec/iv-spec.md` section 1.1).
        options: Estimation options. Defaults to `IVOptions()`
            (`estimator="2sls"`, classical, with intercept,
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
        options: IVOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x_exog = x_exog
        self._x_endog = x_endog
        self._instruments = instruments
        self._options = options if options is not None else IVOptions()

    def fit(self) -> IVResults:
        """Estimate the IV model.

        Returns:
            The estimation results.

        Raises:
            ValidationError: The input or options are invalid (a
                column is missing, contains missing values or
                NaN/infinity, `y`/`x_exog`/`x_endog`/`instruments`
                overlap, `x_endog` or `instruments` is empty,
                insufficient observations, `confidence_level` out of
                range, an unknown `cov_type` or (`estimator="gmm"` only)
                `gmm_type`/`gmm_weight_type` string, an option the
                chosen `estimator`/`gmm_type`/`cov_type` does not
                use (e.g. `cluster` without `cov_type="cluster"`,
                `gmm_max_iter` with a `gmm_type` other than
                `"iterated"`), or too few instruments for
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
        return IVResults(raw)


class IVResults:
    """IV estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing.

    `first_stage()` (per-endogenous-variable first-stage regression
    results) is provided as a separate method rather than a field on
    this class (`docs/spec/iv-spec.md` section 2).

    Args:
        raw: The estimation result object returned by `_lib.fit_iv`
            (`_lib.IVResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `IV.fit()`.
    """

    def __init__(self, raw: _lib.IVResult) -> None:
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
    def test_stats(self) -> dict[str, float]:
        """Coefficient name to test statistic.

        t-statistic for `estimator="2sls"`, z-statistic for
        `estimator="gmm"`; `stat_dist` tells which.
        """
        return dict(zip(self._raw.param_names, self._raw.test_stats))

    @property
    def stat_dist(self) -> Literal["t", "normal"]:
        """Distribution of `test_stats`: `"t"` for `estimator="2sls"`,
        `"normal"` for `estimator="gmm"`."""
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
    def converged(self) -> bool:
        """Whether GMM iteration converged (`estimator="gmm"` only).

        Only meaningful for `gmm_type="iterated"`; always `True` for
        `"one_step"`/`"two_step"` (which never check convergence) and
        for `estimator="2sls"` (2SLS is a closed-form, non-iterative
        estimator).
        """
        return self._raw.converged

    @property
    def n_iter(self) -> int:
        """Number of GMM estimations actually run, counting the initial
        estimate (`estimator="gmm"` only): 1 for `"one_step"`, 2 for
        `"two_step"`, at most `gmm_max_iter` for `"iterated"`. Always
        `1` for `estimator="2sls"`."""
        return self._raw.n_iter

    @property
    def cov_type(self) -> str:
        """Standard error type actually used (normalized to lowercase)."""
        return self._raw.cov_type

    @property
    def estimator(self) -> str:
        """Estimator actually used (normalized to lowercase):
        `"2sls"` or `"gmm"`."""
        return self._raw.estimator

    @property
    def gmm_weight_type(self) -> str | None:
        """Weight matrix actually used for GMM point estimation
        (normalized to lowercase). Only meaningful for `estimator="gmm"`;
        always `None` for `estimator="2sls"`, which has no such concept,
        and for `gmm_type="one_step"`, which does not use a weight
        type."""
        return self._raw.gmm_weight_type

    @property
    def gmm_type(self) -> str | None:
        """GMM estimation type actually used (normalized to
        lowercase): `"one_step"`, `"two_step"` or `"iterated"`. Always
        `None` for `estimator="2sls"`."""
        return self._raw.gmm_type

    @property
    def r_squared(self) -> float:
        """Coefficient of determination (R²)."""
        return self._raw.r_squared

    @property
    def adj_r_squared(self) -> float:
        """Degrees-of-freedom-adjusted R²."""
        return self._raw.adj_r_squared

    @property
    def wald_statistic(self) -> float:
        """Wald test statistic for all slope coefficients being zero.

        For `estimator="2sls"` this is the F-type statistic (the Wald
        statistic divided by the number of slope coefficients; the
        counterpart of `OLSResults.f_statistic`, a classical F-test when
        `cov_type="classical"`, a robust Wald test otherwise). For
        `estimator="gmm"` it is the undivided Wald statistic, which
        follows a chi-squared distribution. `wald_dist` tells which.
        """
        return self._raw.wald_statistic

    @property
    def wald_p_value(self) -> float:
        """P-value of `wald_statistic` (F or chi-squared, see
        `wald_dist`)."""
        return self._raw.wald_p_value

    @property
    def wald_dist(self) -> Literal["f", "chi2"]:
        """Distribution of `wald_statistic`: `"f"` (F distribution) for
        `estimator="2sls"`, `"chi2"` (chi-squared) for `estimator="gmm"`."""
        return self._raw.wald_dist

    @property
    def wald_df_num(self) -> int | None:
        """Numerator degrees of freedom of `wald_statistic` (the number of
        slope coefficients; the only degrees of freedom for `"chi2"`).
        `None` when there are no slope coefficients."""
        return self._raw.wald_df_num

    @property
    def wald_df_denom(self) -> int | None:
        """Denominator degrees of freedom of `wald_statistic` for `"f"`
        (`df_resid`, or `G - 1` with cluster-robust inference). `None` for
        `"chi2"` and when the statistic is NaN."""
        return self._raw.wald_df_denom

    @property
    def weak_instrument_f_statistics(self) -> dict[str, float]:
        """Weak-instrument diagnostic: partial F-statistic for each
        endogenous variable, keyed by variable name.

        Tests the excluded instruments' joint significance after
        partialling out `x_exog`, always under the classical
        (homoskedastic) formula regardless of `cov_type`. Not the
        same as the plain F-statistic of the corresponding regression
        in `first_stage()`, which includes `x_exog`'s contribution
        too. Computed the same way for both `estimator="2sls"` and
        `estimator="gmm"`; see `docs/spec/iv-spec.md` section 3.4.
        """
        return self._raw.weak_instrument_f_statistics

    @property
    def weak_instrument_f_df_num(self) -> int | None:
        """Numerator degrees of freedom of the weak-instrument F
        statistics (the number of excluded instruments; the same for
        every endogenous variable)."""
        return self._raw.weak_instrument_f_df_num

    @property
    def weak_instrument_f_df_denom(self) -> int | None:
        """Denominator degrees of freedom of the weak-instrument F
        statistics (residual degrees of freedom of the first-stage
        regressions; the same for every endogenous variable)."""
        return self._raw.weak_instrument_f_df_denom

    @property
    def overid_statistic(self) -> float | None:
        """Overidentification test statistic: Sargan (`estimator="2sls"`)
        or Hansen J (`estimator="gmm"`).

        `None` when just-identified (`len(instruments) ==
        len(x_endog)`, degrees of freedom 0); see
        `docs/spec/iv-spec.md` section 3.5.
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

    @property
    def wu_hausman_statistic(self) -> float | None:
        """Wu-Hausman endogeneity test statistic (regression-based).

        Adds the first-stage residuals to the structural equation and
        tests their joint significance (`linearmodels`'
        `wooldridge_regression` formulation), jointly over all
        endogenous variables. Unlike
        `weak_instrument_f_statistics`, this is always computed under
        the same `cov_type` passed to `fit()`.

        `None` when the augmented regression cannot be estimated
        (e.g. the first-stage residual has zero variance, such as
        when an instrument perfectly predicts its endogenous
        variable, or there are too few observations for the extra
        residual columns) — this does not affect the validity of
        the other results. **Always `None` for `estimator="gmm"`**
        (not implemented for GMM). See
        `docs/spec/iv-spec.md` section 3.6.
        """
        return self._raw.wu_hausman_statistic

    @property
    def wu_hausman_df_num(self) -> int | None:
        """Numerator degrees of freedom of the Wu-Hausman F test (the
        number of endogenous variables). `None` under the same conditions
        as `wu_hausman_statistic`."""
        return self._raw.wu_hausman_df_num

    @property
    def wu_hausman_df_denom(self) -> int | None:
        """Denominator degrees of freedom of the Wu-Hausman F test.
        `None` under the same conditions as `wu_hausman_statistic`."""
        return self._raw.wu_hausman_df_denom

    @property
    def wu_hausman_p_value(self) -> float | None:
        """P-value of the Wu-Hausman test.

        `None` under the same conditions as `wu_hausman_statistic`.
        """
        return self._raw.wu_hausman_p_value

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
                "test_stat": test_stat,
                "p_value": p,
                "conf_lower": lower,
                "conf_upper": upper,
            }
            for name, coef, se, test_stat, p, lower, upper in zip(
                self._raw.param_names,
                self._raw.params,
                self._raw.std_errors,
                self._raw.test_stats,
                self._raw.p_values,
                self._raw.conf_lower,
                self._raw.conf_upper,
            )
        ]

    def first_stage(self) -> dict[str, OLSResults]:
        """Per-endogenous-variable first-stage regression results.

        Each first-stage regression is `x_endog[i] ~ x_exog +
        instruments`, estimated by plain OLS (`docs/spec/iv-spec.md`
        section 2). Returns the existing
        `OLSResults` type rather than a new IV-specific type — the
        first stage is a genuine, valid OLS regression in its own
        right. Its `f_statistic`/`f_p_value` include `x_exog`'s
        contribution and are **not** the weak-instrument partial
        F-statistic (`weak_instrument_f_statistics`).

        Returns:
            A dictionary keyed by endogenous variable name (matching
            `x_endog`), values are the first-stage `OLSResults`.
        """
        return {
            name: OLSResults(raw)
            for name, raw in self._raw.first_stage().items()
        }
