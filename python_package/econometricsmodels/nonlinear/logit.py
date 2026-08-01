"""Python wrapper for Logit (binary logistic regression).

A thin wrapper around `econometricsmodels._lib.fit_logit` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — a list of column names
for `x`, an options object for estimation settings (CLAUDE.md section 2,
`.claude/rules/python-style.md` "設計方針との整合性").

`LogitOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; same policy as `OLSOptions`, see
`docs/planning/specs/ols-api-design.md` section 3).

`summary()` is not implemented (structured-data-only output policy; see
`docs/planning/specs/nonlinear-api-design.md` section 5 and the
`OlsResults`/`WlsResults` precedent).
"""

from __future__ import annotations

import polars as pl

from .. import _lib
from .._lib import LogitOptions

__all__ = ["Logit", "LogitOptions", "LogitResults"]


class Logit:
    """Binary logistic regression estimator.

    Args:
        data: A polars DataFrame containing the dependent and
            independent variables.
        y: Column name of the dependent variable (coded 0/1).
        x: List of column names of the independent variables.
        options: Estimation options. Defaults to `LogitOptions()`
            (classical, with intercept, Newton-Raphson,
            confidence_level=0.95) when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import Logit
        >>> df = pl.DataFrame({"y": [0.0, 1.0], "x1": [1.0, 2.0]})
        >>> result = Logit(df, y="y", x=["x1"]).fit()
        >>> result.params["x1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        options: LogitOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._options = options if options is not None else LogitOptions()

    def fit(self) -> LogitResults:
        """Estimate the Logit model by maximum likelihood.

        Returns:
            The estimation results.

        Raises:
            ValidationError: The input or options are invalid (a
                column is missing, contains missing values or
                NaN/infinity, `y` contains a value other than 0.0/1.0,
                insufficient observations, `confidence_level` out of
                range, an unknown `cov_type`/`method` string, etc.).
                A subclass of `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. non-convergence, a singular
                Hessian/OPG matrix). A subclass of `RuntimeError`.
        """
        raw = _lib.fit_logit(self._data, self._y, self._x, self._options)
        return LogitResults(raw)


class LogitResults:
    """Logit estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing
    (`docs/planning/specs/nonlinear-api-design.md` section 5).

    `marginal_effects()`, `predict()`, and `pred_table()` are provided
    as separate methods rather than fields on this class (they depend
    on a representative point / threshold not fixed at `fit()` time;
    see `docs/planning/specs/nonlinear-api-design.md` section 6).

    Args:
        raw: The estimation result object returned by `_lib.fit_logit`
            (`_lib.LogitResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `Logit.fit()`.
    """

    def __init__(self, raw: _lib.LogitResult) -> None:
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
    def z_stats(self) -> dict[str, float]:
        """Coefficient name to z-statistic.

        Logit uses a z-test (standard normal), not a t-test (see
        `docs/planning/specs/nonlinear-api-design.md` section 5).
        """
        return dict(zip(self._raw.param_names, self._raw.z_stats))

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
    def log_likelihood(self) -> float:
        """Log-likelihood at the fitted parameters."""
        return self._raw.log_likelihood

    @property
    def log_likelihood_null(self) -> float:
        """Log-likelihood of the intercept-only model."""
        return self._raw.log_likelihood_null

    @property
    def lr_statistic(self) -> float:
        """Likelihood-ratio test statistic (overall significance; the
        Logit analogue of OLS's F-statistic)."""
        return self._raw.lr_statistic

    @property
    def lr_p_value(self) -> float:
        """P-value of the likelihood-ratio test (chi-squared distribution)."""
        return self._raw.lr_p_value

    @property
    def pseudo_r_squared(self) -> float:
        """McFadden pseudo R-squared."""
        return self._raw.pseudo_r_squared

    @property
    def aic(self) -> float:
        """Akaike Information Criterion (AIC)."""
        return self._raw.aic

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion (BIC)."""
        return self._raw.bic

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._raw.n_obs

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (`k - 1`)."""
        return self._raw.df_model

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom (`n - k`)."""
        return self._raw.df_resid

    @property
    def converged(self) -> bool:
        """Whether the solver converged within `max_iter` iterations."""
        return self._raw.converged

    @property
    def n_iter(self) -> int:
        """Actual number of solver iterations."""
        return self._raw.n_iter

    @property
    def cov_type(self) -> str:
        """Standard error type actually used (normalized to lowercase)."""
        return self._raw.cov_type

    def coef_table(self) -> list[dict[str, float | str]]:
        """Row-oriented summary table of the coefficients.

        Shaped to be usable almost as-is in a REST API response. Same
        shape as `OlsResults.coef_table()` except `z_stat` in place of
        `t_stat` (Logit uses a z-test rather than a t-test).

        Returns:
            A list of dictionaries, one per coefficient. Keys are
            `param`, `coef`, `std_err`, `z_stat`, `p_value`,
            `conf_lower`, `conf_upper`.
        """
        return [
            {
                "param": name,
                "coef": coef,
                "std_err": se,
                "z_stat": z,
                "p_value": p,
                "conf_lower": lower,
                "conf_upper": upper,
            }
            for name, coef, se, z, p, lower, upper in zip(
                self._raw.param_names,
                self._raw.params,
                self._raw.std_errors,
                self._raw.z_stats,
                self._raw.p_values,
                self._raw.conf_lower,
                self._raw.conf_upper,
            )
        ]

    def predict(self) -> list[dict[str, float]]:
        """Predicted probabilities `p_i = Λ(x_i'β̂)` for the training data.

        Out-of-sample prediction (a `new_data` argument) is not yet
        supported (tracked separately; see
        `docs/planning/specs/logit-implementation-notes.md`).

        Returns:
            Row-oriented predictions, one dict per observation. Each
            dict currently has a single key, `"probability"`.
        """
        return [{"probability": p} for p in self._raw.predict()]

    def pred_table(self, threshold: float = 0.5) -> list[dict[str, float]]:
        """Classification (confusion) table.

        `actual` always uses a fixed 0.5 split; only the predicted
        class depends on `threshold` (matches statsmodels'
        `BinaryResults.pred_table(threshold)`). Out-of-sample data is
        not yet supported (same limitation as `predict()`).

        Args:
            threshold: Probability threshold above which an
                observation is classified as the positive class.
                Defaults to 0.5.

        Returns:
            A list of two dictionaries, one per actual class (0 then
            1). Keys are `actual`, `predicted_0`, `predicted_1`
            (observation counts).
        """
        raw = self._raw.pred_table(threshold)
        return [
            {"actual": actual, "predicted_0": row[0], "predicted_1": row[1]}
            for actual, row in enumerate(raw)
        ]

    def marginal_effects(
        self, at: str = "overall", confidence_level: float = 0.95
    ) -> list[dict[str, float | str]]:
        """Marginal effects (`dy/dx`) with delta-method standard errors.

        Independent of the `confidence_level` used in `fit()` (may
        differ from it; see
        `docs/planning/specs/nonlinear-api-design.md` section 6). The
        constant term (intercept) is excluded from the output.

        Args:
            at: The representative point at which to evaluate the
                marginal effects. One of `"overall"` (default, average
                marginal effects), `"mean"`, or `"median"`.
                Case-insensitive.
            confidence_level: Confidence level for the confidence
                interval, in the range (0, 1). Defaults to 0.95.

        Returns:
            A list of dictionaries, one per explanatory variable
            (excluding the intercept). Keys are `param`, `dydx`,
            `std_err`, `z`, `p_value`, `conf_low`, `conf_high` (see
            `docs/planning/specs/nonlinear-api-design.md` section 6).

        Raises:
            ValidationError: `at` is not one of `"overall"`, `"mean"`,
                `"median"`, or `confidence_level` is out of range. A
                subclass of `ValueError`.
        """
        raw = self._raw.marginal_effects(at, confidence_level)
        return [
            {
                "param": name,
                "dydx": dydx,
                "std_err": se,
                "z": z,
                "p_value": p,
                "conf_low": lower,
                "conf_high": upper,
            }
            for name, dydx, se, z, p, lower, upper in zip(
                raw.param_names,
                raw.dydx,
                raw.std_errors,
                raw.z_stats,
                raw.p_values,
                raw.conf_lower,
                raw.conf_upper,
            )
        ]
