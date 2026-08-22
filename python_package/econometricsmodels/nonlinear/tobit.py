"""Python wrapper for Tobit (censored regression).

A thin wrapper around `econometricsmodels._lib.fit_tobit` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — a list of column names
for `x`, an options object for estimation settings (CLAUDE.md section 2,
`.claude/rules/python-style.md` "設計方針との整合性").

`TobitOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; same policy as `LogitOptions`/`OLSOptions`, see
`docs/spec/ols-spec.md`, "API引数").

`summary()` is not implemented (structured-data-only output policy; see
`docs/planning/specs/nonlinear-api-design.md` section 5 and the
`OlsResults`/`WlsResults`/`LogitResults` precedent).

Unlike Logit/Probit, Tobit does not have `log_likelihood_null`,
`lr_statistic`, `lr_p_value`, or `pseudo_r_squared` (no closed form
exists for the intercept-only model under censoring); `wald_statistic`/
`wald_p_value` provide the overall model significance test instead (see
`docs/planning/specs/nonlinear-api-design.md` section 5). There is no
`pred_table()`; `censoring_fit_check()` takes its place (`y` is
continuous, so a classification table is not meaningful).
"""

from __future__ import annotations

import polars as pl

from .. import _lib
from .._lib import TobitOptions

__all__ = ["Tobit", "TobitOptions", "TobitResults"]


class Tobit:
    """Censored regression (Tobit) estimator.

    Args:
        data: A polars DataFrame containing the dependent and
            independent variables.
        y: Column name of the dependent variable. May be censored at
            `options.lower`/`options.upper`.
        x: List of column names of the independent variables.
        options: Estimation options. Defaults to `TobitOptions()`
            (classical, with intercept, Newton-Raphson,
            confidence_level=0.95, left-censored at 0.0) when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import Tobit
        >>> df = pl.DataFrame({"y": [0.0, 1.0], "x1": [1.0, 2.0]})
        >>> result = Tobit(df, y="y", x=["x1"]).fit()
        >>> result.params["x1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        options: TobitOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._options = options if options is not None else TobitOptions()

    def fit(self) -> TobitResults:
        """Estimate the Tobit model by maximum likelihood.

        Returns:
            The estimation results.

        Raises:
            ValidationError: The input or options are invalid (a
                column is missing, contains missing values or
                NaN/infinity, `y` is outside the censoring bounds,
                the censoring bounds themselves are invalid,
                insufficient observations, no uncensored observations,
                `confidence_level` out of range, an unknown
                `cov_type`/`method` string, an `x` column named
                `"const"`/`"sigma"`, etc.). A subclass of `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. non-convergence, a singular
                Hessian/OPG/design matrix). A subclass of
                `RuntimeError`.
        """
        raw = _lib.fit_tobit(self._data, self._y, self._x, self._options)
        return TobitResults(raw)


class TobitResults:
    """Tobit estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). They include an entry for `"sigma"` (the error
    term's standard deviation) in addition to the regression
    coefficients, since `_lib.TobitResult` reports `sigma` alongside
    `beta` in a unified `(k+1)`-length representation (see
    `engine_pybind/src/nonlinear/tobit.rs`, `TobitResult`). Use
    `coef_table()` for a row-oriented listing
    (`docs/planning/specs/nonlinear-api-design.md` section 5).

    `marginal_effects()`, `predict()`, and `censoring_fit_check()` are
    provided as separate methods rather than fields on this class (they
    depend on a representative point / prediction target not fixed at
    `fit()` time; see `docs/planning/specs/nonlinear-api-design.md`
    section 6).

    Args:
        raw: The estimation result object returned by `_lib.fit_tobit`
            (`_lib.TobitResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `Tobit.fit()`.
    """

    def __init__(self, raw: _lib.TobitResult) -> None:
        self._raw = raw

    @property
    def param_names(self) -> list[str]:
        """List of coefficient names (`"const"` first when
        `include_intercept=True`; `"sigma"` last)."""
        return self._raw.param_names

    @property
    def params(self) -> dict[str, float]:
        """Coefficient name to coefficient value (includes `"sigma"`)."""
        return dict(zip(self._raw.param_names, self._raw.params))

    @property
    def std_errors(self) -> dict[str, float]:
        """Coefficient name to standard error (includes `"sigma"`)."""
        return dict(zip(self._raw.param_names, self._raw.std_errors))

    @property
    def z_stats(self) -> dict[str, float]:
        """Coefficient name to z-statistic (includes `"sigma"`).

        Tobit uses a z-test (standard normal), not a t-test (see
        `docs/planning/specs/nonlinear-api-design.md` section 5).
        """
        return dict(zip(self._raw.param_names, self._raw.z_stats))

    @property
    def p_values(self) -> dict[str, float]:
        """Coefficient name to two-sided p-value (includes `"sigma"`)."""
        return dict(zip(self._raw.param_names, self._raw.p_values))

    @property
    def conf_int(self) -> dict[str, tuple[float, float]]:
        """Coefficient name to confidence interval `(lower, upper)`
        (includes `"sigma"`)."""
        return {
            name: (lower, upper)
            for name, lower, upper in zip(
                self._raw.param_names,
                self._raw.conf_lower,
                self._raw.conf_upper,
            )
        }

    @property
    def sigma(self) -> float:
        """Point estimate of the error term's standard deviation.

        Equal to `params["sigma"]`; provided as a convenience shortcut.
        """
        return self._raw.sigma

    @property
    def log_likelihood(self) -> float:
        """Log-likelihood at the fitted parameters."""
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
    def n_obs(self) -> int:
        """Number of observations."""
        return self._raw.n_obs

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (number of slope coefficients,
        excluding the intercept and `sigma`)."""
        return self._raw.df_model

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom (`n - (k + 1)`, where `k + 1`
        includes `sigma` as an estimated parameter)."""
        return self._raw.df_resid

    @property
    def wald_statistic(self) -> float:
        """Wald test statistic for overall model significance (all
        slope coefficients jointly zero; the Tobit analogue of OLS's
        F-statistic and Logit/Probit's likelihood-ratio statistic).

        `NaN` when `df_model == 0` (no slope coefficients to test).
        """
        return self._raw.wald_statistic

    @property
    def wald_p_value(self) -> float:
        """P-value of the Wald test (chi-squared distribution with
        `df_model` degrees of freedom). `NaN` when `df_model == 0`."""
        return self._raw.wald_p_value

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

    @property
    def lower(self) -> float | None:
        """Lower censoring bound actually used (echoes
        `TobitOptions.lower`). `None` means no censoring from below."""
        return self._raw.lower

    @property
    def upper(self) -> float | None:
        """Upper censoring bound actually used (echoes
        `TobitOptions.upper`). `None` means no censoring from above."""
        return self._raw.upper

    def coef_table(self) -> list[dict[str, float | str]]:
        """Row-oriented summary table of the coefficients.

        Shaped to be usable almost as-is in a REST API response. Same
        shape as `LogitResults.coef_table()`. Includes a row for
        `"sigma"` in addition to the regression coefficients (see
        `params`).

        Returns:
            A list of dictionaries, one per coefficient (including
            `"sigma"`). Keys are `param`, `coef`, `std_err`, `z_stat`,
            `p_value`, `conf_lower`, `conf_upper`.
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

    def predict(
        self, target: str = "expected_observed"
    ) -> list[dict[str, float]]:
        """Predicted values for the training data used in `fit()`.

        Out-of-sample prediction (a `new_data` argument) is not yet
        supported (same limitation as Logit/Probit's `predict()`).

        Args:
            target: Which quantity to predict. One of
                `"expected_latent"` (`E[y*|x] = x'β`),
                `"expected_observed"` (default; `E[y|x]`, the
                censoring-adjusted conditional expectation, directly
                comparable to the observed `y`), or
                `"prob_uncensored"` (`P(uncensored|x)`).
                Case-insensitive.

        Returns:
            Row-oriented predictions, one dict per observation. Each
            dict has a single key, `"predicted"`.

        Raises:
            ValidationError: `target` is not one of the three known
                values. A subclass of `ValueError`.
        """
        return [{"predicted": p} for p in self._raw.predict(target)]

    def marginal_effects(
        self,
        at: str = "overall",
        target: str = "expected_observed",
        confidence_level: float = 0.95,
    ) -> list[dict[str, float | str]]:
        """Marginal effects (`dy/dx`) with delta-method standard errors.

        Unlike Logit/Probit, this is Tobit's own implementation (not
        the shared `dydx_and_jacobian` pattern) because the formula
        differs per `target` (`nonlinear-api-design.md` section 6,
        Issue #211's conclusion). Independent of the `confidence_level`
        used in `fit()` (may differ from it). The constant term
        (intercept) is excluded from the output.

        Args:
            at: The representative point at which to evaluate the
                marginal effects. One of `"overall"` (default, average
                marginal effects), `"mean"`, or `"median"`.
                Case-insensitive.
            target: Which quantity's marginal effect to compute. Same
                three values as `predict()`. Case-insensitive.
            confidence_level: Confidence level for the confidence
                interval, in the range (0, 1). Defaults to 0.95.

        Returns:
            A list of dictionaries, one per explanatory variable
            (excluding the intercept). Keys are `param`, `dydx`,
            `std_err`, `z`, `p_value`, `conf_low`, `conf_high` (see
            `docs/planning/specs/nonlinear-api-design.md` section 6).

        Raises:
            ValidationError: `at` is not one of `"overall"`, `"mean"`,
                `"median"`, `target` is not one of the three known
                values, or `confidence_level` is out of range. A
                subclass of `ValueError`.
        """
        raw = self._raw.marginal_effects(at, target, confidence_level)
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

    def censoring_fit_check(self) -> list[dict[str, float | str]]:
        """Censoring goodness-of-fit check.

        For each direction (`"lower"`/`"uncensored"`/`"upper"`) that
        applies to this model, compares the observed rate (fraction of
        training observations exactly at that boundary) against the
        model-implied average probability. Replaces Logit/Probit's
        `pred_table()`, which is not meaningful for Tobit's continuous
        `y` (`nonlinear-api-design.md` section 6). A direction is
        omitted from the result when the corresponding
        `TobitOptions.lower`/`upper` was `None` (that direction has no
        censoring).

        Returns:
            A list of dictionaries, one per applicable category (at
            most 3: `"lower"`, `"uncensored"`, `"upper"`). Keys are
            `category`, `observed_rate`, `model_implied_rate`.
        """
        raw = self._raw.censoring_fit_check()
        rows: list[dict[str, float | str]] = []
        if raw.lower is not None:
            rows.append(
                {
                    "category": "lower",
                    "observed_rate": raw.lower.observed_rate,
                    "model_implied_rate": raw.lower.model_implied_rate,
                }
            )
        rows.append(
            {
                "category": "uncensored",
                "observed_rate": raw.uncensored.observed_rate,
                "model_implied_rate": raw.uncensored.model_implied_rate,
            }
        )
        if raw.upper is not None:
            rows.append(
                {
                    "category": "upper",
                    "observed_rate": raw.upper.observed_rate,
                    "model_implied_rate": raw.upper.model_implied_rate,
                }
            )
        return rows
