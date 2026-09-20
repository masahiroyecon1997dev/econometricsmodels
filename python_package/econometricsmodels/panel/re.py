"""Python wrapper for RE (random effects panel regression).

A thin wrapper around `econometricsmodels._lib.fit_re` (the Rust
implementation, `engine`/`engine_pybind`). Validation and estimation
logic live entirely on the Rust side; this module only provides the
Python-facing API shape for polars DataFrames — `entity`/`time` as bare
column-name arguments, `x` as a list, an options object for estimation
settings (CLAUDE.md section 2, `.claude/rules/python-style.md`
"設計方針との整合性", `docs/planning/specs/panel-api-design.md` section 1).

`ReOptions` is re-exported as-is from `_lib` (not redefined as a
separate class; same policy as `OLSOptions`/`IvOptions`/`FeOptions`).

Unlike FE, RE has no separate "additional result" method: the Hausman
test comparing RE against the equivalent FE specification is computed
automatically inside `fit()` and exposed directly as properties on
`ReResults` (`docs/planning/specs/panel-api-design.md` section 2.4).

`summary()` is not implemented (structured-data-only output policy; see
the `OlsResults`/`FeResults` precedent).
"""

from __future__ import annotations

import polars as pl

from .. import _lib
from .._lib import ReOptions

__all__ = ["RE", "ReOptions", "ReResults"]


class RE:
    """Random effects (Swamy-Arora GLS) panel regression estimator.

    Unlike FE, RE has an intercept and supports entity-direction random
    effects only (`docs/planning/specs/panel-api-design.md` section
    7.6, two-way RE is out of scope for v1).

    Args:
        data: A polars DataFrame containing the dependent variable,
            independent variables, entity identifier, and (if
            specified) time/cluster columns.
        y: Column name of the dependent variable.
        x: List of column names of the independent variables. Must
            contain at least one column name.
        entity: Column name of the entity (individual/panel unit)
            identifier.
        options: Estimation options. Defaults to `ReOptions()`
            (`cov_type="cluster"` on `entity`, confidence_level=0.95)
            when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import RE
        >>> df = pl.DataFrame(
        ...     {
        ...         "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ...         "x1": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
        ...         "id": ["a", "a", "b", "b", "c", "c"],
        ...     }
        ... )
        >>> result = RE(df, y="y", x=["x1"], entity="id").fit()
        >>> result.params["x1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        entity: str,
        options: ReOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._entity = entity
        self._options = options if options is not None else ReOptions()

    def fit(self) -> ReResults:
        """Estimate the RE model.

        Returns:
            The estimation results.

        Raises:
            ValidationError: The input or options are invalid (`x` is
                empty, a column is missing, contains missing values or
                NaN/infinity, `y`/`x`/`entity`/`time` overlap,
                insufficient observations, `confidence_level` out of
                range, an unknown `cov_type` (or `cov_type="hc0"`,
                unsupported for RE), a singleton entity group (raised
                by the internal one-way FE regression that RE's σ_ε²
                estimation delegates to — unlike a singleton
                encountered by the separate internal FE comparison
                used for the Hausman test, which falls back to `None`
                on `hausman_statistic`/`hausman_p_value`/`hausman_df`
                instead of failing `fit()`; see `ReResults`'s
                docstring), or a `cov_type="hac"` request with `time`
                unset). A subclass of `ValueError`.
            ComputationError: A problem was detected during
                computation (e.g. a singular quasi-demeaned design
                matrix). A subclass of `RuntimeError`.
        """
        raw = _lib.fit_re(
            self._data, self._y, self._x, self._entity, self._options
        )
        return ReResults(raw)


class ReResults:
    """RE estimation results.

    Array-valued properties (`params`, `std_errors`, etc.) are exposed
    as dictionaries keyed by coefficient name (for O(1) lookup of a
    single parameter). Use `coef_table()` for a row-oriented listing.
    `param_names[0]` is always `"const"` (unlike FE, RE has an
    intercept; see `engine::panel::re`'s module docstring on the Rust
    side for why the quasi-demeaned constant column is named this
    way).

    The Hausman test comparing RE against the equivalent FE
    specification (`hausman_statistic`/`hausman_p_value`/`hausman_df`)
    is computed automatically inside `fit()` and exposed directly as
    properties here, rather than as a separate method like FE's
    `fixed_effects()` (`panel-api-design.md` section 2.4). All three
    are `None` when the internal FE comparison used for the Hausman
    test is unavailable — in practice this only happens when
    `ReOptions.time` is set (requesting the two-way FE comparison,
    `panel-api-design.md` section 7.3) and that two-way regression
    itself fails (e.g. an unbalanced panel or a singleton time
    period), or when `Var(β_FE) - Var(β_RE)` is numerically singular;
    RE's own result is still returned normally in that case. This is
    a narrower condition than it might appear: a failure in RE's
    **own** (always one-way) internal FE call — used to estimate σ_ε², not
    for the Hausman comparison — makes `fit()` itself raise instead
    (e.g. a singleton entity, or a regressor with zero variance after
    the one-way within-transformation), since that failure means
    σ_ε² could not be estimated at all; see `RE.fit()`'s docstring.

    Args:
        raw: The estimation result object returned by `_lib.fit_re`
            (`_lib.ReResult`).

    Note:
        Users normally do not construct this directly; it is returned
        by `RE.fit()`.
    """

    def __init__(self, raw: _lib.ReResult) -> None:
        self._raw = raw

    @property
    def param_names(self) -> list[str]:
        """List of coefficient names (`param_names[0]` is always
        `"const"`)."""
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
        """Quasi-demeaned residuals (in observation order)."""
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
        """Residual degrees of freedom (`n - k`, unlike FE's
        `n - n_entities - k`; RE is a GLS transform that does not
        consume entity-dummy degrees of freedom)."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (`k`, including the intercept)."""
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
        coefficients, excluding the intercept. Unlike FE, this is
        **not** a `cov_type`-dependent Wald test: it always compares
        the residual sum of squares against a total sum of squares
        computed from the simple mean of the quasi-demeaned dependent
        variable, following `linearmodels.RandomEffects`'s definition
        (see `engine::panel::re::ReEstimator::fit`'s docstring on the
        Rust side for the derivation). As a consequence it can be
        negative for extremely unbalanced panels, and is `NaN` when
        there are no slope coefficients (`df_model == 1`)."""
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
        """Within R² (based on the entity-demeaned variables, θ=1;
        unrelated to RE's own Swamy-Arora quasi-demeaning)."""
        return self._raw.r_squared_within

    @property
    def r_squared_between(self) -> float:
        """Between R² (based on entity-mean variables)."""
        return self._raw.r_squared_between

    @property
    def r_squared_overall(self) -> float:
        """Overall R² (based on the untransformed variables)."""
        return self._raw.r_squared_overall

    @property
    def hausman_statistic(self) -> float | None:
        """Classical Hausman test statistic comparing RE against the
        equivalent FE specification. Computed with classical standard
        errors regardless of `cov_type`. `None` if the internal FE
        comparison is unavailable (see the class docstring)."""
        return self._raw.hausman_statistic

    @property
    def hausman_p_value(self) -> float | None:
        """P-value of `hausman_statistic` (upper-tail chi-squared
        probability). `None` under the same conditions as
        `hausman_statistic`."""
        return self._raw.hausman_p_value

    @property
    def hausman_df(self) -> int | None:
        """Degrees of freedom of the Hausman test (number of compared
        slope coefficients, i.e. `df_model - 1`). `None` under the same
        conditions as `hausman_statistic`."""
        return self._raw.hausman_df

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
