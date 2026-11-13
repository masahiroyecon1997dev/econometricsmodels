"""Python wrapper for FEIV (fixed effects with instrumental variables).

A thin wrapper around `econometricsmodels._lib.fit_fe_iv` (the Rust
implementation). Validation and estimation happen on the Rust side; this
module only provides the Python-facing API shape for polars DataFrames:
lists of column names for `x_exog`, `x_endog`, `instruments` and
`effects`, and an options object for the estimation settings.

`FEIVOptions` is re-exported as-is from `_lib`. The first-stage
regressions are returned as `FEResults`, and the weak-instrument
diagnostics use the same result types as `IVResults`. There is no
`summary()`: results are returned as structured data.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import FEIVOptions
from .._results_utils import (
    by_param,
    conf_int_by_param,
    cov_params_by_param,
    cragg_donald_result,
    effective_f_result,
    kleibergen_paap_result,
    params_table_rows,
    results_repr,
)
from ..diagnostics import (
    CraggDonaldResult,
    EffectiveFResult,
    KleibergenPaapResult,
)
from .fe import FEResults

__all__ = ["FEIV", "FEIVOptions", "FEIVResults"]


class FEIV:
    """Fixed effects instrumental variables (FE-2SLS) estimator.

    Removes the fixed effects by the within transformation and then
    applies two-stage least squares to the transformed data. Selecting
    the effects works as in `FE`: one column in `effects` gives one-way
    effects, two columns give two-way effects, and three to five columns
    give higher-dimensional effects.

    Args:
        data: A polars DataFrame containing the dependent variable, the
            exogenous and endogenous independent variables, the
            instruments, the `effects` columns, and (if specified) the
            `cluster` and `dk_time` columns of the options.
        y: Column name of the dependent variable.
        x_exog: List of column names of the exogenous independent
            variables. May be empty.
        x_endog: List of column names of the endogenous independent
            variables. Must contain at least one column name.
        instruments: List of column names of the excluded instruments.
            Must contain at least as many columns as `x_endog`. No
            column may appear in more than one of `y`, `x_exog`,
            `x_endog`, `instruments` and `effects`.
        effects: List of one to five column names that define the fixed
            effects, with the same meaning and order as in `FE`: the
            first is the entity identifier (and the default cluster
            key), the optional second is the time period, and the
            optional third to fifth are further grouping columns.
        options: Estimation options. Defaults to `FEIVOptions()`
            (`cov_type="cluster"` on the first column of `effects`,
            `confidence_level=0.95`) when omitted.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import FEIV
        >>> df = pl.DataFrame(
        ...     {
        ...         "y": [1.0, 2.1, 2.9, 1.5, 2.2, 3.8, 0.9, 2.4, 3.3]
        ...         + [1.8, 2.6, 4.1],
        ...         "endog": [1.0, 2.2, 3.1, 1.2, 2.0, 3.9, 0.8, 2.1, 3.5]
        ...         + [1.9, 2.5, 3.8],
        ...         "z": [0.9, 2.5, 2.7, 1.6, 1.8, 4.0, 1.1, 2.3, 3.0]
        ...         + [1.7, 2.9, 4.2],
        ...         "id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
        ...     }
        ... )
        >>> result = FEIV(
        ...     df,
        ...     y="y",
        ...     x_exog=[],
        ...     x_endog=["endog"],
        ...     instruments=["z"],
        ...     effects=["id"],
        ... ).fit()
        >>> list(result.params)
        ['endog']
        >>> result.n_groups
        {'id': 4}
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x_exog: list[str],
        x_endog: list[str],
        instruments: list[str],
        effects: list[str],
        *,
        options: FEIVOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x_exog = x_exog
        self._x_endog = x_endog
        self._instruments = instruments
        self._effects = effects
        self._options = options if options is not None else FEIVOptions()

    def fit(self) -> FEIVResults:
        """Estimate the FEIV model.

        Returns:
            The estimation results.

        Raises:
            TypeError: An argument has the wrong type (for example
                `x_endog` or `effects` is a string instead of a list of
                column names). A builtin exception, not a
                `ValidationError`.
            ValidationError: The input or options are invalid. A
                subclass of `ValueError`. For example:

                - A column is missing, or a column contains missing
                  values, NaN or infinity.
                - `x_endog` or `instruments` is empty, or there are
                  fewer `instruments` than `x_endog` columns (the model
                  is not identified).
                - `effects` is empty or has more than five columns, or
                  `y`, `x_exog`, `x_endog`, `instruments` and `effects`
                  overlap, or `dk_time` equals `y` or the first column
                  of `effects`.
                - There are insufficient observations, or
                  `confidence_level` is out of range.
                - `cov_type` is unknown or `"hc0"` (not supported for
                  FEIV).
                - `cluster`, `dk_time` or `dk_bandwidth` is given with
                  a `cov_type` that does not use it, or
                  `cov_type="dk"` is requested without `dk_time`.
                - A group of a column in `effects` has a single
                  observation.

            ComputationError: A problem was detected during
                computation (e.g. a singular within-transformed design
                matrix, a perfect fit of the structural equation or of a
                first stage, or alternating projections that did not
                converge within `max_iter`). A subclass of
                `RuntimeError`.
        """
        raw = _lib.fit_fe_iv(
            self._data,
            self._y,
            self._x_exog,
            self._x_endog,
            self._instruments,
            self._effects,
            self._options,
        )
        return FEIVResults(raw, self._x_endog)


class FEIVResults:
    """FEIV estimation results.

    Coefficient-level properties (`params`, `std_errors`, etc.) are
    dictionaries keyed by coefficient name, so one coefficient is
    `result.params["endog"]`. Use `params_table()` for a row-oriented
    listing (a list of dicts), for example for a JSON response.

    Printing a result shows a header line (the dependent variable,
    `n_obs` and `cov_type`) and a short table of the coefficients and
    their standard errors. Its layout may change between versions, so
    read the fields listed below or use `params_table()` rather than
    parsing it.

    Fields by purpose:

    - Coefficients (no intercept): `params`, `std_errors`,
      `test_stats`, `p_values`, `conf_int`; the names in order:
      `param_names`.
    - Covariance of the coefficients: `cov_params`.
    - Which distribution the tests use: `stat_dist`, `stat_df`.
    - Overall Wald test: `wald_statistic`, `wald_p_value`,
      `wald_dist`, `wald_df_num`, `wald_df_denom`.
    - Weak instruments: `weak_instrument_f_statistics`,
      `weak_instrument_f_df_num`, `weak_instrument_f_df_denom`.
    - Overidentification (Sargan): `overid_statistic`,
      `overid_p_value`, `overid_df`.
    - Endogeneity: `wu_hausman_statistic`, `wu_hausman_p_value`,
      `wu_hausman_df_num`, `wu_hausman_df_denom`.
    - Fit: `r_squared_within`, `r_squared_between`,
      `r_squared_overall`.
    - Panel size: `n_obs`, `n_groups`.
    - Model size: `df_model`, `df_resid`, `dep_var_name`.
    - Settings echoed back: `cov_type`, `dk_bandwidth_used`.
    - Residuals: `residuals` (structural residuals of the
      within-transformed data).
    - Methods: `first_stage()` (one `FEResults` per endogenous
      variable), `fixed_effects()` (the estimated effect of every group,
      for each effects column of the model), `params_table()`.
    - Weak-instrument diagnostics (methods, computed when called):
      `cragg_donald()`, `kleibergen_paap()`, `effective_f()`.

    Note:
        Users normally do not construct this directly; it is returned
        by `FEIV.fit()`.
    """

    def __init__(self, raw: _lib.FEIVResult, x_endog: list[str]) -> None:
        self._raw = raw
        self._x_endog = list(x_endog)

    def __repr__(self) -> str:
        return results_repr(type(self).__name__, self._raw)

    @property
    def param_names(self) -> list[str]:
        """List of coefficient names (no intercept: within-transformed
        variables have no constant term)."""
        return self._raw.param_names

    @property
    def params(self) -> dict[str, float]:
        """Coefficient name to coefficient value."""
        return by_param(self._raw.param_names, self._raw.params)

    @property
    def std_errors(self) -> dict[str, float]:
        """Coefficient name to standard error."""
        return by_param(self._raw.param_names, self._raw.std_errors)

    @property
    def cov_params(self) -> dict[str, dict[str, float]]:
        """Covariance matrix of the coefficients, as a dict of dicts by name.

        Computed with the `cov_type` of the options, so `std_errors[a]` is
        the square root of `cov_params[a][a]`. The covariance of `a` and `b`
        is `cov_params["a"]["b"]`, and the keys are in `param_names` order.

        The matrix is symmetric up to rounding error (a relative difference
        of 1e-16 to 1e-15 between `cov_params["a"]["b"]` and
        `cov_params["b"]["a"]`), so compare entries with a tolerance rather
        than with `==`.
        """
        return cov_params_by_param(self._raw.param_names, self._raw.cov_params)

    @property
    def test_stats(self) -> dict[str, float]:
        """Coefficient name to test statistic (t-statistic; see
        `stat_dist`)."""
        return by_param(self._raw.param_names, self._raw.test_stats)

    @property
    def stat_dist(self) -> Literal["t"]:
        """Distribution of `test_stats`. Always `"t"` for the two-stage
        least squares estimator."""
        return self._raw.stat_dist

    @property
    def stat_df(self) -> int | None:
        """Degrees of freedom of the t distribution behind `test_stats`.
        May differ from `df_resid` (cluster-robust inference uses
        `G - 1`, where `G` is the number of clusters, and
        Driscoll-Kraay the number of `dk_time` periods minus 1), so use
        this to recompute p-values from `test_stats`."""
        return self._raw.stat_df

    @property
    def p_values(self) -> dict[str, float]:
        """Coefficient name to two-sided p-value."""
        return by_param(self._raw.param_names, self._raw.p_values)

    @property
    def conf_int(self) -> dict[str, tuple[float, float]]:
        """Coefficient name to confidence interval `(lower, upper)`."""
        return conf_int_by_param(self._raw)

    @property
    def residuals(self) -> list[float]:
        """Structural residuals of the within-transformed data (in
        observation order, `y - Xβ̂`, using the actual endogenous
        variables rather than their first-stage fitted values)."""
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
        """Residual degrees of freedom: `n - df_model`, where `df_model`
        is the model degrees of freedom below."""
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom: the `k` slope coefficients plus the
        absorbed fixed effects (`k + sum(G) - (number of columns - 1)`,
        where the `G` are the group counts in `n_groups`). It is used
        for the degrees of freedom of the inference (`df_resid = n -
        df_model`); the number of slope coefficients is
        `len(param_names)`."""
        return self._raw.df_model

    @property
    def n_groups(self) -> dict[str, int]:
        """Number of groups (unique values) of each column in
        `effects`, keyed by column name in the order given: one entry
        for a one-way model, two for a two-way model, and so on. For
        example `{"id": 3, "t": 2}`."""
        return self._raw.n_groups

    @property
    def cov_type(self) -> str:
        """Standard error type actually used (normalized to lowercase)."""
        return self._raw.cov_type

    @property
    def dk_bandwidth_used(self) -> int | None:
        """Driscoll-Kraay bandwidth actually used: the explicit
        `dk_bandwidth` if given, otherwise the value chosen automatically,
        `floor(4 * (t / 100) ** (2 / 9))` where `t` is the number of
        unique time periods in `FEIVOptions.dk_time`. `None` unless
        `cov_type="dk"`."""
        return self._raw.dk_bandwidth_used

    @property
    def wald_statistic(self) -> float:
        """F-type statistic for all slope coefficients being zero (the
        Wald statistic divided by the number of slope coefficients; a
        classical F-test when `cov_type="classical"`, a robust Wald test
        otherwise)."""
        return self._raw.wald_statistic

    @property
    def wald_p_value(self) -> float:
        """P-value of `wald_statistic`."""
        return self._raw.wald_p_value

    @property
    def wald_dist(self) -> Literal["f"]:
        """Distribution of `wald_statistic`. Always `"f"` (F
        distribution) for the two-stage least squares estimator."""
        return self._raw.wald_dist

    @property
    def wald_df_num(self) -> int | None:
        """Numerator degrees of freedom of `wald_statistic` (the number of
        slope coefficients). Never `None`, because `x_endog` always has
        at least one variable."""
        return self._raw.wald_df_num

    @property
    def wald_df_denom(self) -> int | None:
        """Denominator degrees of freedom of `wald_statistic`. Follows
        `cov_type` like the t-tests, so it equals `stat_df`: the number
        of clusters minus 1 for `cov_type="cluster"`, the number of
        `dk_time` periods minus 1 for `"dk"`, and `df_resid` otherwise.
        Never `None`."""
        return self._raw.wald_df_denom

    @property
    def r_squared_within(self) -> float:
        """Within R² (based on the within-transformed variables, from the
        structural residuals). Can be negative."""
        return self._raw.r_squared_within

    @property
    def r_squared_between(self) -> float:
        """Between R² (based on entity-mean variables, from the structural
        residuals). Can be negative."""
        return self._raw.r_squared_between

    @property
    def r_squared_overall(self) -> float:
        """Overall R² (based on the untransformed variables, from the
        structural residuals). Can be negative."""
        return self._raw.r_squared_overall

    @property
    def weak_instrument_f_statistics(self) -> dict[str, float]:
        """Weak-instrument diagnostic: partial F-statistic for each
        endogenous variable, keyed by variable name in the order of
        `x_endog`.

        Tests the excluded instruments' joint significance after
        partialling out `x_exog` and the fixed effects, always under the
        classical (homoskedastic) formula regardless of `cov_type`.
        `first_stage()` has no overall F-statistic to compare it with.
        """
        stats = self._raw.weak_instrument_f_statistics
        return {name: stats[name] for name in self._x_endog}

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
        regressions after the absorbed fixed effects; the same for every
        endogenous variable)."""
        return self._raw.weak_instrument_f_df_denom

    @property
    def overid_statistic(self) -> float | None:
        """Sargan overidentification test statistic, computed from the
        within-transformed data.

        Always under the classical (homoskedastic) formula, with the
        error variance `e'e / (n - n_absorbed)` (`n_absorbed` is the
        number of absorbed fixed-effect parameters), regardless of
        `cov_type`.

        `None` when just-identified (`len(instruments) ==
        len(x_endog)`, degrees of freedom 0).
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
        tests their joint significance in a fixed-effects regression,
        jointly over all endogenous variables. Unlike
        `weak_instrument_f_statistics`, this is always computed under
        the `cov_type` in `options`.

        `None` when the augmented regression cannot be estimated
        (e.g. the augmented design is singular, `y` is exactly
        explained by the regressors and the first-stage residuals, or
        there are too few observations or clusters for the extra
        residual columns) — this does not affect the validity of the
        other results.
        """
        return self._raw.wu_hausman_statistic

    @property
    def wu_hausman_p_value(self) -> float | None:
        """P-value of the Wu-Hausman test.

        `None` under the same conditions as `wu_hausman_statistic`.
        """
        return self._raw.wu_hausman_p_value

    @property
    def wu_hausman_df_num(self) -> int | None:
        """Numerator degrees of freedom of the Wu-Hausman F test (the
        number of endogenous variables). `None` under the same conditions
        as `wu_hausman_statistic`."""
        return self._raw.wu_hausman_df_num

    @property
    def wu_hausman_df_denom(self) -> int | None:
        """Denominator degrees of freedom of the Wu-Hausman F test. Follows
        `cov_type` like `stat_df`: the number of clusters minus 1 for
        `cov_type="cluster"`, the number of `dk_time` periods minus 1 for
        `"dk"`, and the residual degrees of freedom of the augmented
        regression otherwise. `None` under the same conditions as
        `wu_hausman_statistic`."""
        return self._raw.wu_hausman_df_denom

    def params_table(self) -> list[dict[str, float | str]]:
        """Row-oriented summary table of the coefficients.

        Shaped to be usable almost as-is in a REST API response (same
        policy as `OLSResults.params_table()`). Returned as `list[dict]`
        rather than a polars DataFrame, per the project's policy of
        not using DataFrames for the parameter table itself.

        Returns:
            A list of dictionaries, one per coefficient. Keys are
            `param_name`, `param`, `std_error`, `test_stat`, `p_value`,
            `conf_lower`, `conf_upper`.
        """
        return params_table_rows(self._raw)

    def first_stage(self) -> dict[str, FEResults]:
        """Per-endogenous-variable first-stage regression results.

        Each first-stage regression is `x_endog[i] ~ x_exog +
        instruments` with the same fixed effects and `cov_type` as the
        structural equation, estimated by the within estimator. Returns
        the existing `FEResults` type rather than a new type: the first
        stage is a genuine fixed-effects regression in its own right.
        Its `f_statistic`, `f_p_value`, `f_df_num` and `f_df_denom` are
        `None` (the overall test is not computed for a first stage), and
        it is not the weak-instrument partial F-statistic
        (`weak_instrument_f_statistics`). Each result has its own
        `fixed_effects()`.

        Returns:
            A dictionary keyed by endogenous variable name, in the order
            of `x_endog`; the values are the first-stage `FEResults`.
        """
        stages = self._raw.first_stage()
        return {name: FEResults(stages[name]) for name in self._x_endog}

    def fixed_effects(self) -> dict[str, dict[str, float]]:
        """The fixed effects themselves (`α_i` for the first column of
        `effects`, `γ_t` for the second, and one effect per group of
        each further column), recovered post-hoc from the fitted
        coefficients with the actual endogenous regressors.

        Always a `dict[str, dict[str, float]]` keyed by the column
        names in `effects`, then by group (the value of the column as a
        string). The normalization is the same as
        `FEResults.fixed_effects()`.

        This is a post-estimation computation: it is never done by
        `fit()`. It re-reads the variables from the data passed to
        `fit()` and estimates the model again on every call, so it costs
        about as much as `fit()`.

        Returns:
            The fixed effects, shaped as described above.
        """
        return self._raw.fixed_effects()

    def cragg_donald(self) -> CraggDonaldResult:
        """Cragg-Donald statistic and Stock-Yogo critical values.

        The Cragg-Donald statistic measures the joint strength of the
        excluded instruments for all endogenous variables, assuming
        homoskedastic errors; it does not depend on `cov_type`. The
        absorbed fixed effects are subtracted from its degrees of
        freedom. With one endogenous variable it equals the partial
        F-statistic in `weak_instrument_f_statistics`. The Stock-Yogo
        critical values are tabulated for a limited number of endogenous
        variables and excluded instruments (see `CraggDonaldResult`).

        The Stock-Yogo critical values are derived for independent,
        homoskedastic errors; with heteroskedastic, clustered or
        autocorrelated errors use `kleibergen_paap()` or `effective_f()`.

        This is a post-estimation diagnostic: it is never computed by
        `fit()`. It re-reads the variables from the data passed to
        `fit()` and estimates the model again on every call.

        Returns:
            A `CraggDonaldResult` with the statistic, the numbers of
            endogenous variables and excluded instruments, and the
            critical values (`None` where not tabulated).

        Raises:
            ComputationError: The statistic is undefined for these data,
                for example because the instruments are collinear with
                the exogenous variables or a first stage fits perfectly.
        """
        return cragg_donald_result(self._raw.cragg_donald())

    def kleibergen_paap(self) -> KleibergenPaapResult:
        """Kleibergen-Paap rk LM and rk Wald F statistics.

        Robust versions of the Cragg-Donald analysis. The covariance of
        the first-stage scores follows the `cov_type` (and `cluster`,
        `dk_time`, `dk_bandwidth`) of the options passed to `fit()`, so
        with `cov_type="classical"` the rk Wald F equals the
        Cragg-Donald statistic. The rk LM statistic tests that the first
        stage is underidentified. The result has no critical values; for
        one endogenous variable use `effective_f()`.

        This is a post-estimation diagnostic: it is never computed by
        `fit()`. It re-reads the variables from the data passed to
        `fit()` and estimates the model again on every call.

        Returns:
            A `KleibergenPaapResult`. Either statistic is `None` when
            only the other is defined (the Wald F needs one more cluster
            than the LM).

        Raises:
            ComputationError: Neither statistic is defined for these
                data, for example because the instruments are collinear
                with the exogenous variables or the covariance of the
                scores is singular.
        """
        return kleibergen_paap_result(self._raw.kleibergen_paap())

    def effective_f(self, *, alpha: float = 0.05) -> EffectiveFResult:
        """Montiel Olea-Pflueger effective F statistic.

        A weak-instrument statistic for one endogenous variable that is
        valid under heteroskedasticity, clustering and autocorrelation.
        The covariance follows the `cov_type` (and `cluster`, `dk_time`,
        `dk_bandwidth`) of the options passed to `fit()`; with
        `cov_type="classical"` it equals the Cragg-Donald statistic.
        With one or two excluded instruments, prefer the simplified
        critical values (see `EffectiveFResult`).

        This is a post-estimation diagnostic: it is never computed by
        `fit()`. It re-reads the variables from the data passed to
        `fit()` and estimates the model again on every call.

        Args:
            alpha: Level of the critical values, strictly between 0 and
                1. Defaults to 0.05.

        Returns:
            An `EffectiveFResult` with the statistic and the critical
            values for the worst-case Nagar bias thresholds.

        Raises:
            TypeError: `alpha` is not a real number.
            ValidationError: `alpha` is not strictly between 0 and 1, or
                the model does not have exactly one endogenous variable.
            ComputationError: The statistic is undefined for these data,
                for example because the covariance of the scores is
                singular (too few clusters or observations).
        """
        return effective_f_result(self._raw.effective_f(alpha))
