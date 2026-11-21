"""Python wrapper for FEIV (fixed effects with instrumental variables).

A thin wrapper around `econometricsmodels._lib.fit_fe_iv` (the Rust
implementation). Validation and estimation happen on the Rust side; this
module only provides the Python-facing API shape for polars DataFrames:
lists of column names for `x_exog`, `x_endog`, `instruments` and
`effects`, and an options object for the estimation settings.

`FEIVOptions` is re-exported as-is from `_lib`. `FEIVOptions.estimator`
selects two-stage least squares or GMM, as `IVOptions.estimator` does. The
first-stage regressions are returned as `FEResults`, and the
weak-instrument diagnostics and the C statistic use the same result types
as `IVResults`. There is no `summary()`: results are returned as
structured data.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .. import _lib
from .._lib import FEIVOptions
from .._results_utils import (
    by_param,
    c_statistic_result,
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
    CStatisticResult,
    EffectiveFResult,
    KleibergenPaapResult,
)
from .fe import FEResults

__all__ = ["FEIV", "FEIVOptions", "FEIVResults"]


class FEIV:
    """Fixed effects instrumental variables estimator (2SLS or GMM).

    Removes the fixed effects by the within transformation and then
    applies two-stage least squares (`estimator="2sls"`, the default) or
    the generalized method of moments (`estimator="gmm"`) to the
    transformed data, with the endogenous variables instrumented by
    `instruments`. The default GMM is two-step, weighted by the
    cluster-robust covariance of the moment conditions. With more
    instruments than endogenous variables it is asymptotically efficient
    under arbitrary correlation within the clusters (for example over time
    within an entity), and with few clusters the estimated weight can be
    unstable. With as many instruments as endogenous variables the GMM
    coefficients equal those of two-stage least squares. The `effects`
    argument works as in `FE`: one column gives one-way effects, two
    columns give two-way effects, and three to five columns give
    higher-dimensional effects.

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
            (`estimator="2sls"`, `cov_type="cluster"` on the first
            column of `effects`, `confidence_level=0.95`) when omitted.
            With `estimator="gmm"` the weight `gmm_weight_type` defaults
            to `"cluster"`, unlike `"classical"` in `IV`.

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import FEIV, FEIVOptions
        >>> df = pl.DataFrame(
        ...     {
        ...         "y": [1.0, 2.1, 2.9, 1.5, 2.2, 3.8, 0.9, 2.4, 3.3]
        ...         + [1.8, 2.6, 4.1],
        ...         "endog": [1.0, 2.2, 3.1, 1.2, 2.0, 3.9, 0.8, 2.1, 3.5]
        ...         + [1.9, 2.5, 3.8],
        ...         "z": [0.9, 2.5, 2.7, 1.6, 1.8, 4.0, 1.1, 2.3, 3.0]
        ...         + [1.7, 2.9, 4.2],
        ...         "z2": [1.2, 2.0, 3.4, 1.1, 2.6, 3.5, 1.0, 2.9, 3.1]
        ...         + [2.2, 2.3, 3.6],
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
        >>> result.estimator
        '2sls'
        >>> gmm = FEIV(
        ...     df,
        ...     y="y",
        ...     x_exog=[],
        ...     x_endog=["endog"],
        ...     instruments=["z", "z2"],
        ...     effects=["id"],
        ...     options=FEIVOptions(estimator="gmm"),
        ... ).fit()
        >>> gmm.estimator, gmm.gmm_type, gmm.gmm_weight_type
        ('gmm', 'two_step', 'cluster')
        >>> gmm.stat_dist, gmm.wald_dist
        ('normal', 'chi2')
        >>> gmm.overid_df
        1
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
        """Estimate the FEIV model with the chosen estimator.

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
                - There are too few degrees of freedom (too few
                  observations for the fixed effects and coefficients), or
                  `confidence_level` is out of range.
                - `estimator`, `cov_type`, `gmm_type` or
                  `gmm_weight_type` is unknown, or `cov_type` is
                  `"hc0"` (not supported for FEIV); there is no `"hac"`
                  weight.
                - An option is given where nothing uses it, for
                  example `dk_time` without `cov_type="dk"`, a GMM
                  option (such as `gmm_type`) with
                  `estimator="2sls"`, `gmm_max_iter` without
                  `gmm_type="iterated"`, or `cluster` when neither
                  `cov_type` nor (with GMM) `gmm_weight_type` is
                  `"cluster"`; or `cov_type="dk"` is requested without
                  `dk_time`.
                - A group of a column in `effects` has a single
                  observation.
                - With `estimator="gmm"`, the cluster weight is used with
                  too few clusters: with `l` moment conditions (the
                  columns of `x_exog` and `instruments`), the number of
                  clusters must be at least `l` when the model is
                  overidentified, and more than `l` when it is just
                  identified.

            ComputationError: A problem was detected during
                computation (e.g. a singular within-transformed design
                matrix, a perfect fit of the structural equation or of a
                first stage, a singular matrix in the GMM estimate or its
                weight (collinear regressors or instruments, with
                `estimator="gmm"`), alternating projections that did not
                converge within `max_iter`, or `gmm_type="iterated"`
                that did not converge within `gmm_max_iter` while
                `raise_on_non_convergence` is true). A subclass of
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
    """FEIV estimation results (two-stage least squares or GMM).

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
    - Estimator actually used: `estimator`, `gmm_type`,
      `gmm_weight_type`, `n_iter`, `converged`.
    - Weak instruments (the same for both estimators):
      `weak_instrument_f_statistics`, `weak_instrument_f_df_num`,
      `weak_instrument_f_df_denom`.
    - Overidentification (Sargan with `estimator="2sls"`, Hansen J with
      `estimator="gmm"`; all `None` when just identified):
      `overid_statistic`, `overid_p_value`, `overid_df`.
    - Endogeneity (the same for both estimators, under `cov_type`):
      `wu_hausman_statistic`, `wu_hausman_p_value`, `wu_hausman_df_num`,
      `wu_hausman_df_denom`.
    - Fit: `r_squared_within`, `r_squared_between`,
      `r_squared_overall`.
    - Panel size: `n_obs`, `n_groups`.
    - Model size: `df_model`, `df_resid`.
    - Settings echoed back: `dep_var_name`, `cov_type`,
      `dk_bandwidth_used`.
    - Residuals: `residuals` (structural residuals of the
      within-transformed data).
    - Methods: `first_stage()` (one `FEResults` per endogenous
      variable; the same for both estimators), `fixed_effects()` (the
      estimated effect of every group, for each effects column of the
      model; `estimator="2sls"` only), `params_table()`.
    - Weak-instrument diagnostics (methods, computed when called, the
      same for both estimators): `cragg_donald()`,
      `kleibergen_paap()`, `effective_f()`.
    - Exogeneity and orthogonality (a method, computed when called, with
      `estimator="gmm"` only): `c_stat()`.

    A method that the estimator does not support raises a
    `ValidationError`, and a property that it does not support is
    `None`.

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
        """Coefficient name to test statistic (t-statistic for
        `estimator="2sls"`, z-statistic for `estimator="gmm"`; see
        `stat_dist`)."""
        return by_param(self._raw.param_names, self._raw.test_stats)

    @property
    def stat_dist(self) -> Literal["t", "normal"]:
        """Distribution of `test_stats`: `"t"` for `estimator="2sls"`,
        `"normal"` (standard normal) for `estimator="gmm"`."""
        return self._raw.stat_dist

    @property
    def stat_df(self) -> int | None:
        """Degrees of freedom of the t distribution behind `test_stats`
        (`estimator="2sls"`). May differ from `df_resid` (cluster-robust
        inference uses `G - 1`, where `G` is the number of clusters, and
        Driscoll-Kraay the number of `dk_time` periods minus 1), so use
        this to recompute p-values from `test_stats`. `None` for
        `estimator="gmm"`: the standard normal distribution has none."""
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
        """Structural residuals, in observation order: the
        within-transformed `y` minus the within-transformed `X` times
        `β̂`, with the actual endogenous variables (not their
        first-stage fitted values)."""
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
    def estimator(self) -> Literal["2sls", "gmm"]:
        """Estimator actually used (normalized to lowercase): `"2sls"`
        or `"gmm"`."""
        return self._raw.estimator

    @property
    def gmm_type(
        self,
    ) -> Literal["one_step", "two_step", "iterated"] | None:
        """GMM estimation type actually used (normalized to lowercase):
        `"one_step"`, `"two_step"` or `"iterated"`. `None` for
        `estimator="2sls"`."""
        return self._raw.gmm_type

    @property
    def gmm_weight_type(
        self,
    ) -> Literal["classical", "robust", "cluster"] | None:
        """Weight matrix actually used for the point estimate (normalized
        to lowercase): `"classical"`, `"robust"` or `"cluster"`. `None`
        for `estimator="2sls"` and for `gmm_type="one_step"`, which has
        no weight type."""
        return self._raw.gmm_weight_type

    @property
    def n_iter(self) -> int:
        """Number of GMM estimations actually run, counting the initial
        estimate: 1 for `"one_step"`, 2 for `"two_step"`, at most
        `gmm_max_iter` for `"iterated"`. `1` for `estimator="2sls"`,
        which is a closed-form, non-iterative estimator."""
        return self._raw.n_iter

    @property
    def converged(self) -> bool:
        """Whether the iteration converged. Only meaningful for
        `gmm_type="iterated"`; always `True` for `"one_step"`,
        `"two_step"` and `estimator="2sls"`, which never check
        convergence."""
        return self._raw.converged

    @property
    def wald_statistic(self) -> float:
        """Statistic for all slope coefficients being zero. With
        `estimator="2sls"` it is F-type (the Wald statistic divided by
        the number of slope coefficients; a classical F-test when
        `cov_type="classical"`, a robust Wald test otherwise). With
        `estimator="gmm"` it is the Wald statistic itself, not divided
        by the number of slope coefficients; it follows a chi-squared
        distribution and uses the covariance of the coefficients, so it
        follows `cov_type`."""
        return self._raw.wald_statistic

    @property
    def wald_p_value(self) -> float:
        """P-value of `wald_statistic`."""
        return self._raw.wald_p_value

    @property
    def wald_dist(self) -> Literal["f", "chi2"]:
        """Distribution of `wald_statistic`: `"f"` (F distribution) for
        `estimator="2sls"`, `"chi2"` (chi-squared) for
        `estimator="gmm"`."""
        return self._raw.wald_dist

    @property
    def wald_df_num(self) -> int:
        """Numerator degrees of freedom of `wald_statistic` (for the
        chi-squared distribution of `estimator="gmm"`, its degrees of
        freedom): the number of slope coefficients. Never `None`, because
        `x_endog` always has at least one variable."""
        return self._raw.wald_df_num

    @property
    def wald_df_denom(self) -> int | None:
        """Denominator degrees of freedom of `wald_statistic` for
        `estimator="2sls"`. Follows `cov_type` like the t-tests, so it
        equals `stat_df`: the number of clusters minus 1 for
        `cov_type="cluster"`, the number of `dk_time` periods minus 1
        for `"dk"`, and `df_resid` otherwise. `None` for
        `estimator="gmm"`: the chi-squared distribution has none."""
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
        Computed the same way for both `estimator="2sls"` and
        `estimator="gmm"`.
        """
        stats = self._raw.weak_instrument_f_statistics
        return {name: stats[name] for name in self._x_endog}

    @property
    def weak_instrument_f_df_num(self) -> int | None:
        """Numerator degrees of freedom of the weak-instrument F
        statistics (the number of excluded instruments; the same for
        every endogenous variable). Never `None`, because `x_endog`
        always has at least one variable."""
        return self._raw.weak_instrument_f_df_num

    @property
    def weak_instrument_f_df_denom(self) -> int | None:
        """Denominator degrees of freedom of the weak-instrument F
        statistics (residual degrees of freedom of the first-stage
        regressions after the absorbed fixed effects; the same for every
        endogenous variable). Never `None`, because `x_endog`
        always has at least one variable."""
        return self._raw.weak_instrument_f_df_denom

    @property
    def overid_statistic(self) -> float | None:
        """Overidentification test statistic, computed from the
        within-transformed data: Sargan for `estimator="2sls"`, Hansen J
        for `estimator="gmm"`.

        The Sargan statistic is always under the classical
        (homoskedastic) formula, with the error variance `e'e / (n -
        n_absorbed)` (`n_absorbed` is the number of absorbed
        fixed-effect parameters), regardless of `cov_type`.

        The Hansen J statistic is computed at the estimate. Its
        covariance of the moment conditions follows the weight actually
        used (`gmm_weight_type`; the classical weight for
        `gmm_type="one_step"`). With the classical weight the error
        variance is `e'e / (n - n_absorbed)` like the Sargan statistic;
        the robust and cluster weights are not adjusted. The cluster
        weight needs no adjustment when every column of `effects` is
        nested in the clusters (for example one-way effects clustered by
        the entity, or by a coarser column). With an effect that crosses
        the clusters, such as a time effect in a two-way model, that
        holds only approximately, and not at all for clusters that cross
        the entities (for example the time periods). The robust weight
        makes the test over-reject on a short panel.

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
        """Wu-Hausman endogeneity test statistic (regression-based;
        `estimator="2sls"` only, `None` for `estimator="gmm"`).

        Adds the first-stage residuals to the structural equation and
        tests their joint significance in a fixed-effects regression,
        jointly over all endogenous variables. Unlike
        `weak_instrument_f_statistics`, this is always computed under
        the `cov_type` in `options`.

        `None` when the augmented regression cannot be estimated (the
        augmented design is singular, `y` is exactly explained by the
        regressors and the first-stage residuals, there are too few
        observations, or the covariance of the first-stage residuals'
        coefficients is numerically singular) — this does not affect
        the validity of the other results.
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
        rather than a polars DataFrame.

        Returns:
            A list of dictionaries, one per coefficient. Keys are
            `param_name`, `param`, `std_error`, `test_stat`, `p_value`,
            `conf_lower`, `conf_upper`.
        """
        return params_table_rows(self._raw)

    def first_stage(self) -> dict[str, FEResults]:
        """Per-endogenous-variable first-stage regression results
        (`estimator="2sls"` only).

        Each first-stage regression is `x_endog[i] ~ x_exog +
        instruments` with the same fixed effects and `cov_type` as the
        structural equation, estimated by the within estimator. Each value
        is an `FEResults`, the same type that `FE` returns.
        Its `f_statistic`, `f_p_value`, `f_df_num` and `f_df_denom` are
        `None` (the overall test is not computed for a first stage), and
        it is not the weak-instrument partial F-statistic
        (`weak_instrument_f_statistics`). Each result has its own
        `fixed_effects()`.

        Returns:
            A dictionary keyed by endogenous variable name, in the order
            of `x_endog`; the values are the first-stage `FEResults`.

        Raises:
            ValidationError: The model was fitted with
                `estimator="gmm"`.
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
        about as much as `fit()`. Available with `estimator="2sls"`
        only.

        Returns:
            The fixed effects, shaped as described above.

        Raises:
            ValidationError: The model was fitted with
                `estimator="gmm"`.
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

        Available with `estimator="2sls"` only.

        Returns:
            A `CraggDonaldResult` with the statistic, the numbers of
            endogenous variables and excluded instruments, and the
            critical values (`None` where not tabulated).

        Raises:
            ValidationError: The model was fitted with
                `estimator="gmm"`.
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

        Available with `estimator="2sls"` only.

        Returns:
            A `KleibergenPaapResult`. Either statistic is `None` when
            only the other is defined (the Wald F needs one more cluster
            than the LM).

        Raises:
            ValidationError: The model was fitted with
                `estimator="gmm"`.
            ComputationError: Neither statistic is defined for these
                data, for example because the instruments are collinear
                with the exogenous variables or the covariance of the
                scores is singular. With `cov_type="dk"` this also
                happens when the number of `dk_time` periods is at most
                `len(instruments) - len(x_endog) + 1`: neither statistic
                is defined then.
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
            ValidationError: `alpha` is not strictly between 0 and 1, the
                model does not have exactly one endogenous variable, or
                the model was fitted with `estimator="gmm"` (the
                diagnostic is available with `estimator="2sls"` only).
            ComputationError: The statistic is undefined for these data,
                for example because the covariance of the scores is
                singular (too few clusters or observations, or, with
                `cov_type="dk"`, a number of `dk_time` periods that is at
                most `len(instruments) - len(x_endog) + 1`).
        """
        return effective_f_result(self._raw.effective_f(alpha))

    def c_stat(
        self,
        *,
        variables: list[str] | None = None,
        instruments: list[str] | None = None,
    ) -> CStatisticResult:
        """C statistic (difference-in-Hansen test), for
        `estimator="gmm"` only.

        Tests whether endogenous variables in `variables` are in fact
        exogenous, and whether excluded instruments in `instruments` are
        uncorrelated with the error. The model is estimated by two-step GMM
        on the within-transformed data with and without the tested moment
        conditions, and the statistic is the difference of the two Hansen J
        statistics, which is chi-squared distributed under the null that
        the tested moment conditions are valid. The test is conditional:
        the remaining instruments are taken as valid. Both J statistics use
        the same estimate of the covariance of the moment conditions, so
        the statistic is never negative. Pass `variables`, `instruments`
        or both.

        The covariance of the moment conditions follows `gmm_weight_type`
        (grouped by `cluster` for the cluster weight) when `gmm_type` is
        `"two_step"` or `"iterated"`,
        and is the classical one for `gmm_type="one_step"`, which has no
        weight type. It does not depend on `cov_type`, `gmm_max_iter` or
        `gmm_tol`, and a model fitted with `gmm_type="one_step"` or
        `"iterated"` gets the two-step statistic. As for the Hansen J
        statistic, only the classical weight is adjusted by `n - n_absorbed`
        for the absorbed fixed effects. The cluster weight needs no
        adjustment when every column of `effects` is nested in the
        clusters, but only approximately when an effect crosses the
        clusters and not at all for clusters that cross the entities (for
        example the time periods), and with a robust weight on a short
        panel the statistic over-rejects.

        This is a post-estimation diagnostic: it is never computed by
        `fit()`. It re-reads the variables from the data passed to
        `fit()` (keep that data unchanged) and estimates the model again
        on every call.

        Args:
            variables: Names from `x_endog` whose exogeneity is tested.
                Defaults to `None` (no endogenous variable is tested). The
                names are reported in the order of `x_endog`, not the order
                given.
            instruments: Names from the excluded `instruments` whose
                orthogonality is tested. Defaults to `None` (no instrument
                is tested). At least as many instruments as endogenous
                variables must remain.

        Returns:
            A `CStatisticResult` with the statistic, p-value, degrees of
            freedom and the two J statistics.

        Raises:
            TypeError: `variables` or `instruments` is not a list of str.
            ValidationError: The model was fitted with
                `estimator="2sls"`, neither argument is given, a list is
                empty or repeats a name, a name in `variables` is not in
                `x_endog`, a name in `instruments` is not among the
                excluded instruments, or so many instruments are tested
                that fewer excluded instruments than endogenous variables
                remain.
            ComputationError: The statistic is undefined for these data, for
                example because the instruments are collinear, or because
                there are fewer clusters than the columns of the full
                instrument set (the exogenous regressors, the tested
                endogenous variables and all excluded instruments). Unlike
                `fit()`, which rejects too few clusters for the cluster
                weight with a `ValidationError`, the diagnostic can only
                detect this while computing.
        """
        return c_statistic_result(self._raw.c_stat(variables, instruments))
