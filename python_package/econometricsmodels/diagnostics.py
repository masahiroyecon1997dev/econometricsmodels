"""Result types of post-estimation diagnostics.

Post-estimation diagnostics (for example `OLSResults.white_test()` or
`IVResults.cragg_donald()`) are separate methods that the user calls after
fitting; they are never computed by `fit()`. Each returns a frozen
dataclass so that values are read as attributes (`result.p_value`), like
the properties of the fit results. Use `to_dict()` to get a plain `dict`.
The hypothesis tests of OLS share `DiagnosticResult`; the weak-instrument
diagnostics of IV have no p-value (or two statistics) and each has its own
result type.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = [
    "BreuschGodfreyTestResult",
    "BreuschPaganTestResult",
    "CraggDonaldResult",
    "DiagnosticResult",
    "EffectiveFResult",
    "KleibergenPaapResult",
    "WhiteTestResult",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticResult:
    """Result of a post-estimation diagnostic test.

    Attributes:
        statistic: Value of the test statistic.
        p_value: P-value of the test.
        df: Degrees of freedom of the chi-squared test, or the numerator
            degrees of freedom of the F test.
        df_denom: Denominator degrees of freedom of the F test, or `None`
            for a chi-squared test.
        distribution: Null distribution of `statistic`: `"chi2"` or
            `"f"`.
    """

    statistic: float
    p_value: float
    df: int
    df_denom: int | None
    distribution: Literal["chi2", "f"]

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain `dict` (JSON-ready).

        Returns:
            A dictionary with one key per field.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class WhiteTestResult(DiagnosticResult):
    """Result of `OLSResults.white_test()`.

    The auxiliary regression always includes a constant, whether or not
    the model was fitted with `include_intercept`. The first element of
    `aux_terms` is therefore always that constant, `"const"`, and for the
    LM test `df == len(aux_terms) - 1`.

    The term labels (`"x1"`, `"x1^2"`, `"x1:x2"`) only describe what was
    used. They are not a formula and are never parsed. A column name that
    itself contains `^` or `:` makes a label ambiguous, and so does a
    column named `"const"` in a model fitted with
    `include_intercept=False` (the first element is still the constant).

    Attributes are read-only, but `aux_terms` and `dropped_terms` are plain
    lists: do not modify them in place, and note that the instance is
    therefore not hashable. `to_dict()` returns copies.

    Attributes:
        aux_terms: Terms of the auxiliary regression that was run: the
            constant, then the independent variables, their squares and
            their pairwise products, without the dropped terms.
        dropped_terms: Terms left out because they were constant or
            numerically identical to an earlier term (for example the
            square of a 0/1 dummy, which equals the dummy itself).
    """

    aux_terms: list[str]
    dropped_terms: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class BreuschPaganTestResult(DiagnosticResult):
    """Result of `OLSResults.breusch_pagan_test()`.

    The auxiliary regression always includes a constant, whether or not
    the model was fitted with `include_intercept`. The first element of
    `aux_terms` is therefore always that constant, `"const"`, and for the
    LM test `df == len(aux_terms) - 1`.

    The other terms are the names of the columns that were used, as given.
    A column named `"const"` is ambiguous with the constant (the first
    element is still the constant).

    Attributes are read-only, but `aux_terms` and `dropped_terms` are plain
    lists: do not modify them in place, and note that the instance is
    therefore not hashable. `to_dict()` returns copies.

    Attributes:
        aux_terms: Terms of the auxiliary regression that was run: the
            constant, then the variables that were used, without the
            dropped ones.
        dropped_terms: Variables left out because they were constant or
            numerically identical to an earlier variable.
    """

    aux_terms: list[str]
    dropped_terms: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class BreuschGodfreyTestResult(DiagnosticResult):
    """Result of `OLSResults.breusch_godfrey_test()`.

    `df` always equals `nlags` (the chi-squared degrees of freedom for the
    LM test, the numerator degrees of freedom for the F test). `nlags` is
    kept as a separate field so the lag order reads directly.

    Attributes:
        nlags: Number of lags of the residuals in the auxiliary regression.
    """

    nlags: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CraggDonaldResult:
    """Result of `IVResults.cragg_donald()`.

    The Cragg-Donald statistic measures the joint strength of the excluded
    instruments for all endogenous variables. It is not a test with a
    p-value: compare it with the Stock-Yogo critical values. Use
    `tsls_bias_critical_values` to limit the bias of the 2SLS estimate and
    `tsls_size_critical_values` to limit the size distortion of its Wald
    test. A statistic above the critical value for a level rejects, at the
    5% significance level, that the instruments are weak at that level. With
    one endogenous variable it equals the partial F-statistic in
    `IVResults.weak_instrument_f_statistics`.

    The critical values are tabulated for a limited number of endogenous
    variables and excluded instruments. Outside the table the statistic is
    still valid and the corresponding field is `None`.

    Attributes are read-only, but the critical values are plain dicts: do
    not modify them in place, and note that the instance is therefore not
    hashable. `to_dict()` returns copies. The keys of those dicts are
    numbers, which `json.dumps` writes as strings.

    Attributes:
        statistic: Cragg-Donald Wald F statistic.
        n_endog: Number of endogenous variables.
        n_excluded: Number of excluded instruments.
        tsls_bias_critical_values: Critical values for the relative bias
            of 2SLS, keyed by the largest acceptable relative bias (0.05,
            0.10, 0.20, 0.30), or `None` when not tabulated (tabulated for
            1 to 3 endogenous variables, and from `n_endog + 2` to 30
            excluded instruments).
        tsls_size_critical_values: Critical values for the size distortion
            of the 5% Wald test of 2SLS, keyed by the largest acceptable
            actual size of that test (0.10, 0.15, 0.20, 0.25), or `None`
            when not tabulated (tabulated for 1 or 2 endogenous variables,
            and from `n_endog` to 30 excluded instruments).
    """

    statistic: float
    n_endog: int
    n_excluded: int
    tsls_bias_critical_values: dict[float, float] | None
    tsls_size_critical_values: dict[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain `dict` (the critical-value keys are numbers; see the class note).

        Returns:
            A dictionary with one key per field.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class KleibergenPaapResult:
    """Result of `IVResults.kleibergen_paap()`.

    The Kleibergen-Paap statistics are robust versions of the
    Cragg-Donald analysis: the covariance of the first-stage scores
    follows the `cov_type` of the fit. The rk LM statistic tests that the
    first stage is underidentified (the instruments do not span the
    endogenous variables) and is chi-squared with
    `n_excluded - n_endog + 1` degrees of freedom. The rk Wald F statistic
    is the robust counterpart of the Cragg-Donald statistic. It has no
    p-value and no critical values here, and comparing it with the
    Stock-Yogo values (see `CraggDonaldResult`) is only a heuristic; for
    one endogenous variable use `IVResults.effective_f()`. The rejection of
    the LM test does not show that the instruments are strong, because its
    null is underidentification. The two statistics are defined
    independently, so one of them can be `None`: the Wald F needs one more
    cluster than the LM.

    Attributes are read-only and `to_dict()` returns a new `dict`.

    Attributes:
        rk_lm_statistic: rk LM statistic, or `None` when it is undefined.
        rk_lm_df: Degrees of freedom of the rk LM statistic, or `None`
            exactly when `rk_lm_statistic` is `None`.
        rk_lm_p_value: P-value of the rk LM statistic, or `None` exactly
            when `rk_lm_statistic` is `None`.
        rk_wald_f: rk Wald F statistic, or `None` when it is undefined.
        n_endog: Number of endogenous variables.
        n_excluded: Number of excluded instruments.
    """

    rk_lm_statistic: float | None
    rk_lm_df: int | None
    rk_lm_p_value: float | None
    rk_wald_f: float | None
    n_endog: int
    n_excluded: int

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain `dict` (JSON-ready).

        Returns:
            A dictionary with one key per field.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveFResult:
    """Result of `IVResults.effective_f()`.

    The effective F statistic of Montiel Olea and Pflueger (2013) is a
    weak-instrument statistic for one endogenous variable that stays valid
    under heteroskedasticity, clustering and autocorrelation. Compare it
    with a critical value: if `statistic` exceeds the critical value for
    `tau`, the null that the worst-case Nagar bias of 2SLS exceeds `tau`
    times a worst-case benchmark (the bias when the instruments are
    uninformative and the two errors are perfectly correlated; with
    homoskedastic errors, the OLS bias) is rejected at level `alpha`. The
    critical values are keyed by `tau` (0.05, 0.10, 0.20, 0.30).
    `critical_values`, when not `None`, is the sharper set;
    `simplified_critical_values` is always available and is conservative.
    With one or two excluded instruments the 2SLS bias has no first moment
    in general, so a bias-based critical value is a weaker guide there;
    prefer `simplified_critical_values`. With two excluded instruments and
    `cov_type="classical"`, `bias_bound` is 0 and `critical_values` are the
    same for every `tau`.

    Attributes are read-only, but the critical values are plain dicts: do
    not modify them in place, and note that the instance is therefore not
    hashable. `to_dict()` returns copies. The keys of those dicts are
    numbers, which `json.dumps` writes as strings.

    Attributes:
        statistic: Effective F statistic.
        n_excluded: Number of excluded instruments.
        alpha: Level of the critical values.
        simplified_critical_values: Critical values of the simplified 2SLS
            bias bound, keyed by `tau`.
        critical_values: Critical values that use the data-dependent
            bound `bias_bound` on the worst-case bias (equal to or smaller
            than `simplified_critical_values`), keyed by `tau`, or `None`
            when the bound is not defined.
        bias_bound: The bound on the worst-case Nagar bias of 2SLS relative
            to the benchmark, greater than 0 and at most 1, or `None` when it
            is not defined.
    """

    statistic: float
    n_excluded: int
    alpha: float
    simplified_critical_values: dict[float, float]
    critical_values: dict[float, float] | None
    bias_bound: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain `dict` (the critical-value keys are numbers; see the class note).

        Returns:
            A dictionary with one key per field.
        """
        return asdict(self)
