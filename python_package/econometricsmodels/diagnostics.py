"""Result types of post-estimation diagnostic tests.

Post-estimation diagnostics (for example `OLSResults.white_test()`) are
separate methods that the user calls after fitting; they are never
computed by `fit()`. Each returns a frozen dataclass so that values are
read as attributes (`result.p_value`), like the properties of the fit
results. Use `to_dict()` to get a JSON-ready `dict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = ["DiagnosticResult", "WhiteTestResult"]


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
