"""Shared helpers for the `*Results` wrappers.

Every result class turns the parallel arrays of its `_lib` result object
(`param_names`, `params`, `std_errors`, ...) into name-keyed dicts or
row-oriented tables. Keeping that zip in one place means a fix (such as
guarding against duplicate coefficient names) applies to every method at
once. Property docstrings stay on each result class, since they differ
per method.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

__all__ = [
    "by_param",
    "conf_int_by_param",
    "cov_params_by_param",
    "marginal_effect_rows",
    "params_table_rows",
    "results_repr",
]

# Most coefficient rows `results_repr` shows; with more coefficients, the
# first `_REPR_MAX_ROWS - 1` and the last one are shown.
_REPR_MAX_ROWS = 10
# Longest coefficient name `results_repr` shows before cutting it short.
_REPR_MAX_NAME = 30


class _ConfIntColumns(Protocol):
    @property
    def param_names(self) -> Sequence[str]: ...
    @property
    def conf_lower(self) -> Sequence[float]: ...
    @property
    def conf_upper(self) -> Sequence[float]: ...


class _ParamsTableColumns(_ConfIntColumns, Protocol):
    @property
    def params(self) -> Sequence[float]: ...
    @property
    def std_errors(self) -> Sequence[float]: ...
    @property
    def test_stats(self) -> Sequence[float]: ...
    @property
    def p_values(self) -> Sequence[float]: ...


class _ReprColumns(Protocol):
    @property
    def dep_var_name(self) -> str: ...
    @property
    def n_obs(self) -> int: ...
    @property
    def cov_type(self) -> str: ...
    @property
    def param_names(self) -> Sequence[str]: ...
    @property
    def params(self) -> Sequence[float]: ...
    @property
    def std_errors(self) -> Sequence[float]: ...


class _MarginalEffectColumns(_ConfIntColumns, Protocol):
    @property
    def effect(self) -> Sequence[float]: ...
    @property
    def std_errors(self) -> Sequence[float]: ...
    @property
    def test_stats(self) -> Sequence[float]: ...
    @property
    def p_values(self) -> Sequence[float]: ...


def by_param(
    param_names: Sequence[str], values: Sequence[float]
) -> dict[str, float]:
    """Pair each coefficient name with its value.

    Args:
        param_names: Coefficient names, in estimation order.
        values: One value per coefficient, in the same order.

    Returns:
        A dictionary from coefficient name to value.
    """
    return dict(zip(param_names, values))


def conf_int_by_param(
    raw: _ConfIntColumns,
) -> dict[str, tuple[float, float]]:
    """Pair each coefficient name with its `(lower, upper)` interval.

    Args:
        raw: A `_lib` result object exposing `param_names`,
            `conf_lower` and `conf_upper`.

    Returns:
        A dictionary from coefficient name to `(lower, upper)`.
    """
    return {
        name: (lower, upper)
        for name, lower, upper in zip(
            raw.param_names, raw.conf_lower, raw.conf_upper
        )
    }


def cov_params_by_param(
    param_names: Sequence[str], rows: Sequence[Sequence[float]]
) -> dict[str, dict[str, float]]:
    """Pair each coefficient name with its row of the covariance matrix.

    Args:
        param_names: Coefficient names, in estimation order.
        rows: The covariance matrix as one row per coefficient, in the
            same order on both axes.

    Returns:
        A dictionary of dictionaries, so the covariance of `a` and `b`
        is `result["a"]["b"]`.
    """
    return {
        row_name: dict(zip(param_names, row))
        for row_name, row in zip(param_names, rows)
    }


def params_table_rows(
    raw: _ParamsTableColumns,
) -> list[dict[str, float | str]]:
    """Build the row-oriented parameter table.

    Args:
        raw: A `_lib` result object exposing the coefficient columns.

    Returns:
        One dictionary per parameter with keys `param_name`, `param`,
        `std_error`, `test_stat`, `p_value`, `conf_lower`, `conf_upper`.
    """
    return [
        {
            "param_name": name,
            "param": param,
            "std_error": se,
            "test_stat": test_stat,
            "p_value": p,
            "conf_lower": lower,
            "conf_upper": upper,
        }
        for name, param, se, test_stat, p, lower, upper in zip(
            raw.param_names,
            raw.params,
            raw.std_errors,
            raw.test_stats,
            raw.p_values,
            raw.conf_lower,
            raw.conf_upper,
        )
    ]


def marginal_effect_rows(
    raw: _MarginalEffectColumns,
) -> list[dict[str, float | str]]:
    """Build the row-oriented marginal effects table.

    Same layout as `params_table_rows`, with `effect` in place of `param`.

    Args:
        raw: A `_lib` marginal effects object.

    Returns:
        One dictionary per regressor with keys `param_name`, `effect`,
        `std_error`, `test_stat`, `p_value`, `conf_lower`, `conf_upper`.
    """
    return [
        {
            "param_name": name,
            "effect": effect,
            "std_error": se,
            "test_stat": test_stat,
            "p_value": p,
            "conf_lower": lower,
            "conf_upper": upper,
        }
        for name, effect, se, test_stat, p, lower, upper in zip(
            raw.param_names,
            raw.effect,
            raw.std_errors,
            raw.test_stats,
            raw.p_values,
            raw.conf_lower,
            raw.conf_upper,
        )
    ]


def _repr_name(name: str) -> str:
    """Escape a coefficient name for one line and cut it to a fixed length."""
    shown = repr(name)[1:-1]
    if len(shown) > _REPR_MAX_NAME:
        return shown[: _REPR_MAX_NAME - 1] + "\u2026"
    return shown


def results_repr(class_name: str, raw: _ReprColumns) -> str:
    """Build the short text shown by `repr()` of a `*Results` object.

    The first line names the class, the dependent variable, the number
    of observations and the covariance type. Below it comes one row per
    coefficient with its estimate and standard error. With more than a
    fixed small number of coefficients, the first rows are followed by
    a line counting the omitted ones and then the last row, so a final
    parameter such as the `sigma` of Tobit is always shown. Names are
    escaped to one line and cut when long; widths are counted in
    characters, so wide (for example Japanese) names may misalign the
    columns. The layout is for reading in a terminal or notebook and
    may change between versions; use `params_table()` for
    machine-readable output.

    Args:
        class_name: Name of the result class, for the first line.
        raw: A `_lib` result object exposing the coefficient columns and
            `dep_var_name`, `n_obs` and `cov_type`.

    Returns:
        The text, without a trailing newline.
    """
    head = (
        f"{class_name}(dep_var={raw.dep_var_name!r}, "
        f"n_obs={raw.n_obs}, cov_type={raw.cov_type!r})"
    )
    all_names = raw.param_names
    all_params = raw.params
    all_std_errors = raw.std_errors
    count = len(all_names)
    if count > _REPR_MAX_ROWS:
        shown = [*range(_REPR_MAX_ROWS - 1), count - 1]
        omitted = count - _REPR_MAX_ROWS
    else:
        shown = list(range(count))
        omitted = 0
    names = [_repr_name(all_names[i]) for i in shown]
    params = [f"{all_params[i]:.6g}" for i in shown]
    std_errors = [f"{all_std_errors[i]:.6g}" for i in shown]
    name_width = max(map(len, names), default=0)
    param_width = max([len("param"), *map(len, params)])
    se_width = max([len("std_error"), *map(len, std_errors)])
    rows = [
        f"{name:<{name_width}}  {param:>{param_width}}  {se:>{se_width}}"
        for name, param, se in zip(names, params, std_errors)
    ]
    if omitted:
        rows.insert(len(rows) - 1, f"... ({omitted} more)")
    column_header = (
        f"{'':<{name_width}}  {'param':>{param_width}}  "
        f"{'std_error':>{se_width}}"
    )
    return "\n".join([head, column_header, *rows])
