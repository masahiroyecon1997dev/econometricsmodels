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
]


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
