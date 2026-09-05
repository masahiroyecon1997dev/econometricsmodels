"""Logit の入力・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」）。成功パスの構造・API・
オプション反映は `test_logit_api.py`、主リファレンス（statsmodels）との
数値照合は `test_logit_reference.py`、R クロスチェックは
`test_logit_crosscheck.py`（OLS/WLS の `test_<手法>_validation.py` 等と同じ
4分割、`refactoring-candidates-2.md` 項目68）。

テスト本体は `Logit`/`Probit` で完全に重複するため
`_binary_choice_checks.py` に集約し（`refactoring-candidates-2.md` 項目95）、
このファイルは薄いラッパーに保つ。
"""

from __future__ import annotations

import _binary_choice_checks as _checks
import pytest
from econometricsmodels import Logit, LogitOptions

# ── ValidationError（入力データ） ───────────────────────────────────


def test_y_in_x_raises(binary_dataset):
    _checks.check_y_in_x_raises(binary_dataset, Logit)


def test_duplicate_x_column_raises(binary_dataset):
    _checks.check_duplicate_x_column_raises(binary_dataset, Logit)


def test_const_collision_with_include_intercept_raises():
    _checks.check_const_collision_with_include_intercept_raises(Logit)


def test_empty_x_raises(binary_dataset):
    _checks.check_empty_x_raises(binary_dataset, Logit)


def test_missing_column_raises(binary_dataset):
    _checks.check_missing_column_raises(binary_dataset, Logit)


def test_null_values_raise():
    _checks.check_null_values_raise(Logit)


def test_non_numeric_dtype_raises():
    _checks.check_non_numeric_dtype_raises(Logit)


@pytest.mark.parametrize("bad_value", [0.5, 2.0, -1.0])
def test_non_binary_y_raises(binary_dataset, bad_value):
    _checks.check_non_binary_y_raises(binary_dataset, Logit, bad_value)


def test_insufficient_observations_raises(binary_dataset):
    _checks.check_insufficient_observations_raises(binary_dataset, Logit)


# ── ValidationError（オプション） ──────────────────────────────────


def test_unknown_cov_type_raises(binary_dataset):
    _checks.check_unknown_cov_type_raises(binary_dataset, Logit, LogitOptions)


def test_unknown_method_raises(binary_dataset):
    _checks.check_unknown_method_raises(binary_dataset, Logit, LogitOptions)


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(binary_dataset, confidence_level):
    _checks.check_invalid_confidence_level_raises(
        binary_dataset, Logit, LogitOptions, confidence_level
    )


@pytest.mark.parametrize("tol", [0.0, -1.0])
def test_non_positive_tol_raises(binary_dataset, tol):
    _checks.check_non_positive_tol_raises(
        binary_dataset, Logit, LogitOptions, tol
    )


@pytest.mark.parametrize("max_iter", [0, -1])
def test_non_positive_max_iter_raises(binary_dataset, max_iter):
    _checks.check_non_positive_max_iter_raises(
        binary_dataset, Logit, LogitOptions, max_iter
    )


def test_cluster_cov_type_requires_at_least_two_groups():
    _checks.check_cluster_cov_type_requires_at_least_two_groups(
        Logit, LogitOptions
    )


def test_cluster_col_nonexistent_column_raises(binary_dataset):
    _checks.check_cluster_col_nonexistent_column_raises(
        binary_dataset, Logit, LogitOptions
    )


# ── ValidationError（marginal_effects()） ─────────────────────────


def test_marginal_effects_unknown_at_raises(binary_dataset):
    _checks.check_marginal_effects_unknown_at_raises(binary_dataset, Logit)


def test_marginal_effects_confidence_level_out_of_range_raises(
    binary_dataset,
):
    _checks.check_marginal_effects_confidence_level_out_of_range_raises(
        binary_dataset, Logit
    )


# ── ComputationError ──────────────────────────────────────────────


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_singular_hessian_raises_computation_error(method):
    _checks.check_singular_hessian_raises_computation_error(
        Logit, LogitOptions, method
    )


def test_perfect_multicollinearity_raises_computation_error():
    _checks.check_perfect_multicollinearity_raises_computation_error(
        Logit, "logit"
    )


def test_non_convergence_raises_computation_error_with_tiny_max_iter(
    binary_dataset,
):
    _checks.check_non_convergence_raises_computation_error_with_tiny_max_iter(
        binary_dataset, Logit, LogitOptions
    )


def test_separation_suspected_raises_computation_error_for_near_separation_data():
    _checks.check_separation_suspected_raises_computation_error_for_near_separation_data(
        Logit
    )
