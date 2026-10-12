"""Logit の入力・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」）。成功パスの構造・API・
オプション反映は `test_logit_api.py`、主リファレンス（statsmodels）との
数値照合は `test_logit_reference.py`、R クロスチェックは
`test_logit_crosscheck.py`（OLS/WLS の `test_<手法>_validation.py` 等と同じ
4分割）。

テスト本体は `Logit`/`Probit` で完全に重複するため
`_binary_choice_checks.py` に集約し、このファイルは薄いラッパーに保つ。
"""

from __future__ import annotations

import _binary_choice_checks as _checks
import pytest
from econometricsmodels import Logit, LogitOptions

# ── ValidationError（入力データ） ─────────────────────────────────────


def test_y_in_x_raises(binary_dataset):
    _checks.check_y_in_x_raises(binary_dataset, Logit)


def test_y_empty_string_raises(binary_dataset):
    _checks.check_y_empty_string_raises(binary_dataset, Logit)


def test_duplicate_x_column_raises(binary_dataset):
    _checks.check_duplicate_x_column_raises(binary_dataset, Logit)


def test_const_collision_with_include_intercept_raises():
    _checks.check_const_collision_with_include_intercept_raises(Logit)


def test_empty_x_raises(binary_dataset):
    _checks.check_empty_x_raises(binary_dataset, Logit)


def test_missing_column_raises(binary_dataset):
    _checks.check_missing_column_raises(binary_dataset, Logit)


def test_data_not_polars_raises(binary_dataset):
    _checks.check_data_not_polars_raises(binary_dataset, Logit)


def test_null_values_raise():
    _checks.check_null_values_raise(Logit)


def test_non_finite_values_raise():
    _checks.check_non_finite_values_raise(Logit)


def test_non_numeric_dtype_raises():
    _checks.check_non_numeric_dtype_raises(Logit)


@pytest.mark.parametrize("bad_value", [0.5, 2.0, -1.0])
def test_non_binary_y_raises(binary_dataset, bad_value):
    _checks.check_non_binary_y_raises(binary_dataset, Logit, bad_value)


def test_insufficient_observations_raises(binary_dataset):
    _checks.check_insufficient_observations_raises(binary_dataset, Logit)


# ── ValidationError（predict()のnew_data） ────────────────────────────


def test_predict_missing_column_raises(binary_dataset):
    _checks.check_predict_missing_column_raises(binary_dataset, Logit)


def test_predict_non_numeric_dtype_raises(binary_dataset):
    _checks.check_predict_non_numeric_dtype_raises(binary_dataset, Logit)


def test_predict_null_or_non_finite_values_raise(binary_dataset):
    _checks.check_predict_null_or_non_finite_values_raise(
        binary_dataset, Logit
    )


# ── ValidationError（augment()） ──────────────────────────────────────


def test_augment_column_collision_raises(binary_dataset):
    _checks.check_augment_column_collision_raises(binary_dataset, Logit)


def test_augment_missing_column_raises(binary_dataset):
    _checks.check_augment_missing_column_raises(binary_dataset, Logit)


# ── ValidationError（オプション） ─────────────────────────────────────


@pytest.mark.parametrize("cov_type", ["bogus", ""])
def test_unknown_cov_type_raises(binary_dataset, cov_type):
    _checks.check_unknown_cov_type_raises(
        binary_dataset, Logit, LogitOptions, cov_type
    )


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_rejected(binary_dataset, cov_type):
    _checks.check_nonrobust_is_rejected(
        binary_dataset, Logit, LogitOptions, cov_type
    )


@pytest.mark.parametrize("solver", ["bogus", ""])
def test_unknown_solver_raises(binary_dataset, solver):
    _checks.check_unknown_method_raises(
        binary_dataset, Logit, LogitOptions, solver
    )


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


def test_cluster_count_at_most_slopes_raises_validation_error(binary_dataset):
    _checks.check_cluster_count_at_most_slopes_raises_validation_error(
        binary_dataset, Logit, LogitOptions
    )


def test_mroz_cluster_cov_type_raises_validation_error():
    _checks.check_mroz_cluster_cov_type_raises_validation_error(
        Logit, LogitOptions
    )


def test_cluster_without_col_raises(binary_dataset):
    _checks.check_cluster_without_col_raises(
        binary_dataset, Logit, LogitOptions
    )


def test_cluster_nonexistent_column_raises(binary_dataset):
    _checks.check_cluster_nonexistent_column_raises(
        binary_dataset, Logit, LogitOptions
    )


# ── ValidationError（marginal_effects()） ─────────────────────────────


def test_marginal_effects_unknown_at_raises(binary_dataset):
    _checks.check_marginal_effects_unknown_at_raises(binary_dataset, Logit)


def test_marginal_effects_confidence_level_out_of_range_raises(
    binary_dataset,
):
    _checks.check_marginal_effects_confidence_level_out_of_range_raises(
        binary_dataset, Logit
    )


# ── ComputationError ──────────────────────────────────────────────────


@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_perfect_multicollinearity_raises_computation_error(solver):
    _checks.check_perfect_multicollinearity_raises_computation_error(
        Logit, "logit", LogitOptions, solver
    )


_SEPARATION_SCENARIOS = ["complete_separation", "quasi_complete_separation"]


@pytest.mark.parametrize("scenario", _SEPARATION_SCENARIOS)
@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_separation_raises_computation_error(solver, scenario):
    _checks.check_separation_raises_computation_error(
        Logit, "logit", LogitOptions, solver, scenario
    )


@pytest.mark.parametrize("scenario", _SEPARATION_SCENARIOS)
def test_separation_is_reported_as_separation_by_newton(scenario):
    _checks.check_separation_is_reported_as_separation_by_newton(
        Logit, "logit", LogitOptions, scenario
    )


@pytest.mark.parametrize("scenario", _SEPARATION_SCENARIOS)
@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_separation_with_raise_on_non_convergence_false(solver, scenario):
    _checks.check_separation_with_raise_on_non_convergence_false(
        Logit, "logit", LogitOptions, solver, scenario
    )


@pytest.mark.parametrize("scenario", _SEPARATION_SCENARIOS)
def test_separation_returns_unconverged_result_for_newton(scenario):
    _checks.check_separation_returns_unconverged_result_for_newton(
        Logit, "logit", LogitOptions, scenario
    )


@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_small_sample_separation_is_not_reported_as_success(solver):
    _checks.check_small_sample_separation_is_not_reported_as_success(
        Logit, LogitOptions, solver
    )


@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_multiple_dummy_separation_raises_computation_error(solver):
    _checks.check_multiple_dummy_separation_raises_computation_error(
        Logit, LogitOptions, solver
    )


@pytest.mark.parametrize("overlap", _checks.OVERLAP_ABOVE_BOUNDARY)
@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_single_overlap_above_boundary_is_not_flagged_as_separation(
    solver, overlap
):
    _checks.check_single_overlap_above_boundary_is_not_flagged_as_separation(
        Logit, LogitOptions, solver, overlap, min_slope=250.0
    )


@pytest.mark.parametrize("overlap", _checks.OVERLAP_BELOW_BOUNDARY)
@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_single_overlap_below_boundary_is_flagged_as_separation(
    solver, overlap
):
    _checks.check_single_overlap_below_boundary_is_flagged_as_separation(
        Logit, LogitOptions, solver, overlap
    )


def test_separation_without_intercept_raises_computation_error():
    _checks.check_separation_without_intercept_raises_computation_error(
        Logit, LogitOptions
    )


@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_strongly_collinear_regressors_are_not_rejected_as_separation(solver):
    _checks.check_strongly_collinear_regressors_are_not_rejected_as_separation(
        Logit, LogitOptions, "logit", solver
    )


def test_non_convergence_raises_computation_error_with_tiny_max_iter(
    binary_dataset,
):
    _checks.check_non_convergence_raises_computation_error_with_tiny_max_iter(
        binary_dataset, Logit, LogitOptions
    )


def test_separated_data_raises_computation_error():
    _checks.check_separated_data_raises_computation_error(Logit)


def test_cluster_unused_by_cov_type_raises(binary_dataset):
    _checks.check_cluster_unused_by_cov_type_raises(
        binary_dataset, Logit, LogitOptions
    )
