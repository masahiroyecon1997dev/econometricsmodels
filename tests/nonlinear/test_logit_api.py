"""Logit の成功パスの構造・API・オプション反映・`predict()`/`pred_table()`/
`marginal_effects()` の検証。

確定済み設計（`docs/spec/logit-spec.md`）どおりの結果型・辞書キー・ラベルに
なっていること、`LogitOptions` の各フィールドが engine_pybind 経由で反映される
ことを確認する。`ValidationError`/`ComputationError` パスは
`test_logit_validation.py`、主リファレンス（statsmodels）との数値照合は
`test_logit_reference.py`、R クロスチェックは `test_logit_crosscheck.py`
（OLS/WLS の `test_<手法>_api.py` 等と同じ4分割、`refactoring-candidates-2.md`
項目68）。

`predict()`/`pred_table()`/`marginal_effects()` の構造・オプション反映は
このファイルに集約する（数値照合は `test_logit_reference.py` の
`_check_result`）。`marginal_effects()` の `ValidationError` パスのみ
`test_logit_validation.py`。

`binary_dataset` フィクスチャ（`dataset` の y を中央値で0/1化）は conftest.py で
Logit/Probit 共通定義。

テスト本体は `Logit`/`Probit` で完全に重複するため
`_binary_choice_checks.py` に集約し（`refactoring-candidates-2.md` 項目95）、
このファイルは薄いラッパーに保つ。
"""

from __future__ import annotations

import _binary_choice_checks as _checks
import pytest
from econometricsmodels import Logit, LogitOptions, LogitResults

# ── 成功パス・結果型 ──────────────────────────────────────────────


def test_fit_succeeds_and_returns_logit_results(binary_dataset):
    _checks.check_fit_succeeds_and_returns_results(
        binary_dataset, Logit, LogitResults
    )


def test_default_options_use_classical_and_converge(binary_dataset):
    _checks.check_default_options_use_classical_and_converge(
        binary_dataset, Logit
    )


# ── API構造 ──────────────────────────────────────────────────────


def test_coef_table_structure(binary_dataset):
    _checks.check_coef_table_structure(binary_dataset, Logit)


def test_conf_int_structure(binary_dataset):
    _checks.check_conf_int_structure(binary_dataset, Logit)


def test_params_std_errors_z_stats_p_values_share_keys(binary_dataset):
    _checks.check_params_std_errors_z_stats_p_values_share_keys(
        binary_dataset, Logit
    )


def test_n_obs_matches_dataset_size(binary_dataset):
    _checks.check_n_obs_matches_dataset_size(binary_dataset, Logit)


def test_param_names_include_const_first(binary_dataset):
    _checks.check_param_names_include_const_first(binary_dataset, Logit)


# ── オプションの反映 ──────────────────────────────────────────────
#
# cov_type 以外の LogitOptions フィールド（method・include_intercept・
# confidence_level・raise_on_non_convergence）が、engine_pybind 側の
# 文字列パース・列抽出・分岐ロジックを経て正しく反映されることを確認する。


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_method_option_converges_to_same_params(binary_dataset, method):
    _checks.check_method_option_converges_to_same_params(
        binary_dataset, Logit, LogitOptions, method
    )


def test_include_intercept_false_omits_const_and_converges(binary_dataset):
    _checks.check_include_intercept_false_omits_const_and_converges(
        binary_dataset, Logit, LogitOptions
    )


def test_confidence_level_changes_interval_width(binary_dataset):
    _checks.check_confidence_level_changes_interval_width(
        binary_dataset, Logit, LogitOptions
    )


def test_raise_on_non_convergence_false_returns_result_without_raising(
    binary_dataset,
):
    _checks.check_raise_on_non_convergence_false_returns_result_without_raising(
        binary_dataset, Logit, LogitOptions
    )


def test_cov_type_label(binary_dataset):
    _checks.check_cov_type_label(binary_dataset, Logit, LogitOptions)


@pytest.mark.parametrize(
    "cov_type, expected_label",
    [
        ("CLASSICAL", "classical"),
        ("Classical", "classical"),
        ("OPG", "opg"),
        ("Opg", "opg"),
        ("HC0", "hc0"),
        ("Hc1", "hc1"),
        ("CLUSTER", "cluster"),
        ("nonrobust", "nonrobust"),
        ("NONROBUST", "nonrobust"),
    ],
)
def test_cov_type_is_case_insensitive(
    binary_dataset, cov_type, expected_label
):
    _checks.check_cov_type_is_case_insensitive(
        binary_dataset, Logit, LogitOptions, cov_type, expected_label
    )


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_alias_for_classical(binary_dataset, cov_type):
    _checks.check_nonrobust_is_alias_for_classical(
        binary_dataset, Logit, LogitOptions, cov_type
    )


# ── predict() ────────────────────────────────────────────────────


def test_predict_returns_row_oriented_probabilities(binary_dataset):
    _checks.check_predict_returns_row_oriented_probabilities(
        binary_dataset, Logit
    )


# ── pred_table() ─────────────────────────────────────────────────


def test_pred_table_default_threshold_sums_to_n_obs(binary_dataset):
    _checks.check_pred_table_default_threshold_sums_to_n_obs(
        binary_dataset, Logit
    )


def test_pred_table_actual_counts_invariant_to_threshold(binary_dataset):
    _checks.check_pred_table_actual_counts_invariant_to_threshold(
        binary_dataset, Logit
    )


# ── marginal_effects() ────────────────────────────────────────────


def test_marginal_effects_default_excludes_intercept(binary_dataset):
    _checks.check_marginal_effects_default_excludes_intercept(
        binary_dataset, Logit
    )


def test_marginal_effects_mean_and_median_differ_from_overall(
    binary_dataset,
):
    _checks.check_marginal_effects_mean_and_median_differ_from_overall(
        binary_dataset, Logit
    )


def test_marginal_effects_at_is_case_insensitive(binary_dataset):
    _checks.check_marginal_effects_at_is_case_insensitive(
        binary_dataset, Logit
    )
