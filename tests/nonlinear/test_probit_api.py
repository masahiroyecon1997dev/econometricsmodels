"""Probit の成功パスの構造・API・オプション反映・`predict()`/`pred_table()`/
`marginal_effects()` の検証。

確定済み設計（`docs/spec/probit-spec.md`）どおりの結果型・辞書キー・ラベルに
なっていること、`ProbitOptions` の各フィールドが engine_pybind 経由で反映される
ことを確認する（`test_logit_api.py` と同型）。`ValidationError`/
`ComputationError` パスは `test_probit_validation.py`、主リファレンス
（statsmodels）との数値照合は `test_probit_reference.py`、R クロスチェックは
`test_probit_crosscheck.py`（OLS/WLS の `test_<手法>_api.py` 等と同じ4分割、
`refactoring-candidates-2.md` 項目68）。

`binary_dataset` フィクスチャ（`dataset` の y を中央値で0/1化）は conftest.py で
Logit/Probit 共通定義。

テスト本体は `Logit`/`Probit` で完全に重複するため
`_binary_choice_checks.py` に集約し（`refactoring-candidates-2.md` 項目95）、
このファイルは薄いラッパーに保つ。
"""

from __future__ import annotations

import _binary_choice_checks as _checks
import pytest
from econometricsmodels import Probit, ProbitOptions, ProbitResults

# ── 成功パス・結果型 ──────────────────────────────────────────────


def test_fit_succeeds_and_returns_probit_results(binary_dataset):
    _checks.check_fit_succeeds_and_returns_results(
        binary_dataset, Probit, ProbitResults
    )


def test_default_options_use_classical_and_converge(binary_dataset):
    _checks.check_default_options_use_classical_and_converge(
        binary_dataset, Probit
    )


# ── API構造 ──────────────────────────────────────────────────────


def test_coef_table_structure(binary_dataset):
    _checks.check_coef_table_structure(binary_dataset, Probit)


def test_conf_int_structure(binary_dataset):
    _checks.check_conf_int_structure(binary_dataset, Probit)


def test_params_std_errors_z_stats_p_values_share_keys(binary_dataset):
    _checks.check_params_std_errors_z_stats_p_values_share_keys(
        binary_dataset, Probit
    )


def test_n_obs_matches_dataset_size(binary_dataset):
    _checks.check_n_obs_matches_dataset_size(binary_dataset, Probit)


def test_param_names_include_const_first(binary_dataset):
    _checks.check_param_names_include_const_first(binary_dataset, Probit)


# ── オプションの反映 ──────────────────────────────────────────────
#
# cov_type 以外の ProbitOptions フィールド（method・include_intercept・
# confidence_level・raise_on_non_convergence）が、engine_pybind 側の
# 文字列パース・列抽出・分岐ロジックを経て正しく反映されることを確認する。


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_method_option_converges_to_same_params(binary_dataset, method):
    _checks.check_method_option_converges_to_same_params(
        binary_dataset, Probit, ProbitOptions, method
    )


def test_include_intercept_false_omits_const_and_converges(binary_dataset):
    _checks.check_include_intercept_false_omits_const_and_converges(
        binary_dataset, Probit, ProbitOptions
    )


def test_confidence_level_changes_interval_width(binary_dataset):
    _checks.check_confidence_level_changes_interval_width(
        binary_dataset, Probit, ProbitOptions
    )


def test_raise_on_non_convergence_false_returns_result_without_raising(
    binary_dataset,
):
    _checks.check_raise_on_non_convergence_false_returns_result_without_raising(
        binary_dataset, Probit, ProbitOptions
    )


def test_cov_type_label(binary_dataset):
    _checks.check_cov_type_label(binary_dataset, Probit, ProbitOptions)


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
        binary_dataset, Probit, ProbitOptions, cov_type, expected_label
    )


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_alias_for_classical(binary_dataset, cov_type):
    _checks.check_nonrobust_is_alias_for_classical(
        binary_dataset, Probit, ProbitOptions, cov_type
    )


# ── predict() ────────────────────────────────────────────────────


def test_predict_returns_row_oriented_probabilities(binary_dataset):
    _checks.check_predict_returns_row_oriented_probabilities(
        binary_dataset, Probit
    )


# ── pred_table() ─────────────────────────────────────────────────


def test_pred_table_default_threshold_sums_to_n_obs(binary_dataset):
    _checks.check_pred_table_default_threshold_sums_to_n_obs(
        binary_dataset, Probit
    )


def test_pred_table_actual_counts_invariant_to_threshold(binary_dataset):
    _checks.check_pred_table_actual_counts_invariant_to_threshold(
        binary_dataset, Probit
    )


# ── marginal_effects() ────────────────────────────────────────────


def test_marginal_effects_default_excludes_intercept(binary_dataset):
    _checks.check_marginal_effects_default_excludes_intercept(
        binary_dataset, Probit
    )


def test_marginal_effects_mean_and_median_differ_from_overall(
    binary_dataset,
):
    _checks.check_marginal_effects_mean_and_median_differ_from_overall(
        binary_dataset, Probit
    )


def test_marginal_effects_at_is_case_insensitive(binary_dataset):
    _checks.check_marginal_effects_at_is_case_insensitive(
        binary_dataset, Probit
    )
