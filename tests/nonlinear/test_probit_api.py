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
"""

from __future__ import annotations

import pytest
from econometricsmodels import Probit, ProbitOptions, ProbitResults

# ── 成功パス・結果型 ──────────────────────────────────────────────


def test_fit_succeeds_and_returns_probit_results(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert isinstance(res, ProbitResults)


def test_default_options_use_classical_and_converge(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.cov_type == "classical"
    assert res.converged


# ── API構造 ──────────────────────────────────────────────────────


def test_coef_table_structure(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    table = res.coef_table()

    assert isinstance(table, list)
    assert len(table) == 3  # const, x1, x2
    expected_keys = {
        "param",
        "coef",
        "std_err",
        "z_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert expected_keys <= set(row.keys())
    assert [row["param"] for row in table] == ["const", "x1", "x2"]


def test_conf_int_structure(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    ci = res.conf_int
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_z_stats_p_values_share_keys(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    expected_keys = {"const", "x1", "x2"}
    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.z_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_n_obs_matches_dataset_size(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.n_obs == binary_dataset.height


def test_param_names_include_const_first(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.param_names == ["const", "x1", "x2"]


# ── オプションの反映 ──────────────────────────────────────────────
#
# cov_type 以外の ProbitOptions フィールド（method・include_intercept・
# confidence_level・raise_on_non_convergence）が、engine_pybind 側の
# 文字列パース・列抽出・分岐ロジックを経て正しく反映されることを確認する。


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_method_option_converges_to_same_params(binary_dataset, method):
    """`method`（newton/bfgs/lbfgs）はいずれも同じ最尤解に収束する。

    `engine/src/nonlinear/probit.rs`のRust単体テストは3手法の一致を検証済み
    だが、engine_pybindの文字列→`Method`パースやpython_packageラッパーの
    配線を検出するAPIレベルのテストが無かったため追加した（`test_logit_api.py`と
    同じ理由）。
    """
    baseline = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    res = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(method=method),
    ).fit()
    assert res.converged
    for name in res.param_names:
        assert res.params[name] == pytest.approx(
            baseline.params[name], rel=1e-4
        )


def test_include_intercept_false_omits_const_and_converges(binary_dataset):
    """`include_intercept=False`の構造面での成功パス
    （`test_logit_api.py`と同じ理由、Issue #231フェーズ4）。
    """
    res = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(include_intercept=False),
    ).fit()
    assert res.param_names == ["x1", "x2"]
    assert res.converged
    assert res.df_model == 1


def test_confidence_level_changes_interval_width(binary_dataset):
    """`confidence_level`を下げると信頼区間が狭くなること（`test_logit_api.py`と
    同じ理由、Issue #231フェーズ4）。
    """
    wide = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(confidence_level=0.99),
    ).fit()
    narrow = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(confidence_level=0.80),
    ).fit()

    for name in ["const", "x1", "x2"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width


def test_raise_on_non_convergence_false_returns_result_without_raising(
    binary_dataset,
):
    """`raise_on_non_convergence=False`だと未収束でも例外を投げず、
    `converged=False`の`ProbitResults`を返す（engine側のもう一方の分岐、
    APIレベルでの配線確認。例外を送出する既定挙動側は
    `test_probit_validation.py::test_non_convergence_raises_computation_error_
    with_tiny_max_iter`）。
    """
    res = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(max_iter=1, raise_on_non_convergence=False),
    ).fit()
    assert res.converged is False
    assert res.n_iter == 1


def test_cov_type_label(binary_dataset):
    """`res.cov_type`が指定した`cov_type`（正規化済み小文字）を反映すること
    （`test_logit_api.py`と同じ理由、Issue #231フェーズ4）。
    """
    for cov_type in ["classical", "opg", "hc0", "hc1"]:
        res = Probit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=ProbitOptions(cov_type=cov_type),
        ).fit()
        assert res.cov_type == cov_type

    res = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(cov_type="cluster", cluster_col="cluster"),
    ).fit()
    assert res.cov_type == "cluster"


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
    """`cov_type`が大文字小文字を区別しないこと（`test_logit_api.py`と同じ理由、
    Issue #231フェーズ4）。
    """
    kwargs = {"cluster_col": "cluster"} if cov_type == "CLUSTER" else {}
    options = ProbitOptions(cov_type=cov_type, **kwargs)
    res = Probit(binary_dataset, y="y", x=["x1", "x2"], options=options).fit()
    assert res.cov_type == expected_label


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_alias_for_classical(binary_dataset, cov_type):
    """`"nonrobust"`が`"classical"`と同じ計算方法（標準誤差も一致）のエイリアス
    であること（`test_logit_api.py`と同じ理由、Issue #231フェーズ4）。
    """
    res = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(cov_type=cov_type),
    ).fit()
    classical_res = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(cov_type="classical"),
    ).fit()
    for name in res.param_names:
        assert res.std_errors[name] == classical_res.std_errors[name], name


# ── predict() ────────────────────────────────────────────────────


def test_predict_returns_row_oriented_probabilities(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    predicted = res.predict()

    assert len(predicted) == binary_dataset.height
    for row in predicted:
        assert set(row.keys()) == {"probability"}
        assert 0.0 <= row["probability"] <= 1.0


# ── pred_table() ─────────────────────────────────────────────────


def test_pred_table_default_threshold_sums_to_n_obs(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    table = res.pred_table()

    assert len(table) == 2
    total = sum(row["predicted_0"] + row["predicted_1"] for row in table)
    assert total == binary_dataset.height
    assert {row["actual"] for row in table} == {0, 1}


def test_pred_table_actual_counts_invariant_to_threshold(binary_dataset):
    """`actual`の行合計はthresholdに関わらず一定（固定0.5分割のため）。"""
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    table_default = res.pred_table(0.5)
    table_other = res.pred_table(0.9)

    def row_totals(table):
        return {
            row["actual"]: row["predicted_0"] + row["predicted_1"]
            for row in table
        }

    assert row_totals(table_default) == row_totals(table_other)


# ── marginal_effects() ────────────────────────────────────────────


def test_marginal_effects_default_excludes_intercept(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    effects = res.marginal_effects()

    assert [row["param"] for row in effects] == ["x1", "x2"]
    expected_keys = {
        "param",
        "dydx",
        "std_err",
        "z",
        "p_value",
        "conf_low",
        "conf_high",
    }
    for row in effects:
        assert expected_keys <= set(row.keys())


def test_marginal_effects_mean_and_median_differ_from_overall(
    binary_dataset,
):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    overall = [row["dydx"] for row in res.marginal_effects(at="overall")]
    mean = [row["dydx"] for row in res.marginal_effects(at="mean")]
    median = [row["dydx"] for row in res.marginal_effects(at="median")]

    assert overall != mean
    assert overall != median


def test_marginal_effects_at_is_case_insensitive(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.marginal_effects(at="OVERALL") == res.marginal_effects(
        at="overall"
    )
