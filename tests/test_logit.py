"""Logit python_packageラッパーの構造・API・エラーパスのスモークテスト。

主リファレンス（statsmodels）との厳密な数値比較は別途実施する
（`test_ols_fixtures.py`/`test_wls_fixtures.py`と同じ役割分担）。ここでは
`fit()`の成功パス・`coef_table()`/`predict()`/`pred_table()`/
`marginal_effects()`の構造・`ValidationError`/`ComputationError`パスのみを
検証する。
"""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import separation_suspected_dataset
from econometricsmodels import (
    ComputationError,
    Logit,
    LogitOptions,
    LogitResults,
    ValidationError,
)

# binary_datasetフィクスチャ（`dataset`のyを中央値で0/1化）はconftest.pyで
# Logit/Probit共通定義。

# ── 成功パス・API構造 ────────────────────────────────────────────────


def test_fit_succeeds_and_returns_logit_results(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert isinstance(res, LogitResults)


def test_default_options_use_classical_and_converge(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.cov_type == "classical"
    assert res.converged


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_method_option_converges_to_same_params(binary_dataset, method):
    """`method`（newton/bfgs/lbfgs）はいずれも同じ最尤解に収束する。

    `engine/src/nonlinear/logit.rs`のRust単体テストは3手法の一致を検証済み
    だが、engine_pybindの文字列→`Method`パースやpython_packageラッパーの
    配線（例: "bfgs"と"lbfgs"の実装取り違え）を検出するAPIレベルのテストが
    無かったため追加した。
    """
    baseline = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    res = Logit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=LogitOptions(method=method),
    ).fit()
    assert res.converged
    for name in res.param_names:
        assert res.params[name] == pytest.approx(
            baseline.params[name], rel=1e-4
        )


def test_param_names_include_const_first(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.param_names == ["const", "x1", "x2"]


def test_params_std_errors_z_stats_p_values_share_keys(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    expected_keys = {"const", "x1", "x2"}
    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.z_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_conf_int_structure(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    ci = res.conf_int
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_n_obs_matches_dataset_size(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.n_obs == binary_dataset.height


def test_coef_table_structure(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
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


# ── predict() / pred_table() ─────────────────────────────────────────


def test_predict_returns_row_oriented_probabilities(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    predicted = res.predict()

    assert len(predicted) == binary_dataset.height
    for row in predicted:
        assert set(row.keys()) == {"probability"}
        assert 0.0 <= row["probability"] <= 1.0


def test_pred_table_default_threshold_sums_to_n_obs(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    table = res.pred_table()

    assert len(table) == 2
    total = sum(row["predicted_0"] + row["predicted_1"] for row in table)
    assert total == binary_dataset.height
    assert {row["actual"] for row in table} == {0, 1}


def test_pred_table_actual_counts_invariant_to_threshold(binary_dataset):
    """`actual`の行合計はthresholdに関わらず一定（固定0.5分割のため）。"""
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    table_default = res.pred_table(0.5)
    table_other = res.pred_table(0.9)

    def row_totals(table):
        return {
            row["actual"]: row["predicted_0"] + row["predicted_1"]
            for row in table
        }

    assert row_totals(table_default) == row_totals(table_other)


# ── marginal_effects() ────────────────────────────────────────────────


def test_marginal_effects_default_excludes_intercept(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
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
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    overall = [row["dydx"] for row in res.marginal_effects(at="overall")]
    mean = [row["dydx"] for row in res.marginal_effects(at="mean")]
    median = [row["dydx"] for row in res.marginal_effects(at="median")]

    assert overall != mean
    assert overall != median


def test_marginal_effects_at_is_case_insensitive(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.marginal_effects(at="OVERALL") == res.marginal_effects(
        at="overall"
    )


def test_marginal_effects_unknown_at_raises(binary_dataset):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(at="bogus")


def test_marginal_effects_confidence_level_out_of_range_raises(
    binary_dataset,
):
    res = Logit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(confidence_level=1.5)


# ── エラーハンドリング ──────────────────────────────────────────────


def test_y_in_x_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Logit(binary_dataset, y="y", x=["y", "x1"]).fit()


def test_duplicate_x_column_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Logit(binary_dataset, y="y", x=["x1", "x1"]).fit()


def test_const_collision_with_include_intercept_raises():
    df = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0, 1.0], "const": [1.0, 2.0, 3.0, 3.5]}
    )
    with pytest.raises(ValidationError):
        Logit(df, y="y", x=["const"]).fit()


def test_empty_x_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Logit(binary_dataset, y="y", x=[]).fit()


def test_missing_column_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Logit(binary_dataset, y="y", x=["does_not_exist"]).fit()


def test_unknown_cov_type_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Logit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=LogitOptions(cov_type="bogus"),
        ).fit()


def test_unknown_method_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Logit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=LogitOptions(method="bogus"),
        ).fit()


@pytest.mark.parametrize("tol", [0.0, -1.0])
def test_non_positive_tol_raises(binary_dataset, tol):
    """`tol<=0`は勾配ノルム基準の収束条件が理論上満たされないため`ValidationError`
    （engine側の`MleError::InvalidTol`）。
    """
    with pytest.raises(ValidationError):
        Logit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=LogitOptions(tol=tol),
        ).fit()


@pytest.mark.parametrize("bad_value", [0.5, 2.0, -1.0])
def test_non_binary_y_raises(binary_dataset, bad_value):
    """`y`が`{0.0, 1.0}`以外の値を含む場合は`ValidationError`
    （engine側の`MleError::InvalidBinaryY`）。
    """
    df = binary_dataset.with_columns(binary_dataset["y"].scatter(0, bad_value))
    with pytest.raises(ValidationError):
        Logit(df, y="y", x=["x1", "x2"]).fit()


def test_insufficient_observations_raises(binary_dataset):
    df = binary_dataset.head(2)
    with pytest.raises(ValidationError):
        Logit(df, y="y", x=["x1", "x2"]).fit()


def test_singular_hessian_raises_computation_error():
    """完全な多重共線性は`ComputationError`。"""
    df = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 4.0, 6.0, 8.0, 10.0],  # x2 = 2 * x1
        }
    )
    with pytest.raises(ComputationError):
        Logit(df, y="y", x=["x1", "x2"]).fit()


def test_non_convergence_raises_computation_error_with_tiny_max_iter(
    binary_dataset,
):
    """`max_iter`を人為的に1に絞ると`raise_on_non_convergence=True`（既定）で
    `ComputationError`（engine側の`NonConvergence`）。

    完全分離等の病理的なデータは`NonConvergence`ではなく専用の
    `SeparationSuspected`（`ComputationError`のサブタイプ、
    `test_separation_suspected_raises_computation_error_for_near_separation_data`
    参照）を返すため、`NonConvergence`自体の発生確認には使えない。そのため
    `NonConvergence`の発生確認は、専用データセットに頼らずmax_iterを
    人為的に小さくする方法で行う（`docs/spec/logit-spec.md`参照）。
    """
    with pytest.raises(ComputationError):
        Logit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=LogitOptions(max_iter=1),
        ).fit()


def test_separation_suspected_raises_computation_error_for_near_separation_data():
    """准完全分離データ（`x1`の真の係数を極端に大きくし、ほぼ全観測がx1の符号だけで
    完全に分類できるようにしたDGP）は`ComputationError`（engine側の
    `SeparationSuspected`）。

    勾配ノルム基準の収束判定が浮動小数点アンダーフローにより誤って「収束済み」と
    判定してしまう問題が、Python API境界を通しても正しく検出され
    エラーになることを確認する（`engine`側のRust単体テスト
    `fit_returns_separation_suspected_error_for_near_separation_data`
    のAPIレベル版）。
    """
    df = separation_suspected_dataset()

    with pytest.raises(ComputationError):
        Logit(df, y="y", x=["x1", "x2"]).fit()


def test_raise_on_non_convergence_false_returns_result_without_raising(
    binary_dataset,
):
    """`raise_on_non_convergence=False`だと未収束でも例外を投げず、
    `converged=False`の`LogitResults`を返す（engine側のもう一方の分岐、
    APIレベルでの配線確認）。
    """
    res = Logit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=LogitOptions(max_iter=1, raise_on_non_convergence=False),
    ).fit()
    assert res.converged is False
    assert res.n_iter == 1


def test_cluster_cov_type_requires_at_least_two_groups():
    """クラスター数が1つだけの場合`ValidationError`（engine側の`InsufficientClusters`）。"""
    df = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "cluster": ["a", "a", "a", "a"],
        }
    )
    with pytest.raises(ValidationError):
        Logit(
            df,
            y="y",
            x=["x1"],
            options=LogitOptions(cov_type="cluster", cluster_col="cluster"),
        ).fit()
