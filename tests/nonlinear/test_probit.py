"""Probit python_packageラッパーの構造・API・エラーパスのスモークテスト。

主リファレンス（statsmodels）との厳密な数値比較は別途実施する
（`test_ols_fixtures.py`/`test_wls_fixtures.py`/`test_logit_fixtures.py`と
同じ役割分担）。ここでは`fit()`の成功パス・`coef_table()`/`predict()`/
`pred_table()`/`marginal_effects()`の構造・`ValidationError`/
`ComputationError`パスのみを検証する（`test_logit.py`と同型）。
"""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import separation_suspected_dataset
from econometricsmodels import (
    ComputationError,
    Probit,
    ProbitOptions,
    ProbitResults,
    ValidationError,
)

# binary_datasetフィクスチャ（`dataset`のyを中央値で0/1化）はconftest.pyで
# Logit/Probit共通定義。

# ── 成功パス・API構造 ────────────────────────────────────────────────


def test_fit_succeeds_and_returns_probit_results(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert isinstance(res, ProbitResults)


def test_default_options_use_classical_and_converge(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.cov_type == "classical"
    assert res.converged


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_method_option_converges_to_same_params(binary_dataset, method):
    """`method`（newton/bfgs/lbfgs）はいずれも同じ最尤解に収束する。

    `engine/src/nonlinear/probit.rs`のRust単体テストは3手法の一致を検証済み
    だが、engine_pybindの文字列→`Method`パースやpython_packageラッパーの
    配線を検出するAPIレベルのテストが無かったため追加した（`test_logit.py`と
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


def test_param_names_include_const_first(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.param_names == ["const", "x1", "x2"]


def test_include_intercept_false_omits_const_and_converges(binary_dataset):
    """`include_intercept=False`の構造面での成功パス
    （`test_logit.py`と同じ理由、Issue #231フェーズ4）。
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


def test_params_std_errors_z_stats_p_values_share_keys(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    expected_keys = {"const", "x1", "x2"}
    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.z_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_conf_int_structure(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    ci = res.conf_int
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_n_obs_matches_dataset_size(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.n_obs == binary_dataset.height


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


# ── predict() / pred_table() ─────────────────────────────────────────


def test_predict_returns_row_oriented_probabilities(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    predicted = res.predict()

    assert len(predicted) == binary_dataset.height
    for row in predicted:
        assert set(row.keys()) == {"probability"}
        assert 0.0 <= row["probability"] <= 1.0


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


# ── marginal_effects() ────────────────────────────────────────────────


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


def test_marginal_effects_unknown_at_raises(binary_dataset):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(at="bogus")


def test_marginal_effects_confidence_level_out_of_range_raises(
    binary_dataset,
):
    res = Probit(binary_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(confidence_level=1.5)


# ── エラーハンドリング ──────────────────────────────────────────────


def test_y_in_x_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Probit(binary_dataset, y="y", x=["y", "x1"]).fit()


def test_duplicate_x_column_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Probit(binary_dataset, y="y", x=["x1", "x1"]).fit()


def test_const_collision_with_include_intercept_raises():
    df = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0, 1.0], "const": [1.0, 2.0, 3.0, 3.5]}
    )
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["const"]).fit()


def test_empty_x_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Probit(binary_dataset, y="y", x=[]).fit()


def test_missing_column_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Probit(binary_dataset, y="y", x=["does_not_exist"]).fit()


def test_null_values_raise():
    """欠損値は`column_extraction`の責務で`ValidationError`
    （`test_logit.py`と同じ理由、Issue #231フェーズ4）。
    """
    df = pl.DataFrame({"y": [0.0, None, 1.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["x1"]).fit()


def test_non_numeric_dtype_raises():
    """数値/文字列型にキャストできない列は`ValidationError`
    （`test_logit.py`と同じ理由、Issue #231フェーズ4）。
    """
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["x1"]).fit()


def test_unknown_cov_type_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Probit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=ProbitOptions(cov_type="bogus"),
        ).fit()


def test_unknown_method_raises(binary_dataset):
    with pytest.raises(ValidationError):
        Probit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=ProbitOptions(method="bogus"),
        ).fit()


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(binary_dataset, confidence_level):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合`ValidationError`
    （`test_logit.py`と同じ理由、Issue #231フェーズ4）。
    """
    options = ProbitOptions(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        Probit(binary_dataset, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize("tol", [0.0, -1.0])
def test_non_positive_tol_raises(binary_dataset, tol):
    """`tol<=0`は勾配ノルム基準の収束条件が理論上満たされないため`ValidationError`
    （engine側の`MleError::InvalidTol`、`test_logit.py`と同じ検証）。
    """
    with pytest.raises(ValidationError):
        Probit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=ProbitOptions(tol=tol),
        ).fit()


@pytest.mark.parametrize("bad_value", [0.5, 2.0, -1.0])
def test_non_binary_y_raises(binary_dataset, bad_value):
    """`y`が`{0.0, 1.0}`以外の値を含む場合は`ValidationError`
    （engine側の`MleError::InvalidBinaryY`、`test_logit.py`と同じ検証）。
    """
    df = binary_dataset.with_columns(binary_dataset["y"].scatter(0, bad_value))
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["x1", "x2"]).fit()


def test_insufficient_observations_raises(binary_dataset):
    df = binary_dataset.head(2)
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["x1", "x2"]).fit()


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_singular_hessian_raises_computation_error(method):
    """完全な多重共線性は`ComputationError`。

    `method`のparametrize理由は`test_logit.py`と同じ（`bfgs`/`lbfgs`は
    `newton_step`を経由しないため特異性検出の経路が異なる、Issue #231
    フェーズ4）。
    """
    df = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 4.0, 6.0, 8.0, 10.0],  # x2 = 2 * x1
        }
    )
    with pytest.raises(ComputationError):
        Probit(
            df, y="y", x=["x1", "x2"], options=ProbitOptions(method=method)
        ).fit()


def test_non_convergence_raises_computation_error_with_tiny_max_iter(
    binary_dataset,
):
    """`max_iter`を人為的に1に絞ると`raise_on_non_convergence=True`（既定）で
    `ComputationError`（engine側の`NonConvergence`、`test_logit.py`と同じ理由で
    専用データセットではなくmax_iterを小さくする方法を使う）。
    """
    with pytest.raises(ComputationError):
        Probit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=ProbitOptions(max_iter=1),
        ).fit()


def test_separation_suspected_raises_computation_error_for_near_separation_data():
    """准完全分離データ（`x1`の真の係数を極端に大きくし、ほぼ全観測がx1の符号だけで
    完全に分類できるようにしたDGP）は`ComputationError`（engine側の
    `SeparationSuspected`）。`test_logit.py`のLogit版のProbit版。
    """
    df = separation_suspected_dataset()

    with pytest.raises(ComputationError):
        Probit(df, y="y", x=["x1", "x2"]).fit()


def test_raise_on_non_convergence_false_returns_result_without_raising(
    binary_dataset,
):
    """`raise_on_non_convergence=False`だと未収束でも例外を投げず、
    `converged=False`の`ProbitResults`を返す（engine側のもう一方の分岐、
    APIレベルでの配線確認）。
    """
    res = Probit(
        binary_dataset,
        y="y",
        x=["x1", "x2"],
        options=ProbitOptions(max_iter=1, raise_on_non_convergence=False),
    ).fit()
    assert res.converged is False
    assert res.n_iter == 1


def test_confidence_level_changes_interval_width(binary_dataset):
    """`confidence_level`を下げると信頼区間が狭くなること（`test_logit.py`と
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


@pytest.mark.parametrize("max_iter", [0, -1])
def test_non_positive_max_iter_raises(binary_dataset, max_iter):
    """`max_iter<=0`は`ValidationError`（`test_logit.py`と同じ理由、
    Issue #231フェーズ4）。
    """
    with pytest.raises(ValidationError):
        Probit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=ProbitOptions(max_iter=max_iter),
        ).fit()


def test_cov_type_label(binary_dataset):
    """`res.cov_type`が指定した`cov_type`（正規化済み小文字）を反映すること
    （`test_logit.py`と同じ理由、Issue #231フェーズ4）。
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
    """`cov_type`が大文字小文字を区別しないこと（`test_logit.py`と同じ理由、
    Issue #231フェーズ4）。
    """
    kwargs = {"cluster_col": "cluster"} if cov_type == "CLUSTER" else {}
    options = ProbitOptions(cov_type=cov_type, **kwargs)
    res = Probit(binary_dataset, y="y", x=["x1", "x2"], options=options).fit()
    assert res.cov_type == expected_label


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_alias_for_classical(binary_dataset, cov_type):
    """`"nonrobust"`が`"classical"`と同じ計算方法（標準誤差も一致）のエイリアス
    であること（`test_logit.py`と同じ理由、Issue #231フェーズ4）。
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
        Probit(
            df,
            y="y",
            x=["x1"],
            options=ProbitOptions(cov_type="cluster", cluster_col="cluster"),
        ).fit()


def test_cluster_col_nonexistent_column_raises(binary_dataset):
    """`cluster_col`が実在しない列名を指すと`ValidationError`（`test_logit.py`と
    同じ理由、Issue #231フェーズ4）。
    """
    options = ProbitOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(ValidationError):
        Probit(binary_dataset, y="y", x=["x1", "x2"], options=options).fit()
