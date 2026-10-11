"""Logit/Probitの`test_logit_*.py`/`test_probit_*.py`で重複していた
テスト本体を関数として集約する。

`tests/linear/_ols_helpers.py`と同じ仕組み（pytestが各テストファイルの
ディレクトリを`sys.path`に載せるrootless import）で、`tests/nonlinear/`配下
から`from _binary_choice_checks import ...`の裸importで解決できる。

`Logit`/`Probit`のAPI（`fit()`の引数・`Options`のフィールド構成・`Results`の
プロパティ）が完全に対称に設計されている結果、テストコードもほぼ全て重複していた
（`test_logit_api.py`/`test_probit_api.py`はdocstring以外の差分ゼロ、
`test_logit_validation.py`/`test_probit_validation.py`も同様、
`test_logit_reference.py`/`test_probit_reference.py`はコード構造は同一で
手法ごとの定数値のみ異なる）。ファイル自体は手法ごとに残し（各`test_logit_*.py`/
`test_probit_*.py`を開けばそのファイルの全テストが一覧できる利点を維持するため）、
`Estimator`/`Options`/`Results`クラスを引数で受け取る関数としてテスト本体のみ
ここに切り出す。呼び出し側は同じ関数名・parametrize・fixture引数を持つ薄い
ラッパーになる。

`test_logit_crosscheck.py`/`test_probit_crosscheck.py`はProbit固有の観測情報
行列の説明等、実質的な差分があるため対象外（項目95参照）。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import _error_messages as msgs
import pandas as pd
import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close, check_margeff
from _assertions import rename_intercept as _rename
from _constants import APPLE_CLUSTER, APPLE_X, DATA_DIR, MROZ_X
from _error_messages import escaped
from _helpers import (
    load_apple_adoption_dataset,
    load_wooldridge_dataset,
    multi_dummy_separation_dataset,
    separated_dataset,
    single_overlap_binary_dataset,
    small_sample_separation_dataset,
    strongly_collinear_binary_dataset,
    with_cluster_groups,
)
from econometricsmodels import ComputationError, ValidationError

from benchmark.common import imbalanced_cluster_groups

# ── test_<solver>_api.py: 成功パス・結果型 ──────────────────────────


def check_fit_succeeds_and_returns_results(
    dataset, estimator_cls, results_cls
):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert isinstance(res, results_cls)


def check_default_options_use_classical_and_converge(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.cov_type == "classical"
    assert res.converged


# ── test_<solver>_api.py: API構造 ───────────────────────────────────


def check_params_table_structure(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    table = res.params_table()

    assert isinstance(table, list)
    assert len(table) == 3  # const, x1, x2
    expected_keys = {
        "param_name",
        "param",
        "std_error",
        "test_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert expected_keys <= set(row.keys())
    assert [row["param_name"] for row in table] == ["const", "x1", "x2"]


def check_conf_int_structure(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    ci = res.conf_int
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def check_params_std_errors_test_stats_p_values_share_keys(
    dataset, estimator_cls
):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    expected_keys = {"const", "x1", "x2"}
    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.test_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def check_n_obs_matches_dataset_size(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.n_obs == dataset.height


def check_dep_var_name(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.dep_var_name == "y"


def check_param_names_include_const_first(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.param_names == ["const", "x1", "x2"]


# ── test_<solver>_api.py: オプションの反映 ──────────────────────────
#
# cov_type 以外の Options フィールド（solver・include_intercept・
# confidence_level・raise_on_non_convergence）が、engine_pybind 側の
# 文字列パース・列抽出・分岐ロジックを経て正しく反映されることを確認する。


def check_method_option_converges_to_same_params(
    dataset, estimator_cls, options_cls, solver
):
    """`solver`（newton/bfgs/lbfgs）はいずれも同じ最尤解に収束する。

    engineのRust単体テストは3手法の一致を検証済みだが、engine_pybindの
    文字列→`Method`パースやpython_packageラッパーの配線（例:
    "bfgs"と"lbfgs"の実装取り違え）を検出するAPIレベルのテストが無かった
    ため追加した。
    """
    baseline = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    res = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(solver=solver),
    ).fit()
    assert res.converged
    for name in res.param_names:
        assert res.params[name] == pytest.approx(
            baseline.params[name], rel=1e-4
        )


def check_include_intercept_false_omits_const_and_converges(
    dataset, estimator_cls, options_cls
):
    """`include_intercept=False`の構造面での成功パス（数値照合は
    `test_<method>_reference.py::test_include_intercept_false_matches_statsmodels`）。

    `include_intercept`の値に関わらず`df_model`は常に`k-1`（`docs/spec/
    <solver>-spec.md`参照）となるため、その旨も確認する
    （`testing-completeness-reviewer`指摘）。
    """
    res = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(include_intercept=False),
    ).fit()
    assert res.param_names == ["x1", "x2"]
    assert res.converged
    assert res.df_model == 1


def check_confidence_level_changes_interval_width(
    dataset, estimator_cls, options_cls
):
    """`confidence_level`を下げると信頼区間が狭くなること（既定の0.95以外の
    値が`engine_pybind`経由で実際に反映されることの確認、OLSの
    `test_confidence_level_changes_interval_width`と同型）。
    """
    wide = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(confidence_level=0.99),
    ).fit()
    narrow = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(confidence_level=0.80),
    ).fit()

    for name in ["const", "x1", "x2"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width


def check_raise_on_non_convergence_false_returns_result_without_raising(
    dataset, estimator_cls, options_cls, cov_type="classical"
):
    """`raise_on_non_convergence=False`だと未収束でも例外を投げず、
    `converged=False`の`Results`を返す（engine側のもう一方の分岐、APIレベル
    での配線確認。例外を送出する既定挙動側は
    `test_<solver>_validation.py::test_non_convergence_raises_computation_error_with_tiny_max_iter`）。

    `cov_type`は`classical`以外（`opg`/`hc0`/`hc1`/`cluster`）も検証する。
    打ち切り点（収束未満のパラメータ）でのHessian/スコア評価はcov_typeの
    分岐によって経由する行列演算が異なるため、想定外の例外を投げず、
    標準誤差が有限値であることまで確認する。
    """
    kwargs = {
        "max_iter": 1,
        "raise_on_non_convergence": False,
        "cov_type": cov_type,
    }
    if cov_type == "cluster":
        kwargs["cluster"] = "cluster"
    res = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(**kwargs),
    ).fit()
    assert res.converged is False
    assert res.n_iter == 1
    for name, se in res.std_errors.items():
        assert math.isfinite(se), f"std_errors[{name}]={se} is not finite"


def check_cov_type_label(dataset, estimator_cls, options_cls):
    """`res.cov_type`が指定した`cov_type`（正規化済み小文字）を反映すること
    （OLSの`test_cov_type_label`と同型）。
    """
    for cov_type in ["classical", "opg", "hc0", "hc1"]:
        res = estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(cov_type=cov_type),
        ).fit()
        assert res.cov_type == cov_type

    res = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(cov_type="cluster", cluster="cluster"),
    ).fit()
    assert res.cov_type == "cluster"


def check_method_label(dataset, estimator_cls, options_cls):
    """`res.solver`が指定した`solver`（正規化済み小文字）を反映すること
    （`check_cov_type_label`と同型）。
    """
    for solver in ["newton", "bfgs", "lbfgs"]:
        res = estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(solver=solver),
        ).fit()
        assert res.solver == solver


def check_method_is_case_insensitive(
    dataset, estimator_cls, options_cls, solver, expected_label
):
    """`solver`が大文字小文字を区別しないこと（`check_cov_type_is_case_insensitive`
    と同型）。
    """
    options = options_cls(solver=solver)
    res = estimator_cls(dataset, y="y", x=["x1", "x2"], options=options).fit()
    assert res.solver == expected_label


def check_cov_type_is_case_insensitive(
    dataset, estimator_cls, options_cls, cov_type, expected_label
):
    """`cov_type`が大文字小文字を区別しないこと（`engine_pybind`側の
    `build_<solver>_input`のRust単体テストと対になる、Python API境界での
    確認。OLS/WLSの`test_cov_type_is_case_insensitive`と同型）。
    """
    kwargs = {"cluster": "cluster"} if cov_type == "CLUSTER" else {}
    options = options_cls(cov_type=cov_type, **kwargs)
    res = estimator_cls(dataset, y="y", x=["x1", "x2"], options=options).fit()
    assert res.cov_type == expected_label


def check_nonrobust_is_rejected(dataset, estimator_cls, options_cls, cov_type):
    """`"nonrobust"`（旧別名）は受け付けない（概念ごとに文字列を1つに絞る）。"""
    with pytest.raises(ValidationError, match="unknown cov_type: 'nonrobust'"):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(cov_type=cov_type),
        ).fit()


# ── test_<solver>_api.py: predict() ─────────────────────────────────


def check_predict_returns_probabilities(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    predicted = res.predict()

    assert isinstance(predicted, list)
    assert len(predicted) == dataset.height
    for prob in predicted:
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


def check_predict_new_data_returns_probabilities(dataset, estimator_cls):
    """`predict(new_data)`（out-of-sample）が学習データと構造の
    異なる新規データに対しても同じ形状（確率の`list[float]`）を返すこと。
    """
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    predicted = res.predict(new_data)

    assert isinstance(predicted, list)
    assert len(predicted) == 2
    for prob in predicted:
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


# ── test_<solver>_api.py: augment() ─────────────────────────────────


def check_augment_none_returns_training_data_with_probability_column(
    dataset, estimator_cls
):
    """`augment(new_data=None)`が、学習データの全列＋`"predicted_probability"`列を
    持つDataFrameを、`predict()`と同じ予測確率・元データと同じ行順で
    返すこと。
    """
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()

    augmented = res.augment()

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == dataset.height
    assert augmented.columns == [*dataset.columns, "predicted_probability"]
    for col in dataset.columns:
        assert augmented[col].to_list() == dataset[col].to_list()

    expected = res.predict()
    assert augmented["predicted_probability"].to_list() == expected


def check_augment_new_data_returns_new_data_with_probability_column(
    dataset, estimator_cls
):
    """`augment(new_data)`が、`new_data`の全列＋`"predicted_probability"`列を持つ
    DataFrameを、`predict(new_data)`と同じ予測確率で返すこと。
    """
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    augmented = res.augment(new_data)

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == 2
    assert augmented.columns == ["x1", "x2", "predicted_probability"]

    expected = res.predict(new_data)
    assert augmented["predicted_probability"].to_list() == expected


def check_augment_without_intercept_matches_predict(
    estimator_cls, options_cls
) -> None:
    """`include_intercept=False`でfitした場合も`augment()`が`predict()`と
    同じ予測確率を返すこと（`augment()`はRust側で`predict()`とは別に
    `has_intercept`分岐を実装しているため、個別に確認する。OLSの
    `test_augment_without_intercept_matches_predict`と同型、
    python-reviewer指摘）。
    """
    df = pl.DataFrame(
        {
            "y": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    options = options_cls(include_intercept=False)
    res = estimator_cls(df, y="y", x=["x1"], options=options).fit()

    augmented_none = res.augment()
    expected_none = res.predict()
    assert augmented_none["predicted_probability"].to_list() == expected_none

    new_data = pl.DataFrame({"x1": [10.0, 20.0]})
    augmented_new = res.augment(new_data)
    expected_new = res.predict(new_data)
    assert augmented_new["predicted_probability"].to_list() == expected_new


# ── test_<solver>_api.py: pred_table() ──────────────────────────────


def check_pred_table_default_threshold_sums_to_n_obs(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    table = res.pred_table()

    assert len(table) == 2
    total = sum(row["predicted_0"] + row["predicted_1"] for row in table)
    assert total == dataset.height
    assert {row["actual"] for row in table} == {0, 1}


def check_pred_table_actual_counts_invariant_to_threshold(
    dataset, estimator_cls
):
    """`actual`の行合計はthresholdに関わらず一定（固定0.5分割のため）。"""
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    table_default = res.pred_table(0.5)
    table_other = res.pred_table(0.9)

    def row_totals(table):
        return {
            row["actual"]: row["predicted_0"] + row["predicted_1"]
            for row in table
        }

    assert row_totals(table_default) == row_totals(table_other)


# ── test_<solver>_api.py: marginal_effects() ────────────────────────


def check_marginal_effects_default_excludes_intercept(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    effects = res.marginal_effects()

    assert [row["param_name"] for row in effects] == ["x1", "x2"]
    expected_keys = {
        "param_name",
        "effect",
        "std_error",
        "test_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in effects:
        assert expected_keys <= set(row.keys())


def check_marginal_effects_mean_and_median_differ_from_overall(
    dataset, estimator_cls
):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    overall = [row["effect"] for row in res.marginal_effects(at="overall")]
    mean = [row["effect"] for row in res.marginal_effects(at="mean")]
    median = [row["effect"] for row in res.marginal_effects(at="median")]

    assert overall != mean
    assert overall != median


def check_marginal_effects_at_is_case_insensitive(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.marginal_effects(at="OVERALL") == res.marginal_effects(
        at="overall"
    )


# ── test_<solver>_validation.py: ValidationError（入力データ） ──────


def check_y_in_x_raises(dataset, estimator_cls):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="y",
            single_role="y",
            multi_role="x",
        ),
    ):
        estimator_cls(dataset, y="y", x=["y", "x1"]).fit()


def check_y_empty_string_raises(dataset, estimator_cls):
    """`y`に空文字列を渡した場合`ValidationError`（`x`側の`check_empty_x_raises`と
    対称。`test_ols_validation.py::test_y_empty_string_raises`参照）。
    """
    with pytest.raises(
        ValidationError, match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="")
    ):
        estimator_cls(dataset, y="", x=["x1", "x2"]).fit()


def check_duplicate_x_column_raises(dataset, estimator_cls):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.DUPLICATE_WITHIN_ROLE, name="x1", role="x"),
    ):
        estimator_cls(dataset, y="y", x=["x1", "x1"]).fit()


def check_const_collision_with_include_intercept_raises(estimator_cls):
    df = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0, 1.0], "const": [1.0, 2.0, 3.0, 3.5]}
    )
    with pytest.raises(
        ValidationError, match=escaped(msgs.CONST_COLLISION, role="x")
    ):
        estimator_cls(df, y="y", x=["const"]).fit()


def check_empty_x_raises(dataset, estimator_cls):
    with pytest.raises(ValidationError, match=escaped(msgs.X_EMPTY, role="x")):
        estimator_cls(dataset, y="y", x=[]).fit()


def check_missing_column_raises(dataset, estimator_cls):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="does_not_exist"),
    ):
        estimator_cls(dataset, y="y", x=["does_not_exist"]).fit()


def check_data_not_polars_raises(dataset, estimator_cls):
    """`data`/`new_data`にpolars以外のDataFrame（pandas等）を渡すと、内部実装
    （`pyo3-polars`の`get_columns`呼び出し）が漏れた`AttributeError`ではなく
    `ValidationError`になること（`test_ols_validation.py`と同じ検証）。
    """
    bad = pd.DataFrame({"y": [0.0, 1.0, 0.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="data",
            type_name=msgs.fully_qualified_type_name(bad),
        ),
    ):
        estimator_cls(bad, y="y", x=["x1"]).fit()

    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    bad_new_data = pd.DataFrame({"x1": [1.0], "x2": [0.5]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="new_data",
            type_name=msgs.fully_qualified_type_name(bad_new_data),
        ),
    ):
        res.predict(bad_new_data)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="new_data",
            type_name=msgs.fully_qualified_type_name(bad_new_data),
        ),
    ):
        res.augment(bad_new_data)


def check_null_values_raise(estimator_cls):
    """欠損値は`column_extraction`の責務で`ValidationError`（OLSの
    `test_null_values_raise`と同型、Python API境界で未検証だった）。
    """
    df_y = pl.DataFrame({"y": [0.0, None, 1.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="y", count=1),
    ):
        estimator_cls(df_y, y="y", x=["x1"]).fit()

    df_x = pl.DataFrame({"y": [0.0, 1.0, 0.0], "x1": [1.0, None, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="x1", count=1),
    ):
        estimator_cls(df_x, y="y", x=["x1"]).fit()


def check_non_finite_values_raise(estimator_cls):
    """`y`/`x`にNaN・無限大が含まれる場合`ValidationError`
    （OLSの`test_non_finite_values_raise`と同型。null（`check_null_values_raise`）
    とNaN/Inf は`column_extraction.rs`内で別ロジックのため個別に確認する。
    `predict()`側は`check_predict_null_or_non_finite_values_raise`が`x1`を
    カバー済み）。
    """
    df_y_nan = pl.DataFrame(
        {"y": [0.0, float("nan"), 1.0], "x1": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="y", value="NaN", row=1
        ),
    ):
        estimator_cls(df_y_nan, y="y", x=["x1"]).fit()

    df_y_inf = pl.DataFrame(
        {"y": [0.0, float("inf"), 1.0], "x1": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="y", value="inf", row=1
        ),
    ):
        estimator_cls(df_y_inf, y="y", x=["x1"]).fit()

    df_x_nan = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0], "x1": [1.0, float("nan"), 3.0]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="x1", value="NaN", row=1
        ),
    ):
        estimator_cls(df_x_nan, y="y", x=["x1"]).fit()

    df_x_inf = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0], "x1": [1.0, float("inf"), 3.0]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="x1", value="inf", row=1
        ),
    ):
        estimator_cls(df_x_inf, y="y", x=["x1"]).fit()


def check_non_numeric_dtype_raises(estimator_cls):
    """文字列列は、dtypeの時点で`ValidationError`（OLSの
    `test_non_numeric_dtype_raises`と同型）。
    """
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="y", dtype="String"
        ),
    ):
        estimator_cls(df, y="y", x=["x1"]).fit()


def check_non_binary_y_raises(dataset, estimator_cls, bad_value):
    """`y`が`{0.0, 1.0}`以外の値を含む場合は`ValidationError`
    （engine側の`MleError::InvalidBinaryY`）。
    """
    df = dataset.with_columns(dataset["y"].scatter(0, bad_value))
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_BINARY_Y, row=0, value=msgs.rust_f64(bad_value)
        ),
    ):
        estimator_cls(df, y="y", x=["x1", "x2"]).fit()


def check_insufficient_observations_raises(dataset, estimator_cls):
    df = dataset.head(2)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_OBSERVATIONS, n=2, k=3),
    ):
        estimator_cls(df, y="y", x=["x1", "x2"]).fit()


# ── test_<solver>_validation.py: ValidationError（predict()のnew_data） ──
#
# OLSの`test_predict_missing_column_raises`等と同型。


def check_predict_missing_column_raises(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0]})  # x2が無い

    with pytest.raises(
        ValidationError, match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="x2")
    ):
        res.predict(new_data)


def check_predict_non_numeric_dtype_raises(dataset, estimator_cls):
    """`check_non_numeric_dtype_raises`と同じdtypeのメッセージになる。"""
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": ["a", "b"], "x2": [1.0, 2.0]})

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="x1", dtype="String"
        ),
    ):
        res.predict(new_data)


def check_predict_null_or_non_finite_values_raise(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()

    new_data_null = pl.DataFrame({"x1": [1.0, None], "x2": [1.0, 2.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="x1", count=1),
    ):
        res.predict(new_data_null)

    new_data_inf = pl.DataFrame({"x1": [1.0, float("inf")], "x2": [1.0, 2.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="x1", value="inf", row=1
        ),
    ):
        res.predict(new_data_inf)


def check_augment_column_collision_raises(dataset, estimator_cls):
    """元データ（`new_data=None`）・`new_data`のいずれかに既に`"predicted_probability"`
    列がある場合`ValidationError`（黙って上書きしない、
    OLSの`test_augment_column_collision_raises`と同型）。
    """
    df_with_probability = dataset.with_columns(
        pl.lit(0.0).alias("predicted_probability")
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.EXISTING_COLUMN_COLLISION, name="predicted_probability"
        ),
    ):
        estimator_cls(
            df_with_probability, y="y", x=["x1", "x2"]
        ).fit().augment()

    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame(
        {"x1": [1.0], "x2": [0.5], "predicted_probability": [0.0]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.EXISTING_COLUMN_COLLISION, name="predicted_probability"
        ),
    ):
        res.augment(new_data)


def check_augment_missing_column_raises(dataset, estimator_cls):
    """`augment()`も`predict()`と同じ`extract_f64_column`経路を通るため、
    `new_data`に`x`列が欠けていると`ValidationError`。
    """
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0]})  # x2が無い

    with pytest.raises(
        ValidationError, match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="x2")
    ):
        res.augment(new_data)


# ── test_<solver>_validation.py: ValidationError（オプション） ─────


def check_unknown_cov_type_raises(
    dataset, estimator_cls, options_cls, cov_type="bogus"
):
    """未知の`cov_type`（空文字列を含む）は`ValidationError`
    （テスト網羅性候補・項目46）。
    """
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_COV_TYPE_NONLINEAR, other=cov_type),
    ):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(cov_type=cov_type),
        ).fit()


def check_unknown_method_raises(
    dataset, estimator_cls, options_cls, solver="bogus"
):
    """未知の`solver`（空文字列を含む）は`ValidationError`
    （テスト網羅性候補・項目46）。
    """
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_METHOD_NONLINEAR, other=solver),
    ):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(solver=solver),
        ).fit()


def check_invalid_confidence_level_raises(
    dataset, estimator_cls, options_cls, confidence_level
):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合
    `ValidationError`。

    `marginal_effects(confidence_level=1.5)`側は
    `check_marginal_effects_confidence_level_out_of_range_raises`で既存
    だが、`fit()`本体側（`Options.confidence_level`）が未検証だった
    （`testing-policy.md`「テストの3系統」・OLS/WLSの
    `test_invalid_confidence_level_raises`との非対称）。
    """
    options = options_cls(confidence_level=confidence_level)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_CONFIDENCE_LEVEL,
            confidence_level=msgs.rust_f64(confidence_level),
        ),
    ):
        estimator_cls(dataset, y="y", x=["x1", "x2"], options=options).fit()


def check_non_positive_tol_raises(dataset, estimator_cls, options_cls, tol):
    """`tol<=0`は勾配ノルム基準の収束条件が理論上満たされないため
    `ValidationError`（engine側の`MleError::InvalidTol`）。
    """
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INVALID_TOL, tol=msgs.rust_f64(tol)),
    ):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(tol=tol),
        ).fit()


def check_non_positive_max_iter_raises(
    dataset, estimator_cls, options_cls, max_iter
):
    """`max_iter<=0`は`ValidationError`（engine側の`MleError::InvalidMaxIter`）。

    `tol<=0`側は`check_non_positive_tol_raises`で既存だが、対応する
    `max_iter`側のPython API境界のテストが無かった
    （`testing-completeness-reviewer`指摘）。
    """
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INVALID_MAX_ITER, max_iter=max_iter),
    ):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(max_iter=max_iter),
        ).fit()


def check_cluster_cov_type_requires_at_least_two_groups(
    estimator_cls, options_cls
):
    """クラスター数が1つだけの場合`ValidationError`
    （engine側の`InsufficientClusters`）。
    """
    df = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "cluster": ["a", "a", "a", "a"],
        }
    )
    with pytest.raises(
        ValidationError, match=escaped(msgs.INSUFFICIENT_CLUSTERS, g=1)
    ):
        estimator_cls(
            df,
            y="y",
            x=["x1"],
            options=options_cls(cov_type="cluster", cluster="cluster"),
        ).fit()


def check_cluster_count_at_most_slopes_raises_validation_error(
    binary_dataset, estimator_cls, options_cls
):
    """クラスター数G≤傾き係数の数q（`k - k_constant`）は`ValidationError`
    （engine側の`CommonError::InsufficientClustersForInference`）。

    クラスターロバスト共分散はクラスター寄与スコアの総和がゼロ（MLEの一次条件
    `Σ_i s_i = 0`）で`rank(Ŝ)≤G-1`のため、G≤qだと退化する。Logit/Probitは
    overall検定がLRのため`q×q`反転こそ通らないが、縮退した共分散から読んだSEを
    無警告で返すのは実質バグのため、`fit()`冒頭のバリデーションで弾く
    （Logit/Probitでは新規制約）。`x=["x1", "x2"]`で`q=2`、疑似グループを2個に
    潰して`G=2 == q=2`を作る。
    """
    df = with_cluster_groups(binary_dataset, 2)
    options = options_cls(cov_type="cluster", cluster="cluster_group")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=2, q=2),
    ):
        estimator_cls(df, y="y", x=["x1", "x2"], options=options).fit()


def check_cluster_without_col_raises(dataset, estimator_cls, options_cls):
    """`cov_type="cluster"`なのに`cluster`未指定の場合`ValidationError`
    （OLS/WLS/IVと同じ検証、共通化された経路）。
    """
    options = options_cls(cov_type="cluster")
    with pytest.raises(
        ValidationError, match=escaped(msgs.MISSING_CLUSTER_COLUMN)
    ):
        estimator_cls(dataset, y="y", x=["x1", "x2"], options=options).fit()


def check_cluster_nonexistent_column_raises(
    dataset, estimator_cls, options_cls
):
    """`cluster`が実在しない列名を指すと`ValidationError`（OLSと同じ理由）。"""
    options = options_cls(cov_type="cluster", cluster="does_not_exist")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="does_not_exist"),
    ):
        estimator_cls(dataset, y="y", x=["x1", "x2"], options=options).fit()


# ── test_<solver>_validation.py: ValidationError（marginal_effects()） ─


def check_marginal_effects_unknown_at_raises(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_MARGINAL_EFFECTS_AT, other="bogus"),
    ):
        res.marginal_effects(at="bogus")


def check_marginal_effects_confidence_level_out_of_range_raises(
    dataset, estimator_cls
):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_CONFIDENCE_LEVEL,
            confidence_level=msgs.rust_f64(1.5),
        ),
    ):
        res.marginal_effects(confidence_level=1.5)


# ── test_<solver>_validation.py: ComputationError ───────────────────


def check_perfect_multicollinearity_raises_computation_error(
    estimator_cls, dataset_prefix, options_cls, solver
):
    """完全な多重共線性（合成データセット）は数値比較の対象外
    （`testing-policy.md`「テストの3系統」）。想定エラー（`ComputationError`）が
    発生することのみを確認する。`solver`（newton/bfgs/lbfgs）でparametrizeする。

    以前は、この`solver`網羅を`check_singular_hessian_raises_computation_error`
    （インラインの極小データ、`x2=2*x1`直書き）が担っていた。`newton`は
    `newton_step`内のQR、`bfgs`/`lbfgs`は収束後の`observed_information_cov_params`と
    いう`solver`依存の別経路で特異性を検出しており、過去に`bfgs`だけ検出漏れした
    実バグの回帰ガードだった。その後`engine`内の検出経路は`fit()`冒頭の列ピボットQR
    ランクチェック（`solver`非依存の単一経路、`SingularDesignMatrix`）に一本化された
    が、**`engine_pybind`側のsolver文字列パース（`"bfgs"`/`"lbfgs"` → `EngineSolverType`）
    と配線はsolver固有のまま**なので、「非既定solverの文字列 × 特異入力 ×
    `ComputationError`」を踏むAPI境界テストは引き続き必要（testing-completeness-
    reviewer指摘）。インラインの極小データはCSVフィクスチャ版へ統合した
    （`engine`側の`solver`×`cov_type`網羅は
    `fit_returns_singular_design_matrix_error_for_perfectly_collinear_design_matrix`
    1本に集約）。
    """
    df = pl.read_csv(
        DATA_DIR / f"{dataset_prefix}_perfect_multicollinearity.csv"
    )
    with pytest.raises(ComputationError):
        estimator_cls(
            df,
            y="y",
            x=["x1", "x2", "x3"],
            options=options_cls(solver=solver),
        ).fit()


def check_separation_raises_computation_error(
    estimator_cls, dataset_prefix, options_cls, solver, scenario
):
    """真の分離（合成データセット、有限MLEが存在しない）は数値比較の対象外
    （`testing-policy.md`「テストの3系統」）。想定エラー（`ComputationError`）が
    発生することのみを確認する。`solver`（newton/bfgs/lbfgs）でparametrizeする。

    `scenario`は`complete_separation`（`y`が`x1`の符号のみで決まる完全分離）または
    `quasi_complete_separation`（2値ダミーが1の観測は全て`y=1`となる準完全分離）。

    実際に発生する例外の**メッセージはsolverによって異なる**（newtonは分離検出、
    bfgs/lbfgsは分離検出に至る前にソルバー自身が`NonConvergence`等で失敗することがある）
    ため、基底クラス`ComputationError`のみをアサートする（`check_perfect_
    multicollinearity_raises_computation_error`と同型の設計だが、あちらは
    `solver`によらず常に`SingularDesignMatrix`である点が異なる）。分離検出自体は
    サンプルサイズ・分離の種類によらず厳密（線形計画）なので、無警告の「成功」には
    ならない。
    """
    df = pl.read_csv(DATA_DIR / f"{dataset_prefix}_{scenario}.csv")
    x_cols = [c for c in df.columns if c != "y"]
    with pytest.raises(ComputationError):
        estimator_cls(
            df,
            y="y",
            x=x_cols,
            options=options_cls(solver=solver),
        ).fit()


def check_separation_is_reported_as_separation_by_newton(
    estimator_cls, dataset_prefix, options_cls, scenario
):
    """newtonは分離データで`NonConvergence`ではなく分離検出のメッセージを返す
    （分離の原因と対処が利用者に伝わる）。`ComputationError`の基底クラスだけでは
    区別できないため、メッセージで固定する。"""
    df = pl.read_csv(DATA_DIR / f"{dataset_prefix}_{scenario}.csv")
    x_cols = [c for c in df.columns if c != "y"]
    with pytest.raises(ComputationError, match="separated"):
        estimator_cls(
            df,
            y="y",
            x=x_cols,
            options=options_cls(solver="newton"),
        ).fit()


def check_separation_with_raise_on_non_convergence_false(
    estimator_cls, dataset_prefix, options_cls, solver, scenario
):
    """分離データ（`check_separation_raises_computation_error`と同じフィクスチャ）で
    `raise_on_non_convergence=False`を指定した場合の挙動を固定する。

    分離検出は`raise_on_non_convergence=False`のとき例外を送出せず`converged=False`の
    まま結果を返す（`docs/spec/logit-spec.md`参照）。ただし収束判定とは独立した
    `SingularHessian`等の別エラー経路は`raise_on_non_convergence`に関わらず発生しうる
    （Tobitと同型の挙動。bfgs/lbfgsで入ることがある）ため、「例外を投げず結果を返した
    場合は`converged=False`で標準誤差が有限値であること」のみを保証し、
    `ComputationError`が飛ぶこと自体は許容する（メソッドごとに軌道が異なり、どちらの
    経路に入るかは実装の詳細のため）。newtonは必ず結果を返すことを別途固定する
    （`check_separation_returns_unconverged_result_for_newton`）。
    """
    df = pl.read_csv(DATA_DIR / f"{dataset_prefix}_{scenario}.csv")
    x_cols = [c for c in df.columns if c != "y"]
    options = options_cls(solver=solver, raise_on_non_convergence=False)
    try:
        res = estimator_cls(df, y="y", x=x_cols, options=options).fit()
    except ComputationError:
        return
    assert res.converged is False
    for name, se in res.std_errors.items():
        assert math.isfinite(se), f"std_errors[{name}]={se} is not finite"


def check_separation_returns_unconverged_result_for_newton(
    estimator_cls, dataset_prefix, options_cls, scenario
):
    """newtonは分離データで`raise_on_non_convergence=False`なら例外を送出せず、
    `converged=False`の結果を返す（分離を無警告で「収束」と報告しない）。"""
    df = pl.read_csv(DATA_DIR / f"{dataset_prefix}_{scenario}.csv")
    x_cols = [c for c in df.columns if c != "y"]
    res = estimator_cls(
        df,
        y="y",
        x=x_cols,
        options=options_cls(solver="newton", raise_on_non_convergence=False),
    ).fit()
    assert res.converged is False


def check_small_sample_separation_is_not_reported_as_success(
    estimator_cls, options_cls, solver
):
    """自由度が1（`n=k+1`）の極小標本で完全分離になるデータは、無警告の「収束」として
    返さず`ComputationError`にする。かつては標準化パラメータのノルム基準がこの種の
    データを見逃し、`converged=True`のまま標準誤差が数千の結果を返していた。
    `raise_on_non_convergence=False`でも`converged=True`にならない。"""
    df = small_sample_separation_dataset()
    x = ["x1", "x2", "x3"]
    with pytest.raises(ComputationError):
        estimator_cls(df, y="y", x=x, options=options_cls(solver=solver)).fit()
    try:
        res = estimator_cls(
            df,
            y="y",
            x=x,
            options=options_cls(solver=solver, raise_on_non_convergence=False),
        ).fit()
    except ComputationError:
        return
    assert res.converged is False


def check_multiple_dummy_separation_raises_computation_error(
    estimator_cls, options_cls, solver
):
    """3つのダミーが同時に準完全分離するデータ（分離方向が複数）も、全solverで
    無警告の「成功」にならず`ComputationError`になる。`raise_on_non_convergence=False`でも
    結果を返すなら`converged=False`。"""
    df = multi_dummy_separation_dataset()
    x = [c for c in df.columns if c != "y"]
    with pytest.raises(ComputationError):
        estimator_cls(df, y="y", x=x, options=options_cls(solver=solver)).fit()
    try:
        res = estimator_cls(
            df,
            y="y",
            x=x,
            options=options_cls(solver=solver, raise_on_non_convergence=False),
        ).fit()
    except ComputationError:
        return
    assert res.converged is False


# 1観測だけが分離を壊すデータで、分離判定の境界の両側。実測した境界はこのデータで重なり
# 1e-5〜3e-5の間（標準化後の幾何に依存し、データごとに変わる）。境界そのものではなく、
# 明らかに両側にある値を固定する（`docs/spec/logit-spec.md`3.2節）。
OVERLAP_ABOVE_BOUNDARY = [1e-3, 1e-4]
OVERLAP_BELOW_BOUNDARY = [1e-7, 1e-9, 1e-12]


def check_single_overlap_above_boundary_is_not_flagged_as_separation(
    estimator_cls, options_cls, solver, overlap, min_slope
):
    """1観測だけが分離を壊し、重なりが十分大きければ分離とみなさない。有限のMLEが存在する
    ので推定は収束し、係数は大きいが有限、標準誤差も有限になる。`min_slope`は`x1`の係数の
    下限（実測: Logitは重なり`1e-3`で約307・`1e-4`で約531、Probitは約179・約280）。"""
    df = single_overlap_binary_dataset(overlap)
    res = estimator_cls(
        df,
        y="y",
        x=["x1"],
        options=options_cls(solver=solver, max_iter=200),
    ).fit()
    assert res.converged is True
    assert math.isfinite(res.params["x1"])
    assert res.params["x1"] > min_slope
    for name, se in res.std_errors.items():
        assert math.isfinite(se), f"std_errors[{name}]={se} is not finite"


def check_single_overlap_below_boundary_is_flagged_as_separation(
    estimator_cls, options_cls, solver, overlap
):
    """重なりが小さすぎて`1e-6`未満のマージンしか持たないとき、分離として扱う
    （`ComputationError`）。`raise_on_non_convergence=False`なら`converged=False`の
    結果を返す。"""
    df = single_overlap_binary_dataset(overlap)
    with pytest.raises(ComputationError, match="separated"):
        estimator_cls(
            df,
            y="y",
            x=["x1"],
            options=options_cls(solver=solver, max_iter=200),
        ).fit()
    res = estimator_cls(
        df,
        y="y",
        x=["x1"],
        options=options_cls(
            solver=solver, max_iter=200, raise_on_non_convergence=False
        ),
    ).fit()
    assert res.converged is False


def check_separation_without_intercept_raises_computation_error(
    estimator_cls, options_cls
):
    """切片なし（`include_intercept=False`）でも、完全分離は`ComputationError`になる。"""
    df = separated_dataset()
    with pytest.raises(ComputationError, match="separated"):
        estimator_cls(
            df,
            y="y",
            x=["x1", "x2"],
            options=options_cls(include_intercept=False),
        ).fit()


def check_strongly_collinear_regressors_are_not_rejected_as_separation(
    estimator_cls, options_cls, link, solver
):
    """相関がほぼ1の2列を含む、分離していない2値データは有限のMLEが存在するので、
    分離として棄却せず収束させる（かつてはノルム基準が係数の符号反転した大きな値を
    分離と誤認していた）。"""
    df = strongly_collinear_binary_dataset(link)
    res = estimator_cls(
        df, y="y", x=["x1", "x2", "x3"], options=options_cls(solver=solver)
    ).fit()
    assert res.converged is True


def check_non_convergence_raises_computation_error_with_tiny_max_iter(
    dataset, estimator_cls, options_cls
):
    """`max_iter`を人為的に1に絞ると`raise_on_non_convergence=True`
    （既定）で`ComputationError`（engine側の`NonConvergence`）。

    完全分離等の病理的なデータは`NonConvergence`ではなく専用の
    `SeparationDetected`（engineの`MleError`の別バリアント。Pythonでは
    `NonConvergence`と同じ`ComputationError`で、例外クラスでは区別できない。
    `check_separated_data_raises_computation_error`参照）を返すため、
    `NonConvergence`自体の発生確認には使えない。そのため
    `NonConvergence`の発生確認は、専用データセットに頼らずmax_iterを
    人為的に小さくする方法で行う（`docs/spec/<solver>-spec.md`参照）。
    """
    with pytest.raises(ComputationError):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(max_iter=1),
        ).fit()


def check_separated_data_raises_computation_error(estimator_cls):
    """分離データ（`x1`の真の係数を極端に大きくし、全観測がx1の符号だけで完全に
    分類できるようにしたDGP）は`ComputationError`（engine側の`SeparationDetected`）。

    分離が、Python API境界を通しても正しく検出されエラーになることを確認する
    （`engine`側のRust単体テスト`fit_returns_separation_detected_error_for_separated_data`
    のAPIレベル版）。
    """
    df = separated_dataset()

    with pytest.raises(ComputationError):
        estimator_cls(df, y="y", x=["x1", "x2"]).fit()


# ── test_<method>_reference.py ───────────────────────────────────────
#
# `FIXTURE_PATH`（`logit.json`/`probit.json`）・`SCENARIOS`
# （`generate_logit_fixtures`/`generate_probit_fixtures`由来）・
# `TOLERANCES`キー接頭辞等、手法ごとに異なる値が多いため、個別引数ではなく
# この設定オブジェクトにまとめて渡す。


@dataclass(frozen=True)
class BinaryChoiceReferenceConfig:
    estimator_cls: type
    options_cls: type
    dataset_prefix: (
        str  # "logit" / "probit"（CSVファイル名・statsmodels比較の識別用）
    )
    fixture_path: Path
    scenarios: Sequence[str]
    cov_types: Sequence[str]
    rtol: float
    atol: float
    rtol_solver: float
    rtol_solver_vs_newton: float
    near_separation_tol: float

    def load_fixtures(self) -> dict:
        return json.loads(self.fixture_path.read_text())

    def dataset_path(self, scenario: str) -> Path:
        return DATA_DIR / f"{self.dataset_prefix}_{scenario}.csv"

    def assert_close(self, ours: float, ref: float, label: str) -> None:
        assert_close(ours, ref, label, rtol=self.rtol, atol=self.atol)

    def assert_dict_close(
        self, ours: dict[str, float], ref: dict[str, float], label: str
    ) -> None:
        assert_dict_close(ours, ref, label, rtol=self.rtol, atol=self.atol)

    def assert_dict_close_method(
        self, ours: dict[str, float], ref: dict[str, float], label: str
    ) -> None:
        assert_dict_close(
            ours, ref, label, rtol=self.rtol_solver, atol=self.atol
        )

    def check_margeff(self, res, ref_margeff: dict, label: str) -> None:
        check_margeff(res, ref_margeff, label, rtol=self.rtol, atol=self.atol)


def check_result(
    config: BinaryChoiceReferenceConfig, res, ref: dict, label: str
) -> None:
    config.assert_dict_close(res.params, ref["params"], f"{label}/params")
    config.assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label}/std_errors"
    )
    config.assert_dict_close(
        res.test_stats, ref["test_stats"], f"{label}/test_stats"
    )
    config.assert_dict_close(
        res.p_values, ref["p_values"], f"{label}/p_values"
    )

    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_name = _rename(name)
        our_lower, our_upper = res.conf_int[our_name]
        config.assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        config.assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    config.assert_close(
        res.log_likelihood, ref["log_likelihood"], f"{label}/log_likelihood"
    )
    config.assert_close(
        res.log_likelihood_null,
        ref["log_likelihood_null"],
        f"{label}/log_likelihood_null",
    )
    config.assert_close(
        res.lr_statistic, ref["lr_statistic"], f"{label}/lr_statistic"
    )
    config.assert_close(
        res.lr_p_value, ref["lr_p_value"], f"{label}/lr_p_value"
    )
    config.assert_close(
        res.pseudo_r_squared,
        ref["pseudo_r_squared"],
        f"{label}/pseudo_r_squared",
    )
    config.assert_close(res.aic, ref["aic"], f"{label}/aic")
    config.assert_close(res.bic, ref["bic"], f"{label}/bic")
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_model == ref["df_model"], f"{label}/df_model"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"
    assert res.converged == ref["converged"], f"{label}/converged"

    ours_pred_table = {
        row["actual"]: (row["predicted_0"], row["predicted_1"])
        for row in res.pred_table()
    }
    for i, row in enumerate(ref["pred_table"]):
        assert ours_pred_table[i] == (row[0], row[1]), (
            f"{label}/pred_table/{i}"
        )

    if ref["margeff"] is not None:
        config.check_margeff(res, ref["margeff"], label)


# ── test_<method>_reference.py: 凍結フィクスチャとの数値照合 ─────────


def check_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures, scenario, cov_type
) -> None:
    df = pl.read_csv(config.dataset_path(scenario))
    x_cols = [c for c in df.columns if c != "y"]
    kwargs = (
        {"tol": config.near_separation_tol}
        if scenario == "near_separation"
        else {}
    )
    options = config.options_cls(cov_type=cov_type, **kwargs)
    res = config.estimator_cls(df, y="y", x=x_cols, options=options).fit()

    check_result(
        config,
        res,
        fixtures["synthetic"][scenario][cov_type],
        f"{scenario}/{cov_type}",
    )


def _baseline_cluster_fit(
    config: BinaryChoiceReferenceConfig, kind: str, **option_kwargs
):
    df = pl.read_csv(config.dataset_path("baseline"))
    if kind == "cluster":
        df = with_cluster_groups(df, 10)
    else:
        df = df.with_columns(
            pl.Series("cluster_group", imbalanced_cluster_groups(df.height))
        )
    options = config.options_cls(
        cov_type="cluster", cluster="cluster_group", **option_kwargs
    )
    return config.estimator_cls(
        df, y="y", x=["x1", "x2", "x3"], options=options
    ).fit()


def check_cluster_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures
) -> None:
    """クラスターロバストSE（baselineシナリオ、行番号%10の疑似グループ）。
    係数・標準誤差に加え、z値・p値・信頼区間・適合度統計量・限界効果まで全統計量を比べる。"""
    res = _baseline_cluster_fit(config, "cluster")
    check_result(
        config, res, fixtures["synthetic"]["baseline"]["cluster"], "cluster"
    )


def check_cluster_imbalanced_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures
) -> None:
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。全統計量を比べる。"""
    res = _baseline_cluster_fit(config, "cluster_imbalanced")
    check_result(
        config,
        res,
        fixtures["synthetic"]["baseline"]["cluster_imbalanced"],
        "cluster_imbalanced",
    )


def _apple_cluster_fit(config: BinaryChoiceReferenceConfig, **option_kwargs):
    options = config.options_cls(
        cov_type="cluster", cluster=APPLE_CLUSTER, **option_kwargs
    )
    return config.estimator_cls(
        load_apple_adoption_dataset(), y="y", x=APPLE_X, options=options
    ).fit()


def check_apple_cluster_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures
) -> None:
    """多クラスターの実データ（Wooldridge apple、`ecolbs > 0`の2値`y`、州でG=49、
    グループサイズ1〜66）。mrozの`city`（G=2）はG<=qのため成功パスを持てず、実データの
    クラスターロバストSEはここで全統計量を照合する。"""
    res = _apple_cluster_fit(config)
    check_result(
        config,
        res,
        fixtures["wooldridge"]["apple"]["cluster"],
        "apple/cluster",
    )


# `solver="bfgs"/"lbfgs"`と掛け合わせる代表ケース。newtonは全シナリオ×cov_typeを
# 掛け合わせて照合済みのため、solverの違いが効きやすい点（最も悪条件の
# near_separation、cov_typeごとに別経路のopg/hc0、クラスター）に絞る。
SOLVER_COMBINATION_CASES = [
    "near_separation/classical",
    "near_separation/opg",
    "near_separation/hc0",
    "cluster",
    "cluster_imbalanced",
    "apple_cluster",
]


def check_solver_combination_matches_newton_reference(
    config: BinaryChoiceReferenceConfig, fixtures, case: str, solver: str
) -> None:
    """`solver="bfgs"/"lbfgs"`を、near_separation・cluster・appleと掛け合わせた
    組み合わせで、statsmodelsのnewtonで生成した同じシナリオ・cov_typeの全統計量と比べる。

    参照にstatsmodels自身のbfgs/lbfgsを使わないのは、その収束許容が緩く
    near_separationで本実装と~1e-3差が出る（statsmodels側の誤差）ため。newtonは
    機械精度で収束するので、収束先（MLE）が同じかを`rtol_solver_vs_newton`で厳密に
    確認できる。
    """
    tight = replace(config, rtol=config.rtol_solver_vs_newton)
    label = f"{case}/{solver}"
    if case.startswith("near_separation/"):
        cov_type = case.split("/")[1]
        df = pl.read_csv(config.dataset_path("near_separation"))
        options = config.options_cls(
            cov_type=cov_type, solver=solver, tol=config.near_separation_tol
        )
        res = config.estimator_cls(
            df,
            y="y",
            x=[c for c in df.columns if c != "y"],
            options=options,
        ).fit()
        ref = fixtures["synthetic"]["near_separation"][cov_type]
    elif case == "apple_cluster":
        # appleは平均が標準偏差に比べて大きい列を含む。標準化で平均を引かない
        # 実装ではlbfgsが既定のmax_iter=35では収束しなかった（約95反復）ため、
        # max_iterを指定しないこの呼び出しが既定値での収束の回帰ガードを兼ねる。
        res = _apple_cluster_fit(config, solver=solver)
        ref = fixtures["wooldridge"]["apple"]["cluster"]
    else:
        res = _baseline_cluster_fit(config, case, solver=solver)
        ref = fixtures["synthetic"]["baseline"][case]
    check_result(tight, res, ref, label)


def check_mroz_cluster_cov_type_raises_validation_error(
    estimator_cls, options_cls
) -> None:
    """実データでのクラスターロバストSEの`G <= q`境界。

    mrozの`city`（都市部居住ダミー、484/269の2値）はG=2、`MROZ_X`は7変数なので
    `G=2 <= q=7`。`rank(Ŝ) <= G-1`のためクラスターロバスト共分散が退化するため、
    `fit()`冒頭のバリデーションが`ValidationError`
    （`CommonError::InsufficientClustersForInference`）で弾く。従来はLogit/Probitが
    この縮退した共分散から読んだSEを無警告で返していた（silent-pass、実質バグ。
    数値照合フィクスチャ`mroz/cluster`を持っていたが、この対応で削除）。
    """
    df = load_wooldridge_dataset("mroz")
    options = options_cls(cov_type="cluster", cluster="city")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=2, q=len(MROZ_X)
        ),
    ):
        estimator_cls(df, y="inlf", x=MROZ_X, options=options).fit()


def check_solver_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures, solver
) -> None:
    """`solver="bfgs"/"lbfgs"`が主リファレンス（statsmodelsの同じsolver）と
    フルの統計量（std_errors含む）で一致すること。

    既定の`solver="newton"`のみ全シナリオ×cov_typeで数値照合しており、
    bfgs/lbfgsは`test_<solver>_api.py::test_method_option_converges_to_same_params`
    で自身のnewton結果とparamsのみ緩い許容誤差(rel=1e-4)で比較していたが、
    主リファレンスに対するフルの統計量照合が無かった
    （`testing-completeness-reviewer`指摘）。
    """
    df = pl.read_csv(config.dataset_path("baseline"))
    options = config.options_cls(cov_type="classical", solver=solver)
    res = config.estimator_cls(
        df, y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = fixtures["synthetic"]["solver"][solver]
    label = f"solver/{solver}"
    config.assert_dict_close_method(
        res.params, ref["params"], f"{label}/params"
    )
    config.assert_dict_close_method(
        res.std_errors, ref["std_errors"], f"{label}/std_errors"
    )
    assert res.converged == ref["converged"], f"{label}/converged"


def check_mroz_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures, cov_type
) -> None:
    """Wooldridge実データ（mroz、労働参加モデル）とのクロスチェック。"""
    df = load_wooldridge_dataset("mroz")
    options = config.options_cls(cov_type=cov_type)
    res = config.estimator_cls(df, y="inlf", x=MROZ_X, options=options).fit()

    check_result(
        config,
        res,
        fixtures["wooldridge"]["mroz"][cov_type],
        f"mroz/{cov_type}",
    )


# ── test_<method>_reference.py: ライブ statsmodels との照合 ──────────
# （凍結フィクスチャが対象にしない include_intercept=False の分岐）


def check_include_intercept_false_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, sm_estimator_cls, cov_type
) -> None:
    """`include_intercept=False`の成功パスが構造テスト・数値照合テストとも
    一切検証されていなかった（`df_model`は`include_intercept`の値に関わらず
    常に`k-1`、`log_likelihood_null`は常に「切片のみ」モデルを参照するため
    `include_intercept=False`時は`lr_statistic`が負値になりうる、という特殊
    挙動が`engine`側の単体テストのみで数値照合が無かった。
    `testing-completeness-reviewer`指摘）。frozen
    fixtureではなくstatsmodelsとの直接照合で確認する
    （`test_ols_reference.py`と同じ方針）。

    `statsmodels_ref.run()`はformula API（`patsy`経由でpandasを要求する）を
    使うため、ここでは使わない。OLS側と同じ配列API（`sm.Logit(y, x)`/
    `sm.Probit(y, x)`）で直接比較する（`tests/`はpyarrow等のformula API
    依存パッケージをdev依存に持たない方針、`.claude/rules/testing-policy.md`
    参照）。
    """
    import numpy as np

    df = pl.read_csv(config.dataset_path("baseline"))
    y = df["y"].to_numpy()
    x_cols = ["x1", "x2", "x3"]
    x = np.column_stack([df[c].to_numpy() for c in x_cols])

    if cov_type == "opg":
        base = sm_estimator_cls(y, x).fit(disp=0)
        scores = base.model.score_obs(base.params)
        opg_cov = np.linalg.inv(scores.T @ scores)
        sm_se = np.sqrt(np.diag(opg_cov))
        sm_params = base.params
        fitted = base
    else:
        sm_cov_type = {"classical": "nonrobust"}.get(cov_type, cov_type)
        fitted = sm_estimator_cls(y, x).fit(disp=0, cov_type=sm_cov_type)
        sm_params = fitted.params
        sm_se = fitted.bse

    options = config.options_cls(cov_type=cov_type, include_intercept=False)
    res = config.estimator_cls(df, y="y", x=x_cols, options=options).fit()

    label = f"include_intercept_false/{cov_type}"
    for i, name in enumerate(x_cols):
        config.assert_close(
            res.params[name], sm_params[i], f"{label}/coef/{name}"
        )
        config.assert_close(
            res.std_errors[name], sm_se[i], f"{label}/se/{name}"
        )
    assert res.converged == bool(fitted.mle_retvals["converged"]), (
        f"{label}/converged"
    )
    assert res.df_model == fitted.df_model, f"{label}/df_model"


def check_predict_new_data_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, sm_estimator_cls
) -> None:
    """新規データに対する`predict()`がstatsmodelsの`.predict()`と一致すること
    （OLSの`test_predict_new_data_matches_statsmodels`と同型）。

    列順を学習時（x1, x2, x3）と入れ替えて渡し、列名でマッチングされる
    （列順に依存しない）ことも合わせて確認する。`predict()`の値自体は
    `cov_type`に依存しないため（点推定はcov_typeの影響を受けない）、
    ここでは`classical`のみで確認する。
    """
    import numpy as np
    import statsmodels.api as sm

    df = pl.read_csv(config.dataset_path("baseline"))
    y = df["y"].to_numpy()
    x_cols = ["x1", "x2", "x3"]
    x = np.column_stack([df[c].to_numpy() for c in x_cols])
    sm_fitted = sm_estimator_cls(y, sm.add_constant(x)).fit(disp=0)

    options = config.options_cls(cov_type="classical")
    res = config.estimator_cls(df, y="y", x=x_cols, options=options).fit()

    new_data = pl.DataFrame(
        {"x2": [0.5, -1.0], "x3": [0.2, 0.1], "x1": [1.0, -0.5]}
    )
    predicted = res.predict(new_data)

    sm_new_x = sm.add_constant(
        np.column_stack(
            [
                new_data["x1"].to_numpy(),
                new_data["x2"].to_numpy(),
                new_data["x3"].to_numpy(),
            ]
        ),
        has_constant="add",
    )
    expected = sm_fitted.predict(sm_new_x)

    assert len(predicted) == 2
    for i, (prob, exp) in enumerate(zip(predicted, expected)):
        config.assert_close(prob, exp, f"predict_new_data/{i}")


def check_predict_new_data_without_intercept_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, sm_estimator_cls
) -> None:
    """`include_intercept=False`でfitした場合の`predict(new_data)`も
    statsmodelsと一致すること（OLSの`test_predict_new_data_without_intercept_
    matches_statsmodels`と同型、python-reviewer指摘）。
    """
    df = pl.read_csv(config.dataset_path("baseline"))
    y = df["y"].to_numpy()
    x1 = df["x1"].to_numpy()

    options = config.options_cls(include_intercept=False)
    res = config.estimator_cls(df, y="y", x=["x1"], options=options).fit()
    sm_fitted = sm_estimator_cls(y, x1.reshape(-1, 1)).fit(disp=0)

    new_x1 = [1.0, 2.0, -3.0]
    new_data = pl.DataFrame({"x1": new_x1})
    predicted = res.predict(new_data)
    expected = sm_fitted.predict([[v] for v in new_x1])

    assert len(predicted) == 3
    for i, (prob, exp) in enumerate(zip(predicted, expected)):
        config.assert_close(prob, exp, f"predict_new_data_no_intercept/{i}")


def check_predict_with_include_intercept_false_and_x_named_const(
    estimator_cls, options_cls, link
) -> None:
    """`include_intercept=False`かつ`x`に`"const"`という名前の列を含む場合でも
    `predict(new_data)`が正しく動作すること（OLSの`test_predict_with_include_
    intercept_false_and_x_named_const`と同型の回帰テスト）。

    `include_intercept=True`のときのみ`"const"`列名との衝突チェックが働く仕様
    のため、`include_intercept=False`ならユーザーが`"const"`という名前の
    （切片ではない）通常の説明変数を`x`に含めることは正当な入力。`predict()`の
    内部実装が誤って列名から「自動追加された切片列かどうか」を推測すると、
    この場合に値を無視して1.0固定にしてしまう回帰バグがOLSで過去にあったため
    （`test_ols_api.py`参照）、同型の実装（`has_intercept`フラグのみで判定し
    列名は見ない）を使うLogit/Probitでも固定用に追加する。

    `link`はリンク関数（Logitは`Λ`、Probitは`Φ`）を`z -> probability`の
    呼び出し可能オブジェクトとして呼び出し側から渡す（このcheck自体はLogit/
    Probit共有のため、期待値の計算式だけを手法ごとに差し替える）。
    """
    df = pl.DataFrame(
        {
            "y": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "const": [2.0, 5.0, 1.0, 8.0, 3.0, 6.0, 4.0, 7.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    options = options_cls(include_intercept=False)
    res = estimator_cls(df, y="y", x=["const", "x2"], options=options).fit()

    new_data = pl.DataFrame({"const": [100.0, 200.0], "x2": [10.0, 20.0]})
    predicted = res.predict(new_data)

    coef_const = res.params["const"]
    coef_x2 = res.params["x2"]
    for i, (prob, (c, x2)) in enumerate(
        zip(predicted, [(100.0, 10.0), (200.0, 20.0)])
    ):
        z = coef_const * c + coef_x2 * x2
        expected = link(z)
        assert abs(prob - expected) < 1e-9, f"probability/{i}"


def check_cluster_unused_by_cov_type_raises(
    dataset, estimator_cls, options_cls
):
    """`cov_type="cluster"`以外で`cluster`が指定されたら黙って無視せず
    `ValidationError`。
    """
    for cov_type in ["classical", "opg", "hc0", "hc1"]:
        options = options_cls(cov_type=cov_type, cluster="cluster")
        with pytest.raises(
            ValidationError,
            match=escaped(
                msgs.UNUSED_OPTION,
                option="cluster",
                condition='cov_type="cluster"',
            ),
        ):
            estimator_cls(
                dataset, y="y", x=["x1", "x2"], options=options
            ).fit()
