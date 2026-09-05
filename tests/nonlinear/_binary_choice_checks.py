"""Logit/Probitの`test_logit_*.py`/`test_probit_*.py`で重複していた
テスト本体を関数として集約する（`refactoring-candidates-2.md`項目95）。

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close, check_margeff
from _assertions import rename_intercept as _rename
from _helpers import (
    DATA_DIR,
    MROZ_X,
    load_wooldridge_dataset,
    separation_suspected_dataset,
    with_cluster_groups,
)
from econometricsmodels import ComputationError, ValidationError

from benchmark.common import imbalanced_cluster_groups

# ── test_<method>_api.py: 成功パス・結果型 ──────────────────────────


def check_fit_succeeds_and_returns_results(dataset, estimator_cls, results_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert isinstance(res, results_cls)


def check_default_options_use_classical_and_converge(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.cov_type == "classical"
    assert res.converged


# ── test_<method>_api.py: API構造 ───────────────────────────────────


def check_coef_table_structure(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
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


def check_conf_int_structure(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    ci = res.conf_int
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def check_params_std_errors_z_stats_p_values_share_keys(
    dataset, estimator_cls
):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    expected_keys = {"const", "x1", "x2"}
    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.z_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def check_n_obs_matches_dataset_size(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.n_obs == dataset.height


def check_param_names_include_const_first(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.param_names == ["const", "x1", "x2"]


# ── test_<method>_api.py: オプションの反映 ──────────────────────────
#
# cov_type 以外の Options フィールド（method・include_intercept・
# confidence_level・raise_on_non_convergence）が、engine_pybind 側の
# 文字列パース・列抽出・分岐ロジックを経て正しく反映されることを確認する。


def check_method_option_converges_to_same_params(
    dataset, estimator_cls, options_cls, method
):
    """`method`（newton/bfgs/lbfgs）はいずれも同じ最尤解に収束する。

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
        options=options_cls(method=method),
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
    <method>-spec.md`参照）となるため、その旨も確認する
    （`testing-completeness-reviewer`指摘、Issue #231フェーズ4）。
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
    `test_confidence_level_changes_interval_width`と同型、Issue #231
    フェーズ4）。
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
    dataset, estimator_cls, options_cls
):
    """`raise_on_non_convergence=False`だと未収束でも例外を投げず、
    `converged=False`の`Results`を返す（engine側のもう一方の分岐、APIレベル
    での配線確認。例外を送出する既定挙動側は
    `test_<method>_validation.py::test_non_convergence_raises_computation_error_with_tiny_max_iter`）。
    """
    res = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(max_iter=1, raise_on_non_convergence=False),
    ).fit()
    assert res.converged is False
    assert res.n_iter == 1


def check_cov_type_label(dataset, estimator_cls, options_cls):
    """`res.cov_type`が指定した`cov_type`（正規化済み小文字）を反映すること
    （OLSの`test_cov_type_label`と同型、Issue #231フェーズ4）。
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
        options=options_cls(cov_type="cluster", cluster_col="cluster"),
    ).fit()
    assert res.cov_type == "cluster"


def check_cov_type_is_case_insensitive(
    dataset, estimator_cls, options_cls, cov_type, expected_label
):
    """`cov_type`が大文字小文字を区別しないこと（`engine_pybind`側の
    `build_<method>_input`のRust単体テストと対になる、Python API境界での
    確認。OLS/WLSの`test_cov_type_is_case_insensitive`と同型、Issue #231
    フェーズ4）。
    """
    kwargs = {"cluster_col": "cluster"} if cov_type == "CLUSTER" else {}
    options = options_cls(cov_type=cov_type, **kwargs)
    res = estimator_cls(dataset, y="y", x=["x1", "x2"], options=options).fit()
    assert res.cov_type == expected_label


def check_nonrobust_is_alias_for_classical(
    dataset, estimator_cls, options_cls, cov_type
):
    """`"nonrobust"`が`"classical"`と同じ計算方法（標準誤差も一致）の
    エイリアスであること（OLS/WLSの`test_nonrobust_is_alias_for_classical`と
    同型、Issue #231フェーズ4）。
    """
    res = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(cov_type=cov_type),
    ).fit()
    classical_res = estimator_cls(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=options_cls(cov_type="classical"),
    ).fit()
    for name in res.param_names:
        assert res.std_errors[name] == classical_res.std_errors[name], name


# ── test_<method>_api.py: predict() ─────────────────────────────────


def check_predict_returns_row_oriented_probabilities(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    predicted = res.predict()

    assert len(predicted) == dataset.height
    for row in predicted:
        assert set(row.keys()) == {"probability"}
        assert 0.0 <= row["probability"] <= 1.0


# ── test_<method>_api.py: pred_table() ──────────────────────────────


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


# ── test_<method>_api.py: marginal_effects() ────────────────────────


def check_marginal_effects_default_excludes_intercept(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
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


def check_marginal_effects_mean_and_median_differ_from_overall(
    dataset, estimator_cls
):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    overall = [row["dydx"] for row in res.marginal_effects(at="overall")]
    mean = [row["dydx"] for row in res.marginal_effects(at="mean")]
    median = [row["dydx"] for row in res.marginal_effects(at="median")]

    assert overall != mean
    assert overall != median


def check_marginal_effects_at_is_case_insensitive(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    assert res.marginal_effects(at="OVERALL") == res.marginal_effects(
        at="overall"
    )


# ── test_<method>_validation.py: ValidationError（入力データ） ──────


def check_y_in_x_raises(dataset, estimator_cls):
    with pytest.raises(ValidationError):
        estimator_cls(dataset, y="y", x=["y", "x1"]).fit()


def check_duplicate_x_column_raises(dataset, estimator_cls):
    with pytest.raises(ValidationError):
        estimator_cls(dataset, y="y", x=["x1", "x1"]).fit()


def check_const_collision_with_include_intercept_raises(estimator_cls):
    df = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0, 1.0], "const": [1.0, 2.0, 3.0, 3.5]}
    )
    with pytest.raises(ValidationError):
        estimator_cls(df, y="y", x=["const"]).fit()


def check_empty_x_raises(dataset, estimator_cls):
    with pytest.raises(ValidationError):
        estimator_cls(dataset, y="y", x=[]).fit()


def check_missing_column_raises(dataset, estimator_cls):
    with pytest.raises(ValidationError):
        estimator_cls(dataset, y="y", x=["does_not_exist"]).fit()


def check_null_values_raise(estimator_cls):
    """欠損値は`column_extraction`の責務で`ValidationError`（OLSの
    `test_null_values_raise`と同型、Python API境界で未検証だった、
    Issue #231フェーズ4）。
    """
    df = pl.DataFrame({"y": [0.0, None, 1.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        estimator_cls(df, y="y", x=["x1"]).fit()


def check_non_numeric_dtype_raises(estimator_cls):
    """数値/文字列型にキャストできない列は`ValidationError`（OLSの
    `test_non_numeric_dtype_raises`と同型、Issue #231フェーズ4）。
    """
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        estimator_cls(df, y="y", x=["x1"]).fit()


def check_non_binary_y_raises(dataset, estimator_cls, bad_value):
    """`y`が`{0.0, 1.0}`以外の値を含む場合は`ValidationError`
    （engine側の`MleError::InvalidBinaryY`）。
    """
    df = dataset.with_columns(dataset["y"].scatter(0, bad_value))
    with pytest.raises(ValidationError):
        estimator_cls(df, y="y", x=["x1", "x2"]).fit()


def check_insufficient_observations_raises(dataset, estimator_cls):
    df = dataset.head(2)
    with pytest.raises(ValidationError):
        estimator_cls(df, y="y", x=["x1", "x2"]).fit()


# ── test_<method>_validation.py: ValidationError（オプション） ─────


def check_unknown_cov_type_raises(dataset, estimator_cls, options_cls):
    with pytest.raises(ValidationError):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(cov_type="bogus"),
        ).fit()


def check_unknown_method_raises(dataset, estimator_cls, options_cls):
    with pytest.raises(ValidationError):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(method="bogus"),
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
    `test_invalid_confidence_level_raises`との非対称、Issue #231フェーズ4）。
    """
    options = options_cls(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        estimator_cls(
            dataset, y="y", x=["x1", "x2"], options=options
        ).fit()


def check_non_positive_tol_raises(dataset, estimator_cls, options_cls, tol):
    """`tol<=0`は勾配ノルム基準の収束条件が理論上満たされないため
    `ValidationError`（engine側の`MleError::InvalidTol`）。
    """
    with pytest.raises(ValidationError):
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
    （`testing-completeness-reviewer`指摘、Issue #231フェーズ4）。
    """
    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
        estimator_cls(
            df,
            y="y",
            x=["x1"],
            options=options_cls(cov_type="cluster", cluster_col="cluster"),
        ).fit()


def check_cluster_col_nonexistent_column_raises(
    dataset, estimator_cls, options_cls
):
    """`cluster_col`が実在しない列名を指すと`ValidationError`（OLSと同じ理由、
    Issue #231フェーズ4）。
    """
    options = options_cls(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(ValidationError):
        estimator_cls(dataset, y="y", x=["x1", "x2"], options=options).fit()


# ── test_<method>_validation.py: ValidationError（marginal_effects()） ─


def check_marginal_effects_unknown_at_raises(dataset, estimator_cls):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(at="bogus")


def check_marginal_effects_confidence_level_out_of_range_raises(
    dataset, estimator_cls
):
    res = estimator_cls(dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(confidence_level=1.5)


# ── test_<method>_validation.py: ComputationError ───────────────────


def check_singular_hessian_raises_computation_error(
    estimator_cls, options_cls, method
):
    """完全な多重共線性は`ComputationError`。

    `method`をparametrizeしているのは、`newton`は`newton_step`内の
    ピボット付きQR分解経由でたまたま特異性を検出できていたが、`bfgs`/
    `lbfgs`は準ニュートン法のため`newton_step`を経由せず、収束後の
    `observed_information_cov_params`呼び出しが唯一の検出経路になるという
    構造的な違いがあるため（`docs/planning/specs/nonlinear-implementation-notes.md`
    「`cov_type`共通行列演算の特異性検出」参照。過去に`bfgs`だけ検出漏れし
    桁違いに巨大な標準誤差を含む`Ok`が返る実バグがあり、`engine`側には
    専用の回帰テストがあるが、`method`の文字列パース〜`engine_pybind`配線を
    経由するAPI境界での確認が無かった。`testing-completeness-reviewer`指摘、
    Issue #231フェーズ4）。
    """
    df = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 4.0, 6.0, 8.0, 10.0],  # x2 = 2 * x1
        }
    )
    with pytest.raises(ComputationError):
        estimator_cls(
            df, y="y", x=["x1", "x2"], options=options_cls(method=method)
        ).fit()


def check_perfect_multicollinearity_raises_computation_error(
    estimator_cls, dataset_prefix
):
    """完全な多重共線性（合成データセット）は数値比較の対象外
    （`testing-policy.md`「テストの3系統」）。想定エラー（`ComputationError`）が
    発生することのみを確認する
    （`check_singular_hessian_raises_computation_error`はインラインの
    極小データ、こちらは`benchmark`のCSVフィクスチャ）。
    """
    df = pl.read_csv(
        DATA_DIR / f"{dataset_prefix}_perfect_multicollinearity.csv"
    )
    with pytest.raises(ComputationError):
        estimator_cls(df, y="y", x=["x1", "x2", "x3"]).fit()


def check_non_convergence_raises_computation_error_with_tiny_max_iter(
    dataset, estimator_cls, options_cls
):
    """`max_iter`を人為的に1に絞ると`raise_on_non_convergence=True`
    （既定）で`ComputationError`（engine側の`NonConvergence`）。

    完全分離等の病理的なデータは`NonConvergence`ではなく専用の
    `SeparationSuspected`（`ComputationError`のサブタイプ、
    `check_separation_suspected_raises_computation_error_for_near_separation_data`
    参照）を返すため、`NonConvergence`自体の発生確認には使えない。そのため
    `NonConvergence`の発生確認は、専用データセットに頼らずmax_iterを
    人為的に小さくする方法で行う（`docs/spec/<method>-spec.md`参照）。
    """
    with pytest.raises(ComputationError):
        estimator_cls(
            dataset,
            y="y",
            x=["x1", "x2"],
            options=options_cls(max_iter=1),
        ).fit()


def check_separation_suspected_raises_computation_error_for_near_separation_data(
    estimator_cls,
):
    """准完全分離データ（`x1`の真の係数を極端に大きくし、ほぼ全観測がx1の符号
    だけで完全に分類できるようにしたDGP）は`ComputationError`（engine側の
    `SeparationSuspected`）。

    勾配ノルム基準の収束判定が浮動小数点アンダーフローにより誤って
    「収束済み」と判定してしまう問題が、Python API境界を通しても正しく
    検出されエラーになることを確認する（`engine`側のRust単体テスト
    `fit_returns_separation_suspected_error_for_near_separation_data`
    のAPIレベル版）。
    """
    df = separation_suspected_dataset()

    with pytest.raises(ComputationError):
        estimator_cls(df, y="y", x=["x1", "x2"]).fit()


# ── test_<method>_reference.py ───────────────────────────────────────
#
# `FIXTURE_PATH`（`logit.json`/`probit.json`）・`SCENARIOS`
# （`generate_logit_fixtures`/`generate_probit_fixtures`由来）・
# `TOLERANCES`キー接頭辞等、手法ごとに異なる値が多いため、個別引数ではなく
# この設定オブジェクトにまとめて渡す（`refactoring-candidates-2.md`項目95）。


@dataclass(frozen=True)
class BinaryChoiceReferenceConfig:
    estimator_cls: type
    options_cls: type
    dataset_prefix: str  # "logit" / "probit"（CSVファイル名・statsmodels比較の識別用）
    fixture_path: Path
    scenarios: Sequence[str]
    cov_types: Sequence[str]
    rtol: float
    atol: float
    rtol_method: float
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
            ours, ref, label, rtol=self.rtol_method, atol=self.atol
        )

    def check_margeff(self, res, ref_margeff: dict, label: str) -> None:
        check_margeff(res, ref_margeff, label, rtol=self.rtol, atol=self.atol)


def check_result(
    config: BinaryChoiceReferenceConfig, res, ref: dict, label: str
) -> None:
    config.assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    config.assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    config.assert_dict_close(res.z_stats, ref["z_stats"], f"{label}/z_stats")
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
    kwargs = (
        {"tol": config.near_separation_tol}
        if scenario == "near_separation"
        else {}
    )
    options = config.options_cls(cov_type=cov_type, **kwargs)
    res = config.estimator_cls(
        df, y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    check_result(
        config, res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}"
    )


def check_cluster_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures
) -> None:
    """クラスターロバストSE（baselineシナリオ、行番号%10の疑似グループ）。"""
    df = pl.read_csv(config.dataset_path("baseline"))
    df = with_cluster_groups(df, 10)
    options = config.options_cls(
        cov_type="cluster", cluster_col="cluster_group"
    )
    res = config.estimator_cls(
        df, y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = fixtures["baseline"]["cluster"]
    config.assert_dict_close(res.params, ref["coef"], "cluster/coef")
    config.assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def check_cluster_imbalanced_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures
) -> None:
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。"""
    df = pl.read_csv(config.dataset_path("baseline"))
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = config.options_cls(
        cov_type="cluster", cluster_col="cluster_group"
    )
    res = config.estimator_cls(
        df, y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = fixtures["baseline"]["cluster_imbalanced"]
    config.assert_dict_close(
        res.params, ref["coef"], "cluster_imbalanced/coef"
    )
    config.assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


def check_cluster_g2_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures
) -> None:
    """クラスタ数境界（G=2ちょうど）の成功パス。

    OLSのwald_f_testと異なりLogit/Probitのcluster_cov_paramsはq×q部分行列の
    反転を要求しないため、説明変数を1個に絞る必要はない（k=3のままG=2で正常に
    計算できることを実機確認済み、`generate_<method>_fixtures.py`参照）。
    """
    df = pl.read_csv(config.dataset_path("baseline"))
    df = with_cluster_groups(df, 2)
    options = config.options_cls(
        cov_type="cluster", cluster_col="cluster_group"
    )
    res = config.estimator_cls(
        df, y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = fixtures["baseline"]["cluster_g2"]
    config.assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    config.assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


def check_method_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures, method
) -> None:
    """`method="bfgs"/"lbfgs"`が主リファレンス（statsmodelsの同じmethod）と
    フルの統計量（std_errors含む）で一致すること。

    既定の`method="newton"`のみ全シナリオ×cov_typeで数値照合しており、
    bfgs/lbfgsは`test_<method>_api.py::test_method_option_converges_to_same_params`
    で自身のnewton結果とparamsのみ緩い許容誤差(rel=1e-4)で比較していたが、
    主リファレンスに対するフルの統計量照合が無かった
    （`testing-completeness-reviewer`指摘、Issue #231フェーズ4）。
    """
    df = pl.read_csv(config.dataset_path("baseline"))
    options = config.options_cls(cov_type="classical", method=method)
    res = config.estimator_cls(
        df, y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = fixtures["method"][method]
    label = f"method/{method}"
    config.assert_dict_close_method(res.params, ref["coef"], f"{label}/coef")
    config.assert_dict_close_method(res.std_errors, ref["se"], f"{label}/se")
    assert res.converged == ref["converged"], f"{label}/converged"


def check_mroz_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures, cov_type
) -> None:
    """Wooldridge実データ（mroz、労働参加モデル）とのクロスチェック。"""
    df = load_wooldridge_dataset("mroz")
    options = config.options_cls(cov_type=cov_type)
    res = config.estimator_cls(df, y="inlf", x=MROZ_X, options=options).fit()

    check_result(config, res, fixtures["mroz"][cov_type], f"mroz/{cov_type}")


def check_mroz_cluster_matches_statsmodels(
    config: BinaryChoiceReferenceConfig, fixtures
) -> None:
    """実データでのクラスターロバストSE（`city`＝都市部居住ダミー、484/269の2値）。

    `testing-policy.md`「テスト用データセット」3.の「実データでのグループ列も
    検証する」を満たす（OLS/WLS・Logit/Probit間で同じ趣旨）。
    """
    df = load_wooldridge_dataset("mroz")
    options = config.options_cls(cov_type="cluster", cluster_col="city")
    res = config.estimator_cls(df, y="inlf", x=MROZ_X, options=options).fit()

    ref = fixtures["mroz"]["cluster"]
    config.assert_dict_close(res.params, ref["coef"], "mroz/cluster/coef")
    config.assert_dict_close(res.std_errors, ref["se"], "mroz/cluster/se")


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
    `testing-completeness-reviewer`指摘、Issue #231フェーズ4）。frozen
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
