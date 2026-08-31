"""Logit の入力・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」）。成功パスの構造・API・
オプション反映は `test_logit_api.py`、主リファレンス（statsmodels）との
数値照合は `test_logit_reference.py`、R クロスチェックは
`test_logit_crosscheck.py`（OLS/WLS の `test_<手法>_validation.py` 等と同じ
4分割、`refactoring-candidates-2.md` 項目68）。
"""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import DATA_DIR, separation_suspected_dataset
from econometricsmodels import (
    ComputationError,
    Logit,
    LogitOptions,
    ValidationError,
)

# ── ValidationError（入力データ） ───────────────────────────────────


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


def test_null_values_raise():
    """欠損値は`column_extraction`の責務で`ValidationError`（OLSの
    `test_null_values_raise`と同型、Python API境界で未検証だった、
    Issue #231フェーズ4）。
    """
    df = pl.DataFrame({"y": [0.0, None, 1.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        Logit(df, y="y", x=["x1"]).fit()


def test_non_numeric_dtype_raises():
    """数値/文字列型にキャストできない列は`ValidationError`（OLSの
    `test_non_numeric_dtype_raises`と同型、Issue #231フェーズ4）。
    """
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        Logit(df, y="y", x=["x1"]).fit()


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


# ── ValidationError（オプション） ──────────────────────────────────


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


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(binary_dataset, confidence_level):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合`ValidationError`。

    `marginal_effects(confidence_level=1.5)`側は`test_marginal_effects_
    confidence_level_out_of_range_raises`で既存だが、`fit()`本体側
    （`LogitOptions.confidence_level`）が未検証だった（`testing-policy.md`
    「テストの3系統」・OLS/WLSの`test_invalid_confidence_level_raises`との非対称、
    Issue #231フェーズ4）。
    """
    options = LogitOptions(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        Logit(binary_dataset, y="y", x=["x1", "x2"], options=options).fit()


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


@pytest.mark.parametrize("max_iter", [0, -1])
def test_non_positive_max_iter_raises(binary_dataset, max_iter):
    """`max_iter<=0`は`ValidationError`（engine側の`MleError::InvalidMaxIter`）。

    `tol<=0`側は`test_non_positive_tol_raises`で既存だが、対応する`max_iter`側の
    Python API境界のテストが無かった（`testing-completeness-reviewer`指摘、
    Issue #231フェーズ4）。
    """
    with pytest.raises(ValidationError):
        Logit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=LogitOptions(max_iter=max_iter),
        ).fit()


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


def test_cluster_col_nonexistent_column_raises(binary_dataset):
    """`cluster_col`が実在しない列名を指すと`ValidationError`（OLSと同じ理由、
    Issue #231フェーズ4）。
    """
    options = LogitOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(ValidationError):
        Logit(binary_dataset, y="y", x=["x1", "x2"], options=options).fit()


# ── ValidationError（marginal_effects()） ─────────────────────────


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


# ── ComputationError ──────────────────────────────────────────────


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_singular_hessian_raises_computation_error(method):
    """完全な多重共線性は`ComputationError`。

    `method`をparametrizeしているのは、`newton`は`newton_step`内のピボット付き
    QR分解経由でたまたま特異性を検出できていたが、`bfgs`/`lbfgs`は準ニュートン法
    のため`newton_step`を経由せず、収束後の`observed_information_cov_params`
    呼び出しが唯一の検出経路になるという構造的な違いがあるため
    （`docs/planning/specs/nonlinear-implementation-notes.md`「`cov_type`共通行列演算
    の特異性検出」参照。過去に`bfgs`だけ検出漏れし桁違いに巨大な標準誤差を含む`Ok`
    が返る実バグがあり、`engine`側には専用の回帰テストがあるが、`method`の文字列
    パース〜`engine_pybind`配線を経由するAPI境界での確認が無かった。
    `testing-completeness-reviewer`指摘、Issue #231フェーズ4）。
    """
    df = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 4.0, 6.0, 8.0, 10.0],  # x2 = 2 * x1
        }
    )
    with pytest.raises(ComputationError):
        Logit(
            df, y="y", x=["x1", "x2"], options=LogitOptions(method=method)
        ).fit()


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性（合成データセット）は数値比較の対象外
    （`testing-policy.md`「テストの3系統」）。想定エラー（`ComputationError`）が
    発生することのみを確認する（`test_singular_hessian_raises_computation_error`
    はインラインの極小データ、こちらは`benchmark`のCSVフィクスチャ）。
    """
    df = pl.read_csv(DATA_DIR / "logit_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        Logit(df, y="y", x=["x1", "x2", "x3"]).fit()


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
