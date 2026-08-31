"""Probit の入力・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」、`test_logit_validation.py`
と同型）。成功パスの構造・API・オプション反映は `test_probit_api.py`、
主リファレンス（statsmodels）との数値照合は `test_probit_reference.py`、
R クロスチェックは `test_probit_crosscheck.py`（OLS/WLS の
`test_<手法>_validation.py` 等と同じ4分割、`refactoring-candidates-2.md`
項目68）。
"""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import DATA_DIR, separation_suspected_dataset
from econometricsmodels import (
    ComputationError,
    Probit,
    ProbitOptions,
    ValidationError,
)

# ── ValidationError（入力データ） ───────────────────────────────────


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
    （`test_logit_validation.py`と同じ理由、Issue #231フェーズ4）。
    """
    df = pl.DataFrame({"y": [0.0, None, 1.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["x1"]).fit()


def test_non_numeric_dtype_raises():
    """数値/文字列型にキャストできない列は`ValidationError`
    （`test_logit_validation.py`と同じ理由、Issue #231フェーズ4）。
    """
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["x1"]).fit()


@pytest.mark.parametrize("bad_value", [0.5, 2.0, -1.0])
def test_non_binary_y_raises(binary_dataset, bad_value):
    """`y`が`{0.0, 1.0}`以外の値を含む場合は`ValidationError`
    （engine側の`MleError::InvalidBinaryY`、`test_logit_validation.py`と同じ検証）。
    """
    df = binary_dataset.with_columns(binary_dataset["y"].scatter(0, bad_value))
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["x1", "x2"]).fit()


def test_insufficient_observations_raises(binary_dataset):
    df = binary_dataset.head(2)
    with pytest.raises(ValidationError):
        Probit(df, y="y", x=["x1", "x2"]).fit()


# ── ValidationError（オプション） ──────────────────────────────────


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
    （`test_logit_validation.py`と同じ理由、Issue #231フェーズ4）。
    """
    options = ProbitOptions(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        Probit(binary_dataset, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize("tol", [0.0, -1.0])
def test_non_positive_tol_raises(binary_dataset, tol):
    """`tol<=0`は勾配ノルム基準の収束条件が理論上満たされないため`ValidationError`
    （engine側の`MleError::InvalidTol`、`test_logit_validation.py`と同じ検証）。
    """
    with pytest.raises(ValidationError):
        Probit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=ProbitOptions(tol=tol),
        ).fit()


@pytest.mark.parametrize("max_iter", [0, -1])
def test_non_positive_max_iter_raises(binary_dataset, max_iter):
    """`max_iter<=0`は`ValidationError`（`test_logit_validation.py`と同じ理由、
    Issue #231フェーズ4）。
    """
    with pytest.raises(ValidationError):
        Probit(
            binary_dataset,
            y="y",
            x=["x1", "x2"],
            options=ProbitOptions(max_iter=max_iter),
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
        Probit(
            df,
            y="y",
            x=["x1"],
            options=ProbitOptions(cov_type="cluster", cluster_col="cluster"),
        ).fit()


def test_cluster_col_nonexistent_column_raises(binary_dataset):
    """`cluster_col`が実在しない列名を指すと`ValidationError`（`test_logit_
    validation.py`と同じ理由、Issue #231フェーズ4）。
    """
    options = ProbitOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(ValidationError):
        Probit(binary_dataset, y="y", x=["x1", "x2"], options=options).fit()


# ── ValidationError（marginal_effects()） ─────────────────────────


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


# ── ComputationError ──────────────────────────────────────────────


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_singular_hessian_raises_computation_error(method):
    """完全な多重共線性は`ComputationError`。

    `method`のparametrize理由は`test_logit_validation.py`と同じ（`bfgs`/`lbfgs`は
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


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性（合成データセット）は数値比較の対象外
    （`testing-policy.md`「テストの3系統」）。想定エラー（`ComputationError`）が
    発生することのみを確認する（`test_singular_hessian_raises_computation_error`
    はインラインの極小データ、こちらは`benchmark`のCSVフィクスチャ）。
    """
    df = pl.read_csv(DATA_DIR / "probit_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        Probit(df, y="y", x=["x1", "x2", "x3"]).fit()


def test_non_convergence_raises_computation_error_with_tiny_max_iter(
    binary_dataset,
):
    """`max_iter`を人為的に1に絞ると`raise_on_non_convergence=True`（既定）で
    `ComputationError`（engine側の`NonConvergence`、`test_logit_validation.py`と
    同じ理由で専用データセットではなくmax_iterを小さくする方法を使う）。
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
    `SeparationSuspected`）。`test_logit_validation.py`のLogit版のProbit版。
    """
    df = separation_suspected_dataset()

    with pytest.raises(ComputationError):
        Probit(df, y="y", x=["x1", "x2"]).fit()
