"""OLS の入力・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」）。成功パスの構造・API は
`test_ols_api.py`、主リファレンスとの数値照合は `test_ols_reference.py`、
R クロスチェックは `test_ols_crosscheck.py`。
"""

from __future__ import annotations

import polars as pl
import pytest
from _constants import DATA_DIR
from _helpers import with_cluster_groups
from _ols_helpers import our_fit
from econometricsmodels import (
    OLS,
    ComputationError,
    OLSOptions,
    ValidationError,
)

from benchmark.linear.constants import HAC_MAXLAGS
from benchmark.linear.fixtures.generate_ols_fixtures import COV_TYPES

# ── ValidationError（入力データ） ───────────────────────────────────


def test_missing_column_raises(dataset):
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "nonexistent"]).fit()


def test_null_values_raise():
    df = pl.DataFrame({"y": [1.0, None, 3.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1"]).fit()


def test_non_finite_values_raise():
    """`y`/`x`にNaN・無限大が含まれる場合`ValidationError`。

    null（`test_null_values_raise`）とNaN/無限大は`column_extraction.rs`内で
    別ロジックのため個別に確認する（テスト網羅性レビューで判明した抜け）。
    """
    df_nan = pl.DataFrame(
        {"y": [1.0, float("nan"), 3.0], "x1": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValidationError):
        OLS(df_nan, y="y", x=["x1"]).fit()

    df_inf = pl.DataFrame(
        {"y": [1.0, float("inf"), 3.0], "x1": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValidationError):
        OLS(df_inf, y="y", x=["x1"]).fit()


def test_non_numeric_dtype_raises():
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1"]).fit()


def test_y_in_x_raises(dataset):
    """`y`と同じ列名が`x`にも含まれる場合`ValidationError`。"""
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["y", "x1"]).fit()


def test_duplicate_x_column_raises(dataset):
    """`x`に同じ列名が重複して含まれる場合`ValidationError`。"""
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x1"]).fit()


def test_const_collision_with_include_intercept_raises():
    """`include_intercept=True`のとき`x`に`"const"`という列名を含めると

    自動追加される定数項と衝突し`ValidationError`になること。
    """
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "const": [1.0, 2.0, 3.5]})
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["const"]).fit()


def test_empty_x_raises(dataset):
    """`x`が空リストの場合`ValidationError`。"""
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=[]).fit()


def test_insufficient_observations_raises(dataset):
    """観測数nが説明変数の数k（定数項込み）以下の場合`ValidationError`。"""
    df = dataset.head(2)  # n=2、include_intercept=trueでk=3（const, x1, x2）
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1", "x2"]).fit()


# ── ValidationError（オプション） ──────────────────────────────────


def test_invalid_cov_type_raises(dataset):
    options = OLSOptions(cov_type="invalid")
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_cluster_without_col_raises(dataset):
    options = OLSOptions(cov_type="cluster")
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_cluster_col_nonexistent_column_raises(dataset):
    """`cluster_col`が実在しない列名を指すと`ValidationError`
    （`column_extraction`の責務、既存の欠落を確認するテストが無かった、
    `testing-completeness-reviewer`指摘、Issue #231フェーズ4）。
    """
    options = OLSOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_insufficient_clusters_raises(dataset):
    """クラスターが1種類しかない場合`ValidationError`。"""
    df = dataset.with_columns(pl.lit(0).alias("single_cluster"))
    options = OLSOptions(cov_type="cluster", cluster_col="single_cluster")
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(dataset, confidence_level):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合`ValidationError`。"""
    options = OLSOptions(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize(
    "hac_lags", [-1, 100]
)  # 100 == dataset の n_obs（上限側境界）
def test_invalid_hac_lags_raises(dataset, hac_lags):
    """`hac_lags`が`[0, n)`の範囲外の場合`ValidationError`。"""
    options = OLSOptions(cov_type="hac", hac_lags=hac_lags)
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


# ── ValidationError（predict()） ──────────────────────────────────


def test_predict_missing_column_raises(dataset):
    res = our_fit(dataset)
    new_data = pl.DataFrame({"x1": [1.0, 2.0]})  # x2が無い

    with pytest.raises(ValidationError):
        res.predict(new_data)


def test_predict_non_numeric_dtype_raises(dataset):
    res = our_fit(dataset)
    new_data = pl.DataFrame({"x1": ["a", "b"], "x2": [1.0, 2.0]})

    with pytest.raises(ValidationError):
        res.predict(new_data)


def test_predict_null_or_non_finite_values_raise(dataset):
    res = our_fit(dataset)

    new_data_null = pl.DataFrame({"x1": [1.0, None], "x2": [1.0, 2.0]})
    with pytest.raises(ValidationError):
        res.predict(new_data_null)

    new_data_inf = pl.DataFrame({"x1": [1.0, float("inf")], "x2": [1.0, 2.0]})
    with pytest.raises(ValidationError):
        res.predict(new_data_inf)


# ── ComputationError ──────────────────────────────────────────────


def test_cluster_g2_with_multiple_slopes_raises_computation_error():
    """G=2×説明変数3個（傾き係数q=3）は、ロバストWald検定の共分散部分行列
    （3x3）のランクがクラスタ数G=2以下になり必然的に特異になるため、
    fit()全体がComputationErrorになる（係数・標準誤差自体は計算可能だが、
    F検定の失敗でfit()全体が失敗する仕様。実装中に判明、
    数値比較はしない想定）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 2)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    with pytest.raises(ComputationError):
        OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性（設計行列が特異）は数値比較の対象外
    （`testing-policy.md`「テストの3系統」）。想定エラー（`ComputationError`）が
    発生することのみを確認する。

    以前は手書きの極小 df（`x2 = 2*x1`）による `test_singular_matrix_raises_
    computation_error` も併存していたが、同じ経路の確認で追加検証が無かったため、
    固定済みベンチマーク CSV を使うこのテストへ一本化した
    （`refactoring-candidates-2.md` 項目54）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        OLS(df, y="y", x=["x1", "x2", "x3"]).fit()


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_scale_variance_raises_computation_error(cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    傾き係数の同時共分散部分行列がスケール比の2乗（≈1e18）相当の条件数を持ち
    倍精度浮動小数点の限界を超えて数値的に特異になる（発見・原因調査済み）。`wald_f_test`が固有値分解による相対閾値判定で検出し、
    全cov_typeで`ComputationError`になる（classicalも含む。傾き係数の
    共分散部分行列自体は`cov_type`によらず同じ条件数を持つため）。
    perfect_multicollinearityと同様、数値比較はせずエラーパスのみ確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_scale_variance.csv")
    kwargs = {"hac_lags": HAC_MAXLAGS} if cov_type == "hac" else {}
    options = OLSOptions(cov_type=cov_type, **kwargs)
    with pytest.raises(ComputationError):
        OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()


# ── 例外クラスの継承 ──────────────────────────────────────────────


def test_validation_error_is_value_error():
    """ValidationErrorがValueErrorのサブクラスであること。

    素の`except ValueError`でも捕まえられる
    （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
    """
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    """ComputationErrorがRuntimeErrorのサブクラスであること。"""
    assert issubclass(ComputationError, RuntimeError)
