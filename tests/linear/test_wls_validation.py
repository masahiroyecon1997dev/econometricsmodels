"""WLS の入力・オプションのバリデーション（`ValidationError` パス）の検証。

重み列固有の検証（欠損・非正・NaN・`y`/`x` との衝突）と、OLS と共通化された
経路の検証（cov_type・cluster・confidence_level・hac_lags 等）。想定した例外
クラスが送出されることのみを確認する（`.claude/rules/testing-policy.md`
「テストの3系統」）。成功パスの構造・API は `test_wls_api.py`、主リファレンス
との数値照合は `test_wls_reference.py`、R クロスチェックは
`test_wls_crosscheck.py`。
"""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import DATA_DIR, with_cluster_groups
from econometricsmodels import (
    WLS,
    ComputationError,
    OLSOptions,
    ValidationError,
)

from benchmark.linear.constants import HAC_MAXLAGS
from benchmark.linear.fixtures.generate_wls_fixtures import COV_TYPES

# ── ValidationError（重み列固有） ─────────────────────────────────


def test_missing_weight_column_raises(dataset):
    with pytest.raises(ValidationError):
        WLS(dataset, y="y", x=["x1", "x2"], weight="nonexistent").fit()


def test_weight_equals_y_raises(dataset):
    """`weight`に`y`と同じ列名を指定した場合`ValidationError`。"""
    with pytest.raises(ValidationError):
        WLS(dataset, y="y", x=["x1", "x2"], weight="y").fit()


def test_weight_in_x_raises(dataset):
    """`weight`が`x`にも含まれる場合`ValidationError`。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2", "weight"], weight="weight").fit()


@pytest.mark.parametrize("bad_weight", [0.0, -1.0])
def test_non_positive_weight_raises(dataset, bad_weight):
    """重みに0以下の値が含まれる場合`ValidationError`（analytic weightとして不正）。"""
    n = dataset.height
    weight = [1.0] * (n - 1) + [bad_weight]
    df = dataset.with_columns(pl.Series("weight", weight))
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()


def test_nan_weight_raises(dataset):
    """重みにNaNが含まれる場合`ValidationError`。"""
    n = dataset.height
    weight = [1.0] * (n - 1) + [float("nan")]
    df = dataset.with_columns(pl.Series("weight", weight))
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()


def test_null_weight_raises(dataset):
    """重みに欠損値（null）が含まれる場合`ValidationError`。"""
    n = dataset.height
    weight: list[float | None] = [1.0] * (n - 1) + [None]
    df = dataset.with_columns(pl.Series("weight", weight))
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()


# ── ValidationError（入力データ、OLSと共通の検証） ───────────────


def test_y_in_x_raises(dataset):
    """`y`と同じ列名が`x`にも含まれる場合`ValidationError`（OLSと同じ検証）。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["y", "x1"], weight="weight").fit()


def test_duplicate_x_column_raises(dataset):
    """`x`に同じ列名が重複して含まれる場合`ValidationError`（OLSと同じ検証）。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x1"], weight="weight").fit()


def test_const_collision_with_include_intercept_raises():
    """`include_intercept=True`のとき`x`に`"const"`を含めると`ValidationError`。"""
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "const": [1.0, 2.0, 3.5],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["const"], weight="weight").fit()


def test_empty_x_raises(dataset):
    """`x`が空リストの場合`ValidationError`（OLSと同じ検証）。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=[], weight="weight").fit()


def test_insufficient_observations_raises(dataset):
    """観測数nが説明変数の数k（定数項込み）以下の場合`ValidationError`。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight")).head(2)
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()


def test_missing_column_raises(dataset):
    """`y`/`x`に存在しない列名を指定した場合`ValidationError`
    （`weight`列自体の検証は`test_missing_weight_column_raises`が対象、
    OLSと同じ検証）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "nonexistent"], weight="weight").fit()


def test_null_values_raise():
    """`y`/`x`に欠損値が含まれる場合`ValidationError`（OLSと同じ検証、
    `weight`列自体の欠損値検証は`test_null_weight_raises`が対象）。
    """
    df = pl.DataFrame(
        {"y": [1.0, None, 3.0], "x1": [1.0, 2.0, 3.0], "weight": [1.0] * 3}
    )
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1"], weight="weight").fit()


def test_non_numeric_dtype_raises():
    """`y`が非数値型の場合`ValidationError`（OLSと同じ検証）。"""
    df = pl.DataFrame(
        {"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0], "weight": [1.0] * 3}
    )
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1"], weight="weight").fit()


# ── ValidationError（オプション、OLSと共通化された経路） ─────────


def test_invalid_cov_type_raises(dataset):
    """`cov_type`が未知の文字列の場合`ValidationError`
    （OLSと同じ検証、共通化された経路）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = OLSOptions(cov_type="invalid")
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight", options=options).fit()


def test_cluster_without_col_raises(dataset):
    """`cov_type="cluster"`なのに`cluster_col`未指定の場合`ValidationError`
    （OLSと同じ検証、共通化された経路）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = OLSOptions(cov_type="cluster")
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight", options=options).fit()


def test_cluster_col_nonexistent_column_raises(dataset):
    """`cluster_col`が実在しない列名を指すと`ValidationError`
    （`test_ols_validation.py`と同じ理由、Issue #231フェーズ4）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = OLSOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight", options=options).fit()


def test_insufficient_clusters_raises(dataset):
    """クラスターが1種類しかない場合`ValidationError`（OLSと同じ検証、
    共通化された経路）。
    """
    df = dataset.with_columns(
        pl.lit(1.0).alias("weight"), pl.lit(0).alias("single_cluster")
    )
    options = OLSOptions(cov_type="cluster", cluster_col="single_cluster")
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight", options=options).fit()


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(dataset, confidence_level):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合
    `ValidationError`（OLSと同じ検証、共通化された経路）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = OLSOptions(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight", options=options).fit()


@pytest.mark.parametrize(
    "hac_lags", [-1, 100]
)  # 100 == dataset の n_obs（上限側境界）
def test_invalid_hac_lags_raises(dataset, hac_lags):
    """`hac_lags`が`[0, n)`の範囲外の場合`ValidationError`（OLSと同じ検証、
    共通化された経路）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = OLSOptions(cov_type="hac", hac_lags=hac_lags)
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight", options=options).fit()


# ── ComputationError ──────────────────────────────────────────────


def test_cluster_g2_with_multiple_slopes_raises_computation_error():
    """G=2×説明変数3個（傾き係数q=3）は、ロバストWald検定の共分散部分行列
    （3x3）のランクがクラスタ数G=2以下になり必然的に特異になるため、
    fit()全体がComputationErrorになる（OLSと同じ挙動）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 2)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    with pytest.raises(ComputationError):
        WLS(
            df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
        ).fit()


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。
    想定エラー（`ComputationError`）が発生することのみを確認する
    （OLS・Logitと同じ凍結CSVパターンに統一）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        WLS(df, y="y", x=["x1", "x2", "x3"], weight="weight").fit()


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_scale_variance_raises_computation_error(cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    傾き係数の同時共分散部分行列がスケール比の2乗（≈1e18）相当の条件数を持ち
    倍精度浮動小数点の限界を超えて数値的に特異になる（OLSと同じ理由、
    `test_ols_validation.py`参照。WLSでも実測確認済み）。
    数値比較はせずエラーパスのみ確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_scale_variance.csv")
    kwargs = {"hac_lags": HAC_MAXLAGS} if cov_type == "hac" else {}
    options = OLSOptions(cov_type=cov_type, **kwargs)
    with pytest.raises(ComputationError):
        WLS(
            df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
        ).fit()
