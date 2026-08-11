"""WLS python_packageラッパーの統合テスト。

構造・API・エラーパスの検証、および**OLSとの不変条件回帰テスト**
を行う。主リファレンス（statsmodels）・独立実装（R）との
数値比較は`test_wls_fixtures.py`・`test_wls_crosscheck.py`で行う。

役割分担はOLSの`test_ols.py`と同じ3分割:
    - 構造・API・エラーパスの検証、OLSとの不変条件回帰テスト: このファイル
    - 主リファレンス（statsmodels）との厳密な数値一致: `test_wls_fixtures.py`
    - 独立実装（R）とのクロスチェック: `test_wls_crosscheck.py`
"""

from __future__ import annotations

import polars as pl
import pytest
from econometricsmodels import (
    OLS,
    WLS,
    OLSOptions,
    ValidationError,
    WlsResults,
)

# ── OLSとの不変条件回帰テスト ───────────────────────────────────────
#
# 「共通化・パフォーマンス改善で実装の中身が変わっても結果が変わらない」ことを
# 保証する目的のテスト。内部実装（sqrt(w)変換方式か将来別方式に変わるか等）が
# 変わっても壊れないよう、public API経由でのみ比較する。


@pytest.mark.parametrize(
    "options",
    [
        OLSOptions(),
        OLSOptions(cov_type="hc3"),
        OLSOptions(cov_type="cluster", cluster_col="cluster"),
        OLSOptions(include_intercept=False),
    ],
)
def test_weight_one_matches_ols(dataset, options):
    """重み=1のときWLSの結果がOLSの結果と完全一致すること。

    coef/se/t/p/CI/F統計量/n_obsは、WLSがOLSソルバーを`sqrt(weight)`変換した
    データにそのまま適用する実装であるため（weight=1なら変換が恒等写像になる）
    厳密な`==`で一致する。r_squared・log_likelihood（→aic/bic）・残差は、
    WLS側で元スケールのy・weightsから独立に計算し直す実装のため、加算順序
    等に由来する浮動小数点誤差レベルの差が生じうる（`engine/src/linear/wls.rs`
    の対応するRust単体テストで確認済みの挙動）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))

    ols_res = OLS(df, y="y", x=["x1", "x2"], options=options).fit()
    wls_res = WLS(
        df, y="y", x=["x1", "x2"], weight="weight", options=options
    ).fit()

    assert wls_res.param_names == ols_res.param_names
    for name in ols_res.param_names:
        assert wls_res.params[name] == ols_res.params[name], name
        assert wls_res.std_errors[name] == ols_res.std_errors[name], name
        assert wls_res.t_stats[name] == ols_res.t_stats[name], name
        assert wls_res.p_values[name] == ols_res.p_values[name], name
        assert wls_res.conf_int[name] == ols_res.conf_int[name], name

    assert wls_res.f_statistic == ols_res.f_statistic
    assert wls_res.f_p_value == ols_res.f_p_value
    assert wls_res.n_obs == ols_res.n_obs
    assert wls_res.dep_var_name == ols_res.dep_var_name
    assert wls_res.cov_type == ols_res.cov_type

    assert abs(wls_res.r_squared - ols_res.r_squared) < 1e-9
    assert abs(wls_res.r_squared_adj - ols_res.r_squared_adj) < 1e-9
    assert abs(wls_res.log_likelihood - ols_res.log_likelihood) < 1e-9
    assert abs(wls_res.aic - ols_res.aic) < 1e-9
    assert abs(wls_res.bic - ols_res.bic) < 1e-9
    for wls_r, ols_r in zip(wls_res.residuals, ols_res.residuals):
        assert abs(wls_r - ols_r) < 1e-9


def test_weight_one_matches_ols_coef_table(dataset):
    """coef_table()の内容もOLSと一致すること（辞書系プロパティ以外の確認）。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))

    ols_table = OLS(df, y="y", x=["x1", "x2"]).fit().coef_table()
    wls_table = (
        WLS(df, y="y", x=["x1", "x2"], weight="weight").fit().coef_table()
    )

    assert [row["param"] for row in wls_table] == [
        row["param"] for row in ols_table
    ]
    for wls_row, ols_row in zip(wls_table, ols_table):
        assert wls_row["coef"] == ols_row["coef"]
        assert wls_row["std_err"] == ols_row["std_err"]


# ── エラーハンドリング（重み固有） ──────────────────────────────────


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


def test_invalid_cov_type_raises(dataset):
    """`cov_type`が未知の文字列の場合`ValidationError`
    （OLSと同じ検証、Issue #153で共通化された経路）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = OLSOptions(cov_type="invalid")
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight", options=options).fit()


def test_cluster_without_col_raises(dataset):
    """`cov_type="cluster"`なのに`cluster_col`未指定の場合`ValidationError`
    （OLSと同じ検証、Issue #153で共通化された経路）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = OLSOptions(cov_type="cluster")
    with pytest.raises(ValidationError):
        WLS(df, y="y", x=["x1", "x2"], weight="weight", options=options).fit()


# ── API構造 ─────────────────────────────────────────────────────────


def test_default_options_use_classical(dataset):
    """`options`省略時は`OLSOptions()`の既定値（classical）が使われること。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert res.cov_type == "classical"


def test_residuals_are_original_scale_not_weighted(dataset):
    """`residuals`が元スケール（unweighted）であり、重み付き残差ではないこと。

    重みを大きく偏らせると、重み付き残差 `sqrt(w)(y-ŷ)` は元スケールの
    残差 `y-ŷ` と大きく異なる値になるはず。
    """
    n = dataset.height
    weight = [100.0] * n
    df = dataset.with_columns(pl.Series("weight", weight))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()

    # 元スケールの残差は「予測値からの素の差」であり、重み(=100)を掛けた
    # スケールにはならないはず（sqrt(100)=10倍にはならない）。
    ols_res = OLS(dataset, y="y", x=["x1", "x2"]).fit()
    for wls_r, ols_r in zip(res.residuals, ols_res.residuals):
        assert abs(wls_r - ols_r) < 1e-6


def test_result_is_wls_results_type(dataset):
    """`WLS.fit()`の返り値が`WlsResults`（`OlsResults`とは別型）であること。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert isinstance(res, WlsResults)


def test_coef_table_structure(dataset):
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    table = res.coef_table()

    assert isinstance(table, list)
    assert len(table) == 3  # const, x1, x2
    expected_keys = {
        "param",
        "coef",
        "std_err",
        "t_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert expected_keys <= set(row.keys())
    assert [row["param"] for row in table] == ["const", "x1", "x2"]


def test_conf_int_structure(dataset):
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_t_stats_p_values_share_keys(dataset):
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    expected_keys = {"const", "x1", "x2"}

    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.t_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_nobs_and_dep_var_name(dataset):
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert res.n_obs == 100
    assert res.dep_var_name == "y"
