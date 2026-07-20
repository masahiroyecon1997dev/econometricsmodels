"""OLS python_packageラッパーの統合テスト。

statsmodelsとの数値比較（推定値の正しさ）と、確定済み設計
（`docs/planning/specs/ols-api-design.md`）通りのAPIになっていることの
両方を検証する。

Note:
    以前のこのファイルは設計確定前の草案で、`OLS(df, y="y", x=[...],
    cov_type=cov_type)`のようなフラットなキーワード引数渡しや
    `res.summary()`（テキスト整形。5章で「作らない」と確定済み）、
    `res.to_frame()`がpolars DataFrameを返す（同章「係数テーブルに
    polars DataFrameは使わない」と矛盾）等、確定済み設計と食い違う
    内容だった（7章「既知の不整合」参照）。Issue #15で確定済み設計に
    合わせて全面的に書き直した。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from econometricsmodels import (
    OLS,
    ComputationError,
    OLSOptions,
    OlsResults,
    ValidationError,
)

# 係数・SEの許容絶対誤差（float64の丸め誤差を考慮）
ATOL_COEF = 1e-8
ATOL_SE = 1e-5
ATOL_STAT = 1e-6


# ── statsmodels ラッパー ────────────────────────────────────────────


def _sm_design(df: pl.DataFrame) -> np.ndarray:
    """定数列付き設計行列を返す（statsmodelsと同じ列順）。"""
    return sm.add_constant(
        np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy()])
    )


def _sm_fit(df: pl.DataFrame, cov_type: str = "classical"):
    """statsmodelsでの推定。

    `use_t=True`を明示指定する（本プロジェクトはcov_typeによらず
    t分布で統一する方針だが、statsmodelsの既定は`cov_type="nonrobust"`
    以外でuse_t=False。`docs/planning/specs/ols-api-design.md`
    「検定分布」参照）。
    """
    y = df["y"].to_numpy()
    x = _sm_design(df)
    model = sm.OLS(y, x)
    if cov_type == "classical":
        return model.fit(use_t=True)
    return model.fit(cov_type=cov_type.upper(), use_t=True)


def _sm_fit_cluster(df: pl.DataFrame):
    y = df["y"].to_numpy()
    x = _sm_design(df)
    groups = df["cluster"].to_numpy()
    return sm.OLS(y, x).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, use_t=True
    )


def _our_fit(df: pl.DataFrame, cov_type: str = "classical") -> OlsResults:
    options = OLSOptions(cov_type=cov_type)
    return OLS(df, y="y", x=["x1", "x2"], options=options).fit()


def _our_fit_cluster(df: pl.DataFrame) -> OlsResults:
    options = OLSOptions(cov_type="cluster", cluster_col="cluster")
    return OLS(df, y="y", x=["x1", "x2"], options=options).fit()


# ── 係数・標準誤差の一致 ────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", ["classical", "hc0", "hc1", "hc2", "hc3"])
def test_params_match_statsmodels(dataset, cov_type):
    """回帰係数がstatsmodelsと一致すること（cov_typeによらず係数は同じ）。"""
    sm_res = _sm_fit(dataset, cov_type)
    our_res = _our_fit(dataset, cov_type)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.params):
        our_val = our_res.params[name]
        assert abs(our_val - sm_val) < ATOL_COEF, (
            f"[{cov_type}] params[{name}]: "
            f"ours={our_val:.10f}, sm={sm_val:.10f}"
        )


@pytest.mark.parametrize("cov_type", ["classical", "hc0", "hc1", "hc2", "hc3"])
def test_std_errors_match_statsmodels(dataset, cov_type):
    """標準誤差がstatsmodelsと一致すること。"""
    sm_res = _sm_fit(dataset, cov_type)
    our_res = _our_fit(dataset, cov_type)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.bse):
        our_val = our_res.std_errors[name]
        assert abs(our_val - sm_val) < ATOL_SE, (
            f"[{cov_type}] SE[{name}]: ours={our_val:.8f}, sm={sm_val:.8f}"
        )


def test_cluster_se_match_statsmodels(dataset):
    """クラスター標準誤差がstatsmodelsと一致すること。"""
    sm_res = _sm_fit_cluster(dataset)
    our_res = _our_fit_cluster(dataset)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.bse):
        our_val = our_res.std_errors[name]
        assert abs(our_val - sm_val) < ATOL_SE, (
            f"Cluster SE[{name}]: ours={our_val:.8f}, sm={sm_val:.8f}"
        )


def test_hac_runs_and_returns_finite_std_errors(dataset):
    """HACが（statsmodelsとの数値照合なしで）エラーなく動作すること。"""
    options = OLSOptions(cov_type="hac", hac_lags=2)
    res = OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()

    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


# ── 適合度統計量の一致 ──────────────────────────────────────────────


def test_r_squared_match_statsmodels(dataset):
    """R²と調整済みR²がstatsmodelsと一致すること。"""
    sm_res = _sm_fit(dataset)
    our_res = _our_fit(dataset)

    assert abs(our_res.r_squared - sm_res.rsquared) < ATOL_STAT
    assert abs(our_res.r_squared_adj - sm_res.rsquared_adj) < ATOL_STAT


def test_f_statistic_match_statsmodels(dataset):
    """F統計量がstatsmodelsと一致すること。"""
    sm_res = _sm_fit(dataset)
    our_res = _our_fit(dataset)

    assert abs(our_res.f_statistic - sm_res.fvalue) < 1e-4


def test_residuals_sum_near_zero(dataset):
    """残差の和が0に近いこと（定数項ありOLSの性質）。"""
    our_res = _our_fit(dataset)
    assert abs(sum(our_res.residuals)) < 1e-8


# ── エラーハンドリング ──────────────────────────────────────────────


def test_invalid_cov_type_raises(dataset):
    options = OLSOptions(cov_type="invalid")
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_cluster_without_col_raises(dataset):
    options = OLSOptions(cov_type="cluster")
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_missing_column_raises(dataset):
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "nonexistent"]).fit()


def test_null_values_raise():
    df = pl.DataFrame({"y": [1.0, None, 3.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1"]).fit()


def test_non_numeric_dtype_raises():
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1"]).fit()


def test_singular_matrix_raises_computation_error():
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 4.0, 6.0, 8.0],  # x2 = 2 * x1（完全な多重共線性）
        }
    )
    with pytest.raises(ComputationError):
        OLS(df, y="y", x=["x1", "x2"]).fit()


def test_insufficient_observations_raises(dataset):
    """観測数nが説明変数の数k（定数項込み）以下の場合`ValidationError`

    （Issue #30で追加。engine側の`OlsError::InsufficientObservations`は
    Rust単体テストでのみ検証されており、Python API境界からは未検証だった）。
    """
    df = dataset.head(2)  # n=2、include_intercept=trueでk=3（const, x1, x2）
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1", "x2"]).fit()


def test_insufficient_clusters_raises(dataset):
    """クラスターが1種類しかない場合`ValidationError`

    （Issue #30で追加。従来は`cluster_col`が全く未指定のケースのみ
    検証されており、クラスター数不足自体はPython API境界から未検証だった）。
    """
    df = dataset.with_columns(pl.lit(0).alias("single_cluster"))
    options = OLSOptions(cov_type="cluster", cluster_col="single_cluster")
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(dataset, confidence_level):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合`ValidationError`

    （Issue #30で追加。従来は範囲を大きく外れた1.5のみRust単体テストで
    検証されており、Python API境界・境界値0.0は未検証だった）。
    """
    options = OLSOptions(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize(
    "hac_lags", [-1, 100]
)  # 100 == dataset の nobs（上限側境界）
def test_invalid_hac_lags_raises(dataset, hac_lags):
    """`hac_lags`が`[0, n)`の範囲外の場合`ValidationError`

    （Issue #30で追加。Rust単体テストでは範囲外の境界を検証済みだが、
    Python API境界からは未検証だった）。
    """
    options = OLSOptions(cov_type="hac", hac_lags=hac_lags)
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_y_in_x_raises(dataset):
    """`y`と同じ列名が`x`にも含まれる場合`ValidationError`

    （Issue #30で追加。`engine_pybind::fit`のy/x重複チェックが
    Python API境界から未検証だった）。
    """
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["y", "x1"]).fit()


def test_duplicate_x_column_raises(dataset):
    """`x`に同じ列名が重複して含まれる場合`ValidationError`（Issue #30で追加）。"""
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x1"]).fit()


def test_const_collision_with_include_intercept_raises():
    """`include_intercept=True`のとき`x`に`"const"`という列名を含めると

    自動追加される定数項と衝突し`ValidationError`（Issue #30で追加。
    `engine_pybind::fit`の対応する検証がPython API境界から未検証だった）。
    """
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "const": [1.0, 2.0, 3.5]})
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["const"]).fit()


def test_empty_x_raises(dataset):
    """`x`が空リストの場合`ValidationError`（Issue #30で追加）。"""
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=[]).fit()


def test_validation_error_is_value_error():
    """ValidationErrorがValueErrorのサブクラスであること。

    素の`except ValueError`でも捕まえられる
    （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
    """
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    """ComputationErrorがRuntimeErrorのサブクラスであること。"""
    assert issubclass(ComputationError, RuntimeError)


# ── オプションの反映確認（Issue #30で追加） ──────────────────────────
#
# 従来のtest_ols.pyはcov_type以外のOLSOptionsフィールド
# （include_intercept・confidence_level・hac_lags=None・time_col）を
# Python API境界からほぼ検証しておらず、対応するengine_pybind側の
# 列抽出・分岐ロジックが未検証だった。


def test_include_intercept_false_matches_statsmodels():
    """`include_intercept=False`でstatsmodelsと一致すること（uncentered TSSの

    R²等、Rust単体テストでは検証済みだがPython API境界からは未検証だった）。
    """
    rng = np.random.default_rng(7)
    n = 30
    x1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * x1 + rng.normal(0.0, 0.5, n)
    df = pl.DataFrame({"y": y, "x1": x1})

    sm_res = sm.OLS(y, x1.reshape(-1, 1)).fit(use_t=True)  # 定数項なし
    options = OLSOptions(include_intercept=False)
    our_res = OLS(df, y="y", x=["x1"], options=options).fit()

    assert our_res.param_names == ["x1"]
    assert abs(our_res.params["x1"] - sm_res.params[0]) < ATOL_COEF
    assert abs(our_res.std_errors["x1"] - sm_res.bse[0]) < ATOL_SE
    assert abs(our_res.r_squared - sm_res.rsquared) < ATOL_STAT


def test_confidence_level_changes_interval_width(dataset):
    """`confidence_level`を下げると信頼区間が狭くなること

    （既定の0.95以外の値がengine_pybind経由で実際に反映されることの確認。
    従来はどのテストも既定値0.95のままでしか検証していなかった）。
    """
    wide = OLS(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=OLSOptions(confidence_level=0.99),
    ).fit()
    narrow = OLS(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=OLSOptions(confidence_level=0.80),
    ).fit()

    for name in ["const", "x1", "x2"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width, name


def test_hac_auto_lags_runs_and_returns_finite_std_errors(dataset):
    """`hac_lags`省略時（`None`、自動計算式）でもエラーなく動作すること

    （既存の`test_hac_runs_and_returns_finite_std_errors`は`hac_lags=2`を
    明示していたため、`None`がPython→Rustに正しく伝播する経路は
    未検証だった）。
    """
    options = OLSOptions(cov_type="hac")  # hac_lags省略 = 自動計算
    res = OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()

    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


def test_hac_time_col_reorders_rows_before_computing_lags():
    """`time_col`を指定すると、DataFrameの行順に関わらず時系列順で

    ラグ付き自己共分散を計算すること。データは`engine/src/linear/ols.rs`の
    `fit_computes_hac_std_errors_respecting_time_order`と同一（時系列順で
    x=[1..5], y=[2,4,5,4,5]をtime順=[3,1,5,2,4]にシャッフルして入力し、
    `time_col`無指定・時系列順の入力と同じ結果になることを確認する）。
    engine_pybindの`time_col`列抽出（`extract_f64_column`）はこれまで
    Python API境界から一度も検証されていなかった（Issue #30で追加）。
    """
    ordered_df = pl.DataFrame(
        {"y": [2.0, 4.0, 5.0, 4.0, 5.0], "x1": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    ordered_options = OLSOptions(cov_type="hac", hac_lags=1)
    ordered_res = OLS(
        ordered_df, y="y", x=["x1"], options=ordered_options
    ).fit()

    shuffled_df = pl.DataFrame(
        {
            "y": [5.0, 2.0, 5.0, 4.0, 4.0],
            "x1": [3.0, 1.0, 5.0, 2.0, 4.0],
            "time": [3.0, 1.0, 5.0, 2.0, 4.0],
        }
    )
    shuffled_options = OLSOptions(cov_type="hac", hac_lags=1, time_col="time")
    shuffled_res = OLS(
        shuffled_df, y="y", x=["x1"], options=shuffled_options
    ).fit()

    for name in ["const", "x1"]:
        assert (
            abs(shuffled_res.std_errors[name] - ordered_res.std_errors[name])
            < 1e-9
        ), name


# ── API構造 ─────────────────────────────────────────────────────────


def test_coef_table_structure(dataset):
    res = _our_fit(dataset)
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
    res = _our_fit(dataset)
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_t_stats_p_values_share_keys(dataset):
    res = _our_fit(dataset)
    expected_keys = {"const", "x1", "x2"}

    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.t_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_nobs_and_dep_var_name(dataset):
    res = _our_fit(dataset)
    assert res.nobs == 100
    assert res.dep_var_name == "y"


def test_cov_type_label(dataset):
    for cov_type in ["classical", "hc0", "hc1", "hc2", "hc3"]:
        res = _our_fit(dataset, cov_type)
        assert res.cov_type == cov_type

    res = _our_fit_cluster(dataset)
    assert res.cov_type == "cluster"


def test_default_options_use_classical():
    """`options`省略時は`OLSOptions()`の既定値（classical）が使われること。"""
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [1.0, 2.0, 3.5]})
    res = OLS(df, y="y", x=["x1"]).fit()
    assert res.cov_type == "classical"
