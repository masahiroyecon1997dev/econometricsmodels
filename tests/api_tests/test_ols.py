"""OLS python_packageラッパーの統合テスト。

statsmodelsとの数値比較（推定値の正しさ）と、確定済み設計
（`docs/spec/ols-spec.md`）通りのAPIになっていることの
両方を検証する。
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
    以外でuse_t=False。`docs/spec/ols-spec.md`
    「標準誤差」参照）。
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
    """観測数nが説明変数の数k（定数項込み）以下の場合`ValidationError`。"""
    df = dataset.head(2)  # n=2、include_intercept=trueでk=3（const, x1, x2）
    with pytest.raises(ValidationError):
        OLS(df, y="y", x=["x1", "x2"]).fit()


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
)  # 100 == dataset の nobs（上限側境界）
def test_invalid_hac_lags_raises(dataset, hac_lags):
    """`hac_lags`が`[0, n)`の範囲外の場合`ValidationError`。"""
    options = OLSOptions(cov_type="hac", hac_lags=hac_lags)
    with pytest.raises(ValidationError):
        OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()


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


def test_validation_error_is_value_error():
    """ValidationErrorがValueErrorのサブクラスであること。

    素の`except ValueError`でも捕まえられる
    （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
    """
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    """ComputationErrorがRuntimeErrorのサブクラスであること。"""
    assert issubclass(ComputationError, RuntimeError)


# ── オプションの反映確認 ────────────────────────────────────────────
#
# cov_type以外のOLSOptionsフィールド（include_intercept・confidence_level・
# hac_lags=None・time_col）が、engine_pybind側の列抽出・分岐ロジックを経て
# 正しく反映されることを確認する。


def test_include_intercept_false_matches_statsmodels():
    """`include_intercept=False`でstatsmodelsと一致すること（uncentered TSSのR²等）。"""
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

    （既定の0.95以外の値がengine_pybind経由で実際に反映されることの確認）。
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
    engine_pybindの`time_col`列抽出（`extract_f64_column`）を
    Python API境界から検証する。
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


# ── predict() ──────────────────────────────────────────────────────


def test_predict_none_matches_statsmodels_fitted_values(dataset):
    """`predict(new_data=None)`が学習データに対するstatsmodelsのfittedvaluesと一致すること。"""
    sm_res = _sm_fit(dataset)
    res = _our_fit(dataset)

    predicted = res.predict()

    assert len(predicted) == len(dataset)
    for row, expected in zip(predicted, sm_res.fittedvalues):
        assert row["fitted"] == pytest.approx(expected, abs=ATOL_COEF)


def test_predict_new_data_matches_statsmodels(dataset):
    """新規データに対する`predict()`がstatsmodelsの`.predict()`と一致すること。

    列順を学習時（x1, x2）と入れ替えて渡し、列名でマッチングされる
    （列順に依存しない）ことも合わせて確認する。
    """
    res = _our_fit(dataset)
    sm_res = _sm_fit(dataset)

    new_data = pl.DataFrame({"x2": [0.5, -1.0, 2.0], "x1": [1.0, 2.0, -0.5]})
    predicted = res.predict(new_data)

    sm_new_x = sm.add_constant(
        np.column_stack(
            [
                new_data["x1"].to_numpy(),
                new_data["x2"].to_numpy(),
            ]
        )
    )
    expected = sm_res.predict(sm_new_x)

    assert len(predicted) == 3
    for row, exp in zip(predicted, expected):
        assert row["fitted"] == pytest.approx(exp, abs=ATOL_COEF)


def test_predict_new_data_without_intercept_matches_statsmodels():
    """`include_intercept=False`でfitした場合のpredict()もstatsmodelsと一致すること。"""
    rng = np.random.default_rng(7)
    n = 50
    x1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * x1 + rng.normal(0.0, 0.1, n)
    df = pl.DataFrame({"y": y, "x1": x1})

    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=["x1"], options=options).fit()
    sm_res = sm.OLS(y, x1.reshape(-1, 1)).fit(use_t=True)

    new_x1 = np.array([1.0, 2.0, -3.0])
    new_data = pl.DataFrame({"x1": new_x1})
    predicted = res.predict(new_data)
    expected = sm_res.predict(new_x1.reshape(-1, 1))

    for row, exp in zip(predicted, expected):
        assert row["fitted"] == pytest.approx(exp, abs=ATOL_COEF)


def test_predict_with_include_intercept_false_and_x_named_const():
    """`include_intercept=False`かつ`x`に`"const"`という名前の列を含む場合でも
    predict()が正しく動作すること。

    `include_intercept=True`のときのみ`"const"`という列名との衝突チェックが
    働く仕様のため（`ols-spec.md`「API引数」）、`include_intercept=False`なら
    ユーザーが`"const"`という名前の（切片ではない）通常の説明変数を`x`に
    含めることは正当な入力。`predict()`の内部実装が誤って列名から
    「自動追加された切片列かどうか」を推測すると、この場合に値を無視して
    1.0固定にしてしまう回帰バグがあったため、固定用に追加。
    """
    df = pl.DataFrame(
        {
            "y": [3.0, 7.0, 9.0, 19.0, 11.0],
            "const": [2.0, 5.0, 1.0, 8.0, 3.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=["const", "x2"], options=options).fit()

    new_data = pl.DataFrame({"const": [100.0, 200.0], "x2": [10.0, 20.0]})
    predicted = res.predict(new_data)

    coef_const = res.params["const"]
    coef_x2 = res.params["x2"]
    for row, (c, x2) in zip(predicted, [(100.0, 10.0), (200.0, 20.0)]):
        expected = coef_const * c + coef_x2 * x2
        assert row["fitted"] == pytest.approx(expected, abs=ATOL_COEF)


def test_predict_new_data_structure(dataset):
    res = _our_fit(dataset)
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    predicted = res.predict(new_data)

    assert isinstance(predicted, list)
    assert len(predicted) == 2
    for row in predicted:
        assert set(row.keys()) == {"fitted"}
        assert isinstance(row["fitted"], float)


def test_predict_missing_column_raises(dataset):
    res = _our_fit(dataset)
    new_data = pl.DataFrame({"x1": [1.0, 2.0]})  # x2が無い

    with pytest.raises(ValidationError):
        res.predict(new_data)


def test_predict_non_numeric_dtype_raises(dataset):
    res = _our_fit(dataset)
    new_data = pl.DataFrame({"x1": ["a", "b"], "x2": [1.0, 2.0]})

    with pytest.raises(ValidationError):
        res.predict(new_data)


def test_predict_null_or_non_finite_values_raise(dataset):
    res = _our_fit(dataset)

    new_data_null = pl.DataFrame({"x1": [1.0, None], "x2": [1.0, 2.0]})
    with pytest.raises(ValidationError):
        res.predict(new_data_null)

    new_data_inf = pl.DataFrame({"x1": [1.0, float("inf")], "x2": [1.0, 2.0]})
    with pytest.raises(ValidationError):
        res.predict(new_data_inf)
