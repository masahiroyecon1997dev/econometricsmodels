"""OLS 実装の統合テスト。statsmodels との数値比較 + API 正確性を検証する。"""

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from econometricsmodels import OLS, OlsResults

# 係数・SE の許容絶対誤差（float64 の丸め誤差を考慮）
ATOL_COEF = 1e-8
ATOL_SE = 1e-5
ATOL_STAT = 1e-6


# ── statsmodels ラッパー ─────────────────────────────────────────────────────

def _sm_design(df: pl.DataFrame) -> np.ndarray:
    """定数列付き設計行列を返す（statsmodels と同じ列順）。"""
    return sm.add_constant(
        np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy()])
    )


def _sm_fit(df: pl.DataFrame, cov_type: str = "nonrobust") -> sm.regression.linear_model.RegressionResultsWrapper:
    y = df["y"].to_numpy()
    X = _sm_design(df)
    model = sm.OLS(y, X)
    if cov_type == "nonrobust":
        return model.fit()
    return model.fit(cov_type=cov_type.upper())


def _sm_fit_cluster(df: pl.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    y = df["y"].to_numpy()
    X = _sm_design(df)
    groups = df["cluster"].to_numpy()
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def _our_fit(df: pl.DataFrame, cov_type: str = "nonrobust") -> OlsResults:
    return OLS(df, y="y", x=["x1", "x2"], cov_type=cov_type).fit()


def _our_fit_cluster(df: pl.DataFrame) -> OlsResults:
    return OLS(df, y="y", x=["x1", "x2"], cov_type="cluster", cluster_col="cluster").fit()


# ── 係数の一致 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3"])
def test_params_match_statsmodels(dataset, cov_type):
    """回帰係数が statsmodels と一致すること（SE の種別に関わらず係数は同じ）。"""
    sm_res = _sm_fit(dataset, cov_type)
    our_res = _our_fit(dataset, cov_type)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.params):
        our_val = our_res.params[name]
        assert abs(our_val - sm_val) < ATOL_COEF, (
            f"[{cov_type}] params[{name}]: ours={our_val:.10f}, sm={sm_val:.10f}"
        )


# ── 標準誤差の一致 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3"])
def test_std_errors_match_statsmodels(dataset, cov_type):
    """標準誤差が statsmodels と一致すること。"""
    sm_res = _sm_fit(dataset, cov_type)
    our_res = _our_fit(dataset, cov_type)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.bse):
        our_val = our_res.std_errors[name]
        assert abs(our_val - sm_val) < ATOL_SE, (
            f"[{cov_type}] SE[{name}]: ours={our_val:.8f}, sm={sm_val:.8f}"
        )


def test_cluster_se_match_statsmodels(dataset):
    """クラスター標準誤差が statsmodels と一致すること。"""
    sm_res = _sm_fit_cluster(dataset)
    our_res = _our_fit_cluster(dataset)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.bse):
        our_val = our_res.std_errors[name]
        assert abs(our_val - sm_val) < ATOL_SE, (
            f"Cluster SE[{name}]: ours={our_val:.8f}, sm={sm_val:.8f}"
        )


# ── 適合度統計量の一致 ───────────────────────────────────────────────────────

def test_r_squared_match_statsmodels(dataset):
    """R² と adj-R² が statsmodels と一致すること。"""
    sm_res = _sm_fit(dataset)
    our_res = _our_fit(dataset)

    assert abs(our_res.r_squared - sm_res.rsquared) < ATOL_STAT, (
        f"R²: ours={our_res.r_squared}, sm={sm_res.rsquared}"
    )
    assert abs(our_res.r_squared_adj - sm_res.rsquared_adj) < ATOL_STAT, (
        f"adj-R²: ours={our_res.r_squared_adj}, sm={sm_res.rsquared_adj}"
    )


def test_f_statistic_match_statsmodels(dataset):
    """F 統計量が statsmodels と一致すること。"""
    sm_res = _sm_fit(dataset)
    our_res = _our_fit(dataset)

    assert abs(our_res.f_statistic - sm_res.fvalue) < 1e-4, (
        f"F: ours={our_res.f_statistic}, sm={sm_res.fvalue}"
    )


def test_residuals_sum_near_zero(dataset):
    """残差の和が 0 に近いこと（定数項あり OLS の性質）。"""
    our_res = _our_fit(dataset)
    resid_sum = our_res.residuals.sum()
    assert abs(resid_sum) < 1e-10, f"Σε̂ = {resid_sum}"


def test_fitted_plus_residuals_equals_y(dataset):
    """ŷ + ε̂ = y となること。"""
    our_res = _our_fit(dataset)
    y = dataset["y"].to_numpy()
    reconstructed = our_res.fitted_values.to_numpy() + our_res.residuals.to_numpy()
    np.testing.assert_allclose(reconstructed, y, atol=1e-10)


# ── エラーハンドリング ────────────────────────────────────────────────────────

def test_invalid_cov_type_raises(dataset):
    with pytest.raises(ValueError):
        OLS(dataset, y="y", x=["x1", "x2"], cov_type="invalid")


def test_cluster_without_col_raises(dataset):
    with pytest.raises(ValueError):
        OLS(dataset, y="y", x=["x1", "x2"], cov_type="cluster")


def test_missing_column_raises(dataset):
    with pytest.raises(ValueError):
        OLS(dataset, y="y", x=["x1", "nonexistent"])


def test_null_values_raise():
    df = pl.DataFrame({"y": [1.0, None, 3.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        OLS(df, y="y", x=["x1"])


def test_non_numeric_dtype_raises():
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError):
        OLS(df, y="y", x=["x1"])


# ── API 正確性 ───────────────────────────────────────────────────────────────

def test_residuals_is_polars_series(dataset):
    res = _our_fit(dataset)
    assert isinstance(res.residuals, pl.Series)
    assert len(res.residuals) == len(dataset)


def test_fitted_values_is_polars_series(dataset):
    res = _our_fit(dataset)
    assert isinstance(res.fitted_values, pl.Series)
    assert len(res.fitted_values) == len(dataset)


def test_to_frame_structure(dataset):
    res = _our_fit(dataset)
    df = res.to_frame()
    assert isinstance(df, pl.DataFrame)
    expected_cols = {"param", "coef", "std_err", "t_stat", "p_value", "conf_lower", "conf_upper"}
    assert expected_cols <= set(df.columns)
    assert len(df) == 3  # const, x1, x2


def test_conf_int_structure(dataset):
    res = _our_fit(dataset)
    ci = res.conf_int()
    assert isinstance(ci, pl.DataFrame)
    assert "lower" in ci.columns and "upper" in ci.columns
    assert len(ci) == 3
    # lower < upper for all rows
    assert (ci["lower"] < ci["upper"]).all()


def test_predict_on_training_data(dataset):
    """訓練データでの予測値が fitted_values と一致すること。"""
    res = _our_fit(dataset)
    preds = res.predict(dataset.select(["x1", "x2"]))
    assert isinstance(preds, pl.Series)
    np.testing.assert_allclose(
        preds.to_numpy(), res.fitted_values.to_numpy(), atol=1e-10
    )


def test_summary_contains_key_fields(dataset):
    res = _our_fit(dataset)
    s = res.summary()
    assert "OLS Regression Results" in s
    assert "const" in s
    assert "x1" in s
    assert "x2" in s


def test_nobs_and_df(dataset):
    res = _our_fit(dataset)
    assert res.nobs == 100
    assert res.df_resid == 97   # 100 - 3 (const, x1, x2)
    assert res.df_model == 2    # 2 regressors (excl. const)


def test_cov_type_label(dataset):
    for cov_type, expected_label in [
        ("nonrobust", "nonrobust"),
        ("hc1", "HC1"),
        ("cluster", "cluster"),
    ]:
        if cov_type == "cluster":
            res = _our_fit_cluster(dataset)
        else:
            res = _our_fit(dataset, cov_type)
        assert res.cov_type.lower() == expected_label.lower(), (
            f"Expected cov_type label '{expected_label}', got '{res.cov_type}'"
        )
