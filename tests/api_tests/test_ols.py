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


def test_validation_error_is_value_error():
    """ValidationErrorがValueErrorのサブクラスであること。

    素の`except ValueError`でも捕まえられる
    （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
    """
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    """ComputationErrorがRuntimeErrorのサブクラスであること。"""
    assert issubclass(ComputationError, RuntimeError)


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
