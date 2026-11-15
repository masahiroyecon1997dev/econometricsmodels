"""ほぼ共線な傾きでも、全体のWald/F統計量と係数・標準誤差が、有理数の厳密値と一致すること。

OLS・WLS・IV 2SLS・FE・FE-IVは、共分散と全体のWald/F統計量を直交座標（QR）で求めるため、
傾きの相関が極端に高くても、`cov_type`ごとのサンドイッチ共分散が桁落ちで不正確にならない。
基準値は、データから有理数（`fractions`）で組んだ値（`tests/support/_exact_wald.py`）で、
返された`cov_params`は使わない。REは準偏差化の`θ`が無理数のため、厳密値を作れない
（REは単位テストと参照実装との照合で確かめる）。

受け入れる範囲は、列を揃えた条件数が約`5e7`以下（`test_collinearity_boundary.py`）。
`eps`は、傾きの相関が`1 - O(eps²)`になる強さ（`eps = 1e-7`で条件数は約`2.8e7`）。
"""

from __future__ import annotations

from functools import partial

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _exact_wald import (
    exact_coefficients,
    exact_projected_design,
    exact_slope_wald,
)
from _tolerances import TOLERANCES
from econometricsmodels import (
    FE,
    FEIV,
    IV,
    OLS,
    WLS,
    FEIVOptions,
    FEOptions,
    IVOptions,
    OLSOptions,
    WLSOptions,
)

assert_wald_close = partial(assert_close, **TOLERANCES["near_collinear_wald"])
assert_coef_close = partial(
    assert_close, **TOLERANCES["near_collinear_coefficient"]
)

COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "cluster"]
# FE・FE-IVは`hc0`を持たない。
PANEL_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]
EPS = [1e-5, 1e-6, 1e-7]

N_ENTITIES = 12
N_PERIODS = 5


def _ols_data(eps: float) -> pl.DataFrame:
    """`x2 = x1 + eps·ノイズ`（相関は`1 - O(eps²)`）の小さなデータ（40行、8クラスター）。"""
    n = 40
    i = np.arange(n)
    x1 = ((i * 7) % 13) - 6.0
    x2 = x1 + eps * (((i * 5) % 11) - 5.0)
    noise = np.where(i % 2 == 0, 0.7, -0.4) + 0.1 * (i % 5)
    return pl.DataFrame(
        {
            "y": 1.0 + 2.0 * x1 + 3.0 * x2 + noise,
            "x1": x1,
            "x2": x2,
            "w": 1.0 + 0.25 * (i % 4),
            "g": [f"g{k % 8}" for k in i],
        }
    )


def _iv_data(eps: float) -> pl.DataFrame:
    """外生変数`x1`・`x2`がほぼ共線、内生変数1個、操作変数3個（60行、8クラスター）。"""
    rng = np.random.default_rng(5)
    n = 60
    i = np.arange(n)
    x1 = ((i * 7) % 13) - 6.0
    x2 = x1 + eps * (((i * 5) % 11) - 5.0)
    z = np.column_stack(
        [((i * 3) % 7) - 3.0, ((i * 11) % 9) - 4.0, ((i * 2) % 5) - 2.0]
    )
    endog = z @ np.array([0.8, 0.5, 0.3]) + 0.4 * x1 + rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + 1.5 * endog + rng.normal(size=n)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "endog": endog,
            "z1": z[:, 0],
            "z2": z[:, 1],
            "z3": z[:, 2],
            "g": [f"g{k % 8}" for k in i],
        }
    )


def _panel_data(eps: float) -> pl.DataFrame:
    """個体12×時点5。`x3 = x1 + eps·ノイズ`、内生変数1個、操作変数3個。"""
    rng = np.random.default_rng(1)
    n = N_ENTITIES * N_PERIODS
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    x1 = rng.normal(size=n)
    x3 = x1 + eps * rng.normal(size=n)
    z = rng.normal(size=(n, 3))
    alpha = np.repeat(rng.normal(size=N_ENTITIES), N_PERIODS)
    endog = z @ np.array([0.8, 0.5, 0.3]) + 0.4 * x1 + rng.normal(size=n)
    y = 1 + x1 + 0.5 * x3 + endog + alpha + rng.normal(size=n)
    return pl.DataFrame(
        {
            "entity": entity,
            "y": y,
            "x1": x1,
            "x3": x3,
            "endog": endog,
            "z1": z[:, 0],
            "z2": z[:, 1],
            "z3": z[:, 2],
        }
    )


def _within(df: pl.DataFrame, columns: list[str]) -> list[list[float]]:
    """個体内で平均を引いた列を、行ごとのリスト（行×列）で返す。"""
    demeaned = df.select(
        [
            (pl.col(c) - pl.col(c).mean().over("entity")).alias(c)
            for c in columns
        ]
    )
    return demeaned.rows()


def _column(rows: list[list[float]], j: int) -> list[float]:
    return [row[j] for row in rows]


def _cluster_label(
    cov_type: str, df: pl.DataFrame, column: str
) -> list | None:
    return df[column].to_list() if cov_type == "cluster" else None


def _ols_options(cov_type: str) -> OLSOptions:
    return OLSOptions(
        cov_type=cov_type, cluster="g" if cov_type == "cluster" else None
    )


# ── OLS・WLS ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_ols_wald_matches_the_exact_value(cov_type, eps):
    df = _ols_data(eps)
    res = OLS(df, "y", ["x1", "x2"], options=_ols_options(cov_type)).fit()
    x = [[1.0, a, b] for a, b in zip(df["x1"], df["x2"], strict=True)]
    exact = exact_slope_wald(
        x,
        df["y"].to_list(),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "g"),
    )
    assert_wald_close(res.f_statistic * 2, exact, f"OLS {cov_type} eps={eps}")


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_wls_wald_matches_the_exact_value(cov_type, eps):
    """WLSは`sqrt(w)`倍した変数へのOLSと同じ。厳密値も、同じ`sqrt(w)`倍の値から求める。"""
    df = _ols_data(eps)
    options = WLSOptions(
        cov_type=cov_type, cluster="g" if cov_type == "cluster" else None
    )
    res = WLS(df, "y", ["x1", "x2"], "w", options=options).fit()
    root_w = np.sqrt(df["w"].to_numpy())
    x = np.column_stack(
        [root_w, df["x1"].to_numpy() * root_w, df["x2"].to_numpy() * root_w]
    )
    exact = exact_slope_wald(
        x.tolist(),
        (df["y"].to_numpy() * root_w).tolist(),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "g"),
    )
    assert_wald_close(res.f_statistic * 2, exact, f"WLS {cov_type} eps={eps}")


# ── FE ────────────────────────────────────────────────────────────────


def _fe_exact(df: pl.DataFrame, cov_type: str) -> float:
    rows = _within(df, ["x1", "x3", "y"])
    n = df.height
    return exact_slope_wald(
        [row[:2] for row in rows],
        _column(rows, 2),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "entity"),
        n_constants=0,
        df_resid=n - 2 - N_ENTITIES,
        cluster_k=3,  # 個体でクラスターすると固定効果がネストされ、K = k + 1
        extra_leverage=[1 / N_PERIODS] * n,  # 固定効果ダミーのレバレッジ
    )


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", PANEL_COV_TYPES)
def test_fe_wald_matches_the_exact_value(cov_type, eps):
    df = _panel_data(eps)
    res = FE(
        df,
        y="y",
        x=["x1", "x3"],
        effects=["entity"],
        options=FEOptions(cov_type=cov_type),
    ).fit()
    assert_wald_close(
        res.f_statistic * 2,
        _fe_exact(df, cov_type),
        f"FE {cov_type} eps={eps}",
    )


# ── IV 2SLS・FE-IV ────────────────────────────────────────────────────


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_iv_wald_matches_the_exact_value(cov_type, eps):
    df = _iv_data(eps)
    res = IV(
        df,
        "y",
        ["x1", "x2"],
        ["endog"],
        ["z1", "z2", "z3"],
        options=IVOptions(
            cov_type=cov_type, cluster="g" if cov_type == "cluster" else None
        ),
    ).fit()
    x = [[1.0, *row] for row in df.select("x1", "x2", "endog").rows()]
    z = [[1.0, *row] for row in df.select("x1", "x2", "z1", "z2", "z3").rows()]
    exact = exact_slope_wald(
        x,
        df["y"].to_list(),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "g"),
        x_hat=exact_projected_design(z, x),
    )
    assert_wald_close(
        res.wald_statistic * res.wald_df_num, exact, f"IV {cov_type} eps={eps}"
    )


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", PANEL_COV_TYPES)
def test_fe_iv_wald_matches_the_exact_value(cov_type, eps):
    df = _panel_data(eps)
    res = FEIV(
        df,
        y="y",
        x_exog=["x1", "x3"],
        x_endog=["endog"],
        instruments=["z1", "z2", "z3"],
        effects=["entity"],
        options=FEIVOptions(cov_type=cov_type),
    ).fit()
    n = df.height
    rows = _within(df, ["x1", "x3", "endog", "z1", "z2", "z3", "y"])
    x = [row[:3] for row in rows]
    z = [[row[0], row[1], row[3], row[4], row[5]] for row in rows]
    exact = exact_slope_wald(
        x,
        _column(rows, 6),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "entity"),
        n_constants=0,
        df_resid=n - 3 - N_ENTITIES,
        cluster_k=4,  # 個体でクラスターすると固定効果がネストされ、K = k + 1
        extra_leverage=[1 / N_PERIODS] * n,
        x_hat=exact_projected_design(z, x),
    )
    assert_wald_close(
        res.wald_statistic * res.wald_df_num,
        exact,
        f"FE-IV {cov_type} eps={eps}",
    )


# ── 係数と標準誤差 ────────────────────────────────────────────────────


@pytest.mark.parametrize("eps", EPS)
def test_ols_coefficients_and_standard_errors_match_the_exact_values(eps):
    """係数の誤差は、標準誤差を単位にして測る（ほぼ共線な方向の係数は標準誤差が大きく、
    その幅に比べて数値誤差が十分小さいこと）。標準誤差は相対で比べる。"""
    df = _ols_data(eps)
    res = OLS(df, "y", ["x1", "x2"]).fit()
    x = [[1.0, a, b] for a, b in zip(df["x1"], df["x2"], strict=True)]
    beta, se = exact_coefficients(x, df["y"].to_list())
    for name, b_exact, se_exact in zip(res.param_names, beta, se, strict=True):
        assert_coef_close(
            abs(res.params[name] - b_exact) / se_exact,
            0.0,
            f"OLS {name} error in standard errors, eps={eps}",
        )
        assert_coef_close(
            res.std_errors[name] / se_exact,
            1.0,
            f"OLS {name} standard error ratio, eps={eps}",
        )


@pytest.mark.parametrize("eps", EPS)
def test_fe_coefficients_and_standard_errors_match_the_exact_values(eps):
    df = _panel_data(eps)
    res = FE(
        df,
        y="y",
        x=["x1", "x3"],
        effects=["entity"],
        options=FEOptions(cov_type="classical"),
    ).fit()
    rows = _within(df, ["x1", "x3", "y"])
    beta, se = exact_coefficients(
        [row[:2] for row in rows],
        _column(rows, 2),
        df_resid=df.height - 2 - N_ENTITIES,
    )
    for name, b_exact, se_exact in zip(res.param_names, beta, se, strict=True):
        assert_coef_close(
            abs(res.params[name] - b_exact) / se_exact,
            0.0,
            f"FE {name} error in standard errors, eps={eps}",
        )
        assert_coef_close(
            res.std_errors[name] / se_exact,
            1.0,
            f"FE {name} standard error ratio, eps={eps}",
        )
