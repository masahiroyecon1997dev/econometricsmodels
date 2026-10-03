"""全手法共通の`stat_dist`/`stat_df`（`test_stats`の分布と自由度）のテスト。

`test_stats`の名前と分布情報が手法によらず同じ形で取れること、および`stat_df`が
実際に使われた自由度（`df_resid`とは限らない。OLS・FE・RE・2SLSの
`cov_type="cluster"`は`G-1`、FE・REの`cov_type="dk"`は`T-1`）であることを確認する。各手法の数値検証は手法ごとのreference/crosscheckテストが担う。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    FEOptions,
    IVOptions,
    Logit,
    OLSOptions,
    Probit,
    REOptions,
    Tobit,
    TobitOptions,
)

N_ENTITIES = 20
N_PERIODS = 10
N = N_ENTITIES * N_PERIODS


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(423)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    time = np.tile(np.arange(N_PERIODS), N_ENTITIES)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    z1 = rng.normal(size=N)
    z2 = rng.normal(size=N)
    endog = 0.5 * z1 + 0.5 * z2 + rng.normal(size=N)
    latent = 1.0 + x1 - 0.5 * x2 + rng.normal(size=N)
    return pl.DataFrame(
        {
            "y": latent,
            "y_bin": (latent > 1.0).astype(int),
            "y_cens": np.maximum(latent, 0.5),
            "x1": x1,
            "x2": x2,
            "z1": z1,
            "z2": z2,
            "endog": endog,
            "w": rng.uniform(0.5, 2.0, size=N),
            "entity": entity,
            "time": time,
        }
    )


def _check_t(res, df_expected: int) -> None:
    assert res.stat_dist == "t"
    assert res.stat_df == df_expected
    assert set(res.test_stats) == set(res.params)


def _check_normal(res) -> None:
    assert res.stat_dist == "normal"
    assert res.stat_df is None
    assert set(res.test_stats) == set(res.params)


def test_ols_classical_uses_t_with_residual_df(df):
    res = OLS(df, y="y", x=["x1", "x2"]).fit()
    _check_t(res, N - 3)


def test_ols_cluster_uses_g_minus_one_df(df):
    options = OLSOptions(cov_type="cluster", cluster="entity")
    res = OLS(df, y="y", x=["x1", "x2"], options=options).fit()
    _check_t(res, N_ENTITIES - 1)


def test_wls_uses_t_with_residual_df(df):
    res = WLS(df, y="y", x=["x1", "x2"], weight="w").fit()
    _check_t(res, N - 3)


def test_fe_classical_uses_t_with_panel_residual_df(df):
    options = FEOptions(cov_type="classical")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()
    _check_t(res, res.df_resid)
    assert res.stat_df == N - N_ENTITIES - 2


def test_fe_cluster_uses_g_minus_one_df(df):
    """FEの既定`cov_type`は`"cluster"`（entity単位）で、自由度は`G-1`。"""
    res = FE(df, y="y", x=["x1", "x2"], entity="entity").fit()
    _check_t(res, N_ENTITIES - 1)


def test_fe_dk_uses_t_minus_one_df(df):
    options = FEOptions(cov_type="dk", dk_time="time")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()
    _check_t(res, N_PERIODS - 1)


def test_re_classical_uses_t_with_residual_df(df):
    options = REOptions(cov_type="classical")
    res = RE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()
    _check_t(res, res.df_resid)


def test_re_cluster_uses_g_minus_one_df(df):
    """REの既定`cov_type`は`"cluster"`（entity単位）で、自由度は`G-1`。"""
    res = RE(df, y="y", x=["x1", "x2"], entity="entity").fit()
    _check_t(res, N_ENTITIES - 1)


def test_re_dk_uses_t_minus_one_df(df):
    options = REOptions(cov_type="dk", time="time")
    res = RE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()
    _check_t(res, N_PERIODS - 1)


@pytest.mark.parametrize(
    ("estimator", "dist"), [("2sls", "t"), ("gmm", "normal")]
)
def test_iv_distribution_follows_estimator(df, estimator, dist):
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog"],
        instruments=["z1", "z2"],
        options=IVOptions(estimator=estimator),
    ).fit()
    assert res.stat_dist == dist
    if dist == "t":
        assert res.stat_df == res.df_resid
    else:
        assert res.stat_df is None
    assert set(res.test_stats) == set(res.params)


def test_iv_2sls_cluster_uses_g_minus_one_df(df):
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog"],
        instruments=["z1", "z2"],
        options=IVOptions(cov_type="cluster", cluster="entity"),
    ).fit()
    _check_t(res, N_ENTITIES - 1)


def test_logit_uses_normal(df):
    _check_normal(Logit(df, y="y_bin", x=["x1", "x2"]).fit())


def test_probit_uses_normal(df):
    _check_normal(Probit(df, y="y_bin", x=["x1", "x2"]).fit())


def test_tobit_uses_normal(df):
    options = TobitOptions(lower=0.5)
    _check_normal(Tobit(df, y="y_cens", x=["x1", "x2"], options=options).fit())


def test_test_stats_are_consistent_with_stat_df_for_t_dist(df):
    """`stat_df`から`test_stats`のp値を再計算でき、`p_values`と一致すること
    （`stat_df`が実際に使われた自由度であることの確認、cluster版）。
    """
    from scipy import stats as sps

    options = OLSOptions(cov_type="cluster", cluster="entity")
    res = OLS(df, y="y", x=["x1", "x2"], options=options).fit()
    for name, t in res.test_stats.items():
        p = 2.0 * sps.t.sf(abs(t), res.stat_df)
        assert p == pytest.approx(res.p_values[name], rel=1e-8)
