"""全検定統計量の自由度公開（`*_df`/`f_df_num`/`f_df_denom`/`wald_*`）のテスト。

統計量・自由度・p値が分布に従って整合する（自由度から`p_value`を再計算できる）ことを、
手法ごとに確認する。数値の正しさ自体（統計量・p値）は各手法のreference/crosscheckテストが担う。
"""

from __future__ import annotations

import math

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
from scipy import stats as sps

N_ENTITIES = 20
N_PERIODS = 10
N = N_ENTITIES * N_PERIODS
N_GROUPS = 7


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(425)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    z1 = rng.normal(size=N)
    z2 = rng.normal(size=N)
    z3 = rng.normal(size=N)
    endog = 0.5 * z1 + 0.5 * z2 + rng.normal(size=N)
    latent = 1.0 + x1 - 0.5 * x2 + 0.5 * endog + rng.normal(size=N)
    return pl.DataFrame(
        {
            "y": latent,
            "y_bin": (latent > 1.0).astype(int),
            "y_cens": np.maximum(latent, 0.5),
            "x1": x1,
            "x2": x2,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "endog": endog,
            "w": rng.uniform(0.5, 2.0, size=N),
            "entity": entity,
            "time": np.tile(np.arange(N_PERIODS), N_ENTITIES),
            # entityと入れ子にならないクラスター列（G=7）
            "grp": np.arange(N) % N_GROUPS,
        }
    )


def _f_p(stat, num, denom):
    return float(sps.f.sf(stat, num, denom))


def _chi2_p(stat, df_):
    return float(sps.chi2.sf(stat, df_))


def test_ols_df_resid_df_model_and_f_dfs(df):
    res = OLS(df, y="y", x=["x1", "x2"]).fit()
    assert res.df_resid == N - 3
    assert res.df_model == 2
    assert (res.f_df_num, res.f_df_denom) == (2, N - 3)
    assert _f_p(res.f_statistic, 2, N - 3) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_ols_cluster_f_denominator_is_g_minus_one(df):
    options = OLSOptions(cov_type="cluster", cluster="entity")
    res = OLS(df, y="y", x=["x1", "x2"], options=options).fit()
    assert res.df_resid == N - 3
    assert (res.f_df_num, res.f_df_denom) == (2, N_ENTITIES - 1)
    assert _f_p(res.f_statistic, 2, N_ENTITIES - 1) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_ols_without_intercept_df_model_counts_all_columns(df):
    res = OLS(
        df, y="y", x=["x1"], options=OLSOptions(include_intercept=False)
    ).fit()
    assert res.df_model == 1
    assert res.df_resid == N - 1
    assert (res.f_df_num, res.f_df_denom) == (1, N - 1)


def test_wls_df_resid_df_model_and_f_dfs(df):
    res = WLS(df, y="y", x=["x1", "x2"], weight="w").fit()
    assert (res.df_resid, res.df_model) == (N - 3, 2)
    assert (res.f_df_num, res.f_df_denom) == (2, N - 3)
    assert _f_p(res.f_statistic, 2, N - 3) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_fe_f_dfs(df):
    """FEのF検定の分母自由度は`t`検定と同じ`df_inference`（classicalは
    `df_resid`、clusterは`G-1`、dkは`T-1`）。"""
    classical = FE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=FEOptions(cov_type="classical"),
    ).fit()
    assert (classical.f_df_num, classical.f_df_denom) == (
        2,
        classical.df_resid,
    )
    assert _f_p(classical.f_statistic, 2, classical.df_resid) == pytest.approx(
        classical.f_p_value, rel=1e-8
    )

    cluster = FE(df, y="y", x=["x1", "x2"], entity="entity").fit()
    assert (cluster.f_df_num, cluster.f_df_denom) == (2, N_ENTITIES - 1)
    assert _f_p(cluster.f_statistic, 2, N_ENTITIES - 1) == pytest.approx(
        cluster.f_p_value, rel=1e-8
    )

    dk = FE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=FEOptions(cov_type="dk", dk_time="time"),
    ).fit()
    assert (dk.f_df_num, dk.f_df_denom) == (2, N_PERIODS - 1)


def test_re_f_dfs_and_hausman_df(df):
    res = RE(df, y="y", x=["x1", "x2"], entity="entity").fit()
    assert (res.f_df_num, res.f_df_denom) == (2, res.df_resid)
    assert _f_p(res.f_statistic, 2, res.df_resid) == pytest.approx(
        res.f_p_value, rel=1e-8
    )
    assert res.hausman_df == 2


def _fe(df, *, two_way=False, **options):
    if two_way:
        options["time"] = "time"
    return FE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=FEOptions(**options),
    ).fit()


def _re(df, **options):
    return RE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=REOptions(**options),
    ).fit()


@pytest.mark.parametrize("two_way", [False, True])
@pytest.mark.parametrize("cov_type", ["hc1", "hc2", "hc3"])
def test_fe_hc_f_denominator_is_df_resid(df, cov_type, two_way):
    res = _fe(df, cov_type=cov_type, two_way=two_way)
    assert (res.f_df_num, res.f_df_denom) == (2, res.df_resid)
    assert _f_p(res.f_statistic, 2, res.df_resid) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_fe_two_way_f_dfs_follow_cov_type(df):
    classical = _fe(df, cov_type="classical", two_way=True)
    assert classical.f_df_denom == classical.df_resid

    cluster = _fe(df, cov_type="cluster", two_way=True)
    assert cluster.f_df_denom == N_ENTITIES - 1

    dk = _fe(df, cov_type="dk", two_way=True)
    assert (dk.f_df_num, dk.f_df_denom) == (2, N_PERIODS - 1)
    assert _f_p(dk.f_statistic, 2, N_PERIODS - 1) == pytest.approx(
        dk.f_p_value, rel=1e-8
    )


@pytest.mark.parametrize("two_way", [False, True])
def test_fe_non_entity_cluster_f_denominator_is_g_minus_one(df, two_way):
    res = _fe(df, cov_type="cluster", cluster="grp", two_way=two_way)
    assert (res.f_df_num, res.f_df_denom) == (2, N_GROUPS - 1)
    assert _f_p(res.f_statistic, 2, N_GROUPS - 1) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


@pytest.mark.parametrize(
    "options",
    [
        {"cov_type": "classical"},
        {"cov_type": "hc1"},
        {"cov_type": "hc2"},
        {"cov_type": "hc3"},
        {"cov_type": "cluster"},
        {"cov_type": "cluster", "cluster": "grp"},
        {"cov_type": "dk", "time": "time"},
    ],
)
def test_re_f_denominator_is_df_resid_for_every_cov_type(df, options):
    """REのF統計量（SST/SSR方式）は`cov_type`に依存しないため、分母自由度は
    常に`df_resid`。一方`stat_df`（t検定・信頼区間）は`cluster`で`G-1`、`dk`で
    `T-1`に切り替わるので、この2つは`cluster`/`dk`で食い違う（不整合ではなく仕様）。
    """
    res = _re(df, **options)
    assert (res.f_df_num, res.f_df_denom) == (2, res.df_resid)
    assert _f_p(res.f_statistic, 2, res.df_resid) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_re_f_statistic_does_not_depend_on_cov_type(df):
    base = _re(df, cov_type="classical")
    for options in (
        {"cov_type": "hc1"},
        {"cov_type": "cluster", "cluster": "grp"},
        {"cov_type": "dk", "time": "time"},
    ):
        res = _re(df, **options)
        assert res.f_statistic == pytest.approx(base.f_statistic, rel=1e-12)
        assert res.f_p_value == pytest.approx(base.f_p_value, rel=1e-12)


def test_re_stat_df_differs_from_f_df_denom_under_cluster_and_dk(df):
    cluster = _re(df, cov_type="cluster", cluster="grp")
    assert (cluster.stat_df, cluster.f_df_denom) == (
        N_GROUPS - 1,
        cluster.df_resid,
    )
    dk = _re(df, cov_type="dk", time="time")
    assert (dk.stat_df, dk.f_df_denom) == (N_PERIODS - 1, dk.df_resid)


@pytest.mark.parametrize("model", [Logit, Probit])
def test_logit_probit_lr_df(df, model):
    res = model(df, y="y_bin", x=["x1", "x2"]).fit()
    assert res.lr_df == 2
    assert _chi2_p(res.lr_statistic, 2) == pytest.approx(
        res.lr_p_value, rel=1e-6
    )


def test_tobit_wald_dist_and_df(df):
    res = Tobit(
        df, y="y_cens", x=["x1", "x2"], options=TobitOptions(lower=0.5)
    ).fit()
    assert res.wald_dist == "chi2"
    assert res.wald_df == 2
    assert _chi2_p(res.wald_statistic, 2) == pytest.approx(
        res.wald_p_value, rel=1e-6
    )


def _iv(df, **options):
    return IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog"],
        instruments=["z1", "z2", "z3"],
        options=IVOptions(**options),
    ).fit()


def test_iv_2sls_wald_is_f_type(df):
    res = _iv(df)
    assert res.wald_dist == "f"
    assert (res.wald_df_num, res.wald_df_denom) == (2, res.df_resid)
    assert _f_p(res.wald_statistic, 2, res.df_resid) == pytest.approx(
        res.wald_p_value, rel=1e-8
    )


def test_iv_2sls_cluster_wald_denominator_is_g_minus_one(df):
    res = _iv(df, cov_type="cluster", cluster="entity")
    assert (res.wald_df_num, res.wald_df_denom) == (2, N_ENTITIES - 1)


def test_iv_gmm_wald_is_chi2_type(df):
    res = _iv(df, estimator="gmm")
    assert res.wald_dist == "chi2"
    assert res.wald_df_num == 2
    assert res.wald_df_denom is None
    assert _chi2_p(res.wald_statistic, 2) == pytest.approx(
        res.wald_p_value, rel=1e-8
    )


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_iv_overid_df(df, estimator):
    """`len(instruments) - len(x_endog) = 3 - 1 = 2`（過剰識別）。"""
    res = _iv(df, estimator=estimator)
    assert res.overid_df == 2
    assert _chi2_p(res.overid_statistic, 2) == pytest.approx(
        res.overid_p_value, rel=1e-6
    )


def test_iv_overid_df_is_none_when_just_identified(df):
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog"],
        instruments=["z1"],
    ).fit()
    assert res.overid_statistic is None
    assert res.overid_df is None


def test_iv_wu_hausman_dfs(df):
    res = _iv(df)
    assert res.wu_hausman_df_num == 1
    # 拡張回帰（const, x1, endog, 第一段階残差）の残差自由度
    assert res.wu_hausman_df_denom == N - 4
    assert _f_p(res.wu_hausman_statistic, 1, N - 4) == pytest.approx(
        res.wu_hausman_p_value, rel=1e-6
    )


def test_iv_wu_hausman_dfs_are_none_for_gmm(df):
    res = _iv(df, estimator="gmm")
    assert res.wu_hausman_statistic is None
    assert res.wu_hausman_df_num is None
    assert res.wu_hausman_df_denom is None


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_iv_weak_instrument_f_dfs(df, estimator):
    """分子=除外操作変数の数、分母=第一段階回帰の残差自由度（const, x1, z1〜z3）。"""
    res = _iv(df, estimator=estimator)
    assert res.weak_instrument_f_df_num == 3
    assert res.weak_instrument_f_df_denom == N - 5
    assert all(
        math.isfinite(v) for v in res.weak_instrument_f_statistics.values()
    )
