"""全検定統計量の自由度公開（`*_df`/`f_df_num`/`f_df_denom`/`wald_*`）のテスト。

統計量・自由度・p値が分布に従って整合する（自由度から`p_value`を再計算できる）ことを、
手法ごとに確認する。数値の正しさ自体（統計量・p値）は各手法のreference/crosscheckテストが担う。
"""

from __future__ import annotations

import math

import pytest
from _cross_method_helpers import (
    FE_DF_RESID_ONE_WAY,
    FE_DF_RESID_TWO_WAY,
    N_COARSE,
    N_ENTITIES,
    N_GROUPS,
    N_PERIODS,
    N_PERIODS_COARSE,
    RE_DF_RESID,
    N,
    fit_fe,
    fit_re,
)
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
    Tobit,
    TobitOptions,
)
from scipy import stats as sps


def _f_p(stat, num, denom):
    return float(sps.f.sf(stat, num, denom))


def _chi2_p(stat, df_):
    return float(sps.chi2.sf(stat, df_))


def _iv(panel_df, **options):
    return IV(
        panel_df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog"],
        instruments=["z1", "z2", "z3"],
        options=IVOptions(**options),
    ).fit()


# ── 既定のcov_typeでのF/Wald検定の自由度 ──────────────────────────────


def test_ols_df_resid_df_model_and_f_dfs(panel_df):
    res = OLS(panel_df, y="y", x=["x1", "x2"]).fit()
    assert res.df_resid == N - 3
    assert res.df_model == 2
    assert (res.f_df_num, res.f_df_denom) == (2, N - 3)
    assert _f_p(res.f_statistic, 2, N - 3) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_ols_without_intercept_df_model_counts_all_columns(panel_df):
    res = OLS(
        panel_df, y="y", x=["x1"], options=OLSOptions(include_intercept=False)
    ).fit()
    assert res.df_model == 1
    assert res.df_resid == N - 1
    assert (res.f_df_num, res.f_df_denom) == (1, N - 1)


def test_wls_df_resid_df_model_and_f_dfs(panel_df):
    res = WLS(panel_df, y="y", x=["x1", "x2"], weight="w").fit()
    assert (res.df_resid, res.df_model) == (N - 3, 2)
    assert (res.f_df_num, res.f_df_denom) == (2, N - 3)
    assert _f_p(res.f_statistic, 2, N - 3) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


@pytest.mark.parametrize("model", [Logit, Probit])
def test_logit_probit_lr_df(panel_df, model):
    res = model(panel_df, y="y_bin", x=["x1", "x2"]).fit()
    assert res.lr_df == 2
    assert _chi2_p(res.lr_statistic, 2) == pytest.approx(
        res.lr_p_value, rel=1e-6
    )


def test_tobit_wald_dist_and_df(panel_df):
    res = Tobit(
        panel_df, y="y_cens", x=["x1", "x2"], options=TobitOptions(lower=0.5)
    ).fit()
    assert res.wald_dist == "chi2"
    assert res.wald_df == 2
    assert _chi2_p(res.wald_statistic, 2) == pytest.approx(
        res.wald_p_value, rel=1e-6
    )


def test_iv_2sls_wald_is_f_type(panel_df):
    res = _iv(panel_df)
    assert res.wald_dist == "f"
    assert (res.wald_df_num, res.wald_df_denom) == (2, res.df_resid)
    assert _f_p(res.wald_statistic, 2, res.df_resid) == pytest.approx(
        res.wald_p_value, rel=1e-8
    )


def test_iv_gmm_wald_is_chi2_type(panel_df):
    res = _iv(panel_df, estimator="gmm")
    assert res.wald_dist == "chi2"
    assert res.wald_df_num == 2
    assert res.wald_df_denom is None
    assert _chi2_p(res.wald_statistic, 2) == pytest.approx(
        res.wald_p_value, rel=1e-8
    )


def test_fe_f_dfs(panel_df):
    """FEのF検定の分母自由度は`t`検定と同じ`df_inference`（classicalは
    `df_resid`、clusterは`G-1`、dkは`T-1`）。"""
    classical = FE(
        panel_df,
        y="y",
        x=["x1", "x2"],
        effects=["entity"],
        options=FEOptions(cov_type="classical"),
    ).fit()
    assert (classical.f_df_num, classical.f_df_denom) == (
        2,
        classical.df_resid,
    )
    assert _f_p(classical.f_statistic, 2, classical.df_resid) == pytest.approx(
        classical.f_p_value, rel=1e-8
    )

    cluster = FE(panel_df, y="y", x=["x1", "x2"], effects=["entity"]).fit()
    assert (cluster.f_df_num, cluster.f_df_denom) == (2, N_ENTITIES - 1)
    assert _f_p(cluster.f_statistic, 2, N_ENTITIES - 1) == pytest.approx(
        cluster.f_p_value, rel=1e-8
    )

    dk = FE(
        panel_df,
        y="y",
        x=["x1", "x2"],
        effects=["entity"],
        options=FEOptions(cov_type="dk", dk_time="time"),
    ).fit()
    assert (dk.f_df_num, dk.f_df_denom) == (2, N_PERIODS - 1)
    assert _f_p(dk.f_statistic, 2, N_PERIODS - 1) == pytest.approx(
        dk.f_p_value, rel=1e-8
    )


def test_re_f_dfs_and_hausman_df(panel_df):
    # 既定の`cov_type`は`cluster`（entityクラスター）なので分母自由度は`G-1`。
    res = RE(panel_df, y="y", x=["x1", "x2"], effects=["entity"]).fit()
    assert (res.f_df_num, res.f_df_denom) == (2, N_ENTITIES - 1)
    assert _f_p(res.f_statistic, 2, N_ENTITIES - 1) == pytest.approx(
        res.f_p_value, rel=1e-8
    )
    assert res.hausman_df == 2


# ── cov_typeに連動する分母自由度（cluster: G-1、dk: T-1） ─────────────


def test_ols_cluster_f_denominator_is_g_minus_one(panel_df):
    options = OLSOptions(cov_type="cluster", cluster="entity")
    res = OLS(panel_df, y="y", x=["x1", "x2"], options=options).fit()
    assert res.df_resid == N - 3
    assert (res.f_df_num, res.f_df_denom) == (2, N_ENTITIES - 1)
    assert _f_p(res.f_statistic, 2, N_ENTITIES - 1) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_iv_2sls_cluster_wald_denominator_is_g_minus_one(panel_df):
    res = _iv(panel_df, cov_type="cluster", cluster="entity")
    assert (res.wald_df_num, res.wald_df_denom) == (2, N_ENTITIES - 1)


@pytest.mark.parametrize("two_way", [False, True])
@pytest.mark.parametrize("cov_type", ["hc1", "hc2", "hc3"])
def test_fe_hc_f_denominator_is_df_resid(panel_df, cov_type, two_way):
    res = fit_fe(panel_df, cov_type=cov_type, two_way=two_way)
    expected = FE_DF_RESID_TWO_WAY if two_way else FE_DF_RESID_ONE_WAY
    assert res.df_resid == expected
    assert (res.f_df_num, res.f_df_denom) == (2, expected)
    assert _f_p(res.f_statistic, 2, expected) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_fe_two_way_f_dfs_follow_cov_type(panel_df):
    classical = fit_fe(panel_df, cov_type="classical", two_way=True)
    assert (classical.f_df_num, classical.f_df_denom) == (
        2,
        FE_DF_RESID_TWO_WAY,
    )
    assert _f_p(
        classical.f_statistic, 2, FE_DF_RESID_TWO_WAY
    ) == pytest.approx(classical.f_p_value, rel=1e-8)

    cluster = fit_fe(panel_df, cov_type="cluster", two_way=True)
    assert (cluster.f_df_num, cluster.f_df_denom) == (2, N_ENTITIES - 1)
    assert _f_p(cluster.f_statistic, 2, N_ENTITIES - 1) == pytest.approx(
        cluster.f_p_value, rel=1e-8
    )

    dk = fit_fe(panel_df, cov_type="dk", dk_time="time", two_way=True)
    assert (dk.f_df_num, dk.f_df_denom) == (2, N_PERIODS - 1)
    assert _f_p(dk.f_statistic, 2, N_PERIODS - 1) == pytest.approx(
        dk.f_p_value, rel=1e-8
    )


@pytest.mark.parametrize("two_way", [False, True])
@pytest.mark.parametrize(
    ("cluster", "n_clusters"),
    [
        ("entity", N_ENTITIES),
        ("grp", N_GROUPS),
        ("coarse", N_COARSE),
        ("time", N_PERIODS),
    ],
)
def test_fe_cluster_f_denominator_is_g_minus_one(
    panel_df, cluster, n_clusters, two_way
):
    """F検定の分母自由度も、`cluster`列のクラスター数`G`で`G-1`になる。"""
    res = fit_fe(
        panel_df, cov_type="cluster", cluster=cluster, two_way=two_way
    )
    assert (res.f_df_num, res.f_df_denom) == (2, n_clusters - 1)
    assert _f_p(res.f_statistic, 2, n_clusters - 1) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


@pytest.mark.parametrize(
    ("options", "n_periods"),
    [
        pytest.param({"dk_time": "time"}, N_PERIODS, id="1way-dk_time"),
        pytest.param(
            {"two_way": True, "dk_time": "time"}, N_PERIODS, id="2way-dk_time"
        ),
        pytest.param(
            {"dk_time": "time5"}, N_PERIODS_COARSE, id="1way-dk_time-coarse"
        ),
        pytest.param(
            {"two_way": True, "dk_time": "time5"},
            N_PERIODS_COARSE,
            id="2way-dk_time-overrides-time",
        ),
    ],
)
def test_fe_dk_f_denominator_is_t_minus_one(panel_df, options, n_periods):
    res = fit_fe(panel_df, cov_type="dk", **options)
    assert (res.f_df_num, res.f_df_denom) == (2, n_periods - 1)
    assert _f_p(res.f_statistic, 2, n_periods - 1) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


@pytest.mark.parametrize(
    ("options", "expected_df_denom"),
    [
        ({"cov_type": "classical"}, RE_DF_RESID),
        ({"cov_type": "hc1"}, RE_DF_RESID),
        ({"cov_type": "hc2"}, RE_DF_RESID),
        ({"cov_type": "hc3"}, RE_DF_RESID),
        ({"cov_type": "cluster"}, N_ENTITIES - 1),
        ({"cov_type": "cluster", "cluster": "grp"}, N_GROUPS - 1),
        ({"cov_type": "cluster", "cluster": "coarse"}, N_COARSE - 1),
        ({"cov_type": "cluster", "cluster": "time"}, N_PERIODS - 1),
        ({"cov_type": "dk", "dk_time": "time"}, N_PERIODS - 1),
        ({"cov_type": "dk", "dk_time": "time5"}, N_PERIODS_COARSE - 1),
    ],
)
def test_re_f_denominator_follows_cov_type(
    panel_df, options, expected_df_denom
):
    """REのF統計量はFEと同じく`cov_type`に連動するWald検定で、分母自由度は
    `cluster`で`G-1`、`dk`で`T-1`、それ以外は`df_resid`。t検定・信頼区間の
    `stat_df`と常に一致する。"""
    res = fit_re(panel_df, **options)
    assert res.df_resid == RE_DF_RESID
    assert (res.f_df_num, res.f_df_denom) == (2, expected_df_denom)
    assert res.f_df_denom == res.stat_df
    assert _f_p(res.f_statistic, 2, expected_df_denom) == pytest.approx(
        res.f_p_value, rel=1e-8
    )


def test_re_f_statistic_depends_on_cov_type(panel_df):
    """`classical`以外はロバスト共分散のWald検定のため、F統計量が`classical`と
    異なる（`cov_type`に連動していることの確認）。"""
    base = fit_re(panel_df, cov_type="classical")
    for options in (
        {"cov_type": "hc1"},
        {"cov_type": "cluster", "cluster": "grp"},
        {"cov_type": "dk", "dk_time": "time"},
    ):
        res = fit_re(panel_df, **options)
        assert res.f_statistic != pytest.approx(base.f_statistic, rel=1e-6)


# ── 不均衡パネル ──────────────────────────────────────────────────────


#
# 時点を欠かせ（時点0は全entityで欠測、entityごとに観測数が異なる）、クラスターの
# サイズも偏らせる。期待値は実装を呼ばずデータから直接数える。2-way FEは均衡パネル
# 前提のため、不均衡側は1-way FEとREのみ。


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_unbalanced_fe_re_f_denominator_is_df_resid(
    panel_df_unbalanced, cov_type
):
    n = panel_df_unbalanced.height
    n_entities = panel_df_unbalanced["entity"].n_unique()
    fe = fit_fe(panel_df_unbalanced, cov_type=cov_type)
    assert fe.f_df_denom == n - n_entities - 2
    assert _f_p(fe.f_statistic, 2, fe.f_df_denom) == pytest.approx(
        fe.f_p_value, rel=1e-8
    )
    re = fit_re(panel_df_unbalanced, cov_type=cov_type)
    assert re.f_df_denom == n - 3


@pytest.mark.parametrize("cluster", ["entity", "grp_u", "coarse_u"])
def test_unbalanced_fe_cluster_f_denominator_is_g_minus_one(
    panel_df_unbalanced, cluster
):
    g = panel_df_unbalanced[cluster].n_unique()
    fe = fit_fe(panel_df_unbalanced, cov_type="cluster", cluster=cluster)
    assert (fe.f_df_num, fe.f_df_denom) == (2, g - 1)
    assert _f_p(fe.f_statistic, 2, g - 1) == pytest.approx(
        fe.f_p_value, rel=1e-8
    )
    re = fit_re(panel_df_unbalanced, cov_type="cluster", cluster=cluster)
    assert (re.f_df_num, re.f_df_denom) == (2, g - 1)
    assert _f_p(re.f_statistic, 2, g - 1) == pytest.approx(
        re.f_p_value, rel=1e-8
    )


def test_unbalanced_dk_f_denominator_uses_observed_periods(
    panel_df_unbalanced,
):
    t = panel_df_unbalanced["time"].n_unique()
    fe = fit_fe(panel_df_unbalanced, cov_type="dk", dk_time="time")
    assert fe.f_df_denom == t - 1
    assert _f_p(fe.f_statistic, 2, t - 1) == pytest.approx(
        fe.f_p_value, rel=1e-8
    )
    re = fit_re(panel_df_unbalanced, cov_type="dk", dk_time="time")
    assert re.f_df_denom == t - 1
    assert _f_p(re.f_statistic, 2, t - 1) == pytest.approx(
        re.f_p_value, rel=1e-8
    )


# ── IVの診断検定の自由度 ──────────────────────────────────────────────


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_iv_overid_df(panel_df, estimator):
    """`len(instruments) - len(x_endog) = 3 - 1 = 2`（過剰識別）。"""
    res = _iv(panel_df, estimator=estimator)
    assert res.overid_df == 2
    assert _chi2_p(res.overid_statistic, 2) == pytest.approx(
        res.overid_p_value, rel=1e-6
    )


def test_iv_overid_df_is_none_when_just_identified(panel_df):
    res = IV(
        panel_df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog"],
        instruments=["z1"],
    ).fit()
    assert res.overid_statistic is None
    assert res.overid_df is None


def test_iv_wu_hausman_dfs(panel_df):
    res = _iv(panel_df)
    assert res.wu_hausman_df_num == 1
    # 拡張回帰（const, x1, endog, 第一段階残差）の残差自由度
    assert res.wu_hausman_df_denom == N - 4
    assert _f_p(res.wu_hausman_statistic, 1, N - 4) == pytest.approx(
        res.wu_hausman_p_value, rel=1e-6
    )


def test_iv_wu_hausman_dfs_are_none_for_gmm(panel_df):
    res = _iv(panel_df, estimator="gmm")
    assert res.wu_hausman_statistic is None
    assert res.wu_hausman_df_num is None
    assert res.wu_hausman_df_denom is None


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_iv_weak_instrument_f_dfs(panel_df, estimator):
    """分子=除外操作変数の数、分母=第一段階回帰の残差自由度（const, x1, z1〜z3）。"""
    res = _iv(panel_df, estimator=estimator)
    assert res.weak_instrument_f_df_num == 3
    assert res.weak_instrument_f_df_denom == N - 5
    assert all(
        math.isfinite(v) for v in res.weak_instrument_f_statistics.values()
    )
