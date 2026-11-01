"""全手法共通の`stat_dist`/`stat_df`（`test_stats`の分布と自由度）のテスト。

`test_stats`の名前と分布情報が手法によらず同じ形で取れること、および`stat_df`が
実際に使われた自由度（`df_resid`とは限らない。OLS・FE・RE・2SLSの
`cov_type="cluster"`は`G-1`、FE・REの`cov_type="dk"`は`T-1`）であることを確認する。各手法の数値検証は手法ごとのreference/crosscheckテストが担う。
"""

from __future__ import annotations

import polars as pl
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
    REOptions,
    Tobit,
    TobitOptions,
)
from scipy import stats as sps


def _check_t(res, df_expected: int) -> None:
    assert res.stat_dist == "t"
    assert res.stat_df == df_expected
    assert set(res.test_stats) == set(res.params)
    # p値も`stat_df`のt分布から再計算できる（`stat_df`の値だけでなく、
    # 実際にp値の計算へその自由度が使われていることまで固定する）。
    for name, t in res.test_stats.items():
        p = 2.0 * sps.t.sf(abs(t), res.stat_df)
        assert p == pytest.approx(res.p_values[name], rel=1e-8)


def _check_normal(res) -> None:
    assert res.stat_dist == "normal"
    assert res.stat_df is None
    assert set(res.test_stats) == set(res.params)


# ── 手法別の分布と自由度 ──────────────────────────────────────────────


def test_ols_classical_uses_t_with_residual_df(panel_df):
    res = OLS(panel_df, y="y", x=["x1", "x2"]).fit()
    _check_t(res, N - 3)


def test_ols_cluster_uses_g_minus_one_df(panel_df):
    options = OLSOptions(cov_type="cluster", cluster="entity")
    res = OLS(panel_df, y="y", x=["x1", "x2"], options=options).fit()
    _check_t(res, N_ENTITIES - 1)


def test_wls_uses_t_with_residual_df(panel_df):
    res = WLS(panel_df, y="y", x=["x1", "x2"], weight="w").fit()
    _check_t(res, N - 3)


def test_logit_uses_normal(panel_df):
    _check_normal(Logit(panel_df, y="y_bin", x=["x1", "x2"]).fit())


def test_probit_uses_normal(panel_df):
    _check_normal(Probit(panel_df, y="y_bin", x=["x1", "x2"]).fit())


def test_tobit_uses_normal(panel_df):
    options = TobitOptions(lower=0.5)
    _check_normal(
        Tobit(panel_df, y="y_cens", x=["x1", "x2"], options=options).fit()
    )


@pytest.mark.parametrize(
    ("estimator", "dist"), [("2sls", "t"), ("gmm", "normal")]
)
def test_iv_distribution_follows_estimator(panel_df, estimator, dist):
    res = IV(
        panel_df,
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


def test_iv_2sls_cluster_uses_g_minus_one_df(panel_df):
    res = IV(
        panel_df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog"],
        instruments=["z1", "z2"],
        options=IVOptions(cov_type="cluster", cluster="entity"),
    ).fit()
    _check_t(res, N_ENTITIES - 1)


def test_fe_classical_uses_t_with_panel_residual_df(panel_df):
    options = FEOptions(cov_type="classical")
    res = FE(
        panel_df, y="y", x=["x1", "x2"], effects=["entity"], options=options
    ).fit()
    _check_t(res, res.df_resid)
    assert res.stat_df == N - N_ENTITIES - 2


def test_fe_cluster_uses_g_minus_one_df(panel_df):
    """FEの既定`cov_type`は`"cluster"`（entity単位）で、自由度は`G-1`。"""
    res = FE(panel_df, y="y", x=["x1", "x2"], effects=["entity"]).fit()
    _check_t(res, N_ENTITIES - 1)


def test_fe_dk_uses_t_minus_one_df(panel_df):
    options = FEOptions(cov_type="dk", dk_time="time")
    res = FE(
        panel_df, y="y", x=["x1", "x2"], effects=["entity"], options=options
    ).fit()
    _check_t(res, N_PERIODS - 1)


def test_re_classical_uses_t_with_residual_df(panel_df):
    options = REOptions(cov_type="classical")
    res = RE(
        panel_df, y="y", x=["x1", "x2"], effects=["entity"], options=options
    ).fit()
    _check_t(res, res.df_resid)


def test_re_cluster_uses_g_minus_one_df(panel_df):
    """REの既定`cov_type`は`"cluster"`（entity単位）で、自由度は`G-1`。"""
    res = RE(panel_df, y="y", x=["x1", "x2"], effects=["entity"]).fit()
    _check_t(res, N_ENTITIES - 1)


def test_re_dk_uses_t_minus_one_df(panel_df):
    options = REOptions(cov_type="dk", dk_time="time")
    res = RE(
        panel_df, y="y", x=["x1", "x2"], effects=["entity"], options=options
    ).fit()
    _check_t(res, N_PERIODS - 1)


def test_test_stats_are_consistent_with_stat_df_for_t_dist(panel_df):
    """`stat_df`から`test_stats`のp値を再計算でき、`p_values`と一致すること
    （`stat_df`が実際に使われた自由度であることの確認、cluster版）。
    """
    from scipy import stats as sps

    options = OLSOptions(cov_type="cluster", cluster="entity")
    res = OLS(panel_df, y="y", x=["x1", "x2"], options=options).fit()
    for name, t in res.test_stats.items():
        p = 2.0 * sps.t.sf(abs(t), res.stat_df)
        assert p == pytest.approx(res.p_values[name], rel=1e-8)


# ── FE/REのstat_dfをcov_typeごとに網羅する ────────────────────────────


#
# `stat_df`は`cluster`のとき`G-1`、`dk`のとき`T-1`（時点数）、それ以外
# （`classical`/`hc1`〜`hc3`）は`df_resid`。`G`は`cluster`列のクラスター数で、
# 既定（`cluster`省略）ではentity数だが、他の列を指定すればそのクラスター数になる。


@pytest.mark.parametrize("two_way", [False, True])
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_fe_non_cluster_cov_types_use_residual_df(panel_df, cov_type, two_way):
    res = fit_fe(panel_df, cov_type=cov_type, two_way=two_way)
    expected = FE_DF_RESID_TWO_WAY if two_way else FE_DF_RESID_ONE_WAY
    assert res.df_resid == expected
    _check_t(res, expected)


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_re_non_cluster_cov_types_use_residual_df(panel_df, cov_type):
    res = fit_re(panel_df, cov_type=cov_type)
    assert res.df_resid == RE_DF_RESID
    _check_t(res, RE_DF_RESID)


@pytest.mark.parametrize(
    ("cluster", "n_clusters"),
    [
        ("entity", N_ENTITIES),
        ("grp", N_GROUPS),
        ("coarse", N_COARSE),
        ("time", N_PERIODS),
    ],
)
class TestNonEntityCluster:
    """`cluster`に指定した列のクラスター数`G`で`G-1`になる（entity数ではない）。"""

    def test_fe_one_way(self, panel_df, cluster, n_clusters):
        res = fit_fe(panel_df, cov_type="cluster", cluster=cluster)
        _check_t(res, n_clusters - 1)

    def test_fe_two_way(self, panel_df, cluster, n_clusters):
        res = fit_fe(
            panel_df, cov_type="cluster", cluster=cluster, two_way=True
        )
        _check_t(res, n_clusters - 1)

    def test_re(self, panel_df, cluster, n_clusters):
        res = fit_re(panel_df, cov_type="cluster", cluster=cluster)
        _check_t(res, n_clusters - 1)


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
def test_fe_dk_uses_t_minus_one_of_the_dk_time_column(
    panel_df, options, n_periods
):
    """dkの`T`は`dk_time`列のユニーク数。2-wayでも固定効果の2つ目の列ではなく
    `dk_time`が使われる。"""
    res = fit_fe(panel_df, cov_type="dk", **options)
    _check_t(res, n_periods - 1)


@pytest.mark.parametrize(
    ("time", "n_periods"),
    [("time", N_PERIODS), ("time5", N_PERIODS_COARSE)],
)
def test_re_dk_uses_t_minus_one_of_the_time_column(panel_df, time, n_periods):
    res = fit_re(panel_df, cov_type="dk", dk_time=time)
    _check_t(res, n_periods - 1)


@pytest.mark.parametrize("confidence_level", [0.90, 0.99])
@pytest.mark.parametrize(
    ("kind", "options", "expected_df"),
    [
        ("fe", {"cov_type": "classical"}, FE_DF_RESID_ONE_WAY),
        ("fe", {"cov_type": "hc3"}, FE_DF_RESID_ONE_WAY),
        (
            "fe",
            {"cov_type": "classical", "two_way": True},
            FE_DF_RESID_TWO_WAY,
        ),
        ("fe", {"cov_type": "hc2", "two_way": True}, FE_DF_RESID_TWO_WAY),
        ("fe", {"cov_type": "cluster"}, N_ENTITIES - 1),
        ("fe", {"cov_type": "cluster", "cluster": "grp"}, N_GROUPS - 1),
        ("fe", {"cov_type": "cluster", "cluster": "coarse"}, N_COARSE - 1),
        ("fe", {"cov_type": "cluster", "cluster": "time"}, N_PERIODS - 1),
        (
            "fe",
            {"cov_type": "cluster", "cluster": "grp", "two_way": True},
            N_GROUPS - 1,
        ),
        (
            "fe",
            {"cov_type": "dk", "dk_time": "time5"},
            N_PERIODS_COARSE - 1,
        ),
        (
            "fe",
            {"cov_type": "dk", "dk_time": "time", "two_way": True},
            N_PERIODS - 1,
        ),
        ("re", {"cov_type": "classical"}, RE_DF_RESID),
        ("re", {"cov_type": "hc1"}, RE_DF_RESID),
        ("re", {"cov_type": "cluster"}, N_ENTITIES - 1),
        ("re", {"cov_type": "cluster", "cluster": "grp"}, N_GROUPS - 1),
        ("re", {"cov_type": "cluster", "cluster": "coarse"}, N_COARSE - 1),
        ("re", {"cov_type": "cluster", "cluster": "time"}, N_PERIODS - 1),
        ("re", {"cov_type": "dk", "dk_time": "time5"}, N_PERIODS_COARSE - 1),
    ],
)
def test_confidence_interval_uses_stat_df_critical_value(
    panel_df, kind, options, expected_df, confidence_level
):
    """信頼区間の臨界値が`stat_df`のt分布の両側`confidence_level`点であること
    （既定の0.95以外でも`df_resid`/`G-1`/`T-1`が使われる）。`stat_df`自体の値も
    期待値と照合する。"""
    make = fit_fe if kind == "fe" else fit_re
    res = make(panel_df, confidence_level=confidence_level, **options)
    assert res.stat_df == expected_df
    crit = sps.t.ppf(0.5 + confidence_level / 2.0, res.stat_df)
    for row in res.params_table():
        half = crit * row["std_error"]
        assert row["conf_lower"] == pytest.approx(
            row["param"] - half, rel=1e-8
        )
        assert row["conf_upper"] == pytest.approx(
            row["param"] + half, rel=1e-8
        )


# ── 不均衡パネル・不均衡クラスター ────────────────────────────────────


#
# 完全バランスのパネルでは`n_entities`・`T`・`n/G`が揃うため、取り違えても
# 気づきにくい。ここでは時点を欠かせ（entityごとに観測数が異なり、時点0は
# 全entityで欠測）、クラスターもサイズを偏らせる。期待値は実装を呼ばず
# データから直接数える。2-way FEは均衡パネルが前提（不均衡は`ValidationError`）
# なので、不均衡側は1-way FEとREのみ。


def _unbalanced_counts(data: pl.DataFrame) -> dict[str, int]:
    return {
        "n": data.height,
        "entities": data["entity"].n_unique(),
        "periods": data["time"].n_unique(),
    }


def test_unbalanced_fixture_is_actually_unbalanced(panel_df_unbalanced):
    c = _unbalanced_counts(panel_df_unbalanced)
    sizes = panel_df_unbalanced.group_by("entity").len()["len"]
    assert sizes.n_unique() > 1
    assert c["periods"] == 8  # 時点0は全entityで欠測
    assert c["n"] < N
    assert c["periods"] not in (N_PERIODS, c["entities"])


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_unbalanced_non_cluster_cov_types_use_residual_df(
    panel_df_unbalanced, cov_type
):
    c = _unbalanced_counts(panel_df_unbalanced)
    fe = fit_fe(panel_df_unbalanced, cov_type=cov_type)
    _check_t(fe, c["n"] - c["entities"] - 2)
    re = fit_re(panel_df_unbalanced, cov_type=cov_type)
    _check_t(re, c["n"] - 3)


@pytest.mark.parametrize("cluster", ["entity", "grp_u", "coarse_u"])
def test_unbalanced_cluster_uses_g_minus_one_df(panel_df_unbalanced, cluster):
    g = panel_df_unbalanced[cluster].n_unique()
    _check_t(
        fit_fe(panel_df_unbalanced, cov_type="cluster", cluster=cluster), g - 1
    )
    _check_t(
        fit_re(panel_df_unbalanced, cov_type="cluster", cluster=cluster), g - 1
    )


def test_unbalanced_dk_uses_observed_number_of_periods(panel_df_unbalanced):
    """欠測時点があるパネルでは、dkの`T`は期間の幅ではなく実際に観測された
    ユニークな時点数。"""
    t = panel_df_unbalanced["time"].n_unique()
    _check_t(fit_fe(panel_df_unbalanced, cov_type="dk", dk_time="time"), t - 1)
    _check_t(fit_re(panel_df_unbalanced, cov_type="dk", dk_time="time"), t - 1)
