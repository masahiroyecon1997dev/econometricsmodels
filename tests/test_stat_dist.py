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
from scipy import stats as sps

N_ENTITIES = 20
N_PERIODS = 10
N = N_ENTITIES * N_PERIODS
N_GROUPS = 7
N_COARSE = N_ENTITIES // 4


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
            # entityとは別軸のクラスター列。`grp`はentityと入れ子にならない
            # （G=7、entity内で値が変わる）、`coarse`はentityを束ねる入れ子
            # （G=5）。どちらもGがentity数・時点数・`df_resid`のいずれとも
            # 異なる値になるようにして、自由度の取り違えを区別できる。
            "grp": np.arange(N) % N_GROUPS,
            "coarse": entity // 4,
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


# --- FE/REの`stat_df`を`cov_type`ごとに網羅する --------------------------------
#
# `stat_df`は`cluster`のとき`G-1`、`dk`のとき`T-1`（時点数）、それ以外
# （`classical`/`hc1`〜`hc3`）は`df_resid`。`G`は`cluster`列のクラスター数で、
# 既定（`cluster`省略）ではentity数だが、他の列を指定すればそのクラスター数になる。

FE_DF_RESID_ONE_WAY = N - N_ENTITIES - 2
# 2-way: df_model = k + n_entities + n_periods - 1
FE_DF_RESID_TWO_WAY = N - (2 + N_ENTITIES + N_PERIODS - 1)
RE_DF_RESID = N - 3


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
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_fe_non_cluster_cov_types_use_residual_df(df, cov_type, two_way):
    res = _fe(df, cov_type=cov_type, two_way=two_way)
    expected = FE_DF_RESID_TWO_WAY if two_way else FE_DF_RESID_ONE_WAY
    assert res.df_resid == expected
    _check_t(res, expected)


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_re_non_cluster_cov_types_use_residual_df(df, cov_type):
    res = _re(df, cov_type=cov_type)
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

    def test_fe_one_way(self, df, cluster, n_clusters):
        res = _fe(df, cov_type="cluster", cluster=cluster)
        _check_t(res, n_clusters - 1)

    def test_fe_two_way(self, df, cluster, n_clusters):
        res = _fe(df, cov_type="cluster", cluster=cluster, two_way=True)
        _check_t(res, n_clusters - 1)

    def test_re(self, df, cluster, n_clusters):
        res = _re(df, cov_type="cluster", cluster=cluster)
        _check_t(res, n_clusters - 1)


def test_fe_two_way_dk_uses_t_minus_one_df(df):
    res = _fe(df, cov_type="dk", two_way=True)
    _check_t(res, N_PERIODS - 1)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda d: _fe(d, cov_type="cluster", cluster="grp"), id="fe"
        ),
        pytest.param(
            lambda d: _re(d, cov_type="cluster", cluster="grp"), id="re"
        ),
    ],
)
def test_non_entity_cluster_p_values_follow_stat_df(df, make):
    """p値が`stat_df = G-1`のt分布から再計算できること（`df_resid`や
    entity数ベースの自由度では一致しない）。"""
    res = make(df)
    assert res.stat_df == N_GROUPS - 1
    for name, t in res.test_stats.items():
        p = 2.0 * sps.t.sf(abs(t), res.stat_df)
        assert p == pytest.approx(res.p_values[name], rel=1e-8)


@pytest.mark.parametrize("confidence_level", [0.90, 0.99])
@pytest.mark.parametrize(
    ("kind", "options"),
    [
        ("fe", {"cov_type": "cluster"}),
        ("fe", {"cov_type": "cluster", "cluster": "grp"}),
        ("fe", {"cov_type": "dk", "dk_time": "time"}),
        ("re", {"cov_type": "cluster"}),
        ("re", {"cov_type": "cluster", "cluster": "grp"}),
        ("re", {"cov_type": "dk", "time": "time"}),
    ],
)
def test_confidence_interval_uses_stat_df_critical_value(
    df, kind, options, confidence_level
):
    """信頼区間の臨界値が`stat_df`のt分布の両側`confidence_level`点であること
    （既定の0.95以外でも`G-1`/`T-1`が使われる）。"""
    make = _fe if kind == "fe" else _re
    res = make(df, confidence_level=confidence_level, **options)
    crit = sps.t.ppf(0.5 + confidence_level / 2.0, res.stat_df)
    for row in res.coef_table():
        half = crit * row["std_err"]
        assert row["conf_lower"] == pytest.approx(row["coef"] - half, rel=1e-8)
        assert row["conf_upper"] == pytest.approx(row["coef"] + half, rel=1e-8)
