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
N_PERIODS_COARSE = N_PERIODS // 2


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
            # 時点を2つずつ束ねた列（T=5）。`time`（T=10）と別のTを持つので、
            # dkの時点数`T-1`と`cluster="time"`の`G-1`（どちらも`time`だと9）を
            # 区別できる。
            "time5": time // 2,
        }
    )


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


@pytest.mark.parametrize(
    ("options", "n_periods"),
    [
        pytest.param({"dk_time": "time"}, N_PERIODS, id="1way-dk_time"),
        pytest.param({"time": "time"}, N_PERIODS, id="2way-time"),
        pytest.param(
            {"dk_time": "time5"}, N_PERIODS_COARSE, id="1way-dk_time-coarse"
        ),
        pytest.param(
            {"time": "time", "dk_time": "time5"},
            N_PERIODS_COARSE,
            id="2way-dk_time-overrides-time",
        ),
    ],
)
def test_fe_dk_uses_t_minus_one_of_the_dk_time_column(df, options, n_periods):
    """dkの`T`は`dk_time`（無ければ`time`）列のユニーク数。2-wayで`dk_time`を
    併せて指定した場合は`dk_time`が優先される。"""
    res = _fe(df, cov_type="dk", **options)
    _check_t(res, n_periods - 1)


@pytest.mark.parametrize(
    ("time", "n_periods"),
    [("time", N_PERIODS), ("time5", N_PERIODS_COARSE)],
)
def test_re_dk_uses_t_minus_one_of_the_time_column(df, time, n_periods):
    res = _re(df, cov_type="dk", time=time)
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
        ("fe", {"cov_type": "dk", "two_way": True}, N_PERIODS - 1),
        ("re", {"cov_type": "classical"}, RE_DF_RESID),
        ("re", {"cov_type": "hc1"}, RE_DF_RESID),
        ("re", {"cov_type": "cluster"}, N_ENTITIES - 1),
        ("re", {"cov_type": "cluster", "cluster": "grp"}, N_GROUPS - 1),
        ("re", {"cov_type": "cluster", "cluster": "coarse"}, N_COARSE - 1),
        ("re", {"cov_type": "cluster", "cluster": "time"}, N_PERIODS - 1),
        ("re", {"cov_type": "dk", "time": "time5"}, N_PERIODS_COARSE - 1),
    ],
)
def test_confidence_interval_uses_stat_df_critical_value(
    df, kind, options, expected_df, confidence_level
):
    """信頼区間の臨界値が`stat_df`のt分布の両側`confidence_level`点であること
    （既定の0.95以外でも`df_resid`/`G-1`/`T-1`が使われる）。`stat_df`自体の値も
    期待値と照合する。"""
    make = _fe if kind == "fe" else _re
    res = make(df, confidence_level=confidence_level, **options)
    assert res.stat_df == expected_df
    crit = sps.t.ppf(0.5 + confidence_level / 2.0, res.stat_df)
    for row in res.coef_table():
        half = crit * row["std_err"]
        assert row["conf_lower"] == pytest.approx(row["coef"] - half, rel=1e-8)
        assert row["conf_upper"] == pytest.approx(row["coef"] + half, rel=1e-8)


# --- 不均衡パネル・不均衡クラスター --------------------------------------------
#
# 完全バランスのパネルでは`n_entities`・`T`・`n/G`が揃うため、取り違えても
# 気づきにくい。ここでは時点を欠かせ（entityごとに観測数が異なり、時点0は
# 全entityで欠測）、クラスターもサイズを偏らせる。期待値は実装を呼ばず
# データから直接数える。2-way FEは均衡パネルが前提（不均衡は`ValidationError`）
# なので、不均衡側は1-way FEとREのみ。


@pytest.fixture(scope="module")
def df_unbalanced(df) -> pl.DataFrame:
    rng = np.random.default_rng(424)
    keep = (pl.col("time") >= 1) & (
        pl.col("time") < 4 + (pl.col("entity") % 6)
    )
    sub = df.filter(keep)
    n = sub.height
    # 非入れ子で偏ったクラスター（6群、サイズは概ね3%〜50%）
    grp = rng.choice(6, size=n, p=[0.03, 0.05, 0.07, 0.10, 0.25, 0.50])
    assert len(set(grp.tolist())) == 6
    entity = sub["entity"].to_numpy()
    # entityを束ねる入れ子で偏ったクラスター（サイズ10/4/3/2/1 entity）
    coarse = np.select(
        [entity < 10, entity < 14, entity < 17, entity < 19], [0, 1, 2, 3], 4
    )
    return sub.with_columns(
        pl.Series("grp_u", grp), pl.Series("coarse_u", coarse)
    )


def _unbalanced_counts(data: pl.DataFrame) -> dict[str, int]:
    return {
        "n": data.height,
        "entities": data["entity"].n_unique(),
        "periods": data["time"].n_unique(),
    }


def test_unbalanced_fixture_is_actually_unbalanced(df_unbalanced):
    c = _unbalanced_counts(df_unbalanced)
    sizes = df_unbalanced.group_by("entity").len()["len"]
    assert sizes.n_unique() > 1
    assert c["periods"] == 8  # 時点0は全entityで欠測
    assert c["n"] < N
    assert c["periods"] not in (N_PERIODS, c["entities"])


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_unbalanced_non_cluster_cov_types_use_residual_df(
    df_unbalanced, cov_type
):
    c = _unbalanced_counts(df_unbalanced)
    fe = _fe(df_unbalanced, cov_type=cov_type)
    _check_t(fe, c["n"] - c["entities"] - 2)
    re = _re(df_unbalanced, cov_type=cov_type)
    _check_t(re, c["n"] - 3)


@pytest.mark.parametrize("cluster", ["entity", "grp_u", "coarse_u"])
def test_unbalanced_cluster_uses_g_minus_one_df(df_unbalanced, cluster):
    g = df_unbalanced[cluster].n_unique()
    _check_t(_fe(df_unbalanced, cov_type="cluster", cluster=cluster), g - 1)
    _check_t(_re(df_unbalanced, cov_type="cluster", cluster=cluster), g - 1)


def test_unbalanced_dk_uses_observed_number_of_periods(df_unbalanced):
    """欠測時点があるパネルでは、dkの`T`は期間の幅ではなく実際に観測された
    ユニークな時点数。"""
    t = df_unbalanced["time"].n_unique()
    _check_t(_fe(df_unbalanced, cov_type="dk", dk_time="time"), t - 1)
    _check_t(_re(df_unbalanced, cov_type="dk", time="time"), t - 1)
