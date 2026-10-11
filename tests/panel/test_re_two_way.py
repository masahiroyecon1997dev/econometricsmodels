"""2-way RE（`REOptions.time`、バランスパネル）の構造・API・バリデーションのテスト。

参照実装との網羅的な数値照合（全`cov_type`・不均衡でない複数シナリオ・実データ）は別のテスト
（plmとの照合）が担う。ここでは、plm（`model = "random"`・`effect = "twoways"`・
Swamy-Arora）の値を小さなバランスパネルに埋め込んだスモークテストと、`n_periods`・
暫定値・オプションのバリデーションを確認する。2-wayのハウスマン検定とパネルR²は未実装
（`hausman_*`は`None`・`r_squared_*`は`NaN`）。

データ（6エンティティ×4時点）: `xi`は個体間でのみ、`xt`は時点間でのみ変わる（どちらも
RE（2-way）で許容される）。plmの参照値は`y ~ x1 + xi + xt`。
"""

from __future__ import annotations

import math

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from _error_messages import escaped
from econometricsmodels import (
    RE,
    ComputationError,
    REOptions,
    ValidationError,
)

_Y = [
    2.973, 0.102, -4.005, -1.69, 4.512, 2.697, 1.044, 0.361, 0.457, -1.063,
    -3.188, -3.244, 2.213, 0.875, -0.479, -2.389, 5.936, 5.314, 2.163, -0.585,
    2.081, 0.631, -1.679, -4.541,
]  # fmt: skip
_X1 = [
    -0.828, -0.348, -1.538, -0.256, -1.15, 0.012, -0.223, 0.888, -0.592,
    -0.656, -0.683, -0.016, -0.443, 0.353, 0.073, 0.007, -0.188, -0.766,
    -0.221, -0.984, -1.104, -0.938, 0.679, -1.577,
]  # fmt: skip
_XI = [-0.87, 0.485, -0.186, 1.546, -0.611, -0.348]
_XT = [-1.637, 0.02, 0.892, -0.873]

PLM_PARAMS = {
    "const": 0.307929272368423,
    "x1": 0.89132496100334,
    "xi": -0.195188876604849,
    "xt": -1.09263770513503,
}
PLM_SE = {
    "classical": [
        1.26709012892926,
        0.364650749295775,
        1.00460548400194,
        0.959418507617424,
    ],
    "cluster": [
        1.38606716539328,
        0.352916535514244,
        0.690856338748332,
        1.0762204598989,
    ],
}
PLM_F = 2.3413216172082


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity": [f"e{r // 4 + 1}" for r in range(24)],
            "time": [r % 4 + 1 for r in range(24)],
            "y": _Y,
            "x1": _X1,
            "xi": [_XI[r // 4] for r in range(24)],
            "xt": [_XT[r % 4] for r in range(24)],
        }
    )


def _fit(df, options, x=("x1", "xi", "xt")):
    return RE(df, y="y", x=list(x), entity="entity", options=options).fit()


@pytest.mark.parametrize("cov_type", ["classical", "cluster"])
def test_matches_plm_reference_values(panel, cov_type):
    """係数・標準誤差・F統計量がplmと一致する（一次元でしか変わらない`xi`・`xt`を含む）。"""
    res = _fit(panel, REOptions(cov_type=cov_type, time="time"))

    for name, value in PLM_PARAMS.items():
        assert math.isclose(res.params[name], value, rel_tol=1e-6), name
    for name, value in zip(res.param_names, PLM_SE[cov_type], strict=True):
        assert math.isclose(res.std_errors[name], value, rel_tol=1e-6), name
    assert res.df_resid == 20
    assert res.df_model == 4
    if cov_type == "classical":
        assert math.isclose(res.f_statistic, PLM_F, rel_tol=1e-6)
        assert res.f_df_denom == 20
    else:
        assert res.f_df_denom == 5  # G - 1


def test_two_way_reports_n_periods_and_pending_values(panel):
    res = _fit(panel, REOptions(cov_type="classical", time="time"))

    assert res.n_periods == 4
    assert res.n_entities == 6
    # 2-wayのハウスマン検定・パネルR²は未実装（1-wayの値を流用しない）。
    assert res.hausman_statistic is None
    assert res.hausman_p_value is None
    assert res.hausman_df is None
    assert math.isnan(res.r_squared_within)
    assert math.isnan(res.r_squared_between)
    assert math.isnan(res.r_squared_overall)


def test_one_way_has_no_n_periods(panel):
    """`time`を指定しない1-wayは`n_periods`が`None`で、`dk_time`だけの指定でも同じ。"""
    plain = _fit(panel, REOptions(cov_type="classical"), x=("x1", "xi"))
    dk = _fit(
        panel,
        REOptions(cov_type="dk", dk_time="time"),
        x=("x1", "xi"),
    )

    assert plain.n_periods is None
    assert dk.n_periods is None
    assert plain.hausman_statistic is not None


def test_one_way_accepts_time_only_regressor_like_plm(panel):
    """1-wayでも、時点間でのみ変わる列（`xt`、年トレンド等）を含められる。
    バランスパネルでは個体平均が一定になるためbetween回帰から落ち、plmと同じ
    推定値になる。Hausman検定は`x̃t`が識別できないため`x1`だけを検定する。"""
    res = _fit(panel, REOptions(cov_type="classical"), x=("x1", "xt"))

    expected = {
        "const": 0.117907102600895,
        "x1": 0.400160538463792,
        "xt": -1.029152589790345,
    }
    expected_se = [0.905185788075077, 0.851443915446071, 0.510706679841924]
    for name, value in expected.items():
        assert res.params[name] == pytest.approx(value, rel=1e-8)
    assert list(res.std_errors.values()) == pytest.approx(
        expected_se, rel=1e-8
    )
    assert res.df_resid == 21
    assert res.hausman_df == 1


# plm（`ercomp`）の準偏差変換の重み。`y ~ x1 + xt`（1-way）と`y ~ x1 + xi + xt`
# （2-way: id, time, total）。
THETA_ONE_WAY_X1_XT = 0.422786083747045
THETA_TWO_WAY = (0.764301210325029, 0.791153185965249, 0.713712012990144)


def _group_mean(df: pl.DataFrame, col: str, by: str) -> np.ndarray:
    return df.select(pl.col(col).mean().over(by)).to_series().to_numpy()


def _transformed_design(df, x, thetas, two_way):
    """plmの重みで準偏差変換した`y`と設計行列（切片は`1 - θ…`倍）。"""

    def transform(col):
        v = df[col].to_numpy()
        if not two_way:
            return v - thetas * _group_mean(df, col, "entity")
        t1, t2, t3 = thetas
        return (
            v
            - t1 * _group_mean(df, col, "entity")
            - t2 * _group_mean(df, col, "time")
            + t3 * df[col].mean()
        )

    y = transform("y")
    const = (
        np.full(df.height, 1 - thetas)
        if not two_way
        else np.full(df.height, 1 - thetas[0] - thetas[1] + thetas[2])
    )
    design = np.column_stack([const] + [transform(c) for c in x])
    return y, design


@pytest.mark.parametrize(
    ("cov_type", "sm_kwargs"),
    [
        ("classical", {}),
        ("hc1", {"cov_type": "HC1"}),
        ("hc2", {"cov_type": "HC2"}),
        ("hc3", {"cov_type": "HC3"}),
    ],
)
@pytest.mark.parametrize(
    ("two_way", "x", "thetas"),
    [
        (False, ("x1", "xt"), THETA_ONE_WAY_X1_XT),
        (True, ("x1", "xi", "xt"), THETA_TWO_WAY),
    ],
)
def test_standard_errors_match_ols_on_the_plm_transformed_data(
    panel, two_way, x, thetas, cov_type, sm_kwargs
):
    """plmの重みで変換した`y`・`x`を切片込みの回帰（statsmodels）に当てはめた
    係数・標準誤差・t値・p値が一致する。1-wayの時点間のみ変わる列（`xt`）と
    2-wayの一次元でしか変わらない列を含む。HC2・HC3は素のレバレッジ。"""
    options = REOptions(cov_type=cov_type, time="time" if two_way else None)
    res = _fit(panel, options, x=x)

    y, design = _transformed_design(panel, x, thetas, two_way)
    ref = sm.OLS(y, design).fit(use_t=True, **sm_kwargs)
    names = res.param_names
    assert [res.params[n] for n in names] == pytest.approx(
        list(ref.params), rel=1e-6
    )
    assert [res.std_errors[n] for n in names] == pytest.approx(
        list(ref.bse), rel=1e-6
    )
    assert [res.test_stats[n] for n in names] == pytest.approx(
        list(ref.tvalues), rel=1e-6
    )
    assert [res.p_values[n] for n in names] == pytest.approx(
        list(ref.pvalues), rel=1e-5
    )


@pytest.mark.parametrize(
    ("two_way", "x", "thetas"),
    [
        (False, ("x1", "xt"), THETA_ONE_WAY_X1_XT),
        (True, ("x1", "xi", "xt"), THETA_TWO_WAY),
    ],
)
def test_cluster_standard_errors_match_ols_on_the_plm_transformed_data(
    panel, two_way, x, thetas
):
    """クラスター標準誤差（既定のentityクラスター）も変換後の回帰と一致する。"""
    res = _fit(
        panel,
        REOptions(cov_type="cluster", time="time" if two_way else None),
        x=x,
    )
    y, design = _transformed_design(panel, x, thetas, two_way)
    groups = panel["entity"].to_numpy()
    ref = sm.OLS(y, design).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
    )
    assert [res.std_errors[n] for n in res.param_names] == pytest.approx(
        list(ref.bse), rel=1e-6
    )


def test_two_way_driscoll_kraay_without_lags_equals_time_clusters(panel):
    """`dk_bandwidth=0`のDriscoll-Kraayは、`dk_time`でクラスタリングした
    cluster（`cluster="time"`）と同じ標準誤差になる。"""
    dk = _fit(
        panel,
        REOptions(cov_type="dk", time="time", dk_time="time", dk_bandwidth=0),
    )
    clustered = _fit(
        panel, REOptions(cov_type="cluster", time="time", cluster="time")
    )
    assert list(dk.std_errors.values()) == pytest.approx(
        list(clustered.std_errors.values()), rel=1e-10
    )


@pytest.mark.parametrize("cov_type", ["hc1", "hc2", "hc3", "cluster", "dk"])
def test_two_way_runs_with_every_cov_type(panel, cov_type):
    kwargs = {"dk_time": "time"} if cov_type == "dk" else {}
    res = _fit(panel, REOptions(cov_type=cov_type, time="time", **kwargs))
    assert all(math.isfinite(v) and v > 0 for v in res.std_errors.values())


@pytest.mark.parametrize("x", [("xi", "xt"), ("xt",), ("xi",)])
def test_two_way_accepts_regressors_that_vary_in_only_one_dimension(panel, x):
    """時間変化のある列が無くても（内部の2-way FEが説明変数0個でも）推定できる。"""
    res = _fit(panel, REOptions(time="time"), x=x)
    assert res.n_periods == 4
    assert all(math.isfinite(v) for v in res.params.values())


def test_two_way_constant_column_raises_computation_error(panel):
    """切片と共線な定数列は、between回帰では落ちて最終回帰が特異になる。"""
    df = panel.with_columns(c=pl.lit(2.0))
    with pytest.raises(ComputationError):
        _fit(df, REOptions(time="time"), x=("x1", "c"))


def test_one_way_time_only_column_on_unbalanced_panel_is_tested(panel):
    """1行欠けた不均衡パネルでは、時点間でのみ変わる列も個体平均が一定でなく、
    ハウスマン検定の対象に残る（バランスなら`hausman_df == 1`）。"""
    unbalanced = panel.with_row_index().filter(pl.col("index") != 5)
    balanced = _fit(panel, REOptions(cov_type="classical"), x=("x1", "xt"))
    res = _fit(unbalanced, REOptions(cov_type="classical"), x=("x1", "xt"))
    assert balanced.hausman_df == 1
    assert res.hausman_df == 2


def test_dk_time_is_independent_of_time(panel):
    """2-wayでも`cov_type="dk"`は`dk_time`が必須で、`time`は借用されない。
    `dk_time`が`time`と同じ列でも指定できる。"""
    with pytest.raises(ValidationError, match=escaped(msgs.DK_REQUIRES_TIME)):
        _fit(panel, REOptions(cov_type="dk", time="time"), x=("x1", "xi"))

    res = _fit(
        panel,
        REOptions(cov_type="dk", time="time", dk_time="time"),
        x=("x1", "xi"),
    )
    assert res.n_periods == 4
    assert res.dk_bandwidth_used == 1


def test_unbalanced_panel_raises(panel):
    df = panel.slice(1, panel.height - 1)  # 先頭1行を欠く
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.UNBALANCED_PANEL_FOR_TWO_WAY,
            n_obs=23,
            n_entities=6,
            n_periods=4,
            expected=24,
        ),
    ):
        _fit(df, REOptions(time="time"))


def test_unbalanced_panel_is_allowed_for_one_way(panel):
    df = panel.slice(1, panel.height - 1)
    assert _fit(df, REOptions(cov_type="classical"), x=("x1",)).n_obs == 23


def test_singleton_time_raises(panel):
    """最後の時点を1観測だけ残すと、時点のsingletonで`ValidationError`。"""
    keep = (pl.col("time") != 4) | (pl.int_range(pl.len()).over("time") == 0)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.SINGLETON_GROUP, dimension="time", group_id="4"),
    ):
        _fit(panel.filter(keep), REOptions(time="time"), x=("x1",))


def test_missing_time_column_raises(panel):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        _fit(panel, REOptions(time="nonexistent"))


@pytest.mark.parametrize(
    ("time_col", "x"),
    [("y", ("x1",)), ("entity", ("x1",)), ("x1", ("x1", "xi"))],
)
def test_time_overlapping_another_role_raises(panel, time_col, x):
    with pytest.raises(ValidationError):
        _fit(panel, REOptions(time=time_col), x=x)


def test_time_with_null_values_raises(panel):
    df = panel.with_columns(
        pl.when(pl.int_range(pl.len()) == 3)
        .then(None)
        .otherwise(pl.col("time"))
        .alias("time")
    )
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.GROUP_KEY_COLUMN_HAS_MISSING_VALUES, name="time"),
    ):
        _fit(df, REOptions(time="time"))
