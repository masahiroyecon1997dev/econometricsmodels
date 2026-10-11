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
import polars as pl
import pytest
from _error_messages import escaped
from econometricsmodels import RE, REOptions, ValidationError

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
