"""2-way RE（`REOptions.time`、バランスパネル）の構造・API・バリデーションのテスト。

参照実装（plm）・仕様どおりの手計算との網羅的な数値照合（全`cov_type`・複数シナリオ・
実データ）は`test_re_two_way_reference.py`が担う。ここでは小さなバランスパネル
（`re_two_way_small.csv`）で、`n_periods`・自由度・Hausman検定（手計算との照合、クラスター数・
時点数の境界、`cluster`・`dk_time`の列指定）・分散成分と重みの構造・オプションの
バリデーションを確認する。

データ（6エンティティ×4時点）: `xi`は個体間でのみ、`xt`は時点間でのみ変わる（どちらも
RE（2-way）で許容される）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from _constants import DATA_DIR
from _error_messages import escaped
from _re_oracle import hand_computed
from econometricsmodels import (
    RE,
    ComputationError,
    REOptions,
    ValidationError,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    """6エンティティ×4時点（`benchmark/panel/datasets.py`の`re_two_way_small_panel`を
    固定したCSV）に、テスト用の列を足したもの。"""
    base = pl.read_csv(DATA_DIR / "re_two_way_small.csv")
    entity_idx = pl.col("entity").str.slice(1).cast(pl.Int64)  # 1〜6
    return base.with_columns(
        # 個体平均も時点平均も0の交互作用列。
        z=(entity_idx - 3.5) * (pl.col("time") - 2.5),
        # entityと別のクラスター列（3グループ）。
        grp=pl.format("g{}", (entity_idx - 1) // 2),
        # timeと別の並び順の時点列（値の昇順は t = 2, 4, 1, 3）。
        dk_perm=pl.col("time").replace_strict(
            {1: 30, 2: 10, 3: 40, 4: 20}, return_dtype=pl.Int64
        ),
        # 3時点にまとめた時点列。
        dk3=pl.min_horizontal(pl.col("time"), 3),
    )


def _fit(df, options, x=("x1", "xi", "xt")):
    return RE(df, y="y", x=list(x), entity="entity", options=options).fit()


def test_two_way_reports_degrees_of_freedom(panel):
    """`df_resid = n - k`、`df_model = k + 1`（切片込み）。classicalのF検定の
    分母自由度は`df_resid`、clusterは`G - 1`（plmとの数値照合は
    `test_re_two_way_reference.py`）。"""
    classical = _fit(panel, REOptions(cov_type="classical", time="time"))
    cluster = _fit(panel, REOptions(cov_type="cluster", time="time"))
    assert classical.df_resid == 20
    assert classical.df_model == 4
    assert classical.f_df_denom == 20
    assert cluster.f_df_denom == 5  # G - 1


def test_two_way_reports_n_periods(panel):
    res = _fit(panel, REOptions(cov_type="classical", time="time"))

    assert res.n_periods == 4
    assert res.n_entities == 6


HAUSMAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]


def _oracle(panel, x, cov_type, **kwargs):
    """仕様どおりの手計算（`_re_oracle.py`）。2-wayのDKは`dk_bandwidth=1`（T=4の既定）。"""
    return hand_computed(
        panel, list(x), cov_type, two_way=True, dk_bandwidth=1, **kwargs
    )


def _options(cov_type, **kwargs):
    extra = {"dk_time": "time", "dk_bandwidth": 1} if cov_type == "dk" else {}
    return REOptions(cov_type=cov_type, time="time", **extra, **kwargs)


@pytest.mark.parametrize("cov_type", HAUSMAN_COV_TYPES)
@pytest.mark.parametrize(
    ("x", "df"),
    [
        # `xi`・`xt`は2-way within変換後に全ゼロで、検定対象は`x1`だけ。
        (("x1",), 1),
        (("x1", "xi", "xt"), 1),
        (("x1", "x2"), 2),
        # 交互作用列`z`は識別できず（`x*`との差が定数）検定対象から外れる。
        (("x1", "z", "x2"), 2),
        (("x1", "z"), 1),
    ],
)
def test_two_way_hausman_test_matches_hand_computed(panel, x, df, cov_type):
    """2-wayのHausman検定（補助回帰版）の統計量・p値（`χ²_df`の上側確率）・自由度が
    手計算と一致する。`hausman_df`は2-way within変換後に分散のある列のうち識別できる列の数
    （plmとの照合は`test_re_two_way_reference.py`）。"""
    res = _fit(panel, _options(cov_type), x=x)
    hand = _oracle(panel, x, cov_type)

    assert res.hausman_df == hand["hausman_df"] == df
    assert res.hausman_statistic == pytest.approx(
        hand["hausman_statistic"], rel=1e-8
    )
    assert res.hausman_p_value == pytest.approx(
        hand["hausman_p_value"], rel=1e-7
    )
    assert 0.0 <= res.hausman_p_value <= 1.0


def test_two_way_results_do_not_depend_on_the_column_order(panel):
    """時間変化のある列が先頭以外にあっても結果は同じ（列の対応づけの回帰ガード）。"""
    base = _fit(
        panel,
        REOptions(cov_type="classical", time="time"),
        x=("x1", "xi", "z", "x2"),
    )
    for x in [
        ("xi", "x1", "x2", "z"),
        ("z", "x2", "xi", "x1"),
        ("x2", "z", "x1", "xi"),
    ]:
        res = _fit(panel, REOptions(cov_type="classical", time="time"), x=x)
        assert res.hausman_df == base.hausman_df
        assert res.hausman_statistic == pytest.approx(
            base.hausman_statistic, rel=1e-9
        )
        assert res.r_squared_within == pytest.approx(
            base.r_squared_within, rel=1e-9
        )
        for name in x:
            assert res.params[name] == pytest.approx(
                base.params[name], rel=1e-9
            )


def test_two_way_hausman_cluster_count_must_cover_the_auxiliary_slopes(panel):
    """時点でクラスタリング（G = 4）。RE本体の傾き3個には足りるが、補助回帰の傾き
    （`x*`の3個と`x̃`の1個）には足りず`ValidationError`になる。`x1 + xi`なら通る。"""
    options = REOptions(cov_type="cluster", time="time", cluster="time")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=4, q=4),
    ):
        _fit(panel, options, x=("x1", "xi", "xt"))
    res = _fit(panel, options, x=("x1", "xi"))
    hand = _oracle(
        panel, ("x1", "xi"), "cluster", cluster=panel["time"].to_numpy() - 1
    )
    assert res.hausman_df == 1
    assert res.hausman_statistic == pytest.approx(
        hand["hausman_statistic"], rel=1e-8
    )


def test_two_way_hausman_dk_periods_must_exceed_the_slopes(panel):
    """`dk_time`を3時点にまとめる。RE本体の傾き3個では`T = 3 <= q`で
    `ValidationError`、`x1 + xi`（q = 2）なら通って手計算と一致する。"""
    options = REOptions(
        cov_type="dk", time="time", dk_time="dk3", dk_bandwidth=1
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_DK_PERIODS_FOR_INFERENCE, t_periods=3, q=3
        ),
    ):
        _fit(panel, options, x=("x1", "xi", "xt"))
    res = _fit(panel, options, x=("x1", "xi"))
    hand = _oracle(panel, ("x1", "xi"), "dk", dk_time_col="dk3")
    assert res.hausman_statistic == pytest.approx(
        hand["hausman_statistic"], rel=1e-8
    )


def test_two_way_hausman_uses_the_cluster_and_dk_time_columns(panel):
    """`cluster`をentityと別の列に、`dk_time`をtimeと別の並び順の列にすると、
    補助回帰の共分散もその列を使う（手計算の補助回帰と一致）。別の列で変わることも
    確かめる（既定のentityクラスター・`time`のDKとは異なる値になる）。"""
    groups = np.unique(panel["grp"].to_numpy(), return_inverse=True)[1]
    clustered = _fit(
        panel,
        REOptions(cov_type="cluster", time="time", cluster="grp"),
        x=("x1",),
    )
    hand = _oracle(panel, ("x1",), "cluster", cluster=groups)
    assert clustered.hausman_statistic == pytest.approx(
        hand["hausman_statistic"], rel=1e-8
    )
    default = _fit(panel, _options("cluster"), x=("x1",))
    assert default.hausman_statistic != pytest.approx(
        clustered.hausman_statistic, rel=1e-3
    )

    for x in [("x1",), ("x1", "xi", "xt")]:
        dk = _fit(
            panel,
            REOptions(
                cov_type="dk", time="time", dk_time="dk_perm", dk_bandwidth=1
            ),
            x=x,
        )
        hand = _oracle(panel, x, "dk", dk_time_col="dk_perm")
        assert dk.hausman_statistic == pytest.approx(
            hand["hausman_statistic"], rel=1e-8
        )
        by_time = _fit(panel, _options("dk"), x=x)
        assert by_time.hausman_statistic != pytest.approx(
            dk.hausman_statistic, rel=1e-3
        )


def test_two_way_hausman_test_is_none_without_two_way_variation(panel):
    """一次元でしか変わらない列だけなら検定できず`None`（`fit()`は成功する）。"""
    res = _fit(panel, REOptions(time="time"), x=("xi", "xt"))
    assert res.hausman_statistic is None
    assert res.hausman_p_value is None
    assert res.hausman_df is None
    assert res.r_squared_within == 0.0


def test_two_way_exposes_variance_components_and_weights_but_no_entity_theta(
    panel,
):
    """2-wayは`sigma2_lambda`と3つの重みを持ち、1-wayの`theta`（個体ごと）は`None`
    （値のplm::ercompとの照合は`test_re_two_way_reference.py`）。重みは分散成分から
    `θ₁ = 1 - σ_ε/σ₁`・`θ₂ = 1 - σ_ε/σ₂`・`θ₃ = θ₁ + θ₂ - 1 + σ_ε/σ₄`と定まる。"""
    res = _fit(panel, REOptions(time="time"), x=("x1", "xi", "xt"))
    n, t = 6, 4
    eps, u, lam = res.sigma2_eps, res.sigma2_u, res.sigma2_lambda
    sigma = [
        math.sqrt(eps + t * u),
        math.sqrt(eps + n * lam),
        math.sqrt(eps + t * u + n * lam),
    ]
    theta1 = 1 - math.sqrt(eps) / sigma[0]
    theta2 = 1 - math.sqrt(eps) / sigma[1]
    theta3 = theta1 + theta2 - 1 + math.sqrt(eps) / sigma[2]
    assert res.theta is None
    assert res.theta_entity == pytest.approx(theta1, rel=1e-12)
    assert res.theta_time == pytest.approx(theta2, rel=1e-12)
    assert res.theta_total == pytest.approx(theta3, rel=1e-12)


def test_one_way_exposes_theta_per_entity_and_no_two_way_fields(panel):
    """1-wayは`σ_λ²`と2-wayの重みが`None`で、`theta`はエンティティID→θ_i
    （バランスパネルなので全エンティティで同じ値、`1 - sqrt(σ_ε² / (T·σ_u² + σ_ε²))`）。
    値のplmとの照合は`test_re_time_invariant.py`。"""
    res = _fit(panel, REOptions(cov_type="classical"), x=("x1", "xt"))
    assert res.sigma2_lambda is None
    assert res.theta_entity is None
    assert res.theta_time is None
    assert res.theta_total is None
    assert list(res.theta) == [f"e{i}" for i in range(1, 7)]
    want = 1 - math.sqrt(res.sigma2_eps / (4 * res.sigma2_u + res.sigma2_eps))
    assert list(res.theta.values()) == pytest.approx([want] * 6, rel=1e-12)


def test_theta_total_is_zero_when_a_variance_component_is_zero(panel):
    """分散成分が0に切り上げられると`θ₃`は丸め誤差なしで0になる
    （グループ効果が無いDGPで`σ_u²`・`σ_λ²`がともに0）。"""
    df = panel.with_columns(
        y=2.0 * pl.col("x1")
        + pl.Series([0.1 * ((r * 7) % 5) for r in range(24)])
    )
    res = _fit(df, REOptions(time="time"), x=("x1",))
    assert res.sigma2_u == 0.0
    assert res.sigma2_lambda == 0.0
    assert (res.theta_entity, res.theta_time, res.theta_total) == (
        0.0,
        0.0,
        0.0,
    )


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

    for name, value in _ONE_WAY_REF["params"].items():
        assert res.params[name] == pytest.approx(value, rel=1e-8)
    for name, value in _ONE_WAY_REF["std_errors"].items():
        assert res.std_errors[name] == pytest.approx(value, rel=1e-8)
    assert res.df_resid == 21
    assert res.hausman_df == 1


# plm（`ercomp`）の準偏差変換の重み。`y ~ x1 + xt`（1-way、全エンティティで共通の
# `1 - sqrt(σ_ε² / (T·σ_u² + σ_ε²))`）と`y ~ x1 + xi + xt`（2-way: id, time, total）。
_ONE_WAY_REF = json.loads((_FIXTURES / "re_time_invariant.json").read_text())[
    "synthetic"
]["small_hand_time_only"]["classical"]
_TWO_WAY_REF = json.loads((_FIXTURES / "re_two_way.json").read_text())[
    "synthetic"
]["small_hand_all"]["classical"]
THETA_ONE_WAY_X1_XT = 1 - math.sqrt(
    _ONE_WAY_REF["sigma2_eps"]
    / (4 * _ONE_WAY_REF["sigma2_u"] + _ONE_WAY_REF["sigma2_eps"])
)
THETA_TWO_WAY = (
    _TWO_WAY_REF["theta_entity"],
    _TWO_WAY_REF["theta_time"],
    _TWO_WAY_REF["theta_total"],
)


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
    # Hausmanの補助回帰がクラスター数（4）を要するため、`x1`だけで比べる。
    dk = _fit(
        panel,
        REOptions(cov_type="dk", time="time", dk_time="time", dk_bandwidth=0),
        x=("x1",),
    )
    clustered = _fit(
        panel,
        REOptions(cov_type="cluster", time="time", cluster="time"),
        x=("x1",),
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


@pytest.mark.parametrize(
    ("keep", "means"),
    [
        # 2時点: 時点平均のbetween回帰（`[1, x̄.ₜ]`）が`T = 2 <= K_b + 1`で失敗する。
        (pl.col("time") <= 2, "time"),
        # 2エンティティ: 個体平均のbetween回帰が`N = 2 <= K_b + 1`で失敗する。
        (pl.col("entity").is_in(["e1", "e2"]), "entity"),
    ],
)
def test_between_regression_failure_names_the_regression(panel, keep, means):
    """2-wayは個体平均・時点平均の2本のbetween回帰を推定するため、失敗したメッセージが
    どちらかを示す。"""
    with pytest.raises(
        ValidationError,
        match=f"between-regression on {means} means",
    ):
        _fit(panel.filter(keep), REOptions(time="time"), x=("x1",))


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
