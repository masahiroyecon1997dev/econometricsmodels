"""REの分散成分（`sigma2_*`）と準偏差変換の重み（`theta*`）の公開値のテスト。

参照実装（linearmodels・plm）との数値照合は`test_re_reference.py`・
`test_re_time_invariant.py`・`test_re_two_way.py`が担う。ここでは、行の順序に
よらず重みが正しいエンティティに結び付くこと、IDのdtypeごとのキー、
`cov_type`などのオプションに値が依存しないこと、0への切り上げ・退化ケース、
使わない側が`None`になることを確認する。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _errors import raises_error
from econometricsmodels import RE, ErrorCode, REOptions


def _panel(
    seed: int,
    n_entities: int = 8,
    n_periods: int = 5,
    *,
    entity_sd: float = 0.0,
    time_sd: float = 3.0,
    noise_sd: float = 1.0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    e = np.repeat(np.arange(n_entities), n_periods)
    t = np.tile(np.arange(n_periods), n_entities)
    a = rng.normal(size=n_entities) * entity_sd
    g = rng.normal(size=n_periods) * time_sd
    x = rng.normal(size=n_entities * n_periods)
    y = 1 + x + a[e] + g[t] + noise_sd * rng.normal(size=len(x))
    return pl.DataFrame(
        {"entity": [f"e{i}" for i in e], "time": t, "y": y, "x": x}
    )


def _fit(df, **options):
    return RE(
        df, y="y", x=["x"], entity="entity", options=REOptions(**options)
    ).fit()


def test_theta_is_attached_to_the_right_entity_on_an_unsorted_unbalanced_panel():
    """行がエンティティ順でなく、観測数も違うパネルでも、`theta[id]`は
    そのエンティティの観測数`T_i`から`1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`で
    決まる値になる（出現順と辞書順が食い違う`b`・`a`・`c`）。"""
    rng = np.random.default_rng(5)
    labels = ["b", "b", "a", "a", "a", "a", "a", "c", "c", "c", "b", "a", "c"]
    counts = {k: labels.count(k) for k in "abc"}
    n = len(labels)
    effect = {"a": 5.0, "b": -5.0, "c": 1.0}
    x = rng.normal(size=n)
    y = 1 + x + np.array([effect[k] for k in labels]) + rng.normal(size=n)
    df = pl.DataFrame({"entity": labels, "y": y, "x": x})
    res = _fit(df, cov_type="classical")

    assert list(res.theta) == ["a", "b", "c"]
    for entity, theta in res.theta.items():
        want = 1.0 - np.sqrt(
            res.sigma2_eps / (counts[entity] * res.sigma2_u + res.sigma2_eps)
        )
        assert theta == pytest.approx(want, rel=1e-12), entity
    assert len(set(res.theta.values())) == 3  # T_iが違えばθ_iも違う


@pytest.mark.parametrize(
    "ids",
    [
        list(range(1, 13)),
        [f"unit{i}" for i in (1, 2, 10, 11, 3, 12, 4, 5, 6, 7, 8, 9)],
    ],
    ids=["integer", "string"],
)
def test_theta_keys_are_the_entity_ids_as_strings_in_lexicographic_order(ids):
    """IDは文字列表現をキーにし、辞書順に並べる（整数IDでも`"10"`が`"2"`より前）。"""
    rng = np.random.default_rng(1)
    rows = [(i, t) for i in ids for t in range(4)]
    n = len(rows)
    x = rng.normal(size=n)
    df = pl.DataFrame(
        {
            "entity": [r[0] for r in rows],
            "y": 1 + x + rng.normal(size=n),
            "x": x,
        }
    )
    res = _fit(df, cov_type="classical")

    assert list(res.theta) == sorted(str(i) for i in ids)
    assert len(res.theta) == res.n_entities == len(ids)
    assert all(isinstance(v, float) for v in res.theta.values())


@pytest.mark.parametrize(
    "options",
    [
        {"cov_type": "classical"},
        {"cov_type": "hc1"},
        {"cov_type": "cluster", "cluster": "time"},
        {"cov_type": "dk", "dk_time": "time", "dk_bandwidth": 1},
        {"cov_type": "classical", "confidence_level": 0.9},
    ],
)
def test_variance_components_do_not_depend_on_inference_options(options):
    df = _panel(2, entity_sd=1.0, time_sd=1.0)
    base = _fit(df, time="time", cov_type="classical")
    res = _fit(df, time="time", **options)
    for name in (
        "sigma2_eps",
        "sigma2_u",
        "sigma2_lambda",
        "theta_entity",
        "theta_time",
        "theta_total",
    ):
        assert getattr(res, name) == getattr(base, name), name


def test_two_way_with_only_the_time_variance_keeps_only_the_time_weight():
    """`σ_u² = 0`・`σ_λ² > 0`なら`θ₁ = 0`・`θ₃ = 0`・`θ₂ > 0`で、変換は時点平均
    だけを引く（時点効果だけのRE）。係数は`y - θ₂ȳ.ₜ`・`x - θ₂x̄.ₜ`への最小二乗と一致する。"""
    df = _panel(0)
    res = _fit(df, time="time", cov_type="classical")
    assert res.sigma2_u == 0.0
    assert res.sigma2_lambda > 0.0
    assert res.theta_entity == 0.0
    assert res.theta_total == 0.0
    assert res.theta_time > 0.0

    th = res.theta_time
    y = (
        df["y"].to_numpy()
        - th
        * df.select(pl.col("y").mean().over("time")).to_series().to_numpy()
    )
    x = (
        df["x"].to_numpy()
        - th
        * df.select(pl.col("x").mean().over("time")).to_series().to_numpy()
    )
    design = np.column_stack([np.full(df.height, 1 - th), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert res.params["const"] == pytest.approx(beta[0], rel=1e-9)
    assert res.params["x"] == pytest.approx(beta[1], rel=1e-9)


def test_two_way_with_only_the_entity_variance_keeps_only_the_entity_weight():
    df = _panel(3, entity_sd=3.0, time_sd=0.0)
    res = _fit(df, time="time", cov_type="classical")
    assert res.sigma2_lambda == 0.0
    assert res.sigma2_u > 0.0
    assert res.theta_time == 0.0
    assert res.theta_total == 0.0
    assert res.theta_entity > 0.0
    # 変換は個体平均だけを引く（個体効果だけのRE）。係数は変換後の回帰と一致する。
    th = res.theta_entity
    y = (
        df["y"].to_numpy()
        - th
        * df.select(pl.col("y").mean().over("entity")).to_series().to_numpy()
    )
    x = (
        df["x"].to_numpy()
        - th
        * df.select(pl.col("x").mean().over("entity")).to_series().to_numpy()
    )
    beta = np.linalg.lstsq(
        np.column_stack([np.full(df.height, 1 - th), x]), y, rcond=None
    )[0]
    assert res.params["const"] == pytest.approx(beta[0], rel=1e-9)
    assert res.params["x"] == pytest.approx(beta[1], rel=1e-9)


def test_one_way_negative_estimate_is_raised_to_zero():
    """個体効果の無いデータで`σ_u²`の推定が負になると0に切り上げ、`θ_i`はすべて0
    （変換なし）で、係数はpooled OLSと一致する。"""
    df = _panel(0, entity_sd=0.0, time_sd=0.0)
    res = _fit(df, cov_type="classical")
    assert res.sigma2_u == 0.0
    assert set(res.theta.values()) == {0.0}
    x = df["x"].to_numpy()
    ols = np.linalg.lstsq(
        np.column_stack([np.ones(len(x)), x]), df["y"].to_numpy(), rcond=None
    )[0]
    assert res.params["x"] == pytest.approx(ols[1], rel=1e-9)


def test_one_way_with_only_dk_time_has_theta_and_no_two_way_fields():
    res = _fit(_panel(5), cov_type="dk", dk_time="time", dk_bandwidth=1)
    assert res.theta is not None
    assert res.sigma2_lambda is None
    assert (res.theta_entity, res.theta_time, res.theta_total) == (
        None,
        None,
        None,
    )


def test_two_way_has_no_one_way_theta():
    res = _fit(_panel(5, entity_sd=1.0), time="time", cov_type="classical")
    assert res.theta is None
    assert all(
        isinstance(v, float)
        for v in (
            res.sigma2_eps,
            res.sigma2_u,
            res.sigma2_lambda,
            res.theta_entity,
            res.theta_time,
            res.theta_total,
        )
    )


def test_two_way_without_regressors_that_vary_in_both_dimensions():
    """2-way within変換後に分散のある列が無くても（内部のFEの説明変数が0個）、
    `σ_ε²`は2-way within変換した`y`の残差平方和を`(N-1)(T-1)`で割った値になる。"""
    df = _panel(6, entity_sd=1.0, time_sd=1.0).with_columns(
        xi=pl.col("entity").str.slice(1).cast(pl.Float64) * 0.7
    )
    res = RE(
        df,
        y="y",
        x=["xi"],
        entity="entity",
        options=REOptions(time="time", cov_type="classical"),
    ).fit()
    y = df["y"].to_numpy().reshape(8, 5)
    within = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0) + y.mean()
    want = float((within**2).sum()) / ((8 - 1) * (5 - 1))
    assert res.sigma2_eps == pytest.approx(want, rel=1e-12)


def test_tiny_error_variance_gives_theta_close_to_one_or_a_perfect_fit():
    """誤差分散が極小でも、`θ ≈ 1`で成功する。完全に当てはまる極限
    （`σ_ε² ≈ 1e-18`）では、分散成分を求めるwithin回帰の残差が総平方和の
    `k * epsilon`倍以下になり、完全フィット（`PERFECT_FIT`）として止まる。"""
    ok = _fit(
        _panel(1, entity_sd=1.0, time_sd=0.0, noise_sd=1e-6),
        cov_type="classical",
    )
    assert 0.9999 < min(ok.theta.values()) < 1.0
    with raises_error(
        ErrorCode.PERFECT_FIT,
        stage="VARIANCE_COMPONENT_WITHIN_REGRESSION",
    ):
        _fit(
            _panel(1, entity_sd=1.0, time_sd=0.0, noise_sd=1e-9),
            cov_type="classical",
        )


def test_noise_just_above_the_perfect_fit_threshold_is_a_near_singular_final_regression():
    """完全フィットの閾値（残差の標準偏差が個体内のyの約`3e-8`倍）のすぐ上では、`σ_ε²`が
    極小で`θ ≈ 1`となり、最終回帰の準偏差変換した定数列`1 - θ`がほぼ0になる。この設計行列は
    実際に悪条件なので`MATRIX_NEAR_SINGULAR`のままにする（最終回帰は被説明変数が準偏差変換後の
    値で、元のモデルの総平方和が当てはまらないため完全フィットの判定を足さない）。この帯は
    狭く、下は`PERFECT_FIT`、上（`1e-7`以上）は通常どおり推定できる。"""
    for seed in (1, 2, 3, 4):
        with raises_error(
            ErrorCode.MATRIX_NEAR_SINGULAR,
            matrix="COEFFICIENT_COVARIANCE",
            stage="QUASI_DEMEANED_REGRESSION",
        ):
            _fit(
                _panel(seed, entity_sd=1.0, time_sd=0.0, noise_sd=3e-8),
                cov_type="classical",
            )
    with raises_error(
        ErrorCode.PERFECT_FIT, stage="VARIANCE_COMPONENT_WITHIN_REGRESSION"
    ):
        _fit(
            _panel(1, entity_sd=1.0, time_sd=0.0, noise_sd=1e-8),
            cov_type="classical",
        )
    _fit(
        _panel(1, entity_sd=1.0, time_sd=0.0, noise_sd=1e-7),
        cov_type="classical",
    )


def test_theta_is_computed_once_per_result():
    """`theta`は個体数に比例する辞書のため、同じ結果オブジェクトでは同じ辞書を返す
    （参照のたびに作り直さない）。"""
    res = _fit(_panel(5), cov_type="classical")
    assert res.theta is res.theta
