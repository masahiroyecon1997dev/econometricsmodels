"""ほぼ共線な傾きの検定統計量が、モデルの厳密な再パラメータ化で変わらないこと。

傾きの2列`(x1, x2)`を`(x1, (x2 - x1) / 2^k)`に置き換える（`2^k`は`eps`に近い2の冪）。近い値同士の
引き算は正確で、2の冪での割り算も正確なので、OLS・WLS・IVでは、データの丸めを増やさない
モデルの再表現になる（FE・RE・FE-IVは個体平均を引くときの丸めで、約`1e-9`の摂動が入る。
許容誤差`1e-7`はその分を見込んでいる）。
置き換えた設計は条件が良く、全体のWald/F統計量・Hausman・Wu-Hausman・分散成分は、傾きの正則な
線形変換で変わらない。元の設計（条件数が最大2.8e7）と置き換えた設計の統計量の一致で、
`tests/cross_method/test_near_collinear_exact.py`の有理数の厳密値が作れない場合
（`hac`・`dk`、Hausman・Wu-Hausman）の精度を確かめる。REの全体Fは、返された`θ`を使う厳密値
（同ファイル）でも確かめる。

REのHausman補助回帰とbetween回帰は、相対残差`1e-7`未満の列を「別名」として落とす
（Rの`lm`流。相関ランダム効果のための設計）。その窓（`eps`が約`3e-7`以下）では再パラメータ化の
不変性が成り立たないので、不変性は窓の外（`eps = 1e-5, 1e-6`）で確かめ、窓の中の挙動
（`hausman_df`が落ちること）は別のテストで固定する。
"""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _tolerances import TOLERANCES
from econometricsmodels import (
    FE,
    FEIV,
    IV,
    OLS,
    RE,
    WLS,
    FEIVOptions,
    FEOptions,
    IVOptions,
    OLSOptions,
    REOptions,
    WLSOptions,
)
from test_near_collinear_exact import (
    N_ENTITIES,
    N_PERIODS,
    _iv_data,
    _ols_data,
    _panel_data,
)

if TYPE_CHECKING:
    from collections.abc import Callable

assert_invariant = partial(
    assert_close, **TOLERANCES["near_collinear_reparametrization"]
)

EPS = [1e-5, 1e-6, 1e-7]
# REは別名の窓（`eps`が約3e-7以下）の外だけ。
RE_EPS = [1e-5, 1e-6]


def _reparametrize(
    df: pl.DataFrame, base: str, other: str, eps: float
) -> pl.DataFrame:
    """`other`を`(other - base) / 2^k`に置き換える（`2^k`は`eps`に最も近い2の冪）。"""
    scale = 2.0 ** round(math.log2(eps))
    return df.with_columns(
        ((pl.col(other) - pl.col(base)) / scale).alias(other)
    )


def _with_time(df: pl.DataFrame) -> pl.DataFrame:
    """HAC用の一意な時間列`t`（行番号）と、DK用の個体内の時点`period`を足す。"""
    return df.with_columns(
        pl.int_range(pl.len()).alias("t"),
        pl.Series("period", np.tile(np.arange(N_PERIODS), N_ENTITIES))
        if df.height == N_ENTITIES * N_PERIODS
        else pl.lit(0).alias("period"),
    )


def _check(
    fit: Callable[[pl.DataFrame], dict[str, float]],
    original: pl.DataFrame,
    reparametrized: pl.DataFrame,
    label: str,
) -> None:
    """二つの設計の統計量が一致すること（`fit`は統計量の辞書を返す）。"""
    got, expected = fit(original), fit(reparametrized)
    assert got.keys() == expected.keys()
    for name in got:
        assert_invariant(got[name], expected[name], f"{label} {name}")


# ── OLS・WLS・IV（hac を含む） ─────────────────────────────────────────────


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hac", "cluster"])
def test_ols_overall_f_is_invariant_to_reparametrization(cov_type, eps):
    options = OLSOptions(
        cov_type=cov_type,
        cluster="g" if cov_type == "cluster" else None,
        hac_time="t" if cov_type == "hac" else None,
    )

    def fit(df: pl.DataFrame) -> dict[str, float]:
        res = OLS(df, "y", ["x1", "x2"], options=options).fit()
        return {"f_statistic": res.f_statistic}

    df = _with_time(_ols_data(eps))
    _check(fit, df, _reparametrize(df, "x1", "x2", eps), f"OLS {cov_type}")


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hac", "cluster"])
def test_wls_overall_f_is_invariant_to_reparametrization(cov_type, eps):
    options = WLSOptions(
        cov_type=cov_type,
        cluster="g" if cov_type == "cluster" else None,
        hac_time="t" if cov_type == "hac" else None,
    )

    def fit(df: pl.DataFrame) -> dict[str, float]:
        res = WLS(df, "y", ["x1", "x2"], "w", options=options).fit()
        return {"f_statistic": res.f_statistic}

    df = _with_time(_ols_data(eps))
    _check(fit, df, _reparametrize(df, "x1", "x2", eps), f"WLS {cov_type}")


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hac", "cluster"])
def test_iv_wald_and_wu_hausman_are_invariant_to_reparametrization(
    cov_type, eps
):
    options = IVOptions(
        cov_type=cov_type,
        cluster="g" if cov_type == "cluster" else None,
        hac_time="t" if cov_type == "hac" else None,
    )

    def fit(df: pl.DataFrame) -> dict[str, float]:
        res = IV(
            df,
            "y",
            ["x1", "x2"],
            ["endog"],
            ["z1", "z2", "z3"],
            options=options,
        ).fit()
        return {
            "wald": res.wald_statistic * res.wald_df_num,
            "wu_hausman": res.wu_hausman_statistic,
        }

    df = _with_time(_iv_data(eps))
    _check(fit, df, _reparametrize(df, "x1", "x2", eps), f"IV {cov_type}")


# GMMの設定（`gmm_type`と2段階目以降の重み）。HAC重み・クラスター重みは、重みの弱い方向が入力の丸めに
# 敏感（HAC）、または重みの因子`A`の条件数が設計より大きくなる（クラスター。`eps = 1e-7`で受け入れ
# 限界を超える）ので、`eps`を絞る。
GMM_SETTINGS = [
    ("one_step", None, EPS),
    ("two_step", "robust", EPS),
    ("two_step", "classical", EPS),
    ("iterated", "robust", EPS),
    ("two_step", "cluster", [1e-5, 1e-6]),
    ("two_step", "hac", [1e-5]),
]
_IV_GMM_PARAMS = [
    (gmm_type, weight, cov_type, eps)
    for gmm_type, weight, eps_values in GMM_SETTINGS
    for cov_type in (
        ["classical", "hc1", "hc2", "hc3", "hac", "cluster"]
        if (gmm_type, weight) == ("two_step", "robust")
        else ["classical", "hc1", "hac", "cluster"]
    )
    for eps in eps_values
]


@pytest.mark.parametrize(
    ("gmm_type", "weight", "cov_type", "eps"),
    _IV_GMM_PARAMS,
    ids=[f"{g}-{w}-{c}-{e:g}" for g, w, c, e in _IV_GMM_PARAMS],
)
def test_iv_gmm_wald_and_hansen_j_are_invariant_to_reparametrization(
    gmm_type, weight, cov_type, eps
):
    options = IVOptions(
        estimator="gmm",
        gmm_type=gmm_type,
        gmm_weight_type=weight,
        hac_lags=2 if weight == "hac" else None,
        cov_type=cov_type,
        cluster="g" if "cluster" in (cov_type, weight) else None,
        hac_time="t" if "hac" in (cov_type, weight) else None,
    )

    def fit(df: pl.DataFrame) -> dict[str, float]:
        res = IV(
            df,
            "y",
            ["x1", "x2"],
            ["endog"],
            ["z1", "z2", "z3"],
            options=options,
        ).fit()
        return {
            "wald": res.wald_statistic,
            "hansen_j": res.overid_statistic,
        }

    df = _with_time(_iv_data(eps))
    _check(
        fit,
        df,
        _reparametrize(df, "x1", "x2", eps),
        f"IV GMM {gmm_type} {weight} {cov_type}",
    )


# ── FE・FE-IV（dk を含む） ────────────────────────────────────────────


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster", "dk"])
def test_fe_overall_f_is_invariant_to_reparametrization(cov_type, eps):
    options = FEOptions(
        cov_type=cov_type, dk_time="period" if cov_type == "dk" else None
    )

    def fit(df: pl.DataFrame) -> dict[str, float]:
        res = FE(
            df, y="y", x=["x1", "x3"], effects=["entity"], options=options
        ).fit()
        return {"f_statistic": res.f_statistic}

    df = _with_time(_panel_data(eps))
    _check(fit, df, _reparametrize(df, "x1", "x3", eps), f"FE {cov_type}")


@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster", "dk"])
def test_fe_iv_wald_and_wu_hausman_are_invariant_to_reparametrization(
    cov_type, eps
):
    options = FEIVOptions(
        cov_type=cov_type, dk_time="period" if cov_type == "dk" else None
    )

    def fit(df: pl.DataFrame) -> dict[str, float]:
        res = FEIV(
            df,
            y="y",
            x_exog=["x1", "x3"],
            x_endog=["endog"],
            instruments=["z1", "z2", "z3"],
            effects=["entity"],
            options=options,
        ).fit()
        return {
            "wald": res.wald_statistic * res.wald_df_num,
            "wu_hausman": res.wu_hausman_statistic,
        }

    df = _with_time(_panel_data(eps))
    _check(fit, df, _reparametrize(df, "x1", "x3", eps), f"FE-IV {cov_type}")


_FE_IV_GMM_PARAMS = [
    (gmm_type, weight, cov_type, eps)
    for gmm_type, weight, eps_values in GMM_SETTINGS
    if weight != "hac"
    for cov_type in (
        ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
        if (gmm_type, weight) == ("two_step", "robust")
        else ["classical", "hc1", "cluster", "dk"]
    )
    for eps in eps_values
]


@pytest.mark.parametrize(
    ("gmm_type", "weight", "cov_type", "eps"),
    _FE_IV_GMM_PARAMS,
    ids=[f"{g}-{w}-{c}-{e:g}" for g, w, c, e in _FE_IV_GMM_PARAMS],
)
def test_fe_iv_gmm_wald_and_hansen_j_are_invariant_to_reparametrization(
    gmm_type, weight, cov_type, eps
):
    options = FEIVOptions(
        estimator="gmm",
        gmm_type=gmm_type,
        gmm_weight_type=weight,
        cov_type=cov_type,
        dk_time="period" if cov_type == "dk" else None,
    )

    def fit(df: pl.DataFrame) -> dict[str, float]:
        res = FEIV(
            df,
            y="y",
            x_exog=["x1", "x3"],
            x_endog=["endog"],
            instruments=["z1", "z2", "z3"],
            effects=["entity"],
            options=options,
        ).fit()
        return {
            "wald": res.wald_statistic,
            "hansen_j": res.overid_statistic,
        }

    df = _with_time(_panel_data(eps))
    _check(
        fit,
        df,
        _reparametrize(df, "x1", "x3", eps),
        f"FE-IV GMM {gmm_type} {weight} {cov_type}",
    )


# ── RE ────────────────────────────────────────────────────────────────


def _re_options(cov_type: str) -> REOptions:
    return REOptions(
        cov_type=cov_type, dk_time="period" if cov_type == "dk" else None
    )


@pytest.mark.parametrize("eps", RE_EPS)
@pytest.mark.parametrize(
    "cov_type", ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
)
def test_re_overall_f_and_hausman_are_invariant_to_reparametrization(
    cov_type, eps
):
    """`θ`が無理数のため有理数の厳密値が作れないREの、全体FとHausman統計量の精度。"""

    def fit(df: pl.DataFrame) -> dict[str, float]:
        res = RE(
            df,
            y="y",
            x=["x1", "x3"],
            effects=["entity"],
            options=_re_options(cov_type),
        ).fit()
        assert res.hausman_df == 2
        return {
            "f_statistic": res.f_statistic,
            "hausman": res.hausman_statistic,
        }

    df = _with_time(_panel_data(eps))
    _check(fit, df, _reparametrize(df, "x1", "x3", eps), f"RE {cov_type}")


@pytest.mark.parametrize("cov_type", ["classical", "cluster"])
def test_re_drops_a_nearly_aliased_column_from_the_hausman_regression(
    cov_type,
):
    """傾きの差が小さい（`eps = 1e-7`、列を揃えた条件数は約2.8e7でガードの内側）と、REの
    Hausman補助回帰とbetween回帰は、相対残差`1e-7`未満の列をRの`lm`流に別名として落とす。
    `hausman_df`が2から1に減り、再パラメータ化した設計（別名にならない）とは統計量が変わる。
    相関ランダム効果のための設計で、受け入れ範囲（条件数のガード）とは別の判定。"""
    options = REOptions(cov_type=cov_type)
    df = _panel_data(1e-7)
    inside_window = RE(
        df, y="y", x=["x1", "x3"], effects=["entity"], options=options
    ).fit()
    reparametrized = RE(
        _reparametrize(df, "x1", "x3", 1e-7),
        y="y",
        x=["x1", "x3"],
        effects=["entity"],
        options=options,
    ).fit()
    assert inside_window.hausman_df == 1
    assert reparametrized.hausman_df == 2


@pytest.mark.parametrize("eps", [1e-7, 6e-8])
@pytest.mark.parametrize(
    "cov_type", ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
)
def test_re_returns_finite_statistics_inside_the_alias_window(cov_type, eps):
    """別名の窓の中（`eps = 1e-7`・`6e-8`。条件数のガードの内側）でも、全`cov_type`で全体Fと
    Hausman統計量が有限で返る（有理数の厳密値は`test_near_collinear_exact.py`）。"""
    res = RE(
        _with_time(_panel_data(eps)),
        y="y",
        x=["x1", "x3"],
        effects=["entity"],
        options=_re_options(cov_type),
    ).fit()
    assert math.isfinite(res.f_statistic)
    assert res.f_statistic >= 0
    assert math.isfinite(res.hausman_statistic)
    assert res.hausman_statistic >= 0
