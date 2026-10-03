"""FEのcluster/dk（Driscoll-Kraay）の第2リファレンス（R: plm＋sandwich）との
クロスチェックテスト。

`test_fe_crosscheck.py`（fixest）はcluster/dkの主たる参照値で、linearmodelsは
小標本補正と推論の自由度の規約が違うため使えない。ここでは
`tests/fixtures/benchmarks/fe_plm_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_plm_crosscheck_fixtures.py`で生成）を
用いて、fixestとは別実装（plmのwithin変換・`sandwich::vcovCL`・
`plm::vcovSCC`）と一致することを確認する。

## 検証範囲

1-way FEの`cluster`（entityクラスター）と`dk`のみ。plmは2-wayのwithinに対する
クラスター・SCC共分散行列を持たず、クラスター列もgroup/timeしか指定できない
ため、2-way・entity以外のクラスター列（その`G=q+1`等の境界を含む）はfixest側
だけが参照値になる（`test_fe_crosscheck.py`）。比較する統計量は係数・標準誤差・
t統計量・p値・信頼区間・F統計量・F p値・within R²（AIC/BIC等はplmが提供しない）。

## 境界ケース

fixest側（`test_fe_crosscheck.py`）の境界ケースのうちplmで再現できるものを持つ:
クラスター数`G = q+1`・`G = 2`（plmのクラスターはentity単位のため先頭3・2
entityに絞って作る）、dkの`T = q+1`、dkのバンド幅0・1、entityあたりの観測数が
偏った不均衡（クラスターサイズの偏りの代用）。いずれもplm側で`wald()`のような
分母自由度の切り上げはなく、手計算した自由度`G-1`/`T-1`で比較する。

## 独立性の限界

- cluster: 標準誤差はplm自身ではなく、plmのwithin変換後データに`sandwich::vcovCL`を
  当てたもの。補正係数は`sandwich`が計算するが、`K=k+1`（吸収したentity効果を
  1つ数える。fixestの`K.fixef=nonnested`と同じ数え方）は定数項を加えて選んだ規約で、
  fixestと本実装と同じ前提。t検定の自由度`G-1`とF p値の分母自由度も同様に手計算。
- dk: 補正係数`T/(T-1)·(n-1)/(n-k-G)`はfixestの規約をR側で手計算で掛けたもの
  （plm自身の`type="sss"`は固定効果を数えないため使えない）。したがって
  plmが独立に検証するのはカーネル・バンド幅の規約で、補正係数そのものは
  fixestだけが検証する。自由度`T-1`も同様に手計算。

## `bandwidth == T-1`

fixestは`bandwidth == T-1`で最終ラグ項を落とすため本実装と一致しない。plmは
最終ラグを落とさず標準のBartlettカーネルと一致するので、この境界の参照値は
このファイルだけが持つ（`dk_max_bandwidth`）。

役割分担:
    - 主リファレンス（linearmodels、classical/hc1）: `test_fe_reference.py`
    - 独立実装（R: fixest、全cov_type）: `test_fe_crosscheck.py`
    - 第2リファレンス（R: plm、1-wayのcluster/dk）: このファイル
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _constants import DATA_DIR
from _helpers import load_wooldridge_dataset
from _tolerances import TOLERANCES
from econometricsmodels import FE, FEOptions

from benchmark.common import (
    WAGEPAN_ENTITY,
    WAGEPAN_X,
    WAGEPAN_Y,
)
from benchmark.panel.fixtures.generate_fe_plm_crosscheck_fixtures import (
    EXPLICIT_BANDWIDTH_SCENARIOS,
    EXPLICIT_BANDWIDTHS,
    MAX_BANDWIDTH_SCENARIOS,
    NUMERIC_SCENARIOS,
    WAGEPAN_COV_TYPES,
    _cov_types_for,
    dk_three_periods_frame,
    first_entities_frame,
    skewed_entity_sizes_frame,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "fe_plm_crosscheck.json"
)

RTOL = TOLERANCES["fe_plm_crosscheck"]["rtol"]
ATOL = TOLERANCES["fe_plm_crosscheck"]["atol"]
ATOL_P_VALUE = TOLERANCES["fe_plm_crosscheck"]["atol_p_value"]
RTOL_P_VALUE = TOLERANCES["fe_plm_crosscheck"]["rtol_p_value"]

ONE_WAY_CASES = [
    (scenario, cov_type)
    for scenario in NUMERIC_SCENARIOS
    for cov_type in _cov_types_for(scenario)
]


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)
# p値は絶対誤差の下限を設けず相対誤差だけで比較する（`test_fe_crosscheck.py`と
# 同じ理由）。
_assert_p_close = partial(assert_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE)
_assert_p_dict_close = partial(
    assert_dict_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE
)


def _check_result(
    res, ref: dict, label: str, *, expected_df: int, n_slopes: int
) -> None:
    """係数・標準誤差・検定統計量・p値・信頼区間・F統計量・F p値と、
    推論の自由度（clusterは`G-1`、dkは`T-1`）の検証。

    自由度の規約は参照側（plm）でも手計算のため、公開属性を直接アサートして
    本実装がその規約を使っていることを確認する。
    """
    assert res.stat_df == expected_df, label
    assert (res.f_df_num, res.f_df_denom) == (n_slopes, expected_df), label
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(
        res.test_stats, ref["test_stats"], f"{label}/test_stats"
    )
    _assert_p_dict_close(res.p_values, ref["p_values"], f"{label}/p_values")
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        _assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    _assert_close(res.f_statistic, ref["f_statistic"], f"{label}/f_statistic")
    _assert_p_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")
    # within R²はcov_typeに依存しない（plmのsummary(model)$r.squared）。
    _assert_close(
        res.r_squared_within,
        ref["r_squared_within"],
        f"{label}/r_squared_within",
    )


def _expected_df(
    df: pl.DataFrame, cov_type: str, *, entity: str = "entity"
) -> int:
    """clusterは`G-1`（entity数）、dkは`T-1`（ユニークな時点数）。"""
    if cov_type == "cluster":
        return df[entity].n_unique() - 1
    return df["time"].n_unique() - 1


def _fit_frame(
    df: pl.DataFrame,
    cov_type: str,
    *,
    dk_bandwidth: int | None = None,
):
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    if cov_type == "dk":
        options = FEOptions(
            cov_type="dk", dk_time="time", dk_bandwidth=dk_bandwidth
        )
    else:
        options = FEOptions(cov_type=cov_type)
    return FE(df, y="y", x=x_cols, entity="entity", options=options).fit()


def _fit(scenario: str, cov_type: str, *, dk_bandwidth: int | None = None):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    return _fit_frame(df, cov_type, dk_bandwidth=dk_bandwidth)


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


@pytest.mark.parametrize("scenario, cov_type", ONE_WAY_CASES)
def test_synthetic_one_way_matches_plm(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    _check_result(
        _fit(scenario, cov_type),
        crosscheck[scenario]["one_way"][cov_type],
        f"{scenario}/one_way/{cov_type}",
        expected_df=_expected_df(df, cov_type),
        n_slopes=df.width - 3,
    )


# ── bandwidth == T-1（fixestが最終ラグを落とす境界、plmのみが参照値） ─────


@pytest.mark.parametrize("scenario", MAX_BANDWIDTH_SCENARIOS)
def test_dk_max_bandwidth_matches_plm(crosscheck, scenario):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    n_periods = df["time"].n_unique()
    res = _fit(scenario, "dk", dk_bandwidth=n_periods - 1)

    _check_result(
        res,
        crosscheck["dk_max_bandwidth"][scenario],
        f"{scenario}/one_way/dk_max_bandwidth",
        expected_df=n_periods - 1,
        n_slopes=df.width - 3,
    )


# ── dkのバンド幅を既定値以外で明示指定（0, 1） ─────────────────────────


@pytest.mark.parametrize("bandwidth", EXPLICIT_BANDWIDTHS)
@pytest.mark.parametrize("scenario", EXPLICIT_BANDWIDTH_SCENARIOS)
def test_dk_explicit_bandwidth_matches_plm(crosscheck, scenario, bandwidth):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    res = _fit_frame(df, "dk", dk_bandwidth=bandwidth)

    _check_result(
        res,
        crosscheck["dk_explicit_bandwidth"][scenario][str(bandwidth)],
        f"{scenario}/one_way/dk_bandwidth_{bandwidth}",
        expected_df=df["time"].n_unique() - 1,
        n_slopes=df.width - 3,
    )


# ── 境界ケース（fixestの境界のうちplmで再現できるもの） ──────────────────


def test_dk_three_periods_matches_plm(crosscheck):
    """Driscoll-Kraayの時点数`T = q+1`（q=2、分母自由度`T-1 = q`）の境界。"""
    df = dk_three_periods_frame()
    res = _fit_frame(df, "dk")

    _check_result(
        res,
        crosscheck["boundary"]["dk_three_periods"],
        "boundary/dk_three_periods",
        expected_df=2,
        n_slopes=2,
    )


def test_cluster_g3_two_slopes_matches_plm(crosscheck):
    """クラスター数`G = q+1`（q=2、分母自由度`G-1 = q`）の境界。plmのクラスターは
    entity単位のため、先頭3 entityだけ残して作る。"""
    df = first_entities_frame("fe_baseline.csv", 3)
    res = _fit_frame(df, "cluster")

    _check_result(
        res,
        crosscheck["boundary"]["cluster_g3_two_slopes"],
        "boundary/cluster_g3_two_slopes",
        expected_df=2,
        n_slopes=2,
    )


def test_cluster_g2_matches_plm(crosscheck):
    """クラスタ数境界（G=2、q=1でG>q）。先頭2 entityだけ残して作る。"""
    df = first_entities_frame("fe_baseline_k1.csv", 2)
    res = _fit_frame(df, "cluster")

    _check_result(
        res,
        crosscheck["boundary"]["cluster_g2"],
        "boundary/cluster_g2",
        expected_df=1,
        n_slopes=1,
    )


def test_skewed_entity_sizes_cluster_matches_plm(crosscheck):
    """entityあたりの観測数が偏った（先頭[2,3,5,10]期を循環）不均衡パネルの
    entityクラスター。plmは非entityのクラスター列を扱えないため、クラスター
    サイズの偏りはentityの観測数の偏りで代用する。"""
    df = skewed_entity_sizes_frame()
    res = _fit_frame(df, "cluster")

    _check_result(
        res,
        crosscheck["boundary"]["skewed_entity_sizes"],
        "boundary/skewed_entity_sizes",
        expected_df=df["entity"].n_unique() - 1,
        n_slopes=2,
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_one_way_matches_plm(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    res = FE(
        df,
        y=WAGEPAN_Y,
        x=WAGEPAN_X,
        entity=WAGEPAN_ENTITY,
        options=FEOptions(cov_type=cov_type),
    ).fit()

    _check_result(
        res,
        crosscheck["wagepan"]["one_way"][cov_type],
        f"wagepan/one_way/{cov_type}",
        expected_df=_expected_df(df, cov_type, entity=WAGEPAN_ENTITY),
        n_slopes=len(WAGEPAN_X),
    )
