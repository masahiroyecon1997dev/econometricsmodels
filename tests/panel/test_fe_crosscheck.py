"""FE の独立実装（R: fixest）とのクロスチェックテスト。

主リファレンス（linearmodels）との厳密比較は`test_fe_reference.py`で行う。
ここでは`tests/fixtures/benchmarks/fe_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_crosscheck_fixtures.py`で生成）を
用いて、linearmodelsとは独立した実装（R: fixest）との一致を確認する。

## classical/hc1/hc2/hc3とclusterで許容誤差が異なる理由

classical/hc1/hc2/hc3はfixestと機械精度で一致する（実測相対誤差1e-14程度、
1-way/2-way双方）ため`RTOL_STRICT`で厳密比較する。**clusterのみ**、fixestの
小標本補正の既定慣行（Stata流のG/(G-1)補正）が本実装・linearmodelsと異なり、
`ssc(G.adj=FALSE, K.fixef=...)`で調整してもなお1-way相対誤差1.8e-5程度・
2-way相対誤差0.21%程度の乖離が残る（規約上の系統的な差、実装バグではない。
`run_fixest_benchmark.R`のコメント参照）。追加検証は追跡中。

## このファイルだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.PanelOLS`が提供しないため、fixestが唯一の参照
  実装になる（`benchmark/panel/references/linearmodels_ref.py`モジュールdoc
  参照）。
- **aic/bic/log_likelihood**: `linearmodels.PanelOLS`が提供しないため、fixestのみで検証する。
- **2-way FEのr_squared_within**: `linearmodels`自身がentityのみdemeanの
  別定義を使うため、fixestの`fitstat(m, "wr2")`のみで検証する。

## hacを含まない理由

`cov_type="hac"`（Driscoll-Kraay）はfixestの`vcov="DK"`の既定バンド幅公式・
小標本補正の慣行が本実装・linearmodelsと異なり、明示的にバンド幅を揃えても
標準誤差が実用的な許容誤差でも一致しないため、`linearmodels`のみを参照実装
とする単一参照実装の例外として扱う（ユーザー確認済み）。
`fe_crosscheck.json`にはhacのキー自体が存在しない。

役割分担:
    - 構造・API・`fixed_effects()`: `test_fe_api.py`
    - `ValidationError`/`ComputationError` パス: `test_fe_validation.py`
    - 主リファレンス（linearmodels）との数値照合: `test_fe_reference.py`
    - 独立実装（R: fixest）とのクロスチェック: このファイル
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

from benchmark.common import WAGEPAN_ENTITY, WAGEPAN_TIME, WAGEPAN_X, WAGEPAN_Y
from benchmark.panel.fixtures.generate_fe_crosscheck_fixtures import (
    COV_TYPES,
    ONE_WAY_ONLY_SCENARIOS,
    TWO_WAY_SCENARIOS,
    WAGEPAN_COV_TYPES,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "fe_crosscheck.json"
)

RTOL_STRICT = TOLERANCES["fe_crosscheck"]["rtol_strict"]
RTOL_CLUSTER_ONE_WAY = TOLERANCES["fe_crosscheck"]["rtol_cluster_one_way"]
RTOL_CLUSTER_TWO_WAY = TOLERANCES["fe_crosscheck"]["rtol_cluster_two_way"]
ATOL = TOLERANCES["fe_crosscheck"]["atol"]
ATOL_CLUSTER_P_VALUE = TOLERANCES["fe_crosscheck"]["atol_cluster_p_value"]
ATOL_CLUSTER_CONF_INT = TOLERANCES["fe_crosscheck"]["atol_cluster_conf_int"]

ALL_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS

# small_panel（G=5という極端に少ないクラスタ数）はclusterのp_values/conf_int
# の非線形増幅がさらに拡大し（実測最大絶対誤差conf_int~0.40）、他シナリオ
# 向けの許容誤差では収まらない。coef/se/t_statsはこのシナリオでも問題なく
# 一致するため対象外にはせず、p_values/conf_intの数値比較だけをスコープ外に
# する（`iv_crosscheck`の`rtol_hac_small_n`と同型の「小標本ケースは別枠」、
# `_tolerances.py`参照）。
_SMALL_CLUSTER_COUNT_SCENARIO = "small_panel"


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_scalar_close = partial(assert_close, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, atol=ATOL)


def _rtol_for(cov_type: str, *, two_way: bool) -> float:
    if cov_type != "cluster":
        return RTOL_STRICT
    return RTOL_CLUSTER_TWO_WAY if two_way else RTOL_CLUSTER_ONE_WAY


def _check_result(
    res,
    ref: dict,
    label: str,
    *,
    rtol: float,
    cov_type: str,
    scenario: str | None = None,
) -> None:
    """係数・標準誤差・検定統計量・AIC/BIC/log_likelihood・2-way限定の
    Within R2の検証。

    AIC/BIC/log_likelihood・Within R2（クロスチェック側のみ持つ統計量、
    モジュールdoc「このファイルだけが持つ統計量」参照）と係数自体
    （`cov_type`に依存しない点推定）は常に検証する。標準誤差・検定統計量・
    信頼区間は呼び出し元の`rtol`
    （clusterのみ緩め）を使う。p_values/conf_intのみ、cluster時に別途
    `ATOL_CLUSTER_P_VALUE`/`ATOL_CLUSTER_CONF_INT`（t分布CDF・t臨界値×seに
    よる非線形増幅、`_tolerances.py`参照）を使う。

    `scenario`が`_SMALL_CLUSTER_COUNT_SCENARIO`（G=5）かつ`cov_type=="cluster"`
    のときは、G/(G-1)型補正差の相対的な影響がクラスタ数に反比例して拡大し、
    se・t_stats・p_values・conf_intのいずれも他シナリオ向けの許容誤差に
    収まらない（実測確認済み）ため、この組み合わせに限り
    cov_type依存の統計量の数値比較を丸ごとスキップする（coef/aic/bic/
    log_likelihood/r_squared_withinはcov_type非依存のため引き続き検証、
    `iv_crosscheck`の`rtol_hac_small_n`と同型の「小標本ケースは別枠」判断）。
    """
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef", rtol=rtol)

    skip_cov_type_dependent = (
        scenario == _SMALL_CLUSTER_COUNT_SCENARIO and cov_type == "cluster"
    )
    if not skip_cov_type_dependent:
        _assert_dict_close(res.std_errors, ref["se"], f"{label}/se", rtol=rtol)
        _assert_dict_close(
            res.t_stats, ref["t_stats"], f"{label}/t_stats", rtol=rtol
        )
        p_value_atol = ATOL_CLUSTER_P_VALUE if cov_type == "cluster" else ATOL
        for name, ref_p in ref["p_values"].items():
            _assert_scalar_close(
                res.p_values[name],
                ref_p,
                f"{label}/p_values/{name}",
                rtol=rtol,
                atol=p_value_atol,
            )
        conf_int_atol = (
            ATOL_CLUSTER_CONF_INT if cov_type == "cluster" else ATOL
        )
        for name, (ref_lower, ref_upper) in ref["conf_int"].items():
            our_lower, our_upper = res.conf_int[name]
            _assert_scalar_close(
                our_lower,
                ref_lower,
                f"{label}/conf_lower/{name}",
                rtol=rtol,
                atol=conf_int_atol,
            )
            _assert_scalar_close(
                our_upper,
                ref_upper,
                f"{label}/conf_upper/{name}",
                rtol=rtol,
                atol=conf_int_atol,
            )

    _assert_scalar_close(res.aic, ref["aic"], f"{label}/aic", rtol=RTOL_STRICT)
    _assert_scalar_close(res.bic, ref["bic"], f"{label}/bic", rtol=RTOL_STRICT)
    _assert_scalar_close(
        res.log_likelihood,
        ref["log_likelihood"],
        f"{label}/log_likelihood",
        rtol=RTOL_STRICT,
    )
    if "r_squared_within" in ref:
        _assert_scalar_close(
            res.r_squared_within,
            ref["r_squared_within"],
            f"{label}/r_squared_within",
            rtol=RTOL_STRICT,
        )


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_synthetic_one_way_matches_fixest(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    options = FEOptions(cov_type=cov_type)
    res = FE(df, y="y", x=x_cols, entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck[scenario]["one_way"][cov_type],
        f"{scenario}/one_way/{cov_type}",
        rtol=_rtol_for(cov_type, two_way=False),
        cov_type=cov_type,
        scenario=scenario,
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", TWO_WAY_SCENARIOS)
def test_synthetic_two_way_matches_fixest(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    options = FEOptions(cov_type=cov_type, time="time")
    res = FE(df, y="y", x=x_cols, entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck[scenario]["two_way"][cov_type],
        f"{scenario}/two_way/{cov_type}",
        rtol=_rtol_for(cov_type, two_way=True),
        cov_type=cov_type,
        scenario=scenario,
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_one_way_matches_fixest(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = FEOptions(cov_type=cov_type)
    res = FE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(
        res,
        crosscheck["wagepan"]["one_way"][cov_type],
        f"wagepan/one_way/{cov_type}",
        rtol=_rtol_for(cov_type, two_way=False),
        cov_type=cov_type,
    )


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_two_way_matches_fixest(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = FEOptions(cov_type=cov_type, time=WAGEPAN_TIME)
    res = FE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(
        res,
        crosscheck["wagepan"]["two_way"][cov_type],
        f"wagepan/two_way/{cov_type}",
        rtol=_rtol_for(cov_type, two_way=True),
        cov_type=cov_type,
    )
