"""FE の主リファレンス（linearmodels）との数値照合。

`tests/fixtures/benchmarks/fe.json`（`benchmark/panel/fixtures/
generate_fe_fixtures.py`で生成）を読み込み、合成データの全シナリオ×
classical/hc1 × 1-way/2-way（`unbalanced`のみ1-way限定）で、係数・標準
誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する。Wooldridge実
データ（wagepan）も同じフィクスチャ経由で検証する。

## cluster/dkを含まない理由

本実装のcluster・dk（Driscoll-Kraay）の標準誤差は、小標本補正と推論の自由度
（clusterで`G-1`、dkで`T-1`）をfixest・Stata型に合わせており、linearmodels
（`n/(n-extra_df-k)`、自由度は常に`df_resid`）とは一致しない
（`docs/spec/fe-spec.md`3.3節・4章）。この2つは`test_fe_crosscheck.py`の
fixestだけで検証する。

役割分担:
    - 構造・API・`fixed_effects()`: `test_fe_api.py`
    - `ValidationError`/`ComputationError` パス: `test_fe_validation.py`
    - 主リファレンス（linearmodels）との数値照合: このファイル
    - 独立実装（R: fixest）とのクロスチェック: `test_fe_crosscheck.py`

Note:
    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/panel/freeze.py`参照）から読む。2-way FEの
    `r_squared_within`はlinearmodels自身がentityのみdemeanの別定義を使う
    ため対象外とする（`benchmark/panel/references/linearmodels_ref.py`
    モジュールdoc参照。fixestとの一致は`test_fe_crosscheck.py`が担う）。
    hc2/hc3はlinearmodels.PanelOLSが提供しないためこのファイルの対象外
    （`test_fe_crosscheck.py`のfixestのみで検証する単一参照実装の例外）。
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
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
)
from benchmark.panel.fixtures.generate_fe_fixtures import (
    COV_TYPES,
    ONE_WAY_ONLY_SCENARIOS,
    TWO_WAY_SCENARIOS,
    WAGEPAN_COV_TYPES,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks" / "fe.json"
)

RTOL = TOLERANCES["fe_reference"]["rtol"]
ATOL = TOLERANCES["fe_reference"]["atol"]

ALL_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS

ONE_WAY_CASES = [
    (scenario, cov_type)
    for scenario in ALL_SCENARIOS
    for cov_type in COV_TYPES
]
TWO_WAY_CASES = [
    (scenario, cov_type)
    for scenario in TWO_WAY_SCENARIOS
    for cov_type in COV_TYPES
]


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)


def _check_result(
    res, ref: dict, label: str, *, check_r_squared_within: bool
) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(
        res.test_stats, ref["test_stats"], f"{label}/test_stats"
    )
    _assert_dict_close(res.p_values, ref["p_values"], f"{label}/p_values")

    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        _assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    assert res.n_obs == ref["n_obs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"
    assert res.df_model == ref["df_model"], f"{label}/df_model"
    assert res.n_entities == ref["n_entities"], f"{label}/n_entities"

    _assert_close(res.f_statistic, ref["f_statistic"], f"{label}/f_statistic")
    _assert_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")
    if check_r_squared_within:
        _assert_close(
            res.r_squared_within,
            ref["r_squared_within"],
            f"{label}/r_squared_within",
        )
    _assert_close(
        res.r_squared_between,
        ref["r_squared_between"],
        f"{label}/r_squared_between",
    )
    _assert_close(
        res.r_squared_overall,
        ref["r_squared_overall"],
        f"{label}/r_squared_overall",
    )


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


@pytest.mark.parametrize("scenario, cov_type", ONE_WAY_CASES)
def test_matches_linearmodels_one_way(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    options = FEOptions(cov_type=cov_type)
    res = FE(df, y="y", x=x_cols, entity="entity", options=options).fit()

    _check_result(
        res,
        fixtures[scenario]["one_way"][cov_type],
        f"{scenario}/one_way/{cov_type}",
        check_r_squared_within=True,
    )


@pytest.mark.parametrize("scenario, cov_type", TWO_WAY_CASES)
def test_matches_linearmodels_two_way(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    options = FEOptions(cov_type=cov_type, time="time")
    res = FE(df, y="y", x=x_cols, entity="entity", options=options).fit()

    _check_result(
        res,
        fixtures[scenario]["two_way"][cov_type],
        f"{scenario}/two_way/{cov_type}",
        check_r_squared_within=False,
    )


# ── 境界値 ──────────────────────────────────────────────────────


def test_boundary_df1_one_way_matches_linearmodels(fixtures):
    """df_resid=1境界（1-way）の成功パス。entity=3×period=2・k=2、
    df_resid=6-3-2=1（`benchmark/panel/freeze.py`参照）。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_df1_one_way.csv")
    options = FEOptions(cov_type="classical")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    _check_result(
        res,
        fixtures["baseline_df1"]["one_way"],
        "baseline_df1/one_way",
        check_r_squared_within=True,
    )


def test_boundary_df1_two_way_matches_linearmodels(fixtures):
    """df_resid=1境界（2-way）の成功パス。entity=3×period=3・k=3、
    df_resid=9-(3+3+3-1)=1。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_df1_two_way.csv")
    options = FEOptions(cov_type="classical", time="time")
    res = FE(
        df, y="y", x=["x1", "x2", "x3"], entity="entity", options=options
    ).fit()

    _check_result(
        res,
        fixtures["baseline_df1"]["two_way"],
        "baseline_df1/two_way",
        check_r_squared_within=False,
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_one_way_matches_linearmodels(fixtures, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = FEOptions(cov_type=cov_type)
    res = FE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(
        res,
        fixtures["wagepan"]["one_way"][cov_type],
        f"wagepan/one_way/{cov_type}",
        check_r_squared_within=True,
    )


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_two_way_matches_linearmodels(fixtures, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = FEOptions(cov_type=cov_type, time=WAGEPAN_TIME)
    res = FE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(
        res,
        fixtures["wagepan"]["two_way"][cov_type],
        f"wagepan/two_way/{cov_type}",
        check_r_squared_within=False,
    )
