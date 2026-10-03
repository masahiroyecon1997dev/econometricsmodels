"""FE の独立実装（R: fixest）とのクロスチェックテスト。

主リファレンス（linearmodels）との厳密比較（classical/hc1）は
`test_fe_reference.py`で行う。ここでは`tests/fixtures/benchmarks/fe_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_crosscheck_fixtures.py`で生成）を
用いて、linearmodelsとは独立した実装（R: fixest）との一致を確認する。

## cluster/dkはfixestが唯一の参照実装

本実装のcluster・dk（Driscoll-Kraay）の標準誤差は、小標本補正と推論の自由度
（clusterで`G-1`、dkで`T-1`）をfixestの`ssc()`既定に合わせている
（`docs/spec/fe-spec.md`3.3節）ため、linearmodelsとは一致しない。
`test_fe_reference.py`はclassical/hc1のみをlinearmodelsと比較し、cluster/dkは
このファイルのfixestだけで検証する。fixestの既定`ssc()`のまま、全cov_type
（1-way・2-way双方）で機械精度（実測相対誤差1e-14程度）で一致するため、
`RTOL`で厳密比較する。dkはfixestの既定バンド幅が本実装と異なるため、本実装の
既定式で求めたバンド幅を`DK(lag)`に明示的に渡している
（`generate_fe_crosscheck_fixtures.py`参照）。

## このファイルだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.PanelOLS`が提供しないため、fixestが唯一の参照
  実装になる（`benchmark/panel/references/linearmodels_ref.py`モジュールdoc
  参照）。
- **cluster/dk**: 上記の通り、fixestのみで検証する。
- **aic/bic/log_likelihood**: `linearmodels.PanelOLS`が提供しないため、fixestのみで検証する。
- **2-way FEのr_squared_within**: `linearmodels`自身がentityのみdemeanの
  別定義を使うため、fixestの`fitstat(m, "wr2")`のみで検証する。

## dkの対象外シナリオ

`many_regressors`（k=20、T=6）はk>T-1で同時検定の部分行列が構造的に特異になり
本実装が`ValidationError`にするため対象外。wagepan（T=8）は短いTでのDKのため
`fe.json`と同様に対象外（`SCENARIO_COV_TYPES`・`WAGEPAN_COV_TYPES`）。

役割分担:
    - 構造・API・`fixed_effects()`: `test_fe_api.py`
    - `ValidationError`/`ComputationError` パス: `test_fe_validation.py`
    - 主リファレンス（linearmodels、classical/hc1）との数値照合:
      `test_fe_reference.py`
    - 独立実装（R: fixest）とのクロスチェック（cluster/dkはfixestのみ）:
      このファイル
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
    imbalanced_cluster_groups,
)
from benchmark.panel.fixtures.generate_fe_crosscheck_fixtures import (
    COV_TYPES,
    ONE_WAY_ONLY_SCENARIOS,
    SCENARIO_COV_TYPES,
    TWO_WAY_SCENARIOS,
    WAGEPAN_COV_TYPES,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "fe_crosscheck.json"
)

RTOL = TOLERANCES["fe_crosscheck"]["rtol"]
ATOL = TOLERANCES["fe_crosscheck"]["atol"]

ALL_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS


def _cov_types_for(scenario: str) -> list[str]:
    return SCENARIO_COV_TYPES.get(scenario, COV_TYPES)


ONE_WAY_CASES = [
    (scenario, cov_type)
    for scenario in ALL_SCENARIOS
    for cov_type in _cov_types_for(scenario)
]
TWO_WAY_CASES = [
    (scenario, cov_type)
    for scenario in TWO_WAY_SCENARIOS
    for cov_type in _cov_types_for(scenario)
]


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)


def _check_result(res, ref: dict, label: str) -> None:
    """係数・標準誤差・検定統計量・p値・信頼区間・AIC/BIC/log_likelihood・
    （参照値がある場合の）Within R2の検証。

    AIC/BIC/log_likelihood・Within R2はクロスチェック側のみ持つ統計量
    （モジュールdoc「このファイルだけが持つ統計量」参照）。
    """
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

    _assert_close(res.aic, ref["aic"], f"{label}/aic")
    _assert_close(res.bic, ref["bic"], f"{label}/bic")
    _assert_close(
        res.log_likelihood, ref["log_likelihood"], f"{label}/log_likelihood"
    )
    if "r_squared_within" in ref:
        _assert_close(
            res.r_squared_within,
            ref["r_squared_within"],
            f"{label}/r_squared_within",
        )


def _options(cov_type: str, *, two_way: bool) -> FEOptions:
    """`dk`は1-wayのとき時系列順序（`dk_time`）が別途必要。"""
    if two_way:
        return FEOptions(cov_type=cov_type, time="time")
    if cov_type == "dk":
        return FEOptions(cov_type="dk", dk_time="time")
    return FEOptions(cov_type=cov_type)


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


@pytest.mark.parametrize("scenario, cov_type", ONE_WAY_CASES)
def test_synthetic_one_way_matches_fixest(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    res = FE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=_options(cov_type, two_way=False),
    ).fit()

    _check_result(
        res,
        crosscheck[scenario]["one_way"][cov_type],
        f"{scenario}/one_way/{cov_type}",
    )


@pytest.mark.parametrize("scenario, cov_type", TWO_WAY_CASES)
def test_synthetic_two_way_matches_fixest(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    res = FE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=_options(cov_type, two_way=True),
    ).fit()

    _check_result(
        res,
        crosscheck[scenario]["two_way"][cov_type],
        f"{scenario}/two_way/{cov_type}",
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


# ── 境界値・クラスター不均衡 ────────────────────────────────────


def test_cluster_imbalanced_matches_fixest(crosscheck):
    """クラスター不均衡（サイズ[2,3,5,10,30,50]のタイル、entityとは無関係な
    専用クラスター列）のfixestクロスチェック。`fe_baseline_cluster_imbalanced.
    csv`（entity=20×period=10のn=200）を使う。クラスター列自体はCSVに含めず
    テスト側で都度動的生成する（`generate_fe_crosscheck_fixtures.py`と同じ方針）。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline_cluster_imbalanced.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = FEOptions(cov_type="cluster", cluster="cluster_group")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck["baseline"]["cluster_imbalanced"],
        "baseline/cluster_imbalanced",
    )


def test_cluster_g2_matches_fixest(crosscheck):
    """クラスタ数境界（G=2、q=1でG>q）の成功パスのfixestクロスチェック。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_k1.csv")
    groups = [str(i % 2) for i in range(df.height)]
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = FEOptions(cov_type="cluster", cluster="cluster_group")
    res = FE(df, y="y", x=["x1"], entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck["baseline"]["cluster_g2"],
        "baseline/cluster_g2",
    )


def test_boundary_df1_one_way_matches_fixest(crosscheck):
    """df_resid=1境界（1-way）の成功パスのfixestクロスチェック。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_df1_one_way.csv")
    options = FEOptions(cov_type="classical")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck["baseline_df1"]["one_way"],
        "baseline_df1/one_way",
    )


def test_boundary_df1_two_way_matches_fixest(crosscheck):
    """df_resid=1境界（2-way）の成功パスのfixestクロスチェック。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_df1_two_way.csv")
    options = FEOptions(cov_type="classical", time="time")
    res = FE(
        df, y="y", x=["x1", "x2", "x3"], entity="entity", options=options
    ).fit()

    _check_result(
        res,
        crosscheck["baseline_df1"]["two_way"],
        "baseline_df1/two_way",
    )


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
    )
