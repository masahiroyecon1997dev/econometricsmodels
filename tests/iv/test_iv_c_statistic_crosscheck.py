"""IVのC統計量（`c_stat()`）の独立実装（R: momentfit）とのクロスチェックテスト。

主リファレンス（linearmodelsの`c_stat`と`IVGMM`の部品で組み直した参照実装）との比較は
`test_iv_c_statistic.py`で行う。ここでは`tests/fixtures/benchmarks/
iv_c_statistic_crosscheck.json`（`benchmark/iv/fixtures/
generate_iv_c_statistic_crosscheck_fixtures.py`で生成）を使い、Rのmomentfitの部品で
組み立てた値との一致を確認する。

Rにはこの検定を返すパッケージが無い。`benchmark/iv/references/run_momentfit_c_stat.R`は、
モーメント共分散・GMM推定・Hansen Jをmomentfitの関数で出し、`S`の部分行列を重みにして`J`の
差を取る手順だけを自前で組む。本実装とは別の部品による数値の確認であり、手順そのものの
正しさは`test_iv_c_statistic.py`のlinearmodelsとの照合が担保する。

重みは`estimator="gmm"`の`gmm_weight_type`（classical・robust・hac・cluster）。
HACのラグ数は`C_STAT_HAC_LAGS`、時間順は行順。フィクスチャ生成時と同じ入力データを、
`tests/fixtures/benchmarks/data/`に固定済みのCSVから読む（`benchmark/iv/freeze.py`参照）。
実データは`wooldridge`パッケージから読む。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close
from _constants import DATA_DIR
from _helpers import ROW_TIME, load_wooldridge_dataset, with_row_time
from _tolerances import TOLERANCES
from econometricsmodels import IV, IVOptions

from benchmark.common import balanced_cluster_groups, imbalanced_cluster_groups
from benchmark.iv.constants import (
    C_STAT_CARD_TESTS,
    C_STAT_HAC_LAGS,
    C_STAT_MULTI_ENDOG_TESTS,
    C_STAT_MULTI_ENDOG_TESTS_JUST_IDENTIFIED,
    C_STAT_SINGLE_ENDOG_TESTS,
    C_STAT_SINGLE_ENDOG_TESTS_JUST_IDENTIFIED,
    C_STAT_WEIGHTS,
    CARD_REGION_COLUMN,
    CARD_X_EXOG,
    CLUSTER_ILL_CONDITIONED_SCENARIOS,
)
from benchmark.iv.datasets import (
    MULTI_ENDOG_SCENARIOS,
    NUMERIC_SCENARIOS,
    multi_endog_instruments,
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "iv_c_statistic_crosscheck.json"
)

_CLUSTER = "cluster_group"
_TOL = TOLERANCES["iv_c_statistic_crosscheck"]
_close = partial(assert_close, rtol=_TOL["rtol"], atol=0.0)

_FIXTURES = json.loads(FIXTURE_PATH.read_text())

# フィクスチャの重みのエントリ名 → クラスター列を作る関数（クラスターを使わない重みは`None`）。
_GROUPS: dict[str, Callable[[int], list[int]] | None] = {
    "classical": None,
    "robust": None,
    "hac": None,
    "cluster": balanced_cluster_groups,
    "cluster_imbalanced": imbalanced_cluster_groups,
}


def _weight_of(entry: str) -> str:
    return "cluster" if entry.startswith("cluster") else entry


def _fit(
    df: pl.DataFrame,
    entry: str,
    *,
    y: str,
    x_exog: list[str],
    x_endog: list[str],
    instruments: list[str],
    include_intercept: bool = True,
    cluster_column: str = _CLUSTER,
):
    """`entry`の重み（`gmm_weight_type`）で2段階GMMを推定する。"""
    weight = _weight_of(entry)
    groups = _GROUPS.get(entry)
    if groups is not None:
        df = df.with_columns(pl.Series(_CLUSTER, groups(df.height)))
    options: dict = {}
    if weight == "cluster":
        options["cluster"] = cluster_column
    if weight == "hac":
        options["hac_lags"] = C_STAT_HAC_LAGS
        options["hac_time"] = ROW_TIME
    return IV(
        with_row_time(df),
        y=y,
        x_exog=x_exog,
        x_endog=x_endog,
        instruments=instruments,
        options=IVOptions(
            estimator="gmm",
            gmm_weight_type=weight,
            include_intercept=include_intercept,
            **options,
        ),
    ).fit()


def _check(
    result,
    reference: dict,
    tests: dict[str, tuple[list[str], list[str]]],
    label: str,
) -> None:
    """`tests`の各検定について、`c_stat()`をRの値と比べる。"""
    assert set(reference) == set(tests), label
    for name, (variables, instruments) in tests.items():
        got = result.c_stat(
            variables=variables or None, instruments=instruments or None
        )
        want = reference[name]
        where = f"{label}/{name}"
        assert got.df == want["df"], where
        _close(got.statistic, want["statistic"], f"{where}/statistic")
        _close(got.p_value, want["p_value"], f"{where}/p_value")
        _close(got.j_with_tested, want["j_with_tested"], f"{where}/J with")
        # 検定しない側が丁度識別のとき、J_cは本実装でもRでも厳密に0。
        _close(
            got.j_without_tested,
            want["j_without_tested"],
            f"{where}/J without",
        )


def _single_endog_tests(scenario: str) -> dict:
    if scenario == "just_identified":
        return C_STAT_SINGLE_ENDOG_TESTS_JUST_IDENTIFIED
    return C_STAT_SINGLE_ENDOG_TESTS


def _multi_endog_tests(scenario: str) -> dict:
    if scenario == "just_identified":
        return C_STAT_MULTI_ENDOG_TESTS_JUST_IDENTIFIED
    return C_STAT_MULTI_ENDOG_TESTS


def _cases(section: dict) -> list:
    return [
        pytest.param(scenario, entry, id=f"{scenario}/{entry}")
        for scenario, entries in section.items()
        for entry in entries
    ]


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────────


def test_fixture_covers_every_scenario_weight_and_test() -> None:
    """フィクスチャからシナリオ・重み・検定が欠けても、パラメータ化されたテストが
    黙って減るだけにならないようにする。"""
    synthetic = _FIXTURES["synthetic"]
    scenarios = {
        k: v
        for k, v in synthetic.items()
        if k not in ("multi_endog", "no_intercept")
    }
    assert list(scenarios) == NUMERIC_SCENARIOS
    for scenario, entries in scenarios.items():
        assert set(entries) >= set(C_STAT_WEIGHTS), scenario
        assert set(entries) <= set(_GROUPS), scenario
        for reference in entries.values():
            assert set(reference) == set(_single_endog_tests(scenario))
    assert set(scenarios["baseline"]) == set(_GROUPS)
    for scenario in CLUSTER_ILL_CONDITIONED_SCENARIOS:
        assert "cluster" in scenarios[scenario]
    assert list(synthetic["multi_endog"]) == MULTI_ENDOG_SCENARIOS
    assert "cluster" in synthetic["multi_endog"]["baseline"]
    assert set(synthetic["no_intercept"]["baseline"]) == set(C_STAT_WEIGHTS)


@pytest.mark.parametrize(
    ("scenario", "entry"),
    _cases(
        {
            scenario: entries
            for scenario, entries in _FIXTURES["synthetic"].items()
            if scenario not in ("multi_endog", "no_intercept")
        }
    ),
)
def test_matches_momentfit(scenario: str, entry: str) -> None:
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    result = _fit(
        df,
        entry,
        y="y",
        x_exog=scenario_x_exog(scenario),
        x_endog=["endog1"],
        instruments=scenario_instruments(scenario),
    )
    _check(
        result,
        _FIXTURES["synthetic"][scenario][entry],
        _single_endog_tests(scenario),
        f"{scenario}/{entry}",
    )


@pytest.mark.parametrize(
    ("scenario", "entry"), _cases(_FIXTURES["synthetic"]["multi_endog"])
)
def test_multi_endog_matches_momentfit(scenario: str, entry: str) -> None:
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}_multi_endog.csv")
    result = _fit(
        df,
        entry,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=multi_endog_instruments(scenario),
    )
    _check(
        result,
        _FIXTURES["synthetic"]["multi_endog"][scenario][entry],
        _multi_endog_tests(scenario),
        f"multi_endog/{scenario}/{entry}",
    )


# ── 凍結フィクスチャとの数値照合（オプション） ───────────────────────────


@pytest.mark.parametrize("entry", C_STAT_WEIGHTS)
def test_no_intercept_matches_momentfit(entry: str) -> None:
    """切片なし（式の両辺に`0 +`）。classicalの`σ̂²`は残差を中心化せずに作る。"""
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    result = _fit(
        df,
        entry,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        include_intercept=False,
    )
    _check(
        result,
        _FIXTURES["synthetic"]["no_intercept"]["baseline"][entry],
        C_STAT_SINGLE_ENDOG_TESTS,
        f"no_intercept/baseline/{entry}",
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge card） ──────────────


@pytest.mark.parametrize("entry", [*C_STAT_WEIGHTS, "cluster"])
def test_card_matches_momentfit(entry: str) -> None:
    """教育年数`educ`を`nearc2`・`nearc4`で操作する。クラスターは居住地域の9グループ
    （検定する前のモーメント条件の数と同数）。"""
    df = with_card_region(load_wooldridge_dataset("card"))
    result = _fit(
        df,
        entry,
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        cluster_column=CARD_REGION_COLUMN,
    )
    _check(
        result,
        _FIXTURES["wooldridge"]["card"][entry],
        C_STAT_CARD_TESTS,
        f"card/{entry}",
    )
