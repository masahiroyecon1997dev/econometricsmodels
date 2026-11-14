"""IVの弱操作変数診断の独立実装（R: ivDiag）とのクロスチェックテスト。

主リファレンス（linearmodelsとMOPオラクル）との比較は`test_iv_weak_diagnostics.py`で行う。
ここでは`tests/fixtures/benchmarks/iv_weak_diagnostics_crosscheck.json`
（`benchmark/iv/fixtures/generate_iv_weak_diagnostics_crosscheck_fixtures.py`で生成）を使い、
linearmodelsとは独立した実装（R `ivDiag`）との一致を確認する。

ivDiagが返すのは内生変数が1個のときの次の統計量で、標準誤差はrobustとclusterに限る。

- `F.standard`: Cragg-Donaldの`F`統計量（`cov_type`に依存しない）。
- `F.robust`・`F.cluster`: Kleibergen-Paapのrk Wald `F`（本実装の`hc1`・`cluster`）。
- `eff_F`: 有効F（`hc1`・`cluster`）。

rk LM統計量・Stock-Yogoの臨界値・複数の内生変数・有効Fの臨界値はivDiagが返さないため、
このファイルでは比べない（それらの参照は`test_iv_weak_diagnostics.py`）。

フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`に固定済みのCSV
から読む（`benchmark/iv/freeze.py`参照）。実データは`wooldridge`パッケージから読む。
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
from _helpers import hac_time_for, load_wooldridge_dataset, with_row_time
from _tolerances import TOLERANCES
from econometricsmodels import IV, IVOptions

from benchmark.common import balanced_cluster_groups, imbalanced_cluster_groups
from benchmark.iv.constants import CARD_REGION_COLUMN, CARD_X_EXOG
from benchmark.iv.datasets import (
    NUMERIC_SCENARIOS,
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "iv_weak_diagnostics_crosscheck.json"
)

_CLUSTER = "cluster_group"
_RTOL = TOLERANCES["iv_weak_diagnostics_crosscheck"]["rtol"]
_close = partial(assert_close, rtol=_RTOL, atol=0.0)

_FIXTURES = json.loads(FIXTURE_PATH.read_text())

# フィクスチャのエントリ名 → (`cov_type`, クラスター列を作る関数)。
_GROUPS: dict[str, Callable[[int], list[int]] | None] = {
    "hc1": None,
    "cluster": balanced_cluster_groups,
    "cluster_imbalanced": imbalanced_cluster_groups,
    # クラスター数が除外操作変数の数+1（baselineは操作変数2本）の境界。
    "cluster_g3": partial(balanced_cluster_groups, n_groups=3),
}


def _cov_options(entry: str) -> dict:
    if _GROUPS[entry] is None:
        return {"cov_type": "hc1"}
    return {"cov_type": "cluster", "cluster": _CLUSTER}


def _synthetic_cases() -> list:
    return [
        pytest.param(scenario, entry, id=f"{scenario}/{entry}")
        for scenario, entries in _FIXTURES["synthetic"].items()
        for entry in entries
    ]


def _check(
    df: pl.DataFrame,
    reference: dict,
    label: str,
    *,
    y: str,
    x_exog: list[str],
    x_endog: str,
    instruments: list[str],
    entry: str,
    estimator: str = "2sls",
) -> None:
    groups = _GROUPS[entry]
    if groups is not None:
        df = df.with_columns(pl.Series(_CLUSTER, groups(df.height)))
    result = IV(
        df,
        y=y,
        x_exog=x_exog,
        x_endog=[x_endog],
        instruments=instruments,
        options=IVOptions(estimator=estimator, **_cov_options(entry)),
    ).fit()

    _close(
        result.cragg_donald().statistic,
        reference["cragg_donald_f"],
        f"{label}/cragg_donald.statistic",
    )
    _close(
        result.kleibergen_paap().rk_wald_f,
        reference["rk_wald_f"],
        f"{label}/kleibergen_paap.rk_wald_f",
    )
    _close(
        result.effective_f().statistic,
        reference["effective_f"],
        f"{label}/effective_f.statistic",
    )


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────────


def test_fixture_covers_every_scenario_and_entry() -> None:
    """フィクスチャからシナリオやエントリが欠けても、パラメータ化されたテストが
    黙って減るだけにならないようにする。"""
    synthetic = _FIXTURES["synthetic"]
    assert list(synthetic) == NUMERIC_SCENARIOS
    for scenario, entries in synthetic.items():
        assert "hc1" in entries, scenario
        assert set(entries) <= set(_GROUPS), scenario
    assert set(synthetic["baseline"]) == set(_GROUPS)


@pytest.mark.parametrize(("scenario", "entry"), _synthetic_cases())
def test_matches_ivdiag(scenario: str, entry: str) -> None:
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    _check(
        df,
        _FIXTURES["synthetic"][scenario][entry],
        f"{scenario}/{entry}",
        y="y",
        x_exog=scenario_x_exog(scenario),
        x_endog="endog1",
        instruments=scenario_instruments(scenario),
        entry=entry,
    )


# ── 凍結フィクスチャとの数値照合（オプション） ───────────────────────────


@pytest.mark.parametrize("entry", ["hc1", "cluster"])
def test_gmm_matches_ivdiag(entry: str) -> None:
    """診断は推定量（2SLS/GMM）に依存しない: GMMでも同じ値になる。"""
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    _check(
        df,
        _FIXTURES["synthetic"]["baseline"][entry],
        f"baseline/{entry}/gmm",
        y="y",
        x_exog=scenario_x_exog("baseline"),
        x_endog="endog1",
        instruments=scenario_instruments("baseline"),
        entry=entry,
        estimator="gmm",
    )


@pytest.mark.parametrize("cov_type", ["classical", "hc0", "hc2", "hc3", "hac"])
def test_cragg_donald_matches_ivdiag_under_every_cov_type(
    cov_type: str,
) -> None:
    """Cragg-Donaldは等分散の統計量で、`fit()`の`cov_type`に依存しない。"""
    df = with_row_time(pl.read_csv(DATA_DIR / "iv_baseline.csv"))
    result = IV(
        df,
        y="y",
        x_exog=scenario_x_exog("baseline"),
        x_endog=["endog1"],
        instruments=scenario_instruments("baseline"),
        options=IVOptions(cov_type=cov_type, **hac_time_for(cov_type)),
    ).fit()
    _close(
        result.cragg_donald().statistic,
        _FIXTURES["synthetic"]["baseline"]["hc1"]["cragg_donald_f"],
        f"baseline/{cov_type}/cragg_donald.statistic",
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge card） ──────────────


@pytest.mark.parametrize("entry", ["hc1", "cluster"])
def test_card_matches_ivdiag(entry: str) -> None:
    """教育年数`educ`を`nearc2`・`nearc4`で操作する。クラスターは居住地域の9グループ。"""
    df = with_card_region(load_wooldridge_dataset("card"))
    reference = _FIXTURES["wooldridge"]["card"][entry]
    options = (
        {"cov_type": "hc1"}
        if entry == "hc1"
        else {"cov_type": "cluster", "cluster": CARD_REGION_COLUMN}
    )
    result = IV(
        df,
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        options=IVOptions(**options),
    ).fit()

    _close(
        result.cragg_donald().statistic,
        reference["cragg_donald_f"],
        f"card/{entry}/cragg_donald.statistic",
    )
    _close(
        result.kleibergen_paap().rk_wald_f,
        reference["rk_wald_f"],
        f"card/{entry}/kleibergen_paap.rk_wald_f",
    )
    _close(
        result.effective_f().statistic,
        reference["effective_f"],
        f"card/{entry}/effective_f.statistic",
    )
