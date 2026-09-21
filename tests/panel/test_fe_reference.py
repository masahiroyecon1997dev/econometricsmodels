"""FE の主リファレンス（linearmodels）との数値照合。

`tests/fixtures/benchmarks/fe.json`（`benchmark/panel/fixtures/
generate_fe_fixtures.py`で生成）を読み込み、合成データ6シナリオ×classical/
hc1/cluster/hac × 1-way/2-way（`unbalanced`のみ1-way限定）で、係数・標準
誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する。Wooldridge実
データ（wagepan）も同じフィクスチャ経由で検証する。

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

from benchmark.common import WAGEPAN_ENTITY, WAGEPAN_TIME, WAGEPAN_X, WAGEPAN_Y
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
    _assert_dict_close(res.t_stats, ref["t_stats"], f"{label}/t_stats")
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


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_matches_linearmodels_one_way(fixtures, scenario, cov_type):
    """1-way HACは`time_col`（DK専用の時系列順序、`time`＝2-way構造とは独立の
    フィールド）が必要。フィクスチャ生成側（`linearmodels_ref.py`）は`time`
    列が無い場合エンティティ内の観測順（`groupby(entity).cumcount()`）を
    代用しているが、`benchmark/panel/datasets.py`が生成する行順はエンティ
    ティ内で時系列順そのものなので、`time_col="time"`を明示するのと数学的
    に同じ結果になる（実測確認済み）。
    """
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    kwargs = {"time_col": "time"} if cov_type == "hac" else {}
    options = FEOptions(cov_type=cov_type, **kwargs)
    res = FE(df, y="y", x=x_cols, entity="entity", options=options).fit()

    _check_result(
        res,
        fixtures[scenario]["one_way"][cov_type],
        f"{scenario}/one_way/{cov_type}",
        check_r_squared_within=True,
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", TWO_WAY_SCENARIOS)
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
