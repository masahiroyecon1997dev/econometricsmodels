"""RE の主リファレンス（linearmodels）との数値照合。

`tests/fixtures/benchmarks/re.json`（`benchmark/panel/fixtures/
generate_re_fixtures.py`で生成）を読み込み、合成データ6シナリオ×classical/
hc1/cluster/hac で、係数・標準誤差・検定統計量・適合度統計量を相対誤差1e-8で
厳密比較する。Wooldridge実データ（wagepan）も同じフィクスチャ経由で検証する。

役割分担:
    - 構造・API: `test_re_api.py`
    - `ValidationError`/`ComputationError` パス: `test_re_validation.py`
    - 主リファレンス（linearmodels）との数値照合: このファイル
    - 独立実装（R: plm）とのクロスチェック（hc2/hc3・ハウスマン検定）:
      `test_re_crosscheck.py`

Note:
    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`に
    固定済みのCSV（`fe_{scenario}.csv`、FEと共用）から読む——RE専用の合成
    データセットは無い（`benchmark/panel/references/linearmodels_ref.py`
    モジュールdoc「RE（`run_re()`）固有の相違点」参照、ユーザー確認済み）。
    hc2/hc3・aic/bic・log_likelihood・ハウスマン検定は`linearmodels.
    RandomEffects`が提供しないためこのファイルの対象外
    （`test_re_crosscheck.py`のplmのみで検証する単一参照実装の例外）。
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
from econometricsmodels import RE, ReOptions

from benchmark.common import WAGEPAN_ENTITY, WAGEPAN_X, WAGEPAN_Y
from benchmark.panel.fixtures.generate_re_fixtures import (
    COV_TYPES,
    NUMERIC_SCENARIOS,
    WAGEPAN_COV_TYPES,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks" / "re.json"
)

RTOL = TOLERANCES["re_reference"]["rtol"]
ATOL = TOLERANCES["re_reference"]["atol"]


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)


def _check_result(res, ref: dict, label: str) -> None:
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
@pytest.mark.parametrize("scenario", NUMERIC_SCENARIOS)
def test_matches_linearmodels(fixtures, scenario, cov_type):
    """`hac`は`time`（内部FE呼び出しの1-way/2-way選択とは無関係、`ReOptions`
    には`time_col`が独立に無い。`FeOptions`と違い、REの`ReOptions.time`は
    HAC時系列順序と内部FE1-way/2-way選択を兼ねる1フィールドのため、`hac`
    ケースでも常に`time="time"`を渡す。本フィクスチャの数値比較は`ReOptions.
    time`の値に依存しない（係数・標準誤差はtimeを使わないため、
    `_re_helpers`・`engine/src/panel/CLAUDE.md`参照）。
    """
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    kwargs = {"time": "time"} if cov_type == "hac" else {}
    options = ReOptions(cov_type=cov_type, **kwargs)
    res = RE(df, y="y", x=x_cols, entity="entity", options=options).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_matches_linearmodels(fixtures, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = ReOptions(cov_type=cov_type)
    res = RE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(res, fixtures["wagepan"][cov_type], f"wagepan/{cov_type}")
