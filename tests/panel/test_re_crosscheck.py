"""RE の独立実装（R: plm）とのクロスチェックテスト。

主リファレンス（linearmodels）との厳密比較は`test_re_reference.py`で行う。
ここでは`tests/fixtures/benchmarks/re_crosscheck.json`（`benchmark/panel/
fixtures/generate_re_crosscheck_fixtures.py`で生成）を用いて、linearmodelsとは
独立した実装（R: plm）との一致を確認する。

## このファイルだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.RandomEffects`が提供しないため、plmが唯一の
  参照実装になる（`benchmark/panel/references/linearmodels_ref.py`
  モジュールdoc参照）。
- **ハウスマン検定**（`hausman_statistic`/`hausman_p_value`/`hausman_df`）:
  `linearmodels`に専用実装が無いため、`plm::phtest(method = "aux",
  effect = "individual")`（回帰ベース）が唯一の参照実装（panel-common.md
  5.3節）。比較は常に1-wayで`REOptions.time`の有無によらない
  （`generate_re_crosscheck_fixtures.py`モジュールdoc参照）。

## ハウスマン統計量の方式について

補助回帰版は統計量が構造的に非負になるため符号処理は不要。`plm`は補助回帰の
定数項に準偏差変換前の`1`を使い、本実装もこれに合わせている。バランスパネルでは
機械精度で一致し、不均衡パネルではSwamy-Arora分散成分の差（σ_u²）で数％ずれる
（`_UNBALANCED_HAUSMAN_SCENARIO`）。

## 許容誤差について

plmの変量効果分散成分推定（Swamy-Arora）がlinearmodelsと僅かに異なる実装の
ため、係数・標準誤差自体が不均衡パネルで最大1%程度乖離する（バランスパネル
では機械精度一致）。`_tolerances.py`の`re_crosscheck`参照。

役割分担:
    - 構造・API: `test_re_api.py`
    - `ValidationError`/`ComputationError` パス: `test_re_validation.py`
    - 主リファレンス（linearmodels）との数値照合: `test_re_reference.py`
    - 独立実装（R: plm）とのクロスチェック: このファイル
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
from econometricsmodels import RE, REOptions

from benchmark.common import WAGEPAN_ENTITY, WAGEPAN_X, WAGEPAN_Y
from benchmark.panel.fixtures.generate_fe_fixtures import NUMERIC_SCENARIOS
from benchmark.panel.fixtures.generate_re_crosscheck_fixtures import COV_TYPES

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_crosscheck.json"
)

RTOL = TOLERANCES["re_crosscheck"]["rtol"]
ATOL = TOLERANCES["re_crosscheck"]["atol"]
RTOL_HAUSMAN = TOLERANCES["re_crosscheck"]["rtol_hausman"]
ATOL_HAUSMAN = TOLERANCES["re_crosscheck"]["atol_hausman"]
RTOL_HAUSMAN_UNBALANCED = TOLERANCES["re_crosscheck"][
    "rtol_hausman_unbalanced"
]
ATOL_HAUSMAN_P_VALUE = TOLERANCES["re_crosscheck"]["atol_hausman_p_value"]

# plm/linearmodelsの分散成分（σ_u²）推定の差でθが変わり、ハウスマン統計量への
# 増幅がcoef/seよりさらに大きくなる唯一のシナリオ（モジュールdoc「許容誤差に
# ついて」参照）。
_UNBALANCED_HAUSMAN_SCENARIO = "unbalanced"


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)


def _check_result(
    res, ref: dict, label: str, *, scenario: str | None = None
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

    # ハウスマン検定はcov_typeに依存しない単一の統計量（`ref`のhc2/hc3どちらの
    # エントリにも同じ値が含まれる、`run_plm_benchmark.R`のモジュールコメント
    # 参照）。engine側も`abs()`適用後の値を返すため（モジュールdoc参照）、
    # `plm`の出力と直接比較できる。`df`はシナリオに関わらず常に一致するため
    # 無条件で比較する。
    rtol_hausman = (
        RTOL_HAUSMAN_UNBALANCED
        if scenario == _UNBALANCED_HAUSMAN_SCENARIO
        else RTOL_HAUSMAN
    )
    assert_close(
        res.hausman_statistic,
        ref["hausman_statistic"],
        f"{label}/hausman_statistic",
        rtol=rtol_hausman,
        atol=ATOL_HAUSMAN,
    )
    assert_close(
        res.hausman_p_value,
        ref["hausman_p_value"],
        f"{label}/hausman_p_value",
        rtol=rtol_hausman,
        atol=ATOL_HAUSMAN_P_VALUE,
    )
    assert res.hausman_df == ref["hausman_df"], f"{label}/hausman_df"


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", NUMERIC_SCENARIOS)
def test_synthetic_matches_plm(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    options = REOptions(cov_type=cov_type)
    res = RE(df, y="y", x=x_cols, entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck[scenario][cov_type],
        f"{scenario}/{cov_type}",
        scenario=scenario,
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_wagepan_matches_plm(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = REOptions(cov_type=cov_type)
    res = RE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(res, crosscheck["wagepan"][cov_type], f"wagepan/{cov_type}")
