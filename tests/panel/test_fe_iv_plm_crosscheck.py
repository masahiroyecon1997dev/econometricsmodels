"""FE-2SLS（`FEIV`）の第2リファレンス（R: plm＋sandwich）とのクロスチェックテスト。

`tests/fixtures/benchmarks/fe_iv_plm_crosscheck.json`と`fe_iv_plm_cov_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_iv_plm_crosscheck_fixtures.py`で生成）を読み、
`benchmark/panel/fe_iv_cases.py`の`plm_cases()`の全ケースで、fixestとは別実装の共分散行列
（plmの`model = "within"`の操作変数推定）と一致することを確認する。

## 検証範囲

1-wayの個体FEの`classical`・`hc1`・`cluster`（個体）・`dk`（本実装の既定バンド幅と明示バンド幅1）。
比べる量は、係数・標準誤差・`t`値・`p`値・信頼区間・係数の分散共分散行列・自由度。plmは2-wayのwithinに
対するクラスター・SCC共分散行列を持たず、クラスター列も個体・時点しか指定できないため、2-wayと
他のクラスター列はfixest（`test_fe_iv_reference.py`）だけが参照になる。HC2・HC3は、plmのレバレッジが
固定効果の分を含まず本実装と定義が違うため比べない（`test_fe_iv_dummy_crosscheck.py`のnumpyが参照）。

## 独立性の限界

plmが独立に検証するのはサンドイッチの中身（構造残差と`X̂`）とカーネル・バンド幅の規約で、小標本補正の
係数（`n/(n-k-G)`・`G/(G-1)·(n-1)/(n-k-1)`・`T/(T-1)·(n-1)/(n-k-G)`）とt検定の自由度は、fixestと本実装の
規約をR側でスカラー倍の手計算で掛けたもの（`benchmark/panel/references/run_plm_fe_iv_benchmark.R`）。
補正係数そのものは、fixestだけが独立に検証する。
"""

from __future__ import annotations

from functools import partial

import pytest
from _assertions import assert_dict_close, assert_inference_close
from _fe_iv_helpers import FIXTURE_DIR, fit_case, leaf
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES

from benchmark.panel.fe_iv_cases import plm_cases

_TOL = TOLERANCES["fe_iv_plm_crosscheck"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

_p_dict_close = partial(
    assert_dict_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)

CASES = plm_cases()


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_DIR / "fe_iv_plm_crosscheck.json")


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_matches_plm(fixtures, case) -> None:
    res = fit_case(case)
    ref = leaf(fixtures, case)

    assert_inference_close(
        res,
        ref,
        case.label,
        rtol=RTOL,
        atol=ATOL,
        p_value_check=_p_dict_close,
    )
    assert res.df_resid == ref["df_resid"], case.label
    assert res.stat_df == ref["stat_df"], case.label
