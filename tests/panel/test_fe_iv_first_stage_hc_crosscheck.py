"""FE-2SLS（`FEIV`）のHC2・HC3のKleibergen-Paap rk Wald Fの、R fixestの第一段階との数値照合（独立なクロスチェック）。

`tests/fixtures/benchmarks/fe_iv_first_stage_hc_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_iv_first_stage_hc_crosscheck_fixtures.py`で生成）を読み、
`benchmark/panel/fe_iv_cases.py`の`first_stage_hc_cases()`の全ケースで
`kleibergen_paap().rk_wald_f`を比べる。

内生変数が1個のとき、rk Wald Fは同じ共分散での第一段階の頑健Wald Fに等しい。fixestはIVのHC2・HC3を
提供しない（主リファレンスの`test_fe_iv_reference.py`には含められない）が、第一段階を通常の`feols`として
回せば提供する。参照は、その第一段階の`wald(vcov = "hc2"|"hc3")`の統計量
（`benchmark/panel/references/run_fixest_first_stage_benchmark.R`）。

内生変数が複数のシナリオは、rk Wald Fが第一段階のWald Fに等しくないので比べない。rk LMも比べない。
"""

from __future__ import annotations

import json
from functools import partial

import pytest
from _assertions import assert_close
from _fe_iv_helpers import FIXTURE_DIR, fit_case, leaf
from _tolerances import TOLERANCES

from benchmark.panel.fe_iv_cases import first_stage_hc_cases

_TOL = TOLERANCES["fe_iv_first_stage_hc_crosscheck"]
_close = partial(assert_close, rtol=_TOL["rtol"], atol=_TOL["atol"])

CASES = first_stage_hc_cases()


@pytest.fixture(scope="module")
def fixtures() -> dict:
    path = FIXTURE_DIR / "fe_iv_first_stage_hc_crosscheck.json"
    return json.loads(path.read_text())


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_rk_wald_f_matches_fixest_first_stage(fixtures, case) -> None:
    ref = leaf(fixtures, case)
    assert set(ref) == {"rk_wald_f"}, case.label
    _close(
        fit_case(case).kleibergen_paap().rk_wald_f,
        ref["rk_wald_f"],
        f"{case.label}/rk_wald_f",
    )


def test_fixture_leaves_match_the_cases(fixtures) -> None:
    """フィクスチャの葉の集合が`CASES`と一致する（古い葉が残っても気づける）。"""

    def paths(node: dict, prefix: tuple[str, ...] = ()) -> set[str]:
        if "rk_wald_f" in node:
            return {"/".join(prefix)}
        return {
            path
            for key, child in node.items()
            for path in paths(child, (*prefix, key))
        }

    assert paths(fixtures["synthetic"]) == {case.label for case in CASES}
    assert not fixtures["wooldridge"]


def test_cases_cover_hc2_and_hc3_of_every_effect_structure() -> None:
    """HC2・HC3の両方を、1-way・2-way・3-way、不均衡なパネル、`x_exog`が空、外れ値（高レバレッジ）、
    自由度の境界で比べている（`dummy_cases()`が変わっても守る）。"""
    assert {case.cov.cov_type for case in CASES} == {"hc2", "hc3"}
    assert {case.n_effects for case in CASES} == {1, 2, 3}
    assert {case.scenario for case in CASES} >= {
        "unbalanced",
        "three_way",
        "outlier_regressor",
        "df1_boundary",
    }
    assert {case.path[0] for case in CASES} >= {"x_exog_empty", "two_way"}
    assert all(len(case.x_endog) == 1 for case in CASES)
