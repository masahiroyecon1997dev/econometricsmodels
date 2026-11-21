"""FE-2SLS（`FEIV`）の弱操作変数診断の、R ivDiagとのクロスチェック。

`tests/fixtures/benchmarks/fe_iv_weak_diagnostics_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_iv_weak_diagnostics_crosscheck_fixtures.py`で生成）を読み、
`benchmark/panel/fe_iv_cases.py`の`ivdiag_cases()`の全ケースで、`cragg_donald()`・
`kleibergen_paap()`・`effective_f()`を、ダミー変数を明示した2SLS（`test_fe_iv_dummy_crosscheck.py`）
とは独立な実装（ivDiag。固定効果は`FE`引数で吸収）と比べる。

## 検証範囲

ivDiagが返すのは内生変数が1個のときの次の統計量だけ。

- Cragg-DonaldのF（`F.standard`）: 全ケース。`cov_type`によらない。
- Kleibergen-Paapのrk Wald F: `hc1`は`F.robust`、クラスターは`F.cluster`。
- 有効F（`eff_F`）: `hc1`はrobust、クラスターはcluster。

1-wayの個体FEは`classical`・`hc1`・個体クラスター・時点クラスター・region・不均衡なregion・sector、
2-wayは`classical`・`hc1`・`sector`クラスター。2-wayで個体・時点・regionのクラスターを含めないのは、
ivDiag（lfe）がクラスター頑健共分散の自由度`K`を、固定効果のどれかがクラスターにネストすると、ネストしない
固定効果も含めて外すため（fixestと本実装はネストした固定効果だけを外す）。そのF・有効Fは
`(N - K_lfe) / (N - K)`倍ずれ、本実装ともfixestの第一段階のWald F（`test_fe_iv_reference.py`）とも合わない。
`sector`はどの個体・時点にも複数の水準が現れるのでネストせず、一致する。rk LM・Stock-Yogoの臨界値・内生変数が複数のシナリオ・3-way・DK・
HC2・HC3は比べられず、ダミー変数を明示した2SLSとMOPの独立実装だけが参照になる。
"""

from __future__ import annotations

import json
from functools import partial

import pytest
from _assertions import assert_close
from _fe_iv_helpers import FIXTURE_DIR, fit_case, leaf
from _tolerances import TOLERANCES

from benchmark.panel.fe_iv_cases import ivdiag_cases

_TOL = TOLERANCES["fe_iv_weak_diagnostics_crosscheck"]
_close = partial(assert_close, rtol=_TOL["rtol"], atol=_TOL["atol"])

CASES = ivdiag_cases()


@pytest.fixture(scope="module")
def fixtures() -> dict:
    path = FIXTURE_DIR / "fe_iv_weak_diagnostics_crosscheck.json"
    return json.loads(path.read_text())


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_weak_instrument_diagnostics_match_ivdiag(fixtures, case) -> None:
    res = fit_case(case)
    ref = leaf(fixtures, case)
    label = case.label

    _close(
        res.cragg_donald().statistic,
        ref["cragg_donald_f"],
        f"{label}/cragg_donald",
    )
    if case.cov.cov_type == "classical":
        # ivDiagは等分散の有効F・rk Wald Fを返さない。
        assert set(ref) == {"cragg_donald_f"}
        return
    _close(
        res.kleibergen_paap().rk_wald_f, ref["rk_wald_f"], f"{label}/rk_wald_f"
    )
    _close(res.effective_f().statistic, ref["effective_f"], f"{label}/eff_f")
