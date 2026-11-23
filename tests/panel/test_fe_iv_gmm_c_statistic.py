"""FE-GMM（`FEIVResults.c_stat()`）の参照との数値照合。

`tests/fixtures/benchmarks/fe_iv_gmm_c_statistic.json`（within変換後のデータへのnumpyのオラクル）と
`fe_iv_gmm_c_statistic_crosscheck.json`（同じデータへのR `momentfit`の部品。Rにこの検定を返す
パッケージは無いので、momentfitのモーメント共分散・GMM推定・Hansen Jを使い、Hayashiの手順を
スクリプトが組み立てる）を読み、`benchmark/panel/fe_iv_gmm_cases.py`の`c_stat_cases()`の全ケースで比べる。
オラクルは、IV本体の`c_stat()`の参照（linearmodelsの`c_stat`と照合済み）と同じ実装。

## ケース

1-way・多内生変数・不均衡・2-way・不均衡な2-way・3-way × 重み（classical・robust・cluster）×
検定（内生変数だけ・除外操作変数だけ・両方）。

## 比べる量

統計量・p値・自由度・`J_e`（検定するモーメント条件を使う推定のHansen J）・`J_c`（使わない推定の
Hansen J）。`classical`重みは`S`を`n/(n − n_absorbed)`倍した尺度（参照の値に`(n − n_absorbed)/n`を
掛けたもの）。
"""

from __future__ import annotations

import json
from functools import partial

import pytest
from _assertions import assert_close
from _fe_iv_gmm_helpers import FIXTURE_DIR, fit_c_stat_case, leaf
from _tolerances import TOLERANCES

from benchmark.panel.fe_iv_gmm_cases import c_stat_cases

_TOL = TOLERANCES["fe_iv_gmm_c_statistic"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_p_close = partial(
    assert_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)

CASES = c_stat_cases()
REFERENCES = {
    "numpy_oracle": "fe_iv_gmm_c_statistic.json",
    "momentfit": "fe_iv_gmm_c_statistic_crosscheck.json",
}


@pytest.fixture(scope="module")
def fixtures() -> dict[str, dict]:
    return {
        name: json.loads((FIXTURE_DIR / file).read_text())
        for name, file in REFERENCES.items()
    }


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_c_stat_matches_the_reference(fixtures, case, reference) -> None:
    res = fit_c_stat_case(case)
    ref = leaf(fixtures[reference], case)
    label = f"{case.label}/{reference}"

    assert res.df == ref["df"], label
    assert res.distribution == "chi2"
    assert res.df_denom is None
    assert res.variables == list(case.variables), label
    assert res.instruments == list(case.instruments), label
    _close(res.statistic, ref["statistic"], f"{label}/statistic")
    _p_close(res.p_value, ref["p_value"], f"{label}/p_value")
    _close(res.j_with_tested, ref["j_with_tested"], f"{label}/j_with_tested")
    _close(
        res.j_without_tested,
        ref["j_without_tested"],
        f"{label}/j_without_tested",
    )
