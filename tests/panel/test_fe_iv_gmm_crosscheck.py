"""FE-GMM（`FEIV`）の独立クロスチェック: within変換後のデータへのR `momentfit`。

`tests/fixtures/benchmarks/fe_iv_gmm_crosscheck.json`と`fe_iv_gmm_cov_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_iv_gmm_crosscheck_fixtures.py`で生成）を読み、
`benchmark/panel/fe_iv_gmm_cases.py`の`crosscheck_cases()`の全ケースで比べる。主リファレンス
（linearmodels `IVGMM`）とは別の実装の部品で、係数・重み行列・共分散行列の本体・Hansen Jを作る。

## ケース

1-way（`baseline`）・不均衡・2-way・不均衡な2-way・3-way × 推定方式と重み（one_step・two_step の
classical・robust・cluster・iterated 3回）× classical・hc1・個体クラスター。

## 比べる量

係数・標準誤差・z・p値・信頼区間・係数の分散共分散行列・Wald（χ²）・Hansen J（`classical`重みと
`one_step`は`(n − n_absorbed)/n`倍した尺度）。標準誤差の小標本補正のスカラーは手書き
（`benchmark/panel/references/within_gmm_r.py`のモジュールdoc）で、momentfitが独立に検証するのは
係数・重み行列・共分散行列の本体とHansen J。
"""

from __future__ import annotations

from functools import partial

import pytest
from _assertions import (
    assert_close,
    assert_cov_params_close,
    assert_dict_close,
    assert_inference_close,
)
from _fe_iv_gmm_helpers import FIXTURE_DIR, fit_case, leaf
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES

from benchmark.panel.fe_iv_gmm_cases import crosscheck_cases

_TOL = TOLERANCES["fe_iv_gmm_crosscheck"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_p_close = partial(
    assert_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)
_p_dict_close = partial(
    assert_dict_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)

CASES = crosscheck_cases()


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_DIR / "fe_iv_gmm_crosscheck.json")


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_matches_momentfit(fixtures, case) -> None:
    res = fit_case(case)
    ref = leaf(fixtures, case)
    label = case.label

    assert list(res.params) == list(ref["params"]), label
    assert_inference_close(
        res,
        ref,
        label,
        rtol=RTOL,
        atol=ATOL,
        p_value_check=_p_dict_close,
    )
    assert_cov_params_close(
        res.cov_params, ref["cov_params"], label, rtol=RTOL
    )
    assert res.wald_df_num == ref["wald_df_num"], label
    _close(res.wald_statistic, ref["wald_statistic"], f"{label}/wald")
    _p_close(res.wald_p_value, ref["wald_p_value"], f"{label}/wald_p")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_hansen_j_matches_momentfit(fixtures, case) -> None:
    """Hansen J。丁度識別は`None`（momentfitのスクリプトも出さない）。"""
    res = fit_case(case)
    ref = leaf(fixtures, case)
    label = case.label

    if ref["overid_statistic"] is None:
        assert res.overid_statistic is None, label
        assert res.overid_df is None, label
        return
    assert res.overid_df == ref["overid_df"], label
    _close(res.overid_statistic, ref["overid_statistic"], f"{label}/hansen_j")
    _p_close(res.overid_p_value, ref["overid_p_value"], f"{label}/hansen_j_p")
