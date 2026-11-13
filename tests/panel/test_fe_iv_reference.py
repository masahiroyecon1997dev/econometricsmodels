"""FE-2SLS（`FEIV`）の主リファレンス（R fixest）との数値照合。

`tests/fixtures/benchmarks/fe_iv.json`と`fe_iv_cov.json`
（`benchmark/panel/fixtures/generate_fe_iv_fixtures.py`で生成）を読み、`benchmark/panel/
fe_iv_cases.py`の`fixest_cases()`の全ケース（1-way・2-way・3-way、`x_exog`が空、信頼水準0.90 ×
classical・hc1・cluster（個体・時点・region・不均衡なregion・3つ目の効果）・dk）で、次を
相対誤差1e-8で比べる。

- 係数・標準誤差・`t`値・`p`値・信頼区間・係数の分散共分散行列
- `df_model`・`df_resid`・検定の自由度（`stat_df`は`cov_type`で決まり、clusterは`G-1`、dkは`T-1`）
- 全係数が同時にゼロというWald F（`wald()`）
- 内生変数ごとの部分F（`cov_type`によらず等分散）
- Kleibergen-Paapのrk Wald F（内生変数が1個のとき。同じ`vcov`での第一段階の頑健Wald Fに等しい）
- Wu-Hausman（第一段階の残差を足した補助回帰を、同じ`cov_type`で回したWald F）
- Sargan（fixestの`e'e/n`の尺度を`(n − 吸収した効果数)/n`倍した本実装の尺度）

fixestが提供しないHC2・HC3、定義が違うCragg-Donald・`kpr`・within R²、内生変数が複数のときのrk Wald F、
rk LM・有効Fは比べない
（定義と理由は`benchmark/panel/references/run_fixest_iv_benchmark.R`のヘッダ）。HC2・HC3と
診断の独立した参照は`test_fe_iv_dummy_crosscheck.py`。

役割分担:
    - 構造・API: `test_fe_iv_api.py`、`_lib`の配線・エラー・オラクル: `test_fe_iv_binding.py`
    - 主リファレンス（R: fixest）: このファイル
    - 独立実装（ダミー変数を明示した2SLS）: `test_fe_iv_dummy_crosscheck.py`
    - 独立実装（R: plm）: `test_fe_iv_plm_crosscheck.py`
    - 独立実装（R: ivDiag、弱操作変数診断）: `test_fe_iv_weak_diagnostics_crosscheck.py`
"""

from __future__ import annotations

from functools import partial

import pytest
from _assertions import (
    assert_close,
    assert_dict_close,
    assert_inference_close,
)
from _fe_iv_helpers import FIXTURE_DIR, fit_case, leaf
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES

from benchmark.panel.fe_iv_cases import fixest_cases

_TOL = TOLERANCES["fe_iv_reference"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_p_close = partial(
    assert_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)
_p_dict_close = partial(
    assert_dict_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)

CASES = fixest_cases()


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_DIR / "fe_iv.json")


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_matches_fixest(fixtures, case) -> None:
    res = fit_case(case)
    ref = leaf(fixtures, case)
    label = case.label

    assert_inference_close(
        res,
        ref,
        label,
        rtol=RTOL,
        atol=ATOL,
        p_value_check=_p_dict_close,
    )
    assert res.n_obs == ref["n_obs"], label
    assert res.df_model == ref["df_model"], label
    # 吸収した効果の数（`df_model − 傾き係数の数`）。Sargan・部分Fの自由度に入る。
    assert res.df_model - len(res.params) == ref["n_absorbed"], label
    assert res.df_resid == ref["df_resid"], label
    assert res.stat_df == ref["stat_df"], label

    # 全係数が同時にゼロというWald F。
    assert res.wald_dist == "f"
    assert res.wald_df_num == ref["wald_df_num"], label
    assert res.wald_df_denom == ref["wald_df_denom"], label
    _close(res.wald_statistic, ref["wald_statistic"], f"{label}/wald")
    _p_close(res.wald_p_value, ref["wald_p_value"], f"{label}/wald_p")

    # 内生変数ごとの部分F（`cov_type`によらず等分散）。
    assert list(res.weak_instrument_f_statistics) == list(
        ref["weak_instrument_f"]
    ), label
    for name, expected in ref["weak_instrument_f"].items():
        _close(
            res.weak_instrument_f_statistics[name],
            expected,
            f"{label}/weak_instrument_f/{name}",
        )
    assert res.weak_instrument_f_df_num == ref["weak_instrument_f_df_num"]
    assert res.weak_instrument_f_df_denom == ref["weak_instrument_f_df_denom"]

    # Kleibergen-Paapのrk Wald F。内生変数が1個のときは、同じ`vcov`での第一段階の頑健Wald Fに
    # 等しい（fixestのDriscoll-Kraay・2-way・3-wayも含む）。複数のときは別の統計量で参照が無い。
    if ref["rk_wald_f"] is None:
        assert len(case.x_endog) > 1, label
    else:
        _close(
            res.kleibergen_paap().rk_wald_f,
            ref["rk_wald_f"],
            f"{label}/rk_wald_f",
        )

    # Wu-Hausman（補助回帰は`cov_type`と同じ共分散で回す）。
    assert res.wu_hausman_df_num == ref["wu_hausman_df_num"], label
    assert res.wu_hausman_df_denom == ref["wu_hausman_df_denom"], label
    _close(
        res.wu_hausman_statistic,
        ref["wu_hausman_statistic"],
        f"{label}/wu_hausman",
    )
    _p_close(
        res.wu_hausman_p_value,
        ref["wu_hausman_p_value"],
        f"{label}/wu_hausman_p",
    )

    # Sargan（丁度識別は`None`）。
    if ref["overid_statistic"] is None:
        assert res.overid_statistic is None, label
        assert res.overid_p_value is None, label
        assert res.overid_df is None, label
    else:
        assert res.overid_df == ref["overid_df"], label
        _close(
            res.overid_statistic, ref["overid_statistic"], f"{label}/sargan"
        )
        _p_close(
            res.overid_p_value, ref["overid_p_value"], f"{label}/sargan_p"
        )
