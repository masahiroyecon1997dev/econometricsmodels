"""FE-GMM（`FEIV`）の主リファレンス（within変換後のデータへのlinearmodels `IVGMM`）との数値照合。

`tests/fixtures/benchmarks/fe_iv_gmm.json`と`fe_iv_gmm_cov.json`
（`benchmark/panel/fixtures/generate_fe_iv_gmm_fixtures.py`で生成）を読み、`benchmark/panel/
fe_iv_gmm_cases.py`の`primary_cases()`の全ケースで比べる。固定効果は、群の平均・ダミー変数行列への射影
（`numpy.linalg.lstsq`）で除いた列で、本実装の閉形式・交互射影とは別の計算。

## ケース

- `fe_iv_datasets.SCENARIOS`の全シナリオ（1-way）・2-way・3-way・`x_exog`が空・信頼水準0.90: 既定
  （`two_step`・重み`cluster`）× classical・hc1・個体クラスター。
- `baseline`・`just_identified`・`unbalanced`・2-way（`baseline`・`unbalanced`）・3-way: 既定 × 時点・
  region・sectorのクラスター・DK（既定式と明示のバンド幅）・HC2・HC3。`baseline`と2-way（`baseline`）は、
  one_step・two_step（classical・robust）でも同じ`cov_type`を調べる。
- 外れ値・小標本・時点が少ない・悪条件のシナリオ: 既定 × HC2・HC3。
- `baseline`・`multi_endog`・`unbalanced`・2-way・3-way: one_step・two_step（classical・robust）・
  iterated（3回。cluster・robust重み）× classical・hc1・個体クラスター。

## 比べる量

係数・標準誤差・z・p値・信頼区間・係数の分散共分散行列・`df_model`・`df_resid`・`n_absorbed`・
全係数が同時にゼロというWald（χ²）・Hansen J（`classical`重みと`one_step`は`(n − n_absorbed)/n`倍した
尺度）・反復回数・R²（3-wayは参照なし）。

## 参照が1つだけの量（`docs/guide/verification-details.md`）

hc2・hc3・dkの標準誤差・共分散は、`IVGMM`に無く、numpyの手計算が唯一の参照。丁度識別の
`just_identified`は、FE-2SLSに還元してfixest・plmの値と比べる（`test_fe_iv_gmm_reduction.py`）。

役割分担:
    - 構造・API: `test_fe_iv_gmm_api.py`、`_lib`の配線・エラー・オラクル: `test_fe_iv_gmm_binding.py`
    - 主リファレンス（linearmodels IVGMM）: このファイル
    - 独立実装（R: momentfit）: `test_fe_iv_gmm_crosscheck.py`
    - C統計量: `test_fe_iv_gmm_c_statistic.py`
    - FE-2SLSへの還元: `test_fe_iv_gmm_reduction.py`
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
from _fe_iv_gmm_helpers import (
    FIXTURE_DIR,
    expected_dk_bandwidth,
    fit_case,
    leaf,
)
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES

from benchmark.panel.fe_iv_gmm_cases import primary_cases

_TOL = TOLERANCES["fe_iv_gmm_reference"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_p_close = partial(
    assert_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)
_p_dict_close = partial(
    assert_dict_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)

CASES = primary_cases()
R_SQUARED = ("r_squared_within", "r_squared_between", "r_squared_overall")


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_DIR / "fe_iv_gmm.json")


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_matches_linearmodels_ivgmm(fixtures, case) -> None:
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
    assert res.n_obs == ref["n_obs"], label
    assert res.df_model == ref["df_model"], label
    assert res.df_resid == ref["df_resid"], label
    # 吸収した効果の数（`df_model − 傾き係数の数`）。
    assert res.df_model - len(res.params) == ref["n_absorbed"], label
    if case.base.cov.cov_type == "dk":
        assert res.dk_bandwidth_used == expected_dk_bandwidth(case), label

    # 推定方式と反復回数（`IVGMM`の`iterations`）。
    assert res.gmm_type == case.spec.gmm_type, label
    assert res.gmm_weight_type == case.spec.weight, label
    assert res.n_iter == ref["n_iter"], label

    # 全係数が同時にゼロというWald（χ²。自由度は傾き係数の数）。
    assert res.wald_dist == "chi2"
    assert res.wald_df_num == ref["wald_df_num"], label
    assert res.wald_df_denom is None
    _close(res.wald_statistic, ref["wald_statistic"], f"{label}/wald")
    _p_close(res.wald_p_value, ref["wald_p_value"], f"{label}/wald_p")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_hansen_j_matches_linearmodels_ivgmm(fixtures, case) -> None:
    """Hansen J。`classical`重みと`one_step`は`S`を`n/(n − n_absorbed)`倍した尺度
    （`IVGMM`の値に`(n − n_absorbed)/n`を掛けたもの）。丁度識別は`None`。"""
    res = fit_case(case)
    ref = leaf(fixtures, case)
    label = case.label

    if ref["overid_statistic"] is None:
        assert res.overid_statistic is None, label
        assert res.overid_p_value is None, label
        assert res.overid_df is None, label
        return
    assert res.overid_df == ref["overid_df"], label
    _close(res.overid_statistic, ref["overid_statistic"], f"{label}/hansen_j")
    _p_close(res.overid_p_value, ref["overid_p_value"], f"{label}/hansen_j_p")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_r_squared_matches_the_structural_residual_based_reference(
    fixtures, case
) -> None:
    """R²は構造残差ベース（FE-2SLSと同じ定義）。3-wayは参照を置かない。"""
    res = fit_case(case)
    ref = leaf(fixtures, case)
    if case.base.n_effects > 2:
        assert all(field not in ref for field in R_SQUARED)
        return
    for field in R_SQUARED:
        _close(getattr(res, field), ref[field], f"{case.label}/{field}")
