"""FE-2SLS（`FEIV`）の、固定効果のダミー変数を明示した2SLSとのクロスチェック。

`tests/fixtures/benchmarks/fe_iv_dummy_crosscheck.json`と`fe_iv_dummy_cov_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_iv_dummy_crosscheck_fixtures.py`で生成）を読み、
`benchmark/panel/fe_iv_cases.py`の`dummy_cases()`の全ケースで比べる。ダミー変数を外生変数に入れた
2SLS（`linearmodels.iv.IV2SLS`）はwithin変換（交互射影）を使わないため、本実装のwithin変換・
自由度の数え方（`n_absorbed`）・診断とは独立な参照になる（Frisch-Waugh-Lovell）。

## 比べる量

- 係数・標準誤差・係数の分散共分散行列・`df_resid`・`n_absorbed`（`df_model − 係数の数`）:
  classical・hc1（全構造）、1-wayの個体FE × 時点クラスター、HC2・HC3
- Sargan・Wu-Hausman・Cragg-Donald・Kleibergen-Paap（rk LMとrk Wald F）・部分F・R²:
  classical・hc1・時点クラスター
- 有効F: classical・hc1・hc2・hc3・時点クラスター。linearmodelsではなくMOPの論文の式だけから書いた
  独立実装（`benchmark/iv/references/mop_oracle.py`）で、内生変数が1個のシナリオ

## 参照が1つだけの量（`docs/guide/verification-details.md`）

- HC2・HC3の標準誤差・共分散: linearmodelsに無く、numpyで書いたダミー変数2SLSが唯一の参照
- HC2・HC3・DKのKleibergen-Paap: ダミー変数を入れても、linearmodelsが計算するのはclassical・robust・
  clusteredだけ。有効Fのうちhc2・hc3は上のMOP実装が参照
- hc1のKleibergen-Paap rk LM: linearmodelsの`robust`のLMはHC0基準で、本実装に`hc0`は無い（rk Wald Fは
  hc1と一致するので比べる）

定義の対応（Sargan・Wu-Hausmanの尺度、クラスターの`K`）は
`benchmark/panel/references/dummy_2sls_ref.py`のモジュールdoc。
"""

from __future__ import annotations

from functools import partial

import pytest
from _assertions import assert_close, assert_cov_params_close
from _fe_iv_helpers import FIXTURE_DIR, fit_case, leaf
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES

from benchmark.panel.fe_iv_cases import dummy_cases

_TOL = TOLERANCES["fe_iv_dummy_crosscheck"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_p_close = partial(
    assert_close, rtol=_TOL["rtol_p_value"], atol=_TOL["atol_p_value"]
)

CASES = dummy_cases()
# linearmodelsに無く、numpyの手計算だけが参照のcov_type。
NUMPY_ONLY_COV_TYPES = ("hc2", "hc3")


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_DIR / "fe_iv_dummy_crosscheck.json")


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_estimates_match_the_dummy_variable_2sls(fixtures, case) -> None:
    res = fit_case(case)
    ref = leaf(fixtures, case)
    label = case.label

    assert list(res.params) == list(ref["params"]), label
    for name, expected in ref["params"].items():
        _close(res.params[name], expected, f"{label}/params/{name}")
        _close(
            res.std_errors[name],
            ref["std_errors"][name],
            f"{label}/std_errors/{name}",
        )
    assert_cov_params_close(
        res.cov_params, ref["cov_params"], label, rtol=RTOL
    )
    assert res.df_resid == ref["df_resid"], label
    assert res.df_model - len(res.params) == ref["n_absorbed"], label


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.cov.cov_type not in NUMPY_ONLY_COV_TYPES],
    ids=lambda case: case.label,
)
def test_diagnostics_match_the_dummy_variable_2sls(fixtures, case) -> None:
    res = fit_case(case)
    ref = leaf(fixtures, case)
    label = case.label

    # Sargan（linearmodelsの`e'e/n`の尺度を`(n − n_absorbed)/n`倍した値）。
    if ref["overid_statistic"] is None:
        assert res.overid_statistic is None, label
        assert res.overid_df is None, label
    else:
        assert res.overid_df == ref["overid_df"], label
        _close(
            res.overid_statistic, ref["overid_statistic"], f"{label}/sargan"
        )
        _p_close(
            res.overid_p_value, ref["overid_p_value"], f"{label}/sargan_p"
        )

    # Wu-Hausman（`wooldridge_regression`のWald統計量 ÷ 内生変数の数）。
    _close(
        res.wu_hausman_statistic,
        ref["wu_hausman_statistic"],
        f"{label}/wu_hausman",
    )

    # Cragg-Donaldは`cov_type`によらず等分散。
    _close(
        res.cragg_donald().statistic,
        ref["cragg_donald_f"],
        f"{label}/cragg_donald",
    )

    # Kleibergen-Paap。hc1のrk LMは、linearmodelsがHC0基準のため比べない。
    kp = res.kleibergen_paap()
    _close(kp.rk_wald_f, ref["rk_wald_f"], f"{label}/rk_wald_f")
    if case.cov.cov_type == "hc1":
        assert "rk_lm_statistic" not in ref
    else:
        _close(kp.rk_lm_statistic, ref["rk_lm_statistic"], f"{label}/rk_lm")
        _p_close(kp.rk_lm_p_value, ref["rk_lm_p_value"], f"{label}/rk_lm_p")

    # 部分Fは`cov_type`によらず等分散なので、classicalの葉にだけ置いてある。
    if case.cov.cov_type == "classical":
        assert list(res.weak_instrument_f_statistics) == list(
            ref["weak_instrument_f"]
        ), label
        for name, expected in ref["weak_instrument_f"].items():
            _close(
                res.weak_instrument_f_statistics[name],
                expected,
                f"{label}/weak_instrument_f/{name}",
            )
    else:
        assert "weak_instrument_f" not in ref

    # R²（構造残差ベース。3-wayはPanelOLSが扱えないので置いていない）。
    if case.n_effects > 2:
        assert "r_squared_within" not in ref
    else:
        for field in (
            "r_squared_within",
            "r_squared_between",
            "r_squared_overall",
        ):
            _close(getattr(res, field), ref[field], f"{label}/{field}")


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if len(case.x_endog) == 1],
    ids=lambda case: case.label,
)
def test_effective_f_matches_the_mop_oracle(fixtures, case) -> None:
    """有効Fは、論文の式だけから書いた独立実装（内生変数が1個のシナリオ）と一致する。
    hc2・hc3のこの値が、HC2・HC3の有効Fの唯一の外部の参照になる。"""
    ref = leaf(fixtures, case)
    _close(
        fit_case(case).effective_f().statistic,
        ref["effective_f"],
        f"{case.label}/effective_f",
    )


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if len(case.x_endog) > 1],
    ids=lambda case: case.label,
)
def test_effective_f_is_not_in_the_fixture_for_several_endogenous_variables(
    fixtures, case
) -> None:
    """MOPの独立実装は内生変数が1個のときだけ。複数のシナリオには参照値を置いていない。"""
    assert "effective_f" not in leaf(fixtures, case)
