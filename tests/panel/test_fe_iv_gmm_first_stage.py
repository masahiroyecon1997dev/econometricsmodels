"""FE-GMM（`FEIV(estimator="gmm")`）の第一段階・弱操作変数診断・Wu-Hausmanの照合。

第一段階・弱操作変数診断・Wu-Hausmanは第一段階（と、第一段階の残差を足した最小二乗の補助回帰）だけに依存し、
第二段階の推定量にもGMMの重みにも依らない。同じデータ・同じ`cov_type`なら、GMMはFE-2SLSと同じ値を返す。
`benchmark/panel/fe_iv_gmm_cases.py`の`reduction_cases()`（FE-2SLSの全ケースをGMMに写したもの）で、次を比べる。

- 内生変数ごとの部分F・その自由度・Kleibergen-Paapのrk Wald F・Wu-Hausman（統計量・p値・自由度）を、
  FE-2SLSの主リファレンス（R fixest: `fe_iv.json`）と（**参照テスト**。公開の検証ページの表に数える）
- 第一段階（`first_stage()`の係数・標準誤差・t値）・Cragg-Donald・rk LM・有効Fを、同じケースのFE-2SLSの結果と
  （**整合性テスト**。エンジンの同じ共有実装を2つの推定量から呼んだ結果が一致することを確かめるもので、独立した
  参照ではない。fixestが持たない統計量は、FE-2SLS側で済んでいるダミー変数2SLS・ivDiagとの照合
  （`test_fe_iv_dummy_crosscheck.py`・`test_fe_iv_weak_diagnostics_crosscheck.py`）に載せる橋渡しになる）

前提（GMMの係数がFE-2SLSと同じ）は`reduction_cases()`の定義による（`one_step`と`classical`重みの2段階、
丁度識別の全重み）。診断は係数に依らないので前提が崩れても変わらないが、係数の一致は
`test_fe_iv_gmm_reduction.py`が確かめている。
"""

from __future__ import annotations

from functools import cache

import pytest
from _assertions import assert_close, assert_dict_close
from _fe_iv_gmm_helpers import FIXTURE_DIR
from _fe_iv_gmm_helpers import fit_case as fit_gmm_case
from _fe_iv_helpers import fit_case as fit_2sls_case
from _fe_iv_helpers import leaf
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES
from econometricsmodels import ComputationError, ValidationError

from benchmark.panel.fe_iv_gmm_cases import reduction_cases

_TOL = TOLERANCES["fe_iv_gmm_reduction"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]
_P_RTOL = _TOL["rtol_p_value"]

CASES = reduction_cases()


def _close(actual: float, expected: float, label: str) -> None:
    assert_close(actual, expected, label, rtol=RTOL, atol=ATOL)


@cache
def _fits(case):
    """ケースのGMMとFE-2SLSの推定（ケースごとに1回）。"""
    return fit_gmm_case(case), fit_2sls_case(case.base)


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_DIR / "fe_iv.json")


def test_the_reduction_cases_are_not_empty() -> None:
    assert CASES


# ── fixestの参照値との照合（合成データ） ──────────────────────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_matches_fixest(fixtures, case) -> None:
    res = fit_gmm_case(case)
    ref = leaf(fixtures, case.base)
    label = case.label

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

    # Kleibergen-Paapのrk Wald F（内生変数が1個のとき。複数のときは参照が無い）。
    if ref["rk_wald_f"] is None:
        assert len(case.base.x_endog) > 1, label
    else:
        _close(
            res.kleibergen_paap().rk_wald_f,
            ref["rk_wald_f"],
            f"{label}/rk_wald_f",
        )

    # Wu-Hausman（補助回帰は`cov_type`と同じ共分散で回す。GMMの重みには依らない）。
    assert res.wu_hausman_df_num == ref["wu_hausman_df_num"], label
    assert res.wu_hausman_df_denom == ref["wu_hausman_df_denom"], label
    _close(
        res.wu_hausman_statistic,
        ref["wu_hausman_statistic"],
        f"{label}/wu_hausman",
    )
    assert_close(
        res.wu_hausman_p_value,
        ref["wu_hausman_p_value"],
        f"{label}/wu_hausman_p",
        rtol=_P_RTOL,
        atol=0.0,
    )


# ── FE-2SLSの結果との整合（同じ共有実装の2つの呼び出し） ───────────────


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_first_stage_equals_that_of_fe_2sls(case) -> None:
    gmm, two_sls = _fits(case)
    label = case.label
    gmm_stages, two_sls_stages = gmm.first_stage(), two_sls.first_stage()

    # 内生変数の順、係数・標準誤差・t値・自由度。
    assert list(gmm_stages) == list(two_sls_stages), label
    for name, stage in two_sls_stages.items():
        other = gmm_stages[name]
        for field in ("params", "std_errors", "test_stats"):
            assert_dict_close(
                getattr(other, field),
                getattr(stage, field),
                f"{label}/first_stage/{name}/{field}",
                rtol=RTOL,
                atol=ATOL,
            )
        assert other.df_resid == stage.df_resid, label
        assert other.stat_df == stage.stat_df, label
        assert other.f_statistic is None, label


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_cragg_donald_equals_that_of_fe_2sls(case) -> None:
    gmm, two_sls = _fits(case)
    try:
        expected = two_sls.cragg_donald()
    except ComputationError as err:
        with pytest.raises(ComputationError) as raised:
            gmm.cragg_donald()
        assert raised.value.code == err.code, case.label
        return
    _close(
        gmm.cragg_donald().statistic,
        expected.statistic,
        f"{case.label}/cragg_donald",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_kleibergen_paap_equals_that_of_fe_2sls(case) -> None:
    gmm, two_sls = _fits(case)
    label = case.label
    try:
        expected = two_sls.kleibergen_paap()
    except ComputationError as err:
        with pytest.raises(ComputationError) as raised:
            gmm.kleibergen_paap()
        assert raised.value.code == err.code, label
        return
    actual = gmm.kleibergen_paap()
    assert (actual.rk_lm_statistic is None) == (
        expected.rk_lm_statistic is None
    ), label
    if expected.rk_lm_statistic is not None:
        _close(
            actual.rk_lm_statistic,
            expected.rk_lm_statistic,
            f"{label}/rk_lm",
        )
    assert (actual.rk_wald_f is None) == (expected.rk_wald_f is None), label
    if expected.rk_wald_f is not None:
        _close(actual.rk_wald_f, expected.rk_wald_f, f"{label}/rk_wald_f")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_effective_f_equals_that_of_fe_2sls(case) -> None:
    gmm, two_sls = _fits(case)
    label = case.label
    if len(case.base.x_endog) != 1:
        # 有効Fは内生変数が1個のときだけ定義される（どちらの推定量でも同じ）。
        with pytest.raises(ValidationError):
            two_sls.effective_f()
        with pytest.raises(ValidationError):
            gmm.effective_f()
        return
    try:
        expected = two_sls.effective_f()
    except ComputationError as err:
        with pytest.raises(ComputationError) as raised:
            gmm.effective_f()
        assert raised.value.code == err.code, label
        return
    _close(
        gmm.effective_f().statistic,
        expected.statistic,
        f"{label}/effective_f",
    )


def test_the_diagnostics_are_defined_in_many_cases() -> None:
    """比較が、両方が定義できない入力ばかりで空振りしていない。"""
    defined = {"cragg_donald": 0, "kleibergen_paap": 0, "effective_f": 0}
    for case in CASES:
        _, two_sls = _fits(case)
        for method in defined:
            try:
                getattr(two_sls, method)()
            except (ComputationError, ValidationError):
                continue
            defined[method] += 1
    assert all(count > len(CASES) // 3 for count in defined.values()), defined
