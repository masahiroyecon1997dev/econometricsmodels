"""FE-GMM（`FEIV(estimator="gmm")`）の第一段階・弱操作変数診断・Wu-Hausmanの照合。

第一段階・弱操作変数診断・Wu-Hausmanは第一段階（と、第一段階の残差を足した最小二乗の補助回帰）だけに依存し、
第二段階の推定量にもGMMの重みにも依らない。同じデータ・同じ`cov_type`なら、GMMはFE-2SLSと同じ値を返す。
`benchmark/panel/fe_iv_gmm_cases.py`の`reduction_cases()`（FE-2SLSの全ケースをGMMに写したもの）で、次を比べる。

- 内生変数ごとの部分F・その自由度・Kleibergen-Paapのrk Wald F・Wu-Hausman（統計量・p値・自由度）を、
  FE-2SLSの主リファレンス（R fixest: `fe_iv.json`）と
- 第一段階（`first_stage()`の係数・標準誤差）・Cragg-Donald・有効F・rk LMを、同じケースのFE-2SLSの結果と

後者は、fixestが持たない統計量（Cragg-Donald・有効F・rk LM・第一段階の標準誤差の全て）を、FE-2SLS側で済んでいる
ダミー変数2SLS・ivDiagとの照合に載せるための橋渡しになる（`test_fe_iv_dummy_crosscheck.py`・
`test_fe_iv_weak_diagnostics_crosscheck.py`）。
"""

from __future__ import annotations

import pytest
from _assertions import (
    assert_close,
    assert_dict_close,
)
from _fe_iv_gmm_helpers import FIXTURE_DIR
from _fe_iv_gmm_helpers import fit_case as fit_gmm_case
from _fe_iv_helpers import fit_case as fit_2sls_case
from _fe_iv_helpers import leaf
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES
from econometricsmodels import ComputationError

from benchmark.panel.fe_iv_gmm_cases import reduction_cases

_TOL = TOLERANCES["fe_iv_gmm_reduction"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]
_P_RTOL = _TOL["rtol_p_value"]

CASES = reduction_cases()


class _Fe2slsLeaf:
    """FE-2SLSの葉を`leaf()`で引けるようにする（`path`はFE-2SLSのケースのもの）。"""

    def __init__(self, case) -> None:
        self.path = case.base.path


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_DIR / "fe_iv.json")


def test_the_reduction_cases_are_not_empty() -> None:
    assert CASES


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_gmm_diagnostics_match_fixest(fixtures, case) -> None:
    res = fit_gmm_case(case)
    ref = leaf(fixtures, _Fe2slsLeaf(case))
    label = case.label

    # 内生変数ごとの部分F（`cov_type`によらず等分散）。
    assert list(res.weak_instrument_f_statistics) == list(
        ref["weak_instrument_f"]
    ), label
    for name, expected in ref["weak_instrument_f"].items():
        assert_close(
            res.weak_instrument_f_statistics[name],
            expected,
            f"{label}/weak_instrument_f/{name}",
            rtol=RTOL,
            atol=ATOL,
        )
    assert res.weak_instrument_f_df_num == ref["weak_instrument_f_df_num"]
    assert res.weak_instrument_f_df_denom == ref["weak_instrument_f_df_denom"]

    # Kleibergen-Paapのrk Wald F（内生変数が1個のとき。複数のときは参照が無い）。
    if ref["rk_wald_f"] is not None:
        assert_close(
            res.kleibergen_paap().rk_wald_f,
            ref["rk_wald_f"],
            f"{label}/rk_wald_f",
            rtol=RTOL,
            atol=ATOL,
        )

    # Wu-Hausman（補助回帰は`cov_type`と同じ共分散で回す。GMMの重みには依らない）。
    assert res.wu_hausman_df_num == ref["wu_hausman_df_num"], label
    assert res.wu_hausman_df_denom == ref["wu_hausman_df_denom"], label
    assert_close(
        res.wu_hausman_statistic,
        ref["wu_hausman_statistic"],
        f"{label}/wu_hausman",
        rtol=RTOL,
        atol=ATOL,
    )
    assert_close(
        res.wu_hausman_p_value,
        ref["wu_hausman_p_value"],
        f"{label}/wu_hausman_p",
        rtol=_P_RTOL,
        atol=0.0,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_gmm_first_stage_and_diagnostics_equal_those_of_fe_2sls(case) -> None:
    gmm = fit_gmm_case(case)
    two_sls = fit_2sls_case(case.base)
    label = case.label

    # 第一段階（内生変数の順、係数・標準誤差・t値・自由度）。
    assert list(gmm.first_stage()) == list(two_sls.first_stage()), label
    for name, stage in two_sls.first_stage().items():
        other = gmm.first_stage()[name]
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

    # Cragg-Donald・Kleibergen-Paap・有効F。定義できない入力では、同じ理由で両方が定義できない。
    for method in ("cragg_donald", "kleibergen_paap", "effective_f"):
        try:
            expected = getattr(two_sls, method)()
        except ComputationError as err:
            with pytest.raises(ComputationError) as raised:
                getattr(gmm, method)()
            assert raised.value.code == err.code, f"{label}/{method}"
            continue
        except Exception as err:  # noqa: BLE001 有効Fは内生変数が1個のときだけ
            with pytest.raises(type(err)):
                getattr(gmm, method)()
            continue
        actual = getattr(gmm, method)()
        if method == "cragg_donald":
            assert_close(
                actual.statistic,
                expected.statistic,
                f"{label}/cragg_donald",
                rtol=RTOL,
                atol=ATOL,
            )
        elif method == "kleibergen_paap":
            assert (actual.rk_lm_statistic is None) == (
                expected.rk_lm_statistic is None
            ), label
            if expected.rk_lm_statistic is not None:
                assert_close(
                    actual.rk_lm_statistic,
                    expected.rk_lm_statistic,
                    f"{label}/rk_lm",
                    rtol=RTOL,
                    atol=ATOL,
                )
            if expected.rk_wald_f is not None:
                assert_close(
                    actual.rk_wald_f,
                    expected.rk_wald_f,
                    f"{label}/rk_wald_f",
                    rtol=RTOL,
                    atol=ATOL,
                )
        else:
            assert_close(
                actual.statistic,
                expected.statistic,
                f"{label}/effective_f",
                rtol=RTOL,
                atol=ATOL,
            )
