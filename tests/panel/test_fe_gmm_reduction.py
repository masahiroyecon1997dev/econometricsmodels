"""FE-GMM（`FEGMM`）からFE-2SLS（`FEIV`）への還元の、参照（fixest・plm）との照合。

GMMは次の2つの場合にFE-2SLSと同じ量を返す。`benchmark/panel/fe_gmm_cases.py`の`reduction_cases()`を、
FE-2SLSの参照（R fixest: `fe_iv.json`、R plm: `fe_iv_plm_crosscheck.json`）と比べる。

- **重みが`(Z'Z)⁻¹`の定数倍のとき**（`one_step`と`two_step`の`classical`重み）: 係数は2SLSで、共分散は
  過剰識別でも2SLSの共分散になる（サンドイッチは重みの定数倍に依らない）。FE-2SLSの全ケース
  （全シナリオ・1-way・2-way・3-way・全`cov_type`: クラスターの列・DK・信頼水準を含む）で、係数・標準誤差・
  共分散行列・z統計量（FE-2SLSのt統計量と同じ値）を比べる。Hansen Jもこのとき2SLSのSarganで、fixestの
  `e'e/n`の尺度を`(n − n_absorbed)/n`倍した`fe_iv.json`の値と一致する。
- **丁度識別のとき**: どの重みでも共分散は重みの行列に依らない。`just_identified`の1-wayの全`cov_type`で、
  `two_step`の`cluster`・`robust`重みも比べる。

この還元は、主リファレンス（`IVGMM`）が持たないDK・時点クラスター・入れ子のクラスターの小標本補正
（手書きのスカラー、`K`の数え方を含む）と、Hansen Jの`(n − n_absorbed)/n`の再スケールを、
fixestの値により独立に確かめる。
"""

from __future__ import annotations

import pytest
from _assertions import (
    assert_close,
    assert_cov_params_close,
    assert_dict_close,
)
from _fe_gmm_helpers import FIXTURE_DIR, fit_case, leaf
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES

from benchmark.panel.fe_gmm_cases import reduction_cases
from benchmark.panel.fe_iv_cases import plm_cases

_TOL = TOLERANCES["fe_gmm_reduction"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

CASES = reduction_cases()
# plmが参照にできる`cov_type`（`plm_cases()`のキー）。
_PLM_PATHS = {case.path for case in plm_cases()}
REFERENCES = {
    "fixest": ("fe_iv.json", lambda case: True),
    "plm": (
        "fe_iv_plm_crosscheck.json",
        lambda case: case.base.path in _PLM_PATHS,
    ),
}


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.fixture(scope="module")
def fixtures() -> dict[str, dict]:
    return {
        name: load_fixtures_with_cov(FIXTURE_DIR / file)
        for name, (file, _) in REFERENCES.items()
    }


def _cases(reference: str):
    keep = REFERENCES[reference][1]
    return [case for case in CASES if keep(case)]


class _Fe2slsLeaf:
    """FE-2SLSの葉を`leaf()`で引けるようにする（`path`はFE-2SLSのケースのもの）。"""

    def __init__(self, case) -> None:
        self.path = case.base.path


def _fe_iv_leaf(fixtures: dict, case) -> dict:
    return leaf(fixtures, _Fe2slsLeaf(case))


@pytest.mark.parametrize("reference", REFERENCES)
def test_the_reference_cases_are_not_empty(reference) -> None:
    assert _cases(reference)


@pytest.mark.parametrize(
    ("reference", "case"),
    [(r, c) for r in REFERENCES for c in _cases(r)],
    ids=lambda v: v if isinstance(v, str) else v.label,
)
def test_just_identified_fe_gmm_reduces_to_fe_2sls(
    fixtures, reference, case
) -> None:
    res = fit_case(case)
    ref = _fe_iv_leaf(fixtures[reference], case)
    label = f"{case.label}/{reference}"

    assert list(res.params) == list(ref["params"]), label
    assert_dict_close(
        res.params, ref["params"], f"{label}/params", rtol=RTOL, atol=ATOL
    )
    assert_dict_close(
        res.std_errors,
        ref["std_errors"],
        f"{label}/std_errors",
        rtol=RTOL,
        atol=ATOL,
    )
    # z統計量はFE-2SLSのt統計量と同じ値（`β / SE`）。分布だけが違う。
    assert_dict_close(
        res.test_stats,
        ref["test_stats"],
        f"{label}/test_stats",
        rtol=RTOL,
        atol=ATOL,
    )
    assert_cov_params_close(
        res.cov_params, ref["cov_params"], label, rtol=RTOL
    )


@pytest.mark.parametrize(
    "case",
    [case for case in _cases("fixest") if case.spec.weight != "robust"],
    ids=lambda case: case.label,
)
def test_hansen_j_of_a_classical_weight_is_the_fe_2sls_sargan(
    fixtures, case
) -> None:
    """`one_step`と`classical`重みのHansen Jは、FE-2SLSのSargan（`fe_iv.json`の
    `overid_statistic`。fixestの値に`(n − n_absorbed)/n`を掛けたもの）と同じ。丁度識別は`None`。"""
    res = fit_case(case)
    ref = _fe_iv_leaf(fixtures["fixest"], case)
    label = case.label

    if ref["overid_statistic"] is None:
        assert res.overid_statistic is None, label
        assert res.overid_df is None, label
        return
    assert res.overid_df == ref["overid_df"], label
    assert_close(
        res.overid_statistic,
        ref["overid_statistic"],
        f"{label}/sargan",
        rtol=RTOL,
        atol=ATOL,
    )
    assert_close(
        res.overid_p_value,
        ref["overid_p_value"],
        f"{label}/sargan_p",
        rtol=_TOL["rtol_p_value"],
        atol=_TOL["atol_p_value"],
    )
