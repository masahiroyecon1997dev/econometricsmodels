"""Tobitの独立実装（R `censReg` ＝ `maxLik` エンジン）による数値比較テスト。

`tests/fixtures/benchmarks/tobit_crosscheck.json`（`benchmark/nonlinear/fixtures/
generate_tobit_crosscheck_fixtures.py` で生成）を読み込み、`test_tobit_reference.py`
と同じケース・同じフィールドを `censReg` とクロスチェックする。役割分担は
`test_tobit_reference.py` の docstring 参照（`.claude/rules/testing-policy.md`
「リファレンス実装」）。

`docs/planning/specs/nonlinear-api-design.md` 9章:「`survreg` と `maxLik` は
最適化実装が完全に独立しているため交差検証として組み合わせる価値が高い」。
主リファレンス（`AER::tobit`）と交差検証（`censReg`）がどちらも R 実装のため、
限界効果等の手計算箇所は `run_tobit_crosscheck.R` 内で本実装の閉形式を再現し、
`numDeriv` による数値微分と一致することを別途確認している（同スクリプト参照）。

Note:
    許容誤差は基本 1e-8（`censReg` 側の maxLik 収束を reltol=1e-14 まで詰めたため、
    合成シナリオは点推定・SE・限界効果とも ~2e-9 で一致する）。ただし以下は
    実測乖離に基づき個別に緩める（`tests/_tolerances.py` の `tobit_crosscheck`）:
        - `high_condition_number`（x1,x2 相関 0.999）の SE・z・信頼区間・限界効果 SE:
          悪条件下で2つの独立最適化器の解の僅差が分散系で ~1.9e-8 まで増幅する。
        - `mroz`（`hours` 生スケール）の SE・z・Wald・信頼区間・限界効果 SE:
          `censReg` の maxLik が生スケール悪条件データで `survreg` ほど収束が詰まらず
          ~1e-7〜1.4e-6 乖離する（点推定は ~3e-9 で一致。engine と主リファレンス
          `survreg` は同データで ~3e-10 一致するため `censReg` 側の収束限界）。
"""

from __future__ import annotations

import _tobit_checks as _checks
import pytest
from _tolerances import TOLERANCES

_TOL = TOLERANCES["tobit_crosscheck"]
FIXTURE_PATH = _checks.FIXTURE_DIR / "tobit_crosscheck.json"


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return _checks.load_fixtures(FIXTURE_PATH)


def _tolerances(scenario: str) -> dict:
    rtol = _TOL["rtol"]
    rtol_inference = rtol
    if scenario == "high_condition_number":
        rtol_inference = _TOL["rtol_high_condition_number"]
    elif scenario == "mroz":
        rtol_inference = _TOL["rtol_mroz"]
    return {
        # 点推定・尤度系は全シナリオで基本方針どおり厳密（実測 ~3e-9）。
        "rtol_point": rtol,
        "rtol_inference": rtol_inference,
        "rtol_conf_int": rtol_inference,
        "atol": _TOL["atol"],
    }


def _method_tolerances() -> dict:
    rtol = _TOL["rtol_method"]
    return {
        "rtol_point": rtol,
        "rtol_inference": rtol,
        "rtol_conf_int": rtol,
        "atol": _TOL["atol"],
    }


@pytest.mark.parametrize("cov_type", _checks.COV_TYPES)
@pytest.mark.parametrize("scenario", _checks.SCENARIOS)
def test_matches_censreg(fixtures, scenario, cov_type):
    ref = fixtures[scenario][cov_type]
    res = _checks.build_fit(scenario, cov_type, ref)
    _checks.check_result(
        res, ref, f"{scenario}/{cov_type}", **_tolerances(scenario)
    )


@pytest.mark.parametrize("cov_key", _checks.CLUSTER_COV_KEYS)
def test_cluster_matches_censreg(fixtures, cov_key):
    ref = fixtures[_checks.BASELINE_SCENARIO][cov_key]
    res = _checks.build_fit(_checks.BASELINE_SCENARIO, cov_key, ref)
    _checks.check_result(
        res,
        ref,
        f"{_checks.BASELINE_SCENARIO}/{cov_key}",
        **_tolerances(_checks.BASELINE_SCENARIO),
    )


@pytest.mark.parametrize("method", _checks.METHODS)
def test_method_matches_censreg(fixtures, method):
    ref = fixtures["method"][method]
    res = _checks.build_method_fit(method, ref)
    _checks.check_result(res, ref, f"method/{method}", **_method_tolerances())


@pytest.mark.parametrize("cov_type", _checks.COV_TYPES)
def test_no_intercept_matches_censreg(fixtures, cov_type):
    ref = fixtures["no_intercept"][cov_type]
    res = _checks.build_no_intercept_fit(cov_type, ref)
    _checks.check_result(
        res,
        ref,
        f"no_intercept/{cov_type}",
        **_tolerances(_checks.BASELINE_SCENARIO),
    )


@pytest.mark.parametrize("cov_type", _checks.COV_TYPES)
def test_mroz_matches_censreg(fixtures, cov_type):
    ref = fixtures["mroz"][cov_type]
    res = _checks.build_fit("mroz", cov_type, ref)
    _checks.check_result(res, ref, f"mroz/{cov_type}", **_tolerances("mroz"))
