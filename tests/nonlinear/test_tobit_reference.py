"""Tobitの主リファレンス（R `AER::tobit` ＝ `survival::survreg`）による数値比較テスト。

`tests/fixtures/benchmarks/tobit.json`（`benchmark/nonlinear/fixtures/
generate_tobit_fixtures.py` で生成）を読み込み、打ち切り比率違い・右/区間打ち切り・
構造的悪条件の合成シナリオ × classical/opg/hc0/hc1 + クラスター（均等・不均衡・
G>q 境界）+ method(bfgs/lbfgs) + Wooldridge 実データ（mroz `hours`、Example 17.2）で、
係数・標準誤差・検定統計量・適合度統計量・限界効果・予測値・打ち切り適合度を
相対誤差 1e-8 で厳密比較する（`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。

役割分担（Logit/Probit の `test_<手法>_*.py` と同じ4分割、
`refactoring-candidates-2.md` 項目68）:
    - 成功パスの構造・API・オプション反映・predict/marginal_effects/
      censoring_fit_check の構造・`ValidationError`/`ComputationError` パス:
      `test_tobit.py`
    - 主リファレンス（`AER::tobit`）との厳密な数値一致: このファイル
    - 独立実装（`censReg`）とのクロスチェック: `test_tobit_crosscheck.py`

`AER::tobit` は `survival::survreg` の薄いラッパーで係数・スケール・vcov・logLik は
survreg 由来。survreg は内部で `(β, log σ)` を最適化するが、本実装が公開する
`(β, σ)` 空間へヤコビアン `diag(1,…,1, σ)` で変換した値と実測で係数 ~3e-9・
標準誤差 ~1e-9・対数尤度 ~1e-12 で一致する（Issue #227）。

Note:
    mroz（`hours` 生スケール）は説明変数のスケール差が大きく、信頼区間の端点が
    0 近傍になる係数で相対誤差が ~1.4e-8 まで増幅する（係数・SE 本体は ~3e-10）。
    このシナリオの `conf_int` のみ緩めた許容誤差を使う（`tests/_tolerances.py`）。
    テスト対象フィクスチャは主・交差検証で構造が同一のため、テスト本体は
    `_tobit_checks.py` に集約している（`test_tobit_crosscheck.py` も同モジュールを使う）。
"""

from __future__ import annotations

import _tobit_checks as _checks
import pytest
from _tolerances import TOLERANCES

_TOL = TOLERANCES["tobit_reference"]
FIXTURE_PATH = _checks.FIXTURE_DIR / "tobit.json"


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return _checks.load_fixtures(FIXTURE_PATH)


def _tolerances(scenario: str) -> dict:
    rtol = _TOL["rtol"]
    rtol_conf_int = _TOL["rtol_mroz_conf_int"] if scenario == "mroz" else rtol
    return {
        "rtol_point": rtol,
        "rtol_inference": rtol,
        "rtol_conf_int": rtol_conf_int,
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
def test_matches_aer_tobit(fixtures, scenario, cov_type):
    ref = fixtures[scenario][cov_type]
    res = _checks.build_fit(scenario, cov_type, ref)
    _checks.check_result(
        res, ref, f"{scenario}/{cov_type}", **_tolerances(scenario)
    )


@pytest.mark.parametrize("cov_key", _checks.CLUSTER_COV_KEYS)
def test_cluster_matches_aer_tobit(fixtures, cov_key):
    ref = fixtures[_checks.BASELINE_SCENARIO][cov_key]
    res = _checks.build_fit(_checks.BASELINE_SCENARIO, cov_key, ref)
    _checks.check_result(
        res,
        ref,
        f"{_checks.BASELINE_SCENARIO}/{cov_key}",
        **_tolerances(_checks.BASELINE_SCENARIO),
    )


@pytest.mark.parametrize("method", _checks.METHODS)
def test_method_matches_aer_tobit(fixtures, method):
    ref = fixtures["method"][method]
    res = _checks.build_method_fit(method, ref)
    _checks.check_result(res, ref, f"method/{method}", **_method_tolerances())


@pytest.mark.parametrize("cov_type", _checks.COV_TYPES)
def test_no_intercept_matches_aer_tobit(fixtures, cov_type):
    ref = fixtures["no_intercept"][cov_type]
    res = _checks.build_no_intercept_fit(cov_type, ref)
    _checks.check_result(
        res,
        ref,
        f"no_intercept/{cov_type}",
        **_tolerances(_checks.BASELINE_SCENARIO),
    )


@pytest.mark.parametrize("cov_type", _checks.COV_TYPES)
def test_mroz_matches_aer_tobit(fixtures, cov_type):
    ref = fixtures["mroz"][cov_type]
    res = _checks.build_fit("mroz", cov_type, ref)
    _checks.check_result(res, ref, f"mroz/{cov_type}", **_tolerances("mroz"))
