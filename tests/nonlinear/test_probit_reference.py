"""Probitの主リファレンス（statsmodels）による数値比較テスト。

`tests/fixtures/benchmarks/probit.json`（`benchmark/nonlinear/fixtures/
generate_probit_fixtures.py`で生成）を読み込み、真のprobit DGPによる合成データ
シナリオ×classical/opg/hc0 + クラスター(baseline・mrozの実データ両方) +
Wooldridge実データ（mroz）で、係数・標準誤差・検定統計量・適合度統計量・
限界効果を相対誤差1e-8で厳密比較する（`test_logit_reference.py`と完全に同型。
`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。

役割分担（OLS/WLS の `test_<手法>_*.py` と同じ4分割、
`refactoring-candidates-2.md` 項目68）:
    - 成功パスの構造・API・オプション反映・predict 等: `test_probit_api.py`
    - `ValidationError`/`ComputationError` パス: `test_probit_validation.py`
    - 主リファレンス（statsmodels）との厳密な数値一致: このファイル
    - 独立実装（R）とのクロスチェック: `test_probit_crosscheck.py`

このファイルは凍結フィクスチャ（`probit.json`）との厳密比較を主軸としつつ、
凍結フィクスチャが対象にしない `include_intercept=False` はライブ statsmodels
との直接照合で確認する（`test_ols_reference.py` と同じ構成）。

Note:
    `cov_type="hc1"`はここに含めない（statsmodelsのdiscrete modelがn/(n-k)
    小標本補正を実装しておらずHC0と同一値になるバグ的な欠落があるため、Probitでも
    同じ欠落を実機確認済み。`benchmark/nonlinear/references/statsmodels_ref.py`の
    docstring参照）。`hc1`の数値比較は`test_probit_crosscheck.py`（R側が主リファレンス）
    で行う。

    `cov_type="opg"`の限界効果はstatsmodels側では算出できない（同docstring参照）
    ため、opgのmarginal_effects()数値比較は`test_probit_crosscheck.py`のみで行う。

テスト本体は `Logit`/`Probit` で完全に重複するため
`_binary_choice_checks.py` に集約し（`refactoring-candidates-2.md` 項目95）、
このファイルは手法ごとの設定（`BinaryChoiceReferenceConfig`）を組み立てて
渡す薄いラッパーに保つ。
"""

from __future__ import annotations

from pathlib import Path

import _binary_choice_checks as _checks
import pytest
import statsmodels.api as sm
from _tolerances import TOLERANCES
from econometricsmodels import Probit, ProbitOptions

from benchmark.nonlinear.fixtures.generate_probit_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "probit.json"
)

# OLS（閉形式解）のATOL=1e-10より緩い。Probitは反復最適化（Newton/BFGS/L-BFGS）の
# ため、ゼロ近傍の値（信頼区間の境界等）で閉形式解より1桁大きい浮動小数点誤差が
# 乗ることを実測確認した（Logitと同じ理由、`test_logit_reference.py`参照）。
#
# near_separation（probit特有の準完全分離境界ケース）は、既定のtol=1e-6（勾配ノルム
# 基準）だとstatsmodelsとの数値一致がRTOL=1e-8を満たさない（実測diff~4.4e-8相対、
# Logitのnear_separationと同種の現象）。tol=1e-8まで締めると一致することを確認済み
# だが、既定値自体は変更しない（Logitと同じ理由、`nonlinear-implementation-notes.md`
# 「収束判定のtol」参照）。このシナリオの数値比較テストに限り、明示的にtol=1e-8を
# 指定する。
CONFIG = _checks.BinaryChoiceReferenceConfig(
    estimator_cls=Probit,
    options_cls=ProbitOptions,
    dataset_prefix="probit",
    fixture_path=FIXTURE_PATH,
    scenarios=SCENARIOS,
    cov_types=["classical", "opg", "hc0"],
    rtol=TOLERANCES["probit_reference"]["rtol"],
    atol=TOLERANCES["probit_reference"]["atol"],
    # method="bfgs"/"lbfgs"はnewtonと異なる最適化経路で収束するため、既定の
    # RTOLより緩めた許容誤差を使う（tests/_tolerances.py参照、
    # test_logit_reference.pyと同じ方針）。
    rtol_method=TOLERANCES["probit_reference"]["rtol_method"],
    near_separation_tol=1e-8,
)


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return CONFIG.load_fixtures()


# ── 凍結フィクスチャとの数値照合 ───────────────────────────────────


@pytest.mark.parametrize("cov_type", CONFIG.cov_types)
@pytest.mark.parametrize("scenario", CONFIG.scenarios)
def test_matches_statsmodels(fixtures, scenario, cov_type):
    _checks.check_matches_statsmodels(CONFIG, fixtures, scenario, cov_type)


def test_cluster_matches_statsmodels(fixtures):
    _checks.check_cluster_matches_statsmodels(CONFIG, fixtures)


def test_cluster_imbalanced_matches_statsmodels(fixtures):
    _checks.check_cluster_imbalanced_matches_statsmodels(CONFIG, fixtures)


def test_cluster_g2_matches_statsmodels(fixtures):
    _checks.check_cluster_g2_matches_statsmodels(CONFIG, fixtures)


@pytest.mark.parametrize("method", ["bfgs", "lbfgs"])
def test_method_matches_statsmodels(fixtures, method):
    _checks.check_method_matches_statsmodels(CONFIG, fixtures, method)


@pytest.mark.parametrize("cov_type", CONFIG.cov_types)
def test_mroz_matches_statsmodels(fixtures, cov_type):
    _checks.check_mroz_matches_statsmodels(CONFIG, fixtures, cov_type)


def test_mroz_cluster_matches_statsmodels(fixtures):
    _checks.check_mroz_cluster_matches_statsmodels(CONFIG, fixtures)


# ── ライブ statsmodels との照合（凍結フィクスチャ対象外の分岐） ─────


@pytest.mark.parametrize("cov_type", CONFIG.cov_types)
def test_include_intercept_false_matches_statsmodels(cov_type):
    _checks.check_include_intercept_false_matches_statsmodels(
        CONFIG, sm.Probit, cov_type
    )
