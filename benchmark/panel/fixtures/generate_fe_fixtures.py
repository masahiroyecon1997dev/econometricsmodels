"""FEのテストフィクスチャ（tests/fixtures/benchmarks/fe.json）を生成する。

`benchmark/panel/references/linearmodels_ref.py`（1回呼べば1ケース分の結果を
返す汎用アダプタ）を全シナリオ×全cov_type×1-way/2-wayの組み合わせで呼び出し、
結果を1つのJSONにまとめて書き出す。

このスクリプト自体は`benchmark/`側に置く（ベンチマーク生成ツールの一部）。
生成される`fe.json`は`tests/fixtures/`に置く（テストが読むデータ）。
両者を分けている理由は`.claude/skills/reference-benchmark/SKILL.md`参照。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/panel/freeze.py`参照）。Wooldridgeデータ（wagepan）は
`load_wooldridge.py`経由で都度ロードする。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import linearmodels

from benchmark.common import (
    BENCHMARKS_DIR,
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
    run_fixture_cli,
)
from benchmark.panel.references.linearmodels_ref import run

# hc2/hc3はlinearmodels.PanelOLSが提供しないため対象外（fixestクロスチェック
# 側のみで検証する単一参照実装の例外、`linearmodels_ref.py`モジュールdoc
# 参照）。hacはcross_sectionally_correlatedシナリオが本来の目的（他シナリオ
# でも動くことの確認はできるが統計的な意味は薄い、OLSのHACと同じ扱い）。
COV_TYPES = ["classical", "hc1", "cluster", "hac"]

# unbalancedは1-way専用（6.4節: 2-way FEはバランスパネル必須、2-way要求時の
# ValidationErrorはunbalanced_two_wayシナリオで別途確認する。数値比較対象外）。
ONE_WAY_ONLY_SCENARIOS = ["unbalanced"]

# それ以外はバランスパネルのため1-way/2-way両方で数値比較する。
TWO_WAY_SCENARIOS = [
    "baseline",
    "small_panel",
    "heteroskedastic",
    "autocorrelated",
    "cross_sectionally_correlated",
]

NUMERIC_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS

# singleton_entity/singleton_time/unbalanced_two_way/zero_variance_regressorは
# ここに含まない（いずれもValidationErrorの発生確認のみ、testing-policy.md
# 「テストの3系統」参照。テストコード側で対応）。

# 実データ（Wooldridge wagepan）。T=8年と短くDriscoll-Kraay HACの前提
# （fixestドキュメントが20時点以上を推奨）を満たさないため対象外
# （hacの数値照合は合成データのcross_sectionally_correlatedで十分カバーする）。
WAGEPAN_COV_TYPES = ["classical", "hc1", "cluster"]


def _run_effects(scenario: str, cov_type: str, *, two_way: bool) -> dict:
    # `time_col`は常に実在の"time"列を渡す（`two_way`とは独立、
    # `linearmodels_ref.py`モジュールdoc参照）。1-way + hacで観測順ダミーを
    # 使うと不均衡パネル（unbalancedシナリオ）でバンド幅・カーネル計算が
    # 不正確になるため（実測で発覚、Issue #190）。
    return run(
        scenario,
        ["x1", "x2"],
        cov_type,
        time_col="time",
        two_way=two_way,
        dataset_source="synthetic",
    )


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {"one_way": {}}
        for cov_type in COV_TYPES:
            fixtures[scenario]["one_way"][cov_type] = _run_effects(
                scenario, cov_type, two_way=False
            )
        if scenario in TWO_WAY_SCENARIOS:
            fixtures[scenario]["two_way"] = {
                cov_type: _run_effects(scenario, cov_type, two_way=True)
                for cov_type in COV_TYPES
            }

    fixtures["wagepan"] = {
        "one_way": {
            cov_type: run(
                "wagepan",
                WAGEPAN_X,
                cov_type,
                entity_col=WAGEPAN_ENTITY,
                time_col=WAGEPAN_TIME,
                two_way=False,
                dataset_source="wooldridge",
                y_col=WAGEPAN_Y,
            )
            for cov_type in WAGEPAN_COV_TYPES
        },
        "two_way": {
            cov_type: run(
                "wagepan",
                WAGEPAN_X,
                cov_type,
                entity_col=WAGEPAN_ENTITY,
                time_col=WAGEPAN_TIME,
                two_way=True,
                dataset_source="wooldridge",
                y_col=WAGEPAN_Y,
            )
            for cov_type in WAGEPAN_COV_TYPES
        },
    }

    fixtures["_meta"] = {
        "method": "fe",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "note": (
            "singleton_entity/singleton_time/unbalanced_two_way/"
            "zero_variance_regressorシナリオはここに含まない"
            "（いずれもValidationErrorの発生確認のみ、テストコード側で対応）。"
            "unbalancedは1-way専用（2-way FEはバランスパネル必須のため、"
            "2-way要求時のValidationErrorはunbalanced_two_wayシナリオで確認）。"
            "hc2/hc3はlinearmodels.PanelOLSが提供しないため対象外"
            "（generate_fe_crosscheck_fixtures.pyのfixestクロスチェックのみで"
            "検証する単一参照実装の例外、panel-api-design.md5.4節と同型）。"
            "aic/bicも同じ理由でlinearmodelsに無く、"
            "generate_fe_crosscheck_fixtures.py側のみに含まれる。"
            "2-way FEのr_squared_withinはlinearmodels自身がentityのみdemeanの"
            "別定義を使うため、この値をそのまま数値比較に使わないこと"
            "（fixestのfitstat(m,'wr2')で検証、"
            "linearmodels_ref.pyモジュールdoc参照）。"
            "wagepan（Wooldridge、N=545人×T=8年、1980-1987、バランスパネル）は"
            "married/union/expersqのみ使用——educ/black/hisp等の時間不変変数は"
            "within変換で分散ゼロになり除外、exper自体も2-way FEでentity+time"
            "効果と完全共線になるため除外している"
            "（benchmark/common/constants.pyのWAGEPAN_X参照）。"
            "hacはwagepan（T=8）には適用しない"
            "（Driscoll-Kraay HACはfixestドキュメントが20時点以上を推奨する"
            "ほど時点数に依存するため、合成データのcross_sectionally_"
            "correlatedシナリオ（T=25）でのみ数値照合する）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "fe.json", description=__doc__
    )
