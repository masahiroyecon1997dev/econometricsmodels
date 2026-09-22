"""REのテストフィクスチャ（tests/fixtures/benchmarks/re.json）を生成する。

`benchmark/panel/references/linearmodels_ref.py`の`run_re()`（1回呼べば1ケース
分の結果を返す汎用アダプタ）を全シナリオ×全cov_typeの組み合わせで呼び出し、
結果を1つのJSONにまとめて書き出す。

このスクリプト自体は`benchmark/`側に置く（ベンチマーク生成ツールの一部）。
生成される`re.json`は`tests/fixtures/`に置く（テストが読むデータ）。両者を
分けている理由は`.claude/skills/reference-benchmark/SKILL.md`参照。

**RE専用の合成データセットは無い**: 入力データは`tests/fixtures/benchmarks/
data/`に固定済みのFEのCSV（`fe_{scenario}.csv`）をそのまま再利用する
（`benchmark/panel/references/linearmodels_ref.py`モジュールdoc「RE
（`run_re()`）固有の相違点」参照、ユーザー確認済み・2026-09-20）。Wooldridge
データ（wagepan）は`load_wooldridge.py`経由で都度ロードする。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_fixtures
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
from benchmark.panel.fixtures.generate_fe_fixtures import NUMERIC_SCENARIOS
from benchmark.panel.references.linearmodels_ref import run_re

# hc2/hc3はlinearmodels.RandomEffectsが提供しないため対象外（plmクロスチェック
# 側のみで検証する単一参照実装の例外、`linearmodels_ref.py`モジュールdoc
# 「RE固有の相違点」参照）。hacはFEと同じくcross_sectionally_correlated
# シナリオが本来の目的。
COV_TYPES = ["classical", "hc1", "cluster", "hac"]

# `NUMERIC_SCENARIOS`（unbalanced + baseline/small_panel/heteroskedastic/
# autocorrelated/cross_sectionally_correlated）はFE（generate_fe_fixtures.py）
# のものをそのまま再利用する。REは常にentity方向のみ（1-way相当、`re-spec.md`5章）
# なのでFEのような1-way/2-wayの区別が無く、unbalancedもFEの
# ONE_WAY_ONLY_SCENARIOS制約（2-way FEはバランスパネル必須）を受けずそのまま
# success pathとして扱える。singleton_entity/singleton_time/
# unbalanced_two_way/zero_variance_regressorは含めない（FE同様
# ValidationErrorパス専用、テストコード側で対応）。

# 実データ（Wooldridge wagepan）。FEと同じ変数選定・hac対象外の理由
# （benchmark/common/constants.pyのWAGEPAN_X参照）。
WAGEPAN_COV_TYPES = ["classical", "hc1", "cluster"]


def _run_re(scenario: str, cov_type: str) -> dict:
    # `time_col`は常に実在の"time"列を渡す（`generate_fe_fixtures.py`と同じ
    # 理由——不均衡パネルでcov_type="hac"のバンド幅・カーネル計算が不正確に
    # なることを避けるため、Issue #190）。`REOptions.time`自体は本フィクス
    # チャの対象外（RE.fit()自体はtimeを使わない、`time_col`はlinearmodels
    # 呼び出し側のMultiIndex構築専用）。
    return run_re(
        scenario,
        ["x1", "x2"],
        cov_type,
        time_col="time",
        dataset_source="synthetic",
    )


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {
            cov_type: _run_re(scenario, cov_type) for cov_type in COV_TYPES
        }

    fixtures["wagepan"] = {
        cov_type: run_re(
            "wagepan",
            WAGEPAN_X,
            cov_type,
            entity_col=WAGEPAN_ENTITY,
            time_col=WAGEPAN_TIME,
            dataset_source="wooldridge",
            y_col=WAGEPAN_Y,
        )
        for cov_type in WAGEPAN_COV_TYPES
    }

    fixtures["_meta"] = {
        "method": "re",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "note": (
            "合成データはFE（tests/fixtures/benchmarks/data/fe_*.csv）と"
            "同じ固定済みCSVを再利用する（RE専用のデータセット生成・凍結コード"
            "は追加しない、ユーザー確認済み・2026-09-20。benchmark/linear系統で"
            'OLS/WLSがprefix"synthetic"を共有する前例と同型）。'
            "singleton_entity/singleton_time/unbalanced_two_way/"
            "zero_variance_regressorシナリオはここに含まない（いずれもFE同様"
            "ValidationErrorの発生確認のみ、テストコード側で対応）。"
            "hc2/hc3はlinearmodels.RandomEffectsが提供しないため対象外"
            "（generate_re_crosscheck_fixtures.pyのplmクロスチェックのみで"
            "検証する単一参照実装の例外）。aic/bicも同じ理由でlinearmodelsに"
            '無く、かつplmのmodel="random"もlogLik()未対応のため、REの'
            "aic/bic/log_likelihoodは独立検証していない（式自体の正しさは"
            "OLS本体のテストで別途担保、linearmodels_ref.pyモジュールdoc参照）。"
            "ハウスマン検定（hausman_statistic/hausman_p_value/hausman_df）は"
            "linearmodelsに専用実装が無いため本フィクスチャに含まない——"
            "plm::phtestのみを参照値とする例外として"
            "generate_re_crosscheck_fixtures.json側にのみ含める"
            "（panel-common.md5.3節）。v1のハウスマン検定ベンチマークは"
            "1-way（REOptions.time未指定の内部FE呼び出し）に限定する"
            "（RE自身がv1でentity方向のみをサポートするため、`re-spec.md`5章。2-way内部"
            "FE呼び出しのクロスチェックは別issueで検討、ユーザー確認済み・"
            "2026-09-20）。"
            "wagepan（Wooldridge、N=545人×T=8年、1980-1987、バランスパネル）は"
            "FEと同じ変数選定理由（benchmark/common/constants.pyのWAGEPAN_X"
            "参照）。hacはwagepan（T=8）には適用しない（Driscoll-Kraay HACは"
            "20時点以上を推奨するため、合成データのcross_sectionally_"
            "correlatedシナリオ（T=25）でのみ数値照合する）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "re.json", description=__doc__
    )
