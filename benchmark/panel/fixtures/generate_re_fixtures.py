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
import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.fixtures.generate_fe_fixtures import (
    NUMERIC_SCENARIOS,
    SCENARIO_X_COLS,
)
from benchmark.panel.fixtures.generate_re_crosscheck_fixtures import (
    WAGEPAN_UNBALANCED,
    wagepan_unbalanced,
)
from benchmark.panel.fixtures.generate_re_time_invariant_fixtures import (
    SCENARIO_X_COLS as TIME_INVARIANT_X_COLS,
)
from benchmark.panel.fixtures.generate_re_time_invariant_fixtures import (
    WAGEPAN_X_TIME_INVARIANT,
)
from benchmark.panel.references.linearmodels_ref import run_re

# classical/hc1のみlinearmodelsと比較する。hc2/hc3はlinearmodels.RandomEffectsが
# 提供しないため対象外（plmクロスチェック側のみで検証する単一参照実装の例外、
# `linearmodels_ref.py`モジュールdoc「RE固有の相違点」参照）。cluster/dkは
# 本実装の小標本補正がStata・R型（`G/(G-1)·(n-1)/(n-K)`、t分布の自由度
# `G-1`/`T-1`）に変わりlinearmodelsの`n/(n-k)`とは一致しなくなったため、
# plmのみで検証する（`generate_re_crosscheck_fixtures.py`、
# `docs/spec/re-spec.md`4章）。
COV_TYPES = ["classical", "hc1"]

# `NUMERIC_SCENARIOS`（unbalanced + baseline/small_panel/heteroskedastic/
# autocorrelated/cross_sectionally_correlated）はFE（generate_fe_fixtures.py）
# のものをそのまま再利用する。REは常にentity方向のみ（1-way相当、`re-spec.md`5章）
# なのでFEのような1-way/2-wayの区別が無い。singleton_entity/singleton_time/
# unbalanced_two_way/unbalanced_sparse/unbalanced_disconnected/
# zero_variance_regressorは含めない（FEのシナリオ。REの2-wayは不均衡を拒否する）。
# linearmodelsは`small_sample=True`で、不均衡パネルでも本実装と同じトレース形の
# Swamy-Aroraになる（`linearmodels_ref.py`モジュールdoc「RE固有の相違点」5参照）。
#
# 時間不変の列を含むシナリオ（`generate_re_time_invariant_fixtures.py`と同じCSV・列）も
# 加える。linearmodels 7.1は時間不変の列をwithin回帰の自由度に数えず、σ_ε²・σ_u²とも
# 本実装・plmと機械精度で一致する（バランスの`zero_variance_regressor`と、`T_i`が2〜10に
# 偏る`time_invariant_multi_unbalanced`を含めて実測確認済み）。
TIME_INVARIANT_SCENARIOS = [
    "zero_variance_regressor",
    "time_invariant_unbalanced",
    "time_invariant_multi_unbalanced",
]
SCENARIOS = [*NUMERIC_SCENARIOS, *TIME_INVARIANT_SCENARIOS]

# 不均衡パネルのシナリオ（実データは`WAGEPAN_UNBALANCED`）。linearmodelsの
# `rsquared_between`は個体を重み付けしない（本実装は`T_i`で重み付けする）ため、
# テストはこれらで`r_squared_between`を比べない。
UNBALANCED_SCENARIOS = [
    "unbalanced",
    "time_invariant_unbalanced",
    "time_invariant_multi_unbalanced",
]

# 実データ（Wooldridge wagepan）。FEと同じ変数選定
# （benchmark/common/constants.pyのWAGEPAN_X参照）。行を落とした不均衡版
# （`wagepan_unbalanced`、観測数2・4・7・8）と、時間不変の`educ`・`black`・`hisp`を
# 加えた版（`wagepan_time_invariant`、`generate_re_time_invariant_fixtures.py`の
# wagepanと同じ列）も同じ形で持つ。
WAGEPAN_TIME_INVARIANT = "wagepan_time_invariant"
WAGEPAN_DATASETS = ["wagepan", WAGEPAN_UNBALANCED, WAGEPAN_TIME_INVARIANT]
WAGEPAN_COV_TYPES = ["classical", "hc1"]


def wagepan_dataset(
    wagepan: pl.DataFrame, dataset: str
) -> tuple[pl.DataFrame, list[str]]:
    """`WAGEPAN_DATASETS`の1つについて、データと説明変数を返す（テストと共用）。"""
    if dataset == WAGEPAN_UNBALANCED:
        return wagepan_unbalanced(wagepan), WAGEPAN_X
    if dataset == WAGEPAN_TIME_INVARIANT:
        return wagepan, WAGEPAN_X_TIME_INVARIANT
    return wagepan, WAGEPAN_X


def _scenario_x_cols(scenario: str) -> list[str]:
    if scenario in TIME_INVARIANT_SCENARIOS:
        return TIME_INVARIANT_X_COLS[scenario]
    return SCENARIO_X_COLS.get(scenario, ["x1", "x2"])


def _run_wagepan(dataset: str, cov_type: str) -> dict:
    df, x_cols = wagepan_dataset(load_wooldridge("wagepan"), dataset)
    return run_re(
        dataset,
        x_cols,
        cov_type,
        entity_col=WAGEPAN_ENTITY,
        time_col=WAGEPAN_TIME,
        y_col=WAGEPAN_Y,
        df_override=df,
    )


def _run_re(scenario: str, cov_type: str) -> dict:
    # `time_col`は常に実在の"time"列を渡す（linearmodels呼び出し側の
    # MultiIndex構築専用、RE.fit()自体はtimeを使わない）。
    return run_re(
        scenario,
        _scenario_x_cols(scenario),
        cov_type,
        time_col="time",
        dataset_source="synthetic",
    )


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in SCENARIOS:
        fixtures[scenario] = {
            cov_type: _run_re(scenario, cov_type) for cov_type in COV_TYPES
        }

    for dataset in WAGEPAN_DATASETS:
        fixtures[dataset] = {
            cov_type: _run_wagepan(dataset, cov_type)
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
            "検証する単一参照実装の例外）。cluster/dkは本実装の小標本補正が"
            "Stata・R型に変わりlinearmodelsと一致しないため同様にplmのみで"
            "検証する。aic/bicも同じ理由でlinearmodelsに"
            '無く、かつplmのmodel="random"もlogLik()未対応のため、REの'
            "aic/bic/log_likelihoodは独立検証していない（式自体の正しさは"
            "OLS本体のテストで別途担保、linearmodels_ref.pyモジュールdoc参照）。"
            "ハウスマン検定（hausman_statistic/hausman_p_value/hausman_df）は"
            "linearmodelsに専用実装が無いため本フィクスチャに含まない——"
            "plm::phtestのみを参照値とする例外として"
            "generate_re_crosscheck_fixtures.json側にのみ含める"
            "（panel-common.md5.3節）。v1のハウスマン検定ベンチマークは"
            "1-way（REOptions.dk_time未指定の内部FE呼び出し）に限定する"
            "（RE自身がv1でentity方向のみをサポートするため、`re-spec.md`5章。2-way内部"
            "FE呼び出しのクロスチェックは別issueで検討、ユーザー確認済み・"
            "2026-09-20）。"
            "wagepan（Wooldridge、N=545人×T=8年、1980-1987、バランスパネル）は"
            "FEと同じ変数選定理由（benchmark/common/constants.pyのWAGEPAN_X"
            "参照）。"
            "linearmodelsはfit(small_sample=True)で推定する（不均衡パネルの"
            "1-wayのSwamy-Aroraが本実装・plmのswarと同じトレース形になる。"
            "バランスパネルでは既定と同じ値）。不均衡パネル（unbalanced・"
            "time_invariant_unbalanced・time_invariant_multi_unbalanced・"
            "wagepan_unbalanced）のrsquared_betweenは、linearmodelsが個体を"
            "重み付けせず本実装（T_iで重み付け）と定義が違うため、テストで"
            "比べない。時間不変の列を含むシナリオ（zero_variance_regressor・"
            "time_invariant_unbalanced・time_invariant_multi_unbalanced・"
            "wagepan_time_invariant）もlinearmodels 7.1は時間不変の列を"
            "within回帰の自由度に数えず、本実装・plmと機械精度で一致する。"
            "f_statistic/f_p_valueはlinearmodelsのres.f_statistic_robust"
            "（cov_typeに連動するWald二次形式）。res.f_statistic（変換済みyの"
            "単純平均を基準にしたSST/SSR方式、cov_type非依存）は不均衡パネルで"
            "本実装と一致せず極端な不均衡では負値になるため使わない。"
        ),
    }
    return group_by_dataset_source(fixtures, WAGEPAN_DATASETS)


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "re.json",
        description=__doc__,
    )
