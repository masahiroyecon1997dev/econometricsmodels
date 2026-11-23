"""FEのテストフィクスチャ（tests/fixtures/benchmarks/fe.json）を生成する。

`benchmark/panel/references/linearmodels_ref.py`（1回呼べば1ケース分の結果を
返す汎用アダプタ）を全シナリオ×classical/hc1×1-way/2-wayの組み合わせで呼び出し、
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
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.references.linearmodels_ref import run

# classical/hc1のみlinearmodelsと比較する。hc2/hc3はlinearmodels.PanelOLSが
# 提供しないため対象外（fixestクロスチェック側のみで検証する単一参照実装の例外、
# `linearmodels_ref.py`モジュールdoc参照）。cluster/dkは本実装の小標本補正が
# fixest・Stata型（`G/(G-1)·(n-1)/(n-K)`、t分布の自由度`G-1`/`T-1`）に変わり
# linearmodelsの`n/(n-extra_df-k)`とは一致しなくなったため、linearmodelsでは
# なくfixestのみで検証する（`generate_fe_crosscheck_fixtures.py`、
# `docs/spec/fe-spec.md`4章）。
COV_TYPES = ["classical", "hc1"]

# 組合せが欠ける不均衡パネル。1-wayも2-way（交互射影）も数値比較する。RE（2-wayは不均衡を拒否する）と
# plm（2-wayに非対応）の数値比較の対象でもある。
UNBALANCED_SCENARIOS = ["unbalanced"]

# 不均衡パネルのうちFE（linearmodels・fixest）だけで数値比較するシナリオ。疎なパネルと、個体と時点の
# つながりが2つの連結成分に分かれるパネル（`benchmark/panel/datasets.py`参照）。
UNBALANCED_FE_ONLY_SCENARIOS = ["unbalanced_sparse", "unbalanced_disconnected"]

# バランスパネル。1-way/2-way両方で数値比較する。
TWO_WAY_SCENARIOS = [
    "baseline",
    "small_panel",
    "heteroskedastic",
    "autocorrelated",
    "cross_sectionally_correlated",
    "moderate_multicollinearity",
    "high_condition_number",
    "scale_variance_mild",
    "many_regressors",
    "outlier_regressor",
    "high_variance",
]

NUMERIC_SCENARIOS = UNBALANCED_SCENARIOS + TWO_WAY_SCENARIOS

# FEの数値比較（linearmodels・fixest）の対象。全て1-way・2-wayの両方で比べる。
FE_SCENARIOS = NUMERIC_SCENARIOS + UNBALANCED_FE_ONLY_SCENARIOS

# singleton_entity/singleton_time/zero_variance_regressorはここに含まない（いずれも
# ValidationErrorの発生確認のみ、testing-policy.md「テストの3系統」参照。テストコード側で対応）。
# unbalanced_two_way（先頭1行だけ欠ける）は数値比較ではなく、2-wayで推定できる成功パスの確認用。

# 実データ（Wooldridge wagepan）。
WAGEPAN_COV_TYPES = ["classical", "hc1"]


# many_regressorsのみx1..x20（他シナリオはx1, x2の既定）。
SCENARIO_X_COLS: dict[str, list[str]] = {
    "many_regressors": [f"x{i}" for i in range(1, 21)],
}


def _run_effects(scenario: str, cov_type: str, *, two_way: bool) -> dict:
    # `time_col`は常に実在の"time"列を渡す（`two_way`とは独立、
    # `linearmodels_ref.py`モジュールdoc参照）。1-way + dkで観測順ダミーを
    # 使うと不均衡パネル（unbalancedシナリオ）でバンド幅・カーネル計算が
    # 不正確になるため（実測で発覚）。
    return run(
        scenario,
        SCENARIO_X_COLS.get(scenario, ["x1", "x2"]),
        cov_type,
        time_col="time",
        two_way=two_way,
        dataset_source="synthetic",
    )


def _run_boundary_df1_case(*, two_way: bool) -> dict:
    """df_resid=1境界の成功パス（1-way/2-wayでそれぞれ別データ、
    `benchmark/panel/freeze.py`参照）。"""
    scenario = "baseline_df1_two_way" if two_way else "baseline_df1_one_way"
    x_cols = ["x1", "x2", "x3"] if two_way else ["x1", "x2"]
    return run(
        scenario,
        x_cols,
        "classical",
        time_col="time",
        two_way=two_way,
        dataset_source="synthetic",
    )


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in FE_SCENARIOS:
        fixtures[scenario] = {
            "one_way": {
                cov_type: _run_effects(scenario, cov_type, two_way=False)
                for cov_type in COV_TYPES
            },
            "two_way": {
                cov_type: _run_effects(scenario, cov_type, two_way=True)
                for cov_type in COV_TYPES
            },
        }

    fixtures["baseline_df1"] = {
        "one_way": _run_boundary_df1_case(two_way=False),
        "two_way": _run_boundary_df1_case(two_way=True),
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
            "singleton_entity/singleton_time/zero_variance_regressorシナリオは"
            "ここに含まない（いずれもValidationErrorの発生確認のみ、"
            "テストコード側で対応）。"
            "unbalanced/unbalanced_sparse/unbalanced_disconnectedは組合せが"
            "欠ける不均衡パネルで、2-wayは交互射影で推定する"
            "（unbalanced_disconnectedは個体と時点のつながりが2つの連結成分に"
            "分かれる。自由度はlinearmodelsもk+N+T-1のまま）。"
            "hc2/hc3はlinearmodels.PanelOLSが提供しないため対象外"
            "（generate_fe_crosscheck_fixtures.pyのfixestクロスチェックのみで"
            "検証する単一参照実装の例外、panel-common.md5.4節と同型）。"
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
            "cluster/dkはここに含まない（本実装の小標本補正がfixest・Stata型に"
            "変わりlinearmodelsとは一致しないため、fixestクロスチェックのみで"
            "検証する、モジュールdoc・COV_TYPESのコメント参照）。"
            "moderate_multicollinearity/high_condition_number/"
            "scale_variance_mild/many_regressors/"
            "outlier_regressor/high_varianceは悪条件・"
            "高次元・外れ値シナリオ（`benchmark/panel/datasets.py`参照）。"
            "baseline_df1は自由度ちょうど1の境界成功パス。1-way"
            "（entity=3×period=2、k=2、df_resid=6-3-2=1）と2-way"
            "（entity=3×period=3、k=3、df_resid=9-(3+3+3-1)=1）で別データ"
            "（fe_baseline_df1_one_way.csv/fe_baseline_df1_two_way.csv）。"
        ),
    }
    return group_by_dataset_source(fixtures, ["wagepan"])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "fe.json",
        description=__doc__,
    )
