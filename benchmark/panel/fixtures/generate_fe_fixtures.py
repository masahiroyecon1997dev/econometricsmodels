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
import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.panel.references.linearmodels_ref import run

# hc2/hc3はlinearmodels.PanelOLSが提供しないため対象外（fixestクロスチェック
# 側のみで検証する単一参照実装の例外、`linearmodels_ref.py`モジュールdoc
# 参照）。hacはcross_sectionally_correlatedシナリオが本来の目的（他シナリオ
# でも動くことの確認はできるが統計的な意味は薄い、OLSのHACと同じ扱い）。
COV_TYPES = ["classical", "hc1", "cluster", "hac"]

# many_regressorsはk=20・列ごとに0.1〜100倍のスケール差を持つ高次元シナリオ。
# baseline既定のn_periods=6（Driscoll-Kraay HACがfixestドキュメント推奨の
# 「20時点以上」を大きく下回る）との組み合わせで、HACの傾き係数共分散部分
# 行列が数値的に特異になりComputationErrorになることを実測確認済み（ユーザー
# 確認済み、hacをこのシナリオのcov_type検証から除外する）。
COV_TYPES_NO_HAC = ["classical", "hc1", "cluster"]

# unbalancedは1-way専用（`fe-spec.md`3.1節: 2-way FEはバランスパネル必須、2-way要求時の
# ValidationErrorはunbalanced_two_wayシナリオで別途確認する。数値比較対象外）。
ONE_WAY_ONLY_SCENARIOS = ["unbalanced"]

# それ以外はバランスパネルのため1-way/2-way両方で数値比較する。
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

NUMERIC_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS

# シナリオ別のcov_type一覧（既定はCOV_TYPES、many_regressorsのみ上記の理由でhacを除く）。
SCENARIO_COV_TYPES: dict[str, list[str]] = {
    "many_regressors": COV_TYPES_NO_HAC,
}

# singleton_entity/singleton_time/unbalanced_two_way/zero_variance_regressorは
# ここに含まない（いずれもValidationErrorの発生確認のみ、testing-policy.md
# 「テストの3系統」参照。テストコード側で対応）。

# 実データ（Wooldridge wagepan）。T=8年と短くDriscoll-Kraay HACの前提
# （fixestドキュメントが20時点以上を推奨）を満たさないため対象外
# （hacの数値照合は合成データのcross_sectionally_correlatedで十分カバーする）。
WAGEPAN_COV_TYPES = ["classical", "hc1", "cluster"]


# many_regressorsのみx1..x20（他シナリオはx1, x2の既定）。
SCENARIO_X_COLS: dict[str, list[str]] = {
    "many_regressors": [f"x{i}" for i in range(1, 21)],
}


def _run_effects(scenario: str, cov_type: str, *, two_way: bool) -> dict:
    # `time_col`は常に実在の"time"列を渡す（`two_way`とは独立、
    # `linearmodels_ref.py`モジュールdoc参照）。1-way + hacで観測順ダミーを
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


def _run_cluster_imbalanced_case() -> dict:
    """クラスター不均衡シナリオ（`fe_baseline_cluster_imbalanced.csv`、
    entity=20×period=10のn=200データ）の数値照合。

    entityとは無関係な専用クラスター列（サイズ[2,3,5,10,30,50]のタイル、
    `testing-policy.md`「テスト用データセット」3.）をCSVには含めず、ここで
    都度動的生成する（`benchmark/linear/fixtures/generate_ols_fixtures.py`の
    `_run_cluster_case`と同じ方針）。`entity_nested_within_cluster=false`側の
    `extra_df`分岐（`engine/src/panel/CLAUDE.md`「cov_type対応」参照）を
    意図的に踏む。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline_cluster_imbalanced.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    return run(
        "baseline_cluster_imbalanced",
        ["x1", "x2"],
        "cluster",
        entity_col="entity",
        time_col="time",
        two_way=False,
        cluster_col="cluster_group",
        dataset_source="synthetic",
        df_override=df,
    )


def _run_cluster_g2_case() -> dict:
    """クラスタ数境界（G=2、q=1でG>q）の成功パス確認用
    （`fe_baseline_k1.csv`、k=1に絞ったbaseline、`benchmark/panel/freeze.py`
    参照）。`test_ols_reference.py::test_cluster_g2_matches_statsmodels`と
    同型: 説明変数1個（q=1）に絞ることで、既定のG=40（entity）ではなく
    entityとは無関係な2グループ（G=2）でもロバストWald検定のq×q部分行列が
    特異にならない（`G<=q`ならValidationError、testing-policy.md
    「グループ数が境界値に近いケース」参照）。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline_k1.csv")
    groups = [str(i % 2) for i in range(df.height)]
    df = df.with_columns(pl.Series("cluster_group", groups))
    return run(
        "baseline_k1",
        ["x1"],
        "cluster",
        entity_col="entity",
        time_col="time",
        two_way=False,
        cluster_col="cluster_group",
        dataset_source="synthetic",
        df_override=df,
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

    for scenario in NUMERIC_SCENARIOS:
        cov_types = SCENARIO_COV_TYPES.get(scenario, COV_TYPES)
        fixtures[scenario] = {"one_way": {}}
        for cov_type in cov_types:
            fixtures[scenario]["one_way"][cov_type] = _run_effects(
                scenario, cov_type, two_way=False
            )
        if scenario in TWO_WAY_SCENARIOS:
            fixtures[scenario]["two_way"] = {
                cov_type: _run_effects(scenario, cov_type, two_way=True)
                for cov_type in cov_types
            }

    fixtures["baseline"]["cluster_imbalanced"] = _run_cluster_imbalanced_case()
    fixtures["baseline"]["cluster_g2"] = _run_cluster_g2_case()
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
            "singleton_entity/singleton_time/unbalanced_two_way/"
            "zero_variance_regressorシナリオはここに含まない"
            "（いずれもValidationErrorの発生確認のみ、テストコード側で対応）。"
            "unbalancedは1-way専用（2-way FEはバランスパネル必須のため、"
            "2-way要求時のValidationErrorはunbalanced_two_wayシナリオで確認）。"
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
            "hacはwagepan（T=8）には適用しない"
            "（Driscoll-Kraay HACはfixestドキュメントが20時点以上を推奨する"
            "ほど時点数に依存するため、合成データのcross_sectionally_"
            "correlatedシナリオ（T=25）でのみ数値照合する）。"
            "moderate_multicollinearity/high_condition_number/"
            "scale_variance_mild/many_regressors/"
            "outlier_regressor/high_varianceは悪条件・"
            "高次元・外れ値シナリオ（`benchmark/panel/datasets.py`参照）。"
            "many_regressorsのみn_periods=6（baseline既定）とk=20の組み合わせで"
            "Driscoll-Kraay HACの傾き係数共分散部分行列が数値的に特異になり"
            "ComputationErrorになるため、cov_type検証からhacを除外している"
            "（SCENARIO_COV_TYPES参照、ユーザー確認済み）。"
            "baseline.cluster_imbalancedはentityとは無関係な専用クラスター列"
            "（サイズ[2,3,5,10,30,50]のタイル）での数値照合。"
            "fe_baseline_cluster_imbalanced.csv（entity=20×period=10のn=200、"
            "imbalanced_cluster_groupsが100の倍数のnを要求するためbaseline本体"
            "〔n=240〕とは別データ）を使い、クラスター列自体はCSVに含めず"
            "都度動的生成する（generate_ols_fixtures.pyの_run_cluster_caseと"
            "同じ方針）。"
            "baseline.cluster_g2はクラスタ数境界（G=2、q=1でG>q）の成功パス。"
            "fe_baseline_k1.csv（k=1に絞ったbaseline）にentityとは無関係な"
            "2グループ（行番号%2）を都度動的付与する"
            "（generate_ols_fixtures.pyのcluster_g2ケースと同じ方針）。"
            "baseline_df1は自由度ちょうど1の境界成功パス。1-way"
            "（entity=3×period=2、k=2、df_resid=6-3-2=1）と2-way"
            "（entity=3×period=3、k=3、df_resid=9-(3+3+3-1)=1）で別データ"
            "（fe_baseline_df1_one_way.csv/fe_baseline_df1_two_way.csv）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "fe.json", description=__doc__
    )
