"""IV（2SLS）のテストフィクスチャ（tests/fixtures/benchmarks/iv.json）を
生成するスクリプト。

`benchmark/iv/references/linearmodels_ref.py`（1回呼べば1ケース分の結果を返す
汎用アダプタ）を全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つのJSONに
まとめて書き出す。GMM（`estimator="gmm"`）は`iv_gmm.json`に別途生成する
（`generate_iv_gmm_fixtures.py`）ため、ここでは対象外。

このスクリプト自体は`benchmark/`側に置く。生成される`iv.json`は
`tests/fixtures/benchmarks/`に置く（両者を分ける理由は
`.claude/skills/reference-benchmark/SKILL.md`参照）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/iv/freeze.py`参照）。

使用例（リポジトリルートから）:
    python -m benchmark.iv.fixtures.generate_iv_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import linearmodels
import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    balanced_cluster_groups,
    group_by_dataset_source,
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.iv.constants import (
    CARD_X_EXOG,
    CONFIDENCE_LEVEL_NON_DEFAULT,
    LINEARMODELS_COV_TYPES,
)
from benchmark.iv.datasets import (
    MULTI_ENDOG_SCENARIOS,
    NUMERIC_SCENARIOS,
    multi_endog_instruments,
    scenario_instruments,
    scenario_x_exog,
)
from benchmark.iv.references.linearmodels_ref import run

# include_intercept=False（切片なし）のシナリオ。キーが`fixtures["no_intercept"]`
# 直下のシナリオ名。generate_iv_gmm_fixtures.py・クロスチェック側
# （generate_iv_crosscheck_fixtures.py等）もここをimportして単一の定義元にする。
# x_exog_emptyは`x_exog=[]`かつ切片なしで、弱操作変数診断の制限モデルが
# 回帰変数0個になる退化ケース（`engine/src/iv/CLAUDE.md`参照）。
NO_INTERCEPT_SCENARIOS: dict[str, dict] = {
    "baseline": {
        "dataset": "baseline",
        "x_exog": ["x1"],
        "x_endog": ["endog1"],
        "instruments": ["z1", "z2"],
    },
    "just_identified": {
        "dataset": "just_identified",
        "x_exog": ["x1"],
        "x_endog": ["endog1"],
        "instruments": ["z1"],
    },
    "x_exog_empty": {
        "dataset": "baseline",
        "x_exog": [],
        "x_endog": ["endog1"],
        "instruments": ["z1", "z2"],
    },
    "multi_endog": {
        "dataset": "baseline_multi_endog",
        "x_exog": ["x1"],
        "x_endog": ["endog1", "endog2"],
        "instruments": ["z1", "z2", "z3"],
    },
}


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = scenario_x_exog(scenario)
        instruments = scenario_instruments(scenario)

        fixtures[scenario] = {}
        for cov_type in LINEARMODELS_COV_TYPES:
            result = run(
                dataset=scenario,
                x_exog_cols=x_exog,
                x_endog_cols=["endog1"],
                instrument_cols=instruments,
                cov_type=cov_type,
            )
            fixtures[scenario][cov_type] = result

        if scenario == "baseline":
            n = pl.read_csv(DATA_DIR / "iv_baseline.csv").height
            fixtures[scenario]["cluster"] = _run_cluster_case("baseline")
            fixtures[scenario]["cluster_imbalanced"] = _run_cluster_case(
                "baseline",
                groups=imbalanced_cluster_groups(n),
            )
            fixtures[scenario]["cluster_g2"] = _run_cluster_g2_case()

            # confidence_level非既定（0.95以外）。信頼区間の臨界値はcov_typeごとに
            # 自由度が異なる（clusterはt(G-1)）ため、全cov_typeとclusterで持つ。
            fixtures[scenario]["confidence_level"] = {
                cov_type: run(
                    dataset=scenario,
                    x_exog_cols=x_exog,
                    x_endog_cols=["endog1"],
                    instrument_cols=instruments,
                    cov_type=cov_type,
                    confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
                )
                for cov_type in LINEARMODELS_COV_TYPES
            }
            fixtures[scenario]["confidence_level"]["cluster"] = (
                _run_cluster_case(
                    "baseline",
                    confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
                )
            )
            # G=2境界の成功パス。実装時に`econometricsmodels.IV`で実際に試したところ
            # `TwoSlsEstimator`の第一段階回帰（`x_exog=[]`・instruments=1本・G=2の
            # クラスターロバストF検定）が`ComputationError`（"near-singular"）を
            # 返すバグが発覚したが、原因（`without_baked_in_intercept`未導入による
            # `k_constant`取り違え）は判明・修正済み（`engine/src/iv/CLAUDE.md`
            # 「修正済み」参照）。修正後にフィクスチャ化した。

    # 複数内生変数（k_endog>=2）の成功パス確認。fixtures["multi_endog"][シナリオ]
    # [cov_type]。x_exog=['x1']・x_endog=['endog1', 'endog2']・instruments=
    # ['z1','z2','z3']（過剰識別。just_identifiedのみ['z1','z2']の丁度識別）。
    fixtures["multi_endog"] = {}
    for scenario in MULTI_ENDOG_SCENARIOS:
        fixtures["multi_endog"][scenario] = {}
        for cov_type in LINEARMODELS_COV_TYPES:
            fixtures["multi_endog"][scenario][cov_type] = run(
                dataset=f"{scenario}_multi_endog",
                x_exog_cols=["x1"],
                x_endog_cols=["endog1", "endog2"],
                instrument_cols=multi_endog_instruments(scenario),
                cov_type=cov_type,
            )
        if scenario == "baseline":
            # 複数内生変数×クラスターロバストSE。Wu-HausmanのF分布の自由度は
            # (n_endog, G-1)=(2, 9)になる（n_endog=1のbaselineでは走らない経路）。
            fixtures["multi_endog"][scenario]["cluster"] = _run_cluster_case(
                "baseline_multi_endog",
                x_endog_cols=["endog1", "endog2"],
                instrument_cols=["z1", "z2", "z3"],
            )

    # 実データセット（Wooldridge card、Card 1995の大学近接操作変数による教育の
    # 収益率推定）。testing-policy.md「テスト用データセット」2.（実データセット）の
    # 要求に対しIV系統は未対応だった。
    fixtures["card"] = {}
    for cov_type in LINEARMODELS_COV_TYPES:
        fixtures["card"][cov_type] = run(
            dataset="card",
            x_exog_cols=CARD_X_EXOG,
            x_endog_cols=["educ"],
            instrument_cols=["nearc2", "nearc4"],
            cov_type=cov_type,
            dataset_source="wooldridge",
            y_col="lwage",
        )

    # 自由度1境界（df_resid=1ちょうど）の成功パス確認。
    # x_exog=[]・x_endog=['endog1']・instruments=['z1']（丁度識別、n=3）。
    fixtures["df1"] = {}
    for cov_type in LINEARMODELS_COV_TYPES:
        fixtures["df1"][cov_type] = run(
            dataset="baseline_df1",
            x_exog_cols=[],
            x_endog_cols=["endog1"],
            instrument_cols=["z1"],
            cov_type=cov_type,
        )

    # include_intercept=False（切片なし）。fixtures["no_intercept"][シナリオ][cov_type]。
    fixtures["no_intercept"] = {
        name: {
            cov_type: run(
                dataset=spec["dataset"],
                x_exog_cols=spec["x_exog"],
                x_endog_cols=spec["x_endog"],
                instrument_cols=spec["instruments"],
                cov_type=cov_type,
                include_intercept=False,
            )
            for cov_type in LINEARMODELS_COV_TYPES
        }
        for name, spec in NO_INTERCEPT_SCENARIOS.items()
    }
    # 切片なし×クラスターロバストSE（baselineのみ、切片ありの`cluster`と同じ
    # 疑似グループ）。切片なしでは傾き係数の数qが`k`になり、小標本補正・自由度・
    # Wald検定の次元が切片ありと変わる。
    fixtures["no_intercept"]["baseline"]["cluster"] = _run_cluster_case(
        "baseline", include_intercept=False
    )

    fixtures["_meta"] = {
        "method": "2sls",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "note": (
            "hc2/hc3はlinearmodelsに対応する実装が無いため対象外（`iv-spec.md`"
            "3.1節、`benchmark/iv/references/linearmodels_ref.py`のモジュール"
            "docstring参照）。"
            "GMM（`estimator='gmm'`）は`iv_gmm.json`に別途生成するため対象外。"
            "wu_hausman_statisticはclassical/hc0/hc1/clusterで`res.wooldridge_"
            "regression`をqで割った値を基準とし、hacを含む全cov_typeで機械精度で一致する"
            "（実測確認済み、`benchmark/iv/references/linearmodels_ref.py`の"
            "モジュールdocstring参照）。"
            "weak_instrument_f_linearmodels（linearmodelsのfirst_stage.diagnosticsの"
            "f.stat、常にclassical/debiased=Trueで再fitして計算）と"
            "weak_instrument_f_independent（本スクリプトでSSRベースのnested F検定を"
            "独立計算した値）の両方を含む。実測では機械精度で一致することを確認済み"
            "（独立計算自体のミスパターン検出用）。"
            "perfect_multicollinearityはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "scale_variance_mildはscale_variance（x1*1e6, x2*1e-3、"
            "全cov_typeでComputationError）より緩いスケール差（x1*1e2, x2*1e-1）"
            "の成功パス（OLSの"
            "scale_variance_mildと同じ発想）。"
            "cluster_g2（G=2境界の成功パス）は、`engine/src/iv/CLAUDE.md`"
            "「修正済み」に記録の`k_constant`取り違えバグの修正後にフィクスチャ化"
            "した。"
            "baseline.confidence_levelはconfidence_level="
            f"{CONFIDENCE_LEVEL_NON_DEFAULT}（既定0.95以外）を全cov_type"
            "（classical/hc0/hc1/hac）とclusterで持つ。"
            "invalid_instrument（最後の操作変数を構造誤差と相関させ、Sargan検定が"
            "棄却する）・many_regressors（k_exog=20、列ごとに0.1〜100倍のスケール差）"
            "・outlier_regressor（x1の5%が外れ値）は他のシナリオと同じ構成。"
            "multi_endog（複数内生変数、x_endog=['endog1','endog2']）は"
            "MULTI_ENDOG_SCENARIOS（baseline・just_identified・weak_instruments・"
            "heteroskedastic）ごとに持つ。"
            "benchmark/iv/datasets.pyの第一段階誤差vが内生変数ごとに独立になる"
            "よう修正した後のデータで生成（修正前はv"
            "が全内生変数で単一列のため第一段階回帰残差が事実上完全共線になり、"
            "Wu-Hausman検定の拡張回帰が推定不能だった）。"
            "cardはWooldridge実データ（Card 1995、大学近接ダミーnearc2/nearc4を"
            "操作変数として教育年数educの内生性を補正する教科書的定番例）。"
            "他の実データセット（mroz等）と異なりtrue_betaと比較できないため"
            "`true_beta`キーは持たない。cluster cov_typeは対応する自然な"
            "カテゴリ列が無いため対象外。"
            "df1（自由度1境界、n=3・x_exog=[]・x_endog=['endog1']・"
            "instruments=['z1']）は境界値・悪条件シナリオの一環"
            "（testing-policy.md「テスト用データセット」）。cluster cov_typeは"
            "n=3では意味のあるクラスタ数を確保できないため対象外。"
            "no_intercept（`include_intercept=False`、式に定数項`1`を含めない）は"
            "baseline・just_identified・x_exog_empty（`x_exog=[]`かつ切片なし、"
            "弱操作変数診断の制限モデルが回帰変数0個の退化ケース）・multi_endogの"
            "4シナリオ×classical/hc0/hc1/hac（baselineのみ追加でcluster）。"
            "r_squared/adj_r_squaredは切片なしの"
            "非中心化の定義、f_statistic/f_p_valueは全係数が傾きとして検定される。"
        ),
    }
    return group_by_dataset_source(fixtures, ["card"])


def _run_cluster_case(
    dataset: str,
    groups: list | None = None,
    include_intercept: bool = True,
    x_endog_cols: list[str] | None = None,
    instrument_cols: list[str] | None = None,
    confidence_level: float = 0.95,
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて
    `benchmark/iv/references/linearmodels_ref.py` を呼ぶ
    （`generate_ols_fixtures.py`の`_run_cluster_case`と同じ発想）。
    `x_endog_cols`・`instrument_cols`の既定は`["endog1"]`・`["z1", "z2"]`。
    """
    filename = f"iv_{dataset}.csv"
    df = pl.read_csv(DATA_DIR / filename)
    n = df.height
    cluster_group = (
        groups if groups is not None else balanced_cluster_groups(n)
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = DATA_DIR / f"iv_{dataset}_cluster_tmp.csv"
    grouped.write_csv(tmp_path)
    try:
        return run(
            dataset=f"{dataset}_cluster_tmp",
            x_exog_cols=["x1"],
            x_endog_cols=x_endog_cols or ["endog1"],
            instrument_cols=instrument_cols or ["z1", "z2"],
            cov_type="cluster",
            cluster="cluster_group",
            include_intercept=include_intercept,
            confidence_level=confidence_level,
        )
    finally:
        tmp_path.unlink()


def _run_cluster_g2_case() -> dict:
    """G=2境界の成功パス確認用（`tests/iv/test_iv_api.py::test_cluster_g2_boundary_
    succeeds_when_x_exog_is_empty`と同じ再現条件: `x_exog=[]`・`instruments`1本
    ・行番号%2の疑似グループ、`engine/src/iv/CLAUDE.md`「修正済み」参照）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_g2.csv")
    n = df.height
    grouped = df.with_columns(
        pl.Series("cluster_group", [str(i % 2) for i in range(n)])
    )
    tmp_path = DATA_DIR / "iv_baseline_g2_cluster_tmp.csv"
    grouped.write_csv(tmp_path)
    try:
        return run(
            dataset="baseline_g2_cluster_tmp",
            x_exog_cols=[],
            x_endog_cols=["endog1"],
            instrument_cols=["z1"],
            cov_type="cluster",
            cluster="cluster_group",
        )
    finally:
        tmp_path.unlink()


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "iv.json", description=__doc__
    )
