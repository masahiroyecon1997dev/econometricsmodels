"""IV（2SLS）のテストフィクスチャ（tests/fixtures/benchmarks/iv.json）を
生成するスクリプト。

`benchmark/iv/references/linearmodels_ref.py`（1回呼べば1ケース分の結果を返す
汎用アダプタ）を全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つのJSONに
まとめて書き出す。GMMは`method="gmm"`がまだPython側に配線されていないため対象外
（`linearmodels_ref.py`のモジュールdocstring参照）。

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
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.iv.references.linearmodels_ref import run

# 丁度識別・過剰識別を問わずx_exog=['x1'], x_endog=['endog1'], instruments=[...]の
# 構造で数値比較できるシナリオ。perfect_multicollinearityはComputationErrorの発生
# 確認のみ（testing-policy.md「テストの3系統」）のためここに含めない。
NUMERIC_SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
]

# just_identifiedのみinstruments=['z1']（k_instruments == k_endogに強制される、
# benchmark/iv/datasets.pyのdocstring参照）。他は全シナリオ共通でinstruments=2本。
INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}

# moderate_multicollinearity/high_condition_numberはk_exog=2（x1, x2）で固定済み
# （benchmark/iv/freeze.pyのIV_K_EXOG_OVERRIDES参照）。
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}

# clusterはbaselineのみ、下のcluster専用ケース（_run_cluster_case/
# _run_cluster_g2_case）で個別に扱うためここには含めない（OLS/WLSと同じ書き方、
# 項目14参照）。multi_endog/card/df1がclusterを持たない理由は_metaのnote参照。
COV_TYPES = ["classical", "hc0", "hc1", "hac"]

# Wooldridge card（Card 1995の教育収益率推定、大学近接ダミーnearc2/nearc4を
# 操作変数として教育年数educの内生性を補正する教科書的定番例）。
CARD_X_EXOG = ["exper", "expersq", "black", "smsa", "south"]


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
        instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])

        fixtures[scenario] = {}
        for cov_type in COV_TYPES:
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
            # G=2境界の成功パス。実装時に`econometricsmodels.IV`で実際に試したところ
            # `TwoSlsEstimator`の第一段階回帰（`x_exog=[]`・instruments=1本・G=2の
            # クラスターロバストF検定）が`ComputationError`（"near-singular"）を
            # 返すバグが発覚したが、原因（`without_baked_in_intercept`未導入による
            # `k_constant`取り違え）は判明・修正済み（`engine/src/iv/CLAUDE.md`
            # 「修正済み」参照）。修正後にフィクスチャ化した。

    # 複数内生変数（k_endog>=2）の成功パス確認。x_exog=['x1']・
    # x_endog=['endog1', 'endog2']・instruments=['z1','z2','z3']（過剰識別）。
    fixtures["multi_endog"] = {}
    for cov_type in COV_TYPES:
        fixtures["multi_endog"][cov_type] = run(
            dataset="baseline_multi_endog",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1", "endog2"],
            instrument_cols=["z1", "z2", "z3"],
            cov_type=cov_type,
        )

    # 実データセット（Wooldridge card、Card 1995の大学近接操作変数による教育の
    # 収益率推定）。testing-policy.md「テスト用データセット」2.（実データセット）の
    # 要求に対しIV系統は未対応だった。
    fixtures["card"] = {}
    for cov_type in COV_TYPES:
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
    for cov_type in COV_TYPES:
        fixtures["df1"][cov_type] = run(
            dataset="baseline_df1",
            x_exog_cols=[],
            x_endog_cols=["endog1"],
            instrument_cols=["z1"],
            cov_type=cov_type,
        )

    fixtures["_meta"] = {
        "method": "2sls",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "note": (
            "hc2/hc3はlinearmodelsに対応する実装が無いため対象外（`iv-api-design.md`"
            "3.1節、`benchmark/iv/references/linearmodels_ref.py`のモジュール"
            "docstring参照）。"
            "GMMは`method='gmm'`がまだPython側に配線されていないため対象外。"
            "wu_hausman_statisticはclassical/hc0/hc1/clusterで`res.wooldridge_"
            "regression`をqで割った値を基準とし機械精度で一致する（実測確認済み、"
            "`benchmark/iv/references/linearmodels_ref.py`のモジュールdocstring"
            "参照）。hacのみ"
            "wooldridge_regressionでも一致しないため`None`（原因未特定、"
            "R ivreg クロスチェック実装時に別途確認、ユーザー確認済み）。"
            "weak_instrument_f_linearmodels（linearmodelsのfirst_stage.diagnosticsの"
            "f.stat、常にclassical/debiased=Trueで再fitして計算）と"
            "weak_instrument_f_independent（本スクリプトでSSRベースのnested F検定を"
            "独立計算した値）の両方を含む。実測では機械精度で一致することを確認済み"
            "（独立計算自体のミスパターン検出用、ユーザー確認済み）。"
            "perfect_multicollinearityはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "cluster_g2（G=2境界の成功パス）は、`engine/src/iv/CLAUDE.md`"
            "「修正済み」に記録の`k_constant`取り違えバグの修正後にフィクスチャ化"
            "した。"
            "multi_endog（複数内生変数、x_endog=['endog1','endog2']）は、"
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
        ),
    }
    return fixtures


def _run_cluster_case(dataset: str, groups: list | None = None) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて
    `benchmark/iv/references/linearmodels_ref.py` を呼ぶ
    （`generate_ols_fixtures.py`の`_run_cluster_case`と同じ発想）。
    """
    filename = f"iv_{dataset}.csv"
    df = pl.read_csv(DATA_DIR / filename)
    n = df.height
    cluster_group = (
        groups if groups is not None else [i % 10 for i in range(n)]
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = DATA_DIR / f"iv_{dataset}_cluster_tmp.csv"
    grouped.write_csv(tmp_path)
    try:
        return run(
            dataset=f"{dataset}_cluster_tmp",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1"],
            instrument_cols=["z1", "z2"],
            cov_type="cluster",
            cluster_col="cluster_group",
        )
    finally:
        tmp_path.unlink()


def _run_cluster_g2_case() -> dict:
    """G=2境界の成功パス確認用（`tests/iv/test_iv.py::test_cluster_g2_boundary_
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
            cluster_col="cluster_group",
        )
    finally:
        tmp_path.unlink()


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "iv.json", description=__doc__
    )
