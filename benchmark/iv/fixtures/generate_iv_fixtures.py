"""IV（2SLS）のテストフィクスチャ（tests/api_tests/fixtures/benchmarks/iv.json）を
生成するスクリプト。

`benchmark/iv/run_linearmodels_benchmark.py`（1回呼べば1ケース分の結果を返す汎用
ツール）を全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つのJSONにまとめて
書き出す。GMMは`method="gmm"`がまだPython側に配線されていないため対象外
（`run_linearmodels_benchmark.py`のモジュールdocstring参照）。

このスクリプト自体は`benchmark/`側に置く。生成される`iv.json`は
`tests/api_tests/fixtures/benchmarks/`に置く（両者を分ける理由は
`.claude/skills/reference-benchmark/SKILL.md`参照）。

入力データは`tests/api_tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/freeze_datasets.py`参照）。

使用例:
    python generate_iv_fixtures.py --output ../../../tests/api_tests/fixtures/benchmarks/iv.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # benchmark/iv/ を import path に追加（run_linearmodels_benchmark）
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # benchmark/ を import path に追加（_common）

import linearmodels  # noqa: E402
import polars as pl  # noqa: E402

from _common import DATA_DIR, imbalanced_cluster_groups  # noqa: E402
from run_linearmodels_benchmark import run  # noqa: E402

# 丁度識別・過剰識別を問わずx_exog=['x1'], x_endog=['endog1'], instruments=[...]の
# 構造で数値比較できるシナリオ。perfect_multicollinearityはComputationErrorの発生
# 確認のみ（testing-policy.md「テストの3系統」）のためここに含めない。
NUMERIC_SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "small_n",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
]

# just_identifiedのみinstruments=['z1']（k_instruments == k_endogに強制される、
# generate_iv_datasets.pyのdocstring参照）。他は全シナリオ共通でinstruments=2本。
INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}

# moderate_multicollinearity/high_condition_numberはk_exog=2（x1, x2）で固定済み
# （freeze_datasets.pyのIV_K_EXOG_OVERRIDES参照）。
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}

COV_TYPES = ["classical", "hc0", "hc1", "hac", "cluster"]


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
        instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])

        fixtures[scenario] = {}
        for cov_type in COV_TYPES:
            if cov_type == "cluster":
                continue  # baselineのみ別途複数パターンで確認（下記）
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
            # G=2境界の成功パスは、実装時に`econometricsmodels.IV`で実際に試したところ
            # `TwoSlsEstimator`の第一段階回帰（`x_exog=[]`・instruments=1本・G=2の
            # クラスターロバストF検定）が`ComputationError`（"near-singular"）を返すことが
            # 判明した。同一のデータ・モデル形状（`endog1 ~ const + z1`、G=2クラスター）を
            # 素のOLS（`econometricsmodels.OLS`）で単独fitすると成功する
            # （F統計量が計算できる）ため、IVの第一段階呼び出し経路に固有の問題である
            # 可能性が高い。原因未特定のため、ここではフィクスチャ化を見送り、
            # `iv_baseline_g2.csv`（`freeze_datasets.py`のIV_G2_BOUNDARY_SCENARIOS）は
            # 凍結済みのまま残し、原因究明後に追加する（ユーザー確認済み、
            # `engine/src/iv/CLAUDE.md`に再現手順を記録）。

    fixtures["_meta"] = {
        "method": "2sls",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "note": (
            "hc2/hc3はlinearmodelsに対応する実装が無いため対象外（`iv-api-design.md`"
            "3.1節、`run_linearmodels_benchmark.py`のモジュールdocstring参照）。"
            "GMMは`method='gmm'`がまだPython側に配線されていないため対象外。"
            "wu_hausman_statisticはclassical/hc0/hc1/clusterで`res.wooldridge_"
            "regression`をqで割った値を基準とし機械精度で一致する（実測確認済み、"
            "`run_linearmodels_benchmark.py`のモジュールdocstring参照）。hacのみ"
            "wooldridge_regressionでも一致しないため`None`（原因未特定、"
            "R ivreg クロスチェック実装時に別途確認、ユーザー確認済み）。"
            "weak_instrument_f_linearmodels（linearmodelsのfirst_stage.diagnosticsの"
            "f.stat、常にclassical/debiased=Trueで再fitして計算）と"
            "weak_instrument_f_independent（本スクリプトでSSRベースのnested F検定を"
            "独立計算した値）の両方を含む。実測では機械精度で一致することを確認済み"
            "（独立計算自体のミスパターン検出用、ユーザー確認済み）。"
            "perfect_multicollinearityはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "cluster_g2（G=2境界の成功パス）は、実装時にIVの第一段階回帰で"
            "ComputationErrorが再現したため今回は含めない（本スクリプト内の"
            "コメント・`engine/src/iv/CLAUDE.md`参照、原因究明後に追加予定）。"
        ),
    }
    return fixtures


def _run_cluster_case(dataset: str, groups: list | None = None) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて`run_linearmodels_benchmark`
    を呼ぶ（`generate_ols_fixtures.py`の`_run_cluster_case`と同じ発想）。
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../../../tests/api_tests/fixtures/benchmarks/iv.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
