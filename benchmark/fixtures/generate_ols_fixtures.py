"""OLSのテストフィクスチャ（tests/api_tests/fixtures/ols.json）を生成するスクリプト。

`benchmark/run_statsmodels_benchmark.py`（1回呼べば1ケース分の結果を返す汎用ツール）を
全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つのJSONにまとめて書き出す。

このスクリプト自体は`benchmark/`側に置く（ベンチマーク生成ツールの一部）。
生成される`ols.json`は`tests/api_tests/fixtures/`に置く（テストが読むデータ）。
両者を分けている理由は`.claude/skills/reference-benchmark/SKILL.md`参照。

使用例:
    python fixtures/generate_ols_fixtures.py --output ../tests/api_tests/fixtures/benchmarks/ols.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark/ を import path に追加

import statsmodels  # noqa: E402

from run_statsmodels_benchmark import run  # noqa: E402

# 完全な多重共線性は数値比較の対象外（testing-policy.md「テストの3系統」参照）。
# ComputationErrorが発生することのみをテストコード側で確認する。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
]

# classical/HC系は全シナリオで確認。HACはautocorrelatedシナリオが本来の目的
# （他のシナリオでも動くことの確認はできるが、統計的な意味は薄い）。
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {}
        for cov_type in COV_TYPES:
            result = run(
                dataset_source="synthetic",
                dataset=scenario,
                formula=None,
                cov_type=cov_type,
            )
            fixtures[scenario][cov_type] = result

        # クラスターロバストSEは、シナリオ依存ではなくグルーピングの動作確認が目的のため、
        # baselineシナリオでのみ、決め打ちの疑似グループ（10グループ）を使って確認する。
        # 実際のクラスター構造を統計的に検証するものではない。
        if scenario == "baseline":
            fixtures[scenario]["cluster"] = _run_cluster_case(scenario)

    fixtures["_meta"] = {
        "method": "ols",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "クロスチェック用のRベンチマークは別途 benchmark/run_r_benchmark.R で"
            "生成し、緩い許容誤差で比較する想定（未検証）。"
        ),
    }
    return fixtures


def _run_cluster_case(scenario: str) -> dict:
    """クラスターロバストSE確認用に、決め打ちの疑似グループを付けて実行する。"""
    import statsmodels.formula.api as smf

    from generate_synthetic_datasets import generate_dataset

    df, _ = generate_dataset(scenario)
    pandas_df = df.to_pandas()
    pandas_df["_group"] = [i % 10 for i in range(len(pandas_df))]

    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    formula = "y ~ " + " + ".join(x_cols)

    model = smf.ols(formula=formula, data=pandas_df).fit(
        cov_type="cluster", cov_kwds={"groups": pandas_df["_group"]}
    )

    return {
        "coef": {str(k): float(v) for k, v in model.params.to_dict().items()},
        "se": {str(k): float(v) for k, v in model.bse.to_dict().items()},
        "_meta": {
            "reference": "statsmodels",
            "statsmodels_version": statsmodels.__version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "決め打ちの疑似グループ（行番号%10）。統計的な意味はなく、実装の動作確認用。",
            "formula": formula,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="../tests/api_tests/fixtures/benchmarks/ols.json")
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
