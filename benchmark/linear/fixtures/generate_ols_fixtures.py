"""OLSのテストフィクスチャ（tests/api_tests/fixtures/ols.json）を生成するスクリプト。

`benchmark/linear/run_statsmodels_benchmark.py`（1回呼べば1ケース分の結果を返す汎用ツール）を
全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つのJSONにまとめて書き出す。

このスクリプト自体は`benchmark/`側に置く（ベンチマーク生成ツールの一部）。
生成される`ols.json`は`tests/api_tests/fixtures/`に置く（テストが読むデータ）。
両者を分けている理由は`.claude/skills/reference-benchmark/SKILL.md`参照。

入力データは`tests/api_tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/freeze_datasets.py`参照）。`imbalanced_cluster_groups`（純粋にnから
決定論的にラベルを組み立てるだけで乱数を使わない）のみ、引き続き
`generate_synthetic_datasets.py`を直接呼ぶ。

使用例:
    python generate_ols_fixtures.py --output ../../../tests/api_tests/fixtures/benchmarks/ols.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # benchmark/linear/ を import path に追加（run_statsmodels_benchmark）
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # benchmark/ を import path に追加（generate_synthetic_datasets）

import polars as pl  # noqa: E402
import statsmodels  # noqa: E402

from generate_synthetic_datasets import imbalanced_cluster_groups  # noqa: E402
from run_statsmodels_benchmark import DATA_DIR, run  # noqa: E402

# 完全な多重共線性は数値比較の対象外（testing-policy.md「テストの3系統」参照）。
# ComputationErrorが発生することのみをテストコード側で対応する。
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
        # baselineシナリオでのみ、複数のグルーピングパターンで確認する
        # （testing-policy.md「テスト用データセット」3.、Issue #100）。
        # 実際のクラスター構造を統計的に検証するものではない。
        if scenario == "baseline":
            n = pl.read_csv(DATA_DIR / "synthetic_baseline.csv").height
            fixtures[scenario]["cluster"] = _run_cluster_case()
            fixtures[scenario]["cluster_imbalanced"] = _run_cluster_case(
                groups=imbalanced_cluster_groups(n),
                note="不均衡な疑似グループ（サイズ[2,3,5,10,30,50]のタイル）。Issue #100。",
            )
            # G=2×説明変数3個（既定のbaseline）はロバストWald検定の共分散
            # 部分行列（3x3）のランクがG=2以下になり必然的に特異になるため
            # ComputationError（成功パスではない、test_ols_fixtures.py
            # 側でエラーパスとして確認）。ここでの「G=2境界の成功パス」は
            # 説明変数1個（q=1、Wald検定の部分行列が1x1）に絞って確認する。
            n_g2 = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv").height
            fixtures[scenario]["cluster_g2"] = _run_cluster_case(
                groups=[str(i % 2) for i in range(n_g2)],
                note="クラスタ数境界（G=2ちょうど）の成功パス確認用。"
                "説明変数1個（q=1）に絞っている（Issue #100、"
                "3個だとロバストWald検定の共分散行列が特異になりComputationError）。",
                k1=True,
            )

    fixtures["_meta"] = {
        "method": "ols",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "クロスチェック用のRベンチマークは別途 "
            "benchmark/linear/run_lm_crosscheck_benchmark.R で生成する。"
        ),
    }
    return fixtures


def _run_cluster_case(
    groups: list | None = None,
    note: str = "決め打ちの疑似グループ（行番号%10）。統計的な意味はなく、実装の動作確認用。",
    k1: bool = False,
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて実行する。

    Args:
        groups: 各行のグループラベル。Noneなら既定（行番号%10、10均等グループ）。
        note: フィクスチャの`_meta.note`に記録する説明文。
        k1: TrueならG=2境界ケース用の説明変数1個版（synthetic_baseline_k1.csv）を使う。
    """
    import statsmodels.formula.api as smf

    filename = "synthetic_baseline_k1.csv" if k1 else "synthetic_baseline.csv"
    df = pl.read_csv(DATA_DIR / filename)
    pandas_df = df.to_pandas()
    pandas_df["_group"] = (
        groups
        if groups is not None
        else [i % 10 for i in range(len(pandas_df))]
    )

    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    formula = "y ~ " + " + ".join(x_cols)

    model = smf.ols(formula=formula, data=pandas_df).fit(
        cov_type="cluster", cov_kwds={"groups": pandas_df["_group"]}
    )

    return {
        "coef": {
            str(name): float(v) for name, v in model.params.to_dict().items()
        },
        "se": {str(name): float(v) for name, v in model.bse.to_dict().items()},
        "_meta": {
            "reference": "statsmodels",
            "statsmodels_version": statsmodels.__version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "formula": formula,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../../../tests/api_tests/fixtures/benchmarks/ols.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
