"""Logitのテストフィクスチャ（tests/api_tests/fixtures/benchmarks/logit.json）を
生成するスクリプト。

`benchmark/nonlinear/run_statsmodels_benchmark.py`（1回呼べば1ケース分の結果を返す
汎用ツール）を全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つのJSONに
まとめて書き出す。`benchmark/linear/fixtures/generate_ols_fixtures.py`と同型の設計。

**`cov_type="hc1"`はここに含めない**（statsmodelsのdiscrete modelがn/(n-k)小標本補正を
実装しておらずHC0と同一値になるバグ的な欠落があるため。`run_statsmodels_benchmark.py`の
docstring参照）。`hc1`は`generate_logit_crosscheck_fixtures.py`（R側、正しく補正を
適用する`sandwich::vcovHC`）が主リファレンスの役割を担う（ユーザー確認済み）。

入力データは`tests/api_tests/fixtures/benchmarks/data/`に固定済みのlogit_*.csvを読む
（`benchmark/freeze_datasets.py`参照）。

使用例:
    python generate_logit_fixtures.py --output ../../../tests/api_tests/fixtures/benchmarks/logit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # benchmark/nonlinear/ を import path に追加（run_statsmodels_benchmark）

import polars as pl  # noqa: E402
import statsmodels  # noqa: E402

from run_statsmodels_benchmark import DATA_DIR, run  # noqa: E402

# perfect_multicollinearityは数値比較の対象外（ComputationErrorの発生確認のみ、
# testing-policy.md「テストの3系統」）。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    # logit特有の病理（準完全分離）。収束するが標準誤差が大きく膨らむ境界値ケース。
    "near_separation",
    # 変数間のスケールが極端に異なるケース（generate_binary_choice_datasets.py参照。
    # 真のDGPは未スケーリングのXで計算済みのため成功パス）。
    "scale_variance",
]

# hc1はstatsmodelsで未実装のためここには含めない（上記docstring参照）。
COV_TYPES = ["classical", "opg", "hc0", "cluster"]

MROZ_FORMULA = (
    "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
)


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {}
        for cov_type in COV_TYPES:
            if cov_type == "cluster":
                continue  # clusterはbaselineのみ、下のcluster専用ケースで扱う
            result = run(
                dataset_source="synthetic",
                dataset=scenario,
                formula=None,
                cov_type=cov_type,
            )
            fixtures[scenario][cov_type] = result

    # クラスターロバストSEは、シナリオ依存ではなくグルーピングの動作確認が目的のため、
    # baselineシナリオでのみ、複数のグルーピングパターンで確認する
    # （OLSのgenerate_ols_fixtures.pyと同じ方針、testing-policy.md「テスト用データセット」3.）。
    n = pl.read_csv(DATA_DIR / "logit_baseline.csv").height
    fixtures["baseline"]["cluster"] = _run_cluster_case()
    fixtures["baseline"]["cluster_imbalanced"] = _run_cluster_case(
        groups=_imbalanced_cluster_groups(n),
        note="不均衡な疑似グループ（サイズ[2,3,5,10,30,50]のタイル）。",
    )
    fixtures["baseline"]["cluster_g2"] = _run_cluster_case(
        groups=[str(i % 2) for i in range(n)],
        note=(
            "クラスタ数境界（G=2ちょうど）の成功パス確認用。Logitのcluster_cov_params"
            "はOLSのwald_f_testのようなq×q部分行列の反転を要求しないため、"
            "OLSのcluster_g2ケースと異なり説明変数を1個に絞る必要はない"
            "（k=3のままG=2で正常に計算できることを実機確認済み）。"
        ),
    )

    # 実データセット（Wooldridge mroz、労働参加モデル）。
    fixtures["mroz"] = {}
    for cov_type in COV_TYPES:
        if cov_type == "cluster":
            continue
        fixtures["mroz"][cov_type] = run(
            dataset_source="wooldridge",
            dataset="mroz",
            formula=MROZ_FORMULA,
            cov_type=cov_type,
        )
    # 実データでのクラスターロバストSE（testing-policy.md「テスト用データセット」3.
    # 「実データでのグループ列も検証する」）。mrozの`city`（都市部居住ダミー、
    # 484/269の2値）を実カテゴリ列として使う（OLSのwage1/regionクラスターと同じ趣旨）。
    fixtures["mroz"]["cluster"] = run(
        dataset_source="wooldridge",
        dataset="mroz",
        formula=MROZ_FORMULA,
        cov_type="cluster",
        cluster_col="city",
    )

    fixtures["_meta"] = {
        "method": "logit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "cov_type='hc1'はstatsmodelsのdiscrete modelで未実装（HC0と同一値を返す）"
            "ため含まない。logit_crosscheck.json（R側）が主リファレンスを担う。"
            "cov_type='opg'の限界効果（margeff）はstatsmodels側では算出できないため"
            "nullになっている（run_statsmodels_benchmark.py参照）。"
            "near_separationはlogit特有の病理（準完全分離）の境界値ケース。"
            "完全分離下でのNonConvergence検出には既知の限界があり、専用シナリオは"
            "採用していない（docs/spec/logit-spec.md参照）。"
            "scale_varianceは真のDGPを未スケーリングのXで計算した後に列のみを"
            "スケーリングする設計のため成功パス（generate_binary_choice_datasets.py参照）。"
            "n=k+1（自由度1ちょうど）の境界値ケースはOLSと異なり非採用（n<=kでは"
            "logitのMLEが構造的にほぼ確実に完全分離を起こすため、意味のある成功パスに"
            "ならない。docs/spec/logit-spec.md参照）。"
            "mrozのcluster（city列、都市部居住ダミー）は実データでのクラスターロバスト"
            "SE確認用。"
        ),
    }
    return fixtures


# 疑似グループのパターン生成（OLSのgenerate_synthetic_datasets.imbalanced_cluster_groupsと
# 同じ設計、[2,3,5,10,30,50]のタイルをnに応じて繰り返す）。
_IMBALANCED_CLUSTER_TILE = [2, 3, 5, 10, 30, 50]


def _imbalanced_cluster_groups(n: int) -> list[str]:
    if n % 100 != 0:
        raise ValueError(f"n must be a multiple of 100, got n={n}")
    n_tiles = n // 100
    labels: list[str] = []
    group_idx = 0
    for _ in range(n_tiles):
        for size in _IMBALANCED_CLUSTER_TILE:
            labels.extend([f"g{group_idx}"] * size)
            group_idx += 1
    return labels


def _run_cluster_case(
    groups: list | None = None,
    note: str = "決め打ちの疑似グループ（行番号%10）。統計的な意味はなく、実装の動作確認用。",
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて実行する。"""
    import statsmodels.formula.api as smf

    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    pandas_df = df.to_pandas()
    pandas_df["_group"] = (
        groups
        if groups is not None
        else [i % 10 for i in range(len(pandas_df))]
    )

    formula = "y ~ x1 + x2 + x3"
    model = smf.logit(formula=formula, data=pandas_df).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pandas_df["_group"]}
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
        default="../../../tests/api_tests/fixtures/benchmarks/logit.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
