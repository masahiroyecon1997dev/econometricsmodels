"""Probitのテストフィクスチャ（tests/fixtures/benchmarks/probit.json）を
生成するスクリプト。

`benchmark/nonlinear/run_statsmodels_benchmark.py`（`--model probit`、1回呼べば
1ケース分の結果を返す汎用ツール）を全シナリオ×全cov_typeの組み合わせで呼び出し、
結果を1つのJSONにまとめて書き出す。`generate_logit_fixtures.py`と完全に同型の設計
（シナリオ構成・cov_type構成もLogitと同一、`generate_nonlinear_datasets.py`
参照）。

**`cov_type="hc1"`はここに含めない**（statsmodelsのdiscrete modelがn/(n-k)小標本補正を
実装しておらずHC0と同一値になるバグ的な欠落があるため、Probitでも同じ欠落を実機確認
済み。`run_statsmodels_benchmark.py`のdocstring参照）。`hc1`は
`generate_probit_crosscheck_fixtures.py`（R側、正しく補正を適用する
`sandwich::vcovHC`）が主リファレンスの役割を担う（ユーザー確認済み）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのprobit_*.csvを読む
（`benchmark/freeze_datasets.py`参照）。

使用例:
    python generate_probit_fixtures.py --output ../../../tests/fixtures/benchmarks/probit.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import statsmodels
from _common import DATA_DIR, imbalanced_cluster_groups
from run_statsmodels_benchmark import run

# perfect_multicollinearityは数値比較の対象外（ComputationErrorの発生確認のみ、
# testing-policy.md「テストの3系統」）。generate_logit_fixtures.pyと同じシナリオ構成
# （generate_nonlinear_datasets.py参照）。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    # probit特有の病理（準完全分離）。収束するが標準誤差が大きく膨らむ境界値ケース。
    # 較正値はlogit（beta1=20）と異なりbeta1=10（generate_nonlinear_datasets.py参照）。
    "near_separation",
    # 変数間のスケールが極端に異なるケース（真のDGPは未スケーリングのXで計算済みの
    # ため成功パス）。
    "scale_variance",
]

# hc1はstatsmodelsで未実装のためここには含めない（上記docstring参照）。
# clusterはbaselineのみ、下のcluster専用ケース（_run_cluster_case）で個別に扱うため
# ここには含めない（OLS/WLSと同じ書き方、項目14参照）。
COV_TYPES = ["classical", "opg", "hc0"]

# newton以外のmethod（bfgs/lbfgs）が主リファレンスに対しフルの統計量（std_errors含む）で
# 一致することの確認用（Issue #231フェーズ4のtesting-completeness-reviewer指摘、
# generate_logit_fixtures.pyと同じ方針）。
METHODS = ["bfgs", "lbfgs"]

MROZ_FORMULA = (
    "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
)


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
                model="probit",
            )
            fixtures[scenario][cov_type] = result

    # クラスターロバストSEは、シナリオ依存ではなくグルーピングの動作確認が目的のため、
    # baselineシナリオでのみ、複数のグルーピングパターンで確認する
    # （generate_logit_fixtures.pyと同じ方針、testing-policy.md「テスト用データセット」3.）。
    n = pl.read_csv(DATA_DIR / "probit_baseline.csv").height
    fixtures["baseline"]["cluster"] = _run_cluster_case()
    fixtures["baseline"]["cluster_imbalanced"] = _run_cluster_case(
        groups=imbalanced_cluster_groups(n),
        note="不均衡な疑似グループ（サイズ[2,3,5,10,30,50]のタイル）。",
    )
    fixtures["baseline"]["cluster_g2"] = _run_cluster_case(
        groups=[str(i % 2) for i in range(n)],
        note=(
            "クラスタ数境界（G=2ちょうど）の成功パス確認用。Probitのcluster_cov_params"
            "はLogitと同じくOLSのwald_f_testのようなq×q部分行列の反転を要求しないため、"
            "説明変数を1個に絞る必要はない（k=3のままG=2で正常に計算できることを"
            "実機確認済み）。"
        ),
    )

    # 実データセット（Wooldridge mroz、労働参加モデル）。Logitと同じformula・データ
    # （probit_logitとも定番の比較対象、mrozはWooldridge教科書でも両方の例に使われる）。
    fixtures["mroz"] = {}
    for cov_type in COV_TYPES:
        fixtures["mroz"][cov_type] = run(
            dataset_source="wooldridge",
            dataset="mroz",
            formula=MROZ_FORMULA,
            cov_type=cov_type,
            model="probit",
        )
    # 実データでのクラスターロバストSE（testing-policy.md「テスト用データセット」3.
    # 「実データでのグループ列も検証する」）。mrozの`city`（都市部居住ダミー、
    # 484/269の2値）を実カテゴリ列として使う（Logitと同じ趣旨）。
    fixtures["mroz"]["cluster"] = run(
        dataset_source="wooldridge",
        dataset="mroz",
        formula=MROZ_FORMULA,
        cov_type="cluster",
        cluster_col="city",
        model="probit",
    )

    fixtures["method"] = {
        method: run(
            dataset_source="synthetic",
            dataset="baseline",
            formula=None,
            cov_type="classical",
            model="probit",
            method=method,
        )
        for method in METHODS
    }

    fixtures["_meta"] = {
        "method": "probit",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "cov_type='hc1'はstatsmodelsのdiscrete modelで未実装（HC0と同一値を返す）"
            "ため含まない。probit_crosscheck.json（R側）が主リファレンスを担う。"
            "cov_type='opg'の限界効果（margeff）はstatsmodels側では算出できないため"
            "nullになっている（run_statsmodels_benchmark.py参照）。"
            "near_separationはprobit特有の病理（準完全分離）の境界値ケース"
            "（較正値はlogitと異なりbeta1=10、generate_nonlinear_datasets.py参照）。"
            "完全分離下でのNonConvergence検出には既知の限界（logitと同じ、"
            "nonlinear/common.rsのrun_solverを共有するため）があり、専用シナリオは"
            "採用していない。"
            "scale_varianceは真のDGPを未スケーリングのXで計算した後に列のみを"
            "スケーリングする設計のため成功パス。"
            "n=k+1（自由度1ちょうど）の境界値ケースはLogitと同じ理由で非採用"
            "（n<=kではMLEが構造的にほぼ確実に完全分離を起こすため）。"
            "mrozのcluster（city列、都市部居住ダミー）は実データでのクラスターロバスト"
            "SE確認用。"
            "methodはbfgs/lbfgsがnewtonと同じ最尤解・標準誤差に収束することを主"
            "リファレンスに対して確認するためのfixture（baselineシナリオ・classical"
            "cov_typeの1ケースのみ、Issue #231フェーズ4で追加）。"
        ),
    }
    return fixtures


def _run_cluster_case(
    groups: list | None = None,
    note: str = "決め打ちの疑似グループ（行番号%10）。統計的な意味はなく、実装の動作確認用。",
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて実行する。"""
    import statsmodels.formula.api as smf

    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    pandas_df = df.to_pandas()
    pandas_df["_group"] = (
        groups
        if groups is not None
        else [i % 10 for i in range(len(pandas_df))]
    )

    formula = "y ~ x1 + x2 + x3"
    model = smf.probit(formula=formula, data=pandas_df).fit(
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
            "generated_at": datetime.now(UTC).isoformat(),
            "note": note,
            "formula": formula,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../../../tests/fixtures/benchmarks/probit.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
