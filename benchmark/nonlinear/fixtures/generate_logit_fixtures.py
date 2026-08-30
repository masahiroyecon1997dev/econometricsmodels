"""Logitのテストフィクスチャ（tests/fixtures/benchmarks/logit.json）を
生成するスクリプト。

`benchmark/nonlinear/references/statsmodels_ref.py`（1回呼べば1ケース分の結果を
返す汎用アダプタ）を全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つの
JSONにまとめて書き出す。`benchmark/linear/fixtures/generate_ols_fixtures.py`と
同型の設計。

**`cov_type="hc1"`はここに含めない**（statsmodelsのdiscrete modelがn/(n-k)小標本補正を
実装しておらずHC0と同一値になるバグ的な欠落があるため。`statsmodels_ref.py`の
docstring参照）。`hc1`は`generate_logit_crosscheck_fixtures.py`（R側、正しく補正を
適用する`sandwich::vcovHC`）が主リファレンスの役割を担う（ユーザー確認済み）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのlogit_*.csvを読む
（`benchmark/nonlinear/freeze.py`参照）。

使用例（リポジトリルートから）:
    python -m benchmark.nonlinear.fixtures.generate_logit_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import statsmodels

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    MROZ_FORMULA,
    extract_coef_se,
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.nonlinear.references.statsmodels_ref import run

# perfect_multicollinearityは数値比較の対象外（ComputationErrorの発生確認のみ、
# testing-policy.md「テストの3系統」）。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    # logit特有の病理（準完全分離）。収束するが標準誤差が大きく膨らむ境界値ケース。
    "near_separation",
    # 変数間のスケールが極端に異なるケース（benchmark/nonlinear/datasets.py参照。
    # 真のDGPは未スケーリングのXで計算済みのため成功パス）。
    "scale_variance",
]

# hc1はstatsmodelsで未実装のためここには含めない（上記docstring参照）。
# clusterはbaselineのみ、下のcluster専用ケース（_run_cluster_case）で個別に扱うため
# ここには含めない（OLS/WLSと同じ書き方、項目14参照）。
COV_TYPES = ["classical", "opg", "hc0"]

# newton以外のmethod（bfgs/lbfgs）が主リファレンスに対しフルの統計量（std_errors含む）で
# 一致することの確認用。
# baselineシナリオ・classical cov_typeの1ケースのみで十分（method自体の違いは
# 収束後の最適化点の精度差であり、シナリオ×cov_typeを掛け合わせる必要はない）。
METHODS = ["bfgs", "lbfgs"]


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
    # （OLSのgenerate_ols_fixtures.pyと同じ方針、testing-policy.md「テスト用データセット」3.）。
    n = pl.read_csv(DATA_DIR / "logit_baseline.csv").height
    fixtures["baseline"]["cluster"] = _run_cluster_case()
    fixtures["baseline"]["cluster_imbalanced"] = _run_cluster_case(
        groups=imbalanced_cluster_groups(n),
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

    fixtures["method"] = {
        method: run(
            dataset_source="synthetic",
            dataset="baseline",
            formula=None,
            cov_type="classical",
            method=method,
        )
        for method in METHODS
    }

    fixtures["_meta"] = {
        "method": "logit",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "cov_type='hc1'はstatsmodelsのdiscrete modelで未実装（HC0と同一値を返す）"
            "ため含まない。logit_crosscheck.json（R側）が主リファレンスを担う。"
            "cov_type='opg'の限界効果（margeff）はstatsmodels側では算出できないため"
            "nullになっている"
            "（benchmark/nonlinear/references/statsmodels_ref.py参照）。"
            "near_separationはlogit特有の病理（準完全分離）の境界値ケース。"
            "完全分離下でのNonConvergence検出には既知の限界があり、専用シナリオは"
            "採用していない（docs/spec/logit-spec.md参照）。"
            "scale_varianceは真のDGPを未スケーリングのXで計算した後に列のみを"
            "スケーリングする設計のため成功パス"
            "（benchmark/nonlinear/datasets.py参照）。"
            "n=k+1（自由度1ちょうど）の境界値ケースはOLSと異なり非採用（n<=kでは"
            "logitのMLEが構造的にほぼ確実に完全分離を起こすため、意味のある成功パスに"
            "ならない。docs/spec/logit-spec.md参照）。"
            "mrozのcluster（city列、都市部居住ダミー）は実データでのクラスターロバスト"
            "SE確認用。"
            "methodはbfgs/lbfgsがnewtonと同じ最尤解・標準誤差に収束することを主"
            "リファレンスに対して確認するためのfixture（baselineシナリオ・classical"
            "cov_typeの1ケースのみ）。"
        ),
    }
    return fixtures


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
        **extract_coef_se(model),
        "_meta": {
            "reference": "statsmodels",
            "statsmodels_version": statsmodels.__version__,
            "generated_at": datetime.now(UTC).isoformat(),
            "note": note,
            "formula": formula,
        },
    }


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "logit.json", description=__doc__
    )
