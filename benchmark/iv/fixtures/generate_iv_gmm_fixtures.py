"""GMM（`method="gmm"`）のテストフィクスチャ（tests/fixtures/benchmarks/
iv_gmm.json）を生成するスクリプト。

`benchmark/iv/run_linearmodels_benchmark_iv.py`の`run_gmm()`（1回呼べば1ケース分の
結果を返す汎用ツール）を全シナリオ×cov_type、および代表的なweight_typeの組み合わせで
呼び出し、結果を1つのJSONにまとめて書き出す。

2SLS用の`iv.json`/`generate_iv_fixtures.py`とは別ファイル・別スクリプトにしている
理由: `IV`/`IvOptions`は`method="2sls"`/`"gmm"`を単一クラスで切り替える設計だが、
GMM固有の`weight_type`軸（`cov_type`とは独立、`iv-api-design.md`6.2節）がある分
2SLSとフィクスチャの形状が異なるため、OLS/WLSと同じ「推定量ごとに別ファイル」の
既存方針（`ols.json`/`wls.json`）に倣った（ユーザー確認済み）。

検証範囲（ユーザー確認済み、`cov_type`×`weight_type`の全組み合わせ
（8シナリオ×4weight_type×6cov_type）は規模が大きすぎるため）:
    - `weight_type="unadjusted"`固定で、全8シナリオ×cov_type（classical/hc0/hc1/
      hac、baselineのみ追加でcluster/cluster_imbalanced）を検証する
      （2SLSの`iv.json`と同じ組み合わせ）。
    - 他のweight_type（robust/cluster/kernel）は、`weight_type`と`cov_type`が
      独立な軸であることの確認が目的のため、baselineシナリオ×cov_type=classical
      のみで動作確認する。

このスクリプト自体は`benchmark/`側に置く。生成される`iv_gmm.json`は
`tests/fixtures/benchmarks/`に置く（両者を分ける理由は
`.claude/skills/reference-benchmark/SKILL.md`参照）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/freeze_datasets.py`参照）。

使用例:
    python generate_iv_gmm_fixtures.py --output ../../../tests/fixtures/benchmarks/iv_gmm.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import linearmodels
import polars as pl
from _common import DATA_DIR, imbalanced_cluster_groups
from run_linearmodels_benchmark_iv import run_gmm

# `generate_iv_fixtures.py`のNUMERIC_SCENARIOSと同一（2SLSと同じ合成データセットを
# 再利用する）。
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

INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}

X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}

COV_TYPES = ["classical", "hc0", "hc1", "hac"]
# `weight_type`と`cov_type`が独立な軸であることの確認用（baselineのみ）。
OTHER_WEIGHT_TYPES = ["robust", "cluster", "kernel"]
# 1-step（gmm_iterations=1）・iterated GMM（3以上、固定回数モード）の成功パス確認用
# （既定値2以外、Issue #231フェーズ4）。
GMM_ITERATIONS_SCENARIOS = [1, 3]


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
        instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])

        fixtures[scenario] = {"unadjusted": {}}
        for cov_type in COV_TYPES:
            result = run_gmm(
                dataset=scenario,
                x_exog_cols=x_exog,
                x_endog_cols=["endog1"],
                instrument_cols=instruments,
                weight_type="unadjusted",
                cov_type=cov_type,
            )
            fixtures[scenario]["unadjusted"][cov_type] = result

        if scenario == "baseline":
            n = pl.read_csv(DATA_DIR / "iv_baseline.csv").height
            fixtures[scenario]["unadjusted"]["cluster"] = _run_cluster_case(
                "baseline", weight_type="unadjusted"
            )
            fixtures[scenario]["unadjusted"]["cluster_imbalanced"] = (
                _run_cluster_case(
                    "baseline",
                    weight_type="unadjusted",
                    groups=imbalanced_cluster_groups(n),
                )
            )

            for weight_type in OTHER_WEIGHT_TYPES:
                if weight_type == "cluster":
                    fixtures[scenario][weight_type] = {
                        "classical": _run_cluster_case(
                            "baseline",
                            weight_type="cluster",
                            cov_type="classical",
                        )
                    }
                else:
                    fixtures[scenario][weight_type] = {
                        "classical": run_gmm(
                            dataset="baseline",
                            x_exog_cols=["x1"],
                            x_endog_cols=["endog1"],
                            instrument_cols=["z1", "z2"],
                            weight_type=weight_type,
                            cov_type="classical",
                        )
                    }

    # 複数内生変数（k_endog>=2）。2SLSのiv.jsonと同じ構成（Issue #231フェーズ4、
    # testing-completeness-reviewer指摘のmust fix）。weight_type='unadjusted'固定で
    # cov_typeのみ変える（上記と同じ検証範囲の絞り方）。
    fixtures["multi_endog"] = {"unadjusted": {}}
    for cov_type in COV_TYPES:
        fixtures["multi_endog"]["unadjusted"][cov_type] = run_gmm(
            dataset="baseline_multi_endog",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1", "endog2"],
            instrument_cols=["z1", "z2", "z3"],
            weight_type="unadjusted",
            cov_type=cov_type,
        )

    # weight_type='kernel' × cov_type='hac'の組み合わせ（実務上最も典型的な
    # 「HACカーネル重み＋HAC標準誤差」の組み合わせ経路、Issue #231フェーズ4、
    # testing-completeness-reviewer指摘のshould fix。上記OTHER_WEIGHT_TYPESループは
    # cov_type='classical'固定のためこの組み合わせを通らない）。
    fixtures["kernel_hac"] = run_gmm(
        dataset="baseline",
        x_exog_cols=["x1"],
        x_endog_cols=["endog1"],
        instrument_cols=["z1", "z2"],
        weight_type="kernel",
        cov_type="hac",
    )

    # gmm_iterations: 1（1-step）・3以上（iterated、固定回数モード）の成功パス
    # （Issue #231フェーズ4、testing-completeness-reviewer指摘のshould fix）。
    # baselineシナリオ・weight_type='unadjusted'・cov_type='classical'固定。
    fixtures["gmm_iterations"] = {
        n_iter: run_gmm(
            dataset="baseline",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1"],
            instrument_cols=["z1", "z2"],
            weight_type="unadjusted",
            cov_type="classical",
            gmm_iterations=n_iter,
        )
        for n_iter in GMM_ITERATIONS_SCENARIOS
    }

    fixtures["_meta"] = {
        "method": "gmm",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "note": (
            "weight_type='unadjusted'固定で全8シナリオ×cov_type"
            "（classical/hc0/hc1/hac、baselineのみ追加でcluster/"
            "cluster_imbalanced）を検証する。hc2/hc3は2SLSと同じ理由で対象外"
            "（`run_linearmodels_benchmark_iv.py`のモジュールdocコメント参照）。"
            "他のweight_type（robust/cluster/kernel）はweight_typeとcov_typeが"
            "独立な軸であることの確認が目的のため、baselineシナリオ×"
            "cov_type=classicalのみで検証する（ユーザー確認済み）。"
            "z_stats/p_values/conf_int/f_statistic/f_p_valueは常にz分布・"
            "カイ二乗形式（qで割らない）で独自に計算し直した値（`gmm.rs`の設計、"
            "`run_gmm()`のモジュールdocコメント参照）。hansen_j_statistic/"
            "hansen_j_p_valueは過剰識別のときのみ値を持ち、丁度識別では`None`。"
            "wu_hausman_statistic相当のキーはGMMには存在しないため含まない。"
            "perfect_multicollinearity/G=2クラスター境界はここに含まない"
            "（2SLSの`iv.json`と同じ理由、G=2境界は`engine/src/iv/CLAUDE.md`"
            "「修正済み」参照）。"
            "multi_endog（複数内生変数、x_endog=['endog1','endog2']）は"
            "generate_iv_datasets.pyの第一段階誤差vが内生変数ごとに独立になる"
            "よう修正した後のデータで生成（Issue #231フェーズ4、"
            "generate_iv_fixtures.pyの同名注記参照）。"
            "kernel_hac（weight_type='kernel'×cov_type='hac'）・gmm_iterations"
            "（1/3、既定値2以外の成功パス）も同フェーズで追加。"
        ),
    }
    return fixtures


def _run_cluster_case(
    dataset: str,
    weight_type: str,
    cov_type: str = "cluster",
    groups: list | None = None,
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて`run_gmm`を呼ぶ
    （`generate_iv_fixtures.py`の`_run_cluster_case`と同じ発想）。
    """
    filename = f"iv_{dataset}.csv"
    df = pl.read_csv(DATA_DIR / filename)
    n = df.height
    cluster_group = (
        groups if groups is not None else [i % 10 for i in range(n)]
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = DATA_DIR / f"iv_{dataset}_gmm_cluster_tmp.csv"
    grouped.write_csv(tmp_path)
    try:
        return run_gmm(
            dataset=f"{dataset}_gmm_cluster_tmp",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1"],
            instrument_cols=["z1", "z2"],
            weight_type=weight_type,
            cov_type=cov_type,
            cluster_col="cluster_group",
        )
    finally:
        tmp_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../../../tests/fixtures/benchmarks/iv_gmm.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
