"""panel系統（FE）の合成データセットをCSVとして固定（凍結）する。

`benchmark/regenerate_all.py`（合成データ＋全フィクスチャの一括再生成）から呼ばれる。
単体でも実行できる。

使用例（リポジトリルートから）:
    python -m benchmark.panel.freeze
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.common import freeze_scenarios, run_freeze_cli
from benchmark.panel.datasets import (
    SCENARIOS,
    generate_fe_dataset,
    re_equal_counts_shifted_time_panel,
    re_two_way_small_panel,
    re_zero_variance_component_panels,
)

# df_resid=1境界（1-way）成功パス専用。SCENARIOSには追加せず、baselineを
# n_entities=3, n_periods=2, k=2でオーバーライドした専用データとして固定する
# （OLSのbaseline_df1と同型のパターン）。n=6, df_resid=6-3-2=1。
FE_BOUNDARY_DF1_ONE_WAY_SCENARIOS = ["baseline"]

# df_resid=1境界（2-way）成功パス専用。baselineをn_entities=3, n_periods=3,
# k=3でオーバーライドする。n=9, df_model=3+3+3-1=8, df_resid=9-8=1。
FE_BOUNDARY_DF1_TWO_WAY_SCENARIOS = ["baseline"]

# クラスター不均衡シナリオ専用。baselineをn_entities=20, n_periods=10
# （n=200）でオーバーライドする。`benchmark.common.imbalanced_cluster_groups`
# はnが100の倍数であることを要求するため、既存baselineの既定n=240では
# 使えない（entityとは無関係な専用クラスター列は、OLS/WLSと同じくCSVには
# 含めずフィクスチャ生成・テスト時に都度動的生成する）。
FE_CLUSTER_IMBALANCED_SCENARIOS = ["baseline"]

# クラスター数境界（G=2、q=1でG>q）の成功パス専用。baselineをk=1で
# オーバーライドする（OLSの`baseline_k1`と同型のパターン、G=2の
# クラスター列自体はentityとは無関係のためCSVには含めずフィクスチャ生成・
# テスト時に都度動的生成する）。
FE_CLUSTER_G2_SCENARIOS = ["baseline"]


def freeze(output_dir: Path) -> None:
    true_betas: dict[str, list[float]] = {}
    freeze_scenarios(
        output_dir, generate_fe_dataset, SCENARIOS, "fe", true_betas
    )
    freeze_scenarios(
        output_dir,
        generate_fe_dataset,
        FE_BOUNDARY_DF1_ONE_WAY_SCENARIOS,
        "fe",
        true_betas,
        filename_suffix="_df1_one_way",
        key_suffix="_df1_one_way",
        n_entities=3,
        n_periods=2,
        k=2,
    )
    freeze_scenarios(
        output_dir,
        generate_fe_dataset,
        FE_BOUNDARY_DF1_TWO_WAY_SCENARIOS,
        "fe",
        true_betas,
        filename_suffix="_df1_two_way",
        key_suffix="_df1_two_way",
        n_entities=3,
        n_periods=3,
        k=3,
    )
    freeze_scenarios(
        output_dir,
        generate_fe_dataset,
        FE_CLUSTER_IMBALANCED_SCENARIOS,
        "fe",
        true_betas,
        filename_suffix="_cluster_imbalanced",
        key_suffix="_cluster_imbalanced",
        n_entities=20,
        n_periods=10,
    )
    freeze_scenarios(
        output_dir,
        generate_fe_dataset,
        FE_CLUSTER_G2_SCENARIOS,
        "fe",
        true_betas,
        filename_suffix="_k1",
        key_suffix="_k1",
        k=1,
    )
    # 2-way REの手計算・plm照合用の小さな手作りパネル（乱数ではないため真の係数は無い）。
    re_two_way_small_panel().write_csv(output_dir / "re_two_way_small.csv")
    # 分散成分が0に切り上げられるRE用パネル（`variance_method`のplm照合用）。
    for name, panel in re_zero_variance_component_panels().items():
        panel.write_csv(output_dir / f"{name}.csv")
    re_equal_counts_shifted_time_panel().write_csv(
        output_dir / "re_equal_counts_shifted_time.csv"
    )
    (output_dir / "fe_true_beta.json").write_text(
        json.dumps(true_betas, indent=2)
    )


if __name__ == "__main__":
    run_freeze_cli(
        freeze,
        str(
            Path(__file__).resolve().parents[2]
            / "tests"
            / "fixtures"
            / "benchmarks"
            / "data"
        ),
        "wrote frozen panel (FE) datasets",
        description=__doc__,
    )
