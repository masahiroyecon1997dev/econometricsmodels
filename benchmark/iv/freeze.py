"""iv系統（2SLS/GMM）の合成データセットをCSVとして固定（凍結）する。

`benchmark/regenerate_all.py`（合成データ＋全フィクスチャの一括再生成）から呼ばれる。
単体でも実行できる（リポジトリルートから `python -m benchmark.iv.freeze`）。
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.common import freeze_scenarios, run_freeze_cli
from benchmark.iv.datasets import SCENARIOS as IV_SCENARIOS
from benchmark.iv.datasets import generate_iv_dataset

# datasets.pyのSCENARIOS（IV: 2SLS/GMM用の全シナリオ）をそのまま使う。
# moderate_multicollinearity/high_condition_number/scale_varianceはk_exog=2、
# perfect_multicollinearityはk_exog=3が必要（datasets.pyのdocstring参照）。
IV_K_EXOG_OVERRIDES = {
    "moderate_multicollinearity": 2,
    "high_condition_number": 2,
    "scale_variance": 2,
    "perfect_multicollinearity": 3,
}

# クラスターロバストSEのG=2境界ケース専用（`testing-policy.md`「クラスタ数G」の罠、
# linearのSYNTHETIC_K1_SCENARIOSと同じ理由）。x_exog=0にしてq=1（endog1のみ）に絞る
# （baseline既定のx_exog=['x1']込みだとq=2になりG=2で必然的に特異になるため）。
IV_G2_BOUNDARY_SCENARIOS = ["baseline"]

# 複数内生変数（k_endog>=2）の成功パス確認専用（Issue #231フェーズ4、
# testing-completeness-reviewer指摘のmust fix）。k_instruments=3・k_endog=2で
# 過剰識別（Sargan/Hansen Jが非nullになる）にする。
IV_MULTI_ENDOG_SCENARIOS = ["baseline"]

# 自由度1境界（df_resid=1ちょうど）の成功パス確認専用（Issue #235）。IVは
# OLSと異なり内生変数1本・操作変数1本が最低限必要なため、x_exog=0・k_endog=1・
# k_instruments=1（丁度識別）の最小構成にする。df_resid = n - (k_exog + k_endog + 1) = 1
# を満たすにはn = 0 + 1 + 1 + 1 = 3が必要。
IV_BOUNDARY_DF1_SCENARIOS = ["baseline"]


def freeze(output_dir: Path) -> None:
    iv_true_betas: dict[str, list[float]] = {}
    for scenario in IV_SCENARIOS:
        kwargs = {}
        if scenario in IV_K_EXOG_OVERRIDES:
            kwargs["k_exog"] = IV_K_EXOG_OVERRIDES[scenario]
        freeze_scenarios(
            output_dir,
            generate_iv_dataset,
            [scenario],
            "iv",
            iv_true_betas,
            **kwargs,
        )

    freeze_scenarios(
        output_dir,
        generate_iv_dataset,
        IV_G2_BOUNDARY_SCENARIOS,
        "iv",
        iv_true_betas,
        filename_suffix="_g2",
        key_suffix="_g2",
        k_exog=0,
    )

    freeze_scenarios(
        output_dir,
        generate_iv_dataset,
        IV_MULTI_ENDOG_SCENARIOS,
        "iv",
        iv_true_betas,
        filename_suffix="_multi_endog",
        key_suffix="_multi_endog",
        k_endog=2,
        k_instruments=3,
    )

    freeze_scenarios(
        output_dir,
        generate_iv_dataset,
        IV_BOUNDARY_DF1_SCENARIOS,
        "iv",
        iv_true_betas,
        filename_suffix="_df1",
        key_suffix="_df1",
        n=3,
        k_exog=0,
        k_endog=1,
        k_instruments=1,
    )

    (output_dir / "iv_true_beta.json").write_text(
        json.dumps(iv_true_betas, indent=2)
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
        "wrote frozen iv datasets",
        description=__doc__,
    )
