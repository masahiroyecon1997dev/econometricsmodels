"""iv系統（2SLS/GMM）の合成データセットをCSVとして固定（凍結）する。

`benchmark/freeze_datasets.py`（系統横断のディスパッチャ）から呼ばれる。
単体でも実行できる。

使用例:
    python freeze_iv_datasets.py --output-dir \\
        ../../tests/api_tests/fixtures/benchmarks/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # benchmark/iv/ を import path に追加（generate_iv_datasets）
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # benchmark/ を import path に追加（_common）

from _common import freeze_scenarios  # noqa: E402
from generate_iv_datasets import generate_iv_dataset  # noqa: E402

# generate_iv_datasets.pyのSCENARIOS全て（IV: 2SLS/GMM用）。
# moderate_multicollinearity/high_condition_number/scale_varianceはk_exog=2、
# perfect_multicollinearityはk_exog=3が必要（generate_iv_datasets.pyのdocstring参照）。
IV_SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "small_n",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    "perfect_multicollinearity",
    "scale_variance",
]
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

    (output_dir / "iv_true_beta.json").write_text(
        json.dumps(iv_true_betas, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="../../tests/api_tests/fixtures/benchmarks/data",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze(output_dir)
    print(f"wrote frozen iv datasets to {output_dir}")
