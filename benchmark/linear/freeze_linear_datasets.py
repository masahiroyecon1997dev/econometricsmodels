"""linear系統（OLS/WLS）の合成データセットをCSVとして固定（凍結）する。

`benchmark/freeze_datasets.py`（系統横断のディスパッチャ）から呼ばれる。
単体でも実行できる。

使用例:
    python freeze_linear_datasets.py --output-dir \\
        ../../tests/api_tests/fixtures/benchmarks/data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # benchmark/linear/ を import path に追加（generate_linear_datasets）
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # benchmark/ を import path に追加（_common）

from _common import freeze_scenarios, run_freeze_cli  # noqa: E402
from generate_linear_datasets import generate_linear_dataset  # noqa: E402

# generate_ols_fixtures.py / generate_wls_fixtures.py のNUMERIC_SCENARIOSに
# perfect_multicollinearity（ComputationErrorパスのテストで使う、数値比較はしない）
# を加えた全シナリオ。
SYNTHETIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "perfect_multicollinearity",
    "scale_variance",
    "high_condition_number",
]

# cluster_g2ケース専用。k=1だとrng呼び出し順序が変わるため
# baseline（既定k=3）とは別データになる。
SYNTHETIC_K1_SCENARIOS = ["baseline"]

# n=k+1（自由度1ちょうど）の成功パス確認専用。SCENARIOSには
# 追加せず、cluster_g2ケースと同様にbaselineをn=k+1でオーバーライドした
# 専用データとして固定する。kはbaseline既定と揃え（generate_linear_dataset()の
# k=3、つまりx1..x3）。engine側の`k`は定数項を含む設計行列の列数
# （= generate_linear_dataset()のk + 1 = 4）のため、df_resid=1ちょうどにするには
# n = 4 + 1 = 5 が必要（n = generate_linear_dataset()のk + 2）。
SYNTHETIC_BOUNDARY_DF1_SCENARIOS = ["baseline"]


def freeze(output_dir: Path) -> None:
    true_betas: dict[str, list[float]] = {}
    freeze_scenarios(
        output_dir,
        generate_linear_dataset,
        SYNTHETIC_SCENARIOS,
        "synthetic",
        true_betas,
    )
    freeze_scenarios(
        output_dir,
        generate_linear_dataset,
        SYNTHETIC_K1_SCENARIOS,
        "synthetic",
        true_betas,
        filename_suffix="_k1",
        key_suffix="_k1",
        k=1,
    )
    freeze_scenarios(
        output_dir,
        generate_linear_dataset,
        SYNTHETIC_BOUNDARY_DF1_SCENARIOS,
        "synthetic",
        true_betas,
        filename_suffix="_df1",
        key_suffix="_df1",
        n=5,
        k=3,
    )
    (output_dir / "synthetic_true_beta.json").write_text(
        json.dumps(true_betas, indent=2)
    )


if __name__ == "__main__":
    run_freeze_cli(
        freeze,
        "../../tests/api_tests/fixtures/benchmarks/data",
        "wrote frozen linear datasets",
        description=__doc__,
    )
