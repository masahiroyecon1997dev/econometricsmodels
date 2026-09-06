"""linear系統（OLS/WLS）の合成データセットをCSVとして固定（凍結）する。

`benchmark/regenerate_all.py`（合成データ＋全フィクスチャの一括再生成）から呼ばれる。
単体でも実行できる。

使用例（リポジトリルートから）:
    python -m benchmark.linear.freeze
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.common import freeze_scenarios, run_freeze_cli
from benchmark.linear.datasets import (
    SCENARIOS as SYNTHETIC_SCENARIOS,
)
from benchmark.linear.datasets import generate_linear_dataset

# benchmark/linear/datasets.pyのSCENARIOS（全シナリオ、generate_ols_fixtures.py /
# generate_wls_fixtures.pyのNUMERIC_SCENARIOSにperfect_multicollinearity・
# scale_variance（いずれもComputationErrorパスのテストで使う、数値比較はしない）を
# 加えたもの）をそのまま使う。

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
        str(
            Path(__file__).resolve().parents[2]
            / "tests"
            / "fixtures"
            / "benchmarks"
            / "data"
        ),
        "wrote frozen linear datasets",
        description=__doc__,
    )
