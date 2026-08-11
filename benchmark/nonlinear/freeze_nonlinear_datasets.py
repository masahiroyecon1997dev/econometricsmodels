"""nonlinear系統（Logit/Probit）の合成データセットをCSVとして固定（凍結）する。

`benchmark/freeze_datasets.py`（系統横断のディスパッチャ）から呼ばれる。
単体でも実行できる。

使用例:
    python freeze_nonlinear_datasets.py --output-dir \\
        ../../tests/api_tests/fixtures/benchmarks/data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # benchmark/nonlinear/ を import path に追加（generate_nonlinear_datasets）
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # benchmark/ を import path に追加（_common）

from _common import freeze_scenarios, run_freeze_cli  # noqa: E402
from generate_nonlinear_datasets import (  # noqa: E402
    generate_logit_dataset,
    generate_probit_dataset,
)

# generate_logit_fixtures.pyのNUMERIC_SCENARIOSに、エラーパス確認用の
# perfect_multicollinearityを加えた全シナリオ（generate_nonlinear_datasets.py参照）。
LOGIT_SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    "near_separation",
    "perfect_multicollinearity",
    "scale_variance",
]

# generate_probit_fixtures.pyのNUMERIC_SCENARIOSに、エラーパス確認用の
# perfect_multicollinearityを加えた全シナリオ。LOGIT_SCENARIOSと同じシナリオ構成
# （generate_nonlinear_datasets.py参照）。
PROBIT_SCENARIOS = list(LOGIT_SCENARIOS)


def freeze(output_dir: Path) -> None:
    logit_true_betas: dict[str, list[float]] = {}
    freeze_scenarios(
        output_dir,
        generate_logit_dataset,
        LOGIT_SCENARIOS,
        "logit",
        logit_true_betas,
    )
    (output_dir / "logit_true_beta.json").write_text(
        json.dumps(logit_true_betas, indent=2)
    )

    probit_true_betas: dict[str, list[float]] = {}
    freeze_scenarios(
        output_dir,
        generate_probit_dataset,
        PROBIT_SCENARIOS,
        "probit",
        probit_true_betas,
    )
    (output_dir / "probit_true_beta.json").write_text(
        json.dumps(probit_true_betas, indent=2)
    )


if __name__ == "__main__":
    run_freeze_cli(
        freeze,
        "../../tests/api_tests/fixtures/benchmarks/data",
        "wrote frozen nonlinear datasets",
        description=__doc__,
    )
