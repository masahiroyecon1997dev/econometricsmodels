"""nonlinear系統（Logit/Probit）の合成データセットをCSVとして固定（凍結）する。

`benchmark/freeze_datasets.py`（系統横断のディスパッチャ）から呼ばれる。
単体でも実行できる（リポジトリルートから `python -m benchmark.nonlinear.freeze`）。
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.common import freeze_scenarios, run_freeze_cli
from benchmark.nonlinear.datasets import SCENARIOS as LOGIT_SCENARIOS
from benchmark.nonlinear.datasets import generate_binary_choice_dataset

# datasets.pyのSCENARIOS（全シナリオ、generate_logit_fixtures.py
# のNUMERIC_SCENARIOSに、エラーパス確認用のperfect_multicollinearityを加えたもの）を
# そのまま使う。PROBIT_SCENARIOSはLOGIT_SCENARIOSと同じシナリオ構成
# （datasets.py参照）。
PROBIT_SCENARIOS = list(LOGIT_SCENARIOS)


def freeze(output_dir: Path) -> None:
    logit_true_betas: dict[str, list[float]] = {}
    freeze_scenarios(
        output_dir,
        generate_binary_choice_dataset,
        LOGIT_SCENARIOS,
        "logit",
        logit_true_betas,
        link="logit",
    )
    (output_dir / "logit_true_beta.json").write_text(
        json.dumps(logit_true_betas, indent=2)
    )

    probit_true_betas: dict[str, list[float]] = {}
    freeze_scenarios(
        output_dir,
        generate_binary_choice_dataset,
        PROBIT_SCENARIOS,
        "probit",
        probit_true_betas,
        link="probit",
    )
    (output_dir / "probit_true_beta.json").write_text(
        json.dumps(probit_true_betas, indent=2)
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
        "wrote frozen nonlinear datasets",
        description=__doc__,
    )
