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
from benchmark.panel.datasets import SCENARIOS, generate_fe_dataset


def freeze(output_dir: Path) -> None:
    true_betas: dict[str, list[float]] = {}
    freeze_scenarios(
        output_dir, generate_fe_dataset, SCENARIOS, "fe", true_betas
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
