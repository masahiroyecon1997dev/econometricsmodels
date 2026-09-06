"""nonlinear系統（Logit/Probit）の合成データセットをCSVとして固定（凍結）する。

`benchmark/regenerate_all.py`（合成データ＋全フィクスチャの一括再生成）から呼ばれる。
単体でも実行できる（リポジトリルートから `python -m benchmark.nonlinear.freeze`）。
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.common import freeze_scenarios, run_freeze_cli
from benchmark.nonlinear.datasets import SCENARIOS as LOGIT_SCENARIOS
from benchmark.nonlinear.datasets import (
    TOBIT_SCENARIOS,
    generate_binary_choice_dataset,
    generate_censored_regression_dataset,
)

# datasets.pyのSCENARIOS（全シナリオ、generate_logit_fixtures.py
# のNUMERIC_SCENARIOSに、エラーパス確認用のperfect_multicollinearityを加えたもの）を
# そのまま使う。PROBIT_SCENARIOSはLOGIT_SCENARIOSと同じシナリオ構成
# （datasets.py参照）。
PROBIT_SCENARIOS = list(LOGIT_SCENARIOS)


def _freeze_tobit(output_dir: Path) -> None:
    """Tobit の合成データセットを CSV 固定し、true_beta と打ち切り境界を JSON 化する。

    `generate_censored_regression_dataset` は共通の `freeze_scenarios` が想定する
    `(df, true_beta)` ではなく `(df, true_beta, (lower, upper))` を返す（打ち切り境界は
    y* の分位点として生成時に決まりデータ依存のため）。そのため専用ループを持つ。
    境界は `tobit_censoring_bounds.json` に固定し、フィクスチャ生成・pytest 双方が
    `TobitOptions(lower=, upper=)` およびリファレンス実装呼び出しに使う。
    """
    true_betas: dict[str, list[float]] = {}
    bounds: dict[str, list[float | None]] = {}
    for scenario in TOBIT_SCENARIOS:
        df, true_beta, (lower, upper) = generate_censored_regression_dataset(
            scenario
        )
        df.write_csv(output_dir / f"tobit_{scenario}.csv")
        true_betas[scenario] = true_beta.tolist()
        bounds[scenario] = [lower, upper]
    (output_dir / "tobit_true_beta.json").write_text(
        json.dumps(true_betas, indent=2)
    )
    (output_dir / "tobit_censoring_bounds.json").write_text(
        json.dumps(bounds, indent=2)
    )


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

    _freeze_tobit(output_dir)


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
