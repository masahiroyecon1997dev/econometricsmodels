"""系統（linear/nonlinear/iv）をまたいで使う共通ヘルパー。

Initiative A で関心事ごとにサブモジュールへ分割した
（経緯は `docs/planning/specs/refactoring-issue231-progress.md`「Initiative A」節）。利用側が
`from benchmark.common import DATA_DIR` のように書けるよう、公開 API をここで
re-export する（サブモジュール構成が変わっても利用側の import は変わらない）。

- `datasets_io`: `DATA_DIR` / `load_frozen_dataset` / `freeze_scenarios` /
  `run_freeze_cli`（固定CSV/JSONの読み書き・凍結CLI）。
- `dgp`: `imbalanced_cluster_groups` / `linear_predictor` /
  `correlated_design_matrix` / `apply_perfect_multicollinearity` /
  `hac_auto_lag` / `validate_choice` / `preview_dataset`（DGPまわり）。
- `dgp_constants`: 誤差項・スケール倍率の数値定数（直接 import する）。
- `reference.extract`: `extract_coef_se`（fit結果からの統計量抽出）。
- `load_wooldridge`: Wooldridgeデータローダ（直接 import する）。
"""

from __future__ import annotations

from benchmark.common.constants import (
    MROZ_FORMULA,
    SYNTHETIC_FORMULA,
    WEIGHT_COLUMN_NAME,
)
from benchmark.common.datasets_io import (
    BENCHMARKS_DIR,
    DATA_DIR,
    freeze_scenarios,
    load_frozen_dataset,
    run_freeze_cli,
)
from benchmark.common.dgp import (
    apply_perfect_multicollinearity,
    correlated_design_matrix,
    hac_auto_lag,
    imbalanced_cluster_groups,
    linear_predictor,
    preview_dataset,
    validate_choice,
)
from benchmark.common.driver import run_fixture_cli
from benchmark.common.reference.extract import extract_coef_se

__all__ = [
    "BENCHMARKS_DIR",
    "DATA_DIR",
    "MROZ_FORMULA",
    "SYNTHETIC_FORMULA",
    "WEIGHT_COLUMN_NAME",
    "apply_perfect_multicollinearity",
    "correlated_design_matrix",
    "extract_coef_se",
    "freeze_scenarios",
    "hac_auto_lag",
    "imbalanced_cluster_groups",
    "linear_predictor",
    "load_frozen_dataset",
    "preview_dataset",
    "run_fixture_cli",
    "run_freeze_cli",
    "validate_choice",
]
