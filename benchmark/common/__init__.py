"""系統（linear/nonlinear/iv）をまたいで使う共通ヘルパー。

実体は `helpers.py`（旧 `benchmark/_common.py`）。利用側が
`from benchmark.common import DATA_DIR` のように書けるよう、公開 API をここで
re-export する。内部ファイルの分割（`datasets_io.py` / `dgp.py` 等への細分化。
`docs/planning/specs/benchmark-restructure-design.md` 4章）は後続ステップで行う
予定で、その際もこの re-export により利用側の import は変わらない。

DGP 定数（旧 `benchmark/_dgp_constants.py`）は `benchmark.common.dgp_constants`、
Wooldridge ローダは `benchmark.common.load_wooldridge` に分けたまま
（それぞれ直接 import する）。
"""

from __future__ import annotations

from benchmark.common.helpers import (
    DATA_DIR,
    apply_perfect_multicollinearity,
    correlated_design_matrix,
    extract_coef_se,
    freeze_scenarios,
    hac_auto_lag,
    imbalanced_cluster_groups,
    linear_predictor,
    load_frozen_dataset,
    preview_dataset,
    run_freeze_cli,
    validate_choice,
)

__all__ = [
    "DATA_DIR",
    "apply_perfect_multicollinearity",
    "correlated_design_matrix",
    "extract_coef_se",
    "freeze_scenarios",
    "hac_auto_lag",
    "imbalanced_cluster_groups",
    "linear_predictor",
    "load_frozen_dataset",
    "preview_dataset",
    "run_freeze_cli",
    "validate_choice",
]
