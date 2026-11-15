"""FE-GMM（`FEIVResults.c_stat()`）の主リファレンスのフィクスチャ（tests/fixtures/benchmarks/fe_iv_gmm_c_statistic.json）を生成するスクリプト。

固定効果をwithin変換で除いた列に、numpyで書いた独立のC統計量（`benchmark/iv/references/c_statistic_oracle.py`。
IV本体の`c_stat()`の参照で、linearmodelsの`c_stat`と照合済み）を当てる。
ケースは`benchmark/panel/fe_iv_gmm_cases.py`の`c_stat_cases()`（構造 × 重み × 検定）。
定義とスケール（`classical`重みの`(n − n_absorbed)/n`）は
`benchmark/panel/references/within_c_stat_ref.py`のモジュールdocが正本。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_iv_gmm_c_statistic_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import scipy

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.panel.fe_iv_gmm_cases import CStatCase, c_stat_cases
from benchmark.panel.references.within_c_stat_ref import run_within_c_stat


def _run(case: CStatCase) -> dict:
    return run_within_c_stat(pd.read_csv(case.base.csv_path), case)


def build_fixtures() -> dict:
    cases = c_stat_cases()
    results = [_run(case) for case in cases]
    fixtures: dict = {}
    for case, result in zip(cases, results, strict=True):
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = result
    fixtures["_meta"] = {
        "method": "fe_iv_gmm_c_statistic",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "numpy c_statistic oracle (within-transformed data)",
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "note": (
            "固定効果を除いた列に定数項なしのC統計量（Hayashi 2000、Baum-Schaffer-Stillman 2003）をnumpyで当てる。Sは生の和。classical重みはstatistic・j_*に(n-n_absorbed)/nを掛けた本実装の尺度。pはchi2。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_iv_gmm_c_statistic.json",
        description=__doc__,
    )
