"""FE-GMM（`FEGMMResults.c_stat()`）の独立クロスチェックのフィクスチャ（tests/fixtures/benchmarks/fe_gmm_c_statistic_crosscheck.json）を生成するスクリプト。

同じwithin変換後のデータを、R `momentfit`の部品でC統計量にする
（`benchmark/iv/references/run_momentfit_c_stat.R`。Rにこの検定を返すパッケージは無いので、momentfitの
モーメント共分散・GMM推定・Hansen JでHayashiの手順をスクリプトが組み立てる）。
ケースは`benchmark/panel/fe_gmm_cases.py`の`c_stat_cases()`（構造 × 重み × 検定）。
定義とスケール（`classical`重みの`(n − n_absorbed)/n`）は
`benchmark/panel/references/within_c_stat_ref.py`のモジュールdocが正本。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_gmm_c_statistic_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import scipy

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.panel.fe_gmm_cases import CStatCase, c_stat_cases
from benchmark.panel.references.within_c_stat_ref import (
    run_within_c_stat_momentfit,
)


def _run(case: CStatCase) -> dict:
    return run_within_c_stat_momentfit(pd.read_csv(case.base.csv_path), case)


_MAX_WORKERS = 4


def _r(expression: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f"cat(as.character({expression}))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def build_fixtures() -> dict:
    cases = c_stat_cases()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(_run, cases))
    fixtures: dict = {}
    for case, result in zip(cases, results, strict=True):
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = result
    fixtures["_meta"] = {
        "method": "fe_gmm_c_statistic",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "momentfit (R, within-transformed data)",
        "r_version": _r("getRversion()"),
        "momentfit_version": _r('packageVersion("momentfit")'),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "note": (
            "固定効果を除いた列に定数項なしの式でmomentfitの部品を当てる（run_momentfit_c_stat.R）。classical重みはstatistic・j_*に(n-n_absorbed)/nを掛けた本実装の尺度。pはchi2。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_gmm_c_statistic_crosscheck.json",
        description=__doc__,
    )
