"""FE-GMM（`FEIV`）の独立クロスチェックのフィクスチャ
（tests/fixtures/benchmarks/fe_iv_gmm_crosscheck.json）を生成するスクリプト。

主リファレンス（`generate_fe_iv_gmm_fixtures.py`）と同じwithin変換後のデータを、定数項なしの式で
R `momentfit`（`benchmark/iv/references/run_momentfit.R`）に渡す。`linearmodels`とは別の実装の
部品で、係数・重み行列・共分散行列の本体・Hansen Jを作る。推定方式は`one_step`・`two_step`・
`iterated`（回数を固定）、重みは`classical`・`robust`・`cluster`、標準誤差は`classical`・`hc1`・
`cluster`。標準誤差の小標本補正のスカラー（手書き）は
`benchmark/panel/references/within_gmm_r.py`のモジュールdocが正本。
ケースは`benchmark/panel/fe_iv_gmm_cases.py`の`crosscheck_cases()`。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_iv_gmm_crosscheck_fixtures
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
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.fe_iv_gmm_cases import GmmCase, crosscheck_cases
from benchmark.panel.references.within_gmm_r import run_within_momentfit

_MAX_WORKERS = 4


def _run(case: GmmCase) -> dict:
    return run_within_momentfit(pd.read_csv(case.base.csv_path), case)


def _r(expression: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f"cat(as.character({expression}))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def build_fixtures() -> dict:
    cases = crosscheck_cases()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(_run, cases))
    fixtures: dict = {}
    for case, result in zip(cases, results, strict=True):
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = result
    fixtures["_meta"] = {
        "method": "fe_iv_gmm",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "momentfit (R, within-transformed data)",
        "r_version": _r("getRversion()"),
        "momentfit_version": _r('packageVersion("momentfit")'),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "note": (
            "固定効果を除いた列に定数項なしの式でmomentfitを当てる（initW=tsls、centeredVcov=FALSE）。"
            "one_stepは重み(Z'Z/n)^-1で推定しJはclassicalのS（Sargan）、iteratedは初期推定から回数を"
            "固定して重みを作り直す。標準誤差はrun_momentfit.Rの補正(n-k)を、classical・hc1でdf_resid、"
            "clusterでn-K（fixestのnonnested）に直したスカラー（手書き）を掛けた値。overid_statisticは"
            "classical重みとone_stepで(n-n_absorbed)/nを掛けた本実装の尺度。z分布・chi2。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "fe_iv_gmm_crosscheck.json",
        description=__doc__,
    )
