"""FE-2SLS（`FEIV`）のHC2・HC3のKleibergen-Paap rk Wald Fのfixestによるクロスチェック用フィクスチャ
（tests/fixtures/benchmarks/fe_iv_first_stage_hc_crosscheck.json）を生成するスクリプト。

内生変数が1個のとき、rk Wald Fは同じ共分散での第一段階の頑健Wald Fに等しい。fixestはIVのHC2・HC3を
提供しないため、主リファレンスのフィクスチャ（`generate_fe_iv_fixtures.py`）には含められないが、
第一段階を通常の`feols`として回せばHC2・HC3を提供する
（`benchmark/panel/references/run_fixest_first_stage_benchmark.R`）。

## 比べる量

`FEIVResults.kleibergen_paap().rk_wald_f`だけ。ケースは`benchmark/panel/fe_iv_cases.py`の
`first_stage_hc_cases()`（1-way・2-way・3-way・`x_exog`が空のHC2・HC3。内生変数が複数のシナリオは
rk Wald Fが第一段階のWald Fに等しくないので含めない）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_iv_first_stage_hc_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.panel.fe_iv_cases import (
    Case,
    first_stage_formula,
    first_stage_hc_cases,
)
from benchmark.panel.references.r import run_fixest_first_stage_r

_MAX_WORKERS = 4


def run_case(case: Case) -> dict:
    raw = run_fixest_first_stage_r(
        case.csv_path,
        first_stage_formula(case),
        case.cov.cov_type,
        list(case.instruments),
    )
    return {"rk_wald_f": raw["rk_wald_f"]}


def _r_output(expression: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f"cat(as.character({expression}))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def build_fixtures() -> dict:
    cases = first_stage_hc_cases()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(run_case, cases))
    fixtures: dict = {}
    for case, result in zip(cases, results, strict=True):
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = result
    fixtures["_meta"] = {
        "method": "fe_iv",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "fixest (R)",
        "r_version": _r_output("getRversion()"),
        "fixest_version": _r_output('packageVersion("fixest")'),
        "note": (
            "内生変数が1個のときのrk Wald Fは、同じ共分散での第一段階の頑健Wald F（除外操作変数の係数が"
            "同時にゼロ）に等しい。fixestはIVのHC2・HC3を提供せず、iv_first_stageもIV推定として拒否する"
            "ため、第一段階（内生変数 ~ 外生変数 + 除外操作変数 | 固定効果）を通常のfeolsとして回し、"
            "wald(vcov = 'hc2'|'hc3')の統計量を取る。fixef.tol=1e-11。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_iv_first_stage_hc_crosscheck.json",
        description=__doc__,
    )
