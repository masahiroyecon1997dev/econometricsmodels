"""FE-2SLS（`FEIV`）のplmによるクロスチェック用フィクスチャ
（tests/fixtures/benchmarks/fe_iv_plm_crosscheck.json）を生成するスクリプト。

fixest（`fe_iv.json`）とは別実装のplm（`model = "within"`の操作変数推定）の共分散行列で、係数・
標準誤差・`t`値・`p`値・信頼区間・係数の分散共分散行列を比べる。plmが検証するのはサンドイッチの
中身（構造残差と`X̂`）で、小標本補正はfixest・本実装の規約をスカラー倍で手計算する
（`benchmark/panel/references/run_plm_fe_iv_benchmark.R`のヘッダが正本）。

対象は1-wayの個体FEの`classical`・`hc1`・`cluster`（個体）・`dk`（本実装の既定バンド幅と、
明示バンド幅1・3・`T-2`）。plmは2-wayのwithinに対するクラスター・SCC共分散行列を持たない。ケースは
`benchmark/panel/fe_iv_cases.py`の`plm_cases()`。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_iv_plm_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.fe_iv_cases import Case, plm_cases
from benchmark.panel.references.r import default_dk_bandwidth, run_fe_iv_plm_r


def plm_formula(case: Case) -> str:
    """`plm`の操作変数構文（`y ~ 回帰変数 | 外生変数 + 操作変数`）。"""
    regressors = " + ".join([*case.x_exog, *case.x_endog])
    exogenous = " + ".join([*case.x_exog, *case.instruments])
    return f"y ~ {regressors} | {exogenous}"


def run_case(case: Case) -> dict:
    maxlag = None
    if case.cov.cov_type == "dk":
        maxlag = (
            case.cov.dk_bandwidth
            if case.cov.dk_bandwidth is not None
            else default_dk_bandwidth(case.csv_path)
        )
    return run_fe_iv_plm_r(
        case.csv_path,
        plm_formula(case),
        case.cov.cov_type,
        maxlag=maxlag,
    )


def build_fixtures() -> dict:
    fixtures: dict = {}
    for case in plm_cases():
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = run_case(case)
    fixtures["_meta"] = {
        "method": "fe_iv",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "plm (R) + sandwich",
        "plm_version": subprocess.run(
            ["Rscript", "-e", 'cat(as.character(packageVersion("plm")))'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout,
        "note": (
            "plm(model='within')の操作変数推定。classicalはvcov、hc1はvcovHC(white1, HC0)×n/(n-k-G)、"
            "clusterはvcovHC(arellano, HC0)×G/(G-1)·(n-1)/(n-k-1)、dkはvcovSCC(HC0, maxlag)×"
            "T/(T-1)·(n-1)/(n-k-G)。t検定の自由度はclassical・hc1がn-k-G、clusterがG-1、dkがT-1。"
            "plmのHC2・HC3はレバレッジに固定効果の分を含まないため使わない。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "fe_iv_plm_crosscheck.json",
        description=__doc__,
    )
