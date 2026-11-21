"""FE-2SLS（`FEIV`）の弱操作変数診断のivDiagによるクロスチェック用フィクスチャ
（tests/fixtures/benchmarks/fe_iv_weak_diagnostics_crosscheck.json）を生成するスクリプト。

`FEIVResults.cragg_donald()`・`kleibergen_paap()`・`effective_f()`の、ダミー変数を明示した2SLS
（`generate_fe_iv_dummy_crosscheck_fixtures.py`）とは独立した実装（R: ivDiag。固定効果は
`FE`引数で吸収する）による値を固定する。IV本体の
`generate_iv_weak_diagnostics_crosscheck_fixtures.py`と同じ`run_ivdiag.R`を使う。

## 対象範囲

ivDiagが返すのは内生変数が1個のときの統計量だけで、標準誤差は robust と cluster に限る。

- **Cragg-Donald の F**: `F.standard`（全ケース）。
- **Kleibergen-Paap の rk Wald F**: robust は`F.robust`（本実装の`hc1`）、cluster は`F.cluster`。
- **有効F**: `eff_F`（robust は`hc1`、cluster）。

rk LM統計量・Stock-Yogoの臨界値・内生変数が複数のケース・DK・HC2・HC3は比べられない。
1-wayの固定効果は`classical`・`hc1`・クラスター（個体・時点・region・不均衡なregion・sector）、2-wayは
`classical`・`hc1`・`sector`のクラスター。2-wayで個体・時点・regionのクラスターを含めないのは、
ivDiag（lfe）が、固定効果のどれかがクラスターにネストするとネストしない固定効果も自由度から外すため
（fixestと本実装はネストした固定効果だけを外す）。F・有効Fが`(N - K_lfe) / (N - K)`倍ずれる。
どの固定効果もネストしない`sector`は一致する。ケースは`benchmark/panel/fe_iv_cases.py`の`ivdiag_cases()`。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_iv_weak_diagnostics_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.iv.references.ivdiag_ref import run_ivdiag_r
from benchmark.panel.fe_iv_cases import Case, ivdiag_cases


def run_case(case: Case) -> dict:
    cluster = case.cov.cluster if case.cov.cov_type == "cluster" else None
    raw = run_ivdiag_r(
        case.csv_path,
        "y",
        case.x_endog[0],
        list(case.instruments),
        list(case.x_exog),
        cluster=cluster,
        fixed_effect=",".join(case.effects),
    )
    result: dict = {"cragg_donald_f": raw["f_standard"]}
    if case.cov.cov_type == "hc1":
        result["rk_wald_f"] = raw["f_robust"]
        result["effective_f"] = raw["f_effective"]
    elif cluster is not None:
        result["rk_wald_f"] = raw["f_cluster"]
        result["effective_f"] = raw["f_effective"]
    return result


def build_fixtures() -> dict:
    fixtures: dict = {}
    for case in ivdiag_cases():
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = run_case(case)
    fixtures["_meta"] = {
        "method": "fe_iv",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "ivDiag (R)",
        "ivdiag_version": subprocess.run(
            ["Rscript", "-e", 'cat(as.character(packageVersion("ivDiag")))'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout,
        "note": (
            "ivDiagのFE引数で固定効果を吸収。cragg_donald_fはF.standard（cov_typeによらない）、"
            "rk_wald_fはhc1がF.robust・クラスターがF.cluster、effective_fはeff_F（hc1はrobust、"
            "クラスターはcluster。prec=14）。classicalはCragg-Donaldだけ。"
            "内生変数が複数のシナリオ・3-way・DK・HC2・HC3は比べられない。2-wayのクラスターは、"
            "固定効果がネストしないsectorだけ（ネストするとivDiagの自由度の数え方が違う）。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_iv_weak_diagnostics_crosscheck.json",
        description=__doc__,
    )
