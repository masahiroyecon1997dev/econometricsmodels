"""FEのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/fe_crosscheck.json）を
生成するスクリプト。

`tests/fixtures/benchmarks/fe.json`（linearmodels、主リファレンス）とは別に、
独立実装（R: fixest）によるクロスチェック値を生成する。役割分担は
`docs/planning/specs/panel-api-design.md`5.2節の通り。

## このフィクスチャだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.PanelOLS`が提供しないため、fixestを唯一の参照
  実装として係数・標準誤差を検証する（`linearmodels_ref.py`モジュールdoc参照）。
- **aic/bic**: `linearmodels.PanelOLS`が提供しないため、fixestのみで検証する。
- **2-way FEのr_squared_within**: `linearmodels`自身がentityのみdemeanの
  別定義を使うため、fixestの`fitstat(m, "wr2")`のみで検証する（1-wayは
  `fe.json`側のlinearmodelsの値とも一致するはずの回帰ガードとして機能する）。

## classical/hc1/hc2/hc3とclusterで許容誤差が異なる理由

classical/hc1/hc2/hc3は本実装と機械精度で一致する（実測相対誤差1e-14程度）。
**clusterのみ**、fixestの小標本補正の既定慣行（Stata流のG/(G-1)補正）が
本実装・linearmodelsと異なるため、`ssc(G.adj=FALSE, ...)`で調整してもなお
1-way相対誤差1.8e-5程度・2-way相対誤差0.2%程度の乖離が残る（規約上の系統的な
差、実装バグではない。`run_fixest_benchmark.R`のコメント参照）。テストコード側
では、この2つのグループで許容誤差を分けること（`.claude/rules/
testing-policy.md`「許容誤差」参照）。

## hacを含まない理由

`cov_type="hac"`（Driscoll-Kraay）はfixestの`vcov="DK"`の既定バンド幅公式・
小標本補正の慣行が本実装・linearmodelsと異なり、明示的にバンド幅を揃えても
標準誤差が実用的な許容誤差でも一致しないことを実測確認済みのため、hacは
`linearmodels`のみを参照実装とする単一参照実装の例外として扱う（ユーザー
確認済み）。このフィクスチャにはhacのキー自体が存在しない。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.fixtures.generate_fe_fixtures import (
    ONE_WAY_ONLY_SCENARIOS,
    TWO_WAY_SCENARIOS,
)
from benchmark.panel.references.r import run_fixest_r

NUMERIC_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS

# hc2/hc3はここでのみ検証する（fe.jsonのCOV_TYPESにclassical/hc1/cluster/hacの
# 4つしか無い理由はgenerate_fe_fixtures.py参照）。hacはモジュールdoc「hacを
# 含まない理由」の通り対象外。
COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]

WAGEPAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]
WAGEPAN_FORMULA_RHS = " + ".join(WAGEPAN_X)


def _formula(effects_suffix: str) -> str:
    return f"y ~ x1 + x2 | {effects_suffix}"


def _run_effects(scenario: str, cov_type: str, *, two_way: bool) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    formula = _formula("entity + time" if two_way else "entity")
    cluster_col = "entity" if cov_type == "cluster" else None
    return run_fixest_r(csv_path, formula, cov_type, cluster_col=cluster_col)


def _run_wagepan(csv_path: Path, cov_type: str, *, two_way: bool) -> dict:
    fe_part = (
        f"{WAGEPAN_ENTITY} + {WAGEPAN_TIME}" if two_way else WAGEPAN_ENTITY
    )
    formula = f"{WAGEPAN_Y} ~ {WAGEPAN_FORMULA_RHS} | {fe_part}"
    cluster_col = WAGEPAN_ENTITY if cov_type == "cluster" else None
    return run_fixest_r(csv_path, formula, cov_type, cluster_col=cluster_col)


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {
            "one_way": {
                cov_type: _run_effects(scenario, cov_type, two_way=False)
                for cov_type in COV_TYPES
            }
        }
        if scenario in TWO_WAY_SCENARIOS:
            fixtures[scenario]["two_way"] = {
                cov_type: _run_effects(scenario, cov_type, two_way=True)
                for cov_type in COV_TYPES
            }

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        df = load_wooldridge("wagepan")
        csv_path = tmpdir / "wagepan.csv"
        df.write_csv(csv_path)

        fixtures["wagepan"] = {
            "one_way": {
                cov_type: _run_wagepan(csv_path, cov_type, two_way=False)
                for cov_type in WAGEPAN_COV_TYPES
            },
            "two_way": {
                cov_type: _run_wagepan(csv_path, cov_type, two_way=True)
                for cov_type in WAGEPAN_COV_TYPES
            },
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fixest_version = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(packageVersion("fixest")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "fe",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "fixest (R)",
        "r_version": r_version,
        "fixest_version": fixest_version,
        "note": (
            "hc2/hc3・aic/bic・2-way FEのr_squared_withinは、linearmodelsが"
            "提供しない/別定義のためfixestのみが参照値になる単一参照実装の"
            "例外（モジュールdoc参照）。classical/hc1/hc2/hc3は本実装と機械"
            "精度で一致するが、clusterのみ小標本補正の慣行差により1-way相対"
            "誤差1.8e-5程度・2-way相対誤差0.2%程度の乖離が残る（実装バグでは"
            "ない、run_fixest_benchmark.Rのコメント参照）。hacはこのフィクス"
            "チャに含まない（モジュールdoc「hacを含まない理由」）。wagepanは"
            "fe.jsonと同じmarried/union/expersq・T=8のためhac対象外も同様。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_crosscheck.json",
        description=__doc__,
    )
