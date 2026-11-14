"""FEの3次元以上の固定効果のフィクスチャ（tests/fixtures/benchmarks/fe_multi_way.json）を
生成するスクリプト（主リファレンス: R fixest）。

3次元以上の固定効果の主リファレンスはfixest（`y ~ x1 + x2 | entity + time + c`）。
`linearmodels.PanelOLS`は3次元以上の吸収を扱えないため、独立なクロスチェックは
`AbsorbingLS`（classical・hc1だけ）で行う（`generate_fe_multi_way_crosscheck_fixtures.py`）。
HC2/HC3・cluster・dk・AIC/BIC・Within R²・F統計量はfixestが唯一の参照実装になる
（`fe_crosscheck.json`の1-way・2-wayと同じ理由）。

## 比べる量

- 係数・標準誤差・`t`値・`p`値・信頼区間・係数の分散共分散行列（cov_typeごと）。
- `df_model`・`df_resid`（fixestの`degrees_freedom(model, "k"/"resid")`）。本実装の
  `k + ΣG − (K−1)`と一致する（連結成分の数やネストを引かない数え方）。
- 行ごとの残差。固定効果の和`α+γ+δ`は、残差が一致することで行ごとに照合される
  （個々の効果は基準化で変わるため、和だけが比較できる）。

## cov_type

`classical`・`hc1`・`hc2`・`hc3`・`cluster`（個体）・`cluster_c`・`cluster_time`（効果の次元
そのものの列）・`cluster_region`（効果の外の列。個体の入れ子で、`nested`では`c`も入れ子）・
`dk`（時点列`time`、バンド幅は本実装の既定式）・`dk_bw1`（明示バンド幅1）。fixestの`fixef.tol`は1e-11
（`run_fixest_benchmark.R`参照。組合せが欠ける3次元以上では既定の1e-6だと約5e-9ずれる）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_multi_way_fixtures
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import NamedTuple

import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.multi_way_datasets import MULTI_WAY_SCENARIOS
from benchmark.panel.references.r import default_dk_bandwidth, run_fixest_r

# 固定効果の列（式の`| ...`）と、シナリオの効果の列の一覧。
EFFECT_COLUMNS = {
    scenario: [
        "entity",
        "time",
        "c",
        *(["d"] if scenario == "four_way" else []),
    ]
    for scenario in MULTI_WAY_SCENARIOS
}


class CovCase(NamedTuple):
    """フィクスチャの1ケース。`cluster`はクラスター列、`dk_bandwidth`は`dk`の明示バンド幅
    （`None`なら本実装の既定式）。"""

    key: str
    cov_type: str
    cluster: str | None = None
    dk_bandwidth: int | None = None


# `cluster_c`・`cluster_time`は効果の次元そのものの列でのクラスタリング、`cluster_region`は
# 効果の外にある列（個体の入れ子。`nested`では個体と`c`の両方が入れ子）でのクラスタリング。
COV_CASES = [
    CovCase("classical", "classical"),
    CovCase("hc1", "hc1"),
    CovCase("hc2", "hc2"),
    CovCase("hc3", "hc3"),
    CovCase("cluster", "cluster", "entity"),
    CovCase("cluster_c", "cluster", "c"),
    CovCase("cluster_time", "cluster", "time"),
    CovCase("cluster_region", "cluster", "region"),
    CovCase("dk", "dk"),
    CovCase("dk_bw1", "dk", dk_bandwidth=1),
]

# 水準が2つしかない列でのクラスタリングは、クラスター数が傾き係数の数以下になり
# 本実装が`ValidationError`にするため、`small`（`c`が2水準）では除く。
SKIPPED_CASES = {"small": {"cluster_c"}}


def cov_cases(scenario: str) -> list[CovCase]:
    """シナリオに対応するケース。"""
    skipped = SKIPPED_CASES.get(scenario, set())
    return [case for case in COV_CASES if case.key not in skipped]


# fixestのHC2/HC3は固定効果のダミー行列のCholesky分解を使うため、`c`が個体にネストして
# ダミーが線形従属になるシナリオでは失敗する（leading principal minor is not positive）。
# このシナリオのHC2/HC3だけは、列空間が同じ`entity + time`の式で求めた値を参照にする
# （`c`のダミーは個体のダミーの和なので、ダミー全体の列空間もハット行列も変わらない）。
NESTED_HC_FALLBACK_COV_TYPES = {"hc2", "hc3"}

# その場合に残す量。自由度（`entity + time`の数え方）に依存するp値・信頼区間・F検定・
# 情報量規準・df_model/df_residは、3次元の数え方と異なるため含めない。
NESTED_HC_FALLBACK_KEYS = (
    "params",
    "std_errors",
    "cov_params",
    "test_stats",
    "residuals",
)


def csv_path(scenario: str):
    return DATA_DIR / f"fe_multi_way_{scenario}.csv"


def _run(scenario: str, case: CovCase) -> dict:
    cov_type, cluster = case.cov_type, case.cluster
    path = csv_path(scenario)
    effects = EFFECT_COLUMNS[scenario]
    if scenario == "nested" and cov_type in NESTED_HC_FALLBACK_COV_TYPES:
        effects = effects[:2]
    formula = f"y ~ x1 + x2 | {' + '.join(effects)}"
    dk_lag = None
    if cov_type == "dk":
        dk_lag = (
            case.dk_bandwidth
            if case.dk_bandwidth is not None
            else default_dk_bandwidth(path)
        )
    result = run_fixest_r(
        path,
        formula,
        cov_type,
        cluster=cluster,
        dk_lag=dk_lag,
        with_degrees_and_residuals=True,
    )
    if len(effects) != len(EFFECT_COLUMNS[scenario]):
        result = {key: result[key] for key in NESTED_HC_FALLBACK_KEYS}
    return result


def _package_version(package: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f'cat(as.character(packageVersion("{package}")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def build_fixtures() -> dict:
    fixtures: dict = {
        scenario: {
            case.key: _run(scenario, case) for case in cov_cases(scenario)
        }
        for scenario in MULTI_WAY_SCENARIOS
    }
    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fixtures["_meta"] = {
        "method": "fe_multi_way",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "fixest (R)",
        "r_version": r_version,
        "fixest_version": _package_version("fixest"),
        "rows": {
            scenario: pl.read_csv(csv_path(scenario)).height
            for scenario in MULTI_WAY_SCENARIOS
        },
        "note": (
            "3次元以上の固定効果（effects=entity,time,c[,d]）の主リファレンス。"
            "fixef.tol=1e-11、ssc()は既定のまま（clusterはK.fixef=nonnested、"
            "hc1-hc3とdkはK.fixef=full）。df_model/df_residはdegrees_freedom"
            "(model, 'k'/'resid')で、本実装の k+ΣG-(K-1) と一致する。"
            "residualsは行ごとの残差（CSVの行の順）で、固定効果の和α+γ+δの照合に使う。"
            "dkのバンド幅は本実装の既定floor(4*(T/100)^(2/9))を明示的に渡す。"
            "cluster_c/cluster_timeは効果の次元そのものの列、cluster_regionは効果の外の列"
            "（個体の入れ子。nestedではcも入れ子で、fixestが全次元の入れ子を調べる経路）"
            "でのクラスタリング。smallはcが2水準のためcluster_cを除く。"
            "nestedのhc2/hc3だけは、fixestがダミー行列の線形従属で失敗するため、"
            "列空間が同じentity+timeの式の値を参照にする（df_modelだけは"
            "3次元の数え方k+ΣG-(K-1)と異なるため、このケースの葉は係数・標準誤差・"
            "共分散・t値・残差だけを持つ）。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "fe_multi_way.json",
        description=__doc__,
    )
