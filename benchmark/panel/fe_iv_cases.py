"""FE-2SLS（`FEIV`）のフィクスチャのケース定義。

フィクスチャを生成するスクリプト（`fixtures/generate_fe_iv_*.py`）と、それを読むテスト
（`tests/panel/test_fe_iv_*.py`）が同じ定義を使う（ケースのキーが生成側とテスト側でずれないように、
ここを唯一の定義元にする）。

1つのケース（`Case`）は、データのシナリオ・固定効果・外生変数・`cov_type`の組で、フィクスチャの
JSONでは`path`（キーの並び）で指す。

- `<scenario>/<cov>`: 1-way（`effects=["entity"]`）。全シナリオ。
- `two_way/<scenario>/<cov>`: 2-way（`["entity", "time"]`）。`TWO_WAY_SCENARIOS`。
- `three_way/<cov>`: 3-way（`["entity", "time", "sector"]`）。`three_way`シナリオ。
- `x_exog_empty/<cov>`: `baseline`で`x_exog=[]`。
- `confidence_level/<cov>`: `baseline`で信頼水準0.90。

`<cov>`は`CovCase.key`。`cluster_time`・`cluster_region`・`cluster_region_uneven`は固定効果の外の列
（または次元そのものの列）でのクラスタリング、`dk`は本実装の既定バンド幅、`dk_bw1`は明示バンド幅1。
"""

from __future__ import annotations

from typing import NamedTuple

import polars as pl

from benchmark.common import DATA_DIR
from benchmark.panel.fe_iv_datasets import SCENARIOS, scenario_columns

CONFIDENCE_LEVEL_NON_DEFAULT = 0.90

# 2-wayを回すシナリオ（不均衡と、内生変数・時点数が違うもの）。
TWO_WAY_SCENARIOS = ["baseline", "unbalanced", "multi_endog", "long_panel"]

# クラスター数が`k`（傾き係数の数）以下になる組は`ValidationError`になるため、作らない。
# `short_panel`は時点3、`small_n`は個体8・時点4・region2つ、`long_panel`はregion3つ。
_MIN_CLUSTERS_MARGIN = 1

# DK（Driscoll-Kraay）の明示バンド幅1を作れる最小の時点数（バンド幅は`T-1`未満）。
_DK_EXPLICIT_MIN_PERIODS = 3


class CovCase(NamedTuple):
    """`cov_type`の1ケース。`cluster`はクラスター列、`dk_bandwidth`は`dk`の明示バンド幅
    （`None`なら本実装の既定式）。"""

    key: str
    cov_type: str
    cluster: str | None = None
    dk_bandwidth: int | None = None


CLASSICAL = CovCase("classical", "classical")
HC1 = CovCase("hc1", "hc1")
CLUSTER_ENTITY = CovCase("cluster", "cluster", "entity")
CLUSTER_TIME = CovCase("cluster_time", "cluster", "time")
CLUSTER_REGION = CovCase("cluster_region", "cluster", "region")
CLUSTER_REGION_UNEVEN = CovCase(
    "cluster_region_uneven", "cluster", "region_uneven"
)
CLUSTER_SECTOR = CovCase("cluster_sector", "cluster", "sector")
DK = CovCase("dk", "dk")
DK_BW1 = CovCase("dk_bw1", "dk", dk_bandwidth=1)


class Case(NamedTuple):
    """フィクスチャの1ケース。`path`はJSONでのキーの並び。"""

    path: tuple[str, ...]
    scenario: str
    effects: tuple[str, ...]
    x_exog: tuple[str, ...]
    x_endog: tuple[str, ...]
    instruments: tuple[str, ...]
    cov: CovCase
    confidence_level: float = 0.95

    @property
    def label(self) -> str:
        return "/".join(self.path)

    @property
    def csv_path(self):
        return DATA_DIR / f"fe_iv_{self.scenario}.csv"

    @property
    def n_effects(self) -> int:
        return len(self.effects)


def _dimensions(scenario: str) -> tuple[int, int]:
    """シナリオの個体数と時点数（CSVを読まずに、生成側の規模から決める）。"""
    return {
        "short_panel": (60, 3),
        "long_panel": (12, 20),
        "small_n": (8, 4),
    }.get(scenario, (40, 6))


def _k(scenario: str) -> int:
    """傾き係数の数（外生1＋内生）。"""
    return 1 + len(scenario_columns(scenario)["x_endog"])


def _n_regions(n_entities: int) -> int:
    return -(-n_entities // 5)


def cov_cases(scenario: str, n_effects: int) -> list[CovCase]:
    """シナリオと固定効果の数で作れる`cov_type`のケース。

    Args:
        scenario: データのシナリオ。
        n_effects: 固定効果の列の数（1・2・3）。
    """
    n_entities, n_periods = _dimensions(scenario)
    k = _k(scenario)
    cases = [CLASSICAL, HC1, CLUSTER_ENTITY]
    # 時点でのクラスター（個体の固定効果の外。2-wayでは固定効果の次元そのもの）。
    if n_periods >= k + _MIN_CLUSTERS_MARGIN:
        cases.append(CLUSTER_TIME)
    if n_effects <= 2:
        if _n_regions(n_entities) >= k + _MIN_CLUSTERS_MARGIN:
            cases.append(CLUSTER_REGION)
        if scenario == "baseline" and n_effects == 1:
            cases.append(CLUSTER_REGION_UNEVEN)
    else:
        # 3-wayでは、3つ目の効果の列でのクラスタリングも加える。
        cases.append(CLUSTER_REGION)
        cases.append(CLUSTER_SECTOR)
    cases.append(DK)
    if n_periods >= _DK_EXPLICIT_MIN_PERIODS + 1 and n_effects <= 2:
        cases.append(DK_BW1)
    return cases


def _case(
    path: tuple[str, ...],
    scenario: str,
    effects: tuple[str, ...],
    cov: CovCase,
    *,
    x_exog: tuple[str, ...] | None = None,
    confidence_level: float = 0.95,
) -> Case:
    cols = scenario_columns(scenario)
    return Case(
        path=path,
        scenario=scenario,
        effects=effects,
        x_exog=tuple(cols["x_exog"]) if x_exog is None else x_exog,
        x_endog=tuple(cols["x_endog"]),
        instruments=tuple(cols["instruments"]),
        cov=cov,
        confidence_level=confidence_level,
    )


def fixest_cases() -> list[Case]:
    """主リファレンス（fixest）のケース。HC2・HC3はfixestがIVで提供しないため含めない。"""
    cases: list[Case] = []
    for scenario in SCENARIOS:
        if scenario == "three_way":
            continue
        for cov in cov_cases(scenario, 1):
            cases.append(
                _case((scenario, cov.key), scenario, ("entity",), cov)
            )
    for scenario in TWO_WAY_SCENARIOS:
        for cov in cov_cases(scenario, 2):
            cases.append(
                _case(
                    ("two_way", scenario, cov.key),
                    scenario,
                    ("entity", "time"),
                    cov,
                )
            )
    for cov in cov_cases("three_way", 3):
        cases.append(
            _case(
                ("three_way", cov.key),
                "three_way",
                ("entity", "time", "sector"),
                cov,
            )
        )
    for cov in (CLASSICAL, HC1, CLUSTER_ENTITY):
        cases.append(
            _case(
                ("x_exog_empty", cov.key),
                "baseline",
                ("entity",),
                cov,
                x_exog=(),
            )
        )
    for cov in (CLASSICAL, HC1, CLUSTER_ENTITY, DK):
        cases.append(
            _case(
                ("confidence_level", cov.key),
                "baseline",
                ("entity",),
                cov,
                confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
            )
        )
    return cases


HC2 = CovCase("hc2", "hc2")
HC3 = CovCase("hc3", "hc3")


def _with_cov(case: Case, cov: CovCase) -> Case:
    return case._replace(path=(*case.path[:-1], cov.key), cov=cov)


def dummy_cases() -> list[Case]:
    """ダミー変数を明示した2SLS（linearmodels・numpy）のケース。

    classical・hc1と、1-wayの個体FE × 時点クラスター、全構造のHC2・HC3（numpyだけが参照）。
    個体クラスターと2-wayのクラスターは、`K`の数え方がlinearmodelsと違うため含めない
    （`dummy_2sls_ref.py`のモジュールdoc）。
    """
    cases: list[Case] = []
    for case in fixest_cases():
        if case.path[0] == "confidence_level":
            continue
        key = case.cov.key
        is_cluster_time = key == "cluster_time" and case.n_effects == 1
        if key in ("classical", "hc1") or is_cluster_time:
            cases.append(case)
        if key == "classical":
            cases.extend([_with_cov(case, HC2), _with_cov(case, HC3)])
    return cases


def plm_cases() -> list[Case]:
    """plm（`model = "within"`）のケース。1-wayの個体FE、classical・hc1・個体クラスター・DK。

    plmは2-wayのwithinに対するクラスター・SCC共分散行列を持たず、クラスター列も個体・時点
    しか指定できない。
    """
    keep = {"classical", "hc1", "cluster", "dk", "dk_bw1"}
    return [
        case
        for case in fixest_cases()
        if case.n_effects == 1
        and case.cov.key in keep
        and case.path[0] not in ("x_exog_empty", "confidence_level")
    ]


def ivdiag_cases() -> list[Case]:
    """ivDiagのケース。内生変数が1個のシナリオの、1-wayのclassical・hc1・クラスターと、
    2-wayのclassical・hc1。

    ivDiagは2-wayの固定効果でのクラスター頑健なFが、本実装ともfixestの第一段階のWald Fとも
    一致しない（原因は追っていない）ため、2-wayのクラスターは含めない。
    """
    keep_one_way = {
        "classical",
        "hc1",
        "cluster",
        "cluster_time",
        "cluster_region",
        "cluster_region_uneven",
    }
    cases = []
    for case in fixest_cases():
        if len(case.x_endog) != 1 or case.path[0] in (
            "confidence_level",
            "x_exog_empty",
            "three_way",
        ):
            continue
        keep = (
            keep_one_way
            if case.n_effects == 1
            else {"classical", "hc1"}
            if case.n_effects == 2
            else set()
        )
        if case.cov.key in keep:
            cases.append(case)
    return cases


def formula(case: Case) -> str:
    """fixestの`feols`の式（固定効果・操作変数構文）。"""
    exog = " + ".join(case.x_exog) if case.x_exog else "1"
    fe = " + ".join(case.effects)
    endog = " + ".join(case.x_endog)
    inst = " + ".join(case.instruments)
    return f"y ~ {exog} | {fe} | {endog} ~ {inst}"


def n_rows(case: Case) -> int:
    return pl.scan_csv(case.csv_path).select(pl.len()).collect().item()
