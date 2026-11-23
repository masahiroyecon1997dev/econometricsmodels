"""FE-2SLS（`FEIV`）のフィクスチャ照合テストが共有するヘルパー。

ケースの定義（`benchmark/panel/fe_iv_cases.py`）はフィクスチャの生成側と共通で、ここでは
ケースから`FEIV`を組み立てて推定する部分と、フィクスチャのJSONから葉を引く部分を持つ。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from econometricsmodels import FEIV, FEIVOptions

from benchmark.common import hac_auto_lag
from benchmark.panel.fe_iv_cases import Case

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"


def options_for(case: Case) -> FEIVOptions:
    """ケースの`cov_type`・クラスター・DKバンド幅・信頼水準に対応する`FEIVOptions`。
    DKの時点列は`time`、バンド幅`None`は本実装の既定式。"""
    cov = case.cov
    kwargs: dict = {
        "cov_type": cov.cov_type,
        "confidence_level": case.confidence_level,
    }
    if cov.cov_type == "cluster":
        kwargs["cluster"] = cov.cluster
    elif cov.cov_type == "dk":
        kwargs["dk_time"] = "time"
        kwargs["dk_bandwidth"] = cov.dk_bandwidth
    return FEIVOptions(**kwargs)


def fit_case(case: Case):
    """ケースの凍結CSVを読み、`FEIV`で推定する。"""
    return FEIV(
        pl.read_csv(case.csv_path),
        y="y",
        x_exog=list(case.x_exog),
        x_endog=list(case.x_endog),
        instruments=list(case.instruments),
        effects=list(case.effects),
        options=options_for(case),
    ).fit()


def expected_dk_bandwidth(case: Case) -> int:
    """ケースのDKバンド幅（明示値、`None`なら既定式`floor(4 (T/100)^(2/9))`）。"""
    if case.cov.dk_bandwidth is not None:
        return case.cov.dk_bandwidth
    return hac_auto_lag(pl.read_csv(case.csv_path)["time"].n_unique())


def leaf(fixtures: dict, case: Case) -> dict:
    """フィクスチャの`synthetic`からケースの葉を引く。"""
    node = fixtures["synthetic"]
    for key in case.path:
        node = node[key]
    return node


def overidentified_panel(
    n_entities: int = 30, n_periods: int = 6
) -> pl.DataFrame:
    """内生変数1個・除外操作変数3個（過剰識別）の合成パネル（均衡）。

    列: `y`・`x1`（外生）・`w`（内生）・`z1`〜`z3`（操作変数）・`entity`・`time`・
    `region`（個体を入れ子に含むクラスター。5個体ずつ）。誤差は不均一分散で、ロバスト重みの
    GMMの係数が2SLSから離れる（個体効果が内生変数と相関する）。
    """
    rng = np.random.default_rng(20261121)
    n = n_entities * n_periods
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    alpha = rng.normal(size=n_entities)
    gamma = rng.normal(size=n_periods)
    z1, z2, z3, x1 = rng.normal(size=(4, n))
    v = rng.normal(size=n)
    endog = 0.8 * z1 + 0.5 * z2 + 0.3 * z3 + v + alpha[entity]
    noise = (1.0 + np.abs(z1)) * rng.normal(size=n)
    y = x1 + 1.5 * endog + alpha[entity] + gamma[time] + 0.7 * v + noise
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "w": endog,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "entity": [f"e{e:02d}" for e in entity],
            "time": [f"t{t}" for t in time],
            "region": [f"r{e // 5}" for e in entity],
        }
    )
