"""FE-2SLS（`FEIV`）のフィクスチャ照合テストが共有するヘルパー。

ケースの定義（`benchmark/panel/fe_iv_cases.py`）はフィクスチャの生成側と共通で、ここでは
ケースから`FEIV`を組み立てて推定する部分と、フィクスチャのJSONから葉を引く部分を持つ。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from econometricsmodels import FEIV, FEIVOptions

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


def leaf(fixtures: dict, case: Case) -> dict:
    """フィクスチャの`synthetic`からケースの葉を引く。"""
    node = fixtures["synthetic"]
    for key in case.path:
        node = node[key]
    return node
