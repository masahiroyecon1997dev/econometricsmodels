"""FE-GMM（`FEGMM`）のフィクスチャ照合テストが共有するヘルパー。

ケースの定義（`benchmark/panel/fe_gmm_cases.py`）はフィクスチャの生成側と共通で、ここでは
ケースから`FEGMM`を組み立てて推定する部分と、フィクスチャのJSONから葉を引く部分を持つ。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from econometricsmodels import FEGMM, FEGMMOptions

from benchmark.common import hac_auto_lag
from benchmark.panel.fe_gmm_cases import ITERATED_TOL, CStatCase, GmmCase

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"


def options_for(case: GmmCase) -> FEGMMOptions:
    """ケースの推定方式・重み・`cov_type`・クラスター・DKバンド幅・信頼水準に対応する`FEGMMOptions`。
    DKの時点列は`time`、バンド幅`None`は本実装の既定式。`iterated`は最大回数まで回し、収束しなくても
    エラーにしない。"""
    cov, spec = case.base.cov, case.spec
    kwargs: dict = {
        "cov_type": cov.cov_type,
        "confidence_level": case.base.confidence_level,
        "gmm_type": spec.gmm_type,
        "cluster": case.cluster,
    }
    if spec.weight is not None:
        kwargs["gmm_weight_type"] = spec.weight
    if spec.gmm_type == "iterated":
        kwargs["gmm_max_iter"] = spec.max_iter
        kwargs["gmm_tol"] = ITERATED_TOL
        kwargs["raise_on_non_convergence"] = False
    if cov.cov_type == "dk":
        kwargs["dk_time"] = "time"
        kwargs["dk_bandwidth"] = cov.dk_bandwidth
    return FEGMMOptions(**kwargs)


def fit_case(case: GmmCase):
    """ケースの凍結CSVを読み、`FEGMM`で推定する。"""
    base = case.base
    return FEGMM(
        pl.read_csv(base.csv_path),
        y="y",
        x_exog=list(base.x_exog),
        x_endog=list(base.x_endog),
        instruments=list(base.instruments),
        effects=list(base.effects),
        options=options_for(case),
    ).fit()


def fit_c_stat_case(case: CStatCase):
    """C統計量のケースの`FEGMM`を既定の2段階GMM（重みはケースの`weight`）で推定し、`c_stat()`を返す。"""
    base = case.base
    options = FEGMMOptions(
        gmm_type="two_step",
        gmm_weight_type=case.weight,
        cluster=case.cluster,
    )
    result = FEGMM(
        pl.read_csv(base.csv_path),
        y="y",
        x_exog=list(base.x_exog),
        x_endog=list(base.x_endog),
        instruments=list(base.instruments),
        effects=list(base.effects),
        options=options,
    ).fit()
    return result.c_stat(
        variables=list(case.variables) or None,
        instruments=list(case.instruments) or None,
    )


def expected_dk_bandwidth(case: GmmCase) -> int:
    """ケースのDKバンド幅（明示値、`None`なら既定式`floor(4 (T/100)^(2/9))`）。"""
    if case.base.cov.dk_bandwidth is not None:
        return case.base.cov.dk_bandwidth
    return hac_auto_lag(pl.read_csv(case.base.csv_path)["time"].n_unique())


def leaf(fixtures: dict, case) -> dict:
    """フィクスチャの`synthetic`からケースの葉を引く。"""
    node = fixtures["synthetic"]
    for key in case.path:
        node = node[key]
    return node
