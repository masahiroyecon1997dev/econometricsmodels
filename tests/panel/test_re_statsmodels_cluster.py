"""RE の cluster（t検定の自由度 `G-1`）の statsmodels によるクロスチェック。

`test_re_crosscheck.py`（plm）はREのcluster/dkの参照値だが、plmはz検定を返す
ため、t統計量・p値・信頼区間の自由度（clusterで`G-1`）は
`run_plm_benchmark.R`が本実装と同じ規約で手計算している。plmが検証するのは
標準誤差（補正係数込み）までで、自由度の規約そのものは検証していない。

ここでは`tests/fixtures/benchmarks/re_statsmodels_cluster.json`
（`benchmark/panel/fixtures/generate_re_statsmodels_cluster_fixtures.py`で生成）
を用いる。plmが準偏差変換した応答・設計行列にstatsmodelsのOLS
（`cov_type="cluster"`、`use_t=True`）を当てた値で、statsmodelsが自前で
クラスターSE・t統計量・p値・信頼区間・推論の自由度・傾き係数の同時F検定を
計算する。

## 検証範囲

バランスパネルの`cov_type="cluster"`（entityクラスター）のみ。分散成分はplm
推定のため、不均衡パネルは対象外。DK（Driscoll-Kraay）の自由度`T-1`は
statsmodelsが同じ規約を持たず（`hac-groupsum`の`df_resid_inference`は`T-1`に
ならない）、第2リファレンスがない（`docs/guide/verification.md`参照）。

役割分担:
    - plm（REのcluster/dk、標準誤差・F統計量まで）: `test_re_crosscheck.py`
    - statsmodels（REのclusterの自由度`G-1`、標準誤差・p値・信頼区間）:
      このファイル
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _constants import DATA_DIR
from _helpers import load_wooldridge_dataset
from _tolerances import TOLERANCES
from econometricsmodels import RE, REOptions

from benchmark.common import WAGEPAN_ENTITY, WAGEPAN_X, WAGEPAN_Y
from benchmark.panel.fixtures.generate_re_statsmodels_cluster_fixtures import (
    SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_statsmodels_cluster.json"
)

RTOL = TOLERANCES["re_statsmodels_cluster"]["rtol"]
ATOL = TOLERANCES["re_statsmodels_cluster"]["atol"]
ATOL_P_VALUE = TOLERANCES["re_statsmodels_cluster"]["atol_p_value"]
RTOL_P_VALUE = TOLERANCES["re_statsmodels_cluster"]["rtol_p_value"]

_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)
# p値は絶対誤差の下限を設けず相対誤差だけで比較する（`test_fe_crosscheck.py`と
# 同じ理由）。
_assert_p_close = partial(assert_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE)
_assert_p_dict_close = partial(
    assert_dict_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE
)


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _check(res, ref: dict, n_clusters: int, label: str) -> None:
    # 参照値の自由度がG-1であること（statsmodelsがネイティブに決めた値）。
    # 本実装が同じ自由度を使っていることは、下のp値・信頼区間の一致が示す。
    assert ref["df_resid_inference"] == n_clusters - 1, label
    assert ref["f_df_denom"] == n_clusters - 1, label
    assert res.stat_df == n_clusters - 1, label
    assert res.f_df_denom == n_clusters - 1, label

    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(
        res.test_stats, ref["test_stats"], f"{label}/test_stats"
    )
    _assert_p_dict_close(res.p_values, ref["p_values"], f"{label}/p_values")
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        _assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    _assert_close(res.f_statistic, ref["f_statistic"], f"{label}/f_statistic")
    _assert_p_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synthetic_cluster_matches_statsmodels(reference, scenario):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    res = RE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=REOptions(cov_type="cluster"),
    ).fit()

    _check(
        res,
        reference[scenario],
        df["entity"].n_unique(),
        f"{scenario}/cluster",
    )


def test_wagepan_cluster_matches_statsmodels(reference):
    df = load_wooldridge_dataset("wagepan")
    res = RE(
        df,
        y=WAGEPAN_Y,
        x=WAGEPAN_X,
        entity=WAGEPAN_ENTITY,
        options=REOptions(cov_type="cluster"),
    ).fit()

    _check(
        res,
        reference["wagepan"],
        df[WAGEPAN_ENTITY].n_unique(),
        "wagepan/cluster",
    )
