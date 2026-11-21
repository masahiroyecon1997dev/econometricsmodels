"""FE の3次元以上の固定効果（`effects`が3〜4列）の主リファレンス（R fixest）との数値照合。

`tests/fixtures/benchmarks/fe_multi_way.json`と`fe_multi_way_cov.json`
（`benchmark/panel/fixtures/generate_fe_multi_way_fixtures.py`で生成）を読み、
`MULTI_WAY_SCENARIOS`の全シナリオ×`COV_CASES`の全cov_type（classical・hc1・hc2・hc3・
個体でのcluster・3つ目の効果でのcluster・dk）で、係数・標準誤差・`t`値・`p`値・信頼区間・
係数の分散共分散行列・AIC/BIC/対数尤度・Within R²・F統計量・自由度・行ごとの残差を、
相対誤差1e-8で厳密比較する。

3次元以上の固定効果は`linearmodels.PanelOLS`が扱えないため、fixestが主リファレンスで、
独立なクロスチェックは`linearmodels.iv.absorbing.AbsorbingLS`（classical・hc1だけ、
`test_fe_multi_way_crosscheck.py`）。HC2/HC3・cluster・dkはfixestが唯一の参照実装になる
（`docs/guide/verification-details.md`の単一リファレンスの例外）。

## 固定効果の和の照合

個々の効果は基準化（連結成分ごとに2つ目以降の次元の基準水準を0）で値が変わり、入れ子の次元では
個別に識別されない。そこで行ごとの残差`y − Xβ − (α+γ+δ)`をfixestの`resid()`と比べる。残差が
一致することが、効果の和`α+γ+δ`が行ごとに一致することと同値になる。

## `nested`の3つ目の列でのcluster（fixestとKの数え方が違う）

`nested`は`c`が個体に入れ子で、`cluster="c"`では個体の水準が全て`c`のクラスターに入れ子になる。
本実装は、クラスターに入れ子の固定効果の次元（個体と`c`）を全て`K`から除く（`reghdfe`と同じ考え方）。
fixestの`ssc()`は、クラスター列自身が固定効果の次元のときは他の次元の入れ子を調べない近道を取るため、
`c`だけを除いて`K`が大きくなる（このデータでfixest 31、本実装8）。そのため標準誤差と共分散行列は
fixestと一致せず、係数・残差・自由度は一致する。`K`の差だけで説明できることは
`test_nested_cluster_on_the_fixed_effect_differs_from_fixest_only_by_k`が確かめる。

## `nested`のHC2/HC3

fixestのHC2/HC3は固定効果のダミー行列のCholesky分解を使うため、`c`が個体に入れ子で
ダミーが線形従属になる`nested`では失敗する。このシナリオのHC2/HC3だけは、列空間が同じ
`entity + time`の式の値を参照にし、自由度に依存する量（`p`値・信頼区間・F統計量・AIC/BIC・
`df_*`）は比べない（`generate_fe_multi_way_fixtures.py`参照）。

役割分担:
    - 構造・API・LSDVオラクルとの照合: `test_fe_multi_way.py`
    - 主リファレンス（R: fixest）との数値照合: このファイル
    - 独立実装（linearmodels AbsorbingLS、classical・hc1）とのクロスチェック:
      `test_fe_multi_way_crosscheck.py`
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import (
    assert_close,
    assert_cov_params_close,
    assert_dict_close,
    assert_fields_close,
    assert_inference_close,
)
from _constants import DATA_DIR
from _fixture_cov import load_fixtures_with_cov, mark_cov_consumed
from _tolerances import TOLERANCES
from econometricsmodels import FE, FEOptions

from benchmark.panel.fixtures.generate_fe_multi_way_fixtures import (
    EFFECT_COLUMNS,
    NESTED_HC_FALLBACK_COV_TYPES,
    CovCase,
    cov_cases,
)
from benchmark.panel.multi_way_datasets import MULTI_WAY_SCENARIOS

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "fe_multi_way.json"
)

RTOL = TOLERANCES["fe_multi_way_reference"]["rtol"]
ATOL = TOLERANCES["fe_multi_way_reference"]["atol"]
ATOL_P_VALUE = TOLERANCES["fe_multi_way_reference"]["atol_p_value"]
RTOL_P_VALUE = TOLERANCES["fe_multi_way_reference"]["rtol_p_value"]

CASES = [
    (scenario, case.key)
    for scenario in MULTI_WAY_SCENARIOS
    for case in cov_cases(scenario)
]
CASES_BY_KEY = {
    (scenario, case.key): case
    for scenario in MULTI_WAY_SCENARIOS
    for case in cov_cases(scenario)
}

_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_fields_close = partial(assert_fields_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)
_assert_p_close = partial(assert_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE)
_assert_p_dict_close = partial(
    assert_dict_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE
)


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)


def _options(case: CovCase) -> FEOptions:
    """フィクスチャのケースに対応する`FEOptions`。`dk`は時点列`time`、バンド幅は`None`なら既定。"""
    if case.cov_type == "cluster":
        return FEOptions(cov_type="cluster", cluster=case.cluster)
    if case.cov_type == "dk":
        return FEOptions(
            cov_type="dk", dk_time="time", dk_bandwidth=case.dk_bandwidth
        )
    return FEOptions(cov_type=case.cov_type)


def _fit(scenario: str, key: str):
    df = pl.read_csv(DATA_DIR / f"fe_multi_way_{scenario}.csv")
    return FE(
        df,
        y="y",
        x=["x1", "x2"],
        effects=EFFECT_COLUMNS[scenario],
        options=_options(CASES_BY_KEY[(scenario, key)]),
    ).fit()


# fixestと標準誤差が一致しない組（モジュールdoc「`nested`の3つ目の列でのcluster」）。
DIFFERS_FROM_FIXEST_BY_K = ("nested", "cluster_c")


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("scenario, key", CASES)
def test_matches_fixest(fixtures, scenario, key):
    res = _fit(scenario, key)
    ref = fixtures["synthetic"][scenario][key]
    label = f"{scenario}/{key}"

    # 行ごとの残差の一致は、固定効果の和の一致と同値（モジュールdoc）。
    assert len(res.residuals) == len(ref["residuals"])
    for row, (ours, theirs) in enumerate(zip(res.residuals, ref["residuals"])):
        _assert_close(ours, theirs, f"{label}/residuals/{row}")

    if (scenario, key) == DIFFERS_FROM_FIXEST_BY_K:
        # 標準誤差・共分散行列・それに依存する量はKの数え方が違う（下のテストで説明する）。
        _assert_dict_close(res.params, ref["params"], f"{label}/params")
        assert res.df_model == ref["df_model"], label
        assert res.df_resid == ref["df_resid"], label
        _assert_fields_close(
            res,
            ref,
            ("aic", "bic", "log_likelihood", "r_squared_within"),
            label,
        )
        return

    if "p_values" not in ref:
        # `nested`のHC2/HC3: 自由度に依存しない量だけを比べる（モジュールdoc）。
        assert scenario == "nested"
        assert key in NESTED_HC_FALLBACK_COV_TYPES
        _assert_dict_close(res.params, ref["params"], f"{label}/params")
        _assert_dict_close(
            res.std_errors, ref["std_errors"], f"{label}/std_errors"
        )
        _assert_dict_close(
            res.test_stats, ref["test_stats"], f"{label}/test_stats"
        )
        assert_cov_params_close(
            res.cov_params, ref["cov_params"], label, rtol=RTOL
        )
        return

    assert_inference_close(
        res,
        ref,
        label,
        rtol=RTOL,
        atol=ATOL,
        p_value_check=_assert_p_dict_close,
    )
    assert res.df_model == ref["df_model"], label
    assert res.df_resid == ref["df_resid"], label
    _assert_p_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")
    _assert_fields_close(
        res,
        ref,
        (
            "f_statistic",
            "aic",
            "bic",
            "log_likelihood",
            "r_squared_within",
        ),
        label,
    )


def test_nested_cluster_on_the_fixed_effect_differs_from_fixest_only_by_k(
    fixtures,
):
    """クラスターが固定効果の次元`c`で、個体が`c`に入れ子のとき、本実装の標準誤差はfixestの
    `(n−K_fixest)/(n−K_ours)`の平方根倍。`K`は`df_model`から入れ子の次元の水準数
    （重複する定数項を除く）を引いたもの。"""
    scenario, key = DIFFERS_FROM_FIXEST_BY_K
    res = _fit(scenario, key)
    ref = fixtures["synthetic"][scenario][key]
    n = len(ref["residuals"])
    n_groups = res.n_groups
    # fixestは`c`だけ、本実装は個体と`c`を入れ子とみなす。
    k_fixest = ref["df_model"] - (n_groups["c"] - 1)
    k_ours = ref["df_model"] - (n_groups["entity"] + n_groups["c"] - 2)
    scale = (n - k_fixest) / (n - k_ours)
    for name in ("x1", "x2"):
        _assert_close(
            res.std_errors[name] ** 2,
            scale * ref["std_errors"][name] ** 2,
            f"nested/cluster_c/variance/{name}",
        )
    mark_cov_consumed(ref["cov_params"])
    for row, ref_row in ref["cov_params"].items():
        for col, value in ref_row.items():
            _assert_close(
                res.cov_params[row][col],
                scale * value,
                f"nested/cluster_c/cov_params/{row}/{col}",
            )
