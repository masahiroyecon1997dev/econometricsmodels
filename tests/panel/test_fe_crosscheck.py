"""FE の独立実装（R: fixest）とのクロスチェックテスト。

主リファレンス（linearmodels）との厳密比較（classical/hc1）は
`test_fe_reference.py`で行う。ここでは`tests/fixtures/benchmarks/fe_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_crosscheck_fixtures.py`で生成）を
用いて、linearmodelsとは独立した実装（R: fixest）との一致を確認する。

## cluster/dkはfixestが唯一の参照実装

本実装のcluster・dk（Driscoll-Kraay）の標準誤差は、小標本補正と推論の自由度
（clusterで`G-1`、dkで`T-1`）をfixestの`ssc()`既定に合わせている
（`docs/spec/fe-spec.md`3.3節）ため、linearmodelsとは一致しない。
`test_fe_reference.py`はclassical/hc1のみをlinearmodelsと比較し、cluster/dkは
このファイルのfixestだけで検証する。fixestの既定`ssc()`のまま、全cov_type
（1-way・2-way双方）で機械精度（実測相対誤差1e-14程度）で一致するため、
`RTOL`で厳密比較する。dkはfixestの既定バンド幅が本実装と異なるため、本実装の
既定式で求めたバンド幅を`DK(lag)`に明示的に渡している
（`generate_fe_crosscheck_fixtures.py`参照）。

## このファイルだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.PanelOLS`が提供しないため、fixestが唯一の参照
  実装になる（`benchmark/panel/references/linearmodels_ref.py`モジュールdoc
  参照）。
- **cluster/dk**: 上記の通り、fixestのみで検証する。
- **aic/bic/log_likelihood**: `linearmodels.PanelOLS`が提供しないため、fixestのみで検証する。
- **f_statistic/f_p_value（全cov_type・1-way/2-way）**: linearmodelsはclassical/hc1
  しか比較できないため、`fixest::wald()`で全cov_typeを検証する（p値は`wald()`の
  分母自由度の切り上げを避け、t検定と同じ自由度で計算し直す。
  `run_fixest_benchmark.R`参照）。
- **2-way FEのr_squared_within**: `linearmodels`自身がentityのみdemeanの
  別定義を使うため、fixestの`fitstat(m, "wr2")`のみで検証する。

## dkの対象外シナリオ

`many_regressors`（k=20、T=6）はk>T-1で同時検定の部分行列が構造的に特異になり
本実装が`ValidationError`にするため対象外（`SCENARIO_COV_TYPES`。失敗パスは
`test_fe_validation.py`の`test_many_regressors_dk_raises_validation_error`）。
wagepan（T=8）は`WAGEPAN_X`が3変数で`T-1=7 > q=3`のため対象に含み、時点列は
`year`。

## 既定値以外の指定

- **dkの明示バンド幅**: `DK_EXPLICIT_BANDWIDTHS`のシナリオで`T-1`未満の値を
  fixestと比べる（`bandwidth == T-1`は最終ラグを落とすfixestと食い違うため
  `test_fe_plm_crosscheck.py`が持つ）。
- **2-wayでentity以外のcluster列**: 時間効果とネストする`time`、どちらの
  固定効果ともネストしない列、`G = q+1`の境界、サイズが偏ったクラスター
  （`TWO_WAY_CLUSTER_CASES`）。fixestの`K.fixef="nonnested"`が分岐するのは
  ネストする`time`とそれ以外の2通り。

役割分担:
    - 構造・API・`fixed_effects()`: `test_fe_api.py`
    - `ValidationError`/`ComputationError` パス: `test_fe_validation.py`
    - 主リファレンス（linearmodels、classical/hc1）との数値照合:
      `test_fe_reference.py`
    - 独立実装（R: fixest）とのクロスチェック（cluster/dkはfixestのみ）:
      このファイル
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import (
    assert_close,
    assert_dict_close,
    assert_fields_close,
    assert_inference_close,
)
from _constants import DATA_DIR
from _helpers import load_wooldridge_dataset
from _panel_helpers import bandwidth_cases, x_columns
from _tolerances import TOLERANCES
from econometricsmodels import FE, FEOptions

from benchmark.common import (
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
    imbalanced_cluster_groups,
)
from benchmark.panel.fixtures.generate_fe_crosscheck_fixtures import (
    COV_TYPES,
    DK_EXPLICIT_BANDWIDTHS,
    ONE_WAY_ONLY_SCENARIOS,
    ROBUST_BOUNDARY_COV_TYPES,
    SCENARIO_COV_TYPES,
    TWO_WAY_CLUSTER_CASES,
    TWO_WAY_SCENARIOS,
    WAGEPAN_COV_TYPES,
    dk_three_periods_frame,
    two_way_cluster_frame,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "fe_crosscheck.json"
)

RTOL = TOLERANCES["fe_crosscheck"]["rtol"]
ATOL = TOLERANCES["fe_crosscheck"]["atol"]
ATOL_P_VALUE = TOLERANCES["fe_crosscheck"]["atol_p_value"]
RTOL_P_VALUE = TOLERANCES["fe_crosscheck"]["rtol_p_value"]

ALL_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS


def _cov_types_for(scenario: str) -> list[str]:
    return SCENARIO_COV_TYPES.get(scenario, COV_TYPES)


ONE_WAY_CASES = [
    (scenario, cov_type)
    for scenario in ALL_SCENARIOS
    for cov_type in _cov_types_for(scenario)
]
TWO_WAY_CASES = [
    (scenario, cov_type)
    for scenario in TWO_WAY_SCENARIOS
    for cov_type in _cov_types_for(scenario)
]


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)
_assert_fields_close = partial(assert_fields_close, rtol=RTOL, atol=ATOL)
# p値は絶対誤差の下限を設けず相対誤差だけで比較する（下限があると、参照値が
# 1e-8未満の裾のp値が0.0でも通ってしまう）。
_assert_p_close = partial(assert_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE)
_assert_p_dict_close = partial(
    assert_dict_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE
)


def _check_result(res, ref: dict, label: str) -> None:
    """係数・標準誤差・検定統計量・p値・信頼区間・AIC/BIC/log_likelihood・
    （参照値がある場合の）Within R2の検証。

    AIC/BIC/log_likelihood・Within R2はクロスチェック側のみ持つ統計量
    （モジュールdoc「このファイルだけが持つ統計量」参照）。
    """
    assert_inference_close(
        res,
        ref,
        label,
        rtol=RTOL,
        atol=ATOL,
        p_value_check=_assert_p_dict_close,
    )

    # f_p_valueは裾でも0.0を通さないよう、p値用の許容誤差（下限なし）で比べる。
    _assert_p_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")
    _assert_fields_close(
        res, ref, ("f_statistic", "aic", "bic", "log_likelihood"), label
    )
    if "r_squared_within" in ref:
        _assert_fields_close(res, ref, ("r_squared_within",), label)


def _options(
    cov_type: str,
    *,
    two_way: bool,
    dk_bandwidth: int | None = None,
    time: str = "time",
    dk_time: str = "time",
) -> FEOptions:
    """`time`は2-wayの時間列、`dk_time`はdkの時点列（別のオプション）。`dk`は
    1-way・2-wayとも`dk_time`を明示する必要がある。"""
    kwargs = {"time": time} if two_way else {}
    if cov_type == "dk":
        kwargs["dk_time"] = dk_time
        kwargs["dk_bandwidth"] = dk_bandwidth
    return FEOptions(cov_type=cov_type, **kwargs)


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("scenario, cov_type", ONE_WAY_CASES)
def test_matches_fixest_one_way(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = x_columns(df)
    res = FE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=_options(cov_type, two_way=False),
    ).fit()

    _check_result(
        res,
        crosscheck["synthetic"][scenario]["one_way"][cov_type],
        f"{scenario}/one_way/{cov_type}",
    )


@pytest.mark.parametrize("scenario, cov_type", TWO_WAY_CASES)
def test_matches_fixest_two_way(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = x_columns(df)
    res = FE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=_options(cov_type, two_way=True),
    ).fit()

    _check_result(
        res,
        crosscheck["synthetic"][scenario]["two_way"][cov_type],
        f"{scenario}/two_way/{cov_type}",
    )


# ── dkの明示バンド幅 ──────────────────────────────────────────────────

# fixestは`bandwidth == T-1`で最終ラグを落とすためT-1未満のみ
# （T-1は`test_fe_plm_crosscheck.py`）。
DK_BANDWIDTH_ONE_WAY_CASES = bandwidth_cases(DK_EXPLICIT_BANDWIDTHS)
DK_BANDWIDTH_TWO_WAY_CASES = [
    case for case in DK_BANDWIDTH_ONE_WAY_CASES if case[0] in TWO_WAY_SCENARIOS
]


def _fit_dk_bandwidth(scenario: str, bandwidth: int, *, two_way: bool):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = x_columns(df)
    options = _options("dk", two_way=two_way, dk_bandwidth=bandwidth)
    return FE(df, y="y", x=x_cols, entity="entity", options=options).fit()


@pytest.mark.parametrize("scenario, bandwidth", DK_BANDWIDTH_ONE_WAY_CASES)
def test_dk_explicit_bandwidth_one_way_matches_fixest(
    crosscheck, scenario, bandwidth
):
    res = _fit_dk_bandwidth(scenario, bandwidth, two_way=False)

    assert res.dk_bandwidth_used == bandwidth
    _check_result(
        res,
        crosscheck["synthetic"][scenario]["dk_explicit_bandwidth"]["one_way"][
            str(bandwidth)
        ],
        f"{scenario}/one_way/dk_bandwidth_{bandwidth}",
    )


@pytest.mark.parametrize("scenario, bandwidth", DK_BANDWIDTH_TWO_WAY_CASES)
def test_dk_explicit_bandwidth_two_way_matches_fixest(
    crosscheck, scenario, bandwidth
):
    res = _fit_dk_bandwidth(scenario, bandwidth, two_way=True)

    assert res.dk_bandwidth_used == bandwidth
    _check_result(
        res,
        crosscheck["synthetic"][scenario]["dk_explicit_bandwidth"]["two_way"][
            str(bandwidth)
        ],
        f"{scenario}/two_way/dk_bandwidth_{bandwidth}",
    )


# ── 境界値・クラスター不均衡 ──────────────────────────────────────────


def test_cluster_imbalanced_matches_fixest(crosscheck):
    """クラスター不均衡（サイズ[2,3,5,10,30,50]のタイル、entityとは無関係な
    専用クラスター列）のfixestクロスチェック。`fe_baseline_cluster_imbalanced.
    csv`（entity=20×period=10のn=200）を使う。クラスター列自体はCSVに含めず
    テスト側で都度動的生成する（`generate_fe_crosscheck_fixtures.py`と同じ方針）。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline_cluster_imbalanced.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = FEOptions(cov_type="cluster", cluster="cluster_group")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck["synthetic"]["baseline"]["cluster_imbalanced"],
        "baseline/cluster_imbalanced",
    )


def test_cluster_g2_matches_fixest(crosscheck):
    """クラスタ数境界（G=2、q=1でG>q）の成功パスのfixestクロスチェック。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_k1.csv")
    groups = [str(i % 2) for i in range(df.height)]
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = FEOptions(cov_type="cluster", cluster="cluster_group")
    res = FE(df, y="y", x=["x1"], entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck["synthetic"]["baseline"]["cluster_g2"],
        "baseline/cluster_g2",
    )


def test_cluster_g3_two_slopes_matches_fixest(crosscheck):
    """クラスター数`G = q+1`（q=2、分母自由度`G-1 = q`）の境界。fixestの
    `wald()`が分母自由度を`max(df2, df1+1)`に切り上げる境界で、本実装の
    `F(q, G-1)`のp値が正しいことを確認する（`cluster_g2`はq=1）。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline.csv")
    groups = [str(i % 3) for i in range(df.height)]
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = FEOptions(cov_type="cluster", cluster="cluster_group")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    assert (res.f_df_num, res.f_df_denom) == (2, 2)
    _check_result(
        res,
        crosscheck["synthetic"]["baseline"]["cluster_g3_two_slopes"],
        "baseline/cluster_g3_two_slopes",
    )


@pytest.mark.parametrize("case", list(TWO_WAY_CLUSTER_CASES))
def test_two_way_cluster_on_non_entity_column_matches_fixest(crosscheck, case):
    """2-wayでentity以外の列によるクラスタリング（時間効果とネストする`time`、
    ネストしない列、`G = q+1`の境界、サイズが偏ったクラスター）。fixestの
    `K.fixef="nonnested"`が分岐するのはネストする`time`とそれ以外の2通りで、
    `G = q+1`は自由度の境界、サイズが偏ったクラスターはサイズの偏りを加える
    （`TWO_WAY_CLUSTER_CASES`参照）。分母自由度は全ケースで`G-1`。"""
    df, cluster = two_way_cluster_frame(case)
    options = FEOptions(cov_type="cluster", cluster=cluster, time="time")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    n_clusters = df[cluster].n_unique()
    assert res.stat_df == n_clusters - 1
    assert res.f_df_denom == n_clusters - 1
    _check_result(
        res,
        crosscheck["synthetic"]["baseline"][
            TWO_WAY_CLUSTER_CASES[case].fixture_key
        ],
        f"baseline/two_way/cluster={case}",
    )


def test_dk_three_periods_matches_fixest(crosscheck):
    """Driscoll-Kraayの時点数`T = q+1`（q=2、分母自由度`T-1 = q`）の境界。"""
    df = dk_three_periods_frame()
    options = FEOptions(cov_type="dk", dk_time="time")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    assert (res.f_df_num, res.f_df_denom) == (2, 2)
    _check_result(
        res,
        crosscheck["synthetic"]["baseline"]["dk_three_periods"],
        "baseline/dk_three_periods",
    )


@pytest.mark.parametrize("cov_type", ROBUST_BOUNDARY_COV_TYPES)
@pytest.mark.parametrize("two_way", [False, True])
def test_boundary_df1_robust_matches_fixest(crosscheck, cov_type, two_way):
    """df_resid=1境界のhc1〜hc3（`df_resid <= q`でfixestの`wald()`の分母自由度
    切り上げが効く）。"""
    if two_way:
        df = pl.read_csv(DATA_DIR / "fe_baseline_df1_two_way.csv")
        x = ["x1", "x2", "x3"]
        options = FEOptions(cov_type=cov_type, time="time")
        ref = crosscheck["synthetic"]["baseline_df1"]["two_way_robust"][
            cov_type
        ]
    else:
        df = pl.read_csv(DATA_DIR / "fe_baseline_df1_one_way.csv")
        x = ["x1", "x2"]
        options = FEOptions(cov_type=cov_type)
        ref = crosscheck["synthetic"]["baseline_df1"]["one_way_robust"][
            cov_type
        ]
    res = FE(df, y="y", x=x, entity="entity", options=options).fit()

    assert res.f_df_denom == 1
    _check_result(res, ref, f"baseline_df1/{two_way}/{cov_type}")


def test_boundary_df1_one_way_matches_fixest(crosscheck):
    """df_resid=1境界（1-way）の成功パスのfixestクロスチェック。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_df1_one_way.csv")
    options = FEOptions(cov_type="classical")
    res = FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    _check_result(
        res,
        crosscheck["synthetic"]["baseline_df1"]["one_way"],
        "baseline_df1/one_way",
    )


def test_boundary_df1_two_way_matches_fixest(crosscheck):
    """df_resid=1境界（2-way）の成功パスのfixestクロスチェック。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_df1_two_way.csv")
    options = FEOptions(cov_type="classical", time="time")
    res = FE(
        df, y="y", x=["x1", "x2", "x3"], entity="entity", options=options
    ).fit()

    _check_result(
        res,
        crosscheck["synthetic"]["baseline_df1"]["two_way"],
        "baseline_df1/two_way",
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ──────


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_one_way_matches_fixest(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = _options(cov_type, two_way=False, dk_time=WAGEPAN_TIME)
    res = FE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(
        res,
        crosscheck["wooldridge"]["wagepan"]["one_way"][cov_type],
        f"wagepan/one_way/{cov_type}",
    )


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_two_way_matches_fixest(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = _options(
        cov_type, two_way=True, time=WAGEPAN_TIME, dk_time=WAGEPAN_TIME
    )
    res = FE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(
        res,
        crosscheck["wooldridge"]["wagepan"]["two_way"][cov_type],
        f"wagepan/two_way/{cov_type}",
    )
