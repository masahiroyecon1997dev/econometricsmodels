"""RE の独立実装（R: plm）とのクロスチェックテスト。

主リファレンス（linearmodels）との厳密比較（classical/hc1）は
`test_re_reference.py`で行う。classical/hc1もこのファイルのplmで独立に検証する
（全統計量。F統計量はこのファイルが全cov_typeの独立リファレンス）。
ここでは`tests/fixtures/benchmarks/re_crosscheck.json`（`benchmark/panel/
fixtures/generate_re_crosscheck_fixtures.py`で生成）を用いて、linearmodelsとは
独立した実装（R: plm）との一致を確認する。

## このファイルだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.RandomEffects`が提供しないため、plmが唯一の
  参照実装になる（`benchmark/panel/references/linearmodels_ref.py`
  モジュールdoc参照）。
- **cluster/dk**: 本実装の小標本補正がStata・R型（`G/(G-1)·(n-1)/(n-K)`、dkは
  `G`の代わりに時点数、t分布の自由度は`G-1`/`T-1`）で、linearmodels
  （`n/(n-k)`）とは一致しないため、plm（cluster: `vcovHC(method="arellano",
  type="sss")`、dk: `vcovSCC(maxlag=, type="sss")`）が唯一の参照実装になる。
  `test_re_reference.py`はclassical/hc1のみをlinearmodelsと比較する。
- **f_statistic/f_p_value（全cov_type）**: `plm::pwaldtest(test = "F",
  vcov = ...)`の統計量（p値はt検定と同じ分母自由度から再計算）。linearmodelsの
  `f_statistic_robust`はclassical/hc1しか持たない。
- **ハウスマン検定**（`hausman_statistic`/`hausman_p_value`/`hausman_df`）:
  `linearmodels`に専用実装が無いため、`plm::phtest(method = "aux",
  effect = "individual")`（回帰ベース）が唯一の参照実装（panel-common.md
  5.3節）。比較は常に1-wayで`REOptions.dk_time`の有無によらない
  （`generate_re_crosscheck_fixtures.py`モジュールdoc参照）。

## ハウスマン統計量の方式について

補助回帰版は統計量が構造的に非負になるため符号処理は不要。`plm`は補助回帰の
定数項に準偏差変換前の`1`を使い、本実装もこれに合わせている。バランスパネルでは
機械精度で一致し、不均衡パネルではSwamy-Arora分散成分の差（σ_u²）で数％ずれる
（`_UNBALANCED_HAUSMAN_SCENARIO`）。

## 許容誤差について

plmの変量効果分散成分推定（Swamy-Arora）がlinearmodels準拠の本実装と不均衡
パネルで僅かに異なるため、`unbalanced`シナリオのみ、係数が最大0.18%、標準誤差・
信頼区間等がcov_typeに応じて最大1%台（dkの信頼区間は3.7%）乖離する。許容誤差は一律ではなく、統計量・cov_type別に実測へマージンを載せる
（`_tolerances.py`の`re_crosscheck`の`rtol_unbalanced*`）。バランスパネルでは
機械精度で一致するため、それ以外のシナリオは`rtol_balanced`で厳密に比較する
（cluster・dkの`G/(G-1)`・`T/(T-1)`補正は標準誤差に1%前後しか効かず、緩い
許容誤差では補正式の取り違えを検出できないため、バランスパネルを緩めない）。

役割分担:
    - 構造・API: `test_re_api.py`
    - `ValidationError`/`ComputationError` パス: `test_re_validation.py`
    - 主リファレンス（linearmodels、classical/hc1）との数値照合:
      `test_re_reference.py`
    - 独立実装（R: plm）とのクロスチェック（hc2/hc3・cluster/dk・
      ハウスマン検定）: このファイル
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import _error_messages as msgs
import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close, rename_intercept
from _constants import DATA_DIR
from _error_messages import escaped
from _helpers import load_wooldridge_dataset
from _tolerances import TOLERANCES
from econometricsmodels import (
    RE,
    REOptions,
    ValidationError,
)

from benchmark.common import (
    WAGEPAN_ENTITY,
    WAGEPAN_X,
    WAGEPAN_Y,
    imbalanced_cluster_groups,
)
from benchmark.panel.fixtures.generate_fe_fixtures import NUMERIC_SCENARIOS
from benchmark.panel.fixtures.generate_re_crosscheck_fixtures import (
    COV_TYPES,
    HAUSMAN_COV_TYPES,
    HAUSMAN_DK_BANDWIDTH_KEY,
    HAUSMAN_DK_BANDWIDTH_SCENARIOS,
    HAUSMAN_DK_BANDWIDTHS,
    HAUSMAN_KEY,
    SCENARIO_COV_TYPES,
    WAGEPAN_COV_TYPES,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_crosscheck.json"
)

RTOL_BALANCED = TOLERANCES["re_crosscheck"]["rtol_balanced"]
RTOL_UNBALANCED_COEF = TOLERANCES["re_crosscheck"]["rtol_unbalanced_coef"]
RTOL_UNBALANCED_F = TOLERANCES["re_crosscheck"][
    "rtol_unbalanced_f"
]  # cov_type別
RTOL_UNBALANCED = TOLERANCES["re_crosscheck"]["rtol_unbalanced"]  # cov_type別
ATOL = TOLERANCES["re_crosscheck"]["atol"]
RTOL_P_VALUE = TOLERANCES["re_crosscheck"]["rtol_p_value"]
P_LOG10_UNBALANCED = TOLERANCES["re_crosscheck"]["p_value_log10_unbalanced"]
RTOL_HAUSMAN = TOLERANCES["re_crosscheck"]["rtol_hausman"]
ATOL_HAUSMAN = TOLERANCES["re_crosscheck"]["atol_hausman"]
RTOL_HAUSMAN_UNBALANCED = TOLERANCES["re_crosscheck"][
    "rtol_hausman_unbalanced"
]  # cov_type別
RTOL_HAUSMAN_P_VALUE_UNBALANCED = TOLERANCES["re_crosscheck"][
    "rtol_hausman_p_value_unbalanced"
]
RTOL_HAUSMAN_ILL_CONDITIONED = TOLERANCES["re_crosscheck"][
    "rtol_hausman_ill_conditioned"
]
ATOL_HAUSMAN_P_VALUE = TOLERANCES["re_crosscheck"]["atol_hausman_p_value"]

# plm/linearmodelsの分散成分（σ_u²）推定の差でθが変わり、ハウスマン統計量への
# 増幅がcoef/seよりさらに大きくなる唯一のシナリオ（モジュールdoc「許容誤差に
# ついて」参照）。
_UNBALANCED_HAUSMAN_SCENARIO = "unbalanced"
_ILL_CONDITIONED_HAUSMAN_SCENARIO = "high_condition_number"


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, atol=ATOL)


def _assert_p_close(
    ours: float, ref: float, label: str, *, balanced: bool
) -> None:
    """p値の比較。絶対誤差の下限を設けない（下限があると、参照値が1e-8未満の
    裾のp値が0.0でも通ってしまう）。

    バランスパネルは相対誤差のみで比較する。不均衡パネルはSwamy-Arora分散成分の
    差で統計量が数％ずれ、裾のp値は相対誤差では桁違いにずれる（統計量の0.1%の
    差が`exp(-F)`型に増幅される）ため、常用対数の差で比較する
    （0.0への潰れやオーダーの取り違えは検出できる）。
    """
    if balanced:
        assert_close(ours, ref, label, rtol=RTOL_P_VALUE, atol=0.0)
        return
    if ref == 0.0:
        assert ours == 0.0, f"{label}: ours={ours!r}, ref=0.0"
        return
    assert ours > 0.0, f"{label}: ours={ours!r} underflowed, ref={ref!r}"
    diff = abs(math.log10(ours / ref))
    assert diff <= P_LOG10_UNBALANCED, (
        f"{label}: ours={ours!r}, ref={ref!r}, |log10 ratio|={diff!r} "
        f"> {P_LOG10_UNBALANCED}"
    )


ALL_CASES = [
    (scenario, cov_type)
    for scenario in NUMERIC_SCENARIOS
    for cov_type in SCENARIO_COV_TYPES.get(scenario, COV_TYPES)
]


def _f_rtol_for(scenario: str, cov_type: str) -> float:
    """F統計量のrtol。不均衡パネルのみ分散成分の差をcov_type別に許容する。"""
    if scenario != _UNBALANCED_HAUSMAN_SCENARIO:
        return RTOL_BALANCED
    return RTOL_UNBALANCED_F[cov_type]


def _rtols_for(scenario: str, cov_type: str) -> tuple[float, float]:
    """(係数のrtol, se・t・p値・信頼区間のrtol)。不均衡パネルのみSwamy-Arora
    分散成分の差を統計量・cov_type別に許容する（モジュールdoc参照）。"""
    if scenario != _UNBALANCED_HAUSMAN_SCENARIO:
        return RTOL_BALANCED, RTOL_BALANCED
    return RTOL_UNBALANCED_COEF, RTOL_UNBALANCED[cov_type]


def _check_result(
    res,
    ref: dict,
    label: str,
    *,
    rtol: float,
    coef_rtol: float | None = None,
    f_rtol: float | None = None,
    balanced: bool = True,
) -> None:
    coef_rtol = rtol if coef_rtol is None else coef_rtol
    f_rtol = rtol if f_rtol is None else f_rtol
    _assert_dict_close(
        res.params, ref["coef"], f"{label}/coef", rtol=coef_rtol
    )
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se", rtol=rtol)
    _assert_dict_close(
        res.test_stats, ref["test_stats"], f"{label}/test_stats", rtol=rtol
    )
    for name, p_ref in ref["p_values"].items():
        _assert_p_close(
            res.p_values[rename_intercept(name)],
            p_ref,
            f"{label}/p_values/{name}",
            balanced=balanced,
        )

    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_close(
            our_lower, ref_lower, f"{label}/conf_lower/{name}", rtol=rtol
        )
        _assert_close(
            our_upper, ref_upper, f"{label}/conf_upper/{name}", rtol=rtol
        )

    # F統計量はcov_typeに連動するWald検定（plm::pwaldtest(vcov=...)の統計量と
    # t検定と同じ分母自由度から計算したp値）。
    _assert_close(
        res.f_statistic,
        ref["f_statistic"],
        f"{label}/f_statistic",
        rtol=f_rtol,
    )
    _assert_p_close(
        res.f_p_value,
        ref["f_p_value"],
        f"{label}/f_p_value",
        balanced=balanced,
    )


def _check_hausman(
    res,
    ref: dict,
    label: str,
    *,
    cov_type: str,
    scenario: str | None = None,
) -> None:
    rtol_p_value = None
    if scenario == _UNBALANCED_HAUSMAN_SCENARIO:
        rtol_hausman = RTOL_HAUSMAN_UNBALANCED[cov_type]
        rtol_p_value = RTOL_HAUSMAN_P_VALUE_UNBALANCED[cov_type]
    elif scenario == _ILL_CONDITIONED_HAUSMAN_SCENARIO:
        rtol_hausman = RTOL_HAUSMAN_ILL_CONDITIONED
    else:
        rtol_hausman = RTOL_HAUSMAN
    assert_close(
        res.hausman_statistic,
        ref["hausman_statistic"],
        f"{label}/hausman_statistic",
        rtol=rtol_hausman,
        atol=ATOL_HAUSMAN,
    )
    assert_close(
        res.hausman_p_value,
        ref["hausman_p_value"],
        f"{label}/hausman_p_value",
        rtol=rtol_hausman if rtol_p_value is None else rtol_p_value,
        atol=ATOL_HAUSMAN_P_VALUE,
    )
    assert res.hausman_df == ref["hausman_df"], f"{label}/hausman_df"


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


def _re_options(cov_type: str) -> REOptions:
    # timeはdkのときだけ指定できる（REOptionsのバリデーション）。
    if cov_type == "dk":
        return REOptions(cov_type="dk", dk_time="time")
    return REOptions(cov_type=cov_type)


@pytest.mark.parametrize("scenario, cov_type", ALL_CASES)
def test_synthetic_matches_plm(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    res = RE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=_re_options(cov_type),
    ).fit()

    coef_rtol, rtol = _rtols_for(scenario, cov_type)
    _check_result(
        res,
        crosscheck[scenario][cov_type],
        f"{scenario}/{cov_type}",
        rtol=rtol,
        coef_rtol=coef_rtol,
        f_rtol=_f_rtol_for(scenario, cov_type),
        balanced=scenario != _UNBALANCED_HAUSMAN_SCENARIO,
    )


@pytest.mark.parametrize("cov_type", ["cluster", "dk"])
def test_many_regressors_cluster_dk_raise_validation_error(cov_type):
    """`many_regressors`（k=20、40エンティティ、T=6）はRE本体は成功する入力
    だが、ハウスマン検定の補助回帰のロバスト共分散が構造的に特異になり
    `fit()`が失敗する（cluster: 補助回帰の傾き係数`2k=40`がG=40以下、dk:
    検定対象`k=20`が`T-1=5`超。`re-spec.md`3.7節）。このためplmの標準誤差
    クロスチェックの対象外（`SCENARIO_COV_TYPES`）で、失敗パスのみ確認する。
    """
    df = pl.read_csv(DATA_DIR / "fe_many_regressors.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    with pytest.raises(ValidationError):
        RE(
            df,
            y="y",
            x=x_cols,
            entity="entity",
            options=_re_options(cov_type),
        ).fit()


def _fit_with_cluster_column(csv_name: str, x_cols: list[str], groups):
    df = pl.read_csv(DATA_DIR / csv_name).with_columns(
        pl.Series("cluster_group", groups)
    )
    return RE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=REOptions(cov_type="cluster", cluster="cluster_group"),
    ).fit()


def test_cluster_imbalanced_matches_plm_based_reference(crosscheck):
    """クラスター不均衡（サイズ[2,3,5,10,30,50]のタイル、entityとは無関係な
    専用クラスター列）。plmはgroup/timeしかクラスターにできないため、参照値は
    plmの準偏差変換済みデータに`lm` + `sandwich::vcovCL`を当てたもの
    （`run_plm_benchmark.R`モジュールコメント参照）。`stat_df`はentity数ではなく
    クラスター数`G-1`になる。バランスパネルのため機械精度で比較する。
    """
    n = pl.read_csv(DATA_DIR / "fe_baseline_cluster_imbalanced.csv").height
    groups = imbalanced_cluster_groups(n)
    res = _fit_with_cluster_column(
        "fe_baseline_cluster_imbalanced.csv", ["x1", "x2"], groups
    )

    _check_result(
        res,
        crosscheck["baseline"]["cluster_imbalanced"],
        "baseline/cluster_imbalanced",
        rtol=RTOL_BALANCED,
    )
    assert res.stat_df == len(set(groups)) - 1


def test_cluster_g3_boundary_matches_plm_based_reference(crosscheck):
    """クラスター数の境界の成功パス。REはハウスマン検定の補助回帰の傾き係数
    `2k`に対し`G > 2k`が必要なため、`k=1`・`G=3`（`G = 2k+1`）が最小の成功パス
    （`G=2`は`test_hausman_cluster_count_equal_to_auxiliary_slopes_raises`系の
    `ValidationError`）。
    """
    n = pl.read_csv(DATA_DIR / "fe_baseline_k1.csv").height
    groups = [str(i % 3) for i in range(n)]
    res = _fit_with_cluster_column("fe_baseline_k1.csv", ["x1"], groups)

    _check_result(
        res,
        crosscheck["baseline"]["cluster_g3"],
        "baseline/cluster_g3",
        rtol=RTOL_BALANCED,
    )
    assert res.stat_df == 2


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_matches_plm(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = REOptions(cov_type=cov_type)
    res = RE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(
        res,
        crosscheck["wagepan"][cov_type],
        f"wagepan/{cov_type}",
        rtol=RTOL_BALANCED,
    )


# ── ハウスマン検定（RE本体のcov_typeに連動、plm::phtest(method="aux", vcov=...)） ──


@pytest.mark.parametrize("cov_type", HAUSMAN_COV_TYPES)
@pytest.mark.parametrize("scenario", NUMERIC_SCENARIOS)
def test_synthetic_hausman_matches_plm(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "entity", "time")]
    model = RE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=_re_options(cov_type),
    )
    ref = crosscheck[scenario][HAUSMAN_KEY][cov_type]
    if ref is None:
        # ロバスト共分散が構造的に特異でplmに参照値が無いケース
        # （`generate_re_crosscheck_fixtures.py`の`_STRUCTURALLY_SINGULAR`）。
        # 本実装は`None`ではなくfit()が入力から判定できる`ValidationError`に
        # なる（cluster: `G <= 2k`、dk: `T <= k`。`re-spec.md`3.7節）。
        with pytest.raises(ValidationError):
            model.fit()
        return

    _check_hausman(
        model.fit(),
        ref,
        f"{scenario}/{cov_type}",
        cov_type=cov_type,
        scenario=scenario,
    )


@pytest.mark.parametrize("cov_type", HAUSMAN_COV_TYPES)
def test_wagepan_hausman_matches_plm(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = (
        REOptions(cov_type="dk", dk_time="year")
        if cov_type == "dk"
        else REOptions(cov_type=cov_type)
    )
    res = RE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_hausman(
        res,
        crosscheck["wagepan"][HAUSMAN_KEY][cov_type],
        f"wagepan/{cov_type}",
        cov_type=cov_type,
    )


@pytest.mark.parametrize("bandwidth", HAUSMAN_DK_BANDWIDTHS)
@pytest.mark.parametrize("scenario", HAUSMAN_DK_BANDWIDTH_SCENARIOS)
def test_hausman_dk_explicit_bandwidth_matches_plm(
    crosscheck, scenario, bandwidth
):
    """`dk_bandwidth`を明示指定した場合も`vcovSCC(maxlag=bandwidth)`と一致する。"""
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    res = RE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=REOptions(
            cov_type="dk", dk_time="time", dk_bandwidth=bandwidth
        ),
    ).fit()

    _check_hausman(
        res,
        crosscheck[scenario][HAUSMAN_DK_BANDWIDTH_KEY][str(bandwidth)],
        f"{scenario}/dk/bandwidth={bandwidth}",
        cov_type="dk",
        scenario=scenario,
    )


# ── plmにリファレンスが無いケース・失敗パスの最小再現 ──────────────────


def test_hausman_cluster_with_non_entity_column():
    """`cluster`にentity以外の列を指定できる（plmはgroup/timeしかクラスターに
    できずリファレンスが無いため、entityと同値の別名列で一致・別のグルーピングで
    不一致になることのみ確認する。数値の手計算検証はengineのテスト）。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline.csv").with_columns(
        pl.col("entity").alias("entity_copy"),
        (pl.col("entity").rank("dense") % 10).alias("group10"),
    )

    def fit(cluster: str | None):
        return RE(
            df,
            y="y",
            x=["x1", "x2"],
            entity="entity",
            options=REOptions(cov_type="cluster", cluster=cluster),
        ).fit()

    default = fit(None)
    assert fit("entity_copy").hausman_statistic == pytest.approx(
        default.hausman_statistic, rel=1e-12
    )
    other = fit("group10")
    assert other.hausman_df == 2
    assert other.hausman_statistic != default.hausman_statistic


def _tiny_panel(n_entities: int, n_periods: int) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    n = n_entities * n_periods
    return pl.DataFrame(
        {
            "entity": np.repeat(np.arange(n_entities), n_periods),
            "time": np.tile(np.arange(n_periods), n_entities),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )


def test_hausman_cluster_count_equal_to_auxiliary_slopes_raises():
    """RE本体は`G=4 > q=2`で成功するが、補助回帰の傾き係数`2k=4`に対し
    `G <= 2k`（境界ちょうど）のためfit()が`ValidationError`で失敗する。
    """
    df = _tiny_panel(n_entities=4, n_periods=6)
    with pytest.raises(ValidationError, match="Hausman"):
        RE(
            df,
            y="y",
            x=["x1", "x2"],
            entity="entity",
            options=REOptions(cov_type="cluster"),
        ).fit()
    # G=5 > 2k=4なら成功する（境界の成功パス）。
    ok = RE(
        _tiny_panel(n_entities=5, n_periods=6),
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=REOptions(cov_type="cluster"),
    ).fit()
    assert ok.hausman_df == 2


def test_hausman_dk_too_few_periods_raises():
    """`T=2`ではDK共分散のrankが`T-1=1`で検定対象`k=2`個に足りず、
    fit()が補助回帰を待たず`ValidationError`で失敗する。
    """
    df = _tiny_panel(n_entities=20, n_periods=2)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_DK_PERIODS_FOR_INFERENCE, t_periods=2, q=2
        ),
    ):
        RE(
            df,
            y="y",
            x=["x1", "x2"],
            entity="entity",
            options=REOptions(cov_type="dk", dk_time="time", dk_bandwidth=0),
        ).fit()
