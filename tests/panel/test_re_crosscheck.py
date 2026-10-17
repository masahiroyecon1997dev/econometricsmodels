"""RE の独立実装（R: plm）とのクロスチェックテスト。

主リファレンス（linearmodels）との厳密比較（classical/hc1）は
`test_re_reference.py`で行う。classical/hc1もこのファイルのplmで独立に検証する
（全統計量。F統計量はこのファイルが全cov_typeの独立リファレンス。ただし
`unbalanced`は傾きだけのWald Fを手書きした参照値で、独立性が下がる。下記
「不均衡パネル（unbalanced）の参照値」参照）。
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
  5.3節）。`unbalanced`だけは、`linearmodels`の分散成分から補助回帰を組み立てた
  plmを参照にする（下記）。比較は常に1-wayで`REOptions.dk_time`の有無によらない
  （`generate_re_crosscheck_fixtures.py`モジュールdoc参照）。

## 不均衡パネル（unbalanced）の参照値

plmのSwamy-Arora分散成分は、`linearmodels`準拠の本実装と不均衡パネルで僅かに
異なる（plm自身のREとの差: 係数が最大0.18%、標準誤差・信頼区間等がcov_typeに
応じて1%前後、dkのバンド幅0では信頼区間が12%）。差は共分散推定量や小標本補正
ではなく分散成分の推定にあるため、`unbalanced`の標準誤差・推論統計量（全
cov_type、dkの明示バンド幅を含む）は、plm自身のREではなく`linearmodels`の分散
成分をplmに与えて準偏差変換したデータへの`vcov`・`vcovHC`・`vcovSCC`と比べる
（`run_plm_re_given_variance_benchmark.R`）。他のシナリオと同じく機械精度で
一致し、補正式の取り違え（clusterの`G/(G-1)`欠落は標準誤差に約1.3%）も検出
できる。

この参照でplmが検証するのは係数・共分散行列・小標本補正まで。θの式、傾きだけの
Wald F統計量（`pwaldtest`は変換後の定数列も検定に含めるため手書き）、t検定への
変換は手書きで、分散成分そのもの（Swamy-Arora）は`linearmodels`（`re.json`）が
検証する。θの式とF統計量の式は、plm自身の分散成分を与えた参照値がplm自身のRE
と一致することで、フィクスチャを作るたびに確かめている。

## dkの明示バンド幅・時点数の境界・実データ

- **明示バンド幅**: 既定式は`T=6`でも`T=25`でも2になるため、`dk_bandwidth`を
  0・2・3（`T=25`のシナリオは12も）で明示して`vcovSCC(maxlag=)`と比べる。
- **時点数の境界**: ハウスマン検定の対象数`k=2`に対し`T=k+1=3`の成功パス
  （`baseline.dk_three_periods`）。`T<=k`は`test_re_validation.py`の
  `ValidationError`。
- **wagepan**: `T=8`、検定対象3個（`T-1=7`）でdkも標準誤差・ハウスマン検定の
  両方をplmと比べる（時点列は`year`）。

## ハウスマン統計量の方式について

補助回帰版は統計量が構造的に非負になるため符号処理は不要。`plm`は補助回帰の
定数項に準偏差変換前の`1`を使い、本実装もこれに合わせている。バランスパネルでは
機械精度で一致する。不均衡パネル（`UNBALANCED_SCENARIO`）では、`phtest`の補助回帰が
plmのSwamy-Arora分散成分から作るθに依存し、plm自身の値とは統計量が1〜2%ずれる
（dkのバンド幅2と3の差約2.1%を隠す）。そのためハウスマン検定も、`linearmodels`の
分散成分から補助回帰を組み立ててplmの`vcov`系を当てた値と機械精度で比べる
（`run_plm_re_given_variance_hausman_benchmark.R`）。plmが検証するのは補助回帰の
係数と共分散行列まで。補助回帰の組み立て（準偏差変換した`y*`・`X*`、within変換した
`X̃`、素の定数項、検定対象の選択）は手書きで、plm自身の分散成分を与えた値が
`phtest`と一致することで、フィクスチャを作るたびに確かめている。

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
from functools import partial
from pathlib import Path

import _error_messages_panel as panel_msgs
import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close, assert_inference_close
from _constants import DATA_DIR
from _error_messages import escaped
from _helpers import load_wooldridge_dataset
from _panel_helpers import bandwidth_cases, x_columns
from _tolerances import TOLERANCES
from econometricsmodels import (
    RE,
    REOptions,
    ValidationError,
)

from benchmark.common import (
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
)
from benchmark.panel.datasets import (
    CLUSTER_IMBALANCED_CSV,
    cluster_imbalanced_groups,
)
from benchmark.panel.fixtures.generate_fe_crosscheck_fixtures import (
    dk_three_periods_frame,
)
from benchmark.panel.fixtures.generate_fe_fixtures import NUMERIC_SCENARIOS
from benchmark.panel.fixtures.generate_re_crosscheck_fixtures import (
    COV_TYPES,
    DK_EXPLICIT_BANDWIDTH_KEY,
    DK_EXPLICIT_BANDWIDTHS,
    HAUSMAN_COV_TYPES,
    HAUSMAN_DK_BANDWIDTH_KEY,
    HAUSMAN_KEY,
    SCENARIO_COV_TYPES,
    UNBALANCED_SCENARIO,
    WAGEPAN_COV_TYPES,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_crosscheck.json"
)

RTOL = TOLERANCES["re_crosscheck"]["rtol"]
ATOL = TOLERANCES["re_crosscheck"]["atol"]
RTOL_P_VALUE = TOLERANCES["re_crosscheck"]["rtol_p_value"]
RTOL_HAUSMAN = TOLERANCES["re_crosscheck"]["rtol_hausman"]
ATOL_HAUSMAN = TOLERANCES["re_crosscheck"]["atol_hausman"]
RTOL_HAUSMAN_ILL_CONDITIONED = TOLERANCES["re_crosscheck"][
    "rtol_hausman_ill_conditioned"
]
ATOL_HAUSMAN_P_VALUE = TOLERANCES["re_crosscheck"]["atol_hausman_p_value"]

# 条件数が大きく、ハウスマン統計量にだけ専用の緩めたrtol
# （`rtol_hausman_ill_conditioned`）を使うシナリオ。
_ILL_CONDITIONED_HAUSMAN_SCENARIO = "high_condition_number"


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, atol=ATOL)


def _assert_p_close(ours: float, ref: float, label: str) -> None:
    """p値の比較。絶対誤差の下限を設けない（下限があると、参照値が1e-8未満の
    裾のp値が0.0でも通ってしまう）。"""
    assert_close(ours, ref, label, rtol=RTOL_P_VALUE, atol=0.0)


ALL_CASES = [
    (scenario, cov_type)
    for scenario in NUMERIC_SCENARIOS
    for cov_type in SCENARIO_COV_TYPES.get(scenario, COV_TYPES)
]


def _check_result(
    res,
    ref: dict,
    label: str,
) -> None:
    def p_value_check(
        ours: dict[str, float], ref_p: dict[str, float], p_label: str
    ) -> None:
        for name, p_ref in ref_p.items():
            _assert_p_close(ours[name], p_ref, f"{p_label}/{name}")

    assert_inference_close(
        res,
        ref,
        label,
        rtol=RTOL,
        atol=ATOL,
        p_value_check=p_value_check,
    )

    # F統計量はcov_typeに連動するWald検定（plm::pwaldtest(vcov=...)の統計量と
    # t検定と同じ分母自由度から計算したp値）。
    _assert_close(
        res.f_statistic,
        ref["f_statistic"],
        f"{label}/f_statistic",
        rtol=RTOL,
    )
    _assert_p_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")


def _check_hausman(
    res,
    ref: dict,
    label: str,
    *,
    scenario: str | None = None,
) -> None:
    rtol_hausman = (
        RTOL_HAUSMAN_ILL_CONDITIONED
        if scenario == _ILL_CONDITIONED_HAUSMAN_SCENARIO
        else RTOL_HAUSMAN
    )
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
        rtol=rtol_hausman,
        atol=ATOL_HAUSMAN_P_VALUE,
    )
    assert res.hausman_df == ref["hausman_df"], f"{label}/hausman_df"


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


def _reference_label(scenario: str) -> str:
    """参照値の出どころ。`unbalanced`はplm自身のREではなく、`linearmodels`の
    分散成分を与えたplm（モジュールdoc「不均衡パネル（unbalanced）の参照値」参照）。"""
    if scenario == UNBALANCED_SCENARIO:
        return "plm with linearmodels variance components"
    return "plm"


def _re_options(
    cov_type: str,
    *,
    dk_time: str = "time",
    dk_bandwidth: int | None = None,
) -> REOptions:
    # dk_time・dk_bandwidthはdkのときだけ指定できる（REOptionsの検証）。
    if cov_type == "dk":
        return REOptions(
            cov_type="dk", dk_time=dk_time, dk_bandwidth=dk_bandwidth
        )
    return REOptions(cov_type=cov_type)


@pytest.mark.parametrize("scenario, cov_type", ALL_CASES)
def test_matches_plm(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = x_columns(df)
    res = RE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=_re_options(cov_type),
    ).fit()

    _check_result(
        res,
        crosscheck["synthetic"][scenario][cov_type],
        f"{scenario}/{cov_type} vs {_reference_label(scenario)}",
    )


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
    groups = cluster_imbalanced_groups()
    res = _fit_with_cluster_column(
        CLUSTER_IMBALANCED_CSV, ["x1", "x2"], groups
    )

    _check_result(
        res,
        crosscheck["synthetic"]["baseline"]["cluster_imbalanced"],
        "baseline/cluster_imbalanced",
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
        crosscheck["synthetic"]["baseline"]["cluster_g3"],
        "baseline/cluster_g3",
    )
    assert res.stat_df == 2


# ── dkの明示バンド幅 ──────────────────────────────────────────────────


DK_BANDWIDTH_CASES = bandwidth_cases(DK_EXPLICIT_BANDWIDTHS)


@pytest.mark.parametrize("scenario, bandwidth", DK_BANDWIDTH_CASES)
def test_dk_explicit_bandwidth_matches_plm(crosscheck, scenario, bandwidth):
    """`dk_bandwidth`を明示指定した場合の標準誤差・推論統計量も
    `vcovSCC(maxlag=bandwidth)`と機械精度で一致する。

    参照値は、balancedではplm自身のRE、`unbalanced`では`linearmodels`の分散成分
    を与えて準偏差変換したデータへのplmの`vcovSCC`（モジュールdoc参照）。後者は
    係数・共分散行列・小標本補正までをplmが検証し、θの式・傾きだけのWald F・
    t検定への変換は手書きなので、検証できるのはバンド幅の扱いと小標本補正に
    限られる（等価性は生成時にplm自身の分散成分で確認済み）。バンド幅の扱いは
    wagepan（実データ）のplm自身のREでも確かめている。"""
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    res = RE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=_re_options("dk", dk_bandwidth=bandwidth),
    ).fit()

    assert res.dk_bandwidth_used == bandwidth
    _check_result(
        res,
        crosscheck["synthetic"][scenario][DK_EXPLICIT_BANDWIDTH_KEY][
            str(bandwidth)
        ],
        f"{scenario}/dk/bandwidth={bandwidth} vs {_reference_label(scenario)}",
    )


# ── 境界値（dkの時点数） ──────────────────────────────────────────────


def _fit_dk_three_periods():
    return RE(
        dk_three_periods_frame(),
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=_re_options("dk"),
    ).fit()


def test_dk_three_periods_matches_plm(crosscheck):
    """Driscoll-Kraayの時点数`T = k+1`（k=2、分母自由度`T-1 = k`）の境界の
    成功パス。REは`fit()`のハウスマン検定の対象数`k`に対し`T > k`が必要で、
    `T = 3`が`k = 2`での最小の時点数（`T <= k`は`ValidationError`、
    `test_hausman_dk_too_few_periods_raises`）。"""
    res = _fit_dk_three_periods()

    assert res.stat_df == 2
    _check_result(
        res,
        crosscheck["synthetic"]["baseline"]["dk_three_periods"],
        "baseline/dk_three_periods",
    )


def test_dk_three_periods_hausman_matches_plm(crosscheck):
    """`T = k+1`境界のハウスマン検定（補助回帰の傾き`2k`個ではなく検定対象
    `k`個に対しDK共分散のrank`T-1`が足りるかが境界を決める。`re-spec.md`
    3.7節）。"""
    _check_hausman(
        _fit_dk_three_periods(),
        crosscheck["synthetic"]["baseline"]["dk_three_periods_hausman"],
        "baseline/dk_three_periods",
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_matches_plm(crosscheck, cov_type):
    df = load_wooldridge_dataset("wagepan")
    options = _re_options(cov_type, dk_time=WAGEPAN_TIME)
    res = RE(
        df, y=WAGEPAN_Y, x=WAGEPAN_X, entity=WAGEPAN_ENTITY, options=options
    ).fit()

    _check_result(
        res,
        crosscheck["wooldridge"]["wagepan"][cov_type],
        f"wagepan/{cov_type}",
    )


# ── ハウスマン検定（RE本体のcov_typeに連動、plm::phtest(method="aux", vcov=...)。unbalancedは分散成分を与えた補助回帰） ──


@pytest.mark.parametrize("cov_type", HAUSMAN_COV_TYPES)
@pytest.mark.parametrize("scenario", NUMERIC_SCENARIOS)
def test_hausman_matches_plm(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = x_columns(df)
    model = RE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=_re_options(cov_type),
    )
    ref = crosscheck["synthetic"][scenario][HAUSMAN_KEY][cov_type]
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
        f"{scenario}/{cov_type} vs {_reference_label(scenario)}",
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
        crosscheck["wooldridge"]["wagepan"][HAUSMAN_KEY][cov_type],
        f"wagepan/{cov_type}",
    )


@pytest.mark.parametrize("scenario, bandwidth", DK_BANDWIDTH_CASES)
def test_hausman_dk_explicit_bandwidth_matches_plm(
    crosscheck, scenario, bandwidth
):
    """`dk_bandwidth`を明示指定した場合も`vcovSCC(maxlag=bandwidth)`と一致する。
    `unbalanced`でもバンド幅2と3（統計量の差は約2%）を判別できる。"""
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
        crosscheck["synthetic"][scenario][HAUSMAN_DK_BANDWIDTH_KEY][
            str(bandwidth)
        ],
        f"{scenario}/dk/bandwidth={bandwidth} vs {_reference_label(scenario)}",
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
            panel_msgs.INSUFFICIENT_DK_PERIODS_FOR_INFERENCE, t_periods=2, q=2
        ),
    ):
        RE(
            df,
            y="y",
            x=["x1", "x2"],
            entity="entity",
            options=REOptions(cov_type="dk", dk_time="time", dk_bandwidth=0),
        ).fit()
