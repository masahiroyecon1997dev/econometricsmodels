"""IV(2SLS)の独立実装（R: ivreg + sandwich/lmtest）とのクロスチェックテスト。

主リファレンス（linearmodels）との厳密比較は`test_iv_reference.py`で行う。ここでは
`tests/fixtures/benchmarks/iv_crosscheck.json`
（`benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py`で生成）を用いて、
linearmodelsとは独立した実装（R `ivreg`）との一致を確認する
（`docs/spec/iv-spec.md`4章）。

シナリオ・cov_type・クラスタケースの構成は`test_iv_reference.py`と揃える。

classical/hc0〜hc3/cluster/HACはRとほぼ機械精度で一致する（OLSクロスチェックと同じ実測
傾向）ため`RTOL_STRICT`で厳密比較する。HACのR側は`NeweyWest(adjust = FALSE)`
（既定の`adjust = TRUE`は共分散全体に`n/(n-k)`を掛けるが、本実装・linearmodelsは
掛けない。このスカラー倍だけが差の全てで、small_n（n=40）でも機械精度で一致する。
`benchmark/iv/references/run_ivreg.R`・`test_ols_crosscheck.py`参照）。Wu-Hausmanは
補助回帰が内生変数の第一段階残差の分だけ列が多く、`adjust = TRUE`なら
`n/(n-k_aug)`が掛かる（small_nで統計量が11.1%ずれていたのはこれ）が、同じく
`adjust = FALSE`で揃う。

Note:
    - `hc2`/`hc3`は`vcovHC(model, type="HC2"/"HC3")`（`ivreg:::hatvalues.ivreg`の
      type="stage2"、第二段階OLSのレバレッジをそのまま使う実装）で計算でき、本実装
      （`engine/src/iv/two_sls.rs`の`hc_cov_params`）と数値一致することを実機確認済み
      のため対象に含む（`iv-spec.md`3.1節）。`linearmodels`
      （`test_iv_reference.py`）は対応していないため、hc2/hc3の主リファレンス比較は
      このファイルのみで行う。
    - `weak_instrument_f`・`sargan_statistic`/`sargan_p_value`はivregの
      `summary(diagnostics=TRUE)`が常にclassical vcovで計算する仕様のため、
      全cov_typeで同じ値になる（`benchmark/iv/references/run_ivreg.R`参照）。
    - `wu_hausman_statistic`/`wu_hausman_p_value`は全cov_typeでフィクスチャに
      実測値がある（`summary(diagnostics=TRUE, vcov.=<関数>)`で
      cov_type別のロバスト共分散を診断表に反映できることが判明、
      `benchmark/iv/references/run_ivreg.R`のモジュールコメント参照）。ただしcluster
      cov_typeのみ、ivreg側のWald検定がF分布の分母自由度にクラスター数を
      反映しない既知の制約によりフィクスチャのp値が`null`になる。代わりに
      `_check_result`の`n_clusters`で、Rの統計量とF(n_endog, G-1)から
      p値を再計算して本実装と比較し、分母自由度がG-1であることも確認する。
    - GMMはivregが対応していないため対象外（R momentfitによるクロスチェックは
      `test_iv_gmm_crosscheck.py`）。
    - 第一段階回帰の結果（`first_stage()`）自体はここでは比較しない
      （`test_ols_crosscheck.py`が既にOLSの数値一致を検証済みのため、
      `test_iv_reference.py`と同じ理由）。

    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/iv/freeze.py`参照）から読む。
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import (
    assert_close,
    assert_dict_close,
    assert_fields_close,
    assert_inference_close,
    assert_p_value_close,
)
from _constants import DATA_DIR
from _fixture_cov import load_fixtures_with_cov
from _helpers import (
    hac_time_for,
    load_wooldridge_dataset,
    with_cluster_groups,
    with_row_time,
)
from _tolerances import TOLERANCES
from econometricsmodels import IV, IVOptions
from scipy import stats as sps

from benchmark.common import imbalanced_cluster_groups
from benchmark.iv.constants import (
    CARD_REGION_COLUMN,
    CARD_X_EXOG,
    CLUSTER_ILL_CONDITIONED_SCENARIOS,
    IVREG_ONLY_COV_TYPES,
    LINEARMODELS_COV_TYPES,
    NON_CLUSTER_COV_TYPES,
)
from benchmark.iv.datasets import (
    MULTI_ENDOG_SCENARIOS,
    THREE_ENDOG_INSTRUMENTS,
    THREE_ENDOG_X_ENDOG,
    multi_endog_instruments,
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)
from benchmark.iv.datasets import (
    NUMERIC_SCENARIOS as SCENARIOS,
)
from benchmark.iv.fixtures.generate_iv_crosscheck_fixtures import (
    IVREG_UNSUPPORTED_NO_INTERCEPT,
)
from benchmark.iv.fixtures.generate_iv_fixtures import NO_INTERCEPT_SCENARIOS

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "iv_crosscheck.json"
)

# classical/hc0/hc1/clusterはRとほぼ機械精度で一致する（OLSクロスチェックの実測
# 傾向と同じ、`test_ols_crosscheck.py`参照）。
RTOL_STRICT = TOLERANCES["iv_crosscheck"]["rtol_strict"]

# 絶対誤差フロア（ref値が0近傍のとき相対誤差比較が意味を持たなくなるのを防ぐ）。
ATOL = TOLERANCES["iv_crosscheck"]["atol"]


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)["synthetic"]


@pytest.fixture(scope="module")
def crosscheck_wooldridge() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)["wooldridge"]


# _assert_close/_assert_dict_closeはtests/support/_assertions.pyのassert_close/
# assert_dict_closeに対応する。
_assert_close = partial(assert_close, rtol=RTOL_STRICT, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL_STRICT, atol=ATOL)
_assert_fields_close = partial(
    assert_fields_close, rtol=RTOL_STRICT, atol=ATOL
)


def _check_result(
    res,
    ref: dict,
    label: str,
    *,
    check_wu_hausman: bool = True,
    n_clusters: int | None = None,
    check_sargan: bool = True,
    rtol_wald: float = RTOL_STRICT,
) -> None:
    assert_inference_close(res, ref, label, rtol=RTOL_STRICT, atol=ATOL)
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"

    # `rtol_wald`: Wald/F統計量とp値だけの相対許容誤差。実データ（card）×クラスターで
    # クラスター共分散Vの精度（約3e-12）が`V_slopes`の条件数（約3e7）で増幅され、
    # 他の統計量より緩い値が要る（`_tolerances_iv.py`の`rtol_wald_card_cluster`参照）。
    _assert_fields_close(res, ref, ("r_squared", "adj_r_squared"), label)
    assert_fields_close(
        res,
        ref,
        (("wald_statistic", "f_statistic"), ("wald_p_value", "f_p_value")),
        label,
        rtol=rtol_wald,
        atol=ATOL,
    )

    _assert_dict_close(
        res.weak_instrument_f_statistics,
        ref["weak_instrument_f"],
        f"{label}/weak_instrument_f_statistics",
    )

    # check_sargan=False: 切片なしのivregは残差の中心化TSSを使うSarganで負の値に
    # なりうる（定数項がZに入っている前提の式）ため、フィクスチャ側がnullで
    # 比較不能。本実装側の値はlinearmodelsとの比較（`test_iv_reference.py`）で
    # 検証する。
    if not check_sargan:
        pass
    elif ref["sargan_statistic"] is None:
        assert res.overid_statistic is None, f"{label}/overid_statistic"
        assert res.overid_p_value is None, f"{label}/overid_p_value"
    else:
        _assert_close(
            res.overid_statistic,
            ref["sargan_statistic"],
            f"{label}/overid_statistic",
        )
        assert_p_value_close(
            res.overid_p_value,
            ref["sargan_p_value"],
            f"{label}/overid_p_value",
            rtol=RTOL_STRICT,
        )

    # wu_hausmanは全cov_typeでフィクスチャに実測値を持つ。
    # 境界的なサンプルサイズ（df1シナリオ等、拡張回帰がsaturatedになる）では
    # 本実装・ivreg双方がNoneを返すため、refがNoneのケースは本実装側もNoneに
    # なることだけ確認する。
    # check_wu_hausman=Falseの場合は比較自体を丸ごとスキップする（cluster_g2:
    # 拡張回帰の傾き係数がq=2（endog1・第一段階残差）に対しG=2クラスタでは
    # G-1=1<qとなり構造的にクラスタロバスト共分散が特異になる——本実装のG≤qの
    # 罠と同じ原理（engine/src/iv/CLAUDE.md参照）——ため本実装は正しくNoneを
    # 返すが、ivreg側はこの構造的特異性を検出せず値を返すため比較不能）。
    if not check_wu_hausman:
        pass
    elif ref["wu_hausman_statistic"] is None:
        assert res.wu_hausman_statistic is None, (
            f"{label}/wu_hausman_statistic"
        )
        assert res.wu_hausman_p_value is None, f"{label}/wu_hausman_p_value"
    else:
        _assert_close(
            res.wu_hausman_statistic,
            ref["wu_hausman_statistic"],
            f"{label}/wu_hausman_statistic",
        )
        if n_clusters is None:
            assert_p_value_close(
                res.wu_hausman_p_value,
                ref["wu_hausman_p_value"],
                f"{label}/wu_hausman_p_value",
                rtol=RTOL_STRICT,
            )
        else:
            # cluster cov_typeはivregのWald検定がF分布の分母自由度にクラスター数を
            # 反映せず（n-k_augのまま）、フィクスチャのp値は`null`。本実装は
            # G-1を使うため、Rの統計量からF(n_endog, G-1)でp値を再計算して比較する
            # （モジュールdocコメント参照）。絶対誤差の下限は付けない（裾のp値が
            # 0に潰れても通ってしまうため）。
            assert ref["wu_hausman_p_value"] is None, (
                f"{label}/wu_hausman_p_value"
            )
            # 自由度は本実装の値を検証対象にせず、参照側から独立に決まる値
            # （内生変数の数、クラスター数-1）と照合する。
            n_endog = len(ref["weak_instrument_f"])
            assert res.wu_hausman_df_num == n_endog, (
                f"{label}/wu_hausman_df_num"
            )
            assert res.wu_hausman_df_denom == n_clusters - 1, (
                f"{label}/wu_hausman_df_denom"
            )
            recomputed = sps.f.sf(
                ref["wu_hausman_statistic"], n_endog, n_clusters - 1
            )
            assert_p_value_close(
                res.wu_hausman_p_value,
                float(recomputed),
                f"{label}/wu_hausman_p_value_recomputed",
                rtol=RTOL_STRICT,
            )


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_r(crosscheck, scenario, cov_type):
    x_exog = scenario_x_exog(scenario)
    instruments = scenario_instruments(scenario)
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=x_exog,
        x_endog=["endog1"],
        instruments=instruments,
        options=options,
    ).fit()

    ref = crosscheck[scenario][cov_type]
    _check_result(res, ref, f"{scenario}/{cov_type}/R")


def test_cluster_matches_r(crosscheck):
    """クラスターロバストSE。`generate_iv_crosscheck_fixtures.py`と同じ疑似
    グループ（行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ（`params`/`std_errors`/`r_squared`等が記録されている）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = crosscheck["baseline"]["cluster"]
    _check_result(
        res,
        ref,
        "cluster/R",
        n_clusters=df["cluster_group"].n_unique(),
    )


def test_cluster_imbalanced_matches_r(crosscheck):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = crosscheck["baseline"]["cluster_imbalanced"]
    _check_result(
        res,
        ref,
        "cluster_imbalanced/R",
        n_clusters=df["cluster_group"].n_unique(),
    )


def test_cluster_g2_matches_r(crosscheck):
    """クラスタ数境界（G=2ちょうど）の成功パス（`test_iv_reference.py`の同名テスト
    と同じ再現条件）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_g2.csv")
    df = df.with_columns((pl.int_range(pl.len()) % 2).alias("cluster_group"))
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()

    ref = crosscheck["baseline"]["cluster_g2"]
    _check_result(
        res,
        ref,
        "cluster_g2/R",
        check_wu_hausman=False,
    )
    # 拡張回帰の傾き係数の数q_aug=2がG=2以上のため、本実装はNoneを返す
    # （ivreg側は値を返すので比較はしない）。
    assert res.wu_hausman_statistic is None
    assert res.wu_hausman_p_value is None


@pytest.mark.parametrize("scenario", CLUSTER_ILL_CONDITIONED_SCENARIOS)
def test_cluster_ill_conditioned_matches_r(crosscheck, scenario):
    """悪条件・多重共線性シナリオ×クラスターロバストSE（`test_iv_reference.py`の
    同名テストと同じ再現条件）。
    """
    df = with_cluster_groups(pl.read_csv(DATA_DIR / f"iv_{scenario}.csv"), 10)
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=scenario_x_exog(scenario),
        x_endog=["endog1"],
        instruments=scenario_instruments(scenario),
        options=options,
    ).fit()

    _check_result(
        res,
        crosscheck[scenario]["cluster"],
        f"{scenario}/cluster/R",
        n_clusters=df["cluster_group"].n_unique(),
    )


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
@pytest.mark.parametrize("scenario", MULTI_ENDOG_SCENARIOS)
def test_multi_endog_matches_r(crosscheck, scenario, cov_type):
    """複数内生変数（`x_endog=["endog1", "endog2"]`）の成功パス
    （`test_iv_reference.py`の同名テストと同じ理由・シナリオ）。
    """
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}_multi_endog.csv")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=multi_endog_instruments(scenario),
        options=options,
    ).fit()

    ref = crosscheck["multi_endog"][scenario][cov_type]
    _check_result(res, ref, f"multi_endog/{scenario}/{cov_type}/R")


def test_multi_endog_cluster_matches_r(crosscheck):
    """複数内生変数×クラスターロバストSE（`test_iv_reference.py`の同名テストと
    同じ再現条件）。Wu-Hausmanのp値は、Rの統計量とF(2, G-1)から再計算して比較する。
    """
    df = with_cluster_groups(
        pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv"), 10
    )
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2", "z3"],
        options=options,
    ).fit()

    _check_result(
        res,
        crosscheck["multi_endog"]["baseline"]["cluster"],
        "multi_endog/baseline/cluster/R",
        n_clusters=df["cluster_group"].n_unique(),
    )


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
def test_three_endog_matches_r(crosscheck, cov_type):
    """内生変数3本（過剰識別度1、`test_iv_reference.py`の同名テストと同じ再現条件）。"""
    df = pl.read_csv(DATA_DIR / "iv_baseline_three_endog.csv")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=THREE_ENDOG_X_ENDOG,
        instruments=THREE_ENDOG_INSTRUMENTS,
        options=options,
    ).fit()

    ref = crosscheck["three_endog"][cov_type]
    _check_result(res, ref, f"three_endog/{cov_type}/R")
    assert res.wu_hausman_df_num == 3


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
def test_df1_matches_r(crosscheck, cov_type):
    """自由度1境界（df_resid=1ちょうど）の成功パス（`test_iv_reference.py`の
    同名テストと同じ再現条件）。x_exog=[]・x_endog=['endog1']・
    instruments=['z1']（丁度識別、n=3）。augmented regressionがsaturated
    （残差自由度0）になるため、wu_hausman_statistic/wu_hausman_p_valueは
    本実装・ivreg双方でNoneになる（`_check_result`参照）。hacも含む（R側は
    `adjust = FALSE`で、`adjust = TRUE`の`n/(n-k)`がn=3で標準誤差を42%ずらす
    のを避けている）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_df1.csv")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()

    ref = crosscheck["df1"][cov_type]
    _check_result(res, ref, f"df1/{cov_type}/R")


# ── 凍結フィクスチャとの数値照合（オプション） ────────────────────────


@pytest.mark.parametrize(
    "cov_type", LINEARMODELS_COV_TYPES + IVREG_ONLY_COV_TYPES
)
@pytest.mark.parametrize(
    "scenario",
    [
        name
        for name in NO_INTERCEPT_SCENARIOS
        if name not in IVREG_UNSUPPORTED_NO_INTERCEPT
    ],
)
def test_no_intercept_matches_r(crosscheck, scenario, cov_type):
    """`include_intercept=False`（切片なし）。ivregの式は両辺に`0 +`を付ける。

    - `x_exog_empty`（`x_exog=[]`かつ切片なし）はivregがfitできないため対象外
      （`IVREG_UNSUPPORTED_NO_INTERCEPT`、linearmodelsのテストで検証）。
    - Sarganは比較しない（`_check_result`の`check_sargan`参照）。
    - hc2/hc3はlinearmodelsに実装が無く、ivregだけが参照になる（レバレッジが
      切片の有無で変わる）。
    - hacの小標本補正の慣習差（ivregの`NeweyWest(adjust=TRUE)`が掛ける`n/(n-k)`、
      本実装・linearmodelsは掛けない）は切片ありと同じ。R側を`adjust = FALSE`にして揃えている。
    """
    spec = NO_INTERCEPT_SCENARIOS[scenario]
    df = pl.read_csv(DATA_DIR / f"iv_{spec['dataset']}.csv")
    options = IVOptions(
        include_intercept=False,
        cov_type=cov_type,
        **hac_time_for(cov_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=spec["x_exog"],
        x_endog=spec["x_endog"],
        instruments=spec["instruments"],
        options=options,
    ).fit()

    ref = crosscheck["no_intercept"][scenario][cov_type]
    label = f"no_intercept/{scenario}/{cov_type}/R"
    assert "const" not in res.param_names, label
    _check_result(res, ref, label, check_sargan=False)


def test_no_intercept_cluster_matches_r(crosscheck):
    """`include_intercept=False`×クラスターロバストSE（`baseline`、疑似グループ
    行番号%10）。cluster cov_typeのWu-Hausman p値は、切片ありの
    `test_cluster_matches_r`と同じくRの統計量とG-1から再計算して比較する。
    """
    df = with_cluster_groups(pl.read_csv(DATA_DIR / "iv_baseline.csv"), 10)
    options = IVOptions(
        include_intercept=False, cov_type="cluster", cluster="cluster_group"
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    assert "const" not in res.param_names
    _check_result(
        res,
        crosscheck["no_intercept"]["baseline"]["cluster"],
        "no_intercept/baseline/cluster/R",
        n_clusters=df["cluster_group"].n_unique(),
        check_sargan=False,
    )


# ── 凍結フィクスチャとの数値照合（実データ） ──────────────────────────


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
def test_card_matches_r(crosscheck_wooldridge, cov_type):
    """実データセット（Wooldridge card）。`test_iv_reference.py`の同名テストと
    同じ理由。
    """
    df = load_wooldridge_dataset("card")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        options=options,
    ).fit()

    ref = crosscheck_wooldridge["card"][cov_type]
    _check_result(res, ref, f"card/{cov_type}/R")


def test_card_cluster_matches_r(crosscheck_wooldridge):
    """実データのグループ列（居住地域の9グループ）でのクラスターロバストSE
    （`test_iv_reference.py`の同名テストと同じ理由）。Wu-Hausmanのp値は、Rの統計量と
    F(1, G-1)から再計算して比較する。
    """
    df = with_card_region(load_wooldridge_dataset("card"))
    options = IVOptions(cov_type="cluster", cluster=CARD_REGION_COLUMN)
    res = IV(
        with_row_time(df),
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        options=options,
    ).fit()

    _check_result(
        res,
        crosscheck_wooldridge["card"]["cluster"],
        "card/cluster/R",
        n_clusters=df[CARD_REGION_COLUMN].n_unique(),
        rtol_wald=TOLERANCES["iv_crosscheck"]["rtol_wald_card_cluster"],
    )
