"""IV(2SLS)の独立実装（R: ivreg + sandwich/lmtest）とのクロスチェックテスト。

主リファレンス（linearmodels）との厳密比較は`test_iv_fixtures.py`で行う。ここでは
`tests/fixtures/benchmarks/iv_crosscheck.json`
（`benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py`で生成）を用いて、
linearmodelsとは独立した実装（R `ivreg`）との一致を確認する
（`docs/planning/specs/iv-api-design.md`5.2節）。

シナリオ・cov_type・クラスタケースの構成は`test_iv_fixtures.py`と揃える。

classical/hc0/hc1/clusterはRとほぼ機械精度で一致する（OLSクロスチェックと同じ実測
傾向）ため`RTOL_STRICT`で厳密比較する。HACのみ小標本補正の慣習差により`RTOL_HAC`
（緩め）を使う（`test_ols_crosscheck.py`と同じ方針）。

`small_n`シナリオ（n=40, hac_lag=3）のみHACの乖離が他シナリオ（実測0.3〜0.8%程度）より
大きい（実測: SE最大3.8%、F統計量最大8.1%）。小標本×HACという組み合わせで
Newey-West小標本補正の慣習差がより強く出るためと考えられる（`testing-policy.md`
「許容誤差」の「統計量・cov_typeごとに実測乖離が大きく異なる場合は許容誤差を
分けてよい」という規定に従い、`RTOL_HAC_SMALL_N`を実測最大値にマージンを載せた
値で個別に設定する。ユーザー確認済み）。`high_variance`シナリオも同じ緩めたRTOLを
使う（f_statistic自体は0.6%程度しか違わないが、F分布の裾でp値がその差を増幅する。
実測相対誤差2.37%、Issue #231フェーズ4で発覚）。

`f_p_value`は浮動小数点アンダーフローに近い極小値（1e-9〜1e-12オーダー）になる
ケースがあり、その領域では相対誤差比較が意味を持たない（F統計量自体は0.6%程度
しか違わなくても、F分布の裾の確率はその差を大きく増幅する）。`f_p_value`の比較
のみ`ATOL_F_PVALUE`（絶対誤差フロア、実測最大乖離1.523e-6にマージンを載せた値）を
使う（他の統計量の比較式は変更しない。ユーザー確認済み）。係数ごとの`p_values`
（Issue #232で追加）も同じ理由（t分布の裾での増幅）でhacケースにて実測乖離が
`RTOL_HAC`を超えることがあるため、同じ`ATOL_F_PVALUE`を使う。

Note:
    - `hc2`/`hc3`はivreg側にレバレッジ算出の確立した参照実装が無いため対象外
      （`iv-api-design.md`3.1節）。
    - `weak_instrument_f`・`sargan_statistic`/`sargan_p_value`はivregの
      `summary(diagnostics=TRUE)`が常にclassical vcovで計算する仕様のため、
      全cov_typeで同じ値になる（`run_ivreg_benchmark.R`参照）。
    - `wu_hausman_statistic`/`wu_hausman_p_value`は全cov_typeでフィクスチャに
      実測値がある（Issue #233。`summary(diagnostics=TRUE, vcov.=<関数>)`で
      cov_type別のロバスト共分散を診断表に反映できることが判明、
      `run_ivreg_benchmark.R`のモジュールコメント参照）。ただしcluster
      cov_typeのみ、ivreg側のWald検定がF分布の分母自由度にクラスター数を
      反映しない既知の制約によりp値が一致しないため、`_check_result`の
      `check_wu_hausman_p_value=False`で統計量のみ比較する（ユーザー確認済み）。
    - GMMはivregが対応していないため対象外（5.3節、Rクロスチェック省略の例外規定）。
    - 第一段階回帰の結果（`first_stage()`）自体はここでは比較しない
      （`test_ols_crosscheck.py`が既にOLSの数値一致を検証済みのため、
      `test_iv_fixtures.py`と同じ理由）。

    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/freeze_datasets.py`参照）から読む。
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "benchmark" / "iv" / "fixtures"),
)
from _assertions import assert_close, assert_dict_close
from _common import imbalanced_cluster_groups
from _helpers import DATA_DIR, load_wooldridge_dataset, with_cluster_groups
from _tolerances import TOLERANCES
from econometricsmodels import IV, IvOptions
from generate_iv_crosscheck_fixtures import CARD_X_EXOG
from generate_iv_crosscheck_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "benchmarks"
    / "iv_crosscheck.json"
)

# classical/hc0/hc1/clusterはRとほぼ機械精度で一致する（OLSクロスチェックの実測
# 傾向と同じ、`test_ols_crosscheck.py`参照）。
RTOL_STRICT = TOLERANCES["iv_crosscheck"]["rtol_strict"]

# HACのみ小標本補正の慣習差により実測で相対誤差0.3〜0.8%程度の乖離がある
# （`test_ols_crosscheck.py`の`RTOL_HAC`と同じ理由）。
RTOL_HAC = TOLERANCES["iv_crosscheck"]["rtol_hac"]

# small_nシナリオ（n=40, hac_lag=3）のみ実測乖離がRTOL_HACを超える（SE最大3.8%、
# F統計量最大8.1%）ため専用に緩めた値を使う（モジュールdocコメント参照）。
RTOL_HAC_SMALL_N = TOLERANCES["iv_crosscheck"]["rtol_hac_small_n"]

# f_p_valueが浮動小数点アンダーフローに近い極小値のとき、相対誤差比較の代わりに
# 使う絶対誤差フロア（モジュールdocコメント参照、実測最大乖離1.523e-6にマージン）。
ATOL_F_PVALUE = TOLERANCES["iv_crosscheck"]["atol_f_pvalue"]

# f_p_value以外の統計量向けの絶対誤差フロア（フェーズ3.5でtest_wls_crosscheck.py
# と同じ計算式に修正）。
ATOL = TOLERANCES["iv_crosscheck"]["atol"]

# p_values/wu_hausman_p_value・conf_int・wu_hausman_statistic（Issue #232/#233で
# 追加）のhacケース専用の緩めた許容誤差（モジュールdocコメント・_tolerances.py
# 参照）。
ATOL_HAC_PVALUE = TOLERANCES["iv_crosscheck"]["atol_hac_pvalue"]
ATOL_HAC_CONF_INT = TOLERANCES["iv_crosscheck"]["atol_hac_conf_int"]
RTOL_HAC_WU_HAUSMAN = TOLERANCES["iv_crosscheck"]["rtol_hac_wu_hausman"]
RTOL_HAC_WU_HAUSMAN_SMALL_N = TOLERANCES["iv_crosscheck"][
    "rtol_hac_wu_hausman_small_n"
]

COV_TYPES = ["classical", "hc0", "hc1", "hac"]

INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())["synthetic"]


@pytest.fixture(scope="module")
def crosscheck_wooldridge() -> dict:
    return json.loads(FIXTURE_PATH.read_text())["wooldridge"]


# _assert_close/_assert_dict_closeはtests/_assertions.pyのassert_close/
# assert_dict_closeに対応する（フェーズ3.5で計算式のバグを修正した上で統合）。
_assert_close = partial(assert_close, rtol=RTOL_STRICT, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL_STRICT, atol=ATOL)

# f_p_value専用の比較。アンダーフローに近い極小値では相対誤差比較が意味を
# 持たなくなるため、絶対誤差フロアのみ`ATOL_F_PVALUE`に差し替える
# （モジュールdocコメント参照）。rtolは呼び出し元が都度指定する。
_assert_p_value_close = partial(assert_close, atol=ATOL_F_PVALUE)


def _check_result(
    res,
    ref: dict,
    label: str,
    rtol: float,
    *,
    check_wu_hausman: bool = True,
    check_wu_hausman_p_value: bool = True,
) -> None:
    # hac呼び出し（rtolがRTOL_HAC/RTOL_HAC_SMALL_N）ではp_values/conf_int/
    # wu_hausman_statisticの実測乖離がclassical/hc0/hc1/cluster向けの許容誤差を
    # 超えるため、専用に緩めた値を使う（モジュールdocコメント・_tolerances.py
    # 参照）。
    is_hac = rtol != RTOL_STRICT
    p_value_atol = ATOL_HAC_PVALUE if is_hac else ATOL_F_PVALUE
    conf_int_atol = ATOL_HAC_CONF_INT if is_hac else ATOL
    if rtol == RTOL_HAC_SMALL_N:
        wu_hausman_rtol = RTOL_HAC_WU_HAUSMAN_SMALL_N
    elif is_hac:
        wu_hausman_rtol = RTOL_HAC_WU_HAUSMAN
    else:
        wu_hausman_rtol = rtol

    _assert_dict_close(res.params, ref["coef"], f"{label}/coef", rtol=rtol)
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se", rtol=rtol)
    _assert_dict_close(res.stats, ref["t_stats"], f"{label}/stats", rtol=rtol)
    # p_valuesはf_p_valueと同じ理由（t分布の裾でp値が統計量の僅かな差を増幅する、
    # モジュールdocコメント参照）で絶対誤差フロアを使う。
    _assert_dict_close(
        res.p_values,
        ref["p_values"],
        f"{label}/p_values",
        rtol=rtol,
        atol=p_value_atol,
    )
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_close(
            our_lower,
            ref_lower,
            f"{label}/conf_lower/{name}",
            rtol=rtol,
            atol=conf_int_atol,
        )
        _assert_close(
            our_upper,
            ref_upper,
            f"{label}/conf_upper/{name}",
            rtol=rtol,
            atol=conf_int_atol,
        )
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"

    _assert_close(res.r_squared, ref["r_squared"], f"{label}/r_squared")
    _assert_close(
        res.r_squared_adj, ref["r_squared_adj"], f"{label}/r_squared_adj"
    )
    _assert_close(
        res.f_statistic, ref["f_statistic"], f"{label}/f_statistic", rtol=rtol
    )
    _assert_p_value_close(
        res.f_p_value, ref["f_p_value"], f"{label}/f_p_value", rtol=rtol
    )

    _assert_dict_close(
        res.weak_instrument_f_statistics,
        ref["weak_instrument_f"],
        f"{label}/weak_instrument_f_statistics",
    )

    if ref["sargan_statistic"] is None:
        assert res.overid_statistic is None, f"{label}/overid_statistic"
        assert res.overid_p_value is None, f"{label}/overid_p_value"
    else:
        _assert_close(
            res.overid_statistic,
            ref["sargan_statistic"],
            f"{label}/overid_statistic",
        )
        _assert_close(
            res.overid_p_value,
            ref["sargan_p_value"],
            f"{label}/overid_p_value",
        )

    # wu_hausmanは全cov_typeでフィクスチャに実測値を持つ（Issue #233）。
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
            rtol=wu_hausman_rtol,
        )
        # cluster cov_typeはivregのWald検定がF分布の分母自由度にクラスター数を
        # 反映しない既知の制約によりp値が一致しないため、統計量のみ比較する
        # （呼び出し元がcheck_wu_hausman_p_value=Falseを渡す。モジュール
        # docコメント参照、ユーザー確認済み）。
        if check_wu_hausman_p_value:
            _assert_close(
                res.wu_hausman_p_value,
                ref["wu_hausman_p_value"],
                f"{label}/wu_hausman_p_value",
                rtol=wu_hausman_rtol,
                atol=p_value_atol,
            )


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synthetic_matches_r(crosscheck, scenario, cov_type):
    x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
    instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    options = IvOptions(cov_type=cov_type)
    res = IV(
        df,
        y="y",
        x_exog=x_exog,
        x_endog=["endog1"],
        instruments=instruments,
        options=options,
    ).fit()

    if cov_type == "hac":
        # small_n/high_varianceのみ実測乖離が大きいため専用の緩めたRTOLを使う
        # （モジュールdocコメント参照）。high_varianceはf_statistic自体は0.6%
        # 程度しか違わないが、F分布の裾でp値がその差を増幅する（実測相対誤差
        # 2.37%、Issue #231フェーズ4で発覚）。
        rtol = (
            RTOL_HAC_SMALL_N
            if scenario in ("small_n", "high_variance")
            else RTOL_HAC
        )
    else:
        rtol = RTOL_STRICT
    ref = crosscheck[scenario][cov_type]
    _check_result(res, ref, f"{scenario}/{cov_type}/R", rtol=rtol)


def test_cluster_matches_r(crosscheck):
    """クラスターロバストSE。`generate_iv_crosscheck_fixtures.py`と同じ疑似
    グループ（行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ（`coef`/`se`/`r_squared`等が記録されている）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = IV(
        df,
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
        rtol=RTOL_STRICT,
        check_wu_hausman_p_value=False,
    )


def test_cluster_g2_matches_r(crosscheck):
    """クラスタ数境界（G=2ちょうど）の成功パス（`test_iv_fixtures.py`の同名テスト
    と同じ再現条件、Issue #231フェーズ4）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_g2.csv")
    df = df.with_columns((pl.int_range(pl.len()) % 2).alias("cluster_group"))
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = IV(
        df,
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
        rtol=RTOL_STRICT,
        check_wu_hausman=False,
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_multi_endog_matches_r(crosscheck, cov_type):
    """複数内生変数（`x_endog=["endog1", "endog2"]`）の成功パス
    （`test_iv_fixtures.py`の同名テストと同じ理由、Issue #231フェーズ4）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")
    options = IvOptions(cov_type=cov_type)
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2", "z3"],
        options=options,
    ).fit()

    rtol = RTOL_HAC if cov_type == "hac" else RTOL_STRICT
    ref = crosscheck["multi_endog"][cov_type]
    _check_result(res, ref, f"multi_endog/{cov_type}/R", rtol=rtol)


# df1（n=3）はhacを対象外にする（Newey-Westのラグ選択・小標本補正の慣習差が
# n=3では極端に増幅され統計的に意味のある比較にならない、実測でse最大42%乖離。
# ユーザー確認済み）。
DF1_COV_TYPES = [ct for ct in COV_TYPES if ct != "hac"]


@pytest.mark.parametrize("cov_type", DF1_COV_TYPES)
def test_df1_matches_r(crosscheck, cov_type):
    """自由度1境界（df_resid=1ちょうど）の成功パス（`test_iv_fixtures.py`の
    同名テストと同じ再現条件、Issue #235）。x_exog=[]・x_endog=['endog1']・
    instruments=['z1']（丁度識別、n=3）。augmented regressionがsaturated
    （残差自由度0）になるため、wu_hausman_statistic/wu_hausman_p_valueは
    本実装・ivreg双方でNoneになる（`_check_result`参照）。hacは対象外
    （モジュールdocコメント参照）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_df1.csv")
    options = IvOptions(cov_type=cov_type)
    res = IV(
        df,
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()

    rtol = RTOL_HAC if cov_type == "hac" else RTOL_STRICT
    ref = crosscheck["df1"][cov_type]
    _check_result(res, ref, f"df1/{cov_type}/R", rtol=rtol)


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_card_matches_r(crosscheck_wooldridge, cov_type):
    """実データセット（Wooldridge card）。`test_iv_fixtures.py`の同名テストと
    同じ理由、Issue #231フェーズ4）。
    """
    df = load_wooldridge_dataset("card")
    options = IvOptions(cov_type=cov_type)
    res = IV(
        df,
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        options=options,
    ).fit()

    rtol = RTOL_HAC if cov_type == "hac" else RTOL_STRICT
    ref = crosscheck_wooldridge["card"][cov_type]
    _check_result(res, ref, f"card/{cov_type}/R", rtol=rtol)


def test_cluster_imbalanced_matches_r(crosscheck):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = IV(
        df,
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
        rtol=RTOL_STRICT,
        check_wu_hausman_p_value=False,
    )
