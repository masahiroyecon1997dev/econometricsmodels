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
値で個別に設定する。ユーザー確認済み）。

`f_p_value`は浮動小数点アンダーフローに近い極小値（1e-9〜1e-12オーダー）になる
ケースがあり、その領域では相対誤差比較が意味を持たない（F統計量自体は0.6%程度
しか違わなくても、F分布の裾の確率はその差を大きく増幅する）。`f_p_value`の比較
のみ`ATOL_F_PVALUE`（絶対誤差フロア、実測最大乖離1.523e-6にマージンを載せた値）を
使う（他の統計量の比較式は変更しない。ユーザー確認済み）。

Note:
    - `hc2`/`hc3`はivreg側にレバレッジ算出の確立した参照実装が無いため対象外
      （`iv-api-design.md`3.1節）。
    - `weak_instrument_f`・`sargan_statistic`/`sargan_p_value`はivregの
      `summary(diagnostics=TRUE)`が常にclassical vcovで計算する仕様のため、
      全cov_typeで同じ値になる（`run_ivreg_benchmark.R`参照）。
    - `wu_hausman_statistic`/`wu_hausman_p_value`はclassical cov_typeのときのみ
      フィクスチャに実測値があり、他のcov_typeは`None`（ivregの`diagnostics=TRUE`が
      classical vcov固定のため。hc0/hc1/clusterは`test_iv_fixtures.py`
      （linearmodels）側で既にクロスチェック済み。ユーザー確認済み）。
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
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "benchmark" / "iv" / "fixtures"),
)
from _common import imbalanced_cluster_groups
from _helpers import DATA_DIR, with_cluster_groups
from _tolerances import TOLERANCES
from econometricsmodels import IV, IvOptions
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

COV_TYPES = ["classical", "hc0", "hc1", "hac"]

INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())["synthetic"]


def _assert_close(
    our_val: float, ref_val: float, label: str, rtol: float = RTOL_STRICT
) -> None:
    diff = abs(our_val - ref_val)
    tol = rtol * max(abs(ref_val), 1e-8)
    assert diff <= tol, (
        f"[{label}] ours={our_val:.6f}, reference={ref_val:.6f}, "
        f"diff={diff:.6f} > tol={tol:.6f}"
    )


def _assert_dict_close(
    ours: dict[str, float],
    ref: dict[str, float],
    label: str,
    rtol: float = RTOL_STRICT,
) -> None:
    for name, ref_val in ref.items():
        _assert_close(ours[name], ref_val, f"{label}/{name}", rtol=rtol)


def _assert_p_value_close(
    our_val: float, ref_val: float, label: str, rtol: float
) -> None:
    """f_p_value専用の比較。アンダーフローに近い極小値では相対誤差比較が
    意味を持たなくなるため、絶対誤差フロア（`ATOL_F_PVALUE`）を使う
    （モジュールdocコメント参照）。
    """
    diff = abs(our_val - ref_val)
    tol = max(rtol * abs(ref_val), ATOL_F_PVALUE)
    assert diff <= tol, (
        f"[{label}] ours={our_val!r}, reference={ref_val!r}, "
        f"diff={diff!r} > tol={tol!r}"
    )


def _check_result(res, ref: dict, label: str, rtol: float) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef", rtol=rtol)
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se", rtol=rtol)
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
        {"endog1": ref["weak_instrument_f"]},
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

    # wu_hausmanはclassical cov_typeのときのみフィクスチャに実測値がある
    # （モジュールdocコメント参照）。他のcov_typeはrefがNoneなので比較をスキップする。
    if ref["wu_hausman_statistic"] is not None:
        _assert_close(
            res.wu_hausman_statistic,
            ref["wu_hausman_statistic"],
            f"{label}/wu_hausman_statistic",
        )
        _assert_close(
            res.wu_hausman_p_value,
            ref["wu_hausman_p_value"],
            f"{label}/wu_hausman_p_value",
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
        # small_nのみ実測乖離が大きいため専用の緩めたRTOLを使う
        # （モジュールdocコメント参照）。
        rtol = RTOL_HAC_SMALL_N if scenario == "small_n" else RTOL_HAC
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
    _check_result(res, ref, "cluster/R", rtol=RTOL_STRICT)


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
    _check_result(res, ref, "cluster_imbalanced/R", rtol=RTOL_STRICT)
