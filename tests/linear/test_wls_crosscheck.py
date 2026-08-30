"""WLSの独立実装（R: lm(weights=) + sandwich/lmtest）とのクロスチェックテスト。

主リファレンス（statsmodels）との厳密比較は`test_wls_fixtures.py`で行う。ここでは
`tests/fixtures/benchmarks/wls_crosscheck.json`
（`benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py`で生成）
を用いて、statsmodelsとは独立した実装（R）との一致を確認する。役割分担・
pyfixest除外の理由は`test_ols_crosscheck.py`と同じ。

classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-13〜1e-15
程度）ため`RTOL_STRICT`で厳密比較する。**HACのみOLSより乖離が大きく、実測で
最大相対誤差約4.3%**（OLSの実測約0.4%の10倍程度）だったため、`RTOL_HAC`は
OLSの1e-2ではなく5e-2を採用する（`docs/spec/wls-spec.md`「テスト」参照。`testing-policy.md`「同じクロスチェック用
パッケージでも、統計量・cov_typeごとに実測乖離が大きく異なる場合は、許容誤差を
分けてよい」に従う）。

係数・標準誤差に加え、t値・p値・信頼区間・R²・調整済みR²・AIC・BIC・対数尤度・
F統計量・F検定p値も検証する（`test_ols_crosscheck.py`と同じ方針）。p_valuesのみ
HAC/autocorrelatedで裾確率がゼロ近傍に潰れるため絶対誤差フロア（`ATOL_P_VALUE`）
で比較する。

Note:
    合成データはフィクスチャ生成時と同じ入力データを、`tests/
    fixtures/benchmarks/data/`に固定済みのCSVから読む（重み列`weight`も
    同じCSVに含まれる）。401ksubs（Wooldridge）は`load_wooldridge.py`経由で
    都度ロードする（データの再配布ライセンスが未確認のためCSVとして
    固定しない）。
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _helpers import DATA_DIR, load_wooldridge_dataset, with_cluster_groups
from _tolerances import TOLERANCES
from econometricsmodels import WLS, OLSOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.linear.fixtures.generate_wls_crosscheck_fixtures import (
    NUMERIC_SCENARIOS as SYNTHETIC_SCENARIOS,
)
from benchmark.linear.fixtures.generate_wls_crosscheck_fixtures import (
    WOOLDRIDGE_COV_TYPES,
)
from benchmark.linear.fixtures.generate_wls_fixtures import _add_age_bin

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "wls_crosscheck.json"
)

# classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-13〜1e-15
# 程度）。testing-policy.md「許容誤差」の基本方針（相対誤差1e-8）と揃える。
RTOL_STRICT = TOLERANCES["wls_crosscheck"]["rtol_strict"]

# HACのみ実測最大相対誤差約4.3%（wls-spec.md「テスト」参照）。
RTOL_HAC = TOLERANCES["wls_crosscheck"]["rtol_hac"]

# 絶対誤差フロア（ref値が0近傍のとき相対誤差比較が意味を持たなくなるのを防ぐ）。
ATOL = TOLERANCES["wls_crosscheck"]["atol"]

# p_values専用の絶対誤差フロア（HAC/autocorrelatedで裾確率がゼロ近傍に
# 潰れるため、_tolerances.py参照）。
ATOL_P_VALUE = TOLERANCES["wls_crosscheck"]["atol_p_value"]


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


# _assert_close（dict版）はtests/_assertions.pyのassert_dict_closeに、
# _assert_scalar_close（scalar版）はassert_closeに対応する
# （test_ols_crosscheck.pyと同じ計算式に統合）。
_assert_close = partial(assert_dict_close, rtol=RTOL_STRICT, atol=ATOL)
_assert_scalar_close = partial(assert_close, rtol=RTOL_STRICT, atol=ATOL)


def _assert_fit_stats_close(res, ref: dict, label: str, rtol: float) -> None:
    """R²・調整済みR²・AIC・BIC・対数尤度・F統計量・F検定p値・t値・p値・
    信頼区間の検証（`test_ols_crosscheck.py`と同じ方針）。
    """
    _assert_scalar_close(res.r_squared, ref["r_squared"], f"{label}/r_squared")
    _assert_scalar_close(
        res.r_squared_adj, ref["r_squared_adj"], f"{label}/r_squared_adj"
    )
    _assert_scalar_close(res.aic, ref["aic"], f"{label}/aic")
    _assert_scalar_close(res.bic, ref["bic"], f"{label}/bic")
    _assert_scalar_close(
        res.log_likelihood, ref["log_likelihood"], f"{label}/log_likelihood"
    )
    _assert_scalar_close(
        res.f_statistic, ref["f_statistic"], f"{label}/f_statistic", rtol=rtol
    )
    _assert_scalar_close(
        res.f_p_value, ref["f_p_value"], f"{label}/f_p_value", rtol=rtol
    )
    _assert_close(res.t_stats, ref["t_stats"], f"{label}/t_stats", rtol=rtol)
    _assert_close(
        res.p_values,
        ref["p_values"],
        f"{label}/p_values",
        rtol=rtol,
        atol=ATOL_P_VALUE,
    )
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_scalar_close(
            our_lower, ref_lower, f"{label}/conf_lower/{name}", rtol=rtol
        )
        _assert_scalar_close(
            our_upper, ref_upper, f"{label}/conf_upper/{name}", rtol=rtol
        )


NON_HAC_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3"]


@pytest.mark.parametrize("cov_type", NON_HAC_COV_TYPES)
@pytest.mark.parametrize("scenario", SYNTHETIC_SCENARIOS)
def test_synthetic_matches_r(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    options = OLSOptions(cov_type=cov_type)
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    ref = crosscheck["synthetic"][scenario][cov_type]["r"]
    label = f"{scenario}/{cov_type}/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_cluster_matches_r(crosscheck):
    """クラスターロバストSE。generate_wls_crosscheck_fixtures.pyと同じ疑似
    グループ（行番号%10）を再現する。統計的な意味はなく、実装の動作確認用。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster"]["r"]
    _assert_close(res.params, ref["coef"], "cluster/R coef")
    _assert_close(res.std_errors, ref["se"], "cluster/R se")
    _assert_fit_stats_close(res, ref, "cluster/R", rtol=RTOL_STRICT)


def test_cluster_imbalanced_matches_r(crosscheck):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル、
    OLSの同種ケース相当）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster_imbalanced"]["r"]
    _assert_close(res.params, ref["coef"], "cluster_imbalanced/R coef")
    _assert_close(res.std_errors, ref["se"], "cluster_imbalanced/R se")
    _assert_fit_stats_close(res, ref, "cluster_imbalanced/R", rtol=RTOL_STRICT)


def test_cluster_g2_matches_r(crosscheck):
    """クラスタ数境界（G=2ちょうど）の成功パス（OLSの同種ケース相当）。

    説明変数1個（q=1）に絞っている。baseline既定の3個のままG=2にすると、
    ロバストWald検定の共分散部分行列（3x3）のランクがG=2以下となり必然的に
    特異になりComputationErrorになる（成功パスにならない。
    `test_wls_fixtures.py::test_cluster_g2_with_multiple_slopes_raises_computation_error`
    参照）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv")
    df = with_cluster_groups(df, 2)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(df, y="y", x=["x1"], weight="weight", options=options).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster_g2"]["r"]
    _assert_close(res.params, ref["coef"], "cluster_g2/R coef")
    _assert_close(res.std_errors, ref["se"], "cluster_g2/R se")
    _assert_fit_stats_close(res, ref, "cluster_g2/R", rtol=RTOL_STRICT)


def test_hac_matches_r(crosscheck):
    """HAC標準誤差。フィクスチャ生成時に本実装の自動ラグ式で計算した
    ラグ（`hac_lag`）をそのまま使い、ラグ選択方式自体の違いを比較対象から
    除外した上でNewey-West公式自体の妥当性を確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_autocorrelated.csv")
    entry = crosscheck["synthetic"]["autocorrelated"]["hac"]
    options = OLSOptions(cov_type="hac", hac_lags=entry["hac_lag"])
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    ref = entry["r"]
    _assert_close(res.params, ref["coef"], "hac/R coef")
    _assert_close(res.std_errors, ref["se"], "hac/R se", rtol=RTOL_HAC)
    _assert_fit_stats_close(res, ref, "hac/R", rtol=RTOL_HAC)


@pytest.mark.parametrize("cov_type", WOOLDRIDGE_COV_TYPES)
def test_401ksubs_matches_r(crosscheck, cov_type):
    """実データ（401ksubs、fsize==1）でのWLSクロスチェック。回帰式・重み定義は
    `test_wls_fixtures.py::test_401ksubs_matches_statsmodels`と揃える。
    HACは時系列順の無いクロスセクションデータのため対象外
    （`generate_wls_crosscheck_fixtures.py`のWOOLDRIDGE_COV_TYPESと同じ方針）。
    """
    df = load_wooldridge_dataset("401ksubs").filter(pl.col("fsize") == 1)
    df = df.with_columns((1.0 / pl.col("inc")).alias("inv_inc"))
    options = OLSOptions(cov_type=cov_type)

    res = WLS(
        df,
        y="nettfa",
        x=["inc", "incsq", "age", "agesq", "male", "e401k"],
        weight="inv_inc",
        options=options,
    ).fit()

    ref = crosscheck["401ksubs"][cov_type]["r"]
    label = f"401ksubs/{cov_type}/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_401ksubs_cluster_matches_r(crosscheck):
    """実データ（401ksubs、fsize==1）でのクラスターロバストSE。地域等の実
    カテゴリ列が無いため、ageの分位ビン（8分位、`_add_age_bin`）を疑似的な
    クラスター列として使う（`test_wls_fixtures.py`
    ::test_401ksubs_cluster_matches_statsmodelsと同じグループ構成）。
    """
    df = load_wooldridge_dataset("401ksubs").filter(pl.col("fsize") == 1)
    df = df.with_columns((1.0 / pl.col("inc")).alias("inv_inc"))
    df = _add_age_bin(df)
    options = OLSOptions(cov_type="cluster", cluster_col="age_bin")

    res = WLS(
        df,
        y="nettfa",
        x=["inc", "incsq", "age", "agesq", "male", "e401k"],
        weight="inv_inc",
        options=options,
    ).fit()

    ref = crosscheck["401ksubs"]["cluster"]["r"]
    _assert_close(res.params, ref["coef"], "401ksubs/cluster/R coef")
    _assert_close(res.std_errors, ref["se"], "401ksubs/cluster/R se")
    _assert_fit_stats_close(res, ref, "401ksubs/cluster/R", rtol=RTOL_STRICT)
