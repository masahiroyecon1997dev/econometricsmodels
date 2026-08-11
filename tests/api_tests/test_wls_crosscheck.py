"""WLSの独立実装（R: lm(weights=) + sandwich/lmtest）とのクロスチェックテスト。

主リファレンス（statsmodels）との厳密比較は`test_wls_fixtures.py`で行う。ここでは
`tests/api_tests/fixtures/benchmarks/wls_crosscheck.json`
（`benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py`で生成）
を用いて、statsmodelsとは独立した実装（R）との一致を確認する。役割分担・
pyfixest除外の理由は`test_ols_crosscheck.py`と同じ。

classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-13〜1e-15
程度）ため`RTOL_STRICT`で厳密比較する。**HACのみOLSより乖離が大きく、実測で
最大相対誤差約4.3%**（OLSの実測約0.4%の10倍程度）だったため、`RTOL_HAC`は
OLSの1e-2ではなく5e-2を採用する（`docs/spec/wls-spec.md`「テスト」参照。`testing-policy.md`「同じクロスチェック用
パッケージでも、統計量・cov_typeごとに実測乖離が大きく異なる場合は、許容誤差を
分けてよい」に従う）。

Note:
    合成データはフィクスチャ生成時と同じ入力データを、`tests/api_tests/
    fixtures/benchmarks/data/`に固定済みのCSVから読む（重み列`weight`も
    同じCSVに含まれる）。401ksubs（Wooldridge）は`load_wooldridge.py`経由で
    都度ロードする（データの再配布ライセンスが未確認のためCSVとして
    固定しない）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "linear"
        / "fixtures"
    ),
)
from _common import imbalanced_cluster_groups
from econometricsmodels import WLS, OLSOptions
from generate_wls_crosscheck_fixtures import (
    NUMERIC_SCENARIOS as SYNTHETIC_SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "benchmarks"
    / "wls_crosscheck.json"
)
DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"

# classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-13〜1e-15
# 程度）。testing-policy.md「許容誤差」の基本方針（相対誤差1e-8）と揃える。
RTOL_STRICT = 1e-8

# HACのみ実測最大相対誤差約4.3%（wls-spec.md「テスト」参照）。
RTOL_HAC = 5e-2


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _assert_close(
    ours: dict[str, float],
    reference: dict[str, float],
    label: str,
    rtol: float = RTOL_STRICT,
) -> None:
    for name, ref_val in reference.items():
        our_val = ours[name]
        diff = abs(our_val - ref_val)
        # test_ols_crosscheck.pyと異なりmax(rtol*|ref|, ATOL)の順（絶対誤差フロアとして
        # 使う）。cluster/f_p_valueが5e-13程度まで下がるケースがあり、`rtol*max(|ref|,1e-8)`
        # のままだと許容誤差自体が1e-16まで縮んでしまい、ほぼゼロ同士の比較で偽陽性の
        # 失敗になることが判明したため（WLSのRクロスチェックテスト作成時に発覚）。
        tol = max(rtol * abs(ref_val), 1e-8)
        assert diff <= tol, (
            f"[{label}] {name}: ours={our_val:.6f}, reference={ref_val:.6f}, "
            f"diff={diff:.6f} > tol={tol:.6f}"
        )


def _assert_scalar_close(
    our_val: float, ref_val: float, label: str, rtol: float = RTOL_STRICT
) -> None:
    diff = abs(our_val - ref_val)
    # test_ols_crosscheck.pyと異なりmax(rtol*|ref|, ATOL)の順（絶対誤差フロアとして
    # 使う）。cluster/f_p_valueが5e-13程度まで下がるケースがあり、`rtol*max(|ref|,1e-8)`
    # のままだと許容誤差自体が1e-16まで縮んでしまい、ほぼゼロ同士の比較で偽陽性の
    # 失敗になることが判明したため（WLSのRクロスチェックテスト作成時に発覚）。
    tol = max(rtol * abs(ref_val), 1e-8)
    assert diff <= tol, (
        f"[{label}] ours={our_val:.6f}, reference={ref_val:.6f}, "
        f"diff={diff:.6f} > tol={tol:.6f}"
    )


def _assert_fit_stats_close(res, ref: dict, label: str, rtol: float) -> None:
    """AIC・BIC・対数尤度・F統計量・F検定p値の検証。

    AIC/BIC/対数尤度はcov_typeに依存しないため常にRTOL_STRICTで比較する。
    F統計量・F検定p値はcov_typeごとのロバストWald検定のため呼び出し元の
    rtol（HACのみRTOL_HAC）を使う。
    """
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
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 10).alias("cluster_group"))
        .drop("_row")
    )
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
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 2).alias("cluster_group"))
        .drop("_row")
    )
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


def test_401ksubs_matches_r(crosscheck):
    """実データ（401ksubs、fsize==1）でのWLSクロスチェック。回帰式・重み定義は
    `test_wls_fixtures.py::test_401ksubs_matches_statsmodels`と揃える。
    """
    pytest.importorskip("wooldridge")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
    from load_wooldridge import load as load_wooldridge

    df = load_wooldridge("401ksubs").filter(pl.col("fsize") == 1)
    df = df.with_columns((1.0 / pl.col("inc")).alias("inv_inc"))

    res = WLS(
        df,
        y="nettfa",
        x=["inc", "incsq", "age", "agesq", "male", "e401k"],
        weight="inv_inc",
    ).fit()

    ref = crosscheck["401ksubs"]["r"]
    _assert_close(res.params, ref["coef"], "401ksubs/R coef")
    _assert_close(res.std_errors, ref["se"], "401ksubs/R se")
    _assert_fit_stats_close(res, ref, "401ksubs/R", rtol=RTOL_STRICT)
