"""OLSの独立実装（R: lm + sandwich/lmtest、pyfixest）とのクロスチェックテスト。

主リファレンス（statsmodels）との厳密比較は`test_ols.py`で行う。ここでは
`tests/api_tests/fixtures/benchmarks/ols_crosscheck.json`
（`benchmark/fixtures/generate_ols_crosscheck_fixtures.py`で生成、Issue #18）
を用いて、statsmodelsとは独立した実装との**緩い許容誤差**での一致を確認する。
目的は大きな乖離（実装バグの兆候）の検出であり、厳密一致は期待しない
（`.claude/rules/testing-policy.md`「許容誤差」参照）。

Note:
    合成データセットは`benchmark/generate_synthetic_datasets.py`の
    `generate_dataset()`（seed固定・決定論的）で、フィクスチャ生成時と
    同じデータを再生成する。フィクスチャJSON自体には生データを含めない
    （`ols.json`と同じ設計。`true_beta`等のメタ情報のみを持つ）ため、
    `benchmark/`をimport pathに追加する必要がある。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from generate_synthetic_datasets import generate_dataset  # noqa: E402

from econometricsmodels import OLS, OLSOptions  # noqa: E402

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "benchmarks"
    / "ols_crosscheck.json"
)

# 緩い許容誤差（相対誤差1%）。R（lm+sandwich）はMacKinnon-White系の標準的な
# HC0-3公式を採用しており、本実装・statsmodelsと高い精度で一致する。
RTOL = 1e-2

# pyfixest（fixest）はHC2/HC3に対し、標準的なMacKinnon-White公式に加えて
# sqrt(n/(n-k))倍の追加小標本補正（ssc）を掛ける仕様であることを検証時に確認した
# （R/sandwich・statsmodels・本実装はこの追加補正を行わない）。nが小さいほど
# 乖離が大きくなる（例: small_nシナリオ n=20, k=4 では約11.8%）ため、
# pyfixestとの比較のみ許容誤差を緩める。バグではなく既知の実装差。
RTOL_PYFIXEST = 0.15


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _assert_close(
    ours: dict[str, float],
    reference: dict[str, float],
    label: str,
    rtol: float = RTOL,
) -> None:
    for name, ref_val in reference.items():
        our_val = ours[name]
        diff = abs(our_val - ref_val)
        tol = rtol * max(abs(ref_val), 1e-8)
        assert diff <= tol, (
            f"[{label}] {name}: ours={our_val:.6f}, reference={ref_val:.6f}, "
            f"diff={diff:.6f} > tol={tol:.6f}"
        )


SYNTHETIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
]
NON_HAC_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3"]


@pytest.mark.parametrize("cov_type", NON_HAC_COV_TYPES)
@pytest.mark.parametrize("scenario", SYNTHETIC_SCENARIOS)
def test_synthetic_matches_r(crosscheck, scenario, cov_type):
    df, _ = generate_dataset(scenario)
    options = OLSOptions(cov_type=cov_type)
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"][scenario][cov_type]["r"]
    _assert_close(res.params, ref["coef"], f"{scenario}/{cov_type}/R coef")
    _assert_close(res.std_errors, ref["se"], f"{scenario}/{cov_type}/R se")


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize("scenario", SYNTHETIC_SCENARIOS)
def test_synthetic_matches_pyfixest(crosscheck, scenario, cov_type):
    df, _ = generate_dataset(scenario)
    options = OLSOptions(cov_type=cov_type)
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"][scenario][cov_type]["pyfixest"]
    _assert_close(
        res.params, ref["coef"], f"{scenario}/{cov_type}/pyfixest coef"
    )
    _assert_close(
        res.std_errors,
        ref["se"],
        f"{scenario}/{cov_type}/pyfixest se",
        rtol=RTOL_PYFIXEST,
    )


def test_cluster_matches_r(crosscheck):
    """クラスターロバストSE。generate_ols_crosscheck_fixtures.pyと同じ疑似
    グループ（行番号%10）を再現する。統計的な意味はなく、実装の動作確認用。
    """
    df, _ = generate_dataset("baseline")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 10).alias("cluster_group"))
        .drop("_row")
    )
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster"]["r"]
    _assert_close(res.params, ref["coef"], "cluster/R coef")
    _assert_close(res.std_errors, ref["se"], "cluster/R se")


def test_hac_matches_r(crosscheck):
    """HAC標準誤差。フィクスチャ生成時に本実装の自動ラグ式で計算した
    ラグ（`hac_lag`）をそのまま使い、ラグ選択方式自体の違いを比較対象から
    除外した上でNewey-West公式自体の妥当性を確認する。
    """
    df, _ = generate_dataset("autocorrelated")
    entry = crosscheck["synthetic"]["autocorrelated"]["hac"]
    options = OLSOptions(cov_type="hac", hac_lags=entry["hac_lag"])
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = entry["r"]
    _assert_close(res.params, ref["coef"], "hac/R coef")
    _assert_close(res.std_errors, ref["se"], "hac/R se")


WOOLDRIDGE_DATASETS = {
    "wage1": ("lwage", ["educ", "exper", "tenure"]),
    "gpa2": ("colgpa", ["sat", "hsperc", "tothrs"]),
}


@pytest.fixture(scope="module")
def load_wooldridge():
    """`wooldridge`パッケージ（benchmark依存グループ）が無い環境ではskipする。

    tests/api_tests本体はtest依存グループのみで完結させる方針
    （.claude/rules/testing-policy.md、CLAUDE.md 3章「benchmark/はtests/とは
    別ライフサイクル」）のため、実データクロスチェックのみ任意扱いにする。
    """
    pytest.importorskip("wooldridge")
    from load_wooldridge import load

    return load


@pytest.mark.parametrize("cov_type", ["classical", "hc0", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize("dataset_name", list(WOOLDRIDGE_DATASETS))
def test_wooldridge_matches_r(
    crosscheck, load_wooldridge, dataset_name, cov_type
):
    y, x = WOOLDRIDGE_DATASETS[dataset_name]
    df = load_wooldridge(dataset_name)
    options = OLSOptions(cov_type=cov_type)
    res = OLS(df, y=y, x=x, options=options).fit()

    ref = crosscheck["wooldridge"][dataset_name][cov_type]["r"]
    _assert_close(res.params, ref["coef"], f"{dataset_name}/{cov_type}/R coef")
    _assert_close(res.std_errors, ref["se"], f"{dataset_name}/{cov_type}/R se")


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize("dataset_name", list(WOOLDRIDGE_DATASETS))
def test_wooldridge_matches_pyfixest(
    crosscheck, load_wooldridge, dataset_name, cov_type
):
    y, x = WOOLDRIDGE_DATASETS[dataset_name]
    df = load_wooldridge(dataset_name)
    options = OLSOptions(cov_type=cov_type)
    res = OLS(df, y=y, x=x, options=options).fit()

    ref = crosscheck["wooldridge"][dataset_name][cov_type]["pyfixest"]
    _assert_close(
        res.params, ref["coef"], f"{dataset_name}/{cov_type}/pyfixest coef"
    )
    _assert_close(
        res.std_errors,
        ref["se"],
        f"{dataset_name}/{cov_type}/pyfixest se",
        rtol=RTOL_PYFIXEST,
    )
