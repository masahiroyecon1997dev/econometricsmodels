"""OLSの独立実装（R: lm + sandwich/lmtest）とのクロスチェックテスト。

主リファレンス（statsmodels）との厳密比較は`test_ols.py`で行う。ここでは
`tests/api_tests/fixtures/benchmarks/ols_crosscheck.json`
（`benchmark/fixtures/generate_ols_crosscheck_fixtures.py`で生成、Issue #18）
を用いて、statsmodelsとは独立した実装（R）との一致を確認する。

pyfixestは正確性検証には使わない（Issue #27）。fixest（R）本体のソース
（`vcov_hc2_hc3_internal`）を確認したところ、HC2/HC3にはssc（`n/(n-k)`の小標本
補正）を一切適用しない設計だったが、pyfixest（Python、v0.60.0時点）は
HC1/HC2/HC3を同一分岐で扱っておりHC1用の`N/(N-k)`補正をHC2/HC3にも誤って
適用していた（`sqrt(N/(N-k))`がSEに掛かり、nが小さいほど乖離が拡大する。
small_nシナリオ n=20, k=4で約11.8%）。fixestの仕様ではなくpyfixest自身の
実装バグであり、性能比較専用（別issue）とし正確性検証からは除外する。

classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-14程度）
ため`RTOL_STRICT`で厳密比較する。HACのみ小標本補正の慣習差により
`RTOL_HAC`（緩め）を使う。詳細は`docs/planning/specs/ols-implementation-notes.md`
「クロスチェックの役割分担見直し」参照。

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

# classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-14程度、
# Issue #27で測定）。testing-policy.md「許容誤差」の基本方針（相対誤差1e-8）と
# 揃え、statsmodelsと同水準の厳密比較にする。
RTOL_STRICT = 1e-8

# HACのみ小標本補正の慣習差（prewhite/adjust等）により実測で相対誤差0.4%程度の
# 乖離がある（Issue #18で確認、Issue #27で維持を決定）。バグではなくNewey-West
# 実装の慣習差のため、HACのみ緩めの許容誤差を使う。
RTOL_HAC = 1e-2


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
    # 係数（coef）はcov_typeに依存しない通常のOLS推定値のため厳密比較のまま。
    # HACで乖離しうるのは標準誤差（se）のみ。
    _assert_close(res.params, ref["coef"], "hac/R coef")
    _assert_close(res.std_errors, ref["se"], "hac/R se", rtol=RTOL_HAC)


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
