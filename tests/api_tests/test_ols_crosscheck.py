"""OLSの独立実装（R: lm + sandwich/lmtest）とのクロスチェックテスト。

主リファレンス（statsmodels）との厳密比較は`test_ols.py`で行う。ここでは
`tests/api_tests/fixtures/benchmarks/ols_crosscheck.json`
（`benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py`で生成）
を用いて、statsmodelsとは独立した実装（R）との一致を確認する。

pyfixestは正確性検証には使わない。fixest（R）本体のソース
（`vcov_hc2_hc3_internal`）を確認したところ、HC2/HC3にはssc（`n/(n-k)`の小標本
補正）を一切適用しない設計だったが、pyfixest（Python、v0.60.0時点）は
HC1/HC2/HC3を同一分岐で扱っておりHC1用の`N/(N-k)`補正をHC2/HC3にも誤って
適用していた（`sqrt(N/(N-k))`がSEに掛かり、nが小さいほど乖離が拡大する。
small_nシナリオ n=20, k=4で約11.8%）。fixestの仕様ではなくpyfixest自身の
実装バグであり、性能比較専用とし正確性検証からは除外する。

classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-14程度）
ため`RTOL_STRICT`で厳密比較する。HACのみ小標本補正の慣習差により
`RTOL_HAC`（緩め）を使う。詳細は`docs/planning/specs/ols-implementation-notes.md`
「8. テスト」参照。

係数・標準誤差に加え、AIC・BIC・対数尤度・F統計量・F検定p値も検証する
（`testing-policy.md`「リファレンス実装」章の方針。全統計量を独立実装でも
クロスチェックする）。AIC・BIC・対数尤度はcov_typeに依存しないためHACでも
機械精度で一致するが、F統計量・F検定p値は本実装の`wald_f_test`と同じ
ロバストWald検定（cov_typeごとの共分散行列を使う）のため、HACのみ標準誤差と
同じ小標本補正の慣習差が乗る（実測で相対誤差0.8%程度、`RTOL_HAC`の範囲内）。

Note:
    合成データはフィクスチャ生成時と同じ入力データを、`tests/api_tests/fixtures/
    benchmarks/data/`に固定済みのCSV（`benchmark/freeze_datasets.py`参照）から読む。
    `imbalanced_cluster_groups`（純粋にnから決定論的にラベルを組み立てるだけで
    乱数を使わない）のみ、引き続き`generate_synthetic_datasets.py`を直接呼ぶ。
    Wooldridgeデータは`load_wooldridge.py`経由で都度ロードする（データの再配布
    ライセンスが未確認のためCSVとして固定しない。`freeze_datasets.py`のdocstring
    参照）。`wooldridge`パッケージ（benchmark依存グループ）が無い環境では、
    Wooldridgeクロスチェックのみ任意扱いにする。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from generate_synthetic_datasets import imbalanced_cluster_groups  # noqa: E402

from econometricsmodels import OLS, OLSOptions  # noqa: E402

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "benchmarks"
    / "ols_crosscheck.json"
)
DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"

# classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-14程度）。
# testing-policy.md「許容誤差」の基本方針（相対誤差1e-8）と揃え、statsmodelsと
# 同水準の厳密比較にする。
RTOL_STRICT = 1e-8

# HACのみ小標本補正の慣習差（prewhite/adjust等）により実測で相対誤差0.4%程度の
# 乖離がある。バグではなくNewey-West実装の慣習差のため、HACのみ緩めの許容誤差を使う。
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


def _assert_scalar_close(
    our_val: float, ref_val: float, label: str, rtol: float = RTOL_STRICT
) -> None:
    diff = abs(our_val - ref_val)
    tol = rtol * max(abs(ref_val), 1e-8)
    assert diff <= tol, (
        f"[{label}] ours={our_val:.6f}, reference={ref_val:.6f}, "
        f"diff={diff:.6f} > tol={tol:.6f}"
    )


def _assert_fit_stats_close(res, ref: dict, label: str, rtol: float) -> None:
    """AIC・BIC・対数尤度・F統計量・F検定p値の検証。

    AIC/BIC/対数尤度はcov_typeに依存しないため常にRTOL_STRICTで比較する。
    F統計量・F検定p値はcov_typeごとのロバストWald検定のため呼び出し元の
    rtol（HACのみRTOL_HAC）を使う。

    `ref["f_statistic"]`が`None`の場合はF統計量比較をスキップする
    （scale_varianceシナリオ、Issue #101/#107）。傾き係数の同時共分散
    部分行列がスケール比の2乗相当の条件数を持ち倍精度の限界を超えるため、
    R側の`solve()`が"computationally singular"として計算そのものを拒否する
    （`run_lm_crosscheck_benchmark.R`参照）。係数・SE・AIC・BIC・対数尤度は
    影響を受けないため引き続き比較する。
    """
    _assert_scalar_close(res.aic, ref["aic"], f"{label}/aic")
    _assert_scalar_close(res.bic, ref["bic"], f"{label}/bic")
    _assert_scalar_close(
        res.log_likelihood, ref["log_likelihood"], f"{label}/log_likelihood"
    )
    if ref["f_statistic"] is not None:
        _assert_scalar_close(
            res.f_statistic,
            ref["f_statistic"],
            f"{label}/f_statistic",
            rtol=rtol,
        )
        _assert_scalar_close(
            res.f_p_value, ref["f_p_value"], f"{label}/f_p_value", rtol=rtol
        )


SYNTHETIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "scale_variance",
    "high_condition_number",
    # n=k+1（自由度1ちょうど）の成功パス（Issue #101）。
    "baseline_df1",
]
NON_HAC_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3"]


@pytest.mark.parametrize("cov_type", NON_HAC_COV_TYPES)
@pytest.mark.parametrize("scenario", SYNTHETIC_SCENARIOS)
def test_synthetic_matches_r(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    options = OLSOptions(cov_type=cov_type)
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"][scenario][cov_type]["r"]
    label = f"{scenario}/{cov_type}/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_cluster_matches_r(crosscheck):
    """クラスターロバストSE。generate_ols_crosscheck_fixtures.pyと同じ疑似
    グループ（行番号%10）を再現する。統計的な意味はなく、実装の動作確認用。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
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
    _assert_fit_stats_close(res, ref, "cluster/R", rtol=RTOL_STRICT)


def test_cluster_imbalanced_matches_r(crosscheck):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル、Issue #100）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster_imbalanced"]["r"]
    _assert_close(res.params, ref["coef"], "cluster_imbalanced/R coef")
    _assert_close(res.std_errors, ref["se"], "cluster_imbalanced/R se")
    _assert_fit_stats_close(res, ref, "cluster_imbalanced/R", rtol=RTOL_STRICT)


def test_cluster_g2_matches_r(crosscheck):
    """クラスタ数境界（G=2ちょうど）の成功パス。

    説明変数1個（q=1）に絞っている。baseline既定の3個のままG=2にすると、
    ロバストWald検定の共分散部分行列（3x3）のランクがG=2以下となり必然的に
    特異になりComputationErrorになる（成功パスにならない。
    `test_ols_fixtures.py::test_cluster_g2_with_multiple_slopes_raises_computation_error`
    参照。Issue #100の実装中に判明した境界条件）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 2).alias("cluster_group"))
        .drop("_row")
    )
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1"], options=options).fit()

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
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = entry["r"]
    # 係数（coef）・AIC・BIC・対数尤度はcov_typeに依存しない通常のOLS推定値の
    # ため厳密比較のまま。HACで乖離しうるのは標準誤差（se）とF統計量・F検定p値
    # （cov_typeごとのロバストWald検定のため、標準誤差と同じ慣習差が乗る）。
    _assert_close(res.params, ref["coef"], "hac/R coef")
    _assert_close(res.std_errors, ref["se"], "hac/R se", rtol=RTOL_HAC)
    _assert_fit_stats_close(res, ref, "hac/R", rtol=RTOL_HAC)


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
    Wooldridgeデータはデータの再配布ライセンスが未確認のためCSVとして固定
    せず（`benchmark/freeze_datasets.py`のdocstring参照）、都度ロードする。
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
    label = f"{dataset_name}/{cov_type}/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_wooldridge_wage1_region_cluster_matches_r(
    crosscheck, load_wooldridge
):
    """wage1の実カテゴリ列（northcen/south/westダミーから合成したregion、
    基準カテゴリnortheast、4グループ・不均衡サイズ）でのクラスターロバストSE
    （Issue #100「実データでのグループ列」）。疑似グループ（行番号%N）ではなく
    実データに由来するグループ構造での検証。
    """
    df = load_wooldridge("wage1")
    region = (
        pl.when(pl.col("northcen") == 1)
        .then(pl.lit("northcen"))
        .when(pl.col("south") == 1)
        .then(pl.lit("south"))
        .when(pl.col("west") == 1)
        .then(pl.lit("west"))
        .otherwise(pl.lit("northeast"))
        .alias("region")
    )
    df = df.with_columns(region)
    options = OLSOptions(cov_type="cluster", cluster_col="region")
    res = OLS(
        df, y="lwage", x=["educ", "exper", "tenure"], options=options
    ).fit()

    ref = crosscheck["wooldridge"]["wage1"]["cluster"]["r"]
    label = "wage1/cluster(region)/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)
