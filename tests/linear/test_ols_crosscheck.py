"""OLSの独立実装（R: lm + sandwich/lmtest）とのクロスチェックテスト。

主リファレンス（statsmodels）との厳密比較は`test_ols_reference.py`で行う。ここでは
`tests/fixtures/benchmarks/ols_crosscheck.json`
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
`RTOL_HAC`（緩め）を使う。詳細は`docs/spec/ols-spec.md`「テスト」参照。

係数・標準誤差に加え、AIC・BIC・対数尤度・F統計量・F検定p値も検証する
（`testing-policy.md`「リファレンス実装」章の方針。全統計量を独立実装でも
クロスチェックする）。AIC・BIC・対数尤度はcov_typeに依存しないためHACでも
機械精度で一致するが、F統計量・F検定p値は本実装の`wald_f_test`と同じ
ロバストWald検定（cov_typeごとの共分散行列を使う）のため、HACのみ標準誤差と
同じ小標本補正の慣習差が乗る（実測で相対誤差0.8%程度、`RTOL_HAC`の範囲内）。

`predict()`（`docs/spec/ols-spec.md`「predict()」）も対象に含める。
`run_lm_predict_crosscheck.R`（`fitted()`・`predict(model, newdata=...)`）を使い、
全シナリオで`predict(new_data=None)`（学習データの予測値）を、baselineシナリオのみ
`predict(new_data)`（新規データの予測値、列順を入れ替えて列名マッチングも確認）を
crosscheckする。

`include_intercept=False`（切片なし）・`confidence_level`非既定は、baseline
シナリオのみで全cov_type（classical/HC0-3/cluster/HAC）と組み合わせてRとも
数値照合する（従来ライブstatsmodels比較・相対比較のみだったものを、他の
オプションと同じ凍結フィクスチャでの厳密比較に統合）。

Note:
    合成データはフィクスチャ生成時と同じ入力データを、`tests/fixtures/
    benchmarks/data/`に固定済みのCSV（`benchmark/linear/freeze.py`参照）から読む。
    `imbalanced_cluster_groups`（純粋にnから決定論的にラベルを組み立てるだけで
    乱数を使わない）のみ、引き続き`benchmark/linear/datasets.py`を直接呼ぶ。
    Wooldridgeデータは`load_wooldridge.py`経由で都度ロードする（データの再配布
    ライセンスが未確認のためCSVとして固定しない。`benchmark/linear/freeze.py`のdocstring
    参照）。`wooldridge`パッケージ（benchmark依存グループ）が無い環境では、
    Wooldridgeクロスチェックのみ任意扱いにする。
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _constants import DATA_DIR
from _helpers import with_cluster_groups, wooldridge_loader
from _tolerances import TOLERANCES
from econometricsmodels import OLS, OLSOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.linear.constants import PREDICT_NEW_DATA
from benchmark.linear.fixtures.generate_ols_crosscheck_fixtures import (
    CONFIDENCE_LEVEL_NON_DEFAULT,
)
from benchmark.linear.fixtures.generate_ols_crosscheck_fixtures import (
    NUMERIC_SCENARIOS as SYNTHETIC_SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "ols_crosscheck.json"
)

# classical/HC0-3/clusterはRとほぼ機械精度で一致する（実測で相対誤差1e-14程度）。
# testing-policy.md「許容誤差」の基本方針（相対誤差1e-8）と揃え、statsmodelsと
# 同水準の厳密比較にする。
RTOL_STRICT = TOLERANCES["ols_crosscheck"]["rtol_strict"]

# HACのみ小標本補正の慣習差（prewhite/adjust等）により実測で相対誤差0.4%程度の
# 乖離がある。バグではなくNewey-West実装の慣習差のため、HACのみ緩めの許容誤差を使う。
RTOL_HAC = TOLERANCES["ols_crosscheck"]["rtol_hac"]

# 絶対誤差フロア（ref値が0近傍のとき相対誤差比較が意味を持たなくなるのを防ぐ、
# フェーズ3.5でtest_wls_crosscheck.pyと同じ計算式に修正）。
ATOL = TOLERANCES["ols_crosscheck"]["atol"]

# p_values専用の絶対誤差フロア（HAC/autocorrelatedで裾確率がゼロ近傍に
# 潰れるため、_tolerances.py参照）。
ATOL_P_VALUE = TOLERANCES["ols_crosscheck"]["atol_p_value"]


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


# _assert_close（dict版）はtests/_assertions.pyのassert_dict_closeに、
# _assert_scalar_close（scalar版）はassert_closeに対応する
# （フェーズ3.5で計算式のバグを修正した上で統合）。
_assert_close = partial(assert_dict_close, rtol=RTOL_STRICT, atol=ATOL)
_assert_scalar_close = partial(assert_close, rtol=RTOL_STRICT, atol=ATOL)


def _assert_fit_stats_close(res, ref: dict, label: str, rtol: float) -> None:
    """R²・調整済みR²・AIC・BIC・対数尤度・F統計量・F検定p値・t値・p値・
    信頼区間の検証。

    R²・調整済みR²・AIC/BIC/対数尤度はcov_typeに依存しないため常に
    RTOL_STRICTで比較する。F統計量・F検定p値・t値・p値・信頼区間は
    cov_typeごとの共分散行列（標準誤差）に依存するため呼び出し元の
    rtol（HACのみRTOL_HAC）を使う。
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
    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    options = OLSOptions(cov_type=cov_type)
    res = OLS(df, y="y", x=x_cols, options=options).fit()

    ref = crosscheck["synthetic"][scenario][cov_type]["r"]
    label = f"{scenario}/{cov_type}/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


@pytest.mark.parametrize("scenario", SYNTHETIC_SCENARIOS)
def test_predict_none_matches_r_fitted_values(crosscheck, scenario):
    """`predict(new_data=None)`（学習データに対する予測値）がRの`fitted()`と一致すること。"""
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    res = OLS(df, y="y", x=x_cols).fit()

    predicted = [row["predicted"] for row in res.predict()]
    ref = crosscheck["synthetic"][scenario]["predict"]["fitted"]

    assert len(predicted) == len(ref)
    for i, (our_val, ref_val) in enumerate(zip(predicted, ref)):
        _assert_scalar_close(
            our_val, ref_val, f"{scenario}/predict(None)/R row {i}"
        )


def test_predict_new_data_matches_r(crosscheck):
    """`predict(new_data)`（新規データに対する予測値）がRの`predict(model, newdata=...)`
    と一致すること（baselineシナリオのみ。列順を学習時と入れ替えて渡し、列名マッチングも
    合わせて確認する）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    res = OLS(df, y="y", x=["x1", "x2", "x3"]).fit()

    new_data = pl.DataFrame(
        {
            "x3": PREDICT_NEW_DATA["x3"],
            "x1": PREDICT_NEW_DATA["x1"],
            "x2": PREDICT_NEW_DATA["x2"],
        }
    )
    predicted = [row["predicted"] for row in res.predict(new_data)]
    ref = crosscheck["synthetic"]["baseline"]["predict"]["predicted"]

    assert len(predicted) == len(ref)
    for i, (our_val, ref_val) in enumerate(zip(predicted, ref)):
        _assert_scalar_close(
            our_val, ref_val, f"baseline/predict(new_data)/R row {i}"
        )


def test_cluster_matches_r(crosscheck):
    """クラスターロバストSE。generate_ols_crosscheck_fixtures.pyと同じ疑似
    グループ（行番号%10）を再現する。統計的な意味はなく、実装の動作確認用。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster"]["r"]
    _assert_close(res.params, ref["coef"], "cluster/R coef")
    _assert_close(res.std_errors, ref["se"], "cluster/R se")
    _assert_fit_stats_close(res, ref, "cluster/R", rtol=RTOL_STRICT)


def test_cluster_imbalanced_matches_r(crosscheck):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。

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
    """クラスタ数境界（G=2、q=1でG>q）の成功パス。

    説明変数1個（q=1）に絞っている。baseline既定の3個（q=3）のままG=2にすると、
    `rank(Ŝ)≤G-1`のためロバストWald検定のq×q部分行列が構造的に特異になり、
    `fit()`冒頭のバリデーションが`ValidationError`で弾く（成功パスにならない。
    `test_ols_validation.py::test_cluster_count_at_most_slopes_raises_validation_error`
    参照）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv")
    df = with_cluster_groups(df, 2)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1"], options=options).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster_g2"]["r"]
    _assert_close(res.params, ref["coef"], "cluster_g2/R coef")
    _assert_close(res.std_errors, ref["se"], "cluster_g2/R se")
    _assert_fit_stats_close(res, ref, "cluster_g2/R", rtol=RTOL_STRICT)


@pytest.mark.parametrize(
    "scenario", ["high_condition_number", "moderate_multicollinearity"]
)
def test_cluster_ill_conditioned_matches_r(crosscheck, scenario):
    """悪条件・多重共線性シナリオとクラスターロバストSEの組み合わせ。

    クラスターは従来`baseline`シナリオのみで、他のcov_typeでは全シナリオ検証
    済みの悪条件・多重共線性との組み合わせが未検証だった。均等な疑似グループ
    （行番号%10）のみ確認する（グルーピングパターン自体の網羅性は
    `test_cluster_matches_r`等`baseline`シナリオで確認済み）。
    """
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    df = with_cluster_groups(df, 10)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"][scenario]["cluster"]["r"]
    label = f"{scenario}/cluster/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


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


# ── include_intercept=False・confidence_level非既定（凍結フィクスチャ、
# baselineシナリオ） ─────────────────────────────────────────────
# クラスターも含めるがHACのみ小標本補正の慣習差があるため専用テストで
# RTOL_HACを使う（`test_hac_matches_r`と同じ方針）。
NO_INTERCEPT_AND_CONFIDENCE_LEVEL_STRICT_COV_TYPES = [
    *NON_HAC_COV_TYPES,
    "cluster",
]


def _options_kwargs_for_cov_type(df: pl.DataFrame, cov_type: str):
    """`cov_type="cluster"`のときのみ疑似グループ列を付け、
    `cluster_col`を返す。それ以外は`cluster_col=None`。
    """
    if cov_type == "cluster":
        return with_cluster_groups(df, 10), {"cluster_col": "cluster_group"}
    return df, {"cluster_col": None}


@pytest.mark.parametrize(
    "cov_type", NO_INTERCEPT_AND_CONFIDENCE_LEVEL_STRICT_COV_TYPES
)
def test_no_intercept_matches_r(crosscheck, cov_type):
    """`include_intercept=False`が、cov_typeによらずRとも一致すること
    （baselineシナリオ、classical/HC0-3/cluster）。HACのみ小標本補正の慣習差
    があるため`test_no_intercept_hac_matches_r`で別途確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df, kwargs = _options_kwargs_for_cov_type(df, cov_type)
    options = OLSOptions(include_intercept=False, cov_type=cov_type, **kwargs)
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"]["baseline"]["no_intercept"][cov_type]["r"]
    label = f"no_intercept/{cov_type}/R"
    assert res.param_names == ["x1", "x2", "x3"]
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_no_intercept_hac_matches_r(crosscheck):
    """`include_intercept=False`のHAC標準誤差。フィクスチャ生成時の自動ラグ
    （`hac_lag`）をそのまま使う（`test_hac_matches_r`と同じ方針）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    entry = crosscheck["synthetic"]["baseline"]["no_intercept"]["hac"]
    options = OLSOptions(
        include_intercept=False, cov_type="hac", hac_lags=entry["hac_lag"]
    )
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = entry["r"]
    label = "no_intercept/hac/R"
    assert res.param_names == ["x1", "x2", "x3"]
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se", rtol=RTOL_HAC)
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_HAC)


@pytest.mark.parametrize(
    "cov_type", NO_INTERCEPT_AND_CONFIDENCE_LEVEL_STRICT_COV_TYPES
)
def test_confidence_level_matches_r(crosscheck, cov_type):
    """confidence_level非既定（`CONFIDENCE_LEVEL_NON_DEFAULT`）が、cov_type
    によらずRとも一致すること（baselineシナリオ、classical/HC0-3/cluster）。
    HACのみ`test_confidence_level_hac_matches_r`で別途確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df, kwargs = _options_kwargs_for_cov_type(df, cov_type)
    options = OLSOptions(
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        cov_type=cov_type,
        **kwargs,
    )
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = crosscheck["synthetic"]["baseline"]["confidence_level"][cov_type][
        "r"
    ]
    label = f"confidence_level/{cov_type}/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se")
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_confidence_level_hac_matches_r(crosscheck):
    """confidence_level非既定のHAC標準誤差。フィクスチャ生成時の自動ラグ
    （`hac_lag`）をそのまま使う（`test_hac_matches_r`と同じ方針）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    entry = crosscheck["synthetic"]["baseline"]["confidence_level"]["hac"]
    options = OLSOptions(
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        cov_type="hac",
        hac_lags=entry["hac_lag"],
    )
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = entry["r"]
    label = "confidence_level/hac/R"
    _assert_close(res.params, ref["coef"], f"{label} coef")
    _assert_close(res.std_errors, ref["se"], f"{label} se", rtol=RTOL_HAC)
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_HAC)


WOOLDRIDGE_DATASETS = {
    "wage1": ("lwage", ["educ", "exper", "tenure"]),
    "gpa2": ("colgpa", ["sat", "hsperc", "tothrs"]),
}


@pytest.fixture(scope="module")
def load_wooldridge():
    """`wooldridge_loader`（`tests/_helpers.py`）参照。複数データセット名を
    `pytest.mark.parametrize`で振るため、ロード関数自体をfixtureとして返す。
    """
    return wooldridge_loader()


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
    （「実データでのグループ列」）。疑似グループ（行番号%N）ではなく
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
