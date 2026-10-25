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
ため`RTOL_STRICT`で厳密比較する。HACも同じ（R側は`NeweyWest(adjust = FALSE)`。
既定の`adjust = TRUE`は共分散全体に`n/(n-k)`を掛けるが、本実装・statsmodelsは
掛けず、これだけが差の全てのため、スカラー補正の有無だけを揃えている。
`benchmark/linear/references/run_lm_crosscheck.R`参照）。詳細は
`docs/spec/ols-spec.md`「テスト」参照。

係数・標準誤差に加え、AIC・BIC・対数尤度・F統計量・F検定p値も検証する
（`testing-policy.md`「リファレンス実装」章の方針。全統計量を独立実装でも
クロスチェックする）。F統計量・F検定p値は本実装の`wald_f_test`と同じ
ロバストWald検定（cov_typeごとの共分散行列を使う）で、HACでも機械精度で一致する。

`predict()`（`docs/spec/ols-spec.md`「predict()」）も対象に含める。
`run_lm_predict_crosscheck.R`（`fitted()`・`predict(model, newdata=...)`）を使い、
全シナリオで`predict(new_data=None)`（学習データの予測値）を、baselineシナリオのみ
`predict(new_data)`（新規データの予測値、列順を入れ替えて列名マッチングも確認）を
crosscheckする。

`include_intercept=False`（切片なし）・`confidence_level`非既定は、baseline
シナリオのみで全cov_type（classical/HC0-3/cluster/HAC）と組み合わせてRとも
数値照合する（他のオプションと同じ凍結フィクスチャでの厳密比較）。
HACのラグは、R側の`NeweyWest(lag = )`に本実装の自動選択式（`hac_auto_lag(n)`）の値を
`hac_lag`として渡し、テストはその値を`hac_lags`に明示して照合する（statsmodelsは
`hac_lags`を省略した自動選択で照合する）。自動選択とは別に明示するラグ
（`HAC_LAGS_EXPLICIT`）はbaselineシナリオで確認する。時系列の実データ（Wooldridge
`prminwge`）でもHACを照合する。

Note:
    合成データはフィクスチャ生成時と同じ入力データを、`tests/fixtures/
    benchmarks/data/`に固定済みのCSV（`benchmark/linear/freeze.py`参照）から読む。
    `imbalanced_cluster_groups`（純粋にnから決定論的にラベルを組み立てるだけで
    乱数を使わない）のみ、引き続き`benchmark/linear/datasets.py`を直接呼ぶ。
    Wooldridgeデータは`load_wooldridge.py`経由で都度ロードする（データの再配布
    ライセンスが未確認のためCSVとして固定しない。`benchmark/linear/freeze.py`のdocstring
    参照）。`wooldridge`パッケージ（test依存グループ）が無い環境では、
    Wooldridgeクロスチェックのみ`pytest.importorskip`でskipする。
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close, assert_fields_close
from _constants import DATA_DIR, MROZ_X
from _helpers import (
    ROW_TIME,
    with_cluster_groups,
    with_row_time,
    wooldridge_loader,
)
from _tolerances import TOLERANCES
from econometricsmodels import OLS, OLSOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.linear.constants import (
    CONFIDENCE_LEVEL_NON_DEFAULT,
    HAC_LAGS_EXPLICIT,
    NON_HAC_COV_TYPES,
    PREDICT_NEW_DATA,
)
from benchmark.linear.constants import (
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


# 絶対誤差フロア（ref値が0近傍のとき相対誤差比較が意味を持たなくなるのを防ぐ）。
ATOL = TOLERANCES["ols_crosscheck"]["atol"]


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_dict_close = partial(assert_dict_close, rtol=RTOL_STRICT, atol=ATOL)
_assert_close = partial(assert_close, rtol=RTOL_STRICT, atol=ATOL)
_assert_fields_close = partial(
    assert_fields_close, rtol=RTOL_STRICT, atol=ATOL
)


def _assert_fit_stats_close(res, ref: dict, label: str, rtol: float) -> None:
    """R²・調整済みR²・AIC・BIC・対数尤度・F統計量・F検定p値・t値・p値・
    信頼区間の検証。

    R²・調整済みR²・AIC/BIC/対数尤度はcov_typeに依存しないため常に
    RTOL_STRICTで比較する。F統計量・F検定p値・t値・p値・信頼区間は
    cov_typeごとの共分散行列（標準誤差）に依存するため呼び出し元の
    rtol（全cov_typeでRTOL_STRICT）を使う。
    """
    _assert_fields_close(
        res,
        ref,
        ("r_squared", "adj_r_squared", "aic", "bic", "log_likelihood"),
        label,
    )
    _assert_fields_close(
        res, ref, ("f_statistic", "f_p_value"), label, rtol=rtol
    )
    _assert_dict_close(
        res.test_stats, ref["test_stats"], f"{label}/test_stats", rtol=rtol
    )
    _assert_dict_close(
        res.p_values,
        ref["p_values"],
        f"{label}/p_values",
        rtol=rtol,
    )
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_close(
            our_lower, ref_lower, f"{label}/conf_lower/{name}", rtol=rtol
        )
        _assert_close(
            our_upper, ref_upper, f"{label}/conf_upper/{name}", rtol=rtol
        )


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("cov_type", NON_HAC_COV_TYPES)
@pytest.mark.parametrize("scenario", SYNTHETIC_SCENARIOS)
def test_matches_r(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    options = OLSOptions(cov_type=cov_type)
    res = OLS(with_row_time(df), y="y", x=x_cols, options=options).fit()

    ref = crosscheck["synthetic"][scenario][cov_type]
    label = f"{scenario}/{cov_type}/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


@pytest.mark.parametrize("scenario", SYNTHETIC_SCENARIOS)
def test_predict_none_matches_r_fitted_values(crosscheck, scenario):
    """`predict(new_data=None)`（学習データに対する予測値）がRの`fitted()`と一致すること。"""
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    res = OLS(with_row_time(df), y="y", x=x_cols).fit()

    predicted = res.predict()
    ref = crosscheck["synthetic"][scenario]["predict"]["fitted"]

    assert len(predicted) == len(ref)
    for i, (our_val, ref_val) in enumerate(zip(predicted, ref)):
        _assert_close(our_val, ref_val, f"{scenario}/predict(None)/R row {i}")


def test_predict_new_data_matches_r(crosscheck):
    """`predict(new_data)`（新規データに対する予測値）がRの`predict(model, newdata=...)`
    と一致すること（baselineシナリオのみ。列順を学習時と入れ替えて渡し、列名マッチングも
    合わせて確認する）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    res = OLS(with_row_time(df), y="y", x=["x1", "x2", "x3"]).fit()

    new_data = pl.DataFrame(
        {
            "x3": PREDICT_NEW_DATA["x3"],
            "x1": PREDICT_NEW_DATA["x1"],
            "x2": PREDICT_NEW_DATA["x2"],
        }
    )
    predicted = res.predict(new_data=new_data)
    ref = crosscheck["synthetic"]["baseline"]["predict"]["predicted"]

    assert len(predicted) == len(ref)
    for i, (our_val, ref_val) in enumerate(zip(predicted, ref)):
        _assert_close(
            our_val, ref_val, f"baseline/predict(new_data)/R row {i}"
        )


def test_cluster_matches_r(crosscheck):
    """クラスターロバストSE。generate_ols_crosscheck_fixtures.pyと同じ疑似
    グループ（行番号%10）を再現する。統計的な意味はなく、実装の動作確認用。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = OLSOptions(cov_type="cluster", cluster="cluster_group")
    res = OLS(
        with_row_time(df), y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["params"], "cluster/R params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], "cluster/R std_errors"
    )
    _assert_fit_stats_close(res, ref, "cluster/R", rtol=RTOL_STRICT)


def test_cluster_imbalanced_matches_r(crosscheck):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = OLSOptions(cov_type="cluster", cluster="cluster_group")
    res = OLS(
        with_row_time(df), y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster_imbalanced"]
    _assert_dict_close(
        res.params, ref["params"], "cluster_imbalanced/R params"
    )
    _assert_dict_close(
        res.std_errors, ref["std_errors"], "cluster_imbalanced/R std_errors"
    )
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
    options = OLSOptions(cov_type="cluster", cluster="cluster_group")
    res = OLS(with_row_time(df), y="y", x=["x1"], options=options).fit()

    ref = crosscheck["synthetic"]["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["params"], "cluster_g2/R params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], "cluster_g2/R std_errors"
    )
    _assert_fit_stats_close(res, ref, "cluster_g2/R", rtol=RTOL_STRICT)


@pytest.mark.parametrize(
    "scenario", ["high_condition_number", "moderate_multicollinearity"]
)
def test_cluster_ill_conditioned_matches_r(crosscheck, scenario):
    """悪条件・多重共線性シナリオとクラスターロバストSEの組み合わせ。

    他のcov_typeと同様に、悪条件・多重共線性との組み合わせをクラスターでも
    確認する。均等な疑似グループ（行番号%10）のみ確認する（グルーピング
    パターン自体の網羅性は`test_cluster_matches_r`等`baseline`シナリオで確認）。
    """
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    df = with_cluster_groups(df, 10)
    options = OLSOptions(cov_type="cluster", cluster="cluster_group")
    res = OLS(
        with_row_time(df), y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = crosscheck["synthetic"][scenario]["cluster"]
    label = f"{scenario}/cluster/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


@pytest.mark.parametrize("scenario", SYNTHETIC_SCENARIOS)
def test_hac_matches_r(crosscheck, scenario):
    """HAC標準誤差。フィクスチャ生成時に本実装の自動ラグ式で計算した
    ラグ（`hac_lag`）をそのまま使い、ラグ選択方式自体の違いを比較対象から
    除外した上でNewey-West公式自体の妥当性を確認する。全合成シナリオ
    （小標本・高次元・悪条件を含む）で他のcov_typeと同じ機械精度で一致する
    （R側が`adjust = FALSE`、`run_lm_crosscheck.R`参照）。
    """
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    ref = crosscheck["synthetic"][scenario]["hac"]
    options = OLSOptions(
        cov_type="hac", hac_lags=ref["hac_lag"], hac_time=ROW_TIME
    )
    res = OLS(with_row_time(df), y="y", x=x_cols, options=options).fit()

    label = f"{scenario}/hac/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


# ── 凍結フィクスチャとの数値照合（オプション） ────────────────────────


#
# include_intercept=False・confidence_level非既定（凍結フィクスチャ、baselineシナリオ）。
# クラスターも含める。HACは`hac_lag`を使う専用テスト（`test_*_hac_matches_r`）で、
# 同じ`RTOL_STRICT`で確認する。


NO_INTERCEPT_AND_CONFIDENCE_LEVEL_STRICT_COV_TYPES = [
    *NON_HAC_COV_TYPES,
    "cluster",
]


def _options_kwargs_for_cov_type(df: pl.DataFrame, cov_type: str):
    """`cov_type="cluster"`のときのみ疑似グループ列を付け、
    `cluster`を返す。それ以外は`cluster=None`。
    """
    if cov_type == "cluster":
        return with_cluster_groups(df, 10), {"cluster": "cluster_group"}
    return df, {"cluster": None}


@pytest.mark.parametrize(
    "cov_type", NO_INTERCEPT_AND_CONFIDENCE_LEVEL_STRICT_COV_TYPES
)
def test_no_intercept_matches_r(crosscheck, cov_type):
    """`include_intercept=False`が、cov_typeによらずRとも一致すること
    （baselineシナリオ、classical/HC0-3/cluster）。HACはフィクスチャの`hac_lag`を
    `hac_lags`に渡す必要があるため`test_no_intercept_hac_matches_r`で別途確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df, kwargs = _options_kwargs_for_cov_type(df, cov_type)
    options = OLSOptions(include_intercept=False, cov_type=cov_type, **kwargs)
    res = OLS(
        with_row_time(df), y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = crosscheck["synthetic"]["baseline"]["no_intercept"][cov_type]
    label = f"no_intercept/{cov_type}/R"
    assert res.param_names == ["x1", "x2", "x3"]
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_no_intercept_hac_matches_r(crosscheck):
    """`include_intercept=False`のHAC標準誤差。フィクスチャ生成時の自動ラグ
    （`hac_lag`）をそのまま使う（`test_hac_matches_r`と同じ方針）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    ref = crosscheck["synthetic"]["baseline"]["no_intercept"]["hac"]
    options = OLSOptions(
        include_intercept=False,
        cov_type="hac",
        hac_lags=ref["hac_lag"],
        hac_time=ROW_TIME,
    )
    res = OLS(
        with_row_time(df), y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    label = "no_intercept/hac/R"
    assert res.param_names == ["x1", "x2", "x3"]
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors,
        ref["std_errors"],
        f"{label} std_errors",
        rtol=RTOL_STRICT,
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


@pytest.mark.parametrize(
    "cov_type", NO_INTERCEPT_AND_CONFIDENCE_LEVEL_STRICT_COV_TYPES
)
def test_confidence_level_matches_r(crosscheck, cov_type):
    """confidence_level非既定（`CONFIDENCE_LEVEL_NON_DEFAULT`）が、cov_type
    によらずRとも一致すること（baselineシナリオ、classical/HC0-3/cluster）。
    HACはフィクスチャの`hac_lag`を`hac_lags`に渡す必要があるため
    `test_confidence_level_hac_matches_r`で別途確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df, kwargs = _options_kwargs_for_cov_type(df, cov_type)
    options = OLSOptions(
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        cov_type=cov_type,
        **kwargs,
    )
    res = OLS(
        with_row_time(df), y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    ref = crosscheck["synthetic"]["baseline"]["confidence_level"][cov_type]
    label = f"confidence_level/{cov_type}/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_confidence_level_hac_matches_r(crosscheck):
    """confidence_level非既定のHAC標準誤差。フィクスチャ生成時の自動ラグ
    （`hac_lag`）をそのまま使う（`test_hac_matches_r`と同じ方針）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    ref = crosscheck["synthetic"]["baseline"]["confidence_level"]["hac"]
    options = OLSOptions(
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        cov_type="hac",
        hac_lags=ref["hac_lag"],
        hac_time=ROW_TIME,
    )
    res = OLS(
        with_row_time(df), y="y", x=["x1", "x2", "x3"], options=options
    ).fit()

    label = "confidence_level/hac/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors,
        ref["std_errors"],
        f"{label} std_errors",
        rtol=RTOL_STRICT,
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


@pytest.mark.parametrize("hac_lags", HAC_LAGS_EXPLICIT)
def test_hac_lags_matches_r(crosscheck, hac_lags):
    """自動選択とは別に`hac_lags`を明示指定したHAC標準誤差が、Rの
    `NeweyWest(lag = )`と一致すること（baselineシナリオ）。`hac_lags=0`は
    HC0相当の境界値。ラグはフィクスチャの`hac_lag`ではなくパラメータの値を
    渡し、記録された`hac_lag`と一致することも確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    ref = crosscheck["synthetic"]["baseline"]["hac_lags"][str(hac_lags)]
    assert ref["hac_lag"] == hac_lags
    options = OLSOptions(cov_type="hac", hac_lags=hac_lags, hac_time=ROW_TIME)
    res = OLS(
        with_row_time(df),
        y="y",
        x=["x1", "x2", "x3"],
        options=options,
    ).fit()

    label = f"hac_lags={hac_lags}/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)
    assert res.hac_lags_used == hac_lags, f"{label}/hac_lags_used"


# ── 凍結フィクスチャとの数値照合（実データ） ──────────────────────────


WOOLDRIDGE_DATASETS = {
    "wage1": ("lwage", ["educ", "exper", "tenure"]),
    "gpa2": ("colgpa", ["sat", "hsperc", "tothrs"]),
    "mroz": ("inlf", MROZ_X),
}


@pytest.fixture(scope="module")
def load_wooldridge():
    """`wooldridge_loader`（`tests/support/_helpers.py`）参照。複数データセット名を
    `pytest.mark.parametrize`で振るため、ロード関数自体をfixtureとして返す。
    """
    return wooldridge_loader()


@pytest.mark.parametrize("cov_type", NON_HAC_COV_TYPES)
@pytest.mark.parametrize("dataset_name", list(WOOLDRIDGE_DATASETS))
def test_wooldridge_matches_r(
    crosscheck, load_wooldridge, dataset_name, cov_type
):
    y, x = WOOLDRIDGE_DATASETS[dataset_name]
    df = load_wooldridge(dataset_name)
    options = OLSOptions(cov_type=cov_type)
    res = OLS(with_row_time(df), y=y, x=x, options=options).fit()

    ref = crosscheck["wooldridge"][dataset_name][cov_type]
    label = f"{dataset_name}/{cov_type}/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
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
    options = OLSOptions(cov_type="cluster", cluster="region")
    res = OLS(
        with_row_time(df),
        y="lwage",
        x=["educ", "exper", "tenure"],
        options=options,
    ).fit()

    ref = crosscheck["wooldridge"]["wage1"]["cluster"]
    label = "wage1/cluster(region)/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)


def test_wooldridge_prminwge_hac_matches_r(crosscheck, load_wooldridge):
    """時系列の実データ（Wooldridge `prminwge`、年次昇順でn=38）でのHAC。
    ラグはフィクスチャ生成時の自動選択（`hac_lag`）をそのまま使う
    （`test_hac_matches_r`と同じ方針、statsmodelsとの照合は
    `test_ols_reference.py::test_wooldridge_prminwge_hac_matches_statsmodels`）。
    """
    df = load_wooldridge("prminwge")
    ref = crosscheck["wooldridge"]["prminwge"]["hac"]
    options = OLSOptions(
        cov_type="hac", hac_lags=ref["hac_lag"], hac_time=ROW_TIME
    )
    res = OLS(
        with_row_time(df),
        y="lprepop",
        x=["lmincov", "lusgnp"],
        options=options,
    ).fit()

    label = "prminwge/hac/R"
    _assert_dict_close(res.params, ref["params"], f"{label} params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], f"{label} std_errors"
    )
    _assert_fit_stats_close(res, ref, label, rtol=RTOL_STRICT)
