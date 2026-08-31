"""OLS の主リファレンス（statsmodels）との数値照合。

2種類の照合を行う。

1. **凍結フィクスチャとの厳密比較**: `tests/fixtures/benchmarks/ols.json`
   （`benchmark/linear/fixtures/generate_ols_fixtures.py` で生成）を読み込み、
   6つの合成データシナリオ×classical/HC0-3/HAC + クラスター(baselineのみ)で、
   係数・標準誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する
   （`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。
2. **ライブ statsmodels との照合**: 共有 `dataset` フィクスチャ（n=100）で
   毎回 statsmodels を実行し、係数・標準誤差・R²・F統計量・`include_intercept`
   の扱いが一致することを確認する（凍結フィクスチャが対象にしない
   `include_intercept=False` 等の分岐と、statsmodels の挙動変化そのものの検知）。

役割分担:
    - 構造・API・`predict()`: `test_ols_api.py`
    - `ValidationError`/`ComputationError` パス: `test_ols_validation.py`
    - 主リファレンス（statsmodels）との数値照合: このファイル
    - 独立実装（R）とのクロスチェック: `test_ols_crosscheck.py`

Note:
    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/linear/freeze.py`参照）から読む。ジェネレータ
    （`benchmark/linear/datasets.py`）を直接呼ばないことで、ジェネレータ側の
    コードが将来変わっても既存フィクスチャの期待値と無言で不整合にならない。
    `imbalanced_cluster_groups`（純粋にnから決定論的にラベルを組み立てるだけで
    乱数を使わない）のみ、引き続き`benchmark/linear/datasets.py`を直接呼ぶ。
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from _assertions import assert_close, assert_dict_close
from _assertions import rename_intercept as _rename
from _helpers import DATA_DIR, with_cluster_groups
from _ols_helpers import (
    our_fit,
    our_fit_cluster,
    sm_fit,
    sm_fit_cluster,
)
from _tolerances import TOLERANCES
from econometricsmodels import OLS, OLSOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.linear.fixtures.generate_ols_fixtures import (
    COV_TYPES,
)
from benchmark.linear.fixtures.generate_ols_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "ols.json"
)

RTOL = TOLERANCES["ols_reference"]["rtol"]
ATOL = TOLERANCES["ols_reference"]["atol"]

# SCENARIOS/COV_TYPESはgenerate_ols_fixtures.pyのNUMERIC_SCENARIOS/COV_TYPESと
# 常に一致させる必要があるため、そちらをimportして単一の定義元にする。

# generate_ols_fixtures.py（benchmark/linear/references/statsmodels_ref.py）はHACのラグを
# maxlags=1に固定している。同じラグを明示的に指定し、自動ラグ選択式の
# 違いを比較対象から除外する。
HAC_LAG_IN_FIXTURE = 1


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)


def _check_result(res, ref: dict, label: str) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(res.t_stats, ref["t_stats"], f"{label}/t_stats")
    _assert_dict_close(res.p_values, ref["p_values"], f"{label}/p_values")

    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_name = _rename(name)
        our_lower, our_upper = res.conf_int[our_name]
        _assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        _assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    _assert_close(res.r_squared, ref["r_squared"], f"{label}/r_squared")
    _assert_close(
        res.r_squared_adj, ref["r_squared_adj"], f"{label}/r_squared_adj"
    )
    _assert_close(res.f_statistic, ref["f_statistic"], f"{label}/f_statistic")
    _assert_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")
    _assert_close(res.aic, ref["aic"], f"{label}/aic")
    _assert_close(res.bic, ref["bic"], f"{label}/bic")
    _assert_close(
        res.log_likelihood, ref["log_likelihood"], f"{label}/log_likelihood"
    )
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"


# ── 凍結フィクスチャとの数値照合 ───────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_statsmodels(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    kwargs = {"hac_lags": HAC_LAG_IN_FIXTURE} if cov_type == "hac" else {}
    options = OLSOptions(cov_type=cov_type, **kwargs)
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


def test_cluster_matches_statsmodels(fixtures):
    """クラスターロバストSE。`generate_ols_fixtures.py`と同じ疑似グループ
    （行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ（`coef`/`se`のみが記録されている）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_statsmodels(fixtures):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster_imbalanced"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


def test_cluster_g2_matches_statsmodels(fixtures):
    """クラスタ数境界（G=2ちょうど）の成功パス。

    説明変数1個（q=1）に絞っている。baseline既定の3個のままG=2にすると、
    ロバストWald検定の共分散部分行列（3x3）のランクがG=2以下となり必然的に
    特異になりComputationErrorになる（成功パスにならない。
    `test_ols_validation.py::test_cluster_g2_with_multiple_slopes_raises_`
    `computation_error`参照。実装中に判明した境界条件）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv")
    df = with_cluster_groups(df, 2)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1"], options=options).fit()

    ref = fixtures["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


# ── ライブ statsmodels との照合（共有 dataset フィクスチャ） ────────


@pytest.mark.parametrize("cov_type", ["classical", "hc0", "hc1", "hc2", "hc3"])
def test_params_match_statsmodels(dataset, cov_type):
    """回帰係数がstatsmodelsと一致すること（cov_typeによらず係数は同じ）。"""
    sm_res = sm_fit(dataset, cov_type)
    our_res = our_fit(dataset, cov_type)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.params):
        _assert_close(
            our_res.params[name], sm_val, f"[{cov_type}] params/{name}"
        )


@pytest.mark.parametrize("cov_type", ["classical", "hc0", "hc1", "hc2", "hc3"])
def test_std_errors_match_statsmodels(dataset, cov_type):
    """標準誤差がstatsmodelsと一致すること。"""
    sm_res = sm_fit(dataset, cov_type)
    our_res = our_fit(dataset, cov_type)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.bse):
        _assert_close(
            our_res.std_errors[name], sm_val, f"[{cov_type}] se/{name}"
        )


def test_cluster_se_match_statsmodels(dataset):
    """クラスター標準誤差がstatsmodelsと一致すること。"""
    sm_res = sm_fit_cluster(dataset)
    our_res = our_fit_cluster(dataset)

    for name, sm_val in zip(["const", "x1", "x2"], sm_res.bse):
        _assert_close(our_res.std_errors[name], sm_val, f"cluster_se/{name}")


def test_r_squared_match_statsmodels(dataset):
    """R²と調整済みR²がstatsmodelsと一致すること。"""
    sm_res = sm_fit(dataset)
    our_res = our_fit(dataset)

    _assert_close(our_res.r_squared, sm_res.rsquared, "r_squared")
    _assert_close(our_res.r_squared_adj, sm_res.rsquared_adj, "r_squared_adj")


def test_f_statistic_match_statsmodels(dataset):
    """F統計量がstatsmodelsと一致すること。"""
    sm_res = sm_fit(dataset)
    our_res = our_fit(dataset)

    _assert_close(our_res.f_statistic, sm_res.fvalue, "f_statistic")


def test_include_intercept_false_matches_statsmodels():
    """`include_intercept=False`でstatsmodelsと一致すること（uncentered TSSのR²等）。"""
    rng = np.random.default_rng(7)
    n = 30
    x1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * x1 + rng.normal(0.0, 0.5, n)
    df = pl.DataFrame({"y": y, "x1": x1})

    sm_res = sm.OLS(y, x1.reshape(-1, 1)).fit(use_t=True)  # 定数項なし
    options = OLSOptions(include_intercept=False)
    our_res = OLS(df, y="y", x=["x1"], options=options).fit()

    assert our_res.param_names == ["x1"]
    _assert_close(our_res.params["x1"], sm_res.params[0], "params/x1")
    _assert_close(our_res.std_errors["x1"], sm_res.bse[0], "se/x1")
    _assert_close(our_res.r_squared, sm_res.rsquared, "r_squared")


@pytest.mark.parametrize(
    "cov_type", ["classical", "hc0", "hc1", "hc2", "hc3", "cluster", "hac"]
)
def test_include_intercept_false_matches_statsmodels_robust_cov_types(
    dataset, cov_type
):
    """`include_intercept=False`が、ロバスト系cov_type（HC0-3/cluster/HAC）でも
    statsmodelsと一致すること。

    上の`test_include_intercept_false_matches_statsmodels`はcov_typeを指定
    しない（classical相当）比較のみだったため、include_intercept=Falseが
    engine_pybind側のcov_type分岐ロジックとも独立に正しく配線されていることを
    確認する（テスト網羅性レビュー、Issue #231フェーズ4で判明した抜け）。
    """
    y = dataset["y"].to_numpy()
    x = np.column_stack([dataset["x1"].to_numpy(), dataset["x2"].to_numpy()])

    fit_kwargs: dict = {"use_t": True}
    if cov_type == "cluster":
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": dataset["cluster"].to_numpy()}
    elif cov_type == "hac":
        fit_kwargs["cov_type"] = "HAC"
        fit_kwargs["cov_kwds"] = {"maxlags": 2}
    elif cov_type != "classical":
        fit_kwargs["cov_type"] = cov_type.upper()

    sm_res = sm.OLS(y, x).fit(**fit_kwargs)  # 定数項なし

    options = OLSOptions(
        include_intercept=False,
        cov_type=cov_type,
        cluster_col="cluster" if cov_type == "cluster" else None,
        hac_lags=2 if cov_type == "hac" else None,
    )
    our_res = OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()

    assert our_res.param_names == ["x1", "x2"]
    for i, name in enumerate(["x1", "x2"]):
        _assert_close(
            our_res.params[name],
            sm_res.params[i],
            f"[{cov_type}] params/{name}",
        )
        _assert_close(
            our_res.std_errors[name],
            sm_res.bse[i],
            f"[{cov_type}] se/{name}",
        )
    _assert_close(
        our_res.r_squared, sm_res.rsquared, f"[{cov_type}] r_squared"
    )
