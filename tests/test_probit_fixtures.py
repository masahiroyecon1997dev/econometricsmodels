"""Probitの主リファレンス（statsmodels）による数値比較テスト。

`tests/fixtures/benchmarks/probit.json`（`benchmark/nonlinear/fixtures/
generate_probit_fixtures.py`で生成）を読み込み、真のprobit DGPによる合成データ
シナリオ×classical/opg/hc0 + クラスター(baseline・mrozの実データ両方) +
Wooldridge実データ（mroz）で、係数・標準誤差・検定統計量・適合度統計量・
限界効果を相対誤差1e-8で厳密比較する（`test_logit_fixtures.py`と完全に同型。
`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。

役割分担:
    - 構造・API・エラーパスの検証: `test_probit.py`
    - 主リファレンス（statsmodels）との厳密な数値一致: このファイル
    - 独立実装（R）とのクロスチェック: `test_probit_crosscheck.py`

Note:
    `cov_type="hc1"`はここに含めない（statsmodelsのdiscrete modelがn/(n-k)
    小標本補正を実装しておらずHC0と同一値になるバグ的な欠落があるため、Probitでも
    同じ欠落を実機確認済み。`benchmark/nonlinear/run_statsmodels_benchmark.py`の
    docstring参照）。`hc1`の数値比較は`test_probit_crosscheck.py`（R側が主リファレンス）
    で行う。

    `cov_type="opg"`の限界効果はstatsmodels側では算出できない（同docstring参照）
    ため、opgのmarginal_effects()数値比較は`test_probit_crosscheck.py`のみで行う。
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
    str(
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "nonlinear"
        / "fixtures"
    ),
)
from _assertions import assert_close, assert_dict_close, check_margeff
from _assertions import rename_intercept as _rename
from _common import imbalanced_cluster_groups
from _helpers import (
    DATA_DIR,
    MROZ_X,
    load_wooldridge_dataset,
    with_cluster_groups,
)
from _tolerances import TOLERANCES
from econometricsmodels import (
    ComputationError,
    Probit,
    ProbitOptions,
)
from generate_probit_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)
from run_statsmodels_benchmark import run

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "probit.json"
)

RTOL = TOLERANCES["probit_fixtures"]["rtol"]
# OLS（閉形式解）のATOL=1e-10より緩い。Probitは反復最適化（Newton/BFGS/L-BFGS）の
# ため、ゼロ近傍の値（信頼区間の境界等）で閉形式解より1桁大きい浮動小数点誤差が
# 乗ることを実測確認した（Logitと同じ理由、`test_logit_fixtures.py`参照）。
ATOL = TOLERANCES["probit_fixtures"]["atol"]

# near_separation（probit特有の準完全分離境界ケース）は、既定のtol=1e-6（勾配ノルム
# 基準）だとstatsmodelsとの数値一致がRTOL=1e-8を満たさない（実測diff~4.4e-8相対、
# Logitのnear_separationと同種の現象）。tol=1e-8まで締めると一致することを確認済み
# だが、既定値自体は変更しない（Logitと同じ理由、`nonlinear-implementation-notes.md`
# 「収束判定のtol」参照）。このシナリオの数値比較テストに限り、明示的にtol=1e-8を
# 指定する。
_NEAR_SEPARATION_TOL = 1e-8

COV_TYPES = ["classical", "opg", "hc0"]

MARGEFF_AT = ["overall", "mean", "median"]


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)
_check_margeff = partial(check_margeff, rtol=RTOL, atol=ATOL)

# method="bfgs"/"lbfgs"はnewtonと異なる最適化経路で収束するため、既定のRTOLより
# 緩めた許容誤差を使う（tests/_tolerances.py参照、test_logit_fixtures.pyと同じ方針）。
RTOL_METHOD = TOLERANCES["probit_fixtures"]["rtol_method"]
_assert_dict_close_method = partial(
    assert_dict_close, rtol=RTOL_METHOD, atol=ATOL
)


def _check_result(res, ref: dict, label: str) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(res.z_stats, ref["z_stats"], f"{label}/z_stats")
    _assert_dict_close(res.p_values, ref["p_values"], f"{label}/p_values")

    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_name = _rename(name)
        our_lower, our_upper = res.conf_int[our_name]
        _assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        _assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    _assert_close(
        res.log_likelihood, ref["log_likelihood"], f"{label}/log_likelihood"
    )
    _assert_close(
        res.log_likelihood_null,
        ref["log_likelihood_null"],
        f"{label}/log_likelihood_null",
    )
    _assert_close(
        res.lr_statistic, ref["lr_statistic"], f"{label}/lr_statistic"
    )
    _assert_close(res.lr_p_value, ref["lr_p_value"], f"{label}/lr_p_value")
    _assert_close(
        res.pseudo_r_squared,
        ref["pseudo_r_squared"],
        f"{label}/pseudo_r_squared",
    )
    _assert_close(res.aic, ref["aic"], f"{label}/aic")
    _assert_close(res.bic, ref["bic"], f"{label}/bic")
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_model == ref["df_model"], f"{label}/df_model"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"
    assert res.converged == ref["converged"], f"{label}/converged"

    ours_pred_table = {
        row["actual"]: (row["predicted_0"], row["predicted_1"])
        for row in res.pred_table()
    }
    for i, row in enumerate(ref["pred_table"]):
        assert ours_pred_table[i] == (row[0], row[1]), (
            f"{label}/pred_table/{i}"
        )

    if ref["margeff"] is not None:
        _check_margeff(res, ref["margeff"], label)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_statsmodels(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"probit_{scenario}.csv")
    kwargs = (
        {"tol": _NEAR_SEPARATION_TOL} if scenario == "near_separation" else {}
    )
    options = ProbitOptions(cov_type=cov_type, **kwargs)
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


def test_cluster_matches_statsmodels(fixtures):
    """クラスターロバストSE（baselineシナリオ、行番号%10の疑似グループ）。"""
    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = ProbitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_statsmodels(fixtures):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。"""
    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = ProbitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster_imbalanced"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


def test_cluster_g2_matches_statsmodels(fixtures):
    """クラスタ数境界（G=2ちょうど）の成功パス。

    Logitのcluster_cov_paramsと同じくOLSのwald_f_testのようなq×q部分行列の
    反転を要求しないため、説明変数を1個に絞る必要はない（k=3のままG=2で
    正常に計算できることを実機確認済み、generate_probit_fixtures.py参照）。
    """
    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    df = with_cluster_groups(df, 2)
    options = ProbitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


@pytest.mark.parametrize("method", ["bfgs", "lbfgs"])
def test_method_matches_statsmodels(fixtures, method):
    """`method="bfgs"/"lbfgs"`が主リファレンス（statsmodelsの同じmethod）と
    フルの統計量（std_errors含む）で一致すること（`test_logit_fixtures.py`と
    同じ方針、Issue #231フェーズ4）。
    """
    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    options = ProbitOptions(cov_type="classical", method=method)
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["method"][method]
    label = f"method/{method}"
    _assert_dict_close_method(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close_method(res.std_errors, ref["se"], f"{label}/se")
    assert res.converged == ref["converged"], f"{label}/converged"


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_include_intercept_false_matches_statsmodels(cov_type):
    """`include_intercept=False`の成功パスが検証されていなかった
    （`test_logit_fixtures.py`と同じ理由、Issue #231フェーズ4）。
    """
    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    ref = run(
        dataset_source="synthetic",
        dataset="baseline",
        formula="y ~ x1 + x2 + x3 - 1",
        cov_type=cov_type,
        model="probit",
    )

    options = ProbitOptions(cov_type=cov_type, include_intercept=False)
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    label = f"include_intercept_false/{cov_type}"
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    assert res.converged == ref["converged"], f"{label}/converged"
    assert res.df_model == ref["df_model"], f"{label}/df_model"


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。"""
    df = pl.read_csv(DATA_DIR / "probit_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        Probit(df, y="y", x=["x1", "x2", "x3"]).fit()


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_mroz_matches_statsmodels(fixtures, cov_type):
    """Wooldridge実データ（mroz、労働参加モデル）とのクロスチェック。"""
    df = load_wooldridge_dataset("mroz")
    options = ProbitOptions(cov_type=cov_type)
    res = Probit(df, y="inlf", x=MROZ_X, options=options).fit()

    _check_result(res, fixtures["mroz"][cov_type], f"mroz/{cov_type}")


def test_mroz_cluster_matches_statsmodels(fixtures):
    """実データでのクラスターロバストSE（`city`＝都市部居住ダミー、484/269の2値）。

    `testing-policy.md`「テスト用データセット」3.の「実データでのグループ列も
    検証する」を満たす（Logitのmrozクラスターと同じ趣旨）。
    """
    df = load_wooldridge_dataset("mroz")
    options = ProbitOptions(cov_type="cluster", cluster_col="city")
    res = Probit(df, y="inlf", x=MROZ_X, options=options).fit()

    ref = fixtures["mroz"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "mroz/cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "mroz/cluster/se")
