"""Logitの主リファレンス（statsmodels）による数値比較テスト。

`tests/api_tests/fixtures/benchmarks/logit.json`（`benchmark/nonlinear/fixtures/
generate_logit_fixtures.py`で生成）を読み込み、真のlogit DGPによる合成データ
シナリオ×classical/opg/hc0 + クラスター(baseline・mrozの実データ両方) +
Wooldridge実データ（mroz）で、係数・標準誤差・検定統計量・適合度統計量・
限界効果を相対誤差1e-8で厳密比較する
（`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。

役割分担:
    - 構造・API・エラーパスの検証: `test_logit.py`
    - 主リファレンス（statsmodels）との厳密な数値一致: このファイル
    - 独立実装（R）とのクロスチェック: `test_logit_crosscheck.py`

Note:
    `cov_type="hc1"`はここに含めない（statsmodelsのdiscrete modelがn/(n-k)
    小標本補正を実装しておらずHC0と同一値になるバグ的な欠落があるため。
    `benchmark/nonlinear/run_statsmodels_benchmark.py`のdocstring参照）。
    `hc1`の数値比較は`test_logit_crosscheck.py`（R側が主リファレンス）で行う。

    `cov_type="opg"`の限界効果はstatsmodels側では算出できない（同docstring参照）
    ため、opgのmarginal_effects()数値比較は`test_logit_crosscheck.py`のみで行う。
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
        / "nonlinear"
        / "fixtures"
    ),
)
from _common import imbalanced_cluster_groups  # noqa: E402
from generate_logit_fixtures import (  # noqa: E402
    NUMERIC_SCENARIOS as SCENARIOS,
)

from econometricsmodels import ComputationError, Logit, LogitOptions  # noqa: E402

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "logit.json"
)
DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"

RTOL = 1e-8
# OLS（閉形式解）のATOL=1e-10より緩い。Logitは反復最適化（Newton/BFGS/L-BFGS）の
# ため、ゼロ近傍の値（信頼区間の境界等）で閉形式解より1桁大きい浮動小数点誤差が
# 乗ることを実測確認した（ベンチマーク作成時、diff~2.6e-10のケース）。
ATOL = 1e-9

# near_separation（logit特有の準完全分離境界ケース）は、既定のtol=1e-6（勾配ノルム
# 基準）だとstatsmodelsとの数値一致がRTOL=1e-8を満たさない（実測diff~7e-8相対）。
# tol=1e-8まで締めると一致することを確認済みだが、既定値自体は変更しない
# （BFGSがmax_iter=35のうち34回を要するようになり、他の難しいデータで
# NonConvergenceリスクが上がるため。ユーザー確認済み）。
# このシナリオの数値比較テストに限り、明示的にtol=1e-8を指定する。
_NEAR_SEPARATION_TOL = 1e-8

COV_TYPES = ["classical", "opg", "hc0"]

MARGEFF_AT = ["overall", "mean", "median"]


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _rename(name: str) -> str:
    return "const" if name == "Intercept" else name


def _assert_close(ours: float, ref: float, label: str) -> None:
    diff = abs(ours - ref)
    tol = max(RTOL * abs(ref), ATOL)
    assert diff <= tol, (
        f"{label}: ours={ours!r}, ref={ref!r}, diff={diff!r} > tol={tol!r}"
    )


def _assert_dict_close(
    ours: dict[str, float], ref: dict[str, float], label: str
) -> None:
    for name, ref_val in ref.items():
        _assert_close(ours[_rename(name)], ref_val, f"{label}/{name}")


def _check_margeff(res, ref_margeff: dict, label: str) -> None:
    for at in MARGEFF_AT:
        effects = {row["param"]: row for row in res.marginal_effects(at=at)}
        for name, ref_stats in ref_margeff[at].items():
            row = effects[_rename(name)]
            _assert_close(
                row["dydx"], ref_stats["dydx"], f"{label}/{at}/{name}/dydx"
            )
            _assert_close(
                row["std_err"], ref_stats["se"], f"{label}/{at}/{name}/se"
            )
            _assert_close(row["z"], ref_stats["z"], f"{label}/{at}/{name}/z")
            _assert_close(
                row["p_value"],
                ref_stats["p_value"],
                f"{label}/{at}/{name}/p_value",
            )
            _assert_close(
                row["conf_low"],
                ref_stats["conf_low"],
                f"{label}/{at}/{name}/conf_low",
            )
            _assert_close(
                row["conf_high"],
                ref_stats["conf_high"],
                f"{label}/{at}/{name}/conf_high",
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
    df = pl.read_csv(DATA_DIR / f"logit_{scenario}.csv")
    kwargs = (
        {"tol": _NEAR_SEPARATION_TOL} if scenario == "near_separation" else {}
    )
    options = LogitOptions(cov_type=cov_type, **kwargs)
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


def test_cluster_matches_statsmodels(fixtures):
    """クラスターロバストSE（baselineシナリオ、行番号%10の疑似グループ）。"""
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 10).alias("cluster_group"))
        .drop("_row")
    )
    options = LogitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_statsmodels(fixtures):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。"""
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = LogitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster_imbalanced"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


def test_cluster_g2_matches_statsmodels(fixtures):
    """クラスタ数境界（G=2ちょうど）の成功パス。

    OLSのwald_f_testと異なりLogitのcluster_cov_paramsはq×q部分行列の反転を
    要求しないため、説明変数を1個に絞る必要はない（k=3のままG=2で正常に
    計算できることを実機確認済み、generate_logit_fixtures.py参照）。
    """
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 2).alias("cluster_group"))
        .drop("_row")
    )
    options = LogitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。"""
    df = pl.read_csv(DATA_DIR / "logit_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        Logit(df, y="y", x=["x1", "x2", "x3"]).fit()


MROZ_X = ["nwifeinc", "educ", "exper", "expersq", "age", "kidslt6", "kidsge6"]


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_mroz_matches_statsmodels(fixtures, cov_type):
    """Wooldridge実データ（mroz、労働参加モデル）とのクロスチェック。"""
    wooldridge = pytest.importorskip("wooldridge")
    pandas_df = wooldridge.data("mroz")
    df = pl.from_pandas(pandas_df)
    options = LogitOptions(cov_type=cov_type)
    res = Logit(df, y="inlf", x=MROZ_X, options=options).fit()

    _check_result(res, fixtures["mroz"][cov_type], f"mroz/{cov_type}")


def test_mroz_cluster_matches_statsmodels(fixtures):
    """実データでのクラスターロバストSE（`city`＝都市部居住ダミー、484/269の2値）。

    `testing-policy.md`「テスト用データセット」3.の「実データでのグループ列も
    検証する」を満たす（OLSのwage1/regionクラスターと同じ趣旨）。
    """
    wooldridge = pytest.importorskip("wooldridge")
    pandas_df = wooldridge.data("mroz")
    df = pl.from_pandas(pandas_df)
    options = LogitOptions(cov_type="cluster", cluster_col="city")
    res = Logit(df, y="inlf", x=MROZ_X, options=options).fit()

    ref = fixtures["mroz"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "mroz/cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "mroz/cluster/se")
