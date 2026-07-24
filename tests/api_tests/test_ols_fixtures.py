"""OLSの主リファレンス（statsmodels）による数値比較テスト。

`tests/api_tests/fixtures/benchmarks/ols.json`（`benchmark/fixtures/generate_ols_fixtures.py`
で生成）を読み込み、6つの合成データシナリオ×classical/HC0-3/HAC + クラスター(baselineのみ)で、
係数・標準誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する
（`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。

役割分担:
    - 構造・API・エラーパスの検証: `test_ols.py`
    - 主リファレンス（statsmodels）との厳密な数値一致: このファイル
    - 独立実装（R）とのクロスチェック: `test_ols_crosscheck.py`

Note:
    `generate_ols_fixtures.py`と同じ決定論的データ生成（`generate_dataset()`、
    seed固定）を使ってフィクスチャ生成時と同じデータを再現するため、
    `benchmark/`をimport pathに追加する（`ols_crosscheck.json`と同じ設計。
    フィクスチャJSON自体には生データを含めない）。
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
    Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "ols.json"
)

# testing-policy.md「許容誤差」の基本方針: 相対誤差1e-8。
# ATOLは0近傍の値（p値のアンダーフロー等）向けの下限フロー。
RTOL = 1e-8
ATOL = 1e-10

SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
]
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]

# generate_ols_fixtures.py（run_statsmodels_benchmark.py）はHACのラグを
# maxlags=1に固定している。同じラグを明示的に指定し、自動ラグ選択式の
# 違いを比較対象から除外する。
HAC_LAG_IN_FIXTURE = 1


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _rename(name: str) -> str:
    """statsmodels(formula API)の切片名"Intercept"を本実装の"const"に揃える。"""
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
    assert res.nobs == ref["nobs"], f"{label}/nobs"


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_statsmodels(fixtures, scenario, cov_type):
    df, _ = generate_dataset(scenario)
    kwargs = {"hac_lags": HAC_LAG_IN_FIXTURE} if cov_type == "hac" else {}
    options = OLSOptions(cov_type=cov_type, **kwargs)
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


def test_cluster_matches_statsmodels(fixtures):
    """クラスターロバストSE。`generate_ols_fixtures.py`と同じ疑似グループ
    （行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ（`coef`/`se`のみが記録されている）。
    """
    df, _ = generate_dataset("baseline")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 10).alias("cluster_group"))
        .drop("_row")
    )
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。
    想定エラー（`ComputationError`）が発生することのみを確認する。
    """
    from econometricsmodels import ComputationError

    df, _ = generate_dataset("perfect_multicollinearity")
    with pytest.raises(ComputationError):
        OLS(df, y="y", x=["x1", "x2", "x3"]).fit()
