"""OLSの主リファレンス（statsmodels）による数値比較テスト。

`tests/fixtures/benchmarks/ols.json`（`benchmark/linear/fixtures/generate_ols_fixtures.py`
で生成）を読み込み、6つの合成データシナリオ×classical/HC0-3/HAC + クラスター(baselineのみ)で、
係数・標準誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する
（`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。

役割分担:
    - 構造・API・エラーパスの検証: `test_ols.py`
    - 主リファレンス（statsmodels）との厳密な数値一致: このファイル
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

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _assertions import rename_intercept as _rename
from _helpers import DATA_DIR, with_cluster_groups
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

RTOL = TOLERANCES["ols_fixtures"]["rtol"]
ATOL = TOLERANCES["ols_fixtures"]["atol"]

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
    `test_cluster_g2_with_multiple_slopes_raises_computation_error`参照。
    実装中に判明した境界条件）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv")
    df = with_cluster_groups(df, 2)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = OLS(df, y="y", x=["x1"], options=options).fit()

    ref = fixtures["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


def test_cluster_g2_with_multiple_slopes_raises_computation_error():
    """G=2×説明変数3個（傾き係数q=3）は、ロバストWald検定の共分散部分行列
    （3x3）のランクがクラスタ数G=2以下になり必然的に特異になるため、
    fit()全体がComputationErrorになる（係数・標準誤差自体は計算可能だが、
    F検定の失敗でfit()全体が失敗する仕様。実装中に判明、
    数値比較はしない想定）。
    """
    from econometricsmodels import ComputationError

    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 2)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    with pytest.raises(ComputationError):
        OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。
    想定エラー（`ComputationError`）が発生することのみを確認する。
    """
    from econometricsmodels import ComputationError

    df = pl.read_csv(DATA_DIR / "synthetic_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        OLS(df, y="y", x=["x1", "x2", "x3"]).fit()


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_scale_variance_raises_computation_error(cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    傾き係数の同時共分散部分行列がスケール比の2乗（≈1e18）相当の条件数を持ち
    倍精度浮動小数点の限界を超えて数値的に特異になる（発見・原因調査済み）。`wald_f_test`が固有値分解による相対閾値判定で検出し、
    全cov_typeで`ComputationError`になる（classicalも含む。傾き係数の
    共分散部分行列自体は`cov_type`によらず同じ条件数を持つため）。
    perfect_multicollinearityと同様、数値比較はせずエラーパスのみ確認する。
    """
    from econometricsmodels import ComputationError

    df = pl.read_csv(DATA_DIR / "synthetic_scale_variance.csv")
    kwargs = {"hac_lags": HAC_LAG_IN_FIXTURE} if cov_type == "hac" else {}
    options = OLSOptions(cov_type=cov_type, **kwargs)
    with pytest.raises(ComputationError):
        OLS(df, y="y", x=["x1", "x2", "x3"], options=options).fit()
