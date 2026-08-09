"""GMM（`method="gmm"`）の主リファレンス（linearmodels `IVGMM`）による数値比較
テスト。

`tests/api_tests/fixtures/benchmarks/iv_gmm.json`（`benchmark/iv/fixtures/
generate_iv_gmm_fixtures.py`で生成）を読み込み、8つの合成データシナリオ×
classical/HC0/HC1/HAC（+クラスター、baselineのみ）を`weight_type="unadjusted"`
固定で、加えてbaselineシナリオ×`cov_type="classical"`固定で他の`weight_type`
（robust/cluster/kernel）を検証する（Issue #171、ユーザー確認済みの検証範囲。
`weight_type`×`cov_type`の全組み合わせ（8シナリオ×4weight_type×6cov_type）は
規模が大きすぎるため）。

役割分担:
    - 主リファレンス（linearmodels `IVGMM`）との厳密な数値一致: このファイル
    - `method="2sls"`の同種テスト: `test_iv_fixtures.py`
    - GMM固有の構造・API・エラーパス（`weight_type`×`cov_type`の独立性の構造
      確認、収束/非収束等）: `test_iv.py`

Note:
    - `hc2`/`hc3`はlinearmodelsに対応する実装が無いため対象外
      （`iv_gmm.json`の`_meta.note`参照）。
    - `z_stats`/`p_values`/`conf_int`/`f_statistic`/`f_p_value`は、本実装が
      GMMで常にz分布・カイ二乗形式（qで割らない）を使う設計のため、フィクスチャ
      側も`linearmodels`の値をそのまま使わず独自に計算し直したもの
      （`run_gmm()`のモジュールdocコメント参照）。
    - `hansen_j_statistic`/`hansen_j_p_value`はGMMの過剰識別検定（Hansen J）。
      2SLSのSargan検定に対応し、丁度識別のときは`None`。
    - `wu_hausman_statistic`相当のキーはフィクスチャに存在しない
      （`GmmEstimator`はWu-Hausman検定を実装しないため、`IvResults.
      wu_hausman_statistic`は`method="gmm"`で常に`None`。`test_iv.py`の
      `test_wu_hausman_is_none_for_gmm`で構造確認済み）。
    - `weak_instrument_f_statistics`は本実装が`method`によらず常にclassicalで
      計算する設計のため、フィクスチャの`weak_instrument_f_independent`と比較する
      （`test_iv_fixtures.py`と同じ理由）。

    フィクスチャ生成時と同じ入力データを、`tests/api_tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/freeze_datasets.py`参照）から読む。
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from econometricsmodels import IV, IvOptions

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "iv_gmm.json"
)
DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"

# testing-policy.md「許容誤差」の基本方針: 相対誤差1e-8。
# ATOLは0近傍の値（p値のアンダーフロー等）向けの下限フロー。
RTOL = 1e-8
ATOL = 1e-10

SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "small_n",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
]
COV_TYPES = ["classical", "hc0", "hc1", "hac"]

INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}

# HACラグはIvOptions.hac_lags未指定（自動計算）で、engineとlinearmodelsが
# 同じ式を使うため明示指定不要（`test_iv_fixtures.py`と同じ理由）。


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


def _check_result(res, ref: dict, label: str) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(res.stats, ref["z_stats"], f"{label}/stats")
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
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"

    if ref["hansen_j_statistic"] is None:
        assert res.overid_statistic is None, f"{label}/overid_statistic"
        assert res.overid_p_value is None, f"{label}/overid_p_value"
    else:
        _assert_close(
            res.overid_statistic,
            ref["hansen_j_statistic"],
            f"{label}/overid_statistic",
        )
        _assert_close(
            res.overid_p_value,
            ref["hansen_j_p_value"],
            f"{label}/overid_p_value",
        )

    assert res.wu_hausman_statistic is None, f"{label}/wu_hausman_statistic"

    _assert_dict_close(
        res.weak_instrument_f_statistics,
        ref["weak_instrument_f_independent"],
        f"{label}/weak_instrument_f_statistics",
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_linearmodels(fixtures, scenario, cov_type):
    x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
    instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    options = IvOptions(
        method="gmm", weight_type="unadjusted", cov_type=cov_type
    )
    res = IV(
        df,
        y="y",
        x_exog=x_exog,
        x_endog=["endog1"],
        instruments=instruments,
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures[scenario]["unadjusted"][cov_type],
        f"{scenario}/unadjusted/{cov_type}",
    )


def test_cluster_matches_linearmodels(fixtures):
    """クラスターロバストSE（`weight_type="unadjusted"`固定）。
    `generate_iv_gmm_fixtures.py`と同じ疑似グループ（行番号%10）を再現する。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 10).alias("cluster_group"))
        .drop("_row")
    )
    options = IvOptions(
        method="gmm",
        weight_type="unadjusted",
        cov_type="cluster",
        cluster_col="cluster_group",
    )
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = fixtures["baseline"]["unadjusted"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_linearmodels(fixtures):
    """不均衡クラスタ（`weight_type="unadjusted"`固定、サイズ
    [2, 3, 5, 10, 30, 50]のタイル）。
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
    from generate_synthetic_datasets import imbalanced_cluster_groups

    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = IvOptions(
        method="gmm",
        weight_type="unadjusted",
        cov_type="cluster",
        cluster_col="cluster_group",
    )
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = fixtures["baseline"]["unadjusted"]["cluster_imbalanced"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


@pytest.mark.parametrize("weight_type", ["robust", "cluster", "kernel"])
def test_other_weight_types_match_linearmodels(fixtures, weight_type):
    """`weight_type`（点推定の重み）が`cov_type`（SE計算方式、ここでは`classical`
    固定）と独立な軸であることを、baselineシナリオで数値照合する
    （`weight_type="unadjusted"`は`test_matches_linearmodels`で既に検証済み）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    kwargs = {}
    if weight_type == "cluster":
        df = (
            df.with_row_index("_row")
            .with_columns((pl.col("_row") % 10).alias("cluster_group"))
            .drop("_row")
        )
        kwargs["cluster_col"] = "cluster_group"

    options = IvOptions(
        method="gmm", weight_type=weight_type, cov_type="classical", **kwargs
    )
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["baseline"][weight_type]["classical"],
        f"baseline/{weight_type}/classical",
    )
