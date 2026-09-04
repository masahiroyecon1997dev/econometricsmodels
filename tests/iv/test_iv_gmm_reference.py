"""GMM（`method="gmm"`）の主リファレンス（linearmodels `IVGMM`）による数値比較
テスト。

`tests/fixtures/benchmarks/iv_gmm.json`（`benchmark/iv/fixtures/
generate_iv_gmm_fixtures.py`で生成）を読み込み、8つの合成データシナリオ×
classical/HC0/HC1/HAC（+クラスター、baselineのみ）を`weight_type="unadjusted"`
固定で、加えてbaselineシナリオ×`cov_type="classical"`固定で他の`weight_type`
（robust/cluster/kernel）を検証する（ユーザー確認済みの検証範囲。
`weight_type`×`cov_type`の全組み合わせ（8シナリオ×4weight_type×6cov_type）は
規模が大きすぎるため）。

役割分担（OLS/WLS/Logit/Probit の `test_<手法>_*.py` と同じ4分割、
`refactoring-candidates-2.md` 項目68。IV は主リファレンス数値照合のみ 2SLS/GMM で
ファイルが分かれ、api/validation は 2SLS と共通）:
    - 主リファレンス（linearmodels `IVGMM`）との厳密な数値一致: このファイル
    - `method="2sls"`の同種テスト: `test_iv_reference.py`
    - GMM固有の構造・API・オプション反映（`weight_type`×`cov_type`の独立性の構造
      確認、収束等）: `test_iv_api.py`
    - `ValidationError`/`ComputationError` パス（GMM 固有含む）: `test_iv_validation.py`

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
      wu_hausman_statistic`は`method="gmm"`で常に`None`。`test_iv_api.py`の
      `test_wu_hausman_is_none_for_gmm`で構造確認済み）。
    - `weak_instrument_f_statistics`は本実装が`method`によらず常にclassicalで
      計算する設計のため、フィクスチャの`weak_instrument_f_independent`と比較する
      （`test_iv_reference.py`と同じ理由）。

    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/iv/freeze.py`参照）から読む。
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
from econometricsmodels import IV, IvOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.iv.fixtures.generate_iv_gmm_fixtures import (
    COV_TYPES,
)
from benchmark.iv.fixtures.generate_iv_gmm_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "iv_gmm.json"
)

RTOL = TOLERANCES["iv_gmm_reference"]["rtol"]
ATOL = TOLERANCES["iv_gmm_reference"]["atol"]

# SCENARIOS/COV_TYPESはgenerate_iv_gmm_fixtures.pyのNUMERIC_SCENARIOS/COV_TYPESと
# 常に一致させる必要があるため、そちらをimportして単一の定義元にする。

INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}

# HACラグはIvOptions.hac_lags未指定（自動計算）で、engineとlinearmodelsが
# 同じ式を使うため明示指定不要（`test_iv_reference.py`と同じ理由）。


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)


def _check_result(
    res, ref: dict, label: str, *, check_overid: bool = True
) -> None:
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

    # check_overid=False: gmm_iterations=1のHansen J（過剰識別検定）専用の抜け穴。
    # weight_type="unadjusted"はSがZ'Zに固定され残差に依存しないため、係数・SEは
    # gmm_iterations=1/2/3で完全に一致する（実測確認済み）。本実装のHansen Jも
    # この不変性を保ち、gmm_iterations=1でもgmm_iterations>=2（2SLSのSarganと機械
    # 精度一致）と同じ値を返す（`gmm.rs`のσ̂²·Z'Zスケーリング設計、モジュールdoc
    # コメント「Hansen Jのgmm_iterations=1...のSは...」参照）。しかしlinearmodelsの
    # `IVGMM.fit(iter_limit=1)`のj_statはこの不変性を持たず、iter_limit=2/3の値
    # （2SLSのSarganと一致）とは異なる値を返す（実測: 0.30086708530935663 vs
    # 0.32832429087644643）。原因判明済み（`linearmodels`ソース実機調査、
    # `refactoring-candidates-3.md`旧項目21）: `IVGMM.fit`は`iters < iter_limit`
    # ループの中で初めて残差から重み行列`wmat`を再構築するため、`iter_limit=1`
    # では一度もループが実行されず`wmat`が`(Z'Z/n)⁻¹`という生の初期値のまま
    # J統計量に使われる（σ̂²スケーリング無し）。一方、本実装は
    # `weight_type=Unadjusted`のHansen Jに常にσ̂²スケーリング込みの重みを
    # 使う設計（ユーザー確認済み）のため、`iter_limit=1`固有の規約の違いに
    # よる原理的に一致しえない差異と判明した（バグではない）。
    # gmm_iterations=1のときはoverid_statistic/overid_p_valueの比較のみ除外する
    # （係数・SE等の他の統計量は引き続き比較する）。
    if check_overid:
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


# ── 凍結フィクスチャとの数値照合 ───────────────────────────────────


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
    df = with_cluster_groups(df, 10)
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


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_multi_endog_matches_linearmodels(fixtures, cov_type):
    """複数内生変数（`x_endog=["endog1", "endog2"]`）の成功パス
    （`weak_instrument_f_statistics`・`overid_statistic`（Hansen J、過剰識別）が
    複数内生変数を同時に扱うことをGMMでも確認する。`test_iv_reference.py`の
    同名テストと同じ理由、`testing-completeness-reviewer`指摘、
    Issue #231フェーズ4）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")
    options = IvOptions(
        method="gmm", weight_type="unadjusted", cov_type=cov_type
    )
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2", "z3"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["multi_endog"]["unadjusted"][cov_type],
        f"multi_endog/unadjusted/{cov_type}",
    )


def test_kernel_hac_matches_linearmodels(fixtures):
    """`weight_type="kernel"`×`cov_type="hac"`の組み合わせ（実務上最も典型的な
    「HACカーネル重み＋HAC標準誤差」の組み合わせ経路が、他の`weight_type`×
    `cov_type`の組み合わせと同様に独立して機能することを確認する。
    `test_other_weight_types_match_linearmodels`はcov_type="classical"固定のため
    この組み合わせを通らない（`testing-completeness-reviewer`指摘、
    Issue #231フェーズ4）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    options = IvOptions(method="gmm", weight_type="kernel", cov_type="hac")
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(res, fixtures["kernel_hac"], "kernel_hac")


@pytest.mark.parametrize("n_iter", [1, 3])
def test_gmm_iterations_matches_linearmodels(fixtures, n_iter):
    """`gmm_iterations`が既定値（2）以外（1: 1-step、3: iterated固定回数モード）
    でも主リファレンスと一致することを確認する（`testing-completeness-reviewer`
    指摘、Issue #231フェーズ4）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    options = IvOptions(
        method="gmm",
        weight_type="unadjusted",
        cov_type="classical",
        gmm_iterations=n_iter,
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
        fixtures["gmm_iterations"][str(n_iter)],
        f"gmm_iterations/{n_iter}",
        check_overid=(n_iter != 1),
    )


@pytest.mark.parametrize("weight_type", ["robust", "cluster", "kernel"])
def test_other_weight_types_match_linearmodels(fixtures, weight_type):
    """`weight_type`（点推定の重み）が`cov_type`（SE計算方式、ここでは`classical`
    固定）と独立な軸であることを、baselineシナリオで数値照合する
    （`weight_type="unadjusted"`は`test_matches_linearmodels`で既に検証済み）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    kwargs = {}
    if weight_type == "cluster":
        df = with_cluster_groups(df, 10)
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
