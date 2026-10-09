"""GMM（`estimator="gmm"`）の独立実装によるクロスチェック（R momentfit）。

`tests/fixtures/benchmarks/iv_gmm_crosscheck.json`（`benchmark/iv/fixtures/
generate_iv_gmm_crosscheck_fixtures.py`で生成）を読み込み、主リファレンス
（linearmodels `IVGMM`、`test_iv_gmm_reference.py`）とは別の実装であるmomentfitと
数値比較する。`ivreg`はGMMに対応していないため、2SLSのRクロスチェック
（`test_iv_crosscheck.py`）とは別ファイルにしている。

検証範囲は`test_iv_gmm_reference.py`と同じ構成（10合成シナリオ×classical重み×
cov_type、baselineの他の重み・重みと共分散の組み合わせ、複数内生変数、Wooldridge
card）。構成の定義は`generate_iv_gmm_fixtures.py`からimportして単一の定義元にする。

比較する統計量: 係数・標準誤差・z値・p値・信頼区間・`n_obs`/`df_resid`・
ロバストWald（`wald_statistic`/`wald_p_value`）・Hansen J（過剰識別のときのみ）。
弱操作変数F・R²は2SLSのivregクロスチェックと重複するため含めない。
hc2/hc3（linearmodels・momentfitとも対応なし）と反復GMM（フィクスチャが
classical重みのみで、反復しても2SLSと同じ結果になる）は対象外。

momentfitを本実装・linearmodelsと揃えるための設定と、揃えないと一致しない原因
（momentfit 1.0のHAC・クラスター重みのバグ、`bw`の定義、Hansen Jの重み、標準誤差の
小標本補正の流儀の違い）は`benchmark/iv/references/run_momentfit.R`のヘッダ
コメントに集約している。

役割分担:
    - 主リファレンス（linearmodels）との厳密な数値一致: `test_iv_gmm_reference.py`
    - 独立実装（momentfit）との数値一致: このファイル
    - GMM固有の構造・API・オプション反映: `test_iv_api.py`
    - `ValidationError`/`ComputationError`パス: `test_iv_validation.py`
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _assertions import rename_intercept as _rename
from _constants import DATA_DIR
from _helpers import (
    hac_time_for,
    load_wooldridge_dataset,
    with_cluster_groups,
    with_row_time,
)
from _tolerances import TOLERANCES
from econometricsmodels import IV, IVOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.iv.fixtures.generate_iv_crosscheck_fixtures import CARD_X_EXOG
from benchmark.iv.fixtures.generate_iv_gmm_fixtures import (
    CARD_OTHER_WEIGHT_TYPES,
    COV_TYPES,
    CROSS_WEIGHT_COV_COMBINATIONS,
    INSTRUMENTS_BY_SCENARIO,
    OTHER_WEIGHT_TYPES,
    X_EXOG_BY_SCENARIO,
)
from benchmark.iv.fixtures.generate_iv_gmm_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "iv_gmm_crosscheck.json"
)

RTOL = TOLERANCES["iv_gmm_crosscheck"]["rtol"]
ATOL = TOLERANCES["iv_gmm_crosscheck"]["atol"]

_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)

CLUSTER_COL = "cluster_group"


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(FIXTURE_PATH.read_text())["synthetic"]


@pytest.fixture(scope="module")
def crosscheck_wooldridge() -> dict:
    return json.loads(FIXTURE_PATH.read_text())["wooldridge"]


def _check_result(res, ref: dict, label: str) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(
        res.test_stats, ref["test_stats"], f"{label}/test_stats"
    )
    _assert_dict_close(res.p_values, ref["p_values"], f"{label}/p_values")

    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[_rename(name)]
        _assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        _assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"
    _assert_close(
        res.wald_statistic, ref["f_statistic"], f"{label}/f_statistic"
    )
    _assert_close(res.wald_p_value, ref["f_p_value"], f"{label}/f_p_value")

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

    if "hac_lag" in ref:
        assert res.hac_lags_used == ref["hac_lag"], f"{label}/hac_lags_used"


def _fit(
    df: pl.DataFrame,
    *,
    y: str,
    x_exog: list[str],
    x_endog: list[str],
    instruments: list[str],
    gmm_weight_type: str,
    cov_type: str,
):
    """`gmm_weight_type`・`cov_type`のどちらかがclusterなら`CLUSTER_COL`列を
    クラスター変数として渡す（重みと共分散で共用）。
    """
    uses_cluster = "cluster" in (gmm_weight_type, cov_type)
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type=gmm_weight_type,
        cov_type=cov_type,
        **({"cluster": CLUSTER_COL} if uses_cluster else {}),
        **hac_time_for(gmm_weight_type, cov_type),
    )
    return IV(
        with_row_time(df),
        y=y,
        x_exog=x_exog,
        x_endog=x_endog,
        instruments=instruments,
        options=options,
    ).fit()


def _fit_baseline(
    gmm_weight_type: str, cov_type: str, *, imbalanced: bool = False
):
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    if imbalanced:
        df = df.with_columns(
            pl.Series(CLUSTER_COL, imbalanced_cluster_groups(df.height))
        )
    else:
        df = with_cluster_groups(df, 10, col=CLUSTER_COL)
    return _fit(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        gmm_weight_type=gmm_weight_type,
        cov_type=cov_type,
    )


# ── 合成データ ────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_momentfit(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    res = _fit(
        df,
        y="y",
        x_exog=X_EXOG_BY_SCENARIO.get(scenario, ["x1"]),
        x_endog=["endog1"],
        instruments=INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"]),
        gmm_weight_type="classical",
        cov_type=cov_type,
    )
    _check_result(
        res,
        crosscheck[scenario]["classical"][cov_type],
        f"{scenario}/classical/{cov_type}",
    )


def test_cluster_matches_momentfit(crosscheck):
    """クラスターロバストSE（`gmm_weight_type="classical"`固定、疑似グループ
    行番号%10）。
    """
    res = _fit_baseline("classical", "cluster")
    _check_result(
        res,
        crosscheck["baseline"]["classical"]["cluster"],
        "baseline/classical/cluster",
    )


def test_cluster_imbalanced_matches_momentfit(crosscheck):
    """不均衡クラスター（サイズ[2, 3, 5, 10, 30, 50]のタイル）。"""
    res = _fit_baseline("classical", "cluster", imbalanced=True)
    _check_result(
        res,
        crosscheck["baseline"]["classical"]["cluster_imbalanced"],
        "baseline/classical/cluster_imbalanced",
    )


@pytest.mark.parametrize("gmm_weight_type", OTHER_WEIGHT_TYPES)
def test_other_weight_types_match_momentfit(crosscheck, gmm_weight_type):
    """`gmm_weight_type`（点推定の重み）が`cov_type`（classical固定）と独立な軸
    であることをbaselineで数値照合する。
    """
    res = _fit_baseline(gmm_weight_type, "classical")
    _check_result(
        res,
        crosscheck["baseline"][gmm_weight_type]["classical"],
        f"baseline/{gmm_weight_type}/classical",
    )


def test_hac_weight_hac_cov_matches_momentfit(crosscheck):
    """HACカーネル重み×HAC標準誤差。"""
    res = _fit_baseline("hac", "hac")
    _check_result(
        res, crosscheck["baseline"]["hac"]["hac"], "baseline/hac/hac"
    )


@pytest.mark.parametrize(
    ("gmm_weight_type", "cov_type"), CROSS_WEIGHT_COV_COMBINATIONS
)
def test_cross_weight_cov_matches_momentfit(
    crosscheck, gmm_weight_type, cov_type
):
    """重みと共分散の型が両方非既定かつ異なる組み合わせ。"""
    res = _fit_baseline(gmm_weight_type, cov_type)
    _check_result(
        res,
        crosscheck["baseline"][gmm_weight_type][cov_type],
        f"baseline/{gmm_weight_type}/{cov_type}",
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_multi_endog_matches_momentfit(crosscheck, cov_type):
    """複数内生変数（`x_endog=["endog1", "endog2"]`）。"""
    df = pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")
    res = _fit(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2", "z3"],
        gmm_weight_type="classical",
        cov_type=cov_type,
    )
    _check_result(
        res,
        crosscheck["multi_endog"]["classical"][cov_type],
        f"multi_endog/classical/{cov_type}",
    )


# ── 実データ（Wooldridge card） ──────────────────────────────


def _fit_card(gmm_weight_type: str, cov_type: str):
    return _fit(
        load_wooldridge_dataset("card"),
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        gmm_weight_type=gmm_weight_type,
        cov_type=cov_type,
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_card_matches_momentfit(crosscheck_wooldridge, cov_type):
    """実データセット（Wooldridge card）。classical重みで全cov_type。"""
    _check_result(
        _fit_card("classical", cov_type),
        crosscheck_wooldridge["card"]["classical"][cov_type],
        f"card/classical/{cov_type}",
    )


@pytest.mark.parametrize("gmm_weight_type", CARD_OTHER_WEIGHT_TYPES)
def test_card_other_weight_types_match_momentfit(
    crosscheck_wooldridge, gmm_weight_type
):
    """実データセット（Wooldridge card）で非classicalの重み（cov_type=classical
    固定）。
    """
    _check_result(
        _fit_card(gmm_weight_type, "classical"),
        crosscheck_wooldridge["card"][gmm_weight_type]["classical"],
        f"card/{gmm_weight_type}/classical",
    )
