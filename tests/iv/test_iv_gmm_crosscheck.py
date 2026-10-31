"""GMM（`estimator="gmm"`）の独立実装によるクロスチェック（R momentfit）。

`tests/fixtures/benchmarks/iv_gmm_crosscheck.json`（`benchmark/iv/fixtures/
generate_iv_gmm_crosscheck_fixtures.py`で生成）を読み込み、主リファレンス
（linearmodels `IVGMM`、`test_iv_gmm_reference.py`）とは別の実装であるmomentfitと
数値比較する。`ivreg`はGMMに対応していないため、2SLSのRクロスチェック
（`test_iv_crosscheck.py`）とは別ファイルにしている。

検証範囲は`test_iv_gmm_reference.py`と同じ構成（`NUMERIC_SCENARIOS`の各合成シナリオ×classical重み×
cov_type、baselineの他の重み・重みと共分散の組み合わせ、複数内生変数、Wooldridge
card）。構成の定義は単一の定義元からimportする（シナリオは`benchmark/iv/datasets.py`、cov_typeと
Card実データは`benchmark/iv/constants.py`、重みの組み合わせは`generate_iv_gmm_fixtures.py`）。

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

from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import (
    assert_close,
    assert_dict_close,
    assert_fields_close,
    assert_inference_close,
    assert_p_value_close,
)
from _constants import DATA_DIR
from _fixture_cov import load_fixtures_with_cov
from _helpers import (
    hac_time_for,
    load_wooldridge_dataset,
    with_cluster_groups,
    with_row_time,
)
from _tolerances import TOLERANCES
from econometricsmodels import IV, IVOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.iv.constants import (
    CARD_REGION_COLUMN,
    CARD_X_EXOG,
    CLUSTER_ILL_CONDITIONED_SCENARIOS,
    LINEARMODELS_COV_TYPES,
)
from benchmark.iv.datasets import (
    MULTI_ENDOG_SCENARIOS,
    THREE_ENDOG_INSTRUMENTS,
    THREE_ENDOG_X_ENDOG,
    multi_endog_instruments,
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)
from benchmark.iv.datasets import (
    NUMERIC_SCENARIOS as SCENARIOS,
)
from benchmark.iv.fixtures.generate_iv_fixtures import NO_INTERCEPT_SCENARIOS
from benchmark.iv.fixtures.generate_iv_gmm_fixtures import (
    CARD_OTHER_WEIGHT_TYPES,
    CROSS_WEIGHT_COV_COMBINATIONS,
    MULTI_ENDOG_OTHER_WEIGHT_TYPES,
    OTHER_WEIGHT_TYPES,
    OVERID_REJECTION_WEIGHT_TYPES,
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
_assert_fields_close = partial(assert_fields_close, rtol=RTOL, atol=ATOL)

CLUSTER_COL = "cluster_group"


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)["synthetic"]


@pytest.fixture(scope="module")
def crosscheck_wooldridge() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)["wooldridge"]


def _check_result(
    res, ref: dict, label: str, *, rtol_wald: float = RTOL
) -> None:
    assert_inference_close(res, ref, label, rtol=RTOL, atol=ATOL)

    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"
    # `rtol_wald`: Wald/F統計量とp値だけの相対許容誤差。実データ（card）×クラスターで
    # クラスター共分散Vの精度（約3e-12）が`V_slopes`の条件数（約3e7）で増幅され、
    # 他の統計量より緩い値が要る（`_tolerances_iv.py`の`rtol_wald_card_cluster`参照）。
    assert_fields_close(
        res,
        ref,
        (("wald_statistic", "f_statistic"), ("wald_p_value", "f_p_value")),
        label,
        rtol=rtol_wald,
        atol=ATOL,
    )

    if ref["hansen_j_statistic"] is None:
        assert res.overid_statistic is None, f"{label}/overid_statistic"
        assert res.overid_p_value is None, f"{label}/overid_p_value"
    else:
        _assert_close(
            res.overid_statistic,
            ref["hansen_j_statistic"],
            f"{label}/overid_statistic",
        )
        assert_p_value_close(
            res.overid_p_value,
            ref["hansen_j_p_value"],
            f"{label}/overid_p_value",
            rtol=RTOL,
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
    include_intercept: bool = True,
):
    """`gmm_weight_type`・`cov_type`のどちらかがclusterなら`CLUSTER_COL`列を
    クラスター変数として渡す（重みと共分散で共用）。
    """
    uses_cluster = "cluster" in (gmm_weight_type, cov_type)
    options = IVOptions(
        estimator="gmm",
        include_intercept=include_intercept,
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


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_momentfit(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    res = _fit(
        df,
        y="y",
        x_exog=scenario_x_exog(scenario),
        x_endog=["endog1"],
        instruments=scenario_instruments(scenario),
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


@pytest.mark.parametrize("scenario", CLUSTER_ILL_CONDITIONED_SCENARIOS)
def test_cluster_ill_conditioned_matches_momentfit(crosscheck, scenario):
    """悪条件・多重共線性シナリオ×クラスター共分散（classical重み、均等な疑似グループ
    行番号%10）。
    """
    df = with_cluster_groups(
        pl.read_csv(DATA_DIR / f"iv_{scenario}.csv"), 10, col=CLUSTER_COL
    )
    res = _fit(
        df,
        y="y",
        x_exog=scenario_x_exog(scenario),
        x_endog=["endog1"],
        instruments=scenario_instruments(scenario),
        gmm_weight_type="classical",
        cov_type="cluster",
    )
    _check_result(
        res,
        crosscheck[scenario]["classical"]["cluster"],
        f"{scenario}/classical/cluster",
    )


def test_hac_weight_hac_cov_matches_momentfit(crosscheck):
    """HACカーネル重み×HAC標準誤差。"""
    res = _fit_baseline("hac", "hac")
    _check_result(
        res, crosscheck["baseline"]["hac"]["hac"], "baseline/hac/hac"
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


@pytest.mark.parametrize("gmm_weight_type", OVERID_REJECTION_WEIGHT_TYPES)
def test_invalid_instrument_other_weight_types_match_momentfit(
    crosscheck, gmm_weight_type
):
    """過剰識別の棄却シナリオ（`invalid_instrument`）の、classical以外の重み
    （`test_iv_gmm_reference.py`の同名テストと同じ理由）。
    """
    df = pl.read_csv(DATA_DIR / "iv_invalid_instrument.csv")
    res = _fit(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        gmm_weight_type=gmm_weight_type,
        cov_type="classical",
    )
    _check_result(
        res,
        crosscheck["invalid_instrument"][gmm_weight_type]["classical"],
        f"invalid_instrument/{gmm_weight_type}/classical",
    )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
@pytest.mark.parametrize("scenario", MULTI_ENDOG_SCENARIOS)
def test_multi_endog_matches_momentfit(crosscheck, scenario, cov_type):
    """複数内生変数（`x_endog=["endog1", "endog2"]`）。"""
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}_multi_endog.csv")
    res = _fit(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=multi_endog_instruments(scenario),
        gmm_weight_type="classical",
        cov_type=cov_type,
    )
    _check_result(
        res,
        crosscheck["multi_endog"][scenario]["classical"][cov_type],
        f"multi_endog/{scenario}/classical/{cov_type}",
    )


@pytest.mark.parametrize(
    ("scenario", "gmm_weight_type"),
    [
        (scenario, weight_type)
        for scenario, weight_types in MULTI_ENDOG_OTHER_WEIGHT_TYPES.items()
        for weight_type in weight_types
    ],
)
def test_multi_endog_other_weight_types_match_momentfit(
    crosscheck, scenario, gmm_weight_type
):
    """複数内生変数×classical以外の重み（`cov_type="classical"`、
    `test_iv_gmm_reference.py`の同名テストと同じ理由）。
    """
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}_multi_endog.csv")
    if gmm_weight_type == "cluster":
        df = with_cluster_groups(df, 10, col=CLUSTER_COL)
    res = _fit(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=multi_endog_instruments(scenario),
        gmm_weight_type=gmm_weight_type,
        cov_type="classical",
    )
    _check_result(
        res,
        crosscheck["multi_endog"][scenario][gmm_weight_type]["classical"],
        f"multi_endog/{scenario}/{gmm_weight_type}/classical",
    )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
def test_three_endog_matches_momentfit(crosscheck, cov_type):
    """内生変数3本（過剰識別度1、classical重み）。"""
    df = pl.read_csv(DATA_DIR / "iv_baseline_three_endog.csv")
    res = _fit(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=THREE_ENDOG_X_ENDOG,
        instruments=THREE_ENDOG_INSTRUMENTS,
        gmm_weight_type="classical",
        cov_type=cov_type,
    )
    _check_result(
        res,
        crosscheck["three_endog"]["classical"][cov_type],
        f"three_endog/classical/{cov_type}",
    )


def test_multi_endog_cluster_matches_momentfit(crosscheck):
    """複数内生変数×クラスター共分散（classical重み、疑似グループ行番号%10）。"""
    df = with_cluster_groups(
        pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv"),
        10,
        col=CLUSTER_COL,
    )
    res = _fit(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2", "z3"],
        gmm_weight_type="classical",
        cov_type="cluster",
    )
    _check_result(
        res,
        crosscheck["multi_endog"]["baseline"]["classical"]["cluster"],
        "multi_endog/baseline/classical/cluster",
    )


# ── 凍結フィクスチャとの数値照合（オプション） ────────────────────────


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
@pytest.mark.parametrize("scenario", list(NO_INTERCEPT_SCENARIOS))
def test_no_intercept_matches_momentfit(crosscheck, scenario, cov_type):
    """`include_intercept=False`（切片なし）、`gmm_weight_type="classical"`固定。
    過剰識別のHansen Jは、linearmodelsと異なり非中心化の分散を使うmomentfitと
    比較する（`test_iv_gmm_reference.py`のモジュールdocコメント参照）。
    """
    spec = NO_INTERCEPT_SCENARIOS[scenario]
    df = pl.read_csv(DATA_DIR / f"iv_{spec['dataset']}.csv")
    res = _fit(
        df,
        y="y",
        x_exog=spec["x_exog"],
        x_endog=spec["x_endog"],
        instruments=spec["instruments"],
        gmm_weight_type="classical",
        cov_type=cov_type,
        include_intercept=False,
    )
    label = f"no_intercept/{scenario}/{cov_type}/momentfit"
    assert "const" not in res.param_names, label
    _check_result(
        res, crosscheck["no_intercept"][scenario]["classical"][cov_type], label
    )


def test_no_intercept_cluster_covariance_matches_momentfit(crosscheck):
    """切片なし×クラスターロバストSE（`baseline`、classical重み、疑似グループ
    行番号%10）。
    """
    df = with_cluster_groups(
        pl.read_csv(DATA_DIR / "iv_baseline.csv"), 10, col=CLUSTER_COL
    )
    res = _fit(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        gmm_weight_type="classical",
        cov_type="cluster",
        include_intercept=False,
    )
    _check_result(
        res,
        crosscheck["no_intercept"]["baseline"]["classical"]["cluster"],
        "no_intercept/baseline/classical/cluster/momentfit",
    )


@pytest.mark.parametrize(
    ("scenario", "gmm_weight_type"),
    [
        ("baseline", "robust"),
        ("baseline", "cluster"),
        ("baseline", "hac"),
        ("multi_endog", "robust"),
    ],
)
def test_no_intercept_other_weight_types_match_momentfit(
    crosscheck, scenario, gmm_weight_type
):
    """切片なし×classical以外の重み、`cov_type="classical"`（GMM固有の、残差から
    重み行列を作る経路）。momentfit側の2SLS・重みの扱いは
    `run_momentfit.R`のヘッダコメント8.参照。
    """
    spec = NO_INTERCEPT_SCENARIOS[scenario]
    df = pl.read_csv(DATA_DIR / f"iv_{spec['dataset']}.csv")
    if gmm_weight_type == "cluster":
        df = with_cluster_groups(df, 10, col=CLUSTER_COL)
    res = _fit(
        df,
        y="y",
        x_exog=spec["x_exog"],
        x_endog=spec["x_endog"],
        instruments=spec["instruments"],
        gmm_weight_type=gmm_weight_type,
        cov_type="classical",
        include_intercept=False,
    )
    ref = crosscheck["no_intercept"][scenario][gmm_weight_type]["classical"]
    _check_result(
        res, ref, f"no_intercept/{scenario}/{gmm_weight_type}/momentfit"
    )


# ── 凍結フィクスチャとの数値照合（実データ） ──────────────────────────


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


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
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


def test_card_cluster_matches_momentfit(crosscheck_wooldridge):
    """実データのグループ列（居住地域の9グループ）でのクラスター共分散
    （classical重み、`test_iv_gmm_reference.py`の同名テストと同じ理由）。
    """
    df = with_card_region(load_wooldridge_dataset("card")).rename(
        {CARD_REGION_COLUMN: CLUSTER_COL}
    )
    res = _fit(
        df,
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        gmm_weight_type="classical",
        cov_type="cluster",
    )
    _check_result(
        res,
        crosscheck_wooldridge["card"]["classical"]["cluster"],
        "card/classical/cluster",
        rtol_wald=TOLERANCES["iv_gmm_crosscheck"]["rtol_wald_card_cluster"],
    )
