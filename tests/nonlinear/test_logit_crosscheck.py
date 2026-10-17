"""Logitの独立実装（R: glm + sandwich + marginaleffects）による数値比較テスト。

`tests/fixtures/benchmarks/logit_crosscheck.json`（`benchmark/nonlinear/
fixtures/generate_logit_crosscheck_fixtures.py`で生成）を読み込み、係数・標準誤差・
適合度統計量・限界効果（全統計量）をRとクロスチェックする（cluster 2ケースと、切片なし
（`include_intercept=False`）と非既定の`confidence_level`（0.90）の全cov_typeも全統計量・限界効果）。役割分担は`test_logit_reference.py`
と同じ（`.claude/rules/testing-policy.md`「リファレンス実装」参照）。

Note:
    `cov_type="hc1"`はここが主リファレンスを担う（statsmodelsのdiscrete modelが
    n/(n-k)小標本補正を実装しておらずHC0と同一値になるバグ的な欠落があるため。
    `benchmark/nonlinear/references/statsmodels_ref.py`のdocstring参照）。

    許容誤差はOLSのRクロスチェック（classical/HC0-3/clusterで機械精度一致）より
    緩い。LogitはRのglm（IRLS/Fisher scoring）と本実装（Newton/BFGS/L-BFGS）が
    どちらも反復最適化のため、OLSの閉形式解同士の比較（機械精度一致）ほどの
    精度は出ない。ただしR側の参照値は`glm()`の収束判定と`marginaleffects`の
    数値微分の刻み幅を厳しくして生成しており（`benchmark/nonlinear/references/
    run_glm_crosscheck.R`参照）、基本方針はRTOL=2e-7（実測最大相対誤差~3.5e-8に対する
    マージン。ATOLは1e-12で純粋な相対誤差比較）。全統計量に同じ許容誤差を使う
    （根拠は`tests/support/_tolerances_nonlinear.py`の`logit_crosscheck`のコメント参照）。
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from _assertions import (
    assert_fields_close,
    assert_inference_close,
)
from _constants import APPLE_CLUSTER, APPLE_X, DATA_DIR, MROZ_X
from _helpers import load_wooldridge_dataset, with_cluster_groups
from _nonlinear_helpers import check_margeff, load_apple_adoption_dataset
from _tolerances import TOLERANCES
from econometricsmodels import Logit, LogitOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.nonlinear.constants import (
    CONFIDENCE_LEVEL_NON_DEFAULT,
    NON_CLUSTER_COV_TYPES,
)
from benchmark.nonlinear.fixtures.generate_logit_crosscheck_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "logit_crosscheck.json"
)

RTOL = TOLERANCES["logit_crosscheck"]["rtol"]
ATOL = TOLERANCES["logit_crosscheck"]["atol"]

# near_separationは既定tol=1e-6だとstatsmodels/Rとの一致精度が下がる境界ケース
# （test_logit_reference.py参照）。ここでも同じ理由でtol=1e-8を明示指定する。
_NEAR_SEPARATION_TOL = 1e-8


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _check_result(
    res, ref: dict, label: str, *, confidence_level: float = 0.95
) -> None:
    """全統計量を比べる。`confidence_level`は係数の信頼区間は`fit()`のオプション側で、
    限界効果の信頼区間は`marginal_effects()`の引数側で指定するため、後者に渡す。"""
    assert_inference_close(
        res,
        ref,
        label,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_fields_close(
        res,
        ref,
        (
            "log_likelihood",
            "log_likelihood_null",
            "aic",
            "bic",
            "lr_statistic",
            "lr_p_value",
            "pseudo_r_squared",
        ),
        label,
        rtol=RTOL,
        atol=ATOL,
    )
    check_margeff(
        res,
        ref["margeff"],
        label,
        rtol=RTOL,
        atol=ATOL,
        confidence_level=confidence_level,
    )


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_r_glm(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"logit_{scenario}.csv")
    kwargs = (
        {"tol": _NEAR_SEPARATION_TOL} if scenario == "near_separation" else {}
    )
    options = LogitOptions(cov_type=cov_type, **kwargs)
    x_cols = [c for c in df.columns if c != "y"]
    res = Logit(df, y="y", x=x_cols, options=options).fit()

    ref = fixtures["synthetic"][scenario][cov_type]
    _check_result(res, ref, f"{scenario}/{cov_type}")


def test_cluster_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = LogitOptions(cov_type="cluster", cluster="cluster_group")
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster"]
    _check_result(res, ref, "cluster")


def test_cluster_imbalanced_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = LogitOptions(cov_type="cluster", cluster="cluster_group")
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster_imbalanced"]
    _check_result(res, ref, "cluster_imbalanced")


# ── 凍結フィクスチャとの数値照合（オプション） ────────────────────────


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
def test_no_intercept_matches_r_glm(fixtures, cov_type):
    """`include_intercept=False`（baseline）。hc1（全統計量）とopgの限界効果はRのみが参照。

    `log_likelihood_null`は常に切片のみのモデルを参照するため、`lr_statistic`が
    負値になりうる。
    """
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    options = LogitOptions(cov_type=cov_type, include_intercept=False)
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["no_intercept"][cov_type]
    _check_result(res, ref, f"no_intercept/{cov_type}")


def test_no_intercept_cluster_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = LogitOptions(
        cov_type="cluster", cluster="cluster_group", include_intercept=False
    )
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["no_intercept"]["cluster"]
    _check_result(res, ref, "no_intercept/cluster")


def test_no_intercept_cluster_imbalanced_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = LogitOptions(
        cov_type="cluster", cluster="cluster_group", include_intercept=False
    )
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["no_intercept"][
        "cluster_imbalanced"
    ]
    _check_result(res, ref, "no_intercept/cluster_imbalanced")


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
def test_confidence_level_matches_r_glm(fixtures, cov_type):
    """非既定の`confidence_level`（baseline）。係数と限界効果の信頼区間をRと比べる。

    臨界値を固定した実装（例: 1.96）は0.90の区間で外れる。限界効果の水準は
    `fit()`のオプションとは独立の`marginal_effects(confidence_level=...)`で指定する。
    """
    level = CONFIDENCE_LEVEL_NON_DEFAULT
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    options = LogitOptions(cov_type=cov_type, confidence_level=level)
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["confidence_level"][cov_type]
    _check_result(
        res, ref, f"confidence_level/{cov_type}", confidence_level=level
    )


def test_confidence_level_cluster_matches_r_glm(fixtures):
    level = CONFIDENCE_LEVEL_NON_DEFAULT
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = LogitOptions(
        cov_type="cluster", cluster="cluster_group", confidence_level=level
    )
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["confidence_level"]["cluster"]
    _check_result(res, ref, "confidence_level/cluster", confidence_level=level)


# ── 凍結フィクスチャとの数値照合（実データ） ──────────────────────────


def test_apple_cluster_matches_r_glm(fixtures):
    """多クラスターの実データ（Wooldridge apple、`ecolbs > 0`、州でG=49）。"""
    df = load_apple_adoption_dataset()
    options = LogitOptions(cov_type="cluster", cluster=APPLE_CLUSTER)
    res = Logit(df, y="y", x=APPLE_X, options=options).fit()

    ref = fixtures["wooldridge"]["apple"]["cluster"]
    _check_result(res, ref, "apple/cluster")


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
def test_mroz_matches_r_glm(fixtures, cov_type):
    df = load_wooldridge_dataset("mroz")
    options = LogitOptions(cov_type=cov_type)
    res = Logit(df, y="inlf", x=MROZ_X, options=options).fit()

    ref = fixtures["wooldridge"]["mroz"][cov_type]
    _check_result(res, ref, f"mroz/{cov_type}")
