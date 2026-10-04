"""Logitの独立実装（R: glm + sandwich + marginaleffects）による数値比較テスト。

`tests/fixtures/benchmarks/logit_crosscheck.json`（`benchmark/nonlinear/
fixtures/generate_logit_crosscheck_fixtures.py`で生成）を読み込み、係数・標準誤差・
適合度統計量・限界効果をRとクロスチェックする。役割分担は`test_logit_reference.py`
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
    run_glm_crosscheck.R`参照）、基本方針はRTOL=1e-6（実測最大相対誤差~6e-8に対する
    マージン。ATOLは1e-12で純粋な相対誤差比較）。p値のみ裾での増幅に備えた個別のATOLを設定している
    （根拠は`tests/_tolerances.py`の`logit_crosscheck`のコメント参照。
    `testing-policy.md`「許容誤差」の方針通り）。
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from _constants import DATA_DIR, MROZ_X
from _helpers import load_wooldridge_dataset, with_cluster_groups
from _tolerances import TOLERANCES
from econometricsmodels import Logit, LogitOptions

from benchmark.common import imbalanced_cluster_groups
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

# p値は標準正規分布CDFの裾で係数・zのわずかな数値差が増幅されるため、係数・SE本体
# より緩いATOLを置く（rtolで収まらない実測最大絶対誤差~5.5e-9、baseline/opg/x1）。
ATOL_P_VALUE = TOLERANCES["logit_crosscheck"]["atol_p_value"]

COV_TYPES = ["classical", "opg", "hc0", "hc1"]
MARGEFF_AT = ["overall", "mean", "median"]

# near_separationは既定tol=1e-6だとstatsmodels/Rとの一致精度が下がる境界ケース
# （test_logit_reference.py参照）。ここでも同じ理由でtol=1e-8を明示指定する。
_NEAR_SEPARATION_TOL = 1e-8


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _assert_close(
    ours: float,
    ref: float,
    label: str,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    diff = abs(ours - ref)
    tol = max(rtol * abs(ref), atol)
    assert diff <= tol, (
        f"{label}: ours={ours!r}, ref={ref!r}, diff={diff!r} > tol={tol!r}"
    )


def _assert_dict_close(
    ours: dict[str, float],
    ref: dict[str, float],
    label: str,
    atol: float = ATOL,
) -> None:
    for name, ref_val in ref.items():
        _assert_close(ours[name], ref_val, f"{label}/{name}", atol=atol)


def _check_margeff(res, ref_margeff: dict, label: str) -> None:
    for at in MARGEFF_AT:
        effects = {row["param"]: row for row in res.marginal_effects(at=at)}
        for name, ref_stats in ref_margeff[at].items():
            row = effects[name]
            _assert_close(
                row["effect"],
                ref_stats["effect"],
                f"{label}/{at}/{name}/effect",
            )
            _assert_close(
                row["std_err"],
                ref_stats["std_err"],
                f"{label}/{at}/{name}/std_err",
            )


def _check_result(res, ref: dict, label: str) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(
        res.test_stats, ref["test_stats"], f"{label}/test_stats"
    )
    _assert_dict_close(
        res.p_values, ref["p_values"], f"{label}/p_values", atol=ATOL_P_VALUE
    )
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_close(
            our_lower,
            ref_lower,
            f"{label}/conf_lower/{name}",
        )
        _assert_close(
            our_upper,
            ref_upper,
            f"{label}/conf_upper/{name}",
        )
    for field in (
        "log_likelihood",
        "log_likelihood_null",
        "aic",
        "bic",
        "lr_statistic",
        "lr_p_value",
        "pseudo_r_squared",
    ):
        _assert_close(getattr(res, field), ref[field], f"{label}/{field}")
    if "margeff" in ref:
        _check_margeff(res, ref["margeff"], label)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_r_glm(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"logit_{scenario}.csv")
    kwargs = (
        {"tol": _NEAR_SEPARATION_TOL} if scenario == "near_separation" else {}
    )
    options = LogitOptions(cov_type=cov_type, **kwargs)
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"][scenario][cov_type]["r"]
    _check_result(res, ref, f"{scenario}/{cov_type}")


def test_cluster_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = LogitOptions(cov_type="cluster", cluster="cluster_group")
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster"]["r"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "logit_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = LogitOptions(cov_type="cluster", cluster="cluster_group")
    res = Logit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster_imbalanced"]["r"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_mroz_matches_r_glm(fixtures, cov_type):
    df = load_wooldridge_dataset("mroz")
    options = LogitOptions(cov_type=cov_type)
    res = Logit(df, y="inlf", x=MROZ_X, options=options).fit()

    ref = fixtures["wooldridge"]["mroz"][cov_type]["r"]
    _check_result(res, ref, f"mroz/{cov_type}")
