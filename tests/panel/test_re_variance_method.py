"""REの分散成分の推定法`REOptions.variance_method`（Wallace-Hussain・Amemiya）の検証。

`variance_method`は`plm`の`random.method`（`walhus`・`amemiya`）に対応し、参照実装は`plm`だけ
（`linearmodels.RandomEffects`はSwamy-Arora（既定）のみ。単一リファレンスの例外、
`docs/guide/verification.md`）。Swamy-Aroraの検証は`test_re_reference.py`・
`test_re_time_invariant.py`・`test_re_two_way_reference.py`が担う。

1. **plm**（`re_variance_method.json`、
   `benchmark/panel/fixtures/generate_re_variance_method_fixtures.py`）: 1-way（バランス・
   不均衡・準バランス・時間不変の列・時点間でのみ変わる列・実データ）と2-way（バランス、
   一次元の列・絶対年・実データ）で、分散成分・θ・係数・標準誤差・t・p値・信頼区間・F統計量・
   ハウスマン検定を比べる。代表ケースは全`cov_type`。不均衡の1-wayはplmのトレース項の式、
   バランスは閉形式で、両者は値が異なる（plmの切り替えに従う）。
2. **仕様どおりの独立な手計算**（`_re_oracle.py`）: plmのソースどおりに観測ごとの行列で書いた
   分散成分（`first_stage_variance_components`）を含め、全統計量を比べる。
3. **拒否する入力**: Amemiyaはwithin変動の無い列があると`ValidationError`（plmは不均衡で
   拒否、バランスでは時間不変の列を無視した値を返す。本実装は一律に拒否する）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import _error_messages as msgs
import polars as pl
import pytest
from _assertions import assert_close
from _constants import DATA_DIR
from _error_messages import escaped
from _helpers import load_wooldridge_dataset
from _re_oracle import (
    CONDITIONING_SENSITIVE_KEYS,
    _Panel,
    assert_stats_close,
    first_stage_variance_components,
    hand_computed,
    result_stats,
)
from _tolerances import TOLERANCES
from econometricsmodels import (
    RE,
    ComputationError,
    REOptions,
    ValidationError,
)

from benchmark.panel.fixtures.generate_re_variance_method_fixtures import (
    AMEMIYA_REJECTED_CASES,
    CASES,
    HAUSMAN_KEY,
    WAGEPAN_CSV,
    WAGEPAN_TREND,
    Case,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_variance_method.json"
)

_TOL = TOLERANCES["re_variance_method"]
RTOL = _TOL["rtol"]
ATOL = _TOL["atol"]
RTOL_P_VALUE = _TOL["rtol_p_value"]
RTOL_ABSOLUTE_YEAR = _TOL["rtol_absolute_year"]

# 絶対年の列を含む入力（丸め誤差が拡大する統計量だけ許容誤差を緩める）。
ABSOLUTE_YEAR_CASES = {
    "ow_absolute_year",
    "tw_absolute_year",
    "tw_wagepan_educ_trend",
}

_ONE_WAY_PLM_KEYS = (
    "params",
    "std_errors",
    "test_stats",
    "p_values",
    "f_statistic",
    "f_p_value",
    "sigma2_eps",
    "sigma2_u",
)
_TWO_WAY_PLM_KEYS = (
    *_ONE_WAY_PLM_KEYS,
    "f_df_num",
    "f_df_denom",
    "df_resid",
    "log_likelihood",
    "aic",
    "bic",
    "r_squared_within",
    "r_squared_between",
    "r_squared_overall",
    "sigma2_lambda",
    "theta_entity",
    "theta_time",
    "theta_total",
    "hausman_statistic",
    "hausman_p_value",
    "hausman_df",
)


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _frame(case: Case) -> pl.DataFrame:
    if case.csv == WAGEPAN_CSV:
        return load_wooldridge_dataset("wagepan").with_columns(
            pl.col("year").alias(WAGEPAN_TREND)
        )
    return pl.read_csv(DATA_DIR / f"{case.csv}.csv")


def _plm_entry(reference: dict, case: Case) -> dict:
    section = "wooldridge" if case.csv == WAGEPAN_CSV else "synthetic"
    return reference[section][case.name]


def _options(case: Case, cov_type: str, method: str) -> REOptions:
    return REOptions(
        cov_type=cov_type,
        time=case.time_col if case.two_way else None,
        dk_time=case.time_col if cov_type == "dk" else None,
        variance_method=method,
    )


def _fit(case: Case, cov_type: str, method: str, df=None):
    return RE(
        _frame(case) if df is None else df,
        y=case.y_col,
        x=list(case.x),
        entity=case.entity_col,
        options=_options(case, cov_type, method),
    ).fit()


def _tolerances(case: Case) -> dict:
    tol = {"rtol": RTOL, "atol": ATOL, "rtol_p_value": RTOL_P_VALUE}
    if case.name in ABSOLUTE_YEAR_CASES:
        tol["rtol_by_key"] = dict.fromkeys(
            CONDITIONING_SENSITIVE_KEYS, RTOL_ABSOLUTE_YEAR
        )
    return tol


def _plm_stats(case: Case, entry: dict, cov_type: str) -> dict:
    """plmのフィクスチャを`result_stats`と同じキー名の辞書にそろえる。"""
    plm = entry[cov_type]
    if case.two_way:
        out = {key: plm[key] for key in _TWO_WAY_PLM_KEYS}
    else:
        out = {key: plm[key] for key in _ONE_WAY_PLM_KEYS}
        if case.hausman:
            hausman = entry[HAUSMAN_KEY][cov_type]
            out["hausman_statistic"] = hausman["hausman_statistic"]
            out["hausman_p_value"] = hausman["hausman_p_value"]
            out["hausman_df"] = int(hausman["hausman_df"])
    out["conf_int"] = {k: tuple(v) for k, v in plm["conf_int"].items()}
    return out


PLM_PARAMS = [
    pytest.param(case, method, cov_type, id=f"{case.name}-{method}-{cov_type}")
    for case in CASES
    for method in case.methods
    for cov_type in case.cov_types
]
METHOD_PARAMS = [
    pytest.param(case, method, id=f"{case.name}-{method}")
    for case in CASES
    for method in case.methods
]
ONE_WAY_METHOD_PARAMS = [
    pytest.param(case, method, id=f"{case.name}-{method}")
    for case in CASES
    if not case.two_way
    for method in case.methods
]


# ── plmとの照合 ──────────────────────────────────────────────────


@pytest.mark.parametrize(("case", "method", "cov_type"), PLM_PARAMS)
def test_matches_plm(reference, case, method, cov_type):
    """分散成分・係数・標準誤差・t・p値・信頼区間・F統計量・ハウスマン検定がplmと一致する
    （2-wayはさらに自由度・θ₁〜θ₃、対数尤度・AIC・BIC・R²はplmの係数からの手計算）。"""
    entry = _plm_entry(reference, case)[method]
    res = _fit(case, cov_type, method)

    assert_stats_close(
        result_stats(res),
        _plm_stats(case, entry, cov_type),
        f"{case.name}/{method}/{cov_type}",
        **_tolerances(case),
    )
    assert res.variance_method == method


@pytest.mark.parametrize(("case", "method"), ONE_WAY_METHOD_PARAMS)
def test_one_way_theta_matches_plm_variance_components(
    reference, case, method
):
    """1-wayのθ_iは、plmの分散成分から`1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`で計算した値と
    一致する（不均衡パネルではエンティティごとに異なる。2-wayの重みは`test_matches_plm`）。"""
    plm = _plm_entry(reference, case)[method]["classical"]
    res = _fit(case, "classical", method)
    counts = (
        _frame(case)
        .group_by(case.entity_col, maintain_order=False)
        .len()
        .with_columns(pl.col(case.entity_col).cast(pl.String))
    )
    t = dict(zip(counts[case.entity_col], counts["len"], strict=True))
    assert set(res.theta) == set(t)
    for entity, theta in res.theta.items():
        want = 1.0 - math.sqrt(
            plm["sigma2_eps"]
            / (t[entity] * plm["sigma2_u"] + plm["sigma2_eps"])
        )
        assert_close(
            theta,
            want,
            f"{case.name}/{method}/theta/{entity}",
            rtol=RTOL,
            atol=ATOL,
        )


# ── 仕様どおりの独立な手計算との照合 ─────────────────────────────


@pytest.mark.parametrize(("case", "method", "cov_type"), PLM_PARAMS)
def test_matches_hand_computed(case, method, cov_type):
    """全統計量が、仕様どおりに書いた手計算（`_re_oracle.py`。分散成分はplmのソースどおりに
    観測ごとの行列で書いた独立な実装）と一致する。"""
    df = _frame(case)
    res = _fit(case, cov_type, method, df)
    hand = hand_computed(
        df,
        list(case.x),
        cov_type,
        two_way=case.two_way,
        entity_col=case.entity_col,
        time_col=case.time_col,
        y_col=case.y_col,
        variance_method=method,
    )
    assert_stats_close(
        result_stats(res),
        hand,
        f"{case.name}/{method}/{cov_type}",
        **_tolerances(case),
    )


@pytest.mark.parametrize(("case", "method"), METHOD_PARAMS)
def test_hand_computed_variance_components_match_plm(reference, case, method):
    """手計算の分散成分（閉形式・トレース項の式）がplmの`ercomp`と一致する
    （手計算の実装がplmと同じ推定量であることの確認）。"""
    plm = _plm_entry(reference, case)[method]["classical"]
    panel = _Panel(
        _frame(case), list(case.x), case.entity_col, case.time_col, case.y_col
    )
    eps, u, lam = first_stage_variance_components(panel, method, case.two_way)
    label = f"{case.name}/{method}"
    assert_close(
        eps, plm["sigma2_eps"], f"{label}/sigma2_eps", rtol=RTOL, atol=ATOL
    )
    assert_close(u, plm["sigma2_u"], f"{label}/sigma2_u", rtol=RTOL, atol=ATOL)
    if case.two_way:
        assert_close(
            lam,
            plm["sigma2_lambda"],
            f"{label}/sigma2_lambda",
            rtol=RTOL,
            atol=ATOL,
        )


# ── 推定法の違い・既定値 ─────────────────────────────────────────


def test_default_variance_method_is_swamy_arora():
    case = CASES[0]
    default = RE(
        _frame(case),
        y="y",
        x=list(case.x),
        entity="entity",
        options=REOptions(cov_type="classical"),
    ).fit()
    explicit = _fit(case, "classical", "swamy_arora")
    assert default.variance_method == "swamy_arora"
    assert explicit.params == default.params
    assert explicit.sigma2_u == default.sigma2_u


def test_the_three_methods_give_different_variance_components():
    case = CASES[0]
    sigma = {
        m: (r.sigma2_eps, r.sigma2_u)
        for m in ("swamy_arora", "wallace_hussain", "amemiya")
        for r in [_fit(case, "classical", m)]
    }
    assert len(set(sigma.values())) == 3


@pytest.mark.parametrize(
    "value", ["WALLACE_HUSSAIN", "Amemiya", "Swamy_Arora"]
)
def test_variance_method_is_case_insensitive(value):
    case = CASES[0]
    res = _fit(case, "classical", value)
    assert res.variance_method == value.lower()


def test_unknown_variance_method_raises():
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_VARIANCE_METHOD, value="nerlove"),
    ):
        _fit(CASES[0], "classical", "nerlove")


def test_variance_method_must_be_a_string():
    with pytest.raises(TypeError):
        REOptions(variance_method=1)


def test_swamy_arora_option_does_not_change_the_unbalanced_one_way_formula(
    reference,
):
    """不均衡の1-wayの`swamy_arora`は`linearmodels`準拠の式のまま（plmの`swar`とは異なる）。
    `walhus`・`amemiya`はplmに従うため、plmの`swar`とは別の値になる。"""
    case = next(c for c in CASES if c.name == "ow_unbalanced")
    swar = _fit(case, "classical", "swamy_arora")
    walhus = _fit(case, "classical", "wallace_hussain")
    assert swar.sigma2_u != pytest.approx(walhus.sigma2_u, rel=1e-3)


# ── 拒否する入力・失敗する入力 ───────────────────────────────────


@pytest.mark.parametrize("case", AMEMIYA_REJECTED_CASES, ids=lambda c: c.name)
def test_amemiya_rejects_regressors_without_within_variation(case):
    """時間不変の列（2-wayでは個体間・時点間でのみ変わる列）があるAmemiyaは、
    バランスパネルでも不均衡でも`ValidationError`で、列の名前を示す。"""
    expected = "x_invariant" if not case.two_way else "xi, xt"
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.AMEMIYA_NEEDS_WITHIN_VARIATION, columns=expected),
    ):
        _fit(case, "classical", "amemiya")


def test_amemiya_accepts_a_regressor_that_varies_only_over_time():
    """時点間でのみ変わる列は個体内で変動するため、1-wayのAmemiyaで使える。"""
    case = next(c for c in CASES if c.name == "ow_time_only")
    assert "amemiya" in case.methods
    assert math.isfinite(_fit(case, "classical", "amemiya").sigma2_u)


@pytest.mark.parametrize("method", ["swamy_arora", "wallace_hussain"])
def test_methods_other_than_amemiya_accept_time_invariant_regressors(method):
    case = next(c for c in CASES if c.name == "ow_time_invariant_balanced")
    assert math.isfinite(_fit(case, "classical", method).sigma2_u)


def test_wallace_hussain_pooled_regression_failure_is_a_computation_error():
    """時間不変の列とその定数倍（内部のFEには渡らない）でプールドOLSが特異になる。"""
    case = next(c for c in CASES if c.name == "ow_time_invariant_balanced")
    df = _frame(case).with_columns(z2=2.0 * pl.col("x_invariant"))
    with pytest.raises(ComputationError, match="pooled"):
        RE(
            df,
            y="y",
            x=["x1", "x_invariant", "z2"],
            entity="entity",
            options=REOptions(variance_method="wallace_hussain"),
        ).fit()


@pytest.mark.parametrize("method", ["wallace_hussain", "amemiya"])
def test_two_way_unbalanced_panel_still_raises(method):
    case = next(c for c in CASES if c.name == "tw_baseline")
    df = _frame(case).slice(1)
    with pytest.raises(ValidationError, match="balanced"):
        _fit(case, "classical", method, df)


def test_variance_method_does_not_change_the_hausman_degrees_of_freedom():
    """within推定とハウスマン検定の対象列は`variance_method`によらない。"""
    case = next(c for c in CASES if c.name == "ow_baseline")
    dfs = {m: _fit(case, "classical", m).hausman_df for m in case.methods}
    assert set(dfs.values()) == {2}
