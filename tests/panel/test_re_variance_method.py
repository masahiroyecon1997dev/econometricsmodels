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
3. **statsmodels**: plmの分散成分から求めた重みで準偏差変換した`y`・`X`（切片込み）への
   OLSが計算する対数尤度・AIC・BICと、クラスターロバストの標準誤差・t値・p値・信頼区間・
   傾き係数の同時F検定（自由度`G-1`）を、本実装と比べる（plmが持たない値・plmと規約が
   異なる値を、手計算ではなくstatsmodelsがネイティブに計算する値と照合する）。パネル固有R²は
   plmにもstatsmodelsにも無く、手計算（2）だけが参照になる。
4. **拒否する入力・失敗する入力**: Amemiyaはwithin変動の無い列があると`ValidationError`
   （plmは不均衡で拒否、バランスでは時間不変の列を無視した値を返す。本実装は一律に拒否する）。
   σ_ε²が正でない・1段目のプールドOLSが特異なら`ComputationError`。
5. **plmと値が異なる入力**: 全個体の観測数が同じで時点がずれた1-wayは、本実装が観測数だけで
   閉形式を選ぶのに対し、plmは時点集合も同じことを要求してトレース形を選ぶ。差そのものを固定する。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
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
    ALL_COV_TYPES,
    AMEMIYA_REJECTED_CASES,
    CASES,
    HAUSMAN_KEY,
    WAGEPAN_CSV,
    WAGEPAN_TREND,
    Case,
)
from benchmark.panel.references.statsmodels_ref import run_re_cluster

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


# plmと値が異なる入力（`differs_from_plm`）はplmとの一致を比べない（専用のテスト）。
PLM_CASES = [case for case in CASES if not case.differs_from_plm]
PLM_PARAMS = [
    pytest.param(case, method, cov_type, id=f"{case.name}-{method}-{cov_type}")
    for case in PLM_CASES
    for method in case.methods
    for cov_type in case.cov_types
]
# 手計算との照合は全ケース。plmのフィクスチャが持たない`cov_type`（代表ケース以外のhc1〜hc3・dk）
# も、手計算なら全ケースで比べられる。
HAND_PARAMS = [
    pytest.param(case, method, cov_type, id=f"{case.name}-{method}-{cov_type}")
    for case in CASES
    for method in case.methods
    for cov_type in ALL_COV_TYPES
]
METHOD_PARAMS = [
    pytest.param(case, method, id=f"{case.name}-{method}")
    for case in PLM_CASES
    for method in case.methods
]
ONE_WAY_METHOD_PARAMS = [
    pytest.param(case, method, id=f"{case.name}-{method}")
    for case in PLM_CASES
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


@pytest.mark.parametrize(("case", "method", "cov_type"), HAND_PARAMS)
def test_matches_hand_computed(case, method, cov_type):
    """全統計量が、仕様どおりに書いた手計算（`_re_oracle.py`。分散成分はplmのソースどおりに
    観測ごとの行列で書いた独立な実装）と全ケース×全`cov_type`で一致する（plmのフィクスチャが
    全`cov_type`を持つのは代表ケースだけ）。"""
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


# ── statsmodelsとの照合（plmの重みで準偏差変換したデータへのOLS） ──────


def _group_mean(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """グループ平均を各観測に割り当てた配列。"""
    return (np.bincount(codes, weights=values) / np.bincount(codes))[codes]


def _plm_transformed(case: Case, entry: dict, df: pl.DataFrame):
    """plmの分散成分・重みで準偏差変換した`y`・各説明変数と切片の列。

    1-wayは`σ_ε²`・`σ_u²`から`θ_i = 1 - sqrt(σ_ε² / (T_i σ_u² + σ_ε²))`、
    2-wayはplmの`θ₁`・`θ₂`・`θ₃`（`y - θ₁ȳᵢ. - θ₂ȳ.ₜ + θ₃ȳ..`）。切片の列は変換した定数1。
    """
    entity = np.unique(df[case.entity_col].to_numpy(), return_inverse=True)[1]
    if case.two_way:
        time = np.unique(df[case.time_col].to_numpy(), return_inverse=True)[1]
        th1, th2, th3 = (
            entry["theta_entity"],
            entry["theta_time"],
            entry["theta_total"],
        )

        def transform(v: np.ndarray) -> np.ndarray:
            return (
                v
                - th1 * _group_mean(v, entity)
                - th2 * _group_mean(v, time)
                + th3 * v.mean()
            )

        intercept = np.full(df.height, 1.0 - th1 - th2 + th3)
    else:
        t_i = np.bincount(entity)[entity]
        theta = 1.0 - np.sqrt(
            entry["sigma2_eps"]
            / (t_i * entry["sigma2_u"] + entry["sigma2_eps"])
        )

        def transform(v: np.ndarray) -> np.ndarray:
            return v - theta * _group_mean(v, entity)

        intercept = 1.0 - theta
    y = transform(df[case.y_col].to_numpy().astype(float))
    columns = {c: transform(df[c].to_numpy().astype(float)) for c in case.x}
    return y, intercept, columns, df[case.entity_col].to_list()


@pytest.mark.parametrize(("case", "method"), METHOD_PARAMS)
def test_matches_statsmodels_on_the_plm_transformed_data(
    reference, case, method
):
    """plmの重みで準偏差変換した`y`・`X`（切片込み）へのOLSが、statsmodelsのネイティブな
    計算で出す値と一致する: 対数尤度・AIC・BIC（classical）、クラスターロバストの標準誤差・
    t値・p値・信頼区間・傾き係数の同時F検定と推論の自由度`G-1`（cluster）。"""
    entry = _plm_entry(reference, case)[method]["classical"]
    y, intercept, columns, entity = _plm_transformed(case, entry, _frame(case))
    tol = _tolerances(case)

    ols = sm.OLS(y, np.column_stack([intercept, *columns.values()])).fit()
    classical = _fit(case, "classical", method)
    for ours, theirs, name in [
        (classical.log_likelihood, ols.llf, "log_likelihood"),
        (classical.aic, ols.aic, "aic"),
        (classical.bic, ols.bic, "bic"),
    ]:
        assert_close(
            ours,
            float(theirs),
            f"{case.name}/{method}/{name}",
            rtol=tol["rtol"],
            atol=tol["atol"],
        )

    ref = run_re_cluster(
        {
            "y": y,
            "x": {"(Intercept)": intercept, **columns},
            "entity": entity,
        }
    )
    cluster = _fit(case, "cluster", method)
    assert cluster.stat_df == ref["df_resid_inference"]
    assert_stats_close(
        result_stats(cluster),
        ref,
        f"{case.name}/{method}/cluster",
        keys=[
            "params",
            "std_errors",
            "test_stats",
            "p_values",
            "conf_int",
            "f_statistic",
            "f_p_value",
            "f_df_denom",
        ],
        **tol,
    )


# ── 推定法の違い・既定値 ─────────────────────────────────────────


@pytest.mark.parametrize(
    "case_name", ["ow_baseline", "ow_unbalanced", "tw_baseline"]
)
def test_default_variance_method_is_swamy_arora(case_name):
    """既定は`swamy_arora`で、明示した場合と同じ値（不均衡の1-wayでは`plm`の式ではなく
    従来どおりの`linearmodels`準拠の式のまま）。"""
    case = next(c for c in CASES if c.name == case_name)
    default = RE(
        _frame(case),
        y=case.y_col,
        x=list(case.x),
        entity=case.entity_col,
        options=REOptions(
            cov_type="classical", time=case.time_col if case.two_way else None
        ),
    ).fit()
    explicit = _fit(case, "classical", "swamy_arora")
    assert default.variance_method == "swamy_arora"
    assert explicit.params == default.params
    assert explicit.sigma2_eps == default.sigma2_eps
    assert explicit.sigma2_u == default.sigma2_u


@pytest.mark.parametrize("case_name", ["ow_baseline", "ow_unbalanced"])
def test_the_three_methods_give_different_variance_components(case_name):
    case = next(c for c in CASES if c.name == case_name)
    sigma = {
        m: (r.sigma2_eps, r.sigma2_u)
        for m in ("swamy_arora", "wallace_hussain", "amemiya")
        for r in [_fit(case, "classical", m)]
    }
    assert len(set(sigma.values())) == 3, sigma


@pytest.mark.parametrize(
    "value", ["WALLACE_HUSSAIN", "Amemiya", "Swamy_Arora"]
)
def test_variance_method_is_case_insensitive(value):
    case = CASES[0]
    res = _fit(case, "classical", value)
    assert res.variance_method == value.lower()


def test_unknown_variance_method_raises_with_the_original_value():
    """メッセージには小文字化した値ではなく、渡された値をそのまま載せる。"""
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_VARIANCE_METHOD, value="Nerlove"),
    ):
        _fit(CASES[0], "classical", "Nerlove")


def test_unknown_variance_method_is_reported_before_the_data_is_checked():
    """未知の値は列の検証より前に拒否する（存在しない列があっても未知の値のエラーになる）。"""
    case = CASES[0]
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_VARIANCE_METHOD, value="nerlove"),
    ):
        RE(
            _frame(case),
            y="y",
            x=["no_such_column"],
            entity="entity",
            options=REOptions(variance_method="nerlove"),
        ).fit()


def test_variance_method_must_be_a_string():
    with pytest.raises(TypeError):
        REOptions(variance_method=1)
    options = REOptions()
    with pytest.raises(TypeError):
        options.variance_method = 1


# ── 分散成分が0に切り上げられる入力 ──────────────────────────────


def _case(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


@pytest.mark.parametrize(
    "case_name", ["ow_no_entity_effect", "ow_no_entity_effect_unbalanced"]
)
@pytest.mark.parametrize("method", ["wallace_hussain", "amemiya"])
def test_clipped_entity_variance_makes_one_way_gls_the_pooled_ols(
    case_name, method
):
    """個体効果の無いDGPで`σ_u²`が負になり0に切り上げられる（plmの値は`test_matches_plm`）。
    全個体でθ=0になり、GLSはプールドOLSに退化する（閉形式・トレース形の両方の切り上げ）。"""
    case = _case(case_name)
    res = _fit(case, "classical", method)
    assert res.sigma2_u == 0.0
    assert set(res.theta.values()) == {0.0}
    df = _frame(case)
    pooled = sm.OLS(
        df[case.y_col].to_numpy(),
        sm.add_constant(df.select(list(case.x)).to_numpy()),
    ).fit()
    for got, want in zip(res.params.values(), pooled.params, strict=True):
        assert_close(
            got, float(want), f"{case_name}/{method}", rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("method", ["wallace_hussain", "amemiya"])
def test_clipped_time_variance_drops_the_time_weight(method):
    """時点効果の無いDGPで`σ_λ²`が0に切り上げられ、時点方向の重み`θ₂`が0になる。"""
    res = _fit(_case("tw_no_time_effect"), "classical", method)
    assert res.sigma2_lambda == 0.0
    assert res.theta_time == 0.0
    assert res.sigma2_u > 0.0


def test_clipped_variances_in_both_directions_make_two_way_gls_the_pooled_ols():
    case = _case("tw_no_entity_effect")
    res = _fit(case, "classical", "wallace_hussain")
    assert (res.sigma2_u, res.sigma2_lambda) == (0.0, 0.0)
    df = _frame(case)
    pooled = sm.OLS(
        df[case.y_col].to_numpy(),
        sm.add_constant(df.select(list(case.x)).to_numpy()),
    ).fit()
    for got, want in zip(res.params.values(), pooled.params, strict=True):
        assert_close(
            got, float(want), "tw_no_entity_effect", rtol=RTOL, atol=ATOL
        )


# ── 観測数が等しく時点がずれたパネル（plmと値が異なる既知の差） ────


@pytest.mark.parametrize("method", ["wallace_hussain", "amemiya"])
def test_equal_counts_with_shifted_periods_follow_the_closed_form_unlike_plm(
    reference, method
):
    """1-wayは観測数だけでバランスを判定するため閉形式を使う。plmは時点集合も同じことを
    要求し、このパネルをバランスとみなさずトレース項の式を使う。本実装は手計算の閉形式と、
    plmは手計算のトレース項の式と一致し、2つの値は一致しない（意図した差）。"""
    case = _case("ow_equal_counts_shifted_time")
    plm = _plm_entry(reference, case)[method]["classical"]
    res = _fit(case, "classical", method)
    panel = _Panel(
        _frame(case), list(case.x), case.entity_col, case.time_col, case.y_col
    )
    closed = first_stage_variance_components(panel, method, False)
    trace = first_stage_variance_components(
        panel, method, False, force_trace=True
    )
    for ours, hand, theirs, name in [
        (res.sigma2_eps, closed[0], plm["sigma2_eps"], "sigma2_eps"),
        (res.sigma2_u, closed[1], plm["sigma2_u"], "sigma2_u"),
    ]:
        assert_close(
            ours, hand, f"{method}/{name}/closed", rtol=RTOL, atol=ATOL
        )
        assert_close(
            theirs,
            trace[0] if name == "sigma2_eps" else trace[1],
            f"{method}/{name}/plm",
            rtol=RTOL,
            atol=ATOL,
        )
        assert abs(ours - theirs) > 1e-3 * abs(theirs), (name, ours, theirs)


# ── 拒否する入力・失敗する入力 ───────────────────────────────────


@pytest.mark.parametrize("case", AMEMIYA_REJECTED_CASES, ids=lambda c: c.name)
def test_amemiya_rejects_regressors_without_within_variation(case):
    """時間不変の列（2-wayでは個体間・時点間でのみ変わる列）があるAmemiyaは、
    バランスパネルでも不均衡でも、全列がそうでも`ValidationError`で、列の名前を示す。"""
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.AMEMIYA_NEEDS_WITHIN_VARIATION, columns=case.rejected_columns
        ),
    ):
        _fit(case, "classical", "amemiya")


def test_amemiya_accepts_a_regressor_that_varies_only_over_time():
    """時点間でのみ変わる列は個体内で変動するため、1-wayのAmemiyaで使える。"""
    case = _case("ow_time_only")
    assert "amemiya" in case.methods
    assert math.isfinite(_fit(case, "classical", "amemiya").sigma2_u)


@pytest.mark.parametrize("method", ["swamy_arora", "wallace_hussain"])
def test_methods_other_than_amemiya_accept_time_invariant_regressors(method):
    case = _case("ow_time_invariant_balanced")
    assert math.isfinite(_fit(case, "classical", method).sigma2_u)


def test_wallace_hussain_pooled_regression_failure_is_a_computation_error():
    """時間不変の列とその定数倍（内部のFEには渡らない）でプールドOLSが特異になる。"""
    case = _case("ow_time_invariant_balanced")
    df = _frame(case).with_columns(z2=2.0 * pl.col("x_invariant"))
    with pytest.raises(
        ComputationError, match=escaped(msgs.POOLED_REGRESSION_FAILED)
    ):
        RE(
            df,
            y="y",
            x=["x1", "x_invariant", "z2"],
            entity="entity",
            options=REOptions(variance_method="wallace_hussain"),
        ).fit()


def test_non_positive_sigma2_eps_is_a_computation_error():
    """不均衡パネルのWallace-Hussainは、トレース形の連立が負の`σ_ε²`を返しうる
    （観測数2・6・2のパネル）。plmは0に切り上げてθがNaNになるが、本実装は拒否する。
    同じデータでAmemiya・Swamy-Aroraは推定できる。"""
    df = pl.DataFrame(
        {
            "entity": [0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
            "x1": [2.0, 1.0, 0.0, -2.0, -1.0, -3.0, -3.0, -3.0, -2.0, 2.0],
            "y": [0.0, 1.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 6.0, 7.0],
        }
    )

    def fit(method: str):
        return RE(
            df,
            y="y",
            x=["x1"],
            entity="entity",
            options=REOptions(variance_method=method),
        ).fit()

    pattern = re.escape(
        msgs.NON_POSITIVE_IDIOSYNCRATIC_VARIANCE.format(
            sigma2_eps="@@", method="wallace_hussain"
        )
    ).replace("@@", r"-\d+\.\d+")
    with pytest.raises(ComputationError, match=pattern):
        fit("wallace_hussain")
    assert math.isfinite(fit("amemiya").sigma2_u)
    assert math.isfinite(fit("swamy_arora").sigma2_u)


@pytest.mark.parametrize("method", ["swamy_arora", "amemiya"])
def test_noiseless_data_is_a_computation_error_when_the_first_stage_is_within(
    method,
):
    """ノイズの無い線形DGPはwithin残差が（ほぼ）0で`σ_ε²`が潰れ、準偏差変換後の設計行列が
    特異になる。1段目がwithin推定の`swamy_arora`・`amemiya`が対象（`wallace_hussain`の
    1段目はプールドOLSで、個体効果が残差に残るため`σ_ε²`は潰れない）。"""
    case = _case("ow_baseline")
    df = _frame(case)
    exact = df.with_columns(
        y=2.0 * pl.col("x1") - pl.col("x2") + pl.col("entity").rank("dense")
    )
    with pytest.raises(ComputationError):
        _fit(case, "classical", method, exact)


@pytest.mark.parametrize(
    "method", ["swamy_arora", "wallace_hussain", "amemiya"]
)
def test_perfect_collinearity_is_a_computation_error_for_every_method(method):
    """within変動のある列が完全共線なら、どの推定法でも内部のFE推定が特異になる。"""
    case = _case("ow_baseline")
    df = _frame(case).with_columns(x3=2.0 * pl.col("x1") + 3.0 * pl.col("x2"))
    with pytest.raises(ComputationError):
        RE(
            df,
            y="y",
            x=["x1", "x2", "x3"],
            entity="entity",
            options=REOptions(variance_method=method),
        ).fit()


@pytest.mark.parametrize(
    "method", ["swamy_arora", "wallace_hussain", "amemiya"]
)
@pytest.mark.parametrize("cov_type", ["classical", "cluster"])
def test_extreme_scale_difference_is_a_computation_error_for_every_method(
    method, cov_type
):
    """変数間のスケールが極端に異なる設計行列（`x1`を1e6倍、`x2`を1e-3倍）。"""
    case = _case("ow_baseline")
    df = _frame(case).with_columns(
        x1=pl.col("x1") * 1e6, x2=pl.col("x2") * 1e-3
    )
    with pytest.raises(ComputationError):
        RE(
            df,
            y="y",
            x=["x1", "x2"],
            entity="entity",
            options=REOptions(cov_type=cov_type, variance_method=method),
        ).fit()


def test_entity_count_equal_to_slope_count_plus_one_blocks_only_swamy_arora():
    """個体数が傾き係数の数+1（自由度ちょうど0のbetween回帰）では、between回帰を使う
    `swamy_arora`だけが`ValidationError`になり、1段目に回帰を使うだけの`wallace_hussain`・
    `amemiya`は推定できる（手計算の分散成分と一致）。"""
    rng = np.random.default_rng(3)
    entity = np.repeat(np.arange(3), 5)
    x1, x2 = rng.normal(size=15), rng.normal(size=15)
    y = 1 + x1 + x2 + rng.normal(size=15) + np.repeat(rng.normal(size=3), 5)
    df = pl.DataFrame(
        {
            "entity": entity,
            "time": np.tile(np.arange(5), 3),
            "x1": x1,
            "x2": x2,
            "y": y,
        }
    )

    def fit(method: str):
        return RE(
            df,
            y="y",
            x=["x1", "x2"],
            entity="entity",
            options=REOptions(cov_type="classical", variance_method=method),
        ).fit()

    with pytest.raises(ValidationError, match="between-regression"):
        fit("swamy_arora")
    panel = _Panel(df, ["x1", "x2"], "entity", "time", "y")
    for method in ("wallace_hussain", "amemiya"):
        res = fit(method)
        eps, u, _ = first_stage_variance_components(panel, method, False)
        assert_close(res.sigma2_eps, eps, method, rtol=RTOL, atol=ATOL)
        assert_close(res.sigma2_u, u, method, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "method", ["swamy_arora", "wallace_hussain", "amemiya"]
)
def test_singleton_entity_is_a_validation_error_for_every_method(method):
    """内部のFE推定が全推定法で作られるため、観測数1の個体は全推定法で拒否される。"""
    case = _case("ow_baseline")
    df = _frame(case)
    kept = df.filter((pl.col("entity") != "e00") | (pl.col("time") == "t0"))
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.SINGLETON_GROUP, dimension="entity", group_id="e00"
        ),
    ):
        _fit(case, "classical", method, kept)


@pytest.mark.parametrize("method", ["wallace_hussain", "amemiya"])
def test_two_way_unbalanced_panel_still_raises(method):
    case = _case("tw_baseline")
    df = _frame(case).slice(1)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.UNBALANCED_PANEL_FOR_TWO_WAY,
            n_obs=df.height,
            n_entities=40,
            n_periods=6,
            expected=240,
        ),
    ):
        _fit(case, "classical", method, df)


def test_variance_method_does_not_change_the_hausman_degrees_of_freedom():
    """within推定とハウスマン検定の対象列は`variance_method`によらない。"""
    case = _case("ow_baseline")
    dfs = {m: _fit(case, "classical", m).hausman_df for m in case.methods}
    assert set(dfs.values()) == {2}
