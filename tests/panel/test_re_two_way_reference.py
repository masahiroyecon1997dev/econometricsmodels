"""2-way RE（`REOptions.time`、バランスパネル）のplm・手計算との数値照合。

2-way REを計算する参照実装は`plm`（`model = "random"`・`effect = "twoways"`・Swamy-Arora）
だけで（`linearmodels.RandomEffects`は1-wayのみ）、plmとの照合が主になる
（単一リファレンスの例外、`docs/guide/verification.md`）。構造・API・バリデーションの
テストは`test_re_two_way.py`。

1. **plm**（`re_two_way.json`、`benchmark/panel/fixtures/generate_re_two_way_fixtures.py`）:
   全`cov_type`（classical・HC1〜HC3・cluster・Driscoll-Kraay）×合成データのシナリオ
   （一次元でしか変わらない列・絶対年の列・小パネル・不均一分散・系列相関・T=25の
   パネル等）＋実データ（wagepan）で、係数・標準誤差・t値・p値・信頼区間・F統計量・
   F検定のp値・自由度・分散成分（σ_ε²・σ_u²・σ_λ²）・3つの重み・ハウスマン検定を比べる。
   対数尤度・AIC・BIC・パネル固有R²はplmが持たず、plmの係数からの手計算（独立性は限定的）
   との比較になる。
2. **仕様どおりの独立な手計算**（`_re_oracle.py`）: numpy・statsmodels・scipyだけで
   `re-spec.md`3章を書き下した値と、plmのフィクスチャを持たない入力（説明変数が
   すべて一次元でしか変わらない場合。plmは`empty model`で失敗する）を含めて比べる。
3. **statsmodels**: plmの重みで変換した`y`・`X`の回帰（切片込み）の対数尤度・AIC・BICと、
   本実装の値を比べる（plmが持たない値を、手計算ではなくstatsmodelsがネイティブに計算する値と
   照合する）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from _assertions import assert_close
from _constants import DATA_DIR
from _helpers import load_wooldridge_dataset
from _re_oracle import assert_stats_close, hand_computed, result_stats
from _tolerances import TOLERANCES
from econometricsmodels import RE, REOptions

from benchmark.panel.fixtures.generate_re_two_way_fixtures import (
    COV_TYPES,
    SMALL_PANEL_CASES,
    SMALL_PANEL_CSV,
    SYNTHETIC_CASES,
    WAGEPAN_TREND,
    WAGEPAN_X_TWO_WAY,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_two_way.json"
)

_TOL = TOLERANCES["re_two_way"]
RTOL = _TOL["rtol"]
ATOL = _TOL["atol"]
RTOL_P_VALUE = _TOL["rtol_p_value"]
RTOL_ABSOLUTE_YEAR = _TOL["rtol_absolute_year"]


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@dataclass(frozen=True)
class Case:
    name: str
    scenario: str | None  # 合成データのシナリオ名（実データはNone）
    x_cols: list[str]
    absolute_year: bool = False
    in_fixture: bool = True  # plmのフィクスチャを持つか

    @property
    def y_col(self) -> str:
        return "lwage" if self.scenario is None else "y"

    @property
    def entity_col(self) -> str:
        return "nr" if self.scenario is None else "entity"

    @property
    def time_col(self) -> str:
        return "year" if self.scenario is None else "time"

    def frame(self) -> pl.DataFrame:
        if self.scenario is None:
            return load_wooldridge_dataset("wagepan").with_columns(
                pl.col("year").alias(WAGEPAN_TREND)
            )
        if self.scenario == SMALL_PANEL_CSV:
            return pl.read_csv(DATA_DIR / SMALL_PANEL_CSV)
        return pl.read_csv(DATA_DIR / f"fe_{self.scenario}.csv")

    def plm(self, reference: dict) -> dict:
        if self.scenario is None:
            return reference["wooldridge"]["wagepan"]
        return reference["synthetic"][self.name]

    def fit(self, cov_type: str):
        return RE(
            self.frame(),
            y=self.y_col,
            x=self.x_cols,
            entity=self.entity_col,
            options=self.options(cov_type),
        ).fit()

    def options(self, cov_type: str) -> REOptions:
        dk_time = self.time_col if cov_type == "dk" else None
        return REOptions(
            cov_type=cov_type, time=self.time_col, dk_time=dk_time
        )

    def oracle(self, cov_type: str, **kwargs) -> dict:
        return hand_computed(
            self.frame(),
            self.x_cols,
            cov_type,
            two_way=True,
            entity_col=self.entity_col,
            time_col=self.time_col,
            y_col=self.y_col,
            **kwargs,
        )

    def tolerances(self) -> dict:
        rtol = RTOL_ABSOLUTE_YEAR if self.absolute_year else RTOL
        return {
            "rtol": rtol,
            "atol": ATOL,
            "rtol_p_value": RTOL_P_VALUE if not self.absolute_year else rtol,
        }


PLM_CASES = (
    [
        Case(
            name,
            scenario,
            x_cols,
            absolute_year=scenario == "absolute_year_regressor",
        )
        for name, (scenario, x_cols) in SYNTHETIC_CASES.items()
    ]
    + [
        Case(name, SMALL_PANEL_CSV, x_cols)
        for name, x_cols in SMALL_PANEL_CASES.items()
    ]
    + [Case("wagepan", None, WAGEPAN_X_TWO_WAY, absolute_year=True)]
)

# plmのフィクスチャを持たない入力（説明変数がすべて一次元でしか変わらない場合や、
# 絶対年だけの回帰。plmは`empty model`・推定不能で失敗する）。手計算とだけ比べる。
ORACLE_ONLY_CASES = [
    Case(
        f"one_dimensional_{'_'.join(x)}",
        "one_dimensional_regressors",
        x,
        in_fixture=False,
    )
    for x in (["xi", "xt"], ["xt"], ["xi"], ["xt", "xi"])
] + [
    Case(
        f"absolute_year_{'_'.join(x)}",
        "absolute_year_regressor",
        x,
        absolute_year=True,
        in_fixture=False,
    )
    for x in (["year"], ["year", "x1"])
]

# `hausman_*`・`r_squared_within`を持たない（within変動のある列が無い）入力。
NO_WITHIN_VARIATION = {
    "one_dimensional_xi_xt",
    "one_dimensional_xt",
    "one_dimensional_xi",
    "one_dimensional_xt_xi",
    "absolute_year_year",
}

_PLM_KEYS = (
    "params",
    "std_errors",
    "test_stats",
    "p_values",
    "f_statistic",
    "f_p_value",
    "f_df_num",
    "f_df_denom",
    "df_resid",
    "log_likelihood",
    "aic",
    "bic",
    "r_squared_within",
    "r_squared_between",
    "r_squared_overall",
    "sigma2_eps",
    "sigma2_u",
    "sigma2_lambda",
    "theta_entity",
    "theta_time",
    "theta_total",
    "hausman_statistic",
    "hausman_p_value",
    "hausman_df",
)


def _plm_stats(entry: dict) -> dict:
    out = {key: entry[key] for key in _PLM_KEYS}
    out["conf_int"] = {k: tuple(v) for k, v in entry["conf_int"].items()}
    return out


# ── plmとの照合 ──────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("case", PLM_CASES, ids=lambda c: c.name)
def test_matches_plm(reference, case, cov_type):
    """全統計量がplmと一致する（係数・標準誤差・t・p値・信頼区間・F統計量とそのp値・
    自由度・分散成分・3つの重み・ハウスマン検定、plmが持たない対数尤度・AIC・BIC・
    パネル固有R²はplmの係数からの手計算）。"""
    res = case.fit(cov_type)
    plm = _plm_stats(case.plm(reference)[cov_type])

    assert_stats_close(
        result_stats(res),
        plm,
        f"{case.name}/{cov_type}",
        **case.tolerances(),
    )


@pytest.mark.parametrize("case", PLM_CASES, ids=lambda c: c.name)
def test_variance_components_do_not_depend_on_cov_type(reference, case):
    """分散成分・重みは`cov_type`によらず、全`cov_type`で同じplmの値になる
    （plmの出力側も同じ）。"""
    entry = case.plm(reference)
    for key in ("sigma2_eps", "sigma2_u", "sigma2_lambda", "theta_total"):
        values = {entry[cov_type][key] for cov_type in COV_TYPES}
        assert len(values) == 1, key


# ── 仕様どおりの独立な手計算との照合 ─────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize(
    "case", [*PLM_CASES, *ORACLE_ONLY_CASES], ids=lambda c: c.name
)
def test_matches_hand_computed(case, cov_type):
    """全統計量が、仕様どおりに書いた手計算（`_re_oracle.py`）と一致する。
    説明変数がすべて一次元でしか変わらない入力では、ハウスマン検定は`None`、
    within R²は0になる。"""
    res = case.fit(cov_type)
    hand = case.oracle(cov_type)

    assert_stats_close(
        result_stats(res),
        hand,
        f"{case.name}/{cov_type}",
        **case.tolerances(),
    )
    if case.name in NO_WITHIN_VARIATION:
        assert res.hausman_statistic is None
        assert res.hausman_p_value is None
        assert res.hausman_df is None
        assert res.r_squared_within == 0.0


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("case", PLM_CASES, ids=lambda c: c.name)
def test_hand_computed_with_plm_variance_components_matches_plm(
    reference, case, cov_type
):
    """plmの分散成分を手計算のGLSに与えると、plmの係数・標準誤差・ハウスマン検定と
    一致する（手計算がplmと同じ推定量を計算していることの確認）。"""
    entry = case.plm(reference)[cov_type]
    hand = case.oracle(
        cov_type,
        sigma2=(
            entry["sigma2_eps"],
            entry["sigma2_u"],
            entry["sigma2_lambda"],
        ),
    )
    plm = _plm_stats(entry)
    keys = [
        "params",
        "std_errors",
        "test_stats",
        "p_values",
        "conf_int",
        "f_statistic",
        "f_p_value",
        "hausman_statistic",
        "hausman_p_value",
        "hausman_df",
        "theta_entity",
        "theta_time",
        "theta_total",
    ]
    assert_stats_close(
        hand,
        plm,
        f"{case.name}/{cov_type}",
        keys=keys,
        **case.tolerances(),
    )


# ── statsmodelsとの照合（対数尤度・AIC・BIC） ────────────────────


@pytest.mark.parametrize("case", PLM_CASES, ids=lambda c: c.name)
def test_information_criteria_match_statsmodels_on_the_plm_transformed_data(
    reference, case
):
    """plmの重みで準偏差変換した`y`・`X`（切片は`1 - θ₁ - θ₂ + θ₃`倍）を
    statsmodelsのOLSに当てた対数尤度・AIC・BICと一致する。"""
    entry = case.plm(reference)["classical"]
    th1, th2, th3 = (
        entry["theta_entity"],
        entry["theta_time"],
        entry["theta_total"],
    )
    df = case.frame().sort(case.entity_col, case.time_col)

    def transform(values: np.ndarray) -> np.ndarray:
        frame = pl.DataFrame(
            {
                "v": values,
                "e": df[case.entity_col],
                "t": df[case.time_col],
            }
        )
        return (
            frame.select(
                pl.col("v")
                - th1 * pl.col("v").mean().over("e")
                - th2 * pl.col("v").mean().over("t")
                + th3 * pl.col("v").mean()
            )
            .to_series()
            .to_numpy()
        )

    y = transform(df[case.y_col].to_numpy())
    design = np.column_stack(
        [np.full(df.height, 1.0 - th1 - th2 + th3)]
        + [transform(df[c].to_numpy()) for c in case.x_cols]
    )
    ref = sm.OLS(y, design).fit()

    res = case.fit("classical")
    tol = case.tolerances()
    for ours, theirs, name in [
        (res.log_likelihood, ref.llf, "log_likelihood"),
        (res.aic, ref.aic, "aic"),
        (res.bic, ref.bic, "bic"),
    ]:
        assert_close(
            ours,
            float(theirs),
            f"{case.name}/{name}",
            rtol=tol["rtol"],
            atol=ATOL,
        )
