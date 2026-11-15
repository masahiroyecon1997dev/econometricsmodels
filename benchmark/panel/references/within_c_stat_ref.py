"""FE-GMM（`FEIVResults.c_stat()`）の参照: within変換後のデータへのC統計量。

固定効果をwithin変換で除いた列（`within_gmm_ref.within_columns`。本実装とは別の計算）に、定数項なしの
C統計量（difference-in-Hansen）を当てる。参照は2つ。

- 主: numpyで書いた独立実装（`benchmark/iv/references/c_statistic_oracle.py`。IV本体の`c_stat()`の
  参照で、linearmodelsの`c_stat`と照合済み）。
- 独立クロスチェック: R `momentfit`の部品で`S`・推定・Hansen Jを作る
  （`benchmark/iv/references/run_momentfit_c_stat.R`）。

どちらも`S`は`n`で割らない生の和で、`classical`は`σ̂² = e'e/n`。本実装は`classical`重みのとき`S`を
`n/(n − n_absorbed)`倍する（`σ̂² = e'e/(n − n_absorbed)`）ので、参照の`J_e`・`J_c`・`C`に
`(n − n_absorbed)/n`を掛けた値と比べる（検定は同じ自由度のχ²）。`robust`・`cluster`重みは補正しない。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from scipy import stats

from benchmark.iv.references.c_statistic_oracle import c_statistic
from benchmark.iv.references.r import run_momentfit_c_stat_r
from benchmark.panel.fe_iv_gmm_cases import CStatCase
from benchmark.panel.references.within_gmm_ref import within_columns


def _scaled(
    statistic: float,
    j_with: float,
    j_without: float,
    df_test: int,
    scale: float,
) -> dict:
    stat = statistic * scale
    return {
        "statistic": stat,
        "p_value": float(stats.chi2.sf(stat, df_test)),
        "df": df_test,
        "j_with_tested": j_with * scale,
        "j_without_tested": j_without * scale,
    }


def _absorbed_scale(df: pd.DataFrame, case: CStatCase) -> float:
    """`classical`重みのときだけ`(n − n_absorbed)/n`（本実装の`S`の再スケール）。"""
    base = case.base
    n_absorbed = sum(df[e].nunique() for e in base.effects) - (
        len(base.effects) - 1
    )
    return (
        (len(df) - n_absorbed) / len(df) if case.weight == "classical" else 1.0
    )


def run_within_c_stat(df: pd.DataFrame, case: CStatCase) -> dict:
    """numpyのオラクル（within変換後のデータ）によるC統計量。"""
    base = case.base
    within = within_columns(df, case)
    result = c_statistic(
        within["y"].to_numpy(),
        within[list(base.x_exog)].to_numpy().reshape(len(df), -1),
        within[list(base.x_endog)].to_numpy(),
        within[list(base.instruments)].to_numpy(),
        tested_endog=[base.x_endog.index(v) for v in case.variables],
        tested_instruments=[
            base.instruments.index(v) for v in case.instruments
        ],
        weight=case.weight,
        groups=df[case.cluster].to_numpy() if case.cluster else None,
    )
    return _scaled(
        result.statistic,
        result.j_with_tested,
        result.j_without_tested,
        result.df,
        _absorbed_scale(df, case),
    )


def run_within_c_stat_momentfit(df: pd.DataFrame, case: CStatCase) -> dict:
    """R `momentfit`の部品（within変換後のデータ）によるC統計量。"""
    base = case.base
    within = within_columns(df, case)
    data = within.copy()
    if case.cluster:
        data[case.cluster] = df[case.cluster].to_numpy()
    slopes = [*base.x_exog, *base.x_endog]
    full = [*base.x_exog, *case.variables, *base.instruments]
    formula = f"y ~ 0 + {' + '.join(slopes)} | 0 + {' + '.join(full)}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "within.csv"
        data.to_csv(path, index=False)
        raw = run_momentfit_c_stat_r(
            path,
            formula,
            case.weight,
            [*case.variables, *case.instruments],
            cluster=case.cluster,
        )
    return _scaled(
        raw["statistic"],
        raw["j_with_tested"],
        raw["j_without_tested"],
        int(raw["df"]),
        _absorbed_scale(df, case),
    )
