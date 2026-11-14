"""FE の3次元以上の固定効果の独立実装（linearmodels AbsorbingLS）とのクロスチェック。

主リファレンスはR fixest（`test_fe_multi_way_reference.py`）。ここでは
`tests/fixtures/benchmarks/fe_multi_way_crosscheck.json`と`fe_multi_way_cov_crosscheck.json`
（`benchmark/panel/fixtures/generate_fe_multi_way_crosscheck_fixtures.py`で生成）を読み、
`linearmodels.iv.absorbing.AbsorbingLS`（`absorb`に3〜4つのカテゴリ列、`debiased=True`）と
`MULTI_WAY_SCENARIOS`の全シナリオ×classical・hc1で、係数・標準誤差・`t`値・`p`値・信頼区間・
係数の分散共分散行列・自由度を相対誤差1e-8で厳密比較する。

`AbsorbingLS`はHC2/HC3を持たず、cluster・dkの小標本補正の規約が本実装（fixest型）と違うため、
これらはfixestだけが参照になる（`docs/guide/verification-details.md`の単一リファレンスの例外）。
`AbsorbingLS`の`rsquared`・F統計量は吸収後の別定義のため比べない。
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close, assert_inference_close
from _constants import DATA_DIR
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES
from econometricsmodels import FE, FEOptions

from benchmark.panel.fixtures.generate_fe_multi_way_crosscheck_fixtures import (
    COV_TYPES,
)
from benchmark.panel.fixtures.generate_fe_multi_way_fixtures import (
    EFFECT_COLUMNS,
)
from benchmark.panel.multi_way_datasets import MULTI_WAY_SCENARIOS

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "fe_multi_way_crosscheck.json"
)

RTOL = TOLERANCES["fe_multi_way_crosscheck"]["rtol"]
ATOL = TOLERANCES["fe_multi_way_crosscheck"]["atol"]
RTOL_P_VALUE = TOLERANCES["fe_multi_way_crosscheck"]["rtol_p_value"]
ATOL_P_VALUE = TOLERANCES["fe_multi_way_crosscheck"]["atol_p_value"]

CASES = [
    (scenario, cov_type)
    for scenario in MULTI_WAY_SCENARIOS
    for cov_type in COV_TYPES
]


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("scenario, cov_type", CASES)
def test_matches_absorbing_ls(crosscheck, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_multi_way_{scenario}.csv")
    res = FE(
        df,
        y="y",
        x=["x1", "x2"],
        effects=EFFECT_COLUMNS[scenario],
        options=FEOptions(cov_type=cov_type),
    ).fit()
    ref = crosscheck["synthetic"][scenario][cov_type]
    label = f"{scenario}/{cov_type}"

    assert_inference_close(
        res,
        ref,
        label,
        rtol=RTOL,
        atol=ATOL,
        p_value_check=partial(
            assert_dict_close, rtol=RTOL_P_VALUE, atol=ATOL_P_VALUE
        ),
    )
    for row, (ours, theirs) in enumerate(zip(res.residuals, ref["residuals"])):
        assert_close(
            ours, theirs, f"{label}/residuals/{row}", rtol=RTOL, atol=ATOL
        )
    assert res.df_resid == ref["df_resid"], label
    assert res.df_model == ref["df_model"], label
