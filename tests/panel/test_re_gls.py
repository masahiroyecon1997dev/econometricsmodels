"""不均衡パネルの1-way REの、statsmodels `GLS`とのクロスチェック。

不均衡パネルの主リファレンスは`plm`（`test_re_crosscheck.py`・`test_re_time_invariant.py`）で、
classical・hc1は`linearmodels`（`fit(small_sample=True)`、`test_re_reference.py`）とも比べる。
このファイルは、`plm`の分散成分（`ercomp`の
`sigma2`）から密な`Ω = σ_ε² I + σ_u² ZZ'`を組んだstatsmodelsの`GLS`と、係数・classicalと
個体クラスター（t分布の自由度`G-1`）の標準誤差・係数の分散共分散行列・t統計量・p値・
信頼区間・傾きの同時F検定を比べ、分散成分から
先（GLS変換・共分散・t検定）を`plm`とは独立な実装で確かめる。

statsmodelsは`Ω⁻¹`のCholesky分解で白色化し、本実装の準偏差変換（`Ω^{-1/2}`に比例）とは
個体ごとの直交変換だけ違う。classicalと個体クラスターの共分散はこの変換で変わらないが、
HC1〜HC3は観測ごとのスコアが変わるため対象外（`plm`の`vcovHC`だけが参照）。σ_u²そのものの
参照は`plm`だけで、ここでは本実装の値が`GLS`に与えた値と一致することを前提として確かめる。

フィクスチャ: `tests/fixtures/benchmarks/re_gls.json`・`re_gls_cov.json`
（`benchmark/panel/fixtures/generate_re_gls_fixtures.py`・`generate_re_gls_cov_fixtures.py`）。
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_inference_close
from _constants import DATA_DIR
from _fixture_cov import load_fixtures_with_cov
from _tolerances import TOLERANCES
from econometricsmodels import RE, REOptions

from benchmark.panel.fixtures.generate_re_gls_fixtures import (
    COV_TYPES,
    SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_gls.json"
)

_TOL = TOLERANCES["re_gls"]
RTOL = _TOL["rtol"]
ATOL = _TOL["atol"]
RTOL_P_VALUE = _TOL["rtol_p_value"]

_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)


@pytest.fixture(scope="module")
def reference() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)


def _fit(scenario: str, cov_type: str):
    csv_name, x_cols = SCENARIOS[scenario]
    df = pl.read_csv(DATA_DIR / f"{csv_name}.csv")
    return RE(
        df,
        y="y",
        x=x_cols,
        effects=["entity"],
        options=REOptions(cov_type=cov_type),
    ).fit()


def _check_p_values(ours: dict, ref: dict, label: str) -> None:
    for name, value in ref.items():
        assert_close(
            ours[name],
            value,
            f"{label}/{name}",
            rtol=RTOL_P_VALUE,
            atol=0.0,
        )


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_matches_statsmodels_gls(reference, scenario, cov_type):
    """係数・標準誤差・係数の分散共分散行列・t統計量・p値・信頼区間・傾きの同時F検定と
    推論の自由度が、plmの分散成分を与えた`GLS`と一致する。分散成分は本実装の値がGLSに
    与えた値と一致することも確かめる。"""
    res = _fit(scenario, cov_type)
    ref = reference["synthetic"][scenario]
    label = f"{scenario}/{cov_type} vs statsmodels GLS"
    _assert_close(res.sigma2_eps, ref["sigma2_eps"], f"{label}/sigma2_eps")
    _assert_close(res.sigma2_u, ref["sigma2_u"], f"{label}/sigma2_u")
    gls = ref[cov_type]
    assert_inference_close(
        res,
        gls,
        label,
        rtol=RTOL,
        atol=ATOL,
        p_value_check=_check_p_values,
    )
    assert res.stat_df == gls["df_resid_inference"], f"{label}/stat_df"
    assert res.f_df_denom == gls["f_df_denom"], f"{label}/f_df_denom"
    _assert_close(res.f_statistic, gls["f_statistic"], f"{label}/f_statistic")
    assert_close(
        res.f_p_value,
        gls["f_p_value"],
        f"{label}/f_p_value",
        rtol=RTOL_P_VALUE,
        atol=0.0,
    )
