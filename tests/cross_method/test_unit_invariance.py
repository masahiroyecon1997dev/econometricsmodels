"""1列の単位を極端に変えても、全手法で推定が成功し、統計量が単位を揃えたデータと一致すること。

全体のWald/F検定は、傾き係数の共分散部分行列を相関スケール化（対角を1にそろえる）して
悪条件を判定し二次形式を計算する。係数の標準誤差の比（列の単位の比）が倍精度の閾値を
超えるだけでは失敗しない。1列を`1e9`倍・`1e-9`倍して、単位を揃えたデータと比べる。

ほぼ共線な傾き係数（単位を揃えても悪条件）の受け入れ境界は`test_collinearity_boundary.py`、
その範囲の統計量の精度は`test_near_collinear_exact.py`が固定する。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _errors import raises_error
from _helpers import assert_unit_change_invariant
from econometricsmodels import (
    FE,
    FEIV,
    IV,
    OLS,
    RE,
    WLS,
    ErrorCode,
    FEOptions,
    IVOptions,
    Logit,
    OLSOptions,
    Probit,
    Tobit,
)

N_ENTITIES = 60
N_PERIODS = 8
N = N_ENTITIES * N_PERIODS


def _data() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    x1, x2 = rng.normal(size=N), rng.normal(size=N)
    z1, z2, u = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
    endog = 0.7 * z1 + 0.5 * z2 + 0.5 * u + rng.normal(size=N)
    alpha = np.repeat(rng.normal(size=N_ENTITIES), N_PERIODS)
    y = 1 + x1 + 0.5 * x2 + alpha + rng.normal(size=N)
    return pl.DataFrame(
        {
            "entity": entity,
            "y": y,
            "ybin": (y + rng.logistic(size=N) > 1).astype(float),
            "ycens": np.maximum(y - 1, 0.0),
            "iv_y": 1 + x1 + endog + u + alpha,
            "w": rng.uniform(0.5, 2.0, size=N),
            "x1": x1,
            "x2": x2,
            "endog": endog,
            "z1": z1,
            "z2": z2,
        }
    )


_WALD = ("wald_statistic", "wald_p_value")
_F = ("f_statistic", "f_p_value")
_HAUSMAN = ("hausman_statistic", "hausman_p_value")
_WU_HAUSMAN = ("wu_hausman_statistic", "wu_hausman_p_value")
_IV_COLUMNS = {
    "x_exog": ["x1"],
    "x_endog": ["endog"],
    "instruments": ["z1", "z2"],
}

# 1e9倍・1e-9倍での実測の最大相対差は全手法・全統計量・係数・標準誤差で2e-13（GMM・Tobitも
# 同程度）。その約3桁上に許容誤差を置き、単位に依る小さな退行も見逃さない。
RTOL = 1e-10

# 名前 -> (推定する関数、比べる統計量)。全体のWald/F統計量に加え、同じ`wald_f_test`を使う
# ハウスマン検定（RE）とWu-Hausman検定（2SLS・FEIV）も比べる。ロジット・プロビットは
# 全体のWald検定を持たない（係数と標準誤差だけ）。
CASES = {
    "ols": (lambda d: OLS(d, "y", ["x1", "x2"]).fit(), _F),
    "ols_hc1": (
        lambda d: OLS(
            d, "y", ["x1", "x2"], options=OLSOptions(cov_type="hc1")
        ).fit(),
        _F,
    ),
    "wls": (lambda d: WLS(d, "y", ["x1", "x2"], weight="w").fit(), _F),
    "fe": (
        lambda d: FE(d, y="y", x=["x1", "x2"], effects=["entity"]).fit(),
        _F,
    ),
    "fe_classical": (
        lambda d: FE(
            d,
            y="y",
            x=["x1", "x2"],
            effects=["entity"],
            options=FEOptions(cov_type="classical"),
        ).fit(),
        _F,
    ),
    "re": (
        lambda d: RE(d, y="y", x=["x1", "x2"], effects=["entity"]).fit(),
        (*_F, *_HAUSMAN),
    ),
    "logit": (lambda d: Logit(d, y="ybin", x=["x1", "x2"]).fit(), ()),
    "probit": (lambda d: Probit(d, y="ybin", x=["x1", "x2"]).fit(), ()),
    "tobit": (lambda d: Tobit(d, y="ycens", x=["x1", "x2"]).fit(), _WALD),
    "iv_2sls": (
        lambda d: IV(d, y="iv_y", **_IV_COLUMNS).fit(),
        (*_WALD, *_WU_HAUSMAN),
    ),
    "iv_gmm": (
        lambda d: IV(
            d,
            y="iv_y",
            **_IV_COLUMNS,
            options=IVOptions(estimator="gmm"),
        ).fit(),
        _WALD,
    ),
    "fe_iv": (
        lambda d: FEIV(d, y="iv_y", **_IV_COLUMNS, effects=["entity"]).fit(),
        (*_WALD, *_WU_HAUSMAN),
    ),
    "fe_iv_gmm": (
        lambda d: FEIV(d, y="iv_y", **_IV_COLUMNS, effects=["entity"]).fit(),
        _WALD,
    ),
}


@pytest.mark.parametrize("scale", [1e9, 1e-9])
@pytest.mark.parametrize("name", sorted(CASES))
def test_one_column_in_other_units_gives_the_same_statistics(name, scale):
    fit, stats = CASES[name]
    df = _data().with_columns(pl.col("x1") * scale)
    assert_unit_change_invariant(
        fit, df, {"x1": scale}, stats=stats, rtol=RTOL
    )


# ── 単位を揃えても悪条件のとき ──────────────────────────────────────────
#
# ほぼ共線な傾きでの統計量・係数・標準誤差の厳密値との照合は`test_near_collinear_exact.py`、
# 受け入れ境界は`test_collinearity_boundary.py`。ここには、共分散が実際に特異なときに
# 止まることだけを置く。


def test_a_singular_cluster_meat_raises_near_singular_matrix():
    """クラスター数が`g = q + 1`（4クラスター・傾き3個）の境界で、直交座標のミートの傾き
    部分行列の最小固有値が最大の約`1e-17`倍（実質0）になるデータは、`g > q`で事前検証は通るが、
    全体のWald検定が`MATRIX_NEAR_SINGULAR`（`COEFFICIENT_COVARIANCE`）で止める。桁落ちではなく、
    共分散が実際に特異な場合に止まることの確認で、共線性の程度には依らない。"""
    n = 12
    i = np.arange(n)
    x1 = (i + 1).astype(float)
    df = pl.DataFrame(
        {
            "x1": x1,
            "x2": x1**2,
            "x3": (i % 3).astype(float),
            "g": [f"g{k % 4}" for k in i],
            "y": 1.0
            + 2.0e6 * x1
            + 3.0e-3 * x1**2
            + 0.5 * (i % 3)
            + np.where(i % 2 == 0, 1e3, -1e3),
        }
    )
    options = OLSOptions(cov_type="cluster", cluster="g")
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR, matrix="COEFFICIENT_COVARIANCE"
    ):
        OLS(df, "y", ["x1", "x2", "x3"], options=options).fit()
