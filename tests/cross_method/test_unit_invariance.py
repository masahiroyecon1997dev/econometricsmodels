"""1列の単位を極端に変えても、全手法で推定が成功し、統計量が単位を揃えたデータと一致すること。

全体のWald/F検定は、傾き係数の共分散部分行列を相関スケール化（対角を1にそろえる）して
悪条件を判定し二次形式を計算する。係数の標準誤差の比（列の単位の比）が倍精度の閾値を
超えるだけでは失敗しない。1列を`1e9`倍・`1e-9`倍して、単位を揃えたデータと比べる。

ほぼ共線な傾き係数（単位を揃えても悪条件）は今も`MATRIX_NEAR_SINGULAR`で止まる。
その境界は`docs/guide/validation.md`の記述と合わせて、ここで固定する。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _errors import raises_error
from _exact_wald import exact_slope_wald
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
from test_wald_ill_conditioned import _exact_wald_quadratic_form

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
# 傾きがほぼ共線なら、単位を揃えても全体のWald検定が`MATRIX_NEAR_SINGULAR`で止まる。
# 二つの失敗が別の原因で起きる。
#
# 1. 傾きの相関が倍精度の限界（`1 - 1e-15`程度）を超える本当の準共線。classicalでも起きる。
# 2. `hc1`・`cluster`のサンドイッチ共分散`B M B`の桁落ち。`B = (X'X)⁻¹`の条件数が大きいと
#    共分散が数値的に正定値でなくなり、相関が`1 - 1e-12`程度でも止まる（classicalは通る）。
#
# 2は既知の限界で、`cov_params`を入力にする`wald_f_test`では避けられない（`Rβ`と直交座標の
# ミートから二次形式を直接求める計算経路が要る）。この限界を直したら、下の「限界の固定」の
# テスト（`..._sandwich_...`）は`DID NOT RAISE`で失敗する。その場合は、そのテストを削除して、
# `test_wald_ill_conditioned.py`と同じ方式（返った`cov_params`と係数から有理数で厳密に求めた
# 二次形式との照合）に置き換える。失敗を期待するテストは、1の本当の準共線のものに残す。


def _near_collinear_ols(eps: float, groups: int = 8) -> pl.DataFrame:
    """`x2 = x1 + eps * noise`（相関は`1 - O(eps²)`）の小さなデータ。"""
    n = 40
    i = np.arange(n)
    x1 = ((i * 7) % 13) - 6.0
    x2 = x1 + eps * (((i * 5) % 11) - 5.0)
    noise = np.where(i % 2 == 0, 0.7, -0.4) + 0.1 * (i % 5)
    return pl.DataFrame(
        {
            "y": 1.0 + 2.0 * x1 + 3.0 * x2 + noise,
            "x1": x1,
            "x2": x2,
            "g": [f"g{k % groups}" for k in i],
        }
    )


def _near_collinear_panel(eps: float) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    x1, x2 = rng.normal(size=N), rng.normal(size=N)
    alpha = np.repeat(rng.normal(size=N_ENTITIES), N_PERIODS)
    y = 1 + x1 + 0.5 * x2 + alpha + rng.normal(size=N)
    x3 = x1 + eps * rng.normal(size=N)
    return pl.DataFrame({"entity": entity, "y": y, "x1": x1, "x3": x3})


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


def test_highly_correlated_slopes_with_the_classical_covariance_match_the_exact_value():
    """相関が`1 - 5e-11`（`eps=1e-5`）でも、classicalの共分散は桁落ちしにくく、全体のF統計量は
    返った`cov_params`と係数から有理数で厳密に求めた値と一致する（通るが桁違いの値を返さない）。
    二次形式は逆行列を取るため、精度は相関行列の条件数×機械イプシロン（ここで約2e10×2e-16、
    実測の相対差は数e-6）に限られる。`eps=1e-6`では条件数が約2e12になり相対8e-4まで悪化する。"""
    res = OLS(_near_collinear_ols(1e-5), "y", ["x1", "x2"]).fit()
    n_slopes = len(res.param_names) - 1
    exact = _exact_wald_quadratic_form(res) / n_slopes
    assert res.f_statistic == pytest.approx(exact, rel=1e-4)


@pytest.mark.parametrize("eps", [1e-6, 1e-7])
@pytest.mark.parametrize("cov_type", ["hc1", "cluster"])
def test_ols_near_collinear_slopes_with_a_sandwich_covariance_match_the_exact_value(
    cov_type, eps
):
    """傾きの相関が`1 - 5e-13`・`1 - 5e-15`でも、hc1・clusterの全体F統計量は、データから
    有理数で厳密に求めた値と一致する。直交座標（QR）でミートを作るため、サンドイッチ共分散の
    桁落ちの影響を受けない（以前は`MATRIX_NEAR_SINGULAR`で止まっていた）。"""
    df = _near_collinear_ols(eps)
    options = OLSOptions(
        cov_type=cov_type, cluster="g" if cov_type == "cluster" else None
    )
    res = OLS(df, "y", ["x1", "x2"], options=options).fit()

    x = [[1.0, a, b] for a, b in zip(df["x1"], df["x2"])]
    exact = exact_slope_wald(
        x, df["y"].to_list(), cov_type=cov_type, groups=df["g"].to_list()
    )
    assert res.f_statistic * 2 == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("cov_type", ["hc1", "cluster"])
def test_fe_near_collinear_slopes_with_a_sandwich_covariance_raise(cov_type):
    """【既知の限界の固定】相関が`1 - 5e-15`（`eps=1e-7`、`1e-7`〜`3e-7`で安定）のFEは、
    サンドイッチ共分散の桁落ちで全体F検定が`MATRIX_NEAR_SINGULAR`で止まる。classicalは同じ
    データで通る（上の説明と、限界を直したときの扱いを参照）。"""
    df = _near_collinear_panel(1e-7)
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR,
        matrix="COEFFICIENT_COVARIANCE",
        stage="OVERALL_F_TEST",
    ):
        FE(
            df,
            y="y",
            x=["x1", "x3"],
            effects=["entity"],
            options=FEOptions(cov_type=cov_type),
        ).fit()
    res = FE(
        df,
        y="y",
        x=["x1", "x3"],
        effects=["entity"],
        options=FEOptions(cov_type="classical"),
    ).fit()
    assert np.isfinite(res.f_statistic)
