"""残差に基づくロバスト共分散で全体F/Waldが特異になる原因を、レバレッジが1の観測で名指しするテスト。

`UNIT_LEVERAGE_SLOPE_COVARIANCE`（`ComputationError`）: ある観測だけで値が変わる回帰変数（1行だけが1の
ダミー、`T_i = 2`の個体の中だけで動く変数）があると、その観測は完全にあてはまり残差もスコアも0になる。
その行にだけ値がある傾きの方向は、残差に基づく共分散（`hc0`・`hc1`・`cluster`・`dk`・`hac`）の分散が0で、
全体F/Wald（とREのHausman補助回帰）が定義できない。以前は原因を指さない`MATRIX_NEAR_SINGULAR`
（共線性を疑わせる）だった。

名指しするのは、全体検定が特異で失敗したときだけで、推定できるものは変えない。切片のあるOLS・WLS・IVでは、
行を選ぶ方向が切片を含み傾きの空間の外なので、同じ行があっても推定できる（レバレッジ1だけでは失敗の原因に
ならない）。切片のないモデルとFE・FEIV・REは、ダミーそのものが傾きの空間に入る。

HC2・HC3は行スケール自体が定義できないので、従来どおり`UNIT_LEVERAGE_OBSERVATION`が先に止める。
`test_unit_leverage.py`が固定する。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _errors import raises_error
from econometricsmodels import (
    FE,
    FEIV,
    IV,
    OLS,
    RE,
    WLS,
    ErrorCode,
    FEIVOptions,
    FEOptions,
    IVOptions,
    OLSOptions,
    REOptions,
    WLSOptions,
)

N = 30
DUMMY_ROW = 5
N_ENTITIES = 10
SHORT_ENTITY_PERIODS = 2
PERIODS = 5
UNITS = [1.0, 0.1, 1e3, 1e-6]
ROBUST = ["hc0", "hc1", "hac", "cluster"]
PANEL_ROBUST = ["hc1", "cluster", "dk"]


def _single_equation_data(unit: float) -> pl.DataFrame:
    """`d`が`DUMMY_ROW`だけで非ゼロの単一方程式データ（`unit`は`d`の単位）。"""
    rng = np.random.default_rng(0)
    z = rng.normal(size=N)
    e = rng.normal(size=N)
    endog = z + 0.5 * e + rng.normal(size=N)
    d = np.zeros(N)
    d[DUMMY_ROW] = unit
    return pl.DataFrame(
        {
            "y": endog + e,
            "x1": rng.normal(size=N),
            "d": d,
            "endog": endog,
            "z": z,
            "w": rng.uniform(0.5, 2.0, size=N),
            "t": np.arange(N),
            "g": np.arange(N) % 6,
        }
    )


def _panel_data(unit: float) -> pl.DataFrame:
    """個体0だけが2期間で、`x2`はその個体の最初の期間だけ非ゼロのパネル。"""
    rng = np.random.default_rng(0)
    ids, time, x2 = [], [], []
    for i in range(N_ENTITIES):
        periods = SHORT_ENTITY_PERIODS if i == 0 else PERIODS
        for t in range(periods):
            ids.append(i)
            time.append(t)
            x2.append(unit if (i == 0 and t == 0) else 0.0)
    n = len(ids)
    z = rng.normal(size=n)
    e = rng.normal(size=n)
    endog = z + 0.5 * e + rng.normal(size=n)
    return pl.DataFrame(
        {
            "id": ids,
            "y": endog + e,
            "x1": rng.normal(size=n),
            "x2": x2,
            "endog": endog,
            "z": z,
            "time": time,
        }
    )


def _single_options(cov_type: str) -> dict[str, str]:
    return {"hac": {"hac_time": "t"}, "cluster": {"cluster": "g"}}.get(
        cov_type, {}
    )


def _panel_options(cov_type: str) -> dict[str, str]:
    return {"dk": {"dk_time": "time"}}.get(cov_type, {})


def _fit_single(
    model: str, cov_type: str, unit: float, *, intercept: bool
) -> object:
    data = _single_equation_data(unit)
    extra = _single_options(cov_type)
    if model == "OLS":
        return OLS(
            data,
            y="y",
            x=["x1", "d"],
            options=OLSOptions(
                cov_type=cov_type, include_intercept=intercept, **extra
            ),
        ).fit()
    if model == "WLS":
        return WLS(
            data,
            y="y",
            x=["x1", "d"],
            weight="w",
            options=WLSOptions(
                cov_type=cov_type, include_intercept=intercept, **extra
            ),
        ).fit()
    return IV(
        data,
        y="y",
        x_exog=["x1", "d"],
        x_endog=["endog"],
        instruments=["z"],
        options=IVOptions(
            cov_type=cov_type,
            estimator="gmm" if model == "IV-GMM" else "2sls",
            include_intercept=intercept,
            **extra,
        ),
    ).fit()


def _fit_panel(model: str, cov_type: str, unit: float) -> object:
    panel = _panel_data(unit)
    extra = _panel_options(cov_type)
    if model == "FE":
        return FE(
            panel,
            y="y",
            x=["x1", "x2"],
            effects=["id"],
            options=FEOptions(cov_type=cov_type, **extra),
        ).fit()
    if model == "RE":
        return RE(
            panel,
            y="y",
            x=["x1", "x2"],
            effects=["id"],
            options=REOptions(cov_type=cov_type, **extra),
        ).fit()
    return FEIV(
        panel,
        y="y",
        x_exog=["x1", "x2"],
        x_endog=["endog"],
        instruments=["z"],
        effects=["id"],
        options=FEIVOptions(cov_type=cov_type, estimator="2sls", **extra),
    ).fit()


# 失敗した回帰の`stage`。IVは第一段階が構造式より先に同じ理由で止まる。
SINGLE_STAGE = {
    "OLS": None,
    "WLS": None,
    "IV-2SLS": "FIRST_STAGE",
    "IV-GMM": "FIRST_STAGE",
}


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("cov_type", ROBUST)
@pytest.mark.parametrize("model", list(SINGLE_STAGE))
def test_a_model_without_an_intercept_names_the_single_row_dummy(
    model: str, cov_type: str, unit: float
) -> None:
    """切片がないと、ダミーそのものが傾きの空間に入る。列の単位（丸め）によらず同じ`code`になる。"""
    stage = SINGLE_STAGE[model]
    params = {} if stage is None else {"stage": stage}
    with raises_error(
        ErrorCode.UNIT_LEVERAGE_SLOPE_COVARIANCE,
        n_affected=1,
        first_row=DUMMY_ROW,
        **params,
    ):
        _fit_single(model, cov_type, unit, intercept=False)


@pytest.mark.parametrize("cov_type", ROBUST)
@pytest.mark.parametrize("model", list(SINGLE_STAGE))
def test_the_same_rows_with_an_intercept_are_estimated(
    model: str, cov_type: str
) -> None:
    """切片があれば、行を選ぶ方向が切片を含むので、傾きのブロックは特異にならない。"""
    result = _fit_single(model, cov_type, 1.0, intercept=True)
    assert all(np.isfinite(v) for v in result.std_errors.values())


@pytest.mark.parametrize("model", list(SINGLE_STAGE))
def test_classical_does_not_depend_on_the_rows_one_by_one(model: str) -> None:
    result = _fit_single(model, "classical", 1.0, intercept=False)
    assert all(np.isfinite(v) for v in result.std_errors.values())


@pytest.mark.parametrize("model", list(SINGLE_STAGE)[:2])
def test_hc2_is_still_reported_as_the_undefined_row_scale(model: str) -> None:
    with raises_error(ErrorCode.UNIT_LEVERAGE_OBSERVATION):
        _fit_single(model, "hc2", 1.0, intercept=False)


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("cov_type", PANEL_ROBUST)
@pytest.mark.parametrize("model", ["FE", "FEIV-2SLS"])
def test_fixed_effects_name_the_rows_of_the_short_entity(
    model: str, cov_type: str, unit: float
) -> None:
    """`T_i = 2`の個体0の2行（`x2`が個体内で変わるのはこの個体だけ）。"""
    with raises_error(
        ErrorCode.UNIT_LEVERAGE_SLOPE_COVARIANCE,
        stage="OVERALL_F_TEST",
        n_affected=SHORT_ENTITY_PERIODS,
        first_row=0,
    ):
        _fit_panel(model, cov_type, unit)


@pytest.mark.parametrize("unit", UNITS)
def test_re_hc1_names_the_rows_in_the_hausman_auxiliary_regression(
    unit: float,
) -> None:
    with raises_error(
        ErrorCode.UNIT_LEVERAGE_SLOPE_COVARIANCE,
        stage="HAUSMAN_REGRESSION",
        n_affected=SHORT_ENTITY_PERIODS,
        first_row=0,
    ):
        _fit_panel("RE", "hc1", unit)


@pytest.mark.parametrize("cov_type", ["cluster", "dk"])
def test_re_cluster_and_dk_are_estimated(cov_type: str) -> None:
    result = _fit_panel("RE", cov_type, 1.0)
    assert all(np.isfinite(v) for v in result.std_errors.values())


@pytest.mark.parametrize("model", ["FE", "RE", "FEIV-2SLS"])
def test_panel_classical_is_estimated(model: str) -> None:
    result = _fit_panel(model, "classical", 1.0)
    assert all(np.isfinite(v) for v in result.std_errors.values())


@pytest.mark.parametrize("cov_type", ["hc1", "cluster"])
def test_two_way_effects_estimate_a_single_row_dummy(cov_type: str) -> None:
    """2-wayでは、1行だけ非ゼロの列もwithin変換で個体・時点の行に広がり、その行以外にも値を持つ。
    その行のレバレッジは1（`test_unit_leverage.py`）でも、傾きのブロックは特異にならず推定できる。"""
    rng = np.random.default_rng(1)
    n_entities, n_periods = 8, 5
    n = n_entities * n_periods
    x2 = np.zeros(n)
    x2[12] = 1.0
    panel = pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_entities), n_periods),
            "time": np.tile(np.arange(n_periods), n_entities),
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": x2,
        }
    )
    extra = {"cluster": "id"} if cov_type == "cluster" else {}
    result = FE(
        panel,
        y="y",
        x=["x1", "x2"],
        effects=["id", "time"],
        options=FEOptions(cov_type=cov_type, **extra),
    ).fit()
    assert all(np.isfinite(v) for v in result.std_errors.values())
