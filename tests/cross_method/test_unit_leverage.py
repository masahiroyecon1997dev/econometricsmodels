"""レバレッジが1の観測があるとき、HC2・HC3が全手法で同じ`code`になるテスト。

`UNIT_LEVERAGE_OBSERVATION`（`ComputationError`）: ある観測だけで値が変わる回帰変数（1行だけが1の
ダミー、`T_i = 2`の個体の中だけで動く変数）があると、その観測のレバレッジ`h_ii`が（丸め誤差の範囲で）
1になり、モデルがその観測に完全にあてはまって残差が0になる。HC2・HC3の行スケール`1/(1-h_ii)`は
`0/0`か、丸めノイズが増幅された任意の値になる。以前は、列の単位や丸めで`MATRIX_NEAR_SINGULAR`
（原因を指さない）になる場合と、任意の値の標準誤差が返る場合があった。ここでは、`cov_type`が
`hc2`・`hc3`のとき手法・列の単位によらず同じ`code`になること、レバレッジを使わない`cov_type`は
影響を受けないことを見る。

FEIV-GMMの既定の重み（`robust`）では、同じデータが別の症状（`GMM_BREAD`が特異、`cov_type`によらない）に
なる。それはHC2・HC3のレバレッジの問題ではないので、ここでは重みをclassicalにして検査する。

単一方程式のIV・FEIV・REは、第一段階・Hausmanの補助回帰が構造式より先に同じ理由で止まる（`stage`が付く）。
構造式・各呼び出し箇所が`UnitLeverage`を返すことは、engineの単体テストが直接固定する。
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
UNITS = [1.0, 0.1, 3.0, 10.0, 1e3]
HC_ROBUST = ["hc2", "hc3"]


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


SINGLE_OPTIONS = {
    "hac": {"hac_time": "t"},
    "cluster": {"cluster": "g"},
}
PANEL_OPTIONS = {"dk": {"dk_time": "time"}}


def _fit(model: str, cov_type: str, unit: float) -> object:
    if model == "OLS":
        return OLS(
            _single_equation_data(unit),
            y="y",
            x=["x1", "d"],
            options=OLSOptions(
                cov_type=cov_type, **SINGLE_OPTIONS.get(cov_type, {})
            ),
        ).fit()
    if model == "WLS":
        return WLS(
            _single_equation_data(unit),
            y="y",
            x=["x1", "d"],
            weight="w",
            options=WLSOptions(
                cov_type=cov_type, **SINGLE_OPTIONS.get(cov_type, {})
            ),
        ).fit()
    if model in ("IV-2SLS", "IV-GMM"):
        estimator = "gmm" if model == "IV-GMM" else "2sls"
        return IV(
            _single_equation_data(unit),
            y="y",
            x_exog=["x1", "d"],
            x_endog=["endog"],
            instruments=["z"],
            options=IVOptions(
                cov_type=cov_type,
                estimator=estimator,
                **SINGLE_OPTIONS.get(cov_type, {}),
            ),
        ).fit()
    panel = _panel_data(unit)
    if model == "FE":
        return FE(
            panel,
            y="y",
            x=["x1", "x2"],
            effects=["id"],
            options=FEOptions(
                cov_type=cov_type, **PANEL_OPTIONS.get(cov_type, {})
            ),
        ).fit()
    if model == "RE":
        return RE(
            panel,
            y="y",
            x=["x1", "x2"],
            effects=["id"],
            options=REOptions(
                cov_type=cov_type, **PANEL_OPTIONS.get(cov_type, {})
            ),
        ).fit()
    # 重みが`robust`（FEIV-GMMの既定）だと、`h_ii = 1`の行の残差が0なので`x2`のモーメントの
    # 分散が0になり、`cov_type`によらず`GMM_BREAD`が特異になる（HC2・HC3に限らない症状）。
    # ここは`cov_type`のレバレッジの検査なので、重みはclassicalにする。
    options = (
        FEIVOptions(
            cov_type=cov_type,
            estimator="gmm",
            gmm_weight_type="classical",
            **PANEL_OPTIONS.get(cov_type, {}),
        )
        if model == "FEIV-GMM"
        else FEIVOptions(
            cov_type=cov_type,
            estimator="2sls",
            **PANEL_OPTIONS.get(cov_type, {}),
        )
    )
    return FEIV(
        panel,
        y="y",
        x_exog=["x1", "x2"],
        x_endog=["endog"],
        instruments=["z"],
        effects=["id"],
        options=options,
    ).fit()


MODELS = [
    "OLS",
    "WLS",
    "IV-2SLS",
    "IV-GMM",
    "FE",
    "RE",
    "FEIV-2SLS",
    "FEIV-GMM",
]


# 先に落ちる回帰の`stage`（構造式に到達するモデルは`None`）。
EXPECTED_STAGE = {
    "OLS": None,
    "WLS": None,
    "IV-2SLS": "FIRST_STAGE",
    "IV-GMM": "FIRST_STAGE",
    "FE": None,
    "RE": "HAUSMAN_REGRESSION",
    "FEIV-2SLS": "FIRST_STAGE",
    "FEIV-GMM": None,
}


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("cov_type", HC_ROBUST)
@pytest.mark.parametrize("model", MODELS)
def test_hc2_and_hc3_report_the_unit_leverage_observation_in_every_method(
    model: str, cov_type: str, unit: float
) -> None:
    """列の単位（丸め）によらず、エラーか有限値かが分かれず同じ`code`になる。"""
    stage = EXPECTED_STAGE[model]
    params = {} if stage is None else {"stage": stage}
    with raises_error(ErrorCode.UNIT_LEVERAGE_OBSERVATION, **params) as info:
        _fit(model, cov_type, unit)
    assert info.value.params["n_affected"] >= 1


@pytest.mark.parametrize(
    ("model", "n_affected", "first_row"),
    [
        # `d`が非ゼロの1行だけ。
        ("OLS", 1, DUMMY_ROW),
        ("WLS", 1, DUMMY_ROW),
        ("IV-2SLS", 1, DUMMY_ROW),
        ("IV-GMM", 1, DUMMY_ROW),
        # `T_i = 2`の個体0の2行（`x2`が個体内で変わるのはこの個体だけ）。
        ("FE", SHORT_ENTITY_PERIODS, 0),
        ("RE", SHORT_ENTITY_PERIODS, 0),
        ("FEIV-2SLS", SHORT_ENTITY_PERIODS, 0),
        ("FEIV-GMM", SHORT_ENTITY_PERIODS, 0),
    ],
)
@pytest.mark.parametrize("cov_type", HC_ROBUST)
def test_the_error_reports_how_many_observations_and_the_first_row(
    model: str, cov_type: str, n_affected: int, first_row: int
) -> None:
    with raises_error(
        ErrorCode.UNIT_LEVERAGE_OBSERVATION,
        n_affected=n_affected,
        first_row=first_row,
    ):
        _fit(model, cov_type, 1.0)


def test_several_unit_leverage_rows_are_counted_and_the_first_is_reported() -> (
    None
):
    """行ごとのダミー2列は、各行を別々に完全にあてはめる（ダミー1列に2行が乗るとレバレッジは1/2）。"""
    d = np.zeros(N)
    d[9] = 1.0
    d2 = np.zeros(N)
    d2[13] = 1.0
    df = _single_equation_data(1.0).with_columns(
        d=pl.Series(d), d2=pl.Series(d2)
    )
    with raises_error(
        ErrorCode.UNIT_LEVERAGE_OBSERVATION, n_affected=2, first_row=9
    ):
        OLS(
            df,
            y="y",
            x=["x1", "d", "d2"],
            options=OLSOptions(cov_type="hc3"),
        ).fit()


def test_the_first_row_follows_the_row_order_of_the_data() -> None:
    """行番号は、データの行順（入れ替えた後の位置）で数える。"""
    panel = _panel_data(1.0).sample(fraction=1.0, shuffle=True, seed=7)
    rows = panel.with_row_index("row").filter(pl.col("id") == 0)["row"]
    with raises_error(
        ErrorCode.UNIT_LEVERAGE_OBSERVATION,
        n_affected=SHORT_ENTITY_PERIODS,
        first_row=int(rows.min()),
    ):
        FE(
            panel,
            y="y",
            x=["x1", "x2"],
            effects=["id"],
            options=FEOptions(cov_type="hc3"),
        ).fit()


@pytest.mark.parametrize("cov_type", HC_ROBUST)
def test_two_way_effects_report_a_single_row_dummy(cov_type: str) -> None:
    """2-wayのレバレッジ（吸収した効果の別経路）でも、1行だけ非ゼロの列は1になる。"""
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
    with raises_error(
        ErrorCode.UNIT_LEVERAGE_OBSERVATION, n_affected=1, first_row=12
    ):
        FE(
            panel,
            y="y",
            x=["x1", "x2"],
            effects=["id", "time"],
            options=FEOptions(cov_type=cov_type),
        ).fit()


@pytest.mark.parametrize("model", ["OLS", "WLS", "IV-2SLS", "IV-GMM"])
@pytest.mark.parametrize(
    "cov_type", ["classical", "hc0", "hc1", "hac", "cluster"]
)
def test_single_equation_cov_types_without_leverage_are_not_affected(
    model: str, cov_type: str
) -> None:
    result = _fit(model, cov_type, 1.0)
    assert all(np.isfinite(v) for v in result.std_errors.values())


@pytest.mark.parametrize("model", ["FE", "RE", "FEIV-2SLS", "FEIV-GMM"])
def test_classical_does_not_depend_on_the_leverage(model: str) -> None:
    result = _fit(model, "classical", 1.0)
    assert all(np.isfinite(v) for v in result.std_errors.values())


@pytest.mark.parametrize("cov_type", ["cluster", "dk"])
def test_re_cluster_and_dk_are_not_affected(cov_type: str) -> None:
    """`h = 1`の行はスコアが0になる。FE・FEIVの`hc1`・`cluster`・`dk`は、そのため共分散が
    ほぼ特異になり全体F検定が`MATRIX_NEAR_SINGULAR`で止まる（レバレッジを割る計算ではなく、
    HC2・HC3のガードの対象外）。REの`cluster`・`dk`は推定できる。"""
    result = _fit("RE", cov_type, 1.0)
    assert all(np.isfinite(v) for v in result.std_errors.values())
