"""完全フィットと定数の被説明変数が、全手法で同じ`code`になるテスト。

- 完全フィット（`PERFECT_FIT`、`ComputationError`）: 残差が丸め誤差の範囲で0（残差平方和が
  総平方和の`k * epsilon`倍以下）だと標準誤差とF検定が定義できない。残差が厳密に0になるかは
  データの丸め誤差で決まるため、`cov_type`・手法・`y`の作り方によらず同じ`code`になることを見る。
- 定数の被説明変数: 切片のあるモデルで`y`が全行で同じ値なら`OUTCOME_CONSTANT`
  （`ValidationError`）。FE・REでは個体内の変動が無い場合が`OUTCOME_CONSTANT_WITHIN_ENTITY`
  （`ValidationError`）。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _errors import raises_error
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    ErrorCode,
    FEOptions,
    IVOptions,
    OLSOptions,
    REOptions,
    WLSOptions,
)

N_ENTITIES = 8
N_PERIODS = 5
N = N_ENTITIES * N_PERIODS
COV_TYPES = ["classical", "hc1", "cluster"]


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    x = rng.normal(size=N)
    x2 = rng.normal(size=N)
    z = rng.normal(size=N)
    z2 = rng.normal(size=N)
    endog = 0.8 * z + 0.4 * z2 + 0.3 * x + rng.normal(size=N)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    time = np.tile(np.arange(N_PERIODS), N_ENTITIES)
    return pl.DataFrame(
        {
            "x": x,
            "x2": x2,
            "z": z,
            "z2": z2,
            "endog": endog,
            "w": rng.uniform(0.5, 2.0, size=N),
            "entity": entity,
            "time": time,
            "noise": rng.normal(size=N),
        }
    )


def _cov_options(cov_type: str) -> dict:
    return {"cov_type": cov_type} | (
        {"cluster": "entity"} if cov_type == "cluster" else {}
    )


def _fit(model: str, df: pl.DataFrame, cov_type: str = "classical", **kw):
    """`model`で`y`を`x`・`x2`（IVは`endog`を内生変数、`z`・`z2`を操作変数）に回帰する。"""
    opts = _cov_options(cov_type)
    include_intercept = kw.pop("include_intercept", True)
    if model == "OLS":
        return OLS(
            df,
            "y",
            ["x", "x2"],
            OLSOptions(include_intercept=include_intercept, **opts),
        ).fit()
    if model == "WLS":
        return WLS(
            df,
            "y",
            ["x", "x2"],
            "w",
            WLSOptions(include_intercept=include_intercept, **opts),
        ).fit()
    if model == "FE":
        return FE(
            df, "y", ["x", "x2"], "entity", FEOptions(**kw, **opts)
        ).fit()
    if model == "RE":
        return RE(
            df, "y", ["x", "x2"], "entity", REOptions(**kw, **opts)
        ).fit()
    estimator = "gmm" if model == "IV-GMM" else "2sls"
    return IV(
        df,
        "y",
        ["x"],
        ["endog"],
        ["z", "z2"],
        IVOptions(
            estimator=estimator, include_intercept=include_intercept, **opts
        ),
    ).fit()


ALL_MODELS = ["OLS", "WLS", "FE", "RE", "IV-2SLS", "IV-GMM"]
# GMMは`cov_type`が`classical`・`hc1`・`cluster`とも使える。
SINGLE_EQUATION_MODELS = ["OLS", "WLS", "IV-2SLS", "IV-GMM"]


def _perfect_y(model: str, df: pl.DataFrame) -> pl.Expr:
    """`model`の構造式で厳密に成り立つ`y`（ノイズなし）。"""
    if model.startswith("IV"):
        return 1.0 + 2.0 * pl.col("x") - 1.5 * pl.col("endog")
    if model in ("FE", "RE"):
        # 個体効果つきで厳密に成り立つ。
        return 2.0 * pl.col("x") - 1.5 * pl.col("x2") + 3.0 * pl.col("entity")
    return 1.0 + 2.0 * pl.col("x") - 1.5 * pl.col("x2")


# ── 完全フィット ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("model", ALL_MODELS)
def test_noise_free_outcome_is_a_perfect_fit_for_every_cov_type(
    model, cov_type, df
):
    """`y`が説明変数の厳密な関数だと、`cov_type`によらず`PERFECT_FIT`になる。
    以前は、OLS・WLSはクラスターのときだけ`MATRIX_NEAR_SINGULAR`、FE・REは
    `y`の作り方によって`MATRIX_NEAR_SINGULAR`になり、それ以外は`F`が`1e35`のような
    無意味な値で成功していた。"""
    data = df.with_columns(y=_perfect_y(model, df))
    expected_stage = {
        "FE": "WITHIN_REGRESSION",
        "RE": "VARIANCE_COMPONENT_WITHIN_REGRESSION",
    }
    if model.startswith("IV"):
        # 構造残差が0なので、第一段階は成功したうえで構造式が止まる。
        expected_stage = {}
    stage = expected_stage.get(model)
    with raises_error(
        ErrorCode.PERFECT_FIT, **({"stage": stage} if stage else {})
    ):
        _fit(model, data, cov_type)


@pytest.mark.parametrize("model", ["OLS", "WLS"])
def test_perfect_fit_from_a_random_regressor_with_rounding_error(model, df):
    """`y = 1 + 2*x`を乱数の`x`から作ると、残差は厳密な0ではなく`1e-15`程度の丸め誤差が
    残る。それでも完全フィットとして同じ`code`になる（厳密な0かどうかで分けない）。"""
    data = df.with_columns(y=1.0 + 2.0 * pl.col("x"))
    for cov_type in COV_TYPES:
        with raises_error(ErrorCode.PERFECT_FIT):
            _fit(model, data, cov_type)


@pytest.mark.parametrize("model", SINGLE_EQUATION_MODELS)
def test_perfect_fit_without_an_intercept(model, df):
    """切片なしでも同じ判定（非中心化の総平方和に対する相対閾値）。"""
    data = df.with_columns(
        y=2.0 * pl.col("x") - 1.5 * pl.col("endog")
        if model.startswith("IV")
        else 2.0 * pl.col("x") - 1.5 * pl.col("x2")
    )
    with raises_error(ErrorCode.PERFECT_FIT):
        _fit(model, data, include_intercept=False)


@pytest.mark.parametrize("model", ["OLS", "WLS", "IV-2SLS", "IV-GMM"])
def test_perfect_fit_is_detected_for_an_outcome_with_a_large_mean(model, df):
    """平均が標準偏差より桁違いに大きい`y`でも、丸め誤差の残差を完全フィットとみなす
    （総平方和は平均で中心化した値を基準にする）。"""
    base = (
        1.0 + 2.0 * pl.col("x") - 1.5 * pl.col("endog")
        if model.startswith("IV")
        else 1.0 + 2.0 * pl.col("x") - 1.5 * pl.col("x2")
    )
    data = df.with_columns(y=base + 1e6)
    with raises_error(ErrorCode.PERFECT_FIT):
        _fit(model, data)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_a_small_but_real_error_is_not_a_perfect_fit(model, df):
    """ノイズが総平方和に対して`k * epsilon`を十分に超える（残差の標準偏差が`y`の約`1e-6`）
    なら完全フィットではなく、通常どおり推定できる。"""
    data = df.with_columns(y=_perfect_y(model, df) + 1e-6 * pl.col("noise"))
    result = _fit(model, data)
    assert all(np.isfinite(list(result.std_errors.values())))


def test_perfect_first_stage_is_reported_with_the_first_stage_context(df):
    """操作変数が内生変数を完全に予測する（第一段階の残差が0）と、その内生変数の
    第一段階の文脈つきで`PERFECT_FIT`になる。構造式のノイズの有無とは別に判定される。"""
    data = df.with_columns(
        endog=2.0 * pl.col("z") - pl.col("z2"),
        y=1.0 + pl.col("x") + pl.col("noise"),
    )
    for estimator in ("2sls", "gmm"):
        with raises_error(
            ErrorCode.PERFECT_FIT, stage="FIRST_STAGE", endog_name="endog"
        ):
            IV(
                data,
                "y",
                ["x"],
                ["endog"],
                ["z", "z2"],
                IVOptions(estimator=estimator),
            ).fit()


def test_the_perfect_fit_text_does_not_mention_scale(df):
    """利用者に出る文面は、完全フィットが原因だと伝える（スケール差の対処を勧めない）。"""
    data = df.with_columns(y=1.0 + 2.0 * pl.col("x") - 1.5 * pl.col("x2"))
    with pytest.raises(Exception) as info:
        OLS(data, "y", ["x", "x2"]).fit()
    text = str(info.value)
    assert "fits the data perfectly" in text
    assert "scale" not in text.lower()


# ── 定数の被説明変数 ────────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("model", SINGLE_EQUATION_MODELS)
def test_constant_outcome_with_an_intercept_is_a_validation_error(
    model, cov_type, df
):
    """切片のあるモデルで`y`が全行で同じ値だと、説明するものが無い。以前は
    OLS・WLS・IVが`r_squared = -inf`で成功していた。"""
    data = df.with_columns(y=pl.lit(3.0))
    with raises_error(ErrorCode.OUTCOME_CONSTANT):
        _fit(model, data, cov_type)


@pytest.mark.parametrize("model", ["OLS", "WLS", "IV-2SLS"])
def test_constant_outcome_without_an_intercept_is_allowed(model, df):
    """切片なしでは、定数の`y`の非中心化の総平方和が正なので説明する対象がある。"""
    data = df.with_columns(y=pl.lit(3.0) + 1e-3 * pl.col("noise"))
    result = _fit(model, data, include_intercept=False)
    assert np.isfinite(result.r_squared)
    assert pl.Series(data["y"]).n_unique() > 1


@pytest.mark.parametrize("model", ["OLS", "WLS", "IV-2SLS"])
def test_exactly_constant_outcome_without_an_intercept_is_not_an_outcome_error(
    model, df
):
    """切片なしで`y`が全行で同じ値でも`OUTCOME_CONSTANT`にはならない（説明変数が
    説明できるかは推定の結果で決まる）。"""
    data = df.with_columns(y=pl.lit(3.0))
    try:
        _fit(model, data, include_intercept=False)
    except Exception as exc:  # noqa: BLE001
        assert getattr(exc, "code", None) != ErrorCode.OUTCOME_CONSTANT


def test_constant_outcome_is_reported_before_the_first_stage(df):
    """IVでは、定数の`y`は第一段階より前に報告する（第一段階の失敗に隠れない）。"""
    data = df.with_columns(y=pl.lit(3.0), endog=2.0 * pl.col("z"))
    with raises_error(ErrorCode.OUTCOME_CONSTANT):
        _fit("IV-2SLS", data)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("model", ["FE", "RE"])
def test_outcome_constant_within_every_entity(model, cov_type, df):
    """個体内で定数の`y`（個体間でだけ違う）は、固定効果が全てを説明してwithin変動が残らない。
    `0.1`のように平均が丸め誤差を持つ値でも同じ。"""
    data = df.with_columns(y=0.1 * (pl.col("entity") + 1))
    with raises_error(ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY):
        _fit(model, data, cov_type)


def test_a_globally_constant_outcome_is_constant_within_entity(df):
    data = df.with_columns(y=pl.lit(4.0))
    for model in ("FE", "RE"):
        with raises_error(ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY):
            _fit(model, data)


@pytest.mark.parametrize("model", ["FE", "RE"])
def test_outcome_explained_by_entity_and_time_effects_in_a_two_way_model(
    model, df
):
    """2-wayでは、`y = 個体効果 + 時点効果`も2方向のwithin変換後に0になる。"""
    data = df.with_columns(y=2.0 * pl.col("entity") - 0.5 * pl.col("time"))
    with raises_error(ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY):
        _fit(model, data, time="time")


def test_outcome_constant_within_entity_is_not_staged_in_re(df):
    """REの内部FEが報告するが、入力の事実なので文脈句（`stage`）は付けない。"""
    data = df.with_columns(y=pl.lit(4.0))
    with pytest.raises(Exception) as info:
        RE(data, "y", ["x"], "entity").fit()
    assert getattr(info.value, "code", None) == (
        ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY
    )
    assert getattr(info.value, "params", {}).get("stage") is None
    assert not str(info.value).startswith("In the ")
