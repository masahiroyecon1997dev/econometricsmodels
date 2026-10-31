"""完全フィットと定数の被説明変数が、全手法で同じ`code`になるテスト。

- 完全フィット（`PERFECT_FIT`、`ComputationError`）: 残差が丸め誤差の範囲で0（残差平方和が
  総平方和の`k * epsilon`倍以下）だと標準誤差とF検定が定義できない。残差が厳密に0になるかは
  データの丸め誤差で決まるため、`cov_type`・手法・`y`の作り方によらず同じ`code`になることを見る。
- 定数の被説明変数: 切片のあるモデルで`y`が（丸め誤差の範囲で）全行で同じ値なら
  `OUTCOME_CONSTANT`（`ValidationError`）。FE・REでは個体内の変動が無い場合が
  `OUTCOME_CONSTANT_WITHIN_ENTITY`（`ValidationError`）。
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

LINEAR_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac", "cluster"]
PANEL_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
ALL_MODELS = ["OLS", "WLS", "FE", "RE", "IV-2SLS", "IV-GMM"]
SINGLE_EQUATION_MODELS = ["OLS", "WLS", "IV-2SLS", "IV-GMM"]
MODEL_COV_CASES = [
    (model, cov_type)
    for model in ALL_MODELS
    for cov_type in (
        PANEL_COV_TYPES if model in ("FE", "RE") else LINEAR_COV_TYPES
    )
]
GMM_TYPES = ["one_step", "two_step", "iterated"]


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    x = rng.normal(size=N)
    x2 = rng.normal(size=N)
    z = rng.normal(size=N)
    z2 = rng.normal(size=N)
    endog = 0.8 * z + 0.4 * z2 + 0.3 * x + rng.normal(size=N)
    endog2 = 0.5 * z - 0.7 * z2 + 0.2 * x + rng.normal(size=N)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    time = np.tile(np.arange(N_PERIODS), N_ENTITIES)
    return pl.DataFrame(
        {
            "x": x,
            "x2": x2,
            "z": z,
            "z2": z2,
            "endog": endog,
            "endog2": endog2,
            "w": rng.uniform(0.5, 2.0, size=N),
            "entity": entity,
            "time": time,
            # 重複のない時間順（`cov_type="hac"`の`hac_time`）。
            "t": np.arange(N),
            "noise": rng.normal(size=N),
        }
    )


def _cov_options(cov_type: str) -> dict:
    if cov_type == "cluster":
        return {"cov_type": cov_type, "cluster": "entity"}
    if cov_type == "hac":
        return {"cov_type": cov_type, "hac_time": "t"}
    if cov_type == "dk":
        return {"cov_type": cov_type, "dk_time": "time"}
    return {"cov_type": cov_type}


def _fit(
    model: str,
    df: pl.DataFrame,
    cov_type: str = "classical",
    include_intercept: bool = True,
    gmm_type: str | None = None,
    two_way: bool = False,
    **kw,
):
    """`model`で`y`を`x`・`x2`（IVは`x`を外生変数、`endog`を内生変数、`z`・`z2`を操作変数）に回帰する。"""
    opts = _cov_options(cov_type)
    effects = ["entity", "time"] if two_way else ["entity"]
    if model == "OLS":
        return OLS(
            df,
            "y",
            ["x", "x2"],
            options=OLSOptions(include_intercept=include_intercept, **opts),
        ).fit()
    if model == "WLS":
        return WLS(
            df,
            "y",
            ["x", "x2"],
            "w",
            options=WLSOptions(include_intercept=include_intercept, **opts),
        ).fit()
    if model == "FE":
        return FE(
            df, "y", ["x", "x2"], effects, options=FEOptions(**kw, **opts)
        ).fit()
    if model == "RE":
        return RE(
            df, "y", ["x", "x2"], effects, options=REOptions(**kw, **opts)
        ).fit()
    estimator = "gmm" if model == "IV-GMM" else "2sls"
    gmm = {"gmm_type": gmm_type} if gmm_type else {}
    return IV(
        df,
        "y",
        ["x"],
        ["endog"],
        ["z", "z2"],
        options=IVOptions(
            estimator=estimator,
            include_intercept=include_intercept,
            **opts,
            **gmm,
        ),
    ).fit()


def _perfect_y(model: str) -> pl.Expr:
    """`model`の構造式で厳密に成り立つ`y`（ノイズなし）。"""
    if model.startswith("IV"):
        return 1.0 + 2.0 * pl.col("x") - 1.5 * pl.col("endog")
    if model in ("FE", "RE"):
        # 個体効果つきで厳密に成り立つ。
        return 2.0 * pl.col("x") - 1.5 * pl.col("x2") + 3.0 * pl.col("entity")
    return 1.0 + 2.0 * pl.col("x") - 1.5 * pl.col("x2")


def _stage(model: str) -> dict:
    """`PERFECT_FIT`に付く`stage`（FE・REはwithin回帰の文脈、他は付かない）。"""
    return {
        "FE": {"stage": "WITHIN_REGRESSION"},
        "RE": {"stage": "VARIANCE_COMPONENT_WITHIN_REGRESSION"},
    }.get(model, {})


# ── 完全フィット ────────────────────────────────────────────────────────


@pytest.mark.parametrize(("model", "cov_type"), MODEL_COV_CASES)
def test_noise_free_outcome_is_a_perfect_fit_for_every_cov_type(
    model, cov_type, df
):
    """`y`が説明変数の厳密な関数だと、手法と`cov_type`によらず`PERFECT_FIT`になる。
    残差が厳密に0かどうかはデータの丸め誤差で決まるため、`cov_type`で結果が分かれない
    ことを全ての`cov_type`で確認する。"""
    data = df.with_columns(y=_perfect_y(model))
    with raises_error(ErrorCode.PERFECT_FIT, **_stage(model)):
        _fit(model, data, cov_type)


@pytest.mark.parametrize("gmm_type", GMM_TYPES)
def test_perfect_fit_for_every_gmm_type(gmm_type, df):
    """GMMは`gmm_type`ごとに推定の経路が違うが、初期推定の構造残差で同じ判定をする。"""
    data = df.with_columns(y=_perfect_y("IV-GMM"))
    with raises_error(ErrorCode.PERFECT_FIT):
        _fit("IV-GMM", data, gmm_type=gmm_type)


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
def test_perfect_fit_in_a_two_way_fixed_effects_model(cov_type, df):
    """2-wayでも、within回帰の残差が0なら`PERFECT_FIT`（`y`に個体効果と時点効果を足す）。"""
    data = df.with_columns(
        y=2.0 * pl.col("x")
        - 1.5 * pl.col("x2")
        + 3.0 * pl.col("entity")
        - 0.5 * pl.col("time")
    )
    with raises_error(ErrorCode.PERFECT_FIT, stage="WITHIN_REGRESSION"):
        _fit("FE", data, cov_type, two_way=True)


@pytest.mark.parametrize("model", ["OLS", "WLS"])
def test_perfect_fit_from_a_random_regressor_with_rounding_error(model, df):
    """`y = 1 + 2*x`を乱数の`x`から作ると、残差は厳密な0ではなく`1e-15`程度の丸め誤差が
    残る。それでも完全フィットとして同じ`code`になる（厳密な0かどうかで分けない）。"""
    data = df.with_columns(y=1.0 + 2.0 * pl.col("x"))
    for cov_type in LINEAR_COV_TYPES:
        with raises_error(ErrorCode.PERFECT_FIT):
            _fit(model, data, cov_type)


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
@pytest.mark.parametrize("model", SINGLE_EQUATION_MODELS)
def test_perfect_fit_without_an_intercept(model, cov_type, df):
    """切片なしでも同じ判定（非中心化の総平方和に対する相対閾値）。"""
    data = df.with_columns(
        y=2.0 * pl.col("x") - 1.5 * pl.col("endog")
        if model.startswith("IV")
        else 2.0 * pl.col("x") - 1.5 * pl.col("x2")
    )
    with raises_error(ErrorCode.PERFECT_FIT):
        _fit(model, data, cov_type, include_intercept=False)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_perfect_fit_is_detected_for_an_outcome_with_a_large_mean(model, df):
    """平均が標準偏差より桁違いに大きい`y`でも、丸め誤差の残差を完全フィットとみなす
    （総平方和は平均で中心化した値を基準にする）。"""
    data = df.with_columns(y=_perfect_y(model) + 1e6)
    with raises_error(ErrorCode.PERFECT_FIT, **_stage(model)):
        _fit(model, data)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_a_small_but_real_error_is_not_a_perfect_fit(model, df):
    """ノイズが総平方和に対して`k * epsilon`を十分に超える（残差の標準偏差が`y`の約`1e-6`）
    なら完全フィットではなく、通常どおり推定できる。"""
    data = df.with_columns(y=_perfect_y(model) + 1e-6 * pl.col("noise"))
    result = _fit(model, data)
    assert all(np.isfinite(list(result.std_errors.values())))


@pytest.mark.parametrize("model", ["OLS", "WLS", "FE", "IV-2SLS", "IV-GMM"])
def test_the_threshold_separates_a_real_error_from_rounding_noise(model, df):
    """閾値（残差の標準偏差が`y`の約`3e-8`倍）の両側: `1e-6`は推定でき、`1e-9`は完全フィット。
    RE（閾値のすぐ上の帯は最終回帰が悪条件）は`test_re_variance_components.py`で見る。"""
    perfect = _perfect_y(model)
    above = df.with_columns(y=perfect + 1e-6 * pl.col("noise"))
    below = df.with_columns(y=perfect + 1e-9 * pl.col("noise"))
    assert all(np.isfinite(list(_fit(model, above).std_errors.values())))
    with raises_error(ErrorCode.PERFECT_FIT, **_stage(model)):
        _fit(model, below)


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
                options=IVOptions(estimator=estimator),
            ).fit()


def test_the_first_stage_is_reported_before_the_structural_equation(df):
    """第一段階と構造式がどちらも完全フィットなら、第一段階（先に推定する）が報告される。"""
    data = df.with_columns(
        endog=2.0 * pl.col("z") - pl.col("z2"),
        y=1.0 + 2.0 * pl.col("x") - 1.5 * (2.0 * pl.col("z") - pl.col("z2")),
    )
    with raises_error(
        ErrorCode.PERFECT_FIT, stage="FIRST_STAGE", endog_name="endog"
    ):
        _fit("IV-2SLS", data)


def test_only_the_second_endogenous_regressor_has_a_perfect_first_stage(df):
    """内生変数が2つのとき、完全に予測される側の名前が`endog_name`に入る。"""
    data = df.with_columns(
        endog2=pl.col("z") + 0.5 * pl.col("z2"),
        y=1.0 + pl.col("x") + pl.col("noise"),
    )
    with raises_error(
        ErrorCode.PERFECT_FIT, stage="FIRST_STAGE", endog_name="endog2"
    ):
        IV(
            data,
            "y",
            ["x"],
            ["endog", "endog2"],
            ["z", "z2", "noise"],
            options=IVOptions(),
        ).fit()


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_wu_hausman_is_none_when_only_the_augmented_regression_is_exact(
    estimator, df
):
    """`y = 1 + 2*endog + 3*v`（`v`は第一段階残差）は、構造残差も第一段階も完全フィット
    ではないが、Wu-Hausmanの拡張回帰だけが厳密に成り立つ。`fit()`は成功し、
    2SLSの検定統計量は無意味な値ではなく`None`になる。"""
    first_stage = (
        IV(
            df.with_columns(y=pl.col("x") + pl.col("noise")),
            "y",
            ["x"],
            ["endog"],
            ["z", "z2"],
            options=IVOptions(estimator=estimator),
        )
        .fit()
        .first_stage()["endog"]
    )
    v = np.asarray(first_stage.residuals)
    data = df.with_columns(
        y=pl.Series(1.0 + 2.0 * df["endog"].to_numpy() + 3.0 * v)
    )
    result = IV(
        data,
        "y",
        ["x"],
        ["endog"],
        ["z", "z2"],
        options=IVOptions(estimator=estimator),
    ).fit()
    assert result.wu_hausman_statistic is None
    assert result.wu_hausman_p_value is None


def test_the_second_stage_regression_is_not_judged_for_a_perfect_fit(df):
    """`y = 1 + 2*endog_hat`は第二段階の回帰が厳密に成り立つが、構造残差は0ではない。
    標準誤差の基準は構造残差なので、完全フィットにならず推定できる。"""
    first_stage = (
        IV(
            df.with_columns(y=pl.col("x") + pl.col("noise")),
            "y",
            ["x"],
            ["endog"],
            ["z", "z2"],
            options=IVOptions(),
        )
        .fit()
        .first_stage()["endog"]
    )
    v = np.asarray(first_stage.residuals)
    data = df.with_columns(
        y=pl.Series(1.0 + 2.0 * (df["endog"].to_numpy() - v))
    )
    result = IV(
        data, "y", ["x"], ["endog"], ["z", "z2"], options=IVOptions()
    ).fit()
    assert all(np.isfinite(list(result.std_errors.values())))


def test_the_perfect_fit_text_does_not_mention_scale(df):
    """利用者に出る文面は、完全フィットが原因だと伝える（スケール差の対処を勧めない）。"""
    data = df.with_columns(y=1.0 + 2.0 * pl.col("x") - 1.5 * pl.col("x2"))
    with pytest.raises(Exception) as info:
        OLS(data, "y", ["x", "x2"]).fit()
    text = str(info.value)
    assert "fits the data perfectly" in text
    assert "scale" not in text.lower()


# ── 定数の被説明変数 ────────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
@pytest.mark.parametrize("model", SINGLE_EQUATION_MODELS)
def test_constant_outcome_with_an_intercept_is_a_validation_error(
    model, cov_type, df
):
    """切片のあるモデルで`y`が全行で同じ値だと、説明するものが無い。"""
    data = df.with_columns(y=pl.lit(3.0))
    with raises_error(ErrorCode.OUTCOME_CONSTANT):
        _fit(model, data, cov_type)


@pytest.mark.parametrize("gmm_type", GMM_TYPES)
def test_constant_outcome_for_every_gmm_type(gmm_type, df):
    data = df.with_columns(y=pl.lit(3.0))
    with raises_error(ErrorCode.OUTCOME_CONSTANT):
        _fit("IV-GMM", data, gmm_type=gmm_type)


@pytest.mark.parametrize("model", SINGLE_EQUATION_MODELS)
def test_an_outcome_that_differs_by_one_ulp_is_constant(model, df):
    """計算で作った定数（`0.1 + 0.2`と`0.3`）は1 ulp違うが、FEの分散ゼロ検出と同じ
    相対閾値で定数とみなす（`R² = 0.92`のような意味のない値で成功させない）。"""
    values = np.where(np.arange(N) % 2 == 0, 0.1 + 0.2, 0.3)
    assert len(set(values)) == 2
    data = df.with_columns(y=pl.Series(values))
    with raises_error(ErrorCode.OUTCOME_CONSTANT):
        _fit(model, data)


@pytest.mark.parametrize("model", SINGLE_EQUATION_MODELS)
def test_a_small_but_real_variation_is_not_a_constant_outcome(model, df):
    data = df.with_columns(y=3.0 + 1e-6 * pl.col("noise"))
    result = _fit(model, data)
    assert np.isfinite(result.r_squared)


@pytest.mark.parametrize("model", ["OLS", "WLS", "IV-2SLS", "IV-GMM"])
def test_constant_outcome_without_an_intercept_is_estimated(model, df):
    """切片なしでは、定数の`y`の非中心化の総平方和が正なので説明する対象がある
    （`OUTCOME_CONSTANT`にならず、`x`が説明できる範囲で通常どおり推定する）。"""
    data = df.with_columns(y=pl.lit(3.0))
    result = _fit(model, data, include_intercept=False)
    assert all(np.isfinite(list(result.std_errors.values())))


def test_constant_outcome_is_reported_before_the_first_stage(df):
    """IVでは、定数の`y`は第一段階より前に報告する（第一段階の失敗に隠れない）。"""
    data = df.with_columns(y=pl.lit(3.0), endog=2.0 * pl.col("z"))
    for estimator in ("2sls", "gmm"):
        with raises_error(ErrorCode.OUTCOME_CONSTANT):
            IV(
                data,
                "y",
                ["x"],
                ["endog"],
                ["z", "z2"],
                options=IVOptions(estimator=estimator),
            ).fit()


@pytest.mark.parametrize("model", ["OLS", "WLS"])
def test_constant_outcome_is_reported_before_a_collinear_design(model, df):
    """定数の`y`は入力だけから分かるので、設計行列の共線（QR分解）より先に報告する。"""
    data = df.with_columns(y=pl.lit(3.0), x2=2.0 * pl.col("x"))
    with raises_error(ErrorCode.OUTCOME_CONSTANT):
        _fit(model, data)


def test_constant_outcome_is_reported_before_the_cluster_count(df):
    """クラスター数の検証（`G > q`）は入力だけから分かるものとして先に行う。"""
    data = df.with_columns(y=pl.lit(3.0), few=pl.col("entity") % 2)
    with raises_error(ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES):
        OLS(
            data,
            "y",
            ["x", "x2", "z"],
            options=OLSOptions(cov_type="cluster", cluster="few"),
        ).fit()


def test_a_constant_endogenous_regressor_is_reported_in_its_first_stage(df):
    """定数の内生変数は、その第一段階の被説明変数が定数（切片あり）。第一段階の文脈つき。"""
    data = df.with_columns(
        endog=pl.lit(2.0), y=1.0 + pl.col("x") + pl.col("noise")
    )
    with raises_error(
        ErrorCode.OUTCOME_CONSTANT, stage="FIRST_STAGE", endog_name="endog"
    ):
        IV(data, "y", ["x"], ["endog"], ["z", "z2"], options=IVOptions()).fit()


def test_a_constant_endogenous_regressor_without_an_intercept_is_allowed(df):
    data = df.with_columns(
        endog=pl.lit(2.0), y=1.0 + pl.col("x") + pl.col("noise")
    )
    result = IV(
        data,
        "y",
        ["x"],
        ["endog"],
        ["z", "z2"],
        options=IVOptions(include_intercept=False),
    ).fit()
    assert all(np.isfinite(list(result.std_errors.values())))


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster", "dk"])
@pytest.mark.parametrize("model", ["FE", "RE"])
def test_outcome_constant_within_every_entity(model, cov_type, df):
    """個体内で定数の`y`（個体間でだけ違う）は、固定効果が全てを説明してwithin変動が残らない。
    `0.1`のように平均が丸め誤差を持つ値でも同じ。"""
    data = df.with_columns(y=0.1 * (pl.col("entity") + 1))
    with raises_error(ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY):
        _fit(model, data, cov_type)


@pytest.mark.parametrize("model", ["FE", "RE"])
def test_a_globally_constant_outcome_is_constant_within_entity(model, df):
    data = df.with_columns(y=pl.lit(4.0))
    with raises_error(ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY):
        _fit(model, data)


@pytest.mark.parametrize("model", ["FE", "RE"])
def test_outcome_explained_by_entity_and_time_effects_in_a_two_way_model(
    model, df
):
    """2-wayでは、`y = 個体効果 + 時点効果`も2方向のwithin変換後に0になる。"""
    data = df.with_columns(y=2.0 * pl.col("entity") - 0.5 * pl.col("time"))
    with raises_error(ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY):
        _fit(model, data, two_way=True)


@pytest.mark.parametrize("model", ["FE", "RE"])
def test_an_entity_level_outcome_with_a_small_within_variation_is_estimated(
    model, df
):
    """個体内の変動が相対`1e-6`あれば、個体内で定数とはみなさない。"""
    data = df.with_columns(
        y=3.0 * pl.col("entity")
        + pl.col("x")
        + 1e-6 * pl.col("noise") * pl.col("x2")
    )
    result = _fit(model, data)
    assert all(np.isfinite(list(result.std_errors.values())))


def test_the_regressor_check_comes_before_the_outcome_check_in_fe(df):
    """FEは説明変数の分散ゼロ検出（`REGRESSOR_NO_WITHIN_VARIATION`）を先に行う。"""
    data = df.with_columns(
        y=0.1 * (pl.col("entity") + 1), x2=pl.col("entity") * 1.0
    )
    with raises_error(ErrorCode.REGRESSOR_NO_WITHIN_VARIATION, column="x2"):
        _fit("FE", data)


def test_outcome_constant_within_entity_is_not_staged_in_re(df):
    """REの内部FEが報告するが、入力の事実なので文脈句（`stage`）は付けない。"""
    data = df.with_columns(y=pl.lit(4.0))
    with pytest.raises(Exception) as info:
        RE(data, "y", ["x"], ["entity"]).fit()
    assert getattr(info.value, "code", None) == (
        ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY
    )
    assert getattr(info.value, "params", {}).get("stage") is None
    assert not str(info.value).startswith("In the ")
