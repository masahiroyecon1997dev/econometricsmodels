"""WLS の成功パスの構造・API・オプション反映、および **OLS との不変条件
回帰テスト** の検証。

「共通化・パフォーマンス改善で実装の中身が変わっても結果が変わらない」ことを
public API 経由で保証する。`ValidationError` パスは `test_wls_validation.py`、
主リファレンス（statsmodels）との数値照合は `test_wls_reference.py`、
R クロスチェックは `test_wls_crosscheck.py`。

`predict()`/`augment()` のテストは（statsmodels との照合も含め）このファイルに
集約する（`test_ols_api.py`と同じ方針。predict/augment は独立した API 面で、
その statsmodels 照合はスモーク級）。両者の `ValidationError` パスのみ
`test_wls_validation.py`。
"""

from __future__ import annotations

from functools import partial

import polars as pl
import pytest
from _assertions import (
    assert_close,
    assert_hac_time_reorders_rows,
    assert_params_table_matches_attributes,
    assert_same_estimates,
)
from _constants import WIDE_X
from _helpers import (
    CASE_SPELLINGS,
    HAC_AUTO_LAG_SAMPLE_SIZES,
    ROW_TIME,
    hac_lag_frame,
    hac_time_for,
    reorder_rows,
    respell,
    with_row_time,
)
from _ols_helpers import (
    HAC_REORDER_ORDER,
    hac_reorder_frame,
    sm_wls_fit,
    wide_new_data,
)
from _tolerances import TOLERANCES
from econometricsmodels import (
    OLS,
    WLS,
    OLSOptions,
    WLSOptions,
    WLSResults,
)

from benchmark.common import hac_auto_lag
from benchmark.linear.constants import COV_TYPES, WIDE_BETA

# predict()のstatsmodels照合は主リファレンス照合と同じ許容誤差
# （`_tolerances_linear.py`の"wls_reference"）で行う（`test_ols_api.py`と同じ方針、
# 独自の絶対誤差定数は持たない）。実測の最大相対誤差は、`wide_dataset`の`weight`列
# （0.5〜1.5の独立な一様乱数）でも、説明変数に依存し範囲の広い重み
# （`REGRESSOR_WEIGHT`、最大/最小が約35）でも約5e-14の機械精度で、重みの範囲が
# 広がっても誤差は増えない。
_assert_close = partial(
    assert_close,
    rtol=TOLERANCES["wls_reference"]["rtol"],
    atol=TOLERANCES["wls_reference"]["atol"],
)

# 全観測に同じ重みを置くWLSの重み。重み付き残差`sqrt(w)(y-ŷ)`と元スケールの残差の
# 差が`sqrt(CONSTANT_WEIGHT)`倍として現れるよう、1から十分離した値にする。
CONSTANT_WEIGHT = 100.0

# 説明変数に依存し、範囲の広い重み`exp(0.5*x1 - 0.5*x2)`を持つ列の名前
# （`regressor_weight_dataset`）。`wide_dataset`では最大/最小が約35、
# x1・x2との相関の絶対値が約0.6。
REGRESSOR_WEIGHT = "regressor_weight"
# 上の性質が崩れていないことの確認用の下限（実測に余裕を持たせた値）。
MIN_REGRESSOR_WEIGHT_RATIO = 10.0
MIN_REGRESSOR_WEIGHT_ABS_CORR = 0.4


@pytest.fixture(scope="module")
def regressor_weight_dataset(wide_dataset: pl.DataFrame) -> pl.DataFrame:
    """`wide_dataset`に、複数の説明変数（x1・x2）に依存する広い範囲の重み列
    `REGRESSOR_WEIGHT`を足したもの。

    `wide_dataset`の`weight`列は0.5〜1.5の一様乱数で説明変数と独立なため、
    重み列と説明変数列の取り違えや、重みの変動が大きいときだけ現れる不具合は
    そのままでは検出しにくい。足した列がその目的に足りる性質（範囲の広さ・
    x1・x2との相関）を持つことを、ここで確認してから返す。
    """
    df = wide_dataset.with_columns(
        (0.5 * pl.col("x1") - 0.5 * pl.col("x2")).exp().alias(REGRESSOR_WEIGHT)
    )
    weight = df[REGRESSOR_WEIGHT]
    assert weight.max() / weight.min() >= MIN_REGRESSOR_WEIGHT_RATIO
    for name in ("x1", "x2"):
        corr = df.select(pl.corr(REGRESSOR_WEIGHT, name)).item()
        assert abs(corr) >= MIN_REGRESSOR_WEIGHT_ABS_CORR, name
    return df


# ── 成功パス・結果型 ──────────────────────────────────────────────────


def test_fit_succeeds_and_returns_wls_results(unit_weight_dataset):
    """`WLS.fit()`の返り値が`WLSResults`（`OLSResults`とは別型）であること。"""
    df = unit_weight_dataset
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert isinstance(res, WLSResults)


def test_default_options_use_classical(unit_weight_dataset):
    """`options`省略時は`WLSOptions()`の既定値（classical）が使われること。"""
    df = unit_weight_dataset
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert res.cov_type == "classical"


def test_residuals_are_original_scale_not_weighted(dataset):
    """`residuals`が元スケール（unweighted）であり、重み付き残差ではないこと。

    重みを大きく偏らせると、重み付き残差 `sqrt(w)(y-ŷ)` は元スケールの
    残差 `y-ŷ` と大きく異なる値になるはず。
    """
    df = dataset.with_columns(
        pl.Series("weight", [CONSTANT_WEIGHT] * dataset.height)
    )
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()

    # 元スケールの残差は「予測値からの素の差」であり、重みを掛けたスケール
    # （`sqrt(CONSTANT_WEIGHT)`倍）にはならないはず。
    ols_res = OLS(dataset, y="y", x=["x1", "x2"]).fit()
    for i, (wls_r, ols_r) in enumerate(zip(res.residuals, ols_res.residuals)):
        _assert_rounding_close(wls_r, ols_r, f"residuals[{i}]")


def test_params_follow_the_column_order_of_the_true_coefficients(
    wide_dataset,
):
    """説明変数8列で、各係数の推定値が自分の列の真の係数に近いこと
    （`test_ols_api.py`の同名テストと同じ検証。重みは`wide_dataset`の
    0.5〜1.5の一様乱数で、1ではない）。
    """
    res = WLS(wide_dataset, y="y", x=WIDE_X, weight="weight").fit()

    assert res.param_names == ["const", *WIDE_X]
    for name, true_value in zip(res.param_names, WIDE_BETA):
        assert abs(res.params[name] - true_value) < 5 * res.std_errors[name], (
            name
        )


# ── API構造 ───────────────────────────────────────────────────────────


def test_params_table_structure(wide_dataset):
    res = WLS(wide_dataset, y="y", x=WIDE_X, weight="weight").fit()
    table = res.params_table()
    assert_params_table_matches_attributes(res)

    assert isinstance(table, list)
    assert len(table) == 1 + len(WIDE_X)  # const, x1..x8
    expected_keys = {
        "param_name",
        "param",
        "std_error",
        "test_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert set(row.keys()) == expected_keys
    assert [row["param_name"] for row in table] == ["const", *WIDE_X]


def test_conf_int_structure(wide_dataset):
    res = WLS(wide_dataset, y="y", x=WIDE_X, weight="weight").fit()
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert list(ci) == ["const", *WIDE_X]
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_test_stats_p_values_share_keys(wide_dataset):
    res = WLS(wide_dataset, y="y", x=WIDE_X, weight="weight").fit()
    expected_keys = ["const", *WIDE_X]

    assert list(res.params) == expected_keys
    assert list(res.std_errors) == expected_keys
    assert list(res.test_stats) == expected_keys
    assert list(res.p_values) == expected_keys


def test_n_obs_and_dep_var_name(unit_weight_dataset):
    df = unit_weight_dataset
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert res.n_obs == 100
    assert res.dep_var_name == "y"


# ── オプションの反映 ──────────────────────────────────────────────────


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize("cov_type", [*COV_TYPES, "cluster"])
def test_cov_type_is_case_insensitive(unit_weight_dataset, cov_type, spelling):
    """全`cov_type`で、`res.cov_type`が正規形（小文字）で返り、どの大文字小文字の
    綴りでも正規形と同一の推定結果になること（OLSの
    `test_ols_api.py::test_cov_type_is_case_insensitive`と同じ観点、共通化された
    経路の検証）。HACは`hac_lags`省略時の自動計算式で確認する。
    """
    df = with_row_time(unit_weight_dataset)

    def fit(label: str):
        kwargs = {"cluster": "cluster"} if cov_type == "cluster" else {}
        options = WLSOptions(
            cov_type=label, **kwargs, **hac_time_for(cov_type)
        )
        return WLS(
            df, y="y", x=["x1", "x2"], weight="weight", options=options
        ).fit()

    res = fit(respell(cov_type, spelling))
    assert res.cov_type == cov_type
    assert_same_estimates(res, fit(cov_type))


def test_confidence_level_changes_interval_width(wide_dataset):
    """`confidence_level`を下げると信頼区間が狭くなること（OLSと同じ検証、
    既定の0.95以外の値がengine_pybind経由で実際に反映されることの確認）。
    """
    wide = WLS(
        wide_dataset,
        y="y",
        x=WIDE_X,
        weight="weight",
        options=WLSOptions(confidence_level=0.99),
    ).fit()
    narrow = WLS(
        wide_dataset,
        y="y",
        x=WIDE_X,
        weight="weight",
        options=WLSOptions(confidence_level=0.80),
    ).fit()

    for name in ["const", *WIDE_X]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width, name


def test_include_intercept_false_omits_const(wide_dataset):
    """`include_intercept=False`では`const`が出ず、`df_model`は傾き係数の数
    （切片あり・なしで同じ）になること（数値照合は`test_wls_reference.py`の
    `test_no_intercept_matches_statsmodels`）。
    """
    options = WLSOptions(include_intercept=False)
    res = WLS(
        wide_dataset, y="y", x=WIDE_X, weight="weight", options=options
    ).fit()
    assert res.param_names == WIDE_X
    assert res.df_model == len(WIDE_X)


def test_hac_auto_lags_runs_and_returns_finite_std_errors(unit_weight_dataset):
    """`hac_lags`省略時（`None`、自動計算式）でもエラーなく動作すること
    （OLSと同じ検証）。
    """
    df = with_row_time(unit_weight_dataset)
    # hac_lags省略 = 自動計算
    options = WLSOptions(cov_type="hac", hac_time=ROW_TIME)
    res = WLS(
        df, y="y", x=["x1", "x2"], weight="weight", options=options
    ).fit()

    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


def test_hac_time_reorders_rows_before_computing_lags():
    """`hac_time`を指定すると、DataFrameの行順に関わらず時系列順で
    ラグ付き自己共分散を計算すること（OLSと同じ検証データ・観点、重み=1で
    OLSと同じ結果になることを利用する）。
    """
    ordered_df = hac_reorder_frame().with_columns(pl.lit(1.0).alias("weight"))
    shuffled_df = reorder_rows(ordered_df, HAC_REORDER_ORDER)

    def fit_std_errors(df: pl.DataFrame, hac_time: str) -> dict[str, float]:
        options = WLSOptions(cov_type="hac", hac_lags=1, hac_time=hac_time)
        return (
            WLS(df, y="y", x=["x1"], weight="weight", options=options)
            .fit()
            .std_errors
        )

    assert_hac_time_reorders_rows(
        fit_std_errors(ordered_df, "time"),
        fit_std_errors(shuffled_df, "time"),
        fit_std_errors(with_row_time(shuffled_df), ROW_TIME),
        atol_invariant=TOLERANCES["wls_api"]["atol_hac_time_reorder"],
        min_rel_effect=TOLERANCES["wls_api"][
            "min_rel_hac_time_row_order_effect"
        ],
    )


# ── predict() ─────────────────────────────────────────────────────────


#
# 重み=1でのOLSとの一致は「OLSとの不変条件回帰テスト」節
# （test_weight_one_matches_ols_predict）で確認済み。ここでは重みが
# 予測値の計算に関与しないこと自体を、非自明な（1でない）重みを使った
# statsmodelsとの直接比較で確認する。


def test_predict_none_matches_statsmodels_fitted_values(wide_dataset):
    """`predict(new_data=None)`が学習データに対するstatsmodels `sm.WLS`の
    fittedvaluesと一致すること（重みは1ではない）。
    """
    df = wide_dataset
    sm_res = sm_wls_fit(df, x=WIDE_X)

    res = WLS(df, y="y", x=WIDE_X, weight="weight").fit()
    predicted = res.predict()

    assert len(predicted) == len(df)
    for i, (value, expected) in enumerate(zip(predicted, sm_res.fittedvalues)):
        _assert_close(value, expected, f"predicted/{i}")


def test_predict_new_data_matches_statsmodels(wide_dataset):
    """新規データに対する`predict()`がstatsmodelsの`.predict()`と一致すること
    （重みは1ではない。列順を学習時（x1..x8）と逆順にして渡し、列名マッチングも
    合わせて確認する）。
    """
    import statsmodels.api as sm

    df = wide_dataset
    sm_res = sm_wls_fit(df, x=WIDE_X)

    res = WLS(df, y="y", x=WIDE_X, weight="weight").fit()
    new_data = wide_new_data()
    predicted = res.predict(new_data.select(reversed(WIDE_X)))

    expected = sm_res.predict(sm.add_constant(new_data.to_numpy()))

    assert len(predicted) == 3
    for i, (value, exp) in enumerate(zip(predicted, expected)):
        _assert_close(value, exp, f"predicted/{i}")


def test_predict_none_with_regressor_dependent_weight_matches_statsmodels(
    regressor_weight_dataset,
):
    """重みが説明変数に依存し範囲が広い場合（`REGRESSOR_WEIGHT`）も、
    `predict(new_data=None)`がstatsmodels `sm.WLS`のfittedvaluesと一致すること。
    """
    df = regressor_weight_dataset
    sm_res = sm_wls_fit(df, x=WIDE_X, weight=REGRESSOR_WEIGHT)

    res = WLS(df, y="y", x=WIDE_X, weight=REGRESSOR_WEIGHT).fit()
    predicted = res.predict()

    assert len(predicted) == len(df)
    for i, (value, expected) in enumerate(zip(predicted, sm_res.fittedvalues)):
        _assert_close(value, expected, f"predicted/{i}")


def test_predict_new_data_with_regressor_dependent_weight_matches_statsmodels(
    regressor_weight_dataset,
):
    """重みが説明変数に依存し範囲が広い場合（`REGRESSOR_WEIGHT`）も、
    新規データに対する`predict()`がstatsmodelsの`.predict()`と一致すること
    （列順を学習時（x1..x8）と逆順にして渡す）。
    """
    import statsmodels.api as sm

    df = regressor_weight_dataset
    sm_res = sm_wls_fit(df, x=WIDE_X, weight=REGRESSOR_WEIGHT)

    res = WLS(df, y="y", x=WIDE_X, weight=REGRESSOR_WEIGHT).fit()
    new_data = wide_new_data()
    predicted = res.predict(new_data.select(reversed(WIDE_X)))

    expected = sm_res.predict(sm.add_constant(new_data.to_numpy()))

    assert len(predicted) == 3
    for i, (value, exp) in enumerate(zip(predicted, expected)):
        _assert_close(value, exp, f"predicted/{i}")


def test_predict_with_include_intercept_false_and_x_named_const():
    """`include_intercept=False`かつ`x`に`"const"`という名前の列を含む場合でも
    `predict(new_data)`が、その列の値を使って正しく計算されること。

    `include_intercept=True`のときのみ`"const"`列名との衝突チェックが働く仕様の
    ため（`wls-spec.md`「API引数」）、`include_intercept=False`ならユーザーが
    `"const"`という名前の（切片ではない）通常の説明変数を`x`に含めることは正当な
    入力。`predict()`が列名から「自動追加された切片列かどうか」を推測すると、この
    列の値を無視して1.0固定にしてしまう（OLS・Logit・Probitで過去にあった回帰バグと
    同じ型）。重みは予測に関与しないので、重みは1でない値にして確かめる。
    """
    df = pl.DataFrame(
        {
            "y": [3.0, 7.0, 9.0, 19.0, 11.0],
            "const": [2.0, 5.0, 1.0, 8.0, 3.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "weight": [1.0, 2.0, 0.5, 1.5, 1.0],
        }
    )
    options = WLSOptions(include_intercept=False)
    res = WLS(
        df, y="y", x=["const", "x2"], weight="weight", options=options
    ).fit()

    new_data = pl.DataFrame({"const": [100.0, 200.0], "x2": [10.0, 20.0]})
    predicted = res.predict(new_data)

    coef_const = res.params["const"]
    coef_x2 = res.params["x2"]
    assert len(predicted) == 2
    for i, (value, (c, x2)) in enumerate(
        zip(predicted, [(100.0, 10.0), (200.0, 20.0)], strict=True)
    ):
        expected = coef_const * c + coef_x2 * x2
        _assert_close(value, expected, f"predicted/{i}")


def test_predict_returns_list_of_floats(unit_weight_dataset):
    """`predict()`が観測値ごとの`float`のリストを返すこと（`residuals`と同じ形）。"""
    df = unit_weight_dataset
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()

    predicted = res.predict()
    assert isinstance(predicted, list)
    assert len(predicted) == df.height
    assert all(isinstance(value, float) for value in predicted)


# ── augment() ─────────────────────────────────────────────────────────


def test_augment_none_returns_training_data_with_predicted_column(
    wide_dataset,
):
    """`augment(new_data=None)`が、学習データの全列＋`"predicted"`列を持つ
    DataFrameを、`predict()`と同じ予測値・元データと同じ行順で返すこと。
    """
    df = wide_dataset
    res = WLS(df, y="y", x=WIDE_X, weight="weight").fit()

    augmented = res.augment()

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == df.height
    assert augmented.columns == [*df.columns, "predicted"]
    for col in df.columns:
        assert augmented[col].to_list() == df[col].to_list()

    expected = res.predict()
    assert augmented["predicted"].to_list() == expected


def test_augment_new_data_returns_new_data_with_predicted_column(
    wide_dataset,
):
    """`augment(new_data)`が、`new_data`の全列＋`"predicted"`列を持つ
    DataFrameを、`predict(new_data)`と同じ予測値で返すこと。
    """
    res = WLS(wide_dataset, y="y", x=WIDE_X, weight="weight").fit()
    new_data = wide_new_data(2)

    augmented = res.augment(new_data)

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == 2
    assert augmented.columns == [*WIDE_X, "predicted"]
    for col in WIDE_X:
        assert augmented[col].to_list() == new_data[col].to_list()

    expected = res.predict(new_data)
    assert augmented["predicted"].to_list() == expected


def test_augment_new_data_with_extra_column_preserves_it(wide_dataset):
    """`new_data`が`x`列以外の余分な列（予測に使わない識別子列等）を含む場合、
    その列もそのまま`"predicted"`列と一緒に返されること。
    """
    res = WLS(wide_dataset, y="y", x=WIDE_X, weight="weight").fit()
    new_data = wide_new_data(2).insert_column(0, pl.Series("id", ["a", "b"]))

    augmented = res.augment(new_data)

    assert augmented.columns == ["id", *WIDE_X, "predicted"]
    assert augmented["id"].to_list() == ["a", "b"]


def test_augment_none_with_regressor_dependent_weight_matches_statsmodels(
    regressor_weight_dataset,
):
    """重みが説明変数に依存し範囲が広い場合（`REGRESSOR_WEIGHT`）も、
    `augment(new_data=None)`の`"predicted"`列がstatsmodels `sm.WLS`の
    fittedvaluesと一致し、学習データの全列（重み列を含む）が行順どおり
    保たれること。
    """
    df = regressor_weight_dataset
    sm_res = sm_wls_fit(df, x=WIDE_X, weight=REGRESSOR_WEIGHT)

    res = WLS(df, y="y", x=WIDE_X, weight=REGRESSOR_WEIGHT).fit()
    augmented = res.augment()

    assert augmented.columns == [*df.columns, "predicted"]
    for col in df.columns:
        assert augmented[col].to_list() == df[col].to_list(), col
    for i, (value, expected) in enumerate(
        zip(augmented["predicted"], sm_res.fittedvalues)
    ):
        _assert_close(value, expected, f"predicted/{i}")


def test_augment_new_data_with_regressor_dependent_weight_matches_statsmodels(
    regressor_weight_dataset,
):
    """重みが説明変数に依存し範囲が広い場合（`REGRESSOR_WEIGHT`）も、
    `augment(new_data)`の`"predicted"`列がstatsmodelsの`.predict()`と一致し、
    `new_data`の全列が渡した順のまま保たれること（列順は学習時と逆順）。
    """
    import statsmodels.api as sm

    df = regressor_weight_dataset
    sm_res = sm_wls_fit(df, x=WIDE_X, weight=REGRESSOR_WEIGHT)

    res = WLS(df, y="y", x=WIDE_X, weight=REGRESSOR_WEIGHT).fit()
    new_data = wide_new_data()
    reversed_new_data = new_data.select(reversed(WIDE_X))
    augmented = res.augment(reversed_new_data)

    assert augmented.columns == [*reversed(WIDE_X), "predicted"]
    for col in WIDE_X:
        assert augmented[col].to_list() == new_data[col].to_list(), col
    expected = sm_res.predict(sm.add_constant(new_data.to_numpy()))
    assert augmented.height == 3
    for i, (value, exp) in enumerate(zip(augmented["predicted"], expected)):
        _assert_close(value, exp, f"predicted/{i}")


def test_augment_without_intercept_matches_predict():
    """`include_intercept=False`でfitした場合も`augment()`が`predict()`と
    同じ予測値を返すこと（`augment()`はRust側で`predict()`とは別に
    `has_intercept`分岐を実装しているため、個別に確認する）。
    """
    df = pl.DataFrame(
        {
            "y": [3.0, 7.0, 9.0],
            "x1": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
        },
    )
    options = WLSOptions(include_intercept=False)
    res = WLS(df, y="y", x=["x1"], weight="weight", options=options).fit()

    augmented_none = res.augment()
    expected_none = res.predict()
    assert augmented_none["predicted"].to_list() == expected_none

    new_data = pl.DataFrame({"x1": [10.0, 20.0]})
    augmented_new = res.augment(new_data)
    expected_new = res.predict(new_data)
    assert augmented_new["predicted"].to_list() == expected_new


# ── OLSとの不変条件回帰テスト ─────────────────────────────────────────


#
# 内部実装（sqrt(w)変換方式か将来別方式に変わるか等）が変わっても壊れないよう、
# public API経由でのみ比較する。
#
# WLSとOLSで計算経路が異なる統計量（r_squared・残差・予測値等）は、加算順序に
# 由来する丸め誤差レベルでの一致を、相対誤差なし（rtol=0）の絶対誤差で確認する。


_assert_rounding_close = partial(
    assert_close, rtol=0.0, atol=TOLERANCES["wls_api"]["atol_ols_invariant"]
)


@pytest.mark.parametrize(
    "option_kwargs",
    [
        {},
        {"cov_type": "hc3"},
        {"cov_type": "cluster", "cluster": "cluster"},
        {"include_intercept": False},
    ],
)
def test_weight_one_matches_ols(unit_weight_dataset, option_kwargs):
    """重み=1のときWLSの結果がOLSの結果と完全一致すること。

    params/std_errors/test_stats/p_values/conf_int/F統計量/n_obsは、WLSがOLSソルバーを`sqrt(weight)`変換した
    データにそのまま適用する実装であるため（weight=1なら変換が恒等写像になる）
    厳密な`==`で一致する。r_squared・log_likelihood（→aic/bic）・残差は、
    WLS側で元スケールのy・weightsから独立に計算し直す実装のため、加算順序
    等に由来する浮動小数点誤差レベルの差が生じうる（`engine/src/linear/wls.rs`
    の対応するRust単体テストで確認済みの挙動）。

    `OLSOptions`/`WLSOptions`はフィールド構成が同一の独立クラスのため、
    同じ`option_kwargs`からそれぞれ構築して`OLS`/`WLS`に渡す。
    """
    df = unit_weight_dataset

    ols_res = OLS(
        df, y="y", x=["x1", "x2"], options=OLSOptions(**option_kwargs)
    ).fit()
    wls_res = WLS(
        df,
        y="y",
        x=["x1", "x2"],
        weight="weight",
        options=WLSOptions(**option_kwargs),
    ).fit()

    assert wls_res.param_names == ols_res.param_names
    for name in ols_res.param_names:
        assert wls_res.params[name] == ols_res.params[name], name
        assert wls_res.std_errors[name] == ols_res.std_errors[name], name
        assert wls_res.test_stats[name] == ols_res.test_stats[name], name
        assert wls_res.p_values[name] == ols_res.p_values[name], name
        assert wls_res.conf_int[name] == ols_res.conf_int[name], name

    assert wls_res.f_statistic == ols_res.f_statistic
    assert wls_res.f_p_value == ols_res.f_p_value
    assert wls_res.n_obs == ols_res.n_obs
    assert wls_res.dep_var_name == ols_res.dep_var_name
    assert wls_res.cov_type == ols_res.cov_type

    for stat in ("r_squared", "adj_r_squared", "log_likelihood", "aic", "bic"):
        _assert_rounding_close(
            getattr(wls_res, stat), getattr(ols_res, stat), stat
        )
    for i, (wls_r, ols_r) in enumerate(
        zip(wls_res.residuals, ols_res.residuals)
    ):
        _assert_rounding_close(wls_r, ols_r, f"residuals[{i}]")


def test_weight_one_matches_ols_params_table(unit_weight_dataset):
    """params_table()の内容もOLSと一致すること（辞書系プロパティ以外の確認）。"""
    df = unit_weight_dataset

    ols_table = OLS(df, y="y", x=["x1", "x2"]).fit().params_table()
    wls_table = (
        WLS(df, y="y", x=["x1", "x2"], weight="weight").fit().params_table()
    )

    assert [row["param_name"] for row in wls_table] == [
        row["param_name"] for row in ols_table
    ]
    for wls_row, ols_row in zip(wls_table, ols_table):
        assert wls_row["param"] == ols_row["param"]
        assert wls_row["std_error"] == ols_row["std_error"]


def test_weight_one_matches_ols_predict(unit_weight_dataset):
    """重み=1のとき、`predict()`（学習データ・新規データいずれも）が
    OLSの`predict()`と一致すること（予測値は重みに関与しない、
    という設計の帰結を確認する）。

    学習データ（`new_data=None`）の計算経路自体はwls.rs（手動ループ、
    `original_scale_fitted_and_residuals`）とols.rs（faerの行列演算、
    `OlsEstimator::fitted_values`）で異なるため、丸め誤差レベルでの一致を
    確認する（`test_residuals_are_original_scale_not_weighted`と同じ理由）。
    一方、新規データに対する予測は両者とも同じ純粋関数
    `engine::linear::ols::predict_new_data`を呼ぶため、厳密な`==`で一致する。
    """
    df = unit_weight_dataset

    ols_res = OLS(df, y="y", x=["x1", "x2"]).fit()
    wls_res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()

    for i, (wls_val, ols_val) in enumerate(
        zip(wls_res.predict(), ols_res.predict())
    ):
        _assert_rounding_close(wls_val, ols_val, f"predict()[{i}]")

    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})
    for wls_val, ols_val in zip(
        wls_res.predict(new_data), ols_res.predict(new_data)
    ):
        assert wls_val == ols_val


# ── HACの実使用ラグ数（hac_lags_used） ────────────────────────────────


@pytest.mark.parametrize("n", HAC_AUTO_LAG_SAMPLE_SIZES)
def test_hac_lags_used_matches_python_auto_lag_formula(n):
    """`hac_lags`省略時の`hac_lags_used`が、Python側の独立実装
    （`benchmark.common.hac_auto_lag`）と複数の標本サイズで一致すること。
    """
    options = WLSOptions(cov_type="hac", hac_time=ROW_TIME)
    res = WLS(
        hac_lag_frame(n),
        y="y",
        x=["x1", "x2"],
        weight="weight",
        options=options,
    ).fit()
    assert res.hac_lags_used == hac_auto_lag(n)


def test_hac_lags_used_echoes_explicit_hac_lags_and_is_none_unless_hac():
    df = hac_lag_frame(50)
    kwargs = {"y": "y", "x": ["x1", "x2"], "weight": "weight"}
    hac = WLSOptions(cov_type="hac", hac_lags=3, hac_time=ROW_TIME)
    assert WLS(df, options=hac, **kwargs).fit().hac_lags_used == 3
    hc1 = WLSOptions(cov_type="hc1")
    assert WLS(df, options=hc1, **kwargs).fit().hac_lags_used is None
