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
from _assertions import assert_close
from _tolerances import TOLERANCES
from econometricsmodels import (
    OLS,
    WLS,
    OLSOptions,
    WLSOptions,
    WlsResults,
)

# predict()のstatsmodels照合は主リファレンス照合と同じ許容誤差
# （`_tolerances.py`の"wls_reference"）で行う（`test_ols_api.py`と同じ方針、
# `refactoring-candidates-2.md`項目53/56「独自の絶対誤差定数は持たない」）。
_assert_close = partial(
    assert_close,
    rtol=TOLERANCES["wls_reference"]["rtol"],
    atol=TOLERANCES["wls_reference"]["atol"],
)

# ── OLSとの不変条件回帰テスト ───────────────────────────────────────
#
# 内部実装（sqrt(w)変換方式か将来別方式に変わるか等）が変わっても壊れないよう、
# public API経由でのみ比較する。


@pytest.mark.parametrize(
    "option_kwargs",
    [
        {},
        {"cov_type": "hc3"},
        {"cov_type": "cluster", "cluster_col": "cluster"},
        {"include_intercept": False},
    ],
)
def test_weight_one_matches_ols(dataset, option_kwargs):
    """重み=1のときWLSの結果がOLSの結果と完全一致すること。

    coef/se/t/p/CI/F統計量/n_obsは、WLSがOLSソルバーを`sqrt(weight)`変換した
    データにそのまま適用する実装であるため（weight=1なら変換が恒等写像になる）
    厳密な`==`で一致する。r_squared・log_likelihood（→aic/bic）・残差は、
    WLS側で元スケールのy・weightsから独立に計算し直す実装のため、加算順序
    等に由来する浮動小数点誤差レベルの差が生じうる（`engine/src/linear/wls.rs`
    の対応するRust単体テストで確認済みの挙動）。

    `OLSOptions`/`WLSOptions`はフィールド構成が同一の独立クラス（Issue #308）
    のため、同じ`option_kwargs`からそれぞれ構築して`OLS`/`WLS`に渡す。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))

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
        assert wls_res.t_stats[name] == ols_res.t_stats[name], name
        assert wls_res.p_values[name] == ols_res.p_values[name], name
        assert wls_res.conf_int[name] == ols_res.conf_int[name], name

    assert wls_res.f_statistic == ols_res.f_statistic
    assert wls_res.f_p_value == ols_res.f_p_value
    assert wls_res.n_obs == ols_res.n_obs
    assert wls_res.dep_var_name == ols_res.dep_var_name
    assert wls_res.cov_type == ols_res.cov_type

    assert abs(wls_res.r_squared - ols_res.r_squared) < 1e-9
    assert abs(wls_res.r_squared_adj - ols_res.r_squared_adj) < 1e-9
    assert abs(wls_res.log_likelihood - ols_res.log_likelihood) < 1e-9
    assert abs(wls_res.aic - ols_res.aic) < 1e-9
    assert abs(wls_res.bic - ols_res.bic) < 1e-9
    for wls_r, ols_r in zip(wls_res.residuals, ols_res.residuals):
        assert abs(wls_r - ols_r) < 1e-9


def test_weight_one_matches_ols_coef_table(dataset):
    """coef_table()の内容もOLSと一致すること（辞書系プロパティ以外の確認）。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))

    ols_table = OLS(df, y="y", x=["x1", "x2"]).fit().coef_table()
    wls_table = (
        WLS(df, y="y", x=["x1", "x2"], weight="weight").fit().coef_table()
    )

    assert [row["param"] for row in wls_table] == [
        row["param"] for row in ols_table
    ]
    for wls_row, ols_row in zip(wls_table, ols_table):
        assert wls_row["coef"] == ols_row["coef"]
        assert wls_row["std_err"] == ols_row["std_err"]


def test_weight_one_matches_ols_predict(dataset):
    """重み=1のとき、`predict()`（学習データ・新規データいずれも）が
    OLSの`predict()`と一致すること（Issue #132: 予測値は重みに関与しない、
    という設計の帰結を確認する）。

    学習データ（`new_data=None`）の計算経路自体はwls.rs（手動ループ、
    `original_scale_fitted_and_residuals`）とols.rs（faerの行列演算、
    `OlsEstimator::fitted_values`）で異なるため、丸め誤差レベルでの一致を
    確認する（`test_residuals_are_original_scale_not_weighted`と同じ理由）。
    一方、新規データに対する予測は両者とも同じ純粋関数
    `engine::linear::ols::predict_new_data`を呼ぶため、厳密な`==`で一致する。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))

    ols_res = OLS(df, y="y", x=["x1", "x2"]).fit()
    wls_res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()

    for wls_row, ols_row in zip(wls_res.predict(), ols_res.predict()):
        assert abs(wls_row["predicted"] - ols_row["predicted"]) < 1e-9

    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})
    for wls_row, ols_row in zip(
        wls_res.predict(new_data), ols_res.predict(new_data)
    ):
        assert wls_row["predicted"] == ols_row["predicted"]


# ── 成功パス・結果型 ──────────────────────────────────────────────


def test_default_options_use_classical(dataset):
    """`options`省略時は`WLSOptions()`の既定値（classical）が使われること。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert res.cov_type == "classical"


def test_residuals_are_original_scale_not_weighted(dataset):
    """`residuals`が元スケール（unweighted）であり、重み付き残差ではないこと。

    重みを大きく偏らせると、重み付き残差 `sqrt(w)(y-ŷ)` は元スケールの
    残差 `y-ŷ` と大きく異なる値になるはず。
    """
    n = dataset.height
    weight = [100.0] * n
    df = dataset.with_columns(pl.Series("weight", weight))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()

    # 元スケールの残差は「予測値からの素の差」であり、重み(=100)を掛けた
    # スケールにはならないはず（sqrt(100)=10倍にはならない）。
    ols_res = OLS(dataset, y="y", x=["x1", "x2"]).fit()
    for wls_r, ols_r in zip(res.residuals, ols_res.residuals):
        assert abs(wls_r - ols_r) < 1e-6


def test_result_is_wls_results_type(dataset):
    """`WLS.fit()`の返り値が`WlsResults`（`OlsResults`とは別型）であること。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert isinstance(res, WlsResults)


# ── API構造 ──────────────────────────────────────────────────────


def test_coef_table_structure(dataset):
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    table = res.coef_table()

    assert isinstance(table, list)
    assert len(table) == 3  # const, x1, x2
    expected_keys = {
        "param",
        "coef",
        "std_err",
        "t_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert expected_keys <= set(row.keys())
    assert [row["param"] for row in table] == ["const", "x1", "x2"]


def test_conf_int_structure(dataset):
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_t_stats_p_values_share_keys(dataset):
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    expected_keys = {"const", "x1", "x2"}

    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.t_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_nobs_and_dep_var_name(dataset):
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    assert res.n_obs == 100
    assert res.dep_var_name == "y"


# ── オプションの反映（OLSと同じ観点、共通化された経路の検証） ──────


def test_cov_type_label(dataset):
    """全cov_typeで`res.cov_type`が指定通り反映されること（OLSと同じ検証）。"""
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    for cov_type in ["classical", "hc0", "hc1", "hc2", "hc3"]:
        options = WLSOptions(cov_type=cov_type)
        res = WLS(
            df, y="y", x=["x1", "x2"], weight="weight", options=options
        ).fit()
        assert res.cov_type == cov_type

    cluster_options = WLSOptions(cov_type="cluster", cluster_col="cluster")
    cluster_res = WLS(
        df, y="y", x=["x1", "x2"], weight="weight", options=cluster_options
    ).fit()
    assert cluster_res.cov_type == "cluster"


@pytest.mark.parametrize(
    "cov_type, expected_label",
    [
        ("CLASSICAL", "classical"),
        ("Classical", "classical"),
        ("HC0", "hc0"),
        ("Hc1", "hc1"),
        ("HC2", "hc2"),
        ("hc3", "hc3"),
        ("HAC", "hac"),
        ("Hac", "hac"),
        ("nonrobust", "nonrobust"),
        ("NONROBUST", "nonrobust"),
    ],
)
def test_cov_type_is_case_insensitive(dataset, cov_type, expected_label):
    """`cov_type`が大文字小文字を区別しないこと（OLSの`test_ols_api.py::
    test_cov_type_is_case_insensitive`と同じ観点、共通化された経路の検証。
    HACは`hac_lags`省略時の自動計算式で成功パスを確認する
    （テスト網羅性候補・項目35）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = WLSOptions(cov_type=cov_type)
    res = WLS(
        df, y="y", x=["x1", "x2"], weight="weight", options=options
    ).fit()
    assert res.cov_type == expected_label


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_alias_for_classical(dataset, cov_type):
    """`"nonrobust"`が`"classical"`と同じ計算方法（標準誤差も一致）の
    エイリアスであること（OLSと同じ検証）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = WLSOptions(cov_type=cov_type)
    res = WLS(
        df, y="y", x=["x1", "x2"], weight="weight", options=options
    ).fit()

    classical_options = WLSOptions(cov_type="classical")
    classical_res = WLS(
        df, y="y", x=["x1", "x2"], weight="weight", options=classical_options
    ).fit()
    for name in res.param_names:
        assert res.std_errors[name] == classical_res.std_errors[name], name


def test_confidence_level_changes_interval_width(dataset):
    """`confidence_level`を下げると信頼区間が狭くなること（OLSと同じ検証、
    既定の0.95以外の値がengine_pybind経由で実際に反映されることの確認）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    wide = WLS(
        df,
        y="y",
        x=["x1", "x2"],
        weight="weight",
        options=WLSOptions(confidence_level=0.99),
    ).fit()
    narrow = WLS(
        df,
        y="y",
        x=["x1", "x2"],
        weight="weight",
        options=WLSOptions(confidence_level=0.80),
    ).fit()

    for name in ["const", "x1", "x2"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width, name


def test_hac_auto_lags_runs_and_returns_finite_std_errors(dataset):
    """`hac_lags`省略時（`None`、自動計算式）でもエラーなく動作すること
    （OLSと同じ検証。既存のHAC動作確認テストは`test_weight_one_matches_ols`
    経由で`hac_lags`を明示していなかったが、`cov_type="hac"`自体のテストは
    無かった）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    options = WLSOptions(cov_type="hac")  # hac_lags省略 = 自動計算
    res = WLS(
        df, y="y", x=["x1", "x2"], weight="weight", options=options
    ).fit()

    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


def test_hac_time_col_reorders_rows_before_computing_lags():
    """`time_col`を指定すると、DataFrameの行順に関わらず時系列順で
    ラグ付き自己共分散を計算すること（OLSと同じ検証データ・観点、重み=1で
    OLSと同じ結果になることを利用する）。
    """
    ordered_df = pl.DataFrame(
        {
            "y": [2.0, 4.0, 5.0, 4.0, 5.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "weight": [1.0] * 5,
        }
    )
    ordered_options = WLSOptions(cov_type="hac", hac_lags=1)
    ordered_res = WLS(
        ordered_df, y="y", x=["x1"], weight="weight", options=ordered_options
    ).fit()

    shuffled_df = pl.DataFrame(
        {
            "y": [5.0, 2.0, 5.0, 4.0, 4.0],
            "x1": [3.0, 1.0, 5.0, 2.0, 4.0],
            "time": [3.0, 1.0, 5.0, 2.0, 4.0],
            "weight": [1.0] * 5,
        }
    )
    shuffled_options = WLSOptions(cov_type="hac", hac_lags=1, time_col="time")
    shuffled_res = WLS(
        shuffled_df,
        y="y",
        x=["x1"],
        weight="weight",
        options=shuffled_options,
    ).fit()

    for name in ["const", "x1"]:
        assert (
            abs(shuffled_res.std_errors[name] - ordered_res.std_errors[name])
            < 1e-9
        ), name


# ── predict() ────────────────────────────────────────────────────
#
# 重み=1でのOLSとの一致は「OLSとの不変条件回帰テスト」節
# （test_weight_one_matches_ols_predict）で確認済み。ここでは重みが
# 予測値の計算に関与しないこと自体を、非自明な（1でない）重みを使った
# statsmodelsとの直接比較で確認する。


def _wls_weighted_dataset(dataset: pl.DataFrame) -> pl.DataFrame:
    """`dataset`に非自明な（1でない）重み列を付加する。"""
    weight = 1.0 / (1.0 + dataset["x1"].abs())
    return dataset.with_columns(weight.alias("weight"))


def test_predict_none_matches_statsmodels_fitted_values(dataset):
    """`predict(new_data=None)`が学習データに対するstatsmodels `sm.WLS`の
    fittedvaluesと一致すること（重みは1ではない）。
    """
    import numpy as np
    import statsmodels.api as sm

    df = _wls_weighted_dataset(dataset)
    x = sm.add_constant(
        np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy()])
    )
    sm_res = sm.WLS(
        df["y"].to_numpy(), x, weights=df["weight"].to_numpy()
    ).fit(use_t=True)

    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    predicted = res.predict()

    assert len(predicted) == len(df)
    for i, (row, expected) in enumerate(zip(predicted, sm_res.fittedvalues)):
        _assert_close(row["predicted"], expected, f"predicted/{i}")


def test_predict_new_data_matches_statsmodels(dataset):
    """新規データに対する`predict()`がstatsmodelsの`.predict()`と一致すること
    （重みは1ではない。列順を学習時と入れ替えて渡し、列名マッチングも
    合わせて確認する）。
    """
    import numpy as np
    import statsmodels.api as sm

    df = _wls_weighted_dataset(dataset)
    x = sm.add_constant(
        np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy()])
    )
    sm_res = sm.WLS(
        df["y"].to_numpy(), x, weights=df["weight"].to_numpy()
    ).fit(use_t=True)

    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    new_data = pl.DataFrame({"x2": [0.5, -1.0, 2.0], "x1": [1.0, 2.0, -0.5]})
    predicted = res.predict(new_data)

    sm_new_x = sm.add_constant(
        np.column_stack([new_data["x1"].to_numpy(), new_data["x2"].to_numpy()])
    )
    expected = sm_res.predict(sm_new_x)

    assert len(predicted) == 3
    for i, (row, exp) in enumerate(zip(predicted, expected)):
        _assert_close(row["predicted"], exp, f"predicted/{i}")


def test_predict_returns_predicted_key_only(dataset):
    """`predict()`の各行が`"predicted"`という1つのキーのみを持つこと
    （Issue #309: `"fitted"`固定は統計学的に不正確なため`"predicted"`に統一）。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()

    for row in res.predict():
        assert set(row.keys()) == {"predicted"}
        assert isinstance(row["predicted"], float)


# ── augment() ────────────────────────────────────────────────────


def test_augment_none_returns_training_data_with_predicted_column(dataset):
    """`augment(new_data=None)`が、学習データの全列＋`"predicted"`列を持つ
    DataFrameを、`predict()`と同じ予測値・元データと同じ行順で返すこと。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()

    augmented = res.augment()

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == df.height
    assert augmented.columns == [*df.columns, "predicted"]
    for col in df.columns:
        assert augmented[col].to_list() == df[col].to_list()

    expected = [row["predicted"] for row in res.predict()]
    assert augmented["predicted"].to_list() == expected


def test_augment_new_data_returns_new_data_with_predicted_column(dataset):
    """`augment(new_data)`が、`new_data`の全列＋`"predicted"`列を持つ
    DataFrameを、`predict(new_data)`と同じ予測値で返すこと。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    augmented = res.augment(new_data)

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == 2
    assert augmented.columns == ["x1", "x2", "predicted"]

    expected = [row["predicted"] for row in res.predict(new_data)]
    assert augmented["predicted"].to_list() == expected


def test_augment_new_data_with_extra_column_preserves_it(dataset):
    """`new_data`が`x`列以外の余分な列（予測に使わない識別子列等）を含む場合、
    その列もそのまま`"predicted"`列と一緒に返されること。
    """
    df = dataset.with_columns(pl.lit(1.0).alias("weight"))
    res = WLS(df, y="y", x=["x1", "x2"], weight="weight").fit()
    new_data = pl.DataFrame(
        {
            "id": ["a", "b"],
            "x1": [1.0, 2.0],
            "x2": [0.5, -0.5],
        }
    )

    augmented = res.augment(new_data)

    assert augmented.columns == ["id", "x1", "x2", "predicted"]
    assert augmented["id"].to_list() == ["a", "b"]


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
    expected_none = [row["predicted"] for row in res.predict()]
    assert augmented_none["predicted"].to_list() == expected_none

    new_data = pl.DataFrame({"x1": [10.0, 20.0]})
    augmented_new = res.augment(new_data)
    expected_new = [row["predicted"] for row in res.predict(new_data)]
    assert augmented_new["predicted"].to_list() == expected_new
