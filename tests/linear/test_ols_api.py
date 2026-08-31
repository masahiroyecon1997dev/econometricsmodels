"""OLS の成功パスの構造・API・オプション反映・`predict()` の検証。

確定済み設計（`docs/spec/ols-spec.md`）どおりの結果型・辞書キー・ラベルに
なっていること、`OLSOptions` の各フィールドが engine_pybind 経由で反映される
ことを確認する。`ValidationError`/`ComputationError` パスは
`test_ols_validation.py`、主リファレンスとの数値照合は `test_ols_reference.py`、
R クロスチェックは `test_ols_crosscheck.py`。

`predict()` のテストは（statsmodels との照合も含め）このファイルに集約する
（predict は独立した API 面で、その statsmodels 照合はスモーク級。手法間の
predict の意味の違い〔OLS=予測値／Logit=確率〕を1ファイルで対比できる）。
predict の `ValidationError` パスのみ `test_ols_validation.py`。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from _ols_helpers import ATOL_COEF, our_fit, our_fit_cluster, sm_fit
from econometricsmodels import OLS, OLSOptions

# ── 成功パス・結果型 ──────────────────────────────────────────────


def test_hac_runs_and_returns_finite_std_errors(dataset):
    """HACが（statsmodelsとの数値照合なしで）エラーなく動作すること。"""
    options = OLSOptions(cov_type="hac", hac_lags=2)
    res = OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()

    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


def test_hac_auto_lags_runs_and_returns_finite_std_errors(dataset):
    """`hac_lags`省略時（`None`、自動計算式）でもエラーなく動作すること

    （既存の`test_hac_runs_and_returns_finite_std_errors`は`hac_lags=2`を
    明示していたため、`None`がPython→Rustに正しく伝播する経路は
    未検証だった）。
    """
    options = OLSOptions(cov_type="hac")  # hac_lags省略 = 自動計算
    res = OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()

    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


def test_residuals_sum_near_zero(dataset):
    """残差の和が0に近いこと（定数項ありOLSの性質）。"""
    our_res = our_fit(dataset)
    assert abs(sum(our_res.residuals)) < 1e-8


# ── API構造 ──────────────────────────────────────────────────────


def test_coef_table_structure(dataset):
    res = our_fit(dataset)
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
    res = our_fit(dataset)
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_t_stats_p_values_share_keys(dataset):
    res = our_fit(dataset)
    expected_keys = {"const", "x1", "x2"}

    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.t_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_n_obs_and_dep_var_name(dataset):
    res = our_fit(dataset)
    assert res.n_obs == 100
    assert res.dep_var_name == "y"


# ── オプションの反映 ──────────────────────────────────────────────
#
# cov_type以外のOLSOptionsフィールド（include_intercept・confidence_level・
# hac_lags=None・time_col）が、engine_pybind側の列抽出・分岐ロジックを経て
# 正しく反映されることを確認する。


def test_cov_type_label(dataset):
    for cov_type in ["classical", "hc0", "hc1", "hc2", "hc3"]:
        res = our_fit(dataset, cov_type)
        assert res.cov_type == cov_type

    res = our_fit_cluster(dataset)
    assert res.cov_type == "cluster"


@pytest.mark.parametrize(
    "cov_type, expected_label",
    [
        ("CLASSICAL", "classical"),
        ("Classical", "classical"),
        ("HC0", "hc0"),
        ("Hc1", "hc1"),
        ("HC2", "hc2"),
        ("hc3", "hc3"),
        ("nonrobust", "nonrobust"),
        ("NONROBUST", "nonrobust"),
    ],
)
def test_cov_type_is_case_insensitive(dataset, cov_type, expected_label):
    """`cov_type`が大文字小文字を区別しないこと（`engine_pybind`側の
    `parse_cov_type`のRust単体テストと対になる、Python API境界での確認。
    テスト網羅性レビュー、Issue #231フェーズ4で判明した抜け）。
    """
    options = OLSOptions(cov_type=cov_type)
    res = OLS(dataset, y="y", x=["x1", "x2"], options=options).fit()
    assert res.cov_type == expected_label


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_alias_for_classical(dataset, cov_type):
    """`"nonrobust"`が`"classical"`と同じ計算方法（標準誤差も一致）の
    エイリアスであること。
    """
    res = our_fit(dataset, cov_type)
    classical_res = our_fit(dataset, "classical")
    for name in res.param_names:
        assert res.std_errors[name] == classical_res.std_errors[name], name


def test_default_options_use_classical():
    """`options`省略時は`OLSOptions()`の既定値（classical）が使われること。"""
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [1.0, 2.0, 3.5]})
    res = OLS(df, y="y", x=["x1"]).fit()
    assert res.cov_type == "classical"


def test_confidence_level_changes_interval_width(dataset):
    """`confidence_level`を下げると信頼区間が狭くなること

    （既定の0.95以外の値がengine_pybind経由で実際に反映されることの確認）。
    """
    wide = OLS(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=OLSOptions(confidence_level=0.99),
    ).fit()
    narrow = OLS(
        dataset,
        y="y",
        x=["x1", "x2"],
        options=OLSOptions(confidence_level=0.80),
    ).fit()

    for name in ["const", "x1", "x2"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width, name


def test_hac_time_col_reorders_rows_before_computing_lags():
    """`time_col`を指定すると、DataFrameの行順に関わらず時系列順で

    ラグ付き自己共分散を計算すること。データは`engine/src/linear/ols.rs`の
    `fit_computes_hac_std_errors_respecting_time_order`と同一（時系列順で
    x=[1..5], y=[2,4,5,4,5]をtime順=[3,1,5,2,4]にシャッフルして入力し、
    `time_col`無指定・時系列順の入力と同じ結果になることを確認する）。
    engine_pybindの`time_col`列抽出（`extract_f64_column`）を
    Python API境界から検証する。
    """
    ordered_df = pl.DataFrame(
        {"y": [2.0, 4.0, 5.0, 4.0, 5.0], "x1": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    ordered_options = OLSOptions(cov_type="hac", hac_lags=1)
    ordered_res = OLS(
        ordered_df, y="y", x=["x1"], options=ordered_options
    ).fit()

    shuffled_df = pl.DataFrame(
        {
            "y": [5.0, 2.0, 5.0, 4.0, 4.0],
            "x1": [3.0, 1.0, 5.0, 2.0, 4.0],
            "time": [3.0, 1.0, 5.0, 2.0, 4.0],
        }
    )
    shuffled_options = OLSOptions(cov_type="hac", hac_lags=1, time_col="time")
    shuffled_res = OLS(
        shuffled_df, y="y", x=["x1"], options=shuffled_options
    ).fit()

    for name in ["const", "x1"]:
        assert (
            abs(shuffled_res.std_errors[name] - ordered_res.std_errors[name])
            < 1e-9
        ), name


# ── predict() ────────────────────────────────────────────────────


def test_predict_none_matches_statsmodels_fitted_values(dataset):
    """`predict(new_data=None)`が学習データに対するstatsmodelsのfittedvaluesと一致すること。"""
    sm_res = sm_fit(dataset)
    res = our_fit(dataset)

    predicted = res.predict()

    assert len(predicted) == len(dataset)
    for row, expected in zip(predicted, sm_res.fittedvalues):
        assert row["fitted"] == pytest.approx(expected, abs=ATOL_COEF)


def test_predict_new_data_matches_statsmodels(dataset):
    """新規データに対する`predict()`がstatsmodelsの`.predict()`と一致すること。

    列順を学習時（x1, x2）と入れ替えて渡し、列名でマッチングされる
    （列順に依存しない）ことも合わせて確認する。
    """
    res = our_fit(dataset)
    sm_res = sm_fit(dataset)

    new_data = pl.DataFrame({"x2": [0.5, -1.0, 2.0], "x1": [1.0, 2.0, -0.5]})
    predicted = res.predict(new_data)

    sm_new_x = sm.add_constant(
        np.column_stack(
            [
                new_data["x1"].to_numpy(),
                new_data["x2"].to_numpy(),
            ]
        )
    )
    expected = sm_res.predict(sm_new_x)

    assert len(predicted) == 3
    for row, exp in zip(predicted, expected):
        assert row["fitted"] == pytest.approx(exp, abs=ATOL_COEF)


def test_predict_new_data_without_intercept_matches_statsmodels():
    """`include_intercept=False`でfitした場合のpredict()もstatsmodelsと一致すること。"""
    rng = np.random.default_rng(7)
    n = 50
    x1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * x1 + rng.normal(0.0, 0.1, n)
    df = pl.DataFrame({"y": y, "x1": x1})

    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=["x1"], options=options).fit()
    sm_res = sm.OLS(y, x1.reshape(-1, 1)).fit(use_t=True)

    new_x1 = np.array([1.0, 2.0, -3.0])
    new_data = pl.DataFrame({"x1": new_x1})
    predicted = res.predict(new_data)
    expected = sm_res.predict(new_x1.reshape(-1, 1))

    for row, exp in zip(predicted, expected):
        assert row["fitted"] == pytest.approx(exp, abs=ATOL_COEF)


def test_predict_with_include_intercept_false_and_x_named_const():
    """`include_intercept=False`かつ`x`に`"const"`という名前の列を含む場合でも
    predict()が正しく動作すること。

    `include_intercept=True`のときのみ`"const"`という列名との衝突チェックが
    働く仕様のため（`ols-spec.md`「API引数」）、`include_intercept=False`なら
    ユーザーが`"const"`という名前の（切片ではない）通常の説明変数を`x`に
    含めることは正当な入力。`predict()`の内部実装が誤って列名から
    「自動追加された切片列かどうか」を推測すると、この場合に値を無視して
    1.0固定にしてしまう回帰バグがあったため、固定用に追加。
    """
    df = pl.DataFrame(
        {
            "y": [3.0, 7.0, 9.0, 19.0, 11.0],
            "const": [2.0, 5.0, 1.0, 8.0, 3.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=["const", "x2"], options=options).fit()

    new_data = pl.DataFrame({"const": [100.0, 200.0], "x2": [10.0, 20.0]})
    predicted = res.predict(new_data)

    coef_const = res.params["const"]
    coef_x2 = res.params["x2"]
    for row, (c, x2) in zip(predicted, [(100.0, 10.0), (200.0, 20.0)]):
        expected = coef_const * c + coef_x2 * x2
        assert row["fitted"] == pytest.approx(expected, abs=ATOL_COEF)


def test_predict_new_data_structure(dataset):
    res = our_fit(dataset)
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    predicted = res.predict(new_data)

    assert isinstance(predicted, list)
    assert len(predicted) == 2
    for row in predicted:
        assert set(row.keys()) == {"fitted"}
        assert isinstance(row["fitted"], float)
