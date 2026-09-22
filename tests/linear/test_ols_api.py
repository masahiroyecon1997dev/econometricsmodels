"""OLS の成功パスの構造・API・オプション反映・`predict()`/`augment()` の検証。

確定済み設計（`docs/spec/ols-spec.md`）どおりの結果型・辞書キー・ラベルに
なっていること、`OLSOptions` の各フィールドが engine_pybind 経由で反映される
ことを確認する。`ValidationError`/`ComputationError` パスは
`test_ols_validation.py`、主リファレンスとの数値照合は `test_ols_reference.py`、
R クロスチェックは `test_ols_crosscheck.py`。

`predict()`/`augment()` のテストは（statsmodels との照合も含め）このファイルに
集約する（predict/augment は独立した API 面で、その statsmodels 照合は
スモーク級。手法間の predict の意味の違い〔OLS=予測値／Logit=確率〕を1ファイルで
対比できる）。両者の `ValidationError` パスのみ `test_ols_validation.py`。

上記4分類のいずれにも当てはまらない例外として、末尾に「クラスターロバストSEの
健全性チェック」を1本含む。これはリファレンス実装との数値照合ではなく、
真のクラスター内相関があるDGPでクラスターロバストSEが古典的SEより意図通り
大きくなることを確認する、本実装内で完結した統計的健全性の検証
（詳細は当該テストのdocstring参照）。
"""

from __future__ import annotations

from functools import partial

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from _assertions import assert_close
from _ols_helpers import our_fit, our_fit_cluster, sm_fit
from _tolerances import TOLERANCES
from econometricsmodels import OLS, OLSOptions

# predict() の statsmodels 照合も凍結フィクスチャ照合と同じ許容誤差
# （`_tolerances.py` の "ols_reference"）で行う。`_assertions.assert_close`
# （`tol = max(rtol*|ref|, atol)`）に統一し、独自の絶対誤差定数は持たない
# （`refactoring-candidates-2.md` 項目53/56）。
_assert_close = partial(
    assert_close,
    rtol=TOLERANCES["ols_reference"]["rtol"],
    atol=TOLERANCES["ols_reference"]["atol"],
)

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
        ("HAC", "hac"),
        ("Hac", "hac"),
        ("nonrobust", "nonrobust"),
        ("NONROBUST", "nonrobust"),
    ],
)
def test_cov_type_is_case_insensitive(dataset, cov_type, expected_label):
    """`cov_type`が大文字小文字を区別しないこと（`engine_pybind`側の
    `parse_cov_type`のRust単体テストと対になる、Python API境界での確認。
    テスト網羅性レビュー、Issue #231フェーズ4で判明した抜け。HACは
    `hac_lags`省略時の自動計算式で成功パスを確認する
    （テスト網羅性候補・項目35）。
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
    for i, (row, expected) in enumerate(zip(predicted, sm_res.fittedvalues)):
        _assert_close(row["predicted"], expected, f"predicted/{i}")


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
    for i, (row, exp) in enumerate(zip(predicted, expected)):
        _assert_close(row["predicted"], exp, f"predicted/{i}")


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

    for i, (row, exp) in enumerate(zip(predicted, expected)):
        _assert_close(row["predicted"], exp, f"predicted/{i}")


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
    for i, (row, (c, x2)) in enumerate(
        zip(predicted, [(100.0, 10.0), (200.0, 20.0)])
    ):
        expected = coef_const * c + coef_x2 * x2
        _assert_close(row["predicted"], expected, f"predicted/{i}")


def test_predict_new_data_structure(dataset):
    res = our_fit(dataset)
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    predicted = res.predict(new_data)

    assert isinstance(predicted, list)
    assert len(predicted) == 2
    for row in predicted:
        assert set(row.keys()) == {"predicted"}
        assert isinstance(row["predicted"], float)


# ── augment() ────────────────────────────────────────────────────


def test_augment_none_returns_training_data_with_predicted_column(dataset):
    """`augment(new_data=None)`が、学習データの全列＋`"predicted"`列を持つ
    DataFrameを、`predict()`と同じ予測値・元データと同じ行順で返すこと。
    """
    res = our_fit(dataset)

    augmented = res.augment()

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == dataset.height
    assert augmented.columns == [*dataset.columns, "predicted"]
    for col in dataset.columns:
        assert augmented[col].to_list() == dataset[col].to_list()

    expected = [row["predicted"] for row in res.predict()]
    assert augmented["predicted"].to_list() == expected


def test_augment_new_data_returns_new_data_with_predicted_column(dataset):
    """`augment(new_data)`が、`new_data`の全列＋`"predicted"`列を持つ
    DataFrameを、`predict(new_data)`と同じ予測値で返すこと。
    """
    res = our_fit(dataset)
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    augmented = res.augment(new_data)

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == 2
    assert augmented.columns == ["x1", "x2", "predicted"]
    assert augmented["x1"].to_list() == new_data["x1"].to_list()
    assert augmented["x2"].to_list() == new_data["x2"].to_list()

    expected = [row["predicted"] for row in res.predict(new_data)]
    assert augmented["predicted"].to_list() == expected


def test_augment_new_data_with_extra_column_preserves_it(dataset):
    """`new_data`が`x`列以外の余分な列（予測に使わない識別子列等）を含む場合、
    その列もそのまま`"predicted"`列と一緒に返されること。
    """
    res = our_fit(dataset)
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
        {"y": [3.0, 7.0, 9.0], "x1": [1.0, 2.0, 3.0]},
    )
    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=["x1"], options=options).fit()

    augmented_none = res.augment()
    expected_none = [row["predicted"] for row in res.predict()]
    assert augmented_none["predicted"].to_list() == expected_none

    new_data = pl.DataFrame({"x1": [10.0, 20.0]})
    augmented_new = res.augment(new_data)
    expected_new = [row["predicted"] for row in res.predict(new_data)]
    assert augmented_new["predicted"].to_list() == expected_new


# ── クラスターロバストSEの健全性チェック（真のクラスター内相関） ──────
#
# 上記のクラスター系テスト・test_ols_reference.py/test_ols_crosscheck.pyの
# クラスター系テストは、いずれも誤差i.i.d.なデータに疑似グループラベルを
# 後付けしたものであり、リファレンス実装（statsmodels/R）との数値一致を
# 検証する目的には十分だが、「クラスターロバストSEが真のクラスター内相関が
# ある状況で意図通り機能するか（通常のSEより適切に大きくなるか）」という
# 別種の健全性は検証していなかった（旧test-coverage-candidates.md項目12、
# 対応済みのため同ファイルからは削除済み、ユーザー確認済み）。以下はその
# 健全性のみを確認する専用テストであり、他のテストと異なりリファレンス
# 実装との数値比較は行わない。


def test_cluster_std_error_exceeds_classical_under_true_intra_cluster_correlation():
    """説明変数・誤差の両方にクラスター内相関を持たせたMoulton型DGPで、
    クラスターロバストSEが古典的SEより明確に大きくなることを確認する。

    説明変数x1がクラスターレベルの成分を持たない（個体ごとに独立な）DGPでは、
    誤差だけにクラスター内相関を持たせても、クラスターSEが古典的SEより
    小さくなることさえあることを実測確認済み。クラスターロバストSEの効果を
    検出するには説明変数自体もクラスター内相関を持つ必要がある（古典的な
    Moulton問題の構造）。
    seed=42固定でratio≈3.48（30シードでの実測範囲2.09〜4.49に対し、
    十分なマージンを持たせた閾値1.5を使う）。

    有効性検証について: `testing-policy.md`「property-basedテスト」節の
    本格的なバグ注入要件は`proptest`（engineクレート内）専用のため本テストには
    形式上適用されないが、参考として「assert文の機構自体が正しく機能するか」
    （classical同士の比較でratio=1.0を作りassertが落ちることを確認）のみ
    実施済み。`engine::linear::ols::cluster_cov_params`への実際のバグ注入に
    よる検出力の実証（他手法のproptestが満たす基準）は行っていない。
    """
    rng = np.random.default_rng(42)
    n_groups = 30
    group_size = 20
    n = n_groups * group_size
    group = np.repeat(np.arange(n_groups), group_size)

    # 説明変数x1: クラスターレベルの成分 + 個体ごとの誤差。
    x1_group = rng.normal(scale=1.0, size=n_groups)
    x1 = x1_group[group] + rng.normal(scale=0.5, size=n)

    # 誤差: クラスターレベルのランダム効果 + 個体ごとの誤差。
    u_g = rng.normal(0.0, 2.0, size=n_groups)
    e = u_g[group] + rng.normal(0.0, 1.0, size=n)

    y = 1.0 + 2.0 * x1 + e
    df = pl.DataFrame({"y": y, "x1": x1, "cluster": [str(g) for g in group]})

    classical = OLS(
        df, y="y", x=["x1"], options=OLSOptions(cov_type="classical")
    ).fit()
    cluster = OLS(
        df,
        y="y",
        x=["x1"],
        options=OLSOptions(cov_type="cluster", cluster_col="cluster"),
    ).fit()

    ratio = cluster.std_errors["x1"] / classical.std_errors["x1"]
    assert ratio > 1.5, f"ratio={ratio}"
