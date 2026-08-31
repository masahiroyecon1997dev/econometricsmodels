"""IV の成功パスの構造・API・オプション反映の検証。

確定済み設計（`docs/planning/specs/iv-api-design.md`）どおりの結果型・辞書キー・
ラベルになっていること、`IvOptions` の各フィールド（2SLS/GMM 共通・GMM 固有）が
engine_pybind 経由で反映されることを確認する。`ValidationError`/
`ComputationError` パスは `test_iv_validation.py`、主リファレンス（linearmodels）
との数値照合は `test_iv_reference.py`（2SLS）・`test_iv_gmm_reference.py`（GMM）、
R クロスチェックは `test_iv_crosscheck.py`（OLS/WLS/Logit/Probit の
`test_<手法>_api.py` 等と同じ4分割、`refactoring-candidates-2.md` 項目68）。

`iv_dataset`/`clustered_dataset` フィクスチャと `our_fit` ヘルパーは
`tests/iv/conftest.py`／`tests/iv/_iv_helpers.py`。
"""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import DATA_DIR
from _iv_helpers import our_fit
from econometricsmodels import IV, IvOptions, IvResults

# ── 成功パス・結果型 ──────────────────────────────────────────────


def test_fit_succeeds_and_returns_iv_results(iv_dataset):
    res = our_fit(iv_dataset)
    assert isinstance(res, IvResults)


def test_default_options_use_2sls_classical(iv_dataset):
    """`options`省略時は`IvOptions()`の既定値（method="2sls", classical）が
    使われ、2SLSは常に`converged=True`/`n_iterations=1`（閉形式・非反復）。
    """
    res = our_fit(iv_dataset)
    assert res.cov_type == "classical"
    assert res.converged
    assert res.n_iterations == 1


def test_gmm_method_runs_and_converges(iv_dataset):
    """`method="gmm"`の既定オプション（`gmm_iterations=2`、2-step efficient
    GMM）が成功パスで動作すること。
    """
    options = IvOptions(method="gmm")
    res = our_fit(iv_dataset, options=options)
    assert res.converged
    assert res.n_iterations == 2


def test_cluster_g2_boundary_succeeds_when_x_exog_is_empty():
    """`G=2`クラスター・`x_exog=[]`・丁度識別（`instruments`1本）という、
    ベンチマーク作成中に発見した`ComputationError`（`engine/src/iv/CLAUDE.md`
    「修正済み」参照）の再現条件そのもの。第一段階回帰の`has_intercept`の
    取り違えが原因で、真の傾き係数数`q=1`（`z1`のみ）のところ`q=2`（定数項も
    含めて誤ってカウント）になり、`G=2`クラスターの構造的特異性
    （`rank(Ŝ)≤G-1=1`）で必ず失敗していた（`without_baked_in_intercept`の
    導入で修正、`engine/src/iv/two_sls.rs`の同名Rustテストと対）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_g2.csv")
    df = df.with_columns((pl.int_range(pl.len()) % 2).alias("cluster_group"))
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = IV(
        df,
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()
    assert res.params["endog1"] != 0.0


def test_residuals_length_matches_n_obs(iv_dataset):
    res = our_fit(iv_dataset)
    assert len(res.residuals) == res.n_obs == iv_dataset.height


# ── API構造 ──────────────────────────────────────────────────────


def test_param_names_order(iv_dataset):
    """`param_names`は定数項→x_exog→x_endogの順（`IvInput::from_columns`の
    設計行列の列順、`docs/planning/specs/iv-api-design.md`参照）。
    """
    res = our_fit(iv_dataset)
    assert res.param_names == ["const", "x1", "endog1"]


def test_coef_table_structure(iv_dataset):
    res = our_fit(iv_dataset)
    table = res.coef_table()

    assert isinstance(table, list)
    assert len(table) == 3  # const, x1, endog1
    expected_keys = {
        "param",
        "coef",
        "std_err",
        "stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert expected_keys <= set(row.keys())
    assert [row["param"] for row in table] == ["const", "x1", "endog1"]


def test_conf_int_structure(iv_dataset):
    res = our_fit(iv_dataset)
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert set(ci.keys()) == {"const", "x1", "endog1"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_stats_p_values_share_keys(iv_dataset):
    res = our_fit(iv_dataset)
    expected_keys = {"const", "x1", "endog1"}

    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_n_obs_and_dep_var_name(iv_dataset):
    res = our_fit(iv_dataset)
    assert res.n_obs == iv_dataset.height
    assert res.dep_var_name == "y"


def test_first_stage_structure(iv_dataset):
    """`first_stage()`は`x_endog`の変数名をキーにした`OlsResults`の辞書を返す。"""
    from econometricsmodels import OlsResults

    res = our_fit(iv_dataset)
    first_stage = res.first_stage()

    assert set(first_stage.keys()) == {"endog1"}
    assert isinstance(first_stage["endog1"], OlsResults)
    # 第一段階の設計行列はconst, x_exog, instruments（x_endogは含まない）。
    assert set(first_stage["endog1"].param_names) == {
        "const",
        "x1",
        "z1",
        "z2",
    }


def test_weak_instrument_f_statistics_keyed_by_endog_name(iv_dataset):
    res = our_fit(iv_dataset)
    assert set(res.weak_instrument_f_statistics.keys()) == {"endog1"}
    assert res.weak_instrument_f_statistics["endog1"] > 0.0


def test_weak_instrument_f_statistics_empty_when_no_endog(iv_dataset):
    res = IV(
        iv_dataset, y="y", x_exog=["x1"], x_endog=[], instruments=[]
    ).fit()
    assert res.weak_instrument_f_statistics == {}
    # x_endog=[]のとき、overid/wu_hausmanも意味を持たないためNone
    # （`iv.py`のdocstring参照）。
    assert res.overid_statistic is None
    assert res.wu_hausman_statistic is None


def test_overid_statistic_present_when_over_identified(iv_dataset):
    """`instruments`が2本、`x_endog`が1本（過剰識別）なので`overid_statistic`
    はNoneにならない（2SLSのSargan検定）。
    """
    res = our_fit(iv_dataset)
    assert res.overid_statistic is not None
    assert res.overid_p_value is not None


def test_overid_statistic_none_when_just_identified(iv_dataset):
    res = IV(
        iv_dataset,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1"],
    ).fit()
    assert res.overid_statistic is None
    assert res.overid_p_value is None


def test_overid_statistic_present_for_gmm_hansen_j(iv_dataset):
    """`method="gmm"`でも過剰識別なら`overid_statistic`（Hansen J検定）は
    Noneにならない。
    """
    options = IvOptions(method="gmm")
    res = our_fit(iv_dataset, options=options)
    assert res.overid_statistic is not None
    assert res.overid_p_value is not None


def test_wu_hausman_is_none_for_gmm(iv_dataset):
    """`wu_hausman_statistic`/`wu_hausman_p_value`は`method="gmm"`では常に
    `None`（`GmmEstimator`はWu-Hausman検定を実装しない、
    `engine_pybind/src/iv/CLAUDE.md`参照）。
    """
    options = IvOptions(method="gmm")
    res = our_fit(iv_dataset, options=options)
    assert res.wu_hausman_statistic is None
    assert res.wu_hausman_p_value is None


def test_wu_hausman_is_not_none_for_2sls(iv_dataset):
    res = our_fit(iv_dataset)
    assert res.wu_hausman_statistic is not None
    assert res.wu_hausman_p_value is not None


# ── オプションの反映 ──────────────────────────────────────────────


def test_cov_type_label(iv_dataset):
    for cov_type in ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]:
        options = IvOptions(cov_type=cov_type)
        res = our_fit(iv_dataset, options=options)
        assert res.cov_type == cov_type


def test_cluster_cov_type_label(clustered_dataset):
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = our_fit(clustered_dataset, options=options)
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
        ("nonrobust", "nonrobust"),
        ("NONROBUST", "nonrobust"),
    ],
)
def test_cov_type_is_case_insensitive(iv_dataset, cov_type, expected_label):
    """`cov_type`が大文字小文字を区別しないこと（`engine_pybind`側の
    `parse_iv_cov_type`のRust実装と対になる、Python API境界での確認。
    OLS/WLS/Logit/Probitの`test_cov_type_is_case_insensitive`と同型、
    `testing-completeness-reviewer`指摘、Issue #231フェーズ4）。
    """
    options = IvOptions(cov_type=cov_type)
    res = our_fit(iv_dataset, options=options)
    assert res.cov_type == expected_label


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_alias_for_classical(iv_dataset, cov_type):
    """`"nonrobust"`が`"classical"`と同じ計算方法（標準誤差も一致）のエイリアス
    であること（OLS/WLS/Logit/Probitの同名テストと同型、Issue #231フェーズ4）。
    """
    res = our_fit(iv_dataset, options=IvOptions(cov_type=cov_type))
    classical_res = our_fit(
        iv_dataset, options=IvOptions(cov_type="classical")
    )
    for name in res.param_names:
        assert res.std_errors[name] == classical_res.std_errors[name], name


@pytest.mark.parametrize(
    "weight_type, expected",
    [
        ("UNADJUSTED", "unadjusted"),
        ("Unadjusted", "unadjusted"),
        ("ROBUST", "robust"),
        ("KERNEL", "kernel"),
        ("homoskedastic", "unadjusted"),
        ("HOMOSKEDASTIC", "unadjusted"),
        ("heteroskedastic", "robust"),
        ("HETEROSKEDASTIC", "robust"),
    ],
)
def test_weight_type_is_case_insensitive_and_aliased(
    iv_dataset, weight_type, expected
):
    """`weight_type`が大文字小文字を区別しないこと、および`"homoskedastic"`/
    `"heteroskedastic"`が`"unadjusted"`/`"robust"`のエイリアスであること
    （`engine_pybind`側の`parse_weight_type`と対になる、Python API境界での確認。
    `testing-completeness-reviewer`指摘、Issue #231フェーズ4）。
    """
    options = IvOptions(method="gmm", weight_type=weight_type)
    res = our_fit(iv_dataset, options=options)

    canonical_options = IvOptions(method="gmm", weight_type=expected)
    canonical_res = our_fit(iv_dataset, options=canonical_options)
    for name in res.param_names:
        assert res.params[name] == canonical_res.params[name], name


def test_confidence_level_changes_interval_width(iv_dataset):
    wide = our_fit(iv_dataset, options=IvOptions(confidence_level=0.99))
    narrow = our_fit(iv_dataset, options=IvOptions(confidence_level=0.80))

    for name in ["const", "x1", "endog1"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width, name


def test_hac_auto_lags_runs_and_returns_finite_std_errors(iv_dataset):
    """`hac_lags`省略時（`None`、自動計算式）でもエラーなく動作すること。"""
    options = IvOptions(cov_type="hac")
    res = our_fit(iv_dataset, options=options)
    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


def test_hac_time_col_reorders_rows_before_computing_lags():
    """`time_col`を指定すると、DataFrameの行順に関わらず時系列順で
    ラグ付き自己共分散を計算すること（`test_ols_api.py`の同名テストと同じ発想を
    IVに適用、engine_pybindの`time_col`列抽出経路をAPI境界から検証する）。
    """
    ordered_df = pl.DataFrame(
        {
            "y": [2.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 7.0],
            "endog1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "z1": [1.5, 2.5, 2.0, 4.5, 3.0, 5.5, 6.0, 7.5],
        }
    )
    ordered_options = IvOptions(cov_type="hac", hac_lags=1)
    ordered_res = IV(
        ordered_df,
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=ordered_options,
    ).fit()

    perm = [3, 1, 5, 2, 4, 0, 7, 6]
    shuffled_df = pl.DataFrame(
        {
            "y": [ordered_df["y"][i] for i in perm],
            "endog1": [ordered_df["endog1"][i] for i in perm],
            "z1": [ordered_df["z1"][i] for i in perm],
            "time": [float(i) for i in perm],
        }
    )
    shuffled_options = IvOptions(cov_type="hac", hac_lags=1, time_col="time")
    shuffled_res = IV(
        shuffled_df,
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=shuffled_options,
    ).fit()

    for name in ["const", "endog1"]:
        assert (
            abs(shuffled_res.std_errors[name] - ordered_res.std_errors[name])
            < 1e-9
        ), name


def test_include_intercept_false_omits_const(iv_dataset):
    """`include_intercept=False`だと`param_names`に`"const"`が含まれない
    （`x_endog`/`instruments`には元々自動で切片が付かない仕様と対称）。
    """
    options = IvOptions(include_intercept=False)
    res = our_fit(iv_dataset, options=options)
    assert res.param_names == ["x1", "endog1"]


@pytest.mark.parametrize(
    "weight_type", ["unadjusted", "robust", "cluster", "kernel"]
)
def test_gmm_weight_type_options_run(
    iv_dataset, clustered_dataset, weight_type
):
    """`method="gmm"`の`weight_type`各値が成功パスで動作すること（数値照合は
    `test_iv_gmm_reference.py`）。`cluster`/`kernel`は`cov_type`と同じ
    `cluster_col`/`hac_lags`フィールドを共用する仕様
    （`engine_pybind/src/iv/CLAUDE.md`参照）。
    """
    df = clustered_dataset if weight_type == "cluster" else iv_dataset
    kwargs = (
        {"cluster_col": "cluster_group"} if weight_type == "cluster" else {}
    )
    options = IvOptions(method="gmm", weight_type=weight_type, **kwargs)
    res = our_fit(df, options=options)
    assert res.converged


@pytest.mark.parametrize("cov_type", ["classical", "hc0", "hac", "cluster"])
def test_gmm_cov_type_options_run_independently_of_weight_type(
    iv_dataset, clustered_dataset, cov_type
):
    """`method="gmm"`で`cov_type`（SE計算方式）と`weight_type`（点推定に使う
    重み行列、既定`unadjusted`のまま）が独立な軸であること
    （`engine_pybind/src/iv/common.rs`のモジュールdocコメント参照）を、
    `weight_type`を固定したまま`cov_type`だけ変えても成功パスで動作する
    ことで確認する。
    """
    df = clustered_dataset if cov_type == "cluster" else iv_dataset
    kwargs = {"cluster_col": "cluster_group"} if cov_type == "cluster" else {}
    options = IvOptions(method="gmm", cov_type=cov_type, **kwargs)
    res = our_fit(df, options=options)
    assert res.cov_type == cov_type
    assert res.converged


def test_gmm_convergence_stops_before_max_iterations(iv_dataset):
    """現実的な`gmm_convergence`を指定すると、`gmm_iterations`の上限に達する
    前に収束判定を満たして反復を打ち切ること（`IvOptions.gmm_convergence`の
    「早期収束」という主要な挙動、非収束のみを確認する既存テストと対になる）。
    """
    options = IvOptions(
        method="gmm",
        weight_type="robust",
        gmm_convergence=1e-4,
        gmm_iterations=10,
    )
    res = our_fit(iv_dataset, options=options)
    assert res.converged
    assert res.n_iterations < 10


def test_gmm_raise_on_non_convergence_false_returns_converged_false(
    iv_dataset,
):
    """厳しすぎる`gmm_convergence`でも`raise_on_non_convergence=False`なら
    例外を投げず`converged=False`を返す（例外を送出する既定挙動側は
    `test_iv_validation.py::test_gmm_raise_on_non_convergence_true_raises_
    computation_error`）。
    """
    options = IvOptions(
        method="gmm",
        weight_type="robust",
        gmm_convergence=1e-300,
        gmm_iterations=2,
        raise_on_non_convergence=False,
    )
    res = our_fit(iv_dataset, options=options)
    assert res.converged is False
    assert res.n_iterations == 2
