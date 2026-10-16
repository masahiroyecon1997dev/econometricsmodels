"""IV の成功パスの構造・API・オプション反映の検証。

確定済み設計（`docs/spec/iv-spec.md`）どおりの結果型・辞書キー・
ラベルになっていること、`IVOptions` の各フィールド（2SLS/GMM 共通・GMM 固有）が
engine_pybind 経由で反映されることを確認する。`ValidationError`/
`ComputationError` パスは `test_iv_validation.py`、主リファレンス（linearmodels）
との数値照合は `test_iv_reference.py`（2SLS）・`test_iv_gmm_reference.py`（GMM）、
R クロスチェックは `test_iv_crosscheck.py`（OLS/WLS/Logit/Probit の
`test_<手法>_api.py` 等と同じ4分割）。

`iv_dataset`/`clustered_dataset` フィクスチャと `our_fit` ヘルパーは
`tests/iv/conftest.py`／`tests/support/_iv_helpers.py`。
"""

from __future__ import annotations

import polars as pl
import pytest
from _assertions import (
    assert_hac_time_reorders_rows,
    assert_params_table_matches_attributes,
    assert_same_estimates,
)
from _constants import DATA_DIR
from _helpers import (
    CASE_SPELLINGS,
    HAC_AUTO_LAG_SAMPLE_SIZES,
    ROW_TIME,
    hac_lag_frame,
    hac_time_for,
    reorder_rows,
    respell,
    with_cluster_groups,
    with_row_time,
)
from _iv_helpers import our_fit
from _tolerances import TOLERANCES
from econometricsmodels import IV, IVOptions, IVResults

from benchmark.common import hac_auto_lag
from benchmark.iv.constants import IV_COV_TYPES


def _assert_valid_test_statistic(
    statistic: float | None, p_value: float | None
) -> None:
    """統計量が正、p値が開区間(0, 1)にあること。`is not None`だけでは見逃す
    退化した値（0・NaN・スケールの誤り等）への簡易ガード。
    """
    assert statistic is not None
    assert p_value is not None
    assert statistic > 0.0
    assert 0.0 < p_value < 1.0


# ── 成功パス・結果型 ──────────────────────────────────────────────────


def test_fit_succeeds_and_returns_iv_results(iv_dataset):
    res = our_fit(iv_dataset)
    assert isinstance(res, IVResults)


def test_default_options_use_2sls_classical(iv_dataset):
    """`options`省略時は`IVOptions()`の既定値（estimator="2sls", classical）が
    使われ、2SLSは常に`converged=True`/`n_iter=1`（閉形式・非反復）。
    """
    res = our_fit(iv_dataset)
    assert res.cov_type == "classical"
    assert res.converged
    assert res.n_iter == 1


def test_residuals_length_matches_n_obs(iv_dataset):
    res = our_fit(iv_dataset)
    assert len(res.residuals) == res.n_obs == iv_dataset.height


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
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        df,
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()
    assert res.params["endog1"] != 0.0


# ── API構造 ───────────────────────────────────────────────────────────


def test_param_names_order(iv_dataset):
    """`param_names`は定数項→x_exog→x_endogの順（`IvInput::from_columns`の
    設計行列の列順、`docs/spec/iv-spec.md`参照）。
    """
    res = our_fit(iv_dataset)
    assert res.param_names == ["const", "x1", "endog1"]


def test_params_table_structure(iv_dataset):
    res = our_fit(iv_dataset)
    table = res.params_table()
    assert_params_table_matches_attributes(res)

    assert isinstance(table, list)
    assert len(table) == 3  # const, x1, endog1
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
    assert [row["param_name"] for row in table] == ["const", "x1", "endog1"]


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
    assert set(res.test_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_n_obs_and_dep_var_name(iv_dataset):
    res = our_fit(iv_dataset)
    assert res.n_obs == iv_dataset.height
    assert res.dep_var_name == "y"


def test_first_stage_structure(iv_dataset):
    """`first_stage()`は`x_endog`の変数名をキーにした`OLSResults`の辞書を返す。"""
    from econometricsmodels import OLSResults

    res = our_fit(iv_dataset)
    first_stage = res.first_stage()

    assert set(first_stage.keys()) == {"endog1"}
    assert isinstance(first_stage["endog1"], OLSResults)
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


def test_overid_statistic_present_when_over_identified(iv_dataset):
    """`instruments`が2本、`x_endog`が1本（過剰識別）なので`overid_statistic`
    はNoneにならない（2SLSのSargan検定）。
    """
    res = our_fit(iv_dataset)
    _assert_valid_test_statistic(res.overid_statistic, res.overid_p_value)


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
    """`estimator="gmm"`でも過剰識別なら`overid_statistic`（Hansen J検定）は
    Noneにならない。
    """
    options = IVOptions(estimator="gmm")
    res = our_fit(iv_dataset, options=options)
    _assert_valid_test_statistic(res.overid_statistic, res.overid_p_value)


def test_wu_hausman_is_none_for_gmm(iv_dataset):
    """`wu_hausman_statistic`/`wu_hausman_p_value`は`estimator="gmm"`では常に
    `None`（`GmmEstimator`はWu-Hausman検定を実装しない、
    `engine_pybind/src/iv/CLAUDE.md`参照）。
    """
    options = IVOptions(estimator="gmm")
    res = our_fit(iv_dataset, options=options)
    assert res.wu_hausman_statistic is None
    assert res.wu_hausman_p_value is None


def test_wu_hausman_is_not_none_for_2sls(iv_dataset):
    res = our_fit(iv_dataset)
    _assert_valid_test_statistic(
        res.wu_hausman_statistic, res.wu_hausman_p_value
    )


def test_wu_hausman_degrades_to_none_when_cluster_count_at_most_augmented_slopes(
    iv_dataset,
):
    """`cov_type="cluster"`で構造方程式は`G > q`（`fit()`冒頭のチェックを通過）だが、
    Wu-Hausman拡張回帰は第一段階残差列の分だけ傾き係数が増える
    （`q_aug = q + k_endog`）ため`G <= q_aug`になり、拡張回帰が
    `InsufficientClustersForInference`を返す → `wu_hausman_*`が`None`にdegradeする
    （`fit()`全体は成功。既存の`x_endog=[]`・完全予測退化ケースの
    degradeと同じ意味論）。`x_exog=["x1"]`+`x_endog=["endog1"]`+丁度識別
    （`instruments=["z1"]`、第一段階`q_fs=2`なので`G=3`で第一段階は成功）で
    構造方程式`q=2`・`k_endog=1`・`q_aug=3`、`G=3`（`q=2 < G=3 <= q_aug=3`）。
    """
    df = with_cluster_groups(iv_dataset, 3)
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = our_fit(df, instruments=["z1"], options=options)

    assert res.wu_hausman_statistic is None
    assert res.wu_hausman_p_value is None
    # 主要な推定結果は正常に返る。
    assert res.params["endog1"] is not None


# ── オプションの反映 ──────────────────────────────────────────────────


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize("cov_type", IV_COV_TYPES)
def test_cov_type_is_case_insensitive(
    iv_dataset, clustered_dataset, cov_type, spelling
):
    """全`cov_type`で、`res.cov_type`が正規形（小文字）で返り、どの大文字小文字の
    綴りでも正規形と同一の推定結果になること（`engine_pybind`側の
    `parse_iv_cov_type`のRust実装と対になる、Python API境界での確認。
    OLS/WLS/Logit/Probit/Tobit/FE/REの`test_cov_type_is_case_insensitive`と同型）。
    clusterは`cluster`列を持つ`clustered_dataset`を使う。
    """
    df = clustered_dataset if cov_type == "cluster" else iv_dataset
    kwargs = {"cluster": "cluster_group"} if cov_type == "cluster" else {}

    def fit(label: str):
        options = IVOptions(cov_type=label, **kwargs, **hac_time_for(cov_type))
        return our_fit(df, options=options)

    res = fit(respell(cov_type, spelling))
    assert res.cov_type == cov_type
    assert_same_estimates(res, fit(cov_type))


def test_confidence_level_changes_interval_width(iv_dataset):
    wide = our_fit(iv_dataset, options=IVOptions(confidence_level=0.99))
    narrow = our_fit(iv_dataset, options=IVOptions(confidence_level=0.80))

    for name in ["const", "x1", "endog1"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width, name


def test_include_intercept_false_omits_const(iv_dataset):
    """`include_intercept=False`だと`param_names`に`"const"`が含まれない
    （`x_endog`/`instruments`には元々自動で切片が付かない仕様と対称）。
    """
    options = IVOptions(include_intercept=False)
    res = our_fit(iv_dataset, options=options)
    assert res.param_names == ["x1", "endog1"]


def test_include_intercept_false_allows_const_in_x_endog():
    """`include_intercept=False`なら`x_endog`に`"const"`という名前の（切片
    ではない）通常の内生変数を含められること（衝突チェックは
    `include_intercept=True`のときのみ働く仕様、`test_const_collision_in_
    x_endog_with_include_intercept_raises`と対称）。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
            "const": [5.0, 4.0, 3.0, 6.0, 2.0, 1.0],
            "z1": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        }
    )
    options = IVOptions(include_intercept=False)
    res = our_fit(
        df,
        x_exog=["x1"],
        x_endog=["const"],
        instruments=["z1"],
        options=options,
    )
    assert res.param_names == ["x1", "const"]


def test_include_intercept_false_allows_const_in_instruments():
    """`include_intercept=False`なら`instruments`に`"const"`という名前の
    （切片ではない）通常の操作変数を含められること（前項のテストと対称）。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
            "endog1": [5.0, 4.0, 3.0, 6.0, 2.0, 1.0],
            "const": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        }
    )
    options = IVOptions(include_intercept=False)
    res = our_fit(
        df,
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["const"],
        options=options,
    )
    assert res.first_stage()["endog1"].param_names == ["x1", "const"]


def test_hac_auto_lags_runs_and_returns_finite_std_errors(iv_dataset):
    """`hac_lags`省略時（`None`、自動計算式）でもエラーなく動作すること。"""
    options = IVOptions(cov_type="hac", hac_time=ROW_TIME)
    res = our_fit(iv_dataset, options=options)
    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


def test_hac_time_reorders_rows_before_computing_lags():
    """`hac_time`を指定すると、DataFrameの行順に関わらず時系列順で
    ラグ付き自己共分散を計算すること（`test_ols_api.py`の同名テストと同じ発想を
    IVに適用、engine_pybindの`hac_time`列抽出経路をAPI境界から検証する）。
    """
    ordered_df = pl.DataFrame(
        {
            "y": [2.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 7.0],
            "endog1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "z1": [1.5, 2.5, 2.0, 4.5, 3.0, 5.5, 6.0, 7.5],
            "time": [float(i) for i in range(8)],
        }
    )
    shuffled_df = reorder_rows(ordered_df, [3, 1, 5, 2, 4, 0, 7, 6])

    def fit_std_errors(df: pl.DataFrame, hac_time: str) -> dict[str, float]:
        options = IVOptions(cov_type="hac", hac_lags=1, hac_time=hac_time)
        return (
            IV(
                df,
                y="y",
                x_exog=[],
                x_endog=["endog1"],
                instruments=["z1"],
                options=options,
            )
            .fit()
            .std_errors
        )

    assert_hac_time_reorders_rows(
        fit_std_errors(ordered_df, "time"),
        fit_std_errors(shuffled_df, "time"),
        fit_std_errors(with_row_time(shuffled_df), ROW_TIME),
        atol_invariant=TOLERANCES["iv_api"]["atol_hac_time_reorder"],
        min_rel_effect=TOLERANCES["iv_api"][
            "min_rel_hac_time_row_order_effect"
        ],
    )


@pytest.mark.parametrize("cov_type", ["classical", "hc0", "hac", "cluster"])
def test_gmm_cov_type_options_run_independently_of_weight_type(
    iv_dataset, clustered_dataset, cov_type
):
    """`estimator="gmm"`で`cov_type`（SE計算方式）と`gmm_weight_type`（点推定に使う
    重み行列、既定`classical`のまま）が独立な軸であること
    （`engine_pybind/src/iv/common.rs`のモジュールdocコメント参照）を、
    `gmm_weight_type`を固定したまま`cov_type`だけ変えても成功パスで動作する
    ことで確認する。
    """
    df = clustered_dataset if cov_type == "cluster" else iv_dataset
    kwargs = {"cluster": "cluster_group"} if cov_type == "cluster" else {}
    options = IVOptions(
        estimator="gmm", cov_type=cov_type, **kwargs, **hac_time_for(cov_type)
    )
    res = our_fit(df, options=options)
    assert res.cov_type == cov_type
    assert res.converged


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_estimator_label(iv_dataset, estimator):
    """`res.estimator`が指定した`estimator`（正規化済み小文字）を反映すること。"""
    res = our_fit(iv_dataset, options=IVOptions(estimator=estimator))
    assert res.estimator == estimator


@pytest.mark.parametrize(
    "estimator, expected_label",
    [
        ("2SLS", "2sls"),
        ("2Sls", "2sls"),
        ("GMM", "gmm"),
        ("Gmm", "gmm"),
    ],
)
def test_estimator_is_case_insensitive(iv_dataset, estimator, expected_label):
    """`estimator`が大文字小文字を区別しないこと（`test_cov_type_is_case_insensitive`
    と同型）。
    """
    res = our_fit(iv_dataset, options=IVOptions(estimator=estimator))
    assert res.estimator == expected_label


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize(
    "gmm_weight_type", ["classical", "robust", "cluster", "hac"]
)
def test_weight_type_is_case_insensitive(
    iv_dataset, clustered_dataset, gmm_weight_type, spelling
):
    """`estimator="gmm"`の全`gmm_weight_type`で、`res.gmm_weight_type`が正規形
    （小文字）で返り、どの大文字小文字の綴りでも正規形と同一の推定結果で収束
    すること（`engine_pybind`側の`parse_weight_type`と対になる、Python API
    境界での確認。数値照合は`test_iv_gmm_reference.py`）。`cluster`/`hac`は
    `cov_type`と同じ`cluster`フィールドと`hac_lags`フィールドを共用する仕様
    （`engine_pybind/src/iv/CLAUDE.md`参照）。
    """
    df = clustered_dataset if gmm_weight_type == "cluster" else iv_dataset
    kwargs = (
        {"cluster": "cluster_group"} if gmm_weight_type == "cluster" else {}
    )

    def fit(label: str):
        options = IVOptions(
            estimator="gmm",
            gmm_weight_type=label,
            **kwargs,
            **hac_time_for(gmm_weight_type),
        )
        return our_fit(df, options=options)

    res = fit(respell(gmm_weight_type, spelling))
    assert res.gmm_weight_type == gmm_weight_type
    assert res.converged
    assert_same_estimates(res, fit(gmm_weight_type))


def test_weight_type_is_none_for_2sls(iv_dataset):
    """`gmm_weight_type`はGMM専用の概念のため、`estimator="2sls"`では常に`None`
    であること。
    """
    res = our_fit(iv_dataset, options=IVOptions(estimator="2sls"))
    assert res.gmm_weight_type is None


def test_gmm_tol_stops_before_max_iterations(iv_dataset):
    """現実的な`gmm_tol`を指定すると、`gmm_max_iter`の上限に達する
    前に収束判定を満たして反復を打ち切ること（`IVOptions.gmm_tol`の
    「早期収束」という主要な挙動、非収束のみを確認する既存テストと対になる）。
    """
    options = IVOptions(
        estimator="gmm",
        gmm_type="iterated",
        gmm_weight_type="robust",
        gmm_tol=1e-4,
        gmm_max_iter=10,
    )
    res = our_fit(iv_dataset, options=options)
    assert res.converged
    assert res.n_iter < 10


def test_gmm_raise_on_non_convergence_false_returns_converged_false(
    iv_dataset,
):
    """厳しすぎる`gmm_tol`でも`raise_on_non_convergence=False`なら
    例外を投げず`converged=False`を返す（例外を送出する既定挙動側は
    `test_iv_validation.py::test_gmm_raise_on_non_convergence_true_raises_
    computation_error`）。
    """
    options = IVOptions(
        estimator="gmm",
        gmm_type="iterated",
        gmm_weight_type="robust",
        gmm_tol=1e-300,
        gmm_max_iter=3,
        raise_on_non_convergence=False,
    )
    res = our_fit(iv_dataset, options=options)
    assert res.converged is False
    assert res.n_iter == 3


def test_gmm_only_options_default_to_none():
    """GMM専用オプションの既定値は`None`（指定有無を区別し、使われる
    モードでのみ実効既定値に解決するため）。
    """
    options = IVOptions()
    assert options.gmm_type is None
    assert options.gmm_weight_type is None
    assert options.gmm_max_iter is None
    assert options.gmm_tol is None
    assert options.raise_on_non_convergence is None


# ── GMM（estimator="gmm"） ────────────────────────────────────────────


def test_gmm_method_runs_and_converges(iv_dataset):
    """`estimator="gmm"`の既定オプション（`gmm_type="two_step"`、2-step
    efficient GMM）が成功パスで動作すること。
    """
    options = IVOptions(estimator="gmm")
    res = our_fit(iv_dataset, options=options)
    assert res.converged
    assert res.n_iter == 2
    assert res.gmm_type == "two_step"


def test_gmm_one_step_has_no_weight_type_and_runs_once(iv_dataset):
    """`gmm_type="one_step"`は`gmm_weight_type`を使わず、`gmm_weight_type`は
    `None`、`n_iter=1`で常に収束扱い。
    """
    options = IVOptions(estimator="gmm", gmm_type="one_step")
    res = our_fit(iv_dataset, options=options)
    assert res.gmm_type == "one_step"
    assert res.gmm_weight_type is None
    assert res.n_iter == 1
    assert res.converged


def test_gmm_type_is_case_insensitive_and_echoed_lowercase(iv_dataset):
    options = IVOptions(estimator="gmm", gmm_type="Two_Step")
    assert our_fit(iv_dataset, options=options).gmm_type == "two_step"


def test_gmm_iterated_uses_effective_defaults(iv_dataset):
    """`gmm_max_iter`/`gmm_tol`が`None`（既定）でも`"iterated"`では
    実効既定値（100 / 1e-6）で反復し、収束すること。
    """
    options = IVOptions(
        estimator="gmm", gmm_type="iterated", gmm_weight_type="robust"
    )
    res = our_fit(iv_dataset, options=options)
    assert res.gmm_type == "iterated"
    assert res.converged
    assert 3 <= res.n_iter <= 100


def test_gmm_type_is_none_for_2sls(iv_dataset):
    res = our_fit(iv_dataset, options=IVOptions(estimator="2sls"))
    assert res.gmm_type is None


# ── HACの実使用ラグ数（hac_lags_used） ────────────────────────────────


@pytest.mark.parametrize("n", HAC_AUTO_LAG_SAMPLE_SIZES)
@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_hac_lags_used_matches_python_auto_lag_formula(n, estimator):
    """`hac_lags`省略時の`hac_lags_used`が、Python側の独立実装
    （`benchmark.common.hac_auto_lag`）と複数の標本サイズで一致すること
    （2SLSとGMMはRust側で別々に実装されている）。第一段階の結果も同じ値になる。
    """
    options = IVOptions(estimator=estimator, cov_type="hac", hac_time=ROW_TIME)
    res = our_fit(hac_lag_frame(n), options=options)
    assert res.hac_lags_used == hac_auto_lag(n)
    for first_stage in res.first_stage().values():
        assert first_stage.hac_lags_used == hac_auto_lag(n)


@pytest.mark.parametrize("n", [20, 1000])
@pytest.mark.parametrize("gmm_type", ["two_step", "iterated"])
def test_hac_lags_used_with_hac_weight_type_only(n, gmm_type):
    """GMMで`cov_type`はHacでなく`gmm_weight_type="hac"`だけでも、実際に
    使われたラグ数が返ること。
    """
    options = IVOptions(
        estimator="gmm",
        gmm_type=gmm_type,
        gmm_weight_type="hac",
        hac_time=ROW_TIME,
        # 収束は検証対象外（反復GMMが収束しなくてもラグは報告される）
        **(
            {"raise_on_non_convergence": False}
            if gmm_type == "iterated"
            else {}
        ),
    )
    res = our_fit(hac_lag_frame(n), options=options)
    assert res.cov_type == "classical"
    assert res.hac_lags_used == hac_auto_lag(n)


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_hac_lags_used_echoes_explicit_hac_lags(iv_dataset, estimator):
    options = IVOptions(
        estimator=estimator, cov_type="hac", hac_lags=3, hac_time=ROW_TIME
    )
    assert our_fit(iv_dataset, options=options).hac_lags_used == 3


@pytest.mark.parametrize(
    "options",
    [
        IVOptions(cov_type="hc1"),
        IVOptions(estimator="gmm", cov_type="hc1"),
        # `one_step`は重み行列を持たないため`gmm_weight_type`は使われない。
        IVOptions(estimator="gmm", gmm_type="one_step"),
        IVOptions(
            estimator="gmm", gmm_type="two_step", gmm_weight_type="robust"
        ),
    ],
    ids=["2sls_hc1", "gmm_hc1", "gmm_one_step", "gmm_robust_weight"],
)
def test_hac_lags_used_is_none_without_hac(iv_dataset, options):
    assert our_fit(iv_dataset, options=options).hac_lags_used is None
