"""IV python_packageラッパーの構造・API・エラーパスのスモークテスト。

主リファレンス（linearmodels）との厳密な数値比較は別途`test_iv_fixtures.py`で
実施する（`test_logit.py`/`test_probit.py`と同じ役割分担）。ここでは`fit()`の
成功パス・`coef_table()`/`first_stage()`の構造・オプションの反映確認・
`ValidationError`/`ComputationError`パスのみを検証する。
"""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import DATA_DIR, with_cluster_groups
from econometricsmodels import (
    IV,
    ComputationError,
    IvOptions,
    IvResults,
    ValidationError,
)


@pytest.fixture(scope="module")
def iv_dataset() -> pl.DataFrame:
    """`test_iv_fixtures.py`と同じ固定済みCSV（内生変数`endog1`・操作変数
    `z1`/`z2`を持つ、n=500の合成データセット）を再利用する。
    """
    return pl.read_csv(DATA_DIR / "iv_baseline.csv")


@pytest.fixture(scope="module")
def clustered_dataset(iv_dataset: pl.DataFrame) -> pl.DataFrame:
    """`iv_dataset`に10グループの疑似クラスター列を付与したもの。"""
    return with_cluster_groups(iv_dataset, 10)


def _our_fit(
    df: pl.DataFrame,
    *,
    x_exog: list[str] | None = None,
    x_endog: list[str] | None = None,
    instruments: list[str] | None = None,
    options: IvOptions | None = None,
) -> IvResults:
    """既定は`x_exog=["x1"], x_endog=["endog1"], instruments=["z1", "z2"]`
    （このファイルの大半のテストが使う共通パターン）。異なる変数構成が
    必要なテストのみ明示的に上書きする。
    """
    kwargs = {}
    if options is not None:
        kwargs["options"] = options
    return IV(
        df,
        y="y",
        x_exog=["x1"] if x_exog is None else x_exog,
        x_endog=["endog1"] if x_endog is None else x_endog,
        instruments=["z1", "z2"] if instruments is None else instruments,
        **kwargs,
    ).fit()


# ── 成功パス・API構造 ────────────────────────────────────────────────


def test_fit_succeeds_and_returns_iv_results(iv_dataset):
    res = _our_fit(iv_dataset)
    assert isinstance(res, IvResults)


def test_default_options_use_2sls_classical(iv_dataset):
    """`options`省略時は`IvOptions()`の既定値（method="2sls", classical）が
    使われ、2SLSは常に`converged=True`/`n_iterations=1`（閉形式・非反復）。
    """
    res = _our_fit(iv_dataset)
    assert res.cov_type == "classical"
    assert res.converged
    assert res.n_iterations == 1


def test_gmm_method_runs_and_converges(iv_dataset):
    """`method="gmm"`の既定オプション（`gmm_iterations=2`、2-step efficient
    GMM）が成功パスで動作すること。
    """
    options = IvOptions(method="gmm")
    res = _our_fit(iv_dataset, options=options)
    assert res.converged
    assert res.n_iterations == 2


def test_param_names_order(iv_dataset):
    """`param_names`は定数項→x_exog→x_endogの順（`IvInput::from_columns`の
    設計行列の列順、`docs/planning/specs/iv-api-design.md`参照）。
    """
    res = _our_fit(iv_dataset)
    assert res.param_names == ["const", "x1", "endog1"]


def test_coef_table_structure(iv_dataset):
    res = _our_fit(iv_dataset)
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
    res = _our_fit(iv_dataset)
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert set(ci.keys()) == {"const", "x1", "endog1"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_stats_p_values_share_keys(iv_dataset):
    res = _our_fit(iv_dataset)
    expected_keys = {"const", "x1", "endog1"}

    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_n_obs_and_dep_var_name(iv_dataset):
    res = _our_fit(iv_dataset)
    assert res.n_obs == iv_dataset.height
    assert res.dep_var_name == "y"


def test_cov_type_label(iv_dataset):
    for cov_type in ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]:
        options = IvOptions(cov_type=cov_type)
        res = _our_fit(iv_dataset, options=options)
        assert res.cov_type == cov_type


def test_cluster_cov_type_label(clustered_dataset):
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = _our_fit(clustered_dataset, options=options)
    assert res.cov_type == "cluster"


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
    res = _our_fit(iv_dataset)
    assert len(res.residuals) == res.n_obs == iv_dataset.height


def test_first_stage_structure(iv_dataset):
    """`first_stage()`は`x_endog`の変数名をキーにした`OlsResults`の辞書を返す。"""
    from econometricsmodels import OlsResults

    res = _our_fit(iv_dataset)
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
    res = _our_fit(iv_dataset)
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
    res = _our_fit(iv_dataset)
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
    res = _our_fit(iv_dataset, options=options)
    assert res.overid_statistic is not None
    assert res.overid_p_value is not None


def test_wu_hausman_is_none_for_gmm(iv_dataset):
    """`wu_hausman_statistic`/`wu_hausman_p_value`は`method="gmm"`では常に
    `None`（`GmmEstimator`はWu-Hausman検定を実装しない、
    `engine_pybind/src/iv/CLAUDE.md`参照）。
    """
    options = IvOptions(method="gmm")
    res = _our_fit(iv_dataset, options=options)
    assert res.wu_hausman_statistic is None
    assert res.wu_hausman_p_value is None


def test_wu_hausman_is_not_none_for_2sls(iv_dataset):
    res = _our_fit(iv_dataset)
    assert res.wu_hausman_statistic is not None
    assert res.wu_hausman_p_value is not None


# ── オプションの反映確認 ────────────────────────────────────────────


def test_confidence_level_changes_interval_width(iv_dataset):
    wide = _our_fit(iv_dataset, options=IvOptions(confidence_level=0.99))
    narrow = _our_fit(iv_dataset, options=IvOptions(confidence_level=0.80))

    for name in ["const", "x1", "endog1"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width, name


def test_hac_auto_lags_runs_and_returns_finite_std_errors(iv_dataset):
    """`hac_lags`省略時（`None`、自動計算式）でもエラーなく動作すること。"""
    options = IvOptions(cov_type="hac")
    res = _our_fit(iv_dataset, options=options)
    assert res.cov_type == "hac"
    for se in res.std_errors.values():
        assert se > 0.0


def test_hac_time_col_reorders_rows_before_computing_lags():
    """`time_col`を指定すると、DataFrameの行順に関わらず時系列順で
    ラグ付き自己共分散を計算すること（`test_ols.py`の同名テストと同じ発想を
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
    res = _our_fit(iv_dataset, options=options)
    assert res.param_names == ["x1", "endog1"]


@pytest.mark.parametrize(
    "weight_type", ["unadjusted", "robust", "cluster", "kernel"]
)
def test_gmm_weight_type_options_run(
    iv_dataset, clustered_dataset, weight_type
):
    """`method="gmm"`の`weight_type`各値が成功パスで動作すること（数値照合は
    `test_iv_fixtures.py`の対象外、GMMのlinearmodelsクロスチェックは別issue）。
    `cluster`/`kernel`は`cov_type`と同じ`cluster_col`/`hac_lags`フィールドを
    共用する仕様（`engine_pybind/src/iv/CLAUDE.md`参照）。
    """
    df = clustered_dataset if weight_type == "cluster" else iv_dataset
    kwargs = (
        {"cluster_col": "cluster_group"} if weight_type == "cluster" else {}
    )
    options = IvOptions(method="gmm", weight_type=weight_type, **kwargs)
    res = _our_fit(df, options=options)
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
    res = _our_fit(df, options=options)
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
    res = _our_fit(iv_dataset, options=options)
    assert res.converged
    assert res.n_iterations < 10


def test_gmm_raise_on_non_convergence_true_raises_computation_error(
    iv_dataset,
):
    """厳しすぎる`gmm_convergence`で最大反復回数内に収束しない場合、既定
    （`raise_on_non_convergence=True`）では`ComputationError`（`MleError.
    NonConvergence`と同じ分類、`engine_pybind/src/iv/common.rs`参照）。
    """
    options = IvOptions(
        method="gmm",
        weight_type="robust",
        gmm_convergence=1e-300,
        gmm_iterations=2,
    )
    with pytest.raises(ComputationError):
        _our_fit(iv_dataset, options=options)


def test_gmm_raise_on_non_convergence_false_returns_converged_false(
    iv_dataset,
):
    options = IvOptions(
        method="gmm",
        weight_type="robust",
        gmm_convergence=1e-300,
        gmm_iterations=2,
        raise_on_non_convergence=False,
    )
    res = _our_fit(iv_dataset, options=options)
    assert res.converged is False
    assert res.n_iterations == 2


# ── エラーハンドリング（ValidationErrorパス） ──────────────────────


def test_unknown_method_raises(iv_dataset):
    options = IvOptions(method="invalid")
    with pytest.raises(ValidationError):
        _our_fit(iv_dataset, options=options)


def test_unknown_cov_type_raises(iv_dataset):
    options = IvOptions(cov_type="invalid")
    with pytest.raises(ValidationError):
        _our_fit(iv_dataset, options=options)


def test_unknown_weight_type_raises(iv_dataset):
    options = IvOptions(method="gmm", weight_type="invalid")
    with pytest.raises(ValidationError):
        _our_fit(iv_dataset, options=options)


def test_y_in_x_exog_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["y", "x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_y_in_x_endog_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["y"],
            instruments=["z1", "z2"],
        ).fit()


def test_y_in_instruments_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["y", "z1"],
        ).fit()


def test_x_exog_overlaps_x_endog_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1", "endog1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_instruments_overlaps_x_exog_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["x1", "z2"],
        ).fit()


def test_x_endog_overlaps_instruments_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["endog1", "z2"],
        ).fit()


def test_duplicate_instruments_column_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["z1", "z1"],
        ).fit()


def test_duplicate_x_exog_column_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1", "x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_duplicate_x_endog_column_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1", "endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_const_collision_with_include_intercept_raises():
    """`include_intercept=True`のとき`x_exog`に`"const"`という列名を含めると
    自動追加される定数項と衝突し`ValidationError`になること。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "const": [1.0, 2.0, 3.5, 2.5],
            "endog1": [2.0, 1.0, 4.0, 3.0],
            "z1": [1.0, 3.0, 2.0, 4.0],
        }
    )
    with pytest.raises(ValidationError):
        IV(
            df, y="y", x_exog=["const"], x_endog=["endog1"], instruments=["z1"]
        ).fit()


def test_missing_column_raises(iv_dataset):
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["nonexistent"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_null_values_raise():
    df = pl.DataFrame(
        {
            "y": [1.0, None, 3.0, 4.0],
            "endog1": [2.0, 1.0, 4.0, 3.0],
            "z1": [1.0, 3.0, 2.0, 4.0],
        }
    )
    with pytest.raises(ValidationError):
        IV(df, y="y", x_exog=[], x_endog=["endog1"], instruments=["z1"]).fit()


def test_non_numeric_dtype_raises():
    df = pl.DataFrame(
        {
            "y": ["a", "b", "c", "d"],
            "endog1": [2.0, 1.0, 4.0, 3.0],
            "z1": [1.0, 3.0, 2.0, 4.0],
        }
    )
    with pytest.raises(ValidationError):
        IV(df, y="y", x_exog=[], x_endog=["endog1"], instruments=["z1"]).fit()


def test_singular_first_stage_design_matrix_raises_computation_error():
    """`x_exog`が完全な多重共線性を持つ場合、第一段階回帰
    （`x_endog[j] ~ x_exog + instruments`）の設計行列が特異になり
    `ComputationError`（`IvError::FirstStageFailed`、`test_ols.py`の
    `test_singular_matrix_raises_computation_error`と同じ原理）。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
            ],  # x2 = 2 * x1（完全な多重共線性）
            "endog1": [2.0, 1.0, 4.0, 3.0, 2.5, 3.5],
            "z1": [1.0, 3.0, 2.0, 4.0, 1.5, 2.5],
        }
    )
    with pytest.raises(ComputationError):
        IV(
            df,
            y="y",
            x_exog=["x1", "x2"],
            x_endog=["endog1"],
            instruments=["z1"],
        ).fit()


def test_gmm_cluster_weight_type_raises_computation_error_when_cluster_count_is_less_than_instrument_count(
    iv_dataset,
):
    """`method="gmm"`固有のComputationErrorパス。`weight_type="cluster"`の重み行列`S`
    （l×l、`l`は全操作変数の数）はG個のランク1行列の和のため`rank(S)≤G`
    （`engine/src/iv/CLAUDE.md`「クラスター数Gと操作変数の数lの関係」参照）。
    `G=2 < l=3`（`x_exog=[]`・`instruments=["z1","z2"]`で`l=const+z1+z2=3`）だと
    `S`が構造的に特異になり`ComputationError`（`gmm.rs`の第一段階とは別の、GMM
    自体の重み行列反転経路。2SLS/GMM共通の第一段階回帰の特異性
    （`test_singular_first_stage_design_matrix_raises_computation_error`）とは
    別のGMM固有の失敗パス）。
    """
    n = iv_dataset.height
    df = iv_dataset.with_columns(
        (pl.int_range(pl.len()) < n // 2).cast(pl.Int64).alias("cluster_group")
    )
    options = IvOptions(
        method="gmm",
        weight_type="cluster",
        cluster_col="cluster_group",
        cov_type="classical",
    )
    with pytest.raises(ComputationError):
        IV(
            df,
            y="y",
            x_exog=[],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
            options=options,
        ).fit()


def test_insufficient_observations_raises(iv_dataset):
    """観測数nが説明変数の数k（定数項込み）以下の場合`ValidationError`。"""
    df = iv_dataset.head(2)  # n=2、k=3（const, x1, endog1）
    with pytest.raises(ValidationError):
        _our_fit(df)


def test_insufficient_instruments_raises(iv_dataset):
    """識別の順序条件`len(instruments) >= len(x_endog)`を満たさない場合
    `ValidationError`（`IvError::InsufficientInstruments`）。
    """
    with pytest.raises(ValidationError):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=[],
        ).fit()


def test_cluster_without_col_raises(iv_dataset):
    options = IvOptions(cov_type="cluster")
    with pytest.raises(ValidationError):
        _our_fit(iv_dataset, options=options)


def test_insufficient_clusters_raises(iv_dataset):
    """クラスターが1種類しかない場合`ValidationError`。"""
    df = iv_dataset.with_columns(pl.lit(0).alias("single_cluster"))
    options = IvOptions(cov_type="cluster", cluster_col="single_cluster")
    with pytest.raises(ValidationError):
        _our_fit(df, options=options)


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(iv_dataset, confidence_level):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合
    `ValidationError`。
    """
    options = IvOptions(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        _our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("hac_lags", [-1, 500])  # 500 == iv_dataset の n_obs
def test_invalid_hac_lags_raises(iv_dataset, hac_lags):
    """`hac_lags`が`[0, n)`の範囲外の場合`ValidationError`。"""
    options = IvOptions(cov_type="hac", hac_lags=hac_lags)
    with pytest.raises(ValidationError):
        _our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("gmm_iterations", [0, -1])
def test_invalid_gmm_iterations_raises(iv_dataset, gmm_iterations):
    options = IvOptions(method="gmm", gmm_iterations=gmm_iterations)
    with pytest.raises(ValidationError):
        _our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("gmm_convergence", [0.0, -1.0])
def test_invalid_gmm_convergence_raises(iv_dataset, gmm_convergence):
    options = IvOptions(method="gmm", gmm_convergence=gmm_convergence)
    with pytest.raises(ValidationError):
        _our_fit(iv_dataset, options=options)
