"""Tobit python_packageラッパーの構造・API・エラーパスのスモークテスト。

主リファレンス（R survival::survreg / AER::tobit）との厳密な数値比較は別途
実施する（`test_logit_fixtures.py`/`test_logit_crosscheck.py`と同じ役割分担、
Issue #227）。ここでは`fit()`の成功パス・`coef_table()`/`predict()`/
`censoring_fit_check()`/`marginal_effects()`の構造・`ValidationError`/
`ComputationError`パスのみを検証する（`test_logit.py`のTobit版）。
"""

from __future__ import annotations

import random

import polars as pl
import pytest
from econometricsmodels import (
    ComputationError,
    Tobit,
    TobitOptions,
    TobitResults,
    ValidationError,
)

# censored_datasetフィクスチャ（`dataset`のyを0.0で左打ち切り、打ち切り率21%）は
# conftest.pyで定義。

# ── 成功パス・API構造 ────────────────────────────────────────────────


def test_fit_succeeds_and_returns_tobit_results(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert isinstance(res, TobitResults)


def test_default_options_use_classical_left_censored_at_zero(
    censored_dataset,
):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.cov_type == "classical"
    assert res.lower == 0.0
    assert res.upper is None
    assert res.converged


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_method_option_converges_to_same_params(censored_dataset, method):
    """`method`（newton/bfgs/lbfgs）はいずれも同じ最尤解に収束する。

    `engine/src/nonlinear/tobit.rs`のRust単体テストは3手法の一致を検証済み
    だが、engine_pybindの文字列→`Method`パースやpython_packageラッパーの
    配線を検出するAPIレベルのテストが無かったため追加した（Logitの
    `test_method_option_converges_to_same_params`と同じ理由）。
    """
    baseline = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(method=method),
    ).fit()
    assert res.converged
    for name in res.param_names:
        assert res.params[name] == pytest.approx(
            baseline.params[name], rel=1e-4
        )


def test_param_names_include_const_first_and_sigma_last(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.param_names == ["const", "x1", "x2", "sigma"]


def test_include_intercept_false_omits_const_and_converges(
    censored_dataset,
):
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(include_intercept=False),
    ).fit()
    assert res.param_names == ["x1", "x2", "sigma"]
    assert res.converged
    assert res.df_model == 2


def test_params_std_errors_z_stats_p_values_share_keys(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    expected_keys = {"const", "x1", "x2", "sigma"}
    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.z_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_sigma_property_matches_params_sigma(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.sigma == res.params["sigma"]
    assert res.sigma > 0.0


def test_conf_int_structure(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    ci = res.conf_int
    assert set(ci.keys()) == {"const", "x1", "x2", "sigma"}
    for lower, upper in ci.values():
        assert lower < upper


def test_n_obs_matches_dataset_size(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.n_obs == censored_dataset.height


def test_coef_table_structure(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    table = res.coef_table()

    assert isinstance(table, list)
    assert len(table) == 4  # const, x1, x2, sigma
    expected_keys = {
        "param",
        "coef",
        "std_err",
        "z_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert expected_keys <= set(row.keys())
    assert [row["param"] for row in table] == ["const", "x1", "x2", "sigma"]


def test_wald_statistic_and_p_value_are_present(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.wald_statistic > 0.0
    assert 0.0 <= res.wald_p_value <= 1.0


# ── predict() / censoring_fit_check() ────────────────────────────────


@pytest.mark.parametrize(
    "target", ["expected_latent", "expected_observed", "prob_uncensored"]
)
def test_predict_returns_row_oriented_predictions(censored_dataset, target):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    predicted = res.predict(target=target)

    assert len(predicted) == censored_dataset.height
    for row in predicted:
        assert set(row.keys()) == {"predicted"}


def test_predict_prob_uncensored_is_a_probability(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    for row in res.predict(target="prob_uncensored"):
        assert 0.0 <= row["predicted"] <= 1.0


def test_predict_unknown_target_raises(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.predict(target="bogus")


def test_censoring_fit_check_structure(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    check = res.censoring_fit_check()

    assert isinstance(check, list)
    # 既定の左打ち切りのみ（lower=0.0, upper=None）なので lower/uncensored の2行
    assert {row["category"] for row in check} == {"lower", "uncensored"}
    for row in check:
        assert 0.0 <= row["observed_rate"] <= 1.0
        assert 0.0 <= row["model_implied_rate"] <= 1.0


def test_censoring_fit_check_omits_upper_when_upper_is_none(
    censored_dataset,
):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    check = res.censoring_fit_check()
    assert "upper" not in {row["category"] for row in check}


# ── marginal_effects() ────────────────────────────────────────────────


def test_marginal_effects_default_excludes_intercept(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    effects = res.marginal_effects()

    assert [row["param"] for row in effects] == ["x1", "x2"]
    expected_keys = {
        "param",
        "dydx",
        "std_err",
        "z",
        "p_value",
        "conf_low",
        "conf_high",
    }
    for row in effects:
        assert expected_keys <= set(row.keys())


@pytest.mark.parametrize(
    "target", ["expected_latent", "expected_observed", "prob_uncensored"]
)
def test_marginal_effects_accepts_all_targets(censored_dataset, target):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    effects = res.marginal_effects(target=target)
    assert [row["param"] for row in effects] == ["x1", "x2"]


def test_marginal_effects_mean_and_median_differ_from_overall(
    censored_dataset,
):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    overall = [row["dydx"] for row in res.marginal_effects(at="overall")]
    mean = [row["dydx"] for row in res.marginal_effects(at="mean")]
    median = [row["dydx"] for row in res.marginal_effects(at="median")]

    assert overall != mean
    assert overall != median


def test_marginal_effects_at_is_case_insensitive(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.marginal_effects(at="OVERALL") == res.marginal_effects(
        at="overall"
    )


def test_marginal_effects_unknown_at_raises(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(at="bogus")


def test_marginal_effects_unknown_target_raises(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(target="bogus")


def test_marginal_effects_confidence_level_out_of_range_raises(
    censored_dataset,
):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(ValidationError):
        res.marginal_effects(confidence_level=1.5)


# ── エラーハンドリング ──────────────────────────────────────────────


def test_y_in_x_raises(censored_dataset):
    with pytest.raises(ValidationError):
        Tobit(censored_dataset, y="y", x=["y", "x1"]).fit()


def test_duplicate_x_column_raises(censored_dataset):
    with pytest.raises(ValidationError):
        Tobit(censored_dataset, y="y", x=["x1", "x1"]).fit()


def test_const_collision_with_include_intercept_raises():
    df = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0, 1.0], "const": [1.0, 2.0, 3.0, 3.5]}
    )
    with pytest.raises(ValidationError):
        Tobit(df, y="y", x=["const"]).fit()


def test_sigma_collision_raises():
    """`x`に`"sigma"`という列名があると、`TobitResult`が`param_names`の末尾に
    追加する合成パラメータ名`"sigma"`と衝突する（`"const"`列衝突と同型、
    `engine_pybind`の`validate_no_sigma_collision`のPython API境界での確認）。
    """
    df = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0, 1.0], "sigma": [1.0, 2.0, 3.0, 3.5]}
    )
    with pytest.raises(ValidationError):
        Tobit(df, y="y", x=["sigma"]).fit()


def test_empty_x_raises(censored_dataset):
    with pytest.raises(ValidationError):
        Tobit(censored_dataset, y="y", x=[]).fit()


def test_missing_column_raises(censored_dataset):
    with pytest.raises(ValidationError):
        Tobit(censored_dataset, y="y", x=["does_not_exist"]).fit()


def test_null_values_raise():
    df = pl.DataFrame({"y": [0.0, None, 1.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        Tobit(df, y="y", x=["x1"]).fit()


def test_non_numeric_dtype_raises():
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        Tobit(df, y="y", x=["x1"]).fit()


def test_unknown_cov_type_raises(censored_dataset):
    with pytest.raises(ValidationError):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(cov_type="bogus"),
        ).fit()


def test_unknown_method_raises(censored_dataset):
    with pytest.raises(ValidationError):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(method="bogus"),
        ).fit()


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(censored_dataset, confidence_level):
    options = TobitOptions(confidence_level=confidence_level)
    with pytest.raises(ValidationError):
        Tobit(censored_dataset, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize("tol", [0.0, -1.0])
def test_non_positive_tol_raises(censored_dataset, tol):
    with pytest.raises(ValidationError):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(tol=tol),
        ).fit()


@pytest.mark.parametrize("max_iter", [0, -1])
def test_non_positive_max_iter_raises(censored_dataset, max_iter):
    with pytest.raises(ValidationError):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(max_iter=max_iter),
        ).fit()


def test_insufficient_observations_raises(censored_dataset):
    df = censored_dataset.head(2)
    with pytest.raises(ValidationError):
        Tobit(df, y="y", x=["x1", "x2"]).fit()


def test_invalid_censoring_bounds_raises(censored_dataset):
    """`lower`/`upper`が両方`None`は`ValidationError`（engine側の
    `InvalidCensoringBounds`）。
    """
    with pytest.raises(ValidationError):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(lower=None, upper=None),
        ).fit()


def test_y_out_of_censoring_bounds_raises():
    """`y`が指定した打ち切り境界の範囲外の値を含む場合`ValidationError`
    （engine側の`YOutOfCensoringBounds`）。
    """
    df = pl.DataFrame({"y": [-1.0, 0.0, 1.0, 2.0], "x1": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValidationError):
        Tobit(df, y="y", x=["x1"], options=TobitOptions(lower=0.0)).fit()


def test_no_uncensored_observations_raises():
    """非打ち切り観測が1件も無い（全観測が`lower`ちょうど）場合`ValidationError`
    （engine側の`NoUncensoredObservations`、Issue #223）。
    """
    df = pl.DataFrame({"y": [0.0, 0.0, 0.0, 0.0], "x1": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValidationError):
        Tobit(df, y="y", x=["x1"]).fit()


def test_supports_right_censoring_only():
    """`lower=None`・`upper`指定で右打ち切りのみのモデルとして推定できる
    （`nonlinear-api-design.md`7章）。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 5.0, 5.0, 5.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    res = Tobit(
        df, y="y", x=["x1"], options=TobitOptions(lower=None, upper=5.0)
    ).fit()
    assert res.lower is None
    assert res.upper == 5.0


def test_singular_design_matrix_raises_computation_error():
    """完全な多重共線性は`ComputationError`（engine側の`SingularDesignMatrix`）。

    Logitとは異なり、Tobitは`ols_initial_params`のQR検証が`method`に関わらず
    常に最初に実行されるため、完全な多重共線性は常にこの経路で検出される
    （`method`をparametrizeする必要が無い、`docs/planning/specs/
    nonlinear-implementation-notes.md`参照）。
    """
    df = pl.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 4.0, 6.0, 8.0, 10.0],  # x2 = 2 * x1
        }
    )
    with pytest.raises(ComputationError):
        Tobit(df, y="y", x=["x1", "x2"]).fit()


def test_non_convergence_raises_computation_error_with_tiny_max_iter(
    censored_dataset,
):
    with pytest.raises(ComputationError):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(max_iter=1),
        ).fit()


def test_separation_suspected_raises_computation_error_for_near_separation_data():
    """極端に大きい真の係数（`x1`の係数=100）のDGPは`ComputationError`
    （engine側の`SeparationSuspected`、`run_solver`でLogit/Probit/Tobit共有の
    検出機構）。

    `nonlinear-api-design.md`10章では「Tobitはyが連続なため、非打ち切り観測が
    無いケース等、Logit/Probitとは異なる退化パターンがあり得る」という懸念から
    このケースの検出要否が未確定だった。実際には2種類の異なる退化が存在する
    ことが判明した: 非打ち切り観測ゼロによる`σ→0`退化（`MleError::
    NoUncensoredObservations`、`test_no_uncensored_observations_raises`
    参照、Issue #223）と、本テストが検証する極端な`β`による分離（既存の
    `SeparationSuspected`機構がLogit/Probitと同じ標準化パラメータノルム基準で
    そのまま捕捉できることを本テストで実測確認、Issue #226）。
    """
    rng = random.Random(42)
    n = 200
    x1 = [rng.uniform(-2.0, 2.0) for _ in range(n)]
    x2 = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    y = []
    for i in range(n):
        y_star = 0.0 + 100.0 * x1[i] + 0.5 * x2[i] + rng.gauss(0.0, 1.0)
        y.append(max(0.0, y_star))
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

    with pytest.raises(ComputationError):
        Tobit(df, y="y", x=["x1", "x2"]).fit()


def test_raise_on_non_convergence_false_returns_result_without_raising(
    censored_dataset,
):
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(max_iter=1, raise_on_non_convergence=False),
    ).fit()
    assert res.converged is False
    assert res.n_iter == 1


def test_confidence_level_changes_interval_width(censored_dataset):
    wide = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(confidence_level=0.99),
    ).fit()
    narrow = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(confidence_level=0.80),
    ).fit()

    for name in ["const", "x1", "x2", "sigma"]:
        wide_width = wide.conf_int[name][1] - wide.conf_int[name][0]
        narrow_width = narrow.conf_int[name][1] - narrow.conf_int[name][0]
        assert narrow_width < wide_width


def test_cov_type_label(censored_dataset):
    for cov_type in ["classical", "opg", "hc0", "hc1"]:
        res = Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(cov_type=cov_type),
        ).fit()
        assert res.cov_type == cov_type

    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(cov_type="cluster", cluster_col="cluster"),
    ).fit()
    assert res.cov_type == "cluster"


@pytest.mark.parametrize(
    "cov_type, expected_label",
    [
        ("CLASSICAL", "classical"),
        ("Classical", "classical"),
        ("OPG", "opg"),
        ("Opg", "opg"),
        ("HC0", "hc0"),
        ("Hc1", "hc1"),
        ("CLUSTER", "cluster"),
        ("nonrobust", "nonrobust"),
        ("NONROBUST", "nonrobust"),
    ],
)
def test_cov_type_is_case_insensitive(
    censored_dataset, cov_type, expected_label
):
    kwargs = {"cluster_col": "cluster"} if cov_type == "CLUSTER" else {}
    options = TobitOptions(cov_type=cov_type, **kwargs)
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"], options=options).fit()
    assert res.cov_type == expected_label


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_alias_for_classical(censored_dataset, cov_type):
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(cov_type=cov_type),
    ).fit()
    classical_res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(cov_type="classical"),
    ).fit()
    for name in res.param_names:
        assert res.std_errors[name] == classical_res.std_errors[name], name


def test_cluster_cov_type_requires_at_least_two_groups():
    df = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "cluster": ["a", "a", "a", "a"],
        }
    )
    with pytest.raises(ValidationError):
        Tobit(
            df,
            y="y",
            x=["x1"],
            options=TobitOptions(cov_type="cluster", cluster_col="cluster"),
        ).fit()


def test_cluster_col_nonexistent_column_raises(censored_dataset):
    options = TobitOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(ValidationError):
        Tobit(censored_dataset, y="y", x=["x1", "x2"], options=options).fit()
