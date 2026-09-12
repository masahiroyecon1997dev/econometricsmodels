"""Tobit python_packageラッパーの構造・API・エラーパスのスモークテスト。

主リファレンス（R survival::survreg / AER::tobit）との厳密な数値比較は別途
実施する（`test_logit_reference.py`/`test_logit_crosscheck.py`と同じ役割分担、
Issue #227）。ここでは`fit()`の成功パス・`coef_table()`/`predict()`/
`censoring_fit_check()`/`marginal_effects()`の構造・`ValidationError`/
`ComputationError`パスのみを検証する（`test_logit_api.py`/
`test_logit_validation.py`のTobit版）。
"""

from __future__ import annotations

import json
import math
import random

import _error_messages as msgs
import polars as pl
import pytest
from _constants import DATA_DIR
from _error_messages import escaped
from econometricsmodels import (
    ComputationError,
    Tobit,
    TobitOptions,
    TobitResults,
    ValidationError,
)

_TOBIT_CENSORING_BOUNDS = json.loads(
    (DATA_DIR / "tobit_censoring_bounds.json").read_text()
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


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_method_label(censored_dataset, method):
    """`res.method`が指定した`method`（正規化済み小文字）を反映すること
    （Logit/Probitの`check_method_label`と同型、Issue #307）。
    """
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(method=method),
    ).fit()
    assert res.method == method


@pytest.mark.parametrize(
    "method, expected_label",
    [
        ("NEWTON", "newton"),
        ("Newton", "newton"),
        ("BFGS", "bfgs"),
        ("Bfgs", "bfgs"),
        ("LBFGS", "lbfgs"),
        ("Lbfgs", "lbfgs"),
    ],
)
def test_method_is_case_insensitive(censored_dataset, method, expected_label):
    """`method`が大文字小文字を区別しないこと（Logit/Probitの
    `check_method_is_case_insensitive`と同型、Issue #307）。
    """
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(method=method),
    ).fit()
    assert res.method == expected_label


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
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_MARGINAL_EFFECTS_TARGET, other="bogus"),
    ):
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
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_MARGINAL_EFFECTS_AT, other="bogus"),
    ):
        res.marginal_effects(at="bogus")


def test_marginal_effects_unknown_target_raises(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_MARGINAL_EFFECTS_TARGET, other="bogus"),
    ):
        res.marginal_effects(target="bogus")


def test_marginal_effects_confidence_level_out_of_range_raises(
    censored_dataset,
):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_CONFIDENCE_LEVEL,
            confidence_level=msgs.rust_f64(1.5),
        ),
    ):
        res.marginal_effects(confidence_level=1.5)


# ── エラーハンドリング ──────────────────────────────────────────────


def test_y_in_x_raises(censored_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="y",
            single_role="y",
            multi_role="x",
        ),
    ):
        Tobit(censored_dataset, y="y", x=["y", "x1"]).fit()


def test_y_empty_string_raises(censored_dataset):
    """`y`に空文字列を渡した場合`ValidationError`
    （`test_ols_validation.py::test_y_empty_string_raises`参照）。
    """
    with pytest.raises(
        ValidationError, match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="")
    ):
        Tobit(censored_dataset, y="", x=["x1", "x2"]).fit()


def test_duplicate_x_column_raises(censored_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.DUPLICATE_WITHIN_ROLE, name="x1", role="x"),
    ):
        Tobit(censored_dataset, y="y", x=["x1", "x1"]).fit()


def test_const_collision_with_include_intercept_raises():
    df = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0, 1.0], "const": [1.0, 2.0, 3.0, 3.5]}
    )
    with pytest.raises(ValidationError, match=escaped(msgs.CONST_COLLISION)):
        Tobit(df, y="y", x=["const"]).fit()


def test_sigma_collision_raises():
    """`x`に`"sigma"`という列名があると、`TobitResult`が`param_names`の末尾に
    追加する合成パラメータ名`"sigma"`と衝突する（`"const"`列衝突と同型、
    `engine_pybind`の`validate_no_sigma_collision`のPython API境界での確認）。
    """
    df = pl.DataFrame(
        {"y": [0.0, 1.0, 0.0, 1.0], "sigma": [1.0, 2.0, 3.0, 3.5]}
    )
    with pytest.raises(ValidationError, match=escaped(msgs.SIGMA_COLLISION)):
        Tobit(df, y="y", x=["sigma"]).fit()


def test_empty_x_raises(censored_dataset):
    with pytest.raises(ValidationError, match=escaped(msgs.X_EMPTY, role="x")):
        Tobit(censored_dataset, y="y", x=[]).fit()


def test_missing_column_raises(censored_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="does_not_exist"),
    ):
        Tobit(censored_dataset, y="y", x=["does_not_exist"]).fit()


def test_null_values_raise():
    df = pl.DataFrame({"y": [0.0, None, 1.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="y", count=1),
    ):
        Tobit(df, y="y", x=["x1"]).fit()


def test_null_values_in_x_raise():
    """`x` 列に null が含まれる場合も `ValidationError`（`y` だけでなく `x` も
    欠損チェックの対象、テスト網羅性レビュー 観点5）。"""
    df = pl.DataFrame({"y": [0.0, 1.0, 2.0, 3.0], "x1": [1.0, None, 3.0, 4.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="x1", count=1),
    ):
        Tobit(df, y="y", x=["x1"]).fit()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_values_raise(bad):
    """`y`/`x` に NaN・無限大が含まれる場合 `ValidationError`。

    null（`test_null_values_raise` / `test_null_values_in_x_raise`）と NaN/無限大は
    `column_extraction.rs` 内で別ロジックのため個別に確認する（OLS の
    `test_non_finite_values_raise` と同じ、テスト網羅性レビュー 観点5）。
    """
    bad_repr = "NaN" if math.isnan(bad) else "inf"
    df_y = pl.DataFrame(
        {"y": [0.0, bad, 1.0, 2.0], "x1": [1.0, 2.0, 3.0, 4.0]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="y", value=bad_repr, row=1
        ),
    ):
        Tobit(df_y, y="y", x=["x1"]).fit()

    df_x = pl.DataFrame(
        {"y": [0.0, 1.0, 2.0, 3.0], "x1": [1.0, bad, 3.0, 4.0]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="x1", value=bad_repr, row=1
        ),
    ):
        Tobit(df_x, y="y", x=["x1"]).fit()


def test_non_numeric_dtype_raises():
    """文字列を数値キャストするとnullになるため`COLUMN_HAS_MISSING_VALUES`経路
    になる（`test_ols_validation.py::test_non_numeric_dtype_raises`参照）。
    """
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="y", count=3),
    ):
        Tobit(df, y="y", x=["x1"]).fit()


def test_unknown_cov_type_raises(censored_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_COV_TYPE_NONLINEAR, other="bogus"),
    ):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(cov_type="bogus"),
        ).fit()


def test_unknown_method_raises(censored_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_METHOD_NONLINEAR, other="bogus"),
    ):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(method="bogus"),
        ).fit()


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(censored_dataset, confidence_level):
    options = TobitOptions(confidence_level=confidence_level)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_CONFIDENCE_LEVEL,
            confidence_level=msgs.rust_f64(confidence_level),
        ),
    ):
        Tobit(censored_dataset, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize("tol", [0.0, -1.0])
def test_non_positive_tol_raises(censored_dataset, tol):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INVALID_TOL, tol=msgs.rust_f64(tol)),
    ):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(tol=tol),
        ).fit()


@pytest.mark.parametrize("max_iter", [0, -1])
def test_non_positive_max_iter_raises(censored_dataset, max_iter):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INVALID_MAX_ITER, max_iter=max_iter),
    ):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(max_iter=max_iter),
        ).fit()


def test_insufficient_observations_raises(censored_dataset):
    """観測数nが説明変数の数k（定数項込み）以下の場合`ValidationError`。

    Tobitは`sigma`（誤差項の標準偏差）もMLEパラメータとして数えるため
    `k=4`（const, x1, x2, sigma）。Logit/Probitの同名テスト（`k=3`）とは
    ここが異なる（実測確認済み）。
    """
    df = censored_dataset.head(2)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_OBSERVATIONS, n=2, k=4),
    ):
        Tobit(df, y="y", x=["x1", "x2"]).fit()


def test_invalid_censoring_bounds_raises(censored_dataset):
    """`lower`/`upper`が両方`None`は`ValidationError`（engine側の
    `InvalidCensoringBounds`）。
    """
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_CENSORING_BOUNDS,
            lower=msgs.rust_option_f64_debug(None),
            upper=msgs.rust_option_f64_debug(None),
        ),
    ):
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
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.Y_OUT_OF_CENSORING_BOUNDS,
            row=0,
            value=msgs.rust_f64(-1.0),
            lower=msgs.rust_option_f64_debug(0.0),
            upper=msgs.rust_option_f64_debug(None),
        ),
    ):
        Tobit(df, y="y", x=["x1"], options=TobitOptions(lower=0.0)).fit()


def test_no_uncensored_observations_raises():
    """非打ち切り観測が1件も無い（全観測が`lower`ちょうど）場合`ValidationError`
    （engine側の`NoUncensoredObservations`、Issue #223）。
    """
    df = pl.DataFrame({"y": [0.0, 0.0, 0.0, 0.0], "x1": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NO_UNCENSORED_OBSERVATIONS,
            lower=msgs.rust_option_f64_debug(0.0),
            upper=msgs.rust_option_f64_debug(None),
        ),
    ):
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


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。
    想定エラー（`ComputationError`、engine 側の `SingularDesignMatrix`）が発生する
    ことのみを確認する。OLS/WLS/Logit と同じく凍結 CSV
    （`tobit_perfect_multicollinearity.csv`、`x3 = 2·x1 + 3·x2`）を使う
    （テスト網羅性レビュー 観点3。旧テストは inline n=5 データだった）。

    Logit とは異なり、Tobit は `ols_initial_params` の QR 検証が `method` に
    関わらず常に最初に実行されるため、完全な多重共線性は常にこの経路で検出される
    （`method` を parametrize する必要が無い、`docs/spec/tobit-spec.md` 3.2節参照）。
    """
    df = pl.read_csv(DATA_DIR / "tobit_perfect_multicollinearity.csv")
    lower, upper = _TOBIT_CENSORING_BOUNDS["perfect_multicollinearity"]
    with pytest.raises(ComputationError):
        Tobit(
            df,
            y="y",
            x=["x1", "x2", "x3"],
            options=TobitOptions(lower=lower, upper=upper),
        ).fit()


@pytest.mark.parametrize("cov_type", ["classical", "opg", "hc0", "hc1"])
def test_scale_variance_raises_computation_error(cov_type):
    """変数間のスケールが極端に異なる設計行列（x1 を `*1e6`、x2 を `*1e-3`）は、
    傾き係数の同時共分散部分行列がスケール比の 2 乗（≈1e18）相当の条件数を持ち
    倍精度浮動小数点の限界を超えて数値的に特異になる（OLS/WLS と同じ理由・同じ
    凍結 CSV パターン、`test_ols_validation.py` 参照）。

    Tobit の全体 Wald 検定が OLS の F 検定と同型でこの部分行列の反転を要求するため、
    classical を含む全 cov_type で `ComputationError` になる。`scale_variance_mild`
    （スケール比 1e3）が数値リグレッション検知用の成功パス
    （`test_tobit_reference.py`）。数値比較はせずエラーパスのみ確認する
    （テスト網羅性レビュー 観点3、`TOBIT_ERROR_PATH_SCENARIOS`）。
    """
    df = pl.read_csv(DATA_DIR / "tobit_scale_variance.csv")
    lower, upper = _TOBIT_CENSORING_BOUNDS["scale_variance"]
    options = TobitOptions(cov_type=cov_type, lower=lower, upper=upper)
    with pytest.raises(ComputationError):
        Tobit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()


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


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_large_true_coefficient_dgp_converges_and_recovers_truth(method):
    """大きい真の係数（`x1`の係数=100）でもノイズがあれば識別可能で、真値を回復する
    （Issue #286の回帰テスト）。

    Issue #286（`y`のスケール由来の分離ヒューリスティック誤発火）の修正前は、この
    DGP（`y* = 100·x1 + 0.5·x2 + N(0,1)`、n=200、左打ち切り約51%）が`run_solver`
    共有の`SeparationSuspected`（標準化パラメータノルム基準、Logit/Probitの
    `y∈{0,1}`で較正）に誤って引っかかり`ComputationError`になっていた。`y`の
    標準偏差が約65あり標準化パラメータノルムが閾値100を超えていたのが原因で、
    真の分離ではなかった。#286（`TobitScaling`導入）と#288（Tobitでは
    `SeparationNormCheck.Disabled`）を経て、Newton/BFGS/LBFGSのいずれでも
    `x1≈100`・`σ≈1`（真値`β1=100`, `σ=1`）で収束する。

    真の（準完全）分離が`ComputationError`になることは
    `test_true_separation_noise_free_dgp_raises_computation_error`で検証する。
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

    res = Tobit(
        df, y="y", x=["x1", "x2"], options=TobitOptions(method=method)
    ).fit()

    # これはリファレンス数値照合ではなく#286の回帰テスト。許容幅は「真値を回復し、
    # かつσ→0退化に倒れていない」ことだけを担保する緩いバンド。実測は3メソッドとも
    # x1≈99.98・x2≈0.5・const≈0・σ≈1.02（相互のズレは1e-3未満）だが、将来の
    # ソルバー変更でのメソッド間変動を吸収するためマージンを広く取る。σの下限0.5は
    # 「(準)分離で σ→0 へ退化していない」ことの実質的なガード。
    assert res.converged
    assert abs(res.params["x1"] - 100.0) < 2.0
    assert abs(res.params["x2"] - 0.5) < 0.5
    assert abs(res.params["const"]) < 5.0
    assert 0.5 < res.sigma < 2.0


@pytest.mark.parametrize("method", ["newton", "bfgs", "lbfgs"])
def test_true_separation_noise_free_dgp_raises_computation_error(method):
    """ノイズを除いた完全分離DGP（`y* = 100·x1 + 0.5·x2`）は`ComputationError`。

    Tobitの「真の」分離は、Logit/Probitのように係数が±∞へ発散するのではなく
    `σ→0`退化として現れる。実測ではNewtonは`NonConvergence`（`max_iter`到達）、
    BFGS/L-BFGSも`ComputationError`になる（変種は問わない）。`max_iter`を
    35→2000に増やしてもNewton/BFGSは`NonConvergence`のまま。標準化パラメータ
    ノルム基準の`SeparationSuspected`はTobitでは無効
    （Issue #288、`run_solver`に`SeparationNormCheck.Disabled`）。

    **全件打ち切りとの棲み分け**: 非打ち切り観測が1件も無い（全観測が境界値
    ちょうど）ケースは`fit()`冒頭の`validate_has_uncensored_observations`が
    `ValidationError`（`NoUncensoredObservations`）で先に弾く
    （`test_no_uncensored_observations_raises`、Issue #223）。本ケースは
    非打ち切り観測が存在するため、そのバリデーションは通過し、最適化の
    非収束＝`ComputationError`（`ValueError`系ではない）として現れる。
    """
    rng = random.Random(42)
    n = 200
    x1 = [rng.uniform(-2.0, 2.0) for _ in range(n)]
    x2 = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    y = [max(0.0, 100.0 * x1[i] + 0.5 * x2[i]) for i in range(n)]
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

    with pytest.raises(ComputationError):
        Tobit(
            df, y="y", x=["x1", "x2"], options=TobitOptions(method=method)
        ).fit()


def test_quasi_separation_tiny_noise_reports_unconverged_without_raising():
    """境界レジーム（軽度の準完全分離＋ごく小さいノイズ）で
    `raise_on_non_convergence=False`のとき、旧実装と同様に`converged=False`を
    返す（無言で`converged=True`を返さない）ことを固定する（Issue #288、
    rust-reviewer指摘の「中間レジーム」）。

    `y* = 100·x1 + 0.5·x2 + N(0, 0.001)`。ノイズがあるため理屈上は識別可能だが、
    数値的には(準)分離的で`σ→0`方向へ退化し、Newtonは`max_iter`まで収束しない。
    実測では**真値自体は回復する**（`x1≈100`, `x2≈0.5`, `const≈0`, `σ≈ノイズsd`）
    ——「有限だが巨大な誤った`β̂`で収束扱いになる」病理ではなく、
    「正しい`β̂`だが収束判定は満たさない」状態。`raise_on_non_convergence=True`
    （既定）なら`ComputationError`（`NonConvergence`）になる。
    """
    rng = random.Random(7)
    n = 200
    x1 = [rng.uniform(-2.0, 2.0) for _ in range(n)]
    x2 = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    y = [
        max(0.0, 100.0 * x1[i] + 0.5 * x2[i] + rng.gauss(0.0, 0.001))
        for i in range(n)
    ]
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

    res = Tobit(
        df,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(raise_on_non_convergence=False),
    ).fit()

    assert res.converged is False
    # 退化しているのは σ のみ。傾きは真値近傍（巨大な誤った β̂ ではない）。
    assert abs(res.params["x1"] - 100.0) < 1.0
    assert abs(res.params["x2"] - 0.5) < 0.5
    assert 0.0 < res.sigma < 0.1

    with pytest.raises(ComputationError):
        Tobit(df, y="y", x=["x1", "x2"]).fit()


def test_many_regressors_no_false_separation():
    """説明変数を15本に増やしても、健全なDGPで偽の`SeparationSuspected`無しに
    収束する（Issue #288）。

    `SeparationNormCheck.Disabled`採用の根拠の一つが「多変量モデルでは
    標準化パラメータノルムが`√k`オーダーで増え、#286型の偽陽性が再発しうる」
    （Tobitでは係数由来でもノルムが増える）。Tobitは検出自体を通らないため
    ここで`ComputationError`になってはいけない。`TobitScaling`が設計行列を
    列標準化・平均センタリングしてノルムを抑える回帰ガードでもある。
    """
    rng = random.Random(3)
    n = 400
    k = 15
    cols = {f"x{j}": [rng.gauss(0.0, 1.0) for _ in range(n)] for j in range(k)}
    betas = [1.0 if j % 2 == 0 else -0.7 for j in range(k)]
    y = []
    for i in range(n):
        lin = 2.0 + sum(betas[j] * cols[f"x{j}"][i] for j in range(k))
        y.append(max(0.0, lin + rng.gauss(0.0, 1.5)))
    df = pl.DataFrame({"y": y, **cols})

    res = Tobit(df, y="y", x=[f"x{j}" for j in range(k)]).fit()

    assert res.converged
    # 代表的な係数が真値近傍（厳密照合はIssue #227の数値テストの領分）。
    assert abs(res.params["x0"] - 1.0) < 0.5
    assert abs(res.params["x1"] - (-0.7)) < 0.5
    assert 1.0 < res.sigma < 2.0


def test_mroz_hours_raw_scale_converges_without_false_separation():
    """実データ（Wooldridge mroz `hours`、生スケール）で偽の`SeparationSuspected`
    無しに収束する（Issue #286の実データ回帰、#288で無効化を確定）。

    `hours`（0〜4950、左打ち切り約43%）を Example 17.2 の RHS 7変数で推定する。
    `y`の標準偏差が大きく（σ̂≈1122）、#286修正前は標準化パラメータノルムが
    閾値100を超え`ComputationError`になっていた。R `AER::tobit`（survreg）との
    厳密な数値照合はIssue #227の別テストの領分。ここでは「生スケールでも
    収束し、教科書的な係数（`educ`≈80）を返す」ことのみ確認する。
    """
    from _constants import MROZ_X
    from _helpers import load_wooldridge_dataset

    mroz = load_wooldridge_dataset("mroz")
    res = Tobit(
        mroz, y="hours", x=MROZ_X, options=TobitOptions(lower=0.0)
    ).fit()

    assert res.converged
    assert res.n_obs == 753
    # Wooldridge Example 17.2 の水準（educ の係数 ≈ 80.6）。
    assert abs(res.params["educ"] - 80.6) < 5.0
    assert 1000.0 < res.sigma < 1250.0


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
    with pytest.raises(
        ValidationError, match=escaped(msgs.INSUFFICIENT_CLUSTERS, g=1)
    ):
        Tobit(
            df,
            y="y",
            x=["x1"],
            options=TobitOptions(cov_type="cluster", cluster_col="cluster"),
        ).fit()


def test_cluster_count_at_most_slopes_raises_validation_error(
    censored_dataset,
):
    """クラスター数G≤傾き係数の数q（ここで`G=2 == q=2`、x1/x2）は`ValidationError`
    （engine側の`CommonError::InsufficientClustersForInference`、Issue #289 / #287）。

    `rank(Ŝ)≤G-1`のため全体Wald検定のq×q部分行列がG≤qで構造的に特異になる。
    従来は`wald_chi2_test`内の`ComputationError`だったが、GもqもR行列計算なしで
    即座に判定できるため`fit()`冒頭の`ValidationError`へ前倒しした（#287のmroz
    `hours`クラスターケースがこの経路。`G<q`側はOLSの同名テストで確認）。
    """
    cluster = pl.Series(
        "cluster", [i % 2 for i in range(censored_dataset.height)]
    )
    df = censored_dataset.with_columns(cluster)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=2, q=2),
    ):
        Tobit(
            df,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(cov_type="cluster", cluster_col="cluster"),
        ).fit()


def test_mroz_hours_cluster_cov_type_raises_validation_error():
    """実データでの`G <= q`境界（#287の顕在化ケース、Issue #289で解決）。

    Wooldridge mroz `hours` Tobit（Wooldridge Example 17.2、RHS 7変数 → q=7）を
    `cluster_col="city"`（都市部居住ダミー、G=2）で推定すると`G=2 <= q=7`。
    `rank(Ŝ) <= G-1 = 1`のため全体Wald検定の`7×7`部分行列が構造的に特異になり、
    `fit()`冒頭のバリデーションが`ValidationError`
    （`CommonError::InsufficientClustersForInference`）で弾く。従来は
    `wald_chi2_test`内の`ComputationError`で`fit()`全体が失敗していた
    （参照実装Rの`linearHypothesis`相当も同データで計算不能）。
    """
    from _constants import MROZ_X
    from _helpers import load_wooldridge_dataset

    mroz = load_wooldridge_dataset("mroz")
    options = TobitOptions(cov_type="cluster", cluster_col="city", lower=0.0)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=2, q=len(MROZ_X)
        ),
    ):
        Tobit(mroz, y="hours", x=MROZ_X, options=options).fit()


def test_cluster_col_nonexistent_column_raises(censored_dataset):
    options = TobitOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="does_not_exist"),
    ):
        Tobit(censored_dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_cluster_col_with_null_raises(censored_dataset):
    """`cluster_col` に null（欠損）が含まれる場合 `ValidationError`
    （`extract_group_key_column` の null チェック、テスト網羅性レビュー 観点5）。"""
    n = censored_dataset.height
    groups = [None] + [str(i % 5) for i in range(n - 1)]
    df = censored_dataset.with_columns(pl.Series("grp", groups, dtype=pl.Utf8))
    options = TobitOptions(cov_type="cluster", cluster_col="grp")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.GROUP_KEY_COLUMN_HAS_MISSING_VALUES, name="grp"),
    ):
        Tobit(df, y="y", x=["x1", "x2"], options=options).fit()
