"""Tobit の成功パスの構造・API・オプション反映・`predict()`/`augment()`/
`censoring_fit_check()`/`marginal_effects()` の検証。

確定済み設計（`docs/spec/tobit-spec.md`）どおりの結果型・辞書キー・ラベルに
なっていること、`TobitOptions` の各フィールドが engine_pybind 経由で反映される
ことを確認する。`ValidationError`/`ComputationError` パスは
`test_tobit_validation.py`、主リファレンス（R `AER::tobit`）との数値照合は
`test_tobit_reference.py`、R クロスチェック（`censReg`）は
`test_tobit_crosscheck.py`（`test_logit_api.py` 等と同じ分割）。

`predict()`/`augment()`/`marginal_effects()` の構造・オプション反映はこの
ファイルに集約する。それぞれの `ValidationError` パスのみ
`test_tobit_validation.py`。
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from _assertions import (
    assert_close,
    assert_option_case_insensitive,
    assert_params_table_matches_attributes,
    assert_same_estimates,
)
from _helpers import CASE_SPELLINGS, respell
from _tolerances import TOLERANCES
from econometricsmodels import Tobit, TobitOptions, TobitResults
from scipy.stats import norm

from benchmark.nonlinear.constants import MLE_COV_TYPES, MLE_SOLVERS

# censored_datasetフィクスチャ（`dataset`のyを0.0で左打ち切り、打ち切り率21%）は
# conftest.pyで定義。


# ── 成功パス・結果型 ──────────────────────────────────────────────────


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


# ── API構造 ───────────────────────────────────────────────────────────


def test_param_names_include_const_first_and_sigma_last(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.param_names == ["const", "x1", "x2", "sigma"]


def test_params_table_structure(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    table = res.params_table()
    assert_params_table_matches_attributes(res)

    assert isinstance(table, list)
    assert len(table) == 4  # const, x1, x2, sigma
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
    assert [row["param_name"] for row in table] == [
        "const",
        "x1",
        "x2",
        "sigma",
    ]
    # `sigma`は係数ではないが、他の行と同じキーで値が表せる。
    sigma_row = table[-1]
    assert sigma_row["param_name"] == "sigma"
    assert sigma_row["param"] == res.params["sigma"]


def test_conf_int_structure(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    ci = res.conf_int
    assert set(ci.keys()) == {"const", "x1", "x2", "sigma"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_test_stats_p_values_share_keys(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    expected_keys = {"const", "x1", "x2", "sigma"}
    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.test_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_n_obs_and_dep_var_name(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.n_obs == censored_dataset.height
    assert res.dep_var_name == "y"


def test_sigma_property_matches_params_sigma(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.sigma == res.params["sigma"]
    assert res.sigma > 0.0


def test_wald_statistic_and_p_value_are_present(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.wald_statistic > 0.0
    assert 0.0 <= res.wald_p_value <= 1.0


# ── オプションの反映 ──────────────────────────────────────────────────


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize("cov_type", MLE_COV_TYPES)
def test_cov_type_is_case_insensitive(censored_dataset, cov_type, spelling):
    """全`cov_type`で、`res.cov_type`が正規形（小文字）で返り、どの大文字小文字の
    綴りでも正規形と同一の推定結果になること（`engine_pybind`側の
    `build_<solver>_input`のRust単体テストと対になる、Python API境界での
    確認。OLS/WLS/IV/Logit/Probit/FE/REの`test_cov_type_is_case_insensitive`と同型）。
    """
    kwargs = {"cluster": "cluster"} if cov_type == "cluster" else {}

    def fit(label: str):
        options = TobitOptions(cov_type=label, **kwargs)
        return Tobit(
            censored_dataset, y="y", x=["x1", "x2"], options=options
        ).fit()

    res = fit(respell(cov_type, spelling))
    assert res.cov_type == cov_type
    assert_same_estimates(res, fit(cov_type))


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


@pytest.mark.parametrize("solver", MLE_SOLVERS)
def test_solver_option_converges(censored_dataset, solver):
    """`solver`（newton/bfgs/lbfgs）はいずれも収束する。

    engine_pybindの文字列→`Method`パースやpython_packageラッパーの配線が
    成功パスで動くことの確認に留める。最尤解との数値一致は主リファレンスとの
    照合（`test_tobit_reference.py`の`test_solver_matches_aer_tobit`）が、
    より厳しい許容誤差で担う。
    """
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(solver=solver),
    ).fit()
    assert res.converged


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize("solver", MLE_SOLVERS)
def test_solver_is_case_insensitive(censored_dataset, solver, spelling):
    """`res.solver`が正規形で返り、どの大文字小文字の綴りでも正規形と同一の推定結果に
    なること（Logit/Probitの`check_solver_is_case_insensitive`と同型）。
    """

    def fit(label):
        return Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(solver=label),
        ).fit()

    assert_option_case_insensitive(fit, "solver", solver, spelling)


@pytest.mark.parametrize("cov_type", MLE_COV_TYPES)
def test_raise_on_non_convergence_false_returns_result_without_raising(
    censored_dataset, cov_type
):
    """`raise_on_non_convergence=False`だと未収束でも例外を投げず、
    `converged=False`の`Results`を返す。`cov_type`は`classical`以外
    （`opg`/`hc0`/`hc1`/`cluster`）も検証する（`_binary_choice_checks.py`の
    Logit/Probit版と同じ懸念——打ち切り点でのHessian/スコア評価はcov_typeの
    分岐によって経由する行列演算が異なるため、想定外の例外を投げず標準誤差が
    有限値であることまで確認する）。
    """
    kwargs = {
        "max_iter": 1,
        "raise_on_non_convergence": False,
        "cov_type": cov_type,
    }
    if cov_type == "cluster":
        kwargs["cluster"] = "cluster"
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(**kwargs),
    ).fit()
    assert res.converged is False
    assert res.n_iter == 1
    for name, se in res.std_errors.items():
        assert math.isfinite(se), f"std_errors[{name}]={se} is not finite"


@pytest.mark.parametrize("include_intercept", [True, False])
def test_aic_bic_count_sigma_as_a_parameter(
    censored_dataset, include_intercept
):
    """`aic`/`bic`の罰則は`σ`を含む総パラメータ数（`params`の長さ）を使う。
    一方`df_model`はWald検定の対象となる傾き係数の数で、`σ`と切片を含まない
    （`docs/spec/tobit-spec.md`「適合度統計量」）。
    """
    res = Tobit(
        censored_dataset,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(include_intercept=include_intercept),
    ).fit()
    n_params = len(res.params)

    assert "sigma" in res.params
    rtol = TOLERANCES["tobit_api"]["rtol_information_criterion"]
    assert_close(
        res.aic,
        -2.0 * res.log_likelihood + 2.0 * n_params,
        "aic",
        rtol=rtol,
        atol=0.0,
    )
    assert_close(
        res.bic,
        -2.0 * res.log_likelihood + math.log(res.n_obs) * n_params,
        "bic",
        rtol=rtol,
        atol=0.0,
    )
    assert res.df_resid == res.n_obs - n_params
    assert res.df_model == 2
    assert res.df_model == n_params - 1 - int(include_intercept)


def test_supports_right_censoring_only():
    """`lower=None`・`upper`指定で右打ち切りのみのモデルとして推定できる
    （`nonlinear-common.md`7章）。
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


# ── predict() ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "target",
    [
        "expected_latent",
        "expected_observed",
        "prob_uncensored",
        "expected_uncensored",
    ],
)
def test_predict_returns_list_of_floats(censored_dataset, target):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    predicted = res.predict(target=target)

    assert isinstance(predicted, list)
    assert len(predicted) == censored_dataset.height
    assert all(isinstance(value, float) for value in predicted)


def test_predict_prob_uncensored_is_a_probability(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    for prob in res.predict(target="prob_uncensored"):
        assert 0.0 <= prob <= 1.0


def test_predict_expected_uncensored_satisfies_decomposition_identity(
    censored_dataset,
):
    """`censored_dataset`は下限0のみの左打ち切り。`E[y|x] = P(y>0|x) *
    E[y|y>0,x]`（`expected_observed = prob_uncensored * expected_uncensored`）
    が成り立ち、`E[y|y>0,x]`は常に下限0以上。
    """
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    observed = res.predict(target="expected_observed")
    prob = res.predict(target="prob_uncensored")
    conditional = res.predict(target="expected_uncensored")

    for obs, p, cond in zip(observed, prob, conditional, strict=True):
        assert cond > 0.0
        assert obs == pytest.approx(p * cond, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize(
    "target",
    [
        "expected_latent",
        "expected_observed",
        "prob_uncensored",
        "expected_uncensored",
    ],
)
def test_predict_new_data_returns_list_of_floats(censored_dataset, target):
    """`predict(new_data=...)`（out-of-sample）が学習データと構造の
    異なる新規データに対しても同じ形状（`float`のリスト）を返すこと。
    """
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    predicted = res.predict(target=target, new_data=new_data)

    assert isinstance(predicted, list)
    assert len(predicted) == 2
    assert all(isinstance(value, float) for value in predicted)


@pytest.mark.parametrize(
    "target", ["expected_latent", "prob_uncensored", "expected_observed"]
)
def test_predict_with_include_intercept_false_and_x_named_const(target):
    """`include_intercept=False`かつ`x`に`"const"`という名前の列を含む場合でも
    `predict(new_data)`が、その列の値を使って正しく計算されること。

    `include_intercept=True`のときのみ`"const"`列名との衝突チェックが働く仕様の
    ため、`include_intercept=False`ならユーザーが`"const"`という名前の（切片では
    ない）通常の説明変数を`x`に含めることは正当な入力。`predict()`が列名から
    「自動追加された切片列かどうか」を推測すると、この列の値を無視して1.0固定にして
    しまう（OLS・Logit・Probitで過去にあった回帰バグと同じ型）。期待値は、同じ推定結果の
    係数と`sigma`から閉形式（下限0の左打ち切り）で組み立てる。
    """
    rng = np.random.default_rng(0)
    n = 60
    const = rng.uniform(1.0, 4.0, n)
    x2 = rng.normal(size=n)
    y = np.maximum(-1.0 + 1.2 * const - 0.5 * x2 + rng.normal(size=n), 0.0)
    df = pl.DataFrame({"y": y, "const": const, "x2": x2})
    options = TobitOptions(include_intercept=False)
    res = Tobit(df, y="y", x=["const", "x2"], options=options).fit()
    assert res.converged

    # `const`が1.0固定扱いになると予測値が大きく変わる、1から離れた値を含める。
    new_data = pl.DataFrame({"const": [3.0, 0.2, 1.0], "x2": [1.0, -0.5, 0.0]})
    xb = (
        res.params["const"] * new_data["const"].to_numpy()
        + res.params["x2"] * new_data["x2"].to_numpy()
    )
    z = xb / res.sigma
    expected = {
        "expected_latent": xb,
        "prob_uncensored": norm.cdf(z),
        "expected_observed": xb * norm.cdf(z) + res.sigma * norm.pdf(z),
    }[target]

    predicted = res.predict(new_data, target=target)

    assert len(predicted) == 3
    rtol = TOLERANCES["tobit_api"]["rtol_predict_closed_form"]
    for i, (value, exp) in enumerate(zip(predicted, expected, strict=True)):
        assert_close(value, exp, f"{target}/{i}", rtol=rtol, atol=0.0)


def test_predict_augment_take_new_data_as_first_positional_arg(
    censored_dataset,
):
    """`predict`/`augment`の第1引数が`new_data`であること（他手法と同じ
    `res.predict(new_df)`の書き方がそのまま通る）。
    """
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    assert res.predict(new_data) == res.predict(new_data=new_data)
    augmented = res.augment(new_data)
    assert augmented.height == 2
    assert "predicted_expected_observed" in augmented.columns


# ── augment() ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "target",
    [
        "expected_latent",
        "expected_observed",
        "prob_uncensored",
        "expected_uncensored",
    ],
)
def test_augment_none_returns_training_data_with_predicted_column(
    censored_dataset, target
):
    """`augment(new_data=None)`が、学習データの全列＋`"predicted_{target}"`
    列を持つDataFrameを、`predict()`と同じ予測値・元データと同じ行順で
    返すこと。
    """
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()

    augmented = res.augment(target=target)
    column_name = f"predicted_{target}"

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == censored_dataset.height
    assert augmented.columns == [*censored_dataset.columns, column_name]
    for col in censored_dataset.columns:
        assert augmented[col].to_list() == censored_dataset[col].to_list()

    expected = res.predict(target=target)
    assert augmented[column_name].to_list() == expected


def test_augment_new_data_returns_new_data_with_predicted_column(
    censored_dataset,
):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})

    augmented = res.augment(target="expected_observed", new_data=new_data)

    assert isinstance(augmented, pl.DataFrame)
    assert augmented.height == 2
    assert augmented.columns == ["x1", "x2", "predicted_expected_observed"]

    expected = res.predict(target="expected_observed", new_data=new_data)
    assert augmented["predicted_expected_observed"].to_list() == expected


def test_augment_without_intercept_matches_predict(censored_dataset):
    """`include_intercept=False`でfitした場合も`augment()`が`predict()`と
    同じ予測値を返すこと（`augment()`はRust側で`predict()`とは別に
    `has_intercept`分岐を実装しているため、個別に確認する。OLSの
    `test_augment_without_intercept_matches_predict`と同型）。
    """
    options = TobitOptions(include_intercept=False)
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"], options=options).fit()

    augmented_none = res.augment()
    expected_none = res.predict()
    assert (
        augmented_none["predicted_expected_observed"].to_list()
        == expected_none
    )

    new_data = pl.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})
    augmented_new = res.augment(new_data=new_data)
    expected_new = res.predict(new_data=new_data)
    assert (
        augmented_new["predicted_expected_observed"].to_list() == expected_new
    )


def test_augment_different_targets_do_not_collide_on_same_dataframe(
    censored_dataset,
):
    """`predicted_{target}`という列名にした狙い:
    Logit/Probitのような固定名`"predicted_probability"`だと、複数の`target`を
    同じDataFrameに積み上げようとした2回目の`augment()`呼び出しが列名衝突で
    失敗する。`target`依存の列名ならこれが起きない。
    """
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()

    once = res.augment(target="expected_observed")
    twice = res.augment(target="prob_uncensored", new_data=once)

    assert "predicted_expected_observed" in twice.columns
    assert "predicted_prob_uncensored" in twice.columns


# ── censoring_fit_check() ─────────────────────────────────────────────


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


@pytest.mark.parametrize(
    ("lower_quantile", "expected_categories"),
    [
        (None, {"uncensored", "upper"}),
        (0.05, {"lower", "uncensored", "upper"}),
    ],
    ids=["right_only", "two_sided"],
)
def test_censoring_fit_check_includes_upper_when_upper_is_set(
    dataset, lower_quantile, expected_categories
):
    """`upper`を指定したモデルでは`"upper"`カテゴリが現れる（右打ち切りのみ・
    両側打ち切り）。観測された割合・モデルが示す割合はそれぞれ全カテゴリで
    合計1になり、`"upper"`の観測割合は`y`が上限ちょうどの観測の割合に一致する。
    モデルが示す割合は、各カテゴリで観測割合の近く（0.05以内）にある。両側の
    ケースは下限（約5%）と上限（約20%）の打ち切り率を大きく変えてあり、
    `"lower"`と`"upper"`のモデル割合の取り違えもこの近さの確認で検出できる。
    共有`dataset`の`y`を分位点で打ち切るのは、`censored_dataset`が下限0の
    左打ち切りのみで上限側を持たないため。
    """
    upper = float(dataset["y"].quantile(0.8))
    lower = (
        None
        if lower_quantile is None
        else float(dataset["y"].quantile(lower_quantile))
    )
    y = dataset["y"].clip(lower_bound=lower, upper_bound=upper)
    df = dataset.with_columns(y.alias("y"))
    res = Tobit(
        df,
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(lower=lower, upper=upper),
    ).fit()
    check = {row["category"]: row for row in res.censoring_fit_check()}

    assert set(check) == expected_categories
    for key in ("observed_rate", "model_implied_rate"):
        total = sum(row[key] for row in check.values())
        assert total == pytest.approx(1.0), key
    for category, row in check.items():
        assert row["model_implied_rate"] == pytest.approx(
            row["observed_rate"], abs=0.05
        ), category
    observed_upper = float((df["y"] == upper).mean())
    assert 0.0 < observed_upper < 1.0
    assert check["upper"]["observed_rate"] == pytest.approx(observed_upper)


# ── marginal_effects() ────────────────────────────────────────────────


def test_marginal_effects_default_excludes_intercept(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    effects = res.marginal_effects()

    assert [row["param_name"] for row in effects] == ["x1", "x2"]
    expected_keys = {
        "param_name",
        "effect",
        "std_error",
        "test_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in effects:
        assert set(row.keys()) == expected_keys


@pytest.mark.parametrize(
    "target",
    [
        "expected_latent",
        "expected_observed",
        "prob_uncensored",
        "expected_uncensored",
    ],
)
def test_marginal_effects_accepts_all_targets(censored_dataset, target):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    effects = res.marginal_effects(target=target)
    assert [row["param_name"] for row in effects] == ["x1", "x2"]


def test_marginal_effects_mean_and_median_differ_from_overall(
    censored_dataset,
):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    overall = [row["effect"] for row in res.marginal_effects(at="overall")]
    mean = [row["effect"] for row in res.marginal_effects(at="mean")]
    median = [row["effect"] for row in res.marginal_effects(at="median")]

    assert overall != mean
    assert overall != median


@pytest.mark.parametrize("at", ["overall", "mean", "median"])
@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
def test_marginal_effects_at_is_case_insensitive(
    censored_dataset, at, spelling
):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    assert res.marginal_effects(at=respell(at, spelling)) == (
        res.marginal_effects(at=at)
    )
