"""OLS の Breusch-Pagan 検定（`OLSResults.breusch_pagan_test()`）のテスト。

役割分担（`test_ols_{api,validation,reference,crosscheck}.py`の4分割と同じ区分を、
診断検定1つを1ファイルにまとめて持つ）:

- 構造・オプション反映（`statistic`の選択・`variables`の既定と明示・`aux_terms`/
  `dropped_terms`・`include_intercept=False`・`cov_type`非依存・スケール不変・結果型）
- `TypeError`/`ValidationError`/`ComputationError`パス
- 主リファレンス（statsmodels `het_breuschpagan(robust=True)`、`ols_breusch_pagan.json`）
  との数値照合。教科書（Wooldridge 例8.4）の公表値とも照合する
- 独立実装（R `lmtest::bptest`＋同じ補助回帰のlm、`ols_breusch_pagan_crosscheck.json`）との
  クロスチェック

補助回帰の変数`Z`はモデルの説明変数（既定）に限らず、モデルの一部・モデルに入っていない列でもよい。
`Z`に入った定数列・重複列を除いてランクに基づく自由度を使う挙動は、合成データで
statsmodels（除いた後の列を渡す）・R（そのまま渡す）の両方と照合する。
"""

from __future__ import annotations

import dataclasses
import json
from functools import partial
from pathlib import Path

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from _assertions import assert_close
from _constants import DATA_DIR
from _error_messages import escaped, fully_qualified_type_name
from _helpers import (
    ROW_TIME,
    with_cluster_groups,
    with_row_time,
    wooldridge_loader,
)
from _tolerances import TOLERANCES
from econometricsmodels import (
    OLS,
    BreuschPaganTestResult,
    ComputationError,
    DiagnosticResult,
    IVOptions,
    OLSOptions,
    ValidationError,
)
from statsmodels.stats.diagnostic import het_breuschpagan

from benchmark.linear.constants import (
    BP_SYNTHETIC_CASES,
    BP_WOOLDRIDGE_CASES,
)
from benchmark.linear.datasets import resolve_bp_case

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"

REF_RTOL = TOLERANCES["ols_breusch_pagan_reference"]["rtol"]
REF_ATOL = TOLERANCES["ols_breusch_pagan_reference"]["atol"]
REF_ATOL_P = TOLERANCES["ols_breusch_pagan_reference"]["atol_p_value"]
CC_RTOL = TOLERANCES["ols_breusch_pagan_crosscheck"]["rtol_strict"]
CC_ATOL = TOLERANCES["ols_breusch_pagan_crosscheck"]["atol"]
CC_ATOL_P = TOLERANCES["ols_breusch_pagan_crosscheck"]["atol_p_value"]

# 統計量は絶対誤差フロアを併用し、p値は相対誤差のみで比較する（裾のp値を検証するため）。
_ref_close = partial(assert_close, rtol=REF_RTOL, atol=REF_ATOL)
_ref_close_p = partial(assert_close, rtol=REF_RTOL, atol=REF_ATOL_P)
_cc_close = partial(assert_close, rtol=CC_RTOL, atol=CC_ATOL)
_cc_close_p = partial(assert_close, rtol=CC_RTOL, atol=CC_ATOL_P)

# 教科書の公表値（Wooldridge, Introductory Econometrics, 例8.4）。小数2桁の丸めなので
# 絶対誤差0.005で比較する。
TEXTBOOK_HPRICE1 = {
    "hprice1": {"lm": 14.09, "f": 5.34},
    "hprice1_log": {"lm": 4.22, "f": 1.41},
}
TEXTBOOK_ATOL = 0.005

# 合成データのケースのうち、補助回帰から除かれる変数が決まっているもの。
SYNTHETIC_EXPECTED_DROPPED = {
    "baseline_constant_and_duplicate": ["one", "x1_copy"],
}


def _synthetic(scenario: str) -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    return df, [c for c in df.columns if c not in ("y", "weight")]


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    lhs, rhs = formula.split("~")
    return lhs.strip(), [t.strip() for t in rhs.split("+")]


def _fit_synthetic_case(name: str):
    """ケース定義どおりに当てはめた結果と、`breusch_pagan_test`に渡す`variables`。"""
    case = BP_SYNTHETIC_CASES[name]
    df, _ = _synthetic(case["scenario"])
    df, x_cols, variables = resolve_bp_case(case, df)
    options = OLSOptions(include_intercept=case.get("include_intercept", True))
    return OLS(df, y="y", x=x_cols, options=options).fit(), variables


def _fit_wooldridge_case(name: str, load_wooldridge):
    case = BP_WOOLDRIDGE_CASES[name]
    y, x_cols = _parse_formula(case["formula"])
    fitted = OLS(load_wooldridge(case["dataset"]), y=y, x=x_cols).fit()
    return fitted, case.get("variables")


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads((FIXTURES_DIR / "ols_breusch_pagan.json").read_text())


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads(
        (FIXTURES_DIR / "ols_breusch_pagan_crosscheck.json").read_text()
    )


@pytest.fixture(scope="module")
def load_wooldridge():
    return wooldridge_loader()


@pytest.fixture
def baseline() -> OLS:
    df, x_cols = _synthetic("baseline")
    return OLS(df, y="y", x=x_cols)


# ── 構造・オプション反映 ────────────────────────────────────────────


def test_default_returns_lm_version_with_chi2_distribution(baseline):
    res = baseline.fit().breusch_pagan_test()

    assert isinstance(res, BreuschPaganTestResult)
    assert isinstance(res, DiagnosticResult)
    assert res.distribution == "chi2"
    assert res.df_denom is None
    assert res.df == 3
    assert 0.0 <= res.p_value <= 1.0


def test_f_version_has_f_distribution_and_denominator_degrees_of_freedom(
    baseline,
):
    fitted = baseline.fit()
    res = fitted.breusch_pagan_test(statistic="f")

    assert res.distribution == "f"
    assert res.df == 3
    assert res.df_denom == fitted.n_obs - res.df - 1
    assert 0.0 <= res.p_value <= 1.0


def test_statistic_argument_is_case_insensitive(baseline):
    fitted = baseline.fit()
    assert fitted.breusch_pagan_test(statistic="LM") == (
        fitted.breusch_pagan_test(statistic="lm")
    )
    assert fitted.breusch_pagan_test(statistic="F") == (
        fitted.breusch_pagan_test(statistic="f")
    )


def test_statistic_can_be_given_as_the_second_positional_argument(baseline):
    fitted = baseline.fit()
    assert fitted.breusch_pagan_test(None, "f") == fitted.breusch_pagan_test(
        statistic="f"
    )


def test_variables_default_to_the_model_independent_variables(baseline):
    fitted = baseline.fit()
    default = fitted.breusch_pagan_test()

    assert default.aux_terms == ["const", "x1", "x2", "x3"]
    assert default.dropped_terms == []
    assert default.df == len(default.aux_terms) - 1
    assert fitted.breusch_pagan_test(["x1", "x2", "x3"]) == default
    assert fitted.breusch_pagan_test(variables=None) == default


def test_explicit_variables_define_the_auxiliary_regression(baseline):
    res = baseline.fit().breusch_pagan_test(["x2", "x1"])

    assert res.aux_terms == ["const", "x2", "x1"]
    assert res.df == 2


def test_variables_may_be_columns_that_are_not_in_the_model():
    df, _ = _synthetic("baseline")
    fitted = OLS(df, y="y", x=["x1"]).fit()
    res = fitted.breusch_pagan_test(["x2", "x3"])

    assert res.aux_terms == ["const", "x2", "x3"]
    assert res.df == 2
    assert res != fitted.breusch_pagan_test()


def test_variables_may_include_the_dependent_variable(baseline):
    """`y`を指定しても拒否しない（`y`に依存する分散は意味のあるケースがある）。"""
    res = baseline.fit().breusch_pagan_test(["y"])
    assert res.aux_terms == ["const", "y"]
    assert res.df == 1


def test_boolean_and_integer_variables_are_accepted():
    df, _ = _synthetic("baseline")
    df = df.with_columns(
        (pl.col("x3") > 0).alias("flag"),
        (pl.col("x3") > 0).cast(pl.Float64).alias("flag_f"),
        (pl.col("x3") > 0).cast(pl.Int64).alias("flag_int"),
    )
    fitted = OLS(df, y="y", x=["x1"]).fit()

    flag = fitted.breusch_pagan_test(["flag"])
    assert flag.aux_terms == ["const", "flag"]
    assert flag.statistic == fitted.breusch_pagan_test(["flag_f"]).statistic
    assert fitted.breusch_pagan_test(["flag_int"]).statistic == flag.statistic


def test_lm_and_f_versions_report_the_same_terms(baseline):
    fitted = baseline.fit()
    lm, f = (
        fitted.breusch_pagan_test(),
        fitted.breusch_pagan_test(statistic="f"),
    )
    assert lm.aux_terms == f.aux_terms
    assert lm.dropped_terms == f.dropped_terms
    assert lm.df == f.df


def test_aux_regression_has_a_constant_without_an_intercept_in_the_model():
    """`include_intercept=False`でも補助回帰には定数を含める（`aux_terms`の
    先頭は常に`"const"`）。"""
    df, x_cols = _synthetic("baseline")
    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=x_cols, options=options).fit().breusch_pagan_test()

    assert res.aux_terms == ["const", "x1", "x2", "x3"]
    assert res.df == 3


def test_constant_and_duplicate_variables_are_dropped():
    """定数列と、先に指定した列と同一の列は補助回帰から除き、自由度は残った列数で数える。"""
    fitted, variables = _fit_synthetic_case("baseline_constant_and_duplicate")
    res = fitted.breusch_pagan_test(variables)

    assert res.aux_terms == ["const", "x1", "x2"]
    assert res.dropped_terms == ["one", "x1_copy"]
    assert res.df == 2
    # 除いた後の変数を直接指定した結果と一致する。
    assert res == dataclasses.replace(
        fitted.breusch_pagan_test(["x1", "x2"]),
        dropped_terms=["one", "x1_copy"],
    )


def test_result_does_not_depend_on_cov_type(baseline):
    """Breusch-Pagan検定は古典的な等分散を仮定する補助回帰のLM検定で、
    `cov_type`に依存しない。"""
    df, x_cols = _synthetic("baseline")
    classical = baseline.fit().breusch_pagan_test()
    robust = (
        OLS(df, y="y", x=x_cols, options=OLSOptions(cov_type="hc1"))
        .fit()
        .breusch_pagan_test()
    )
    assert robust == classical


@pytest.mark.parametrize(
    "cov_type", ["hc0", "hc1", "hc2", "hc3", "hac", "cluster"]
)
def test_result_does_not_depend_on_any_cov_type(cov_type):
    """`hac`（時間列）・`cluster`（クラスター列）のように追加の列引数を持つ`cov_type`でも、
    説明変数の再抽出が崩れず結果が変わらない。"""
    df, x_cols = _synthetic("baseline")
    df = with_cluster_groups(with_row_time(df), 10)
    kwargs = {
        "hac": {"hac_lags": 1, "hac_time": ROW_TIME},
        "cluster": {"cluster": "cluster_group"},
    }.get(cov_type, {})

    classical = OLS(df, y="y", x=x_cols).fit().breusch_pagan_test()
    res = (
        OLS(
            df,
            y="y",
            x=x_cols,
            options=OLSOptions(cov_type=cov_type, **kwargs),
        )
        .fit()
        .breusch_pagan_test()
    )
    assert res == classical


def test_result_is_invariant_to_scale_and_location_of_the_variables():
    """補助回帰の`R²`はスケール・平行移動で変わらない。平均が標準偏差より桁違いに
    大きい変数（賃金・人口等）でも失敗せず同じ結果になる。"""
    df, x_cols = _synthetic("baseline")
    base = OLS(df, y="y", x=x_cols).fit().breusch_pagan_test()

    shifted = df.with_columns(
        (pl.col("x1") * 1.0e3 + 5.0e3).alias("x1"),
        (pl.col("x2") * 10.0 + 4.0e4).alias("x2"),
    )
    res = OLS(shifted, y="y", x=x_cols).fit().breusch_pagan_test()

    assert res.df == base.df
    _cc_close(res.statistic, base.statistic, "statistic")
    # 残差自体が丸め誤差の分だけ変わるため、p値は少し緩く比較する。
    assert res.p_value == pytest.approx(base.p_value, rel=1e-6)


def test_result_is_invariant_to_variable_order_and_scale_of_y(baseline):
    df, x_cols = _synthetic("baseline")
    fitted = baseline.fit()
    base = fitted.breusch_pagan_test()

    reordered = fitted.breusch_pagan_test(list(reversed(x_cols)))
    assert reordered.df == base.df
    _cc_close(reordered.statistic, base.statistic, "reordered/statistic")
    assert reordered.aux_terms[1:] == list(reversed(x_cols))

    scaled_y = df.with_columns((pl.col("y") * 1.0e4).alias("y"))
    res = OLS(scaled_y, y="y", x=x_cols).fit().breusch_pagan_test()
    _cc_close(res.statistic, base.statistic, "scaled_y/statistic")


def test_to_dict_is_json_ready(baseline):
    res = baseline.fit().breusch_pagan_test(statistic="f")
    as_dict = res.to_dict()

    assert set(as_dict) == {
        "statistic",
        "p_value",
        "df",
        "df_denom",
        "distribution",
        "aux_terms",
        "dropped_terms",
    }
    assert json.loads(json.dumps(as_dict)) == as_dict


def test_result_is_frozen(baseline):
    res = baseline.fit().breusch_pagan_test()
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.p_value = 0.0  # type: ignore[misc]


def test_breusch_pagan_test_is_not_computed_by_fit(baseline):
    """事後診断は`fit()`では計算しない。結果の公開属性に検定の値（プロパティ）は無く、
    `breusch_pagan_test`というメソッドだけがある。"""
    fitted = baseline.fit()
    assert [n for n in dir(fitted) if "pagan" in n] == ["breusch_pagan_test"]
    assert callable(fitted.breusch_pagan_test)


def test_regressor_named_const_gets_an_unambiguous_first_constant():
    """`x`に`"const"`という名前の列（定数ではない）があると`aux_terms`に`"const"`が
    2つ現れるが、先頭が補助回帰の定数である。"""
    df, _ = _synthetic("baseline")
    df = df.rename({"x1": "const"})
    options = OLSOptions(include_intercept=False)
    fitted = OLS(df, y="y", x=["const", "x2"], options=options).fit()
    res = fitted.breusch_pagan_test()

    assert res.aux_terms == ["const", "const", "x2"]


# ── TypeError / ValidationError / ComputationError ──────────────────


@pytest.mark.parametrize(
    "variables", ["x1", ("x1", "x2"), {"x1"}, 1, pl.Series(["x1"])]
)
def test_variables_that_is_not_a_list_raises_type_error(baseline, variables):
    """型違いは`ValidationError`ではなく組み込みの`TypeError`
    （`docs/guide/validation.md`の分担）。"""
    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.VARIABLES_NOT_A_LIST,
            type_name=fully_qualified_type_name(variables),
        ),
    ):
        baseline.fit().breusch_pagan_test(variables)


def test_variables_element_that_is_not_a_str_raises_type_error(baseline):
    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.VARIABLES_ELEMENT_NOT_STR, index=1, type_name="int"
        ),
    ):
        baseline.fit().breusch_pagan_test(["x1", 2])


def test_non_string_statistic_raises_type_error(baseline):
    with pytest.raises(TypeError):
        baseline.fit().breusch_pagan_test(statistic=1)  # type: ignore[arg-type]


def test_none_statistic_raises_type_error(baseline):
    """`statistic`に`None`は渡せない（既定は`"lm"`。`variables`と違い`None`は無効）。"""
    with pytest.raises(TypeError):
        baseline.fit().breusch_pagan_test(statistic=None)  # type: ignore[arg-type]


def test_unknown_statistic_raises_validation_error(baseline):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_DIAGNOSTIC_STATISTIC, other="chi2"),
    ):
        baseline.fit().breusch_pagan_test(statistic="chi2")


def test_empty_variables_raises_validation_error(baseline):
    with pytest.raises(
        ValidationError, match=escaped(msgs.X_EMPTY, role="variables")
    ):
        baseline.fit().breusch_pagan_test([])


def test_duplicate_variables_raise_validation_error(baseline):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.DUPLICATE_WITHIN_ROLE, name="x1", role="variables"),
    ):
        baseline.fit().breusch_pagan_test(["x1", "x2", "x1"])


def test_missing_column_raises_validation_error(baseline):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nope"),
    ):
        baseline.fit().breusch_pagan_test(["x1", "nope"])


def test_variable_with_an_unsupported_dtype_raises_validation_error():
    df, _ = _synthetic("baseline")
    df = df.with_columns(pl.lit("a").alias("label"))
    fitted = OLS(df, y="y", x=["x1"]).fit()

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="label", dtype="String"
        ),
    ):
        fitted.breusch_pagan_test(["label"])


def test_variable_with_missing_values_raises_validation_error():
    """モデルに入れていない列の欠損値も自動除外せず拒否する（欠損値方針は`fit()`と同じ）。
    当てはめは`z`を使わないので通り、`Z`に使うとき初めて拒否される。"""
    df, _ = _synthetic("baseline")
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 3)
        .then(None)
        .otherwise(pl.col("x2"))
        .alias("z")
    )
    fitted = OLS(df, y="y", x=["x1"]).fit()

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="z", count=1),
    ):
        fitted.breusch_pagan_test(["z"])


@pytest.mark.parametrize(
    ("value", "kind"), [(float("nan"), "NaN"), (float("inf"), "inf")]
)
def test_variable_with_non_finite_values_raises_validation_error(value, kind):
    df, _ = _synthetic("baseline")
    bad = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(value)
        .otherwise(pl.col("x2"))
        .alias("z")
    )
    # 当てはめは`z`を使わないので通る。`Z`に使うとき初めて拒否される。
    fitted = OLS(bad, y="y", x=["x1"]).fit()

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="z", value=kind, row=2
        ),
    ):
        fitted.breusch_pagan_test(["z"])


def test_too_few_observations_for_the_auxiliary_regression_raises():
    """`n = q + 1`（定数込みの列数と同数）は拒否される。`n = q + 2`は成功する
    （`test_smallest_sample_matches_statsmodels`）。境界の`<=`/`<`の取り違えを
    検出するための組。"""
    rng = np.random.default_rng(11)
    n = 3
    x1, x2 = rng.normal(size=(2, n))
    y = 1.0 + x1 + rng.normal(size=n)
    fitted = OLS(
        pl.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y", x=["x1"]
    ).fit()

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_OBSERVATIONS_AUX_REGRESSION, n=3, k=3),
    ):
        fitted.breusch_pagan_test(["x1", "x2"])


def test_sample_size_is_checked_against_the_variables_that_remain_after_dropping():
    """名目の補助回帰は定数込み4列（n=4では足りない）だが、定数列を除くと3列でn=4は
    足りる。メッセージの`k`も除外後の列数。"""
    x1 = [0.5, 1.2, -0.3, 2.1]
    y = [1.1, 2.9, 0.4, 5.2]
    df = pl.DataFrame(
        {"y": y, "x1": x1, "x2": [0.3, -1.0, 0.8, 0.1], "one": [1.0] * 4}
    )
    fitted = OLS(df, y="y", x=["x1"]).fit()

    ok = fitted.breusch_pagan_test(["one", "x1", "x2"], "f")
    assert ok.df == 2
    assert ok.df_denom == 1

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_OBSERVATIONS_AUX_REGRESSION, n=3, k=3),
    ):
        OLS(df.head(3), y="y", x=["x1"]).fit().breusch_pagan_test(
            ["one", "x1", "x2"]
        )


def test_result_without_training_data_raises_validation_error():
    """`IVResults.first_stage()`が返す`OLSResults`は単一のソースDataFrameを持たず、
    変数を再抽出できない（`white_test()`と同じ）。"""
    rng = np.random.default_rng(3)
    n = 120
    z1, z2, x1 = rng.normal(size=(3, n))
    endog = 0.7 * z1 + 0.4 * z2 + 0.3 * x1 + rng.normal(size=n)
    y = 1.0 + 0.5 * endog + 0.2 * x1 + rng.normal(size=n)
    df = pl.DataFrame({"y": y, "endog1": endog, "x1": x1, "z1": z1, "z2": z2})
    from econometricsmodels import IV

    res = IV(
        df,
        y="y",
        x_endog=["endog1"],
        x_exog=["x1"],
        instruments=["z1", "z2"],
        options=IVOptions(),
    ).fit()
    first_stage = res.first_stage()["endog1"]

    with pytest.raises(
        ValidationError, match=escaped(msgs.BP_NO_TRAINING_DATA)
    ):
        first_stage.breusch_pagan_test()


def test_all_constant_variables_raise_computation_error():
    df, _ = _synthetic("baseline")
    df = df.with_columns(pl.lit(3.0).alias("c"))
    fitted = OLS(df, y="y", x=["x1"]).fit()

    with pytest.raises(
        ComputationError, match=escaped(msgs.BP_AUX_REGRESSION_PREFIX)
    ):
        fitted.breusch_pagan_test(["c"])


def test_collinear_but_not_identical_variables_raise_computation_error():
    """定数と完全に同一な列だけを除く。`2 * x1`のように同一ではない共線列は
    除かれず、補助回帰が特異になる（`fit()`と同じく共線は`ComputationError`）。"""
    df, _ = _synthetic("baseline")
    df = df.with_columns((pl.col("x1") * 2.0).alias("x1_double"))
    fitted = OLS(df, y="y", x=["x1"]).fit()

    with pytest.raises(
        ComputationError, match=escaped(msgs.BP_AUX_REGRESSION_PREFIX)
    ):
        fitted.breusch_pagan_test(["x1", "x1_double"])


def test_full_set_of_dummies_without_intercept_raises_computation_error():
    """全カテゴリのダミー（和が定数列）を`Z`にすると補助回帰の定数と共線になる。
    重複・定数列の除外では取れないため`ComputationError`にする（`white_test()`と同じ制限）。"""
    rng = np.random.default_rng(5)
    n = 90
    group = np.arange(n) % 3
    df = pl.DataFrame(
        {
            "y": rng.normal(size=n) + group,
            "x1": rng.normal(size=n),
            "d1": (group == 0).astype(float),
            "d2": (group == 1).astype(float),
            "d3": (group == 2).astype(float),
        }
    )
    fitted = OLS(df, y="y", x=["x1"]).fit()

    with pytest.raises(
        ComputationError, match=escaped(msgs.BP_AUX_REGRESSION_PREFIX)
    ):
        fitted.breusch_pagan_test(["d1", "d2", "d3"])


# ── 主リファレンス（statsmodels `het_breuschpagan`）との数値照合 ───────


def _check_against(fitted, variables, ref: dict, label: str) -> None:
    lm = fitted.breusch_pagan_test(variables, "lm")
    f = fitted.breusch_pagan_test(variables, "f")

    _ref_close(lm.statistic, ref["lm"], f"{label}/lm")
    _ref_close_p(lm.p_value, ref["lm_p_value"], f"{label}/lm_p_value")
    _ref_close(f.statistic, ref["f"], f"{label}/f")
    _ref_close_p(f.p_value, ref["f_p_value"], f"{label}/f_p_value")
    assert lm.df == ref["df"], f"{label}/df"
    assert fitted.n_obs == ref["n_obs"], f"{label}/n_obs"


@pytest.mark.parametrize("case", list(BP_SYNTHETIC_CASES))
def test_synthetic_matches_statsmodels(reference, case):
    fitted, variables = _fit_synthetic_case(case)

    _check_against(fitted, variables, reference["synthetic"][case], case)
    res = fitted.breusch_pagan_test(variables)
    assert res.dropped_terms == SYNTHETIC_EXPECTED_DROPPED.get(case, [])


def test_baseline_df1_is_the_smallest_sample_with_one_denominator_degree():
    """`baseline_df1`（n=5、`q=3`）は`n = q + 2`で`df_denom = 1`の成功パス。
    White検定は補助回帰の列数が足りず拒否される（`n = 5`、定数込み10列）ので、この境界は
    BP検定のフィクスチャ（statsmodels・R）でだけ確認できる。"""
    fitted, variables = _fit_synthetic_case("baseline_df1")
    res = fitted.breusch_pagan_test(variables, "f")

    assert fitted.n_obs == 5
    assert res.df == 3
    assert res.df_denom == 1


def test_heteroskedastic_variance_is_rejected_when_z_is_the_driving_variable():
    """`heteroskedastic`の誤差分散は`|x1|`に比例して増える。`Z`をモデルのxにすると
    （`x1`に対して対称なので）棄却されないが、`Z = |x1|`なら強く棄却される
    （裾のp値の経路。フィクスチャでstatsmodels・Rと照合する）。"""
    fitted, variables = _fit_synthetic_case("heteroskedastic_abs_x1")
    assert variables == ["abs_x1"]
    assert fitted.breusch_pagan_test(variables).p_value < 1e-10

    fitted_default, _ = _fit_synthetic_case("heteroskedastic")
    assert fitted_default.breusch_pagan_test().p_value > 0.05


@pytest.mark.parametrize("case", list(BP_WOOLDRIDGE_CASES))
def test_wooldridge_matches_statsmodels(reference, load_wooldridge, case):
    fitted, variables = _fit_wooldridge_case(case, load_wooldridge)
    _check_against(fitted, variables, reference["wooldridge"][case], case)


@pytest.mark.parametrize("case", list(TEXTBOOK_HPRICE1))
def test_hprice1_matches_the_textbook_values(load_wooldridge, case):
    """教科書（Wooldridge 例8.4）に載っている住宅価格モデルのLM・F統計量。価格の水準では
    不均一分散が強く、対数にすると弱まる。"""
    fitted, variables = _fit_wooldridge_case(case, load_wooldridge)
    expected = TEXTBOOK_HPRICE1[case]

    assert fitted.breusch_pagan_test(variables, "lm").statistic == (
        pytest.approx(expected["lm"], abs=TEXTBOOK_ATOL)
    )
    assert fitted.breusch_pagan_test(variables, "f").statistic == (
        pytest.approx(expected["f"], abs=TEXTBOOK_ATOL)
    )


def test_smallest_sample_matches_statsmodels():
    """観測数が補助回帰の列数（定数込み）より1つだけ多い境界（`n = q + 2`、
    `df_denom = 1`）の成功パスを、ライブのstatsmodelsと照合する。"""
    rng = np.random.default_rng(11)
    n = 4  # 2変数の補助回帰は定数込み3列
    x1, x2 = rng.normal(size=(2, n))
    y = 1.0 + x1 - 0.5 * x2 + rng.normal(size=n)
    fitted = OLS(
        pl.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y", x=["x1"]
    ).fit()

    lm = fitted.breusch_pagan_test(["x1", "x2"], "lm")
    f = fitted.breusch_pagan_test(["x1", "x2"], "f")
    assert lm.df == 2
    assert f.df_denom == 1

    resid = sm.OLS(y, sm.add_constant(x1)).fit().resid
    sm_lm, sm_lm_p, sm_f, sm_f_p = het_breuschpagan(
        resid, sm.add_constant(np.column_stack([x1, x2])), robust=True
    )
    _ref_close(lm.statistic, sm_lm, "n=q+2/lm")
    _ref_close_p(lm.p_value, sm_lm_p, "n=q+2/lm_p_value")
    _ref_close(f.statistic, sm_f, "n=q+2/f")
    _ref_close_p(f.p_value, sm_f_p, "n=q+2/f_p_value")


# ── 独立実装（R）とのクロスチェック ─────────────────────────────────


def _check_against_r(fitted, variables, ref: dict, label: str) -> None:
    lm = fitted.breusch_pagan_test(variables, "lm")
    f = fitted.breusch_pagan_test(variables, "f")

    _cc_close(lm.statistic, ref["lm"], f"{label}/lm")
    _cc_close_p(lm.p_value, ref["lm_p_value"], f"{label}/lm_p_value")
    _cc_close(f.statistic, ref["f"], f"{label}/f")
    _cc_close_p(f.p_value, ref["f_p_value"], f"{label}/f_p_value")
    # 自由度: Rのbptestのparameter（ランクに基づく）と補助回帰のlmのF検定の自由度。
    assert lm.df == ref["df"], f"{label}/df"
    assert f.df == ref["f_df_num"], f"{label}/f_df_num"
    assert f.df_denom == ref["f_df_denom"], f"{label}/f_df_denom"
    assert fitted.n_obs == ref["n_obs"], f"{label}/n_obs"


@pytest.mark.parametrize("case", list(BP_SYNTHETIC_CASES))
def test_synthetic_matches_r(crosscheck, case):
    """`baseline_constant_and_duplicate`は`Z`の定数列・重複列をそのままRに渡す。Rは
    エイリアスとしてランクに基づく自由度を使い、本実装が除いて数えた自由度と一致する。"""
    fitted, variables = _fit_synthetic_case(case)
    _check_against_r(fitted, variables, crosscheck["synthetic"][case], case)


@pytest.mark.parametrize("case", list(BP_WOOLDRIDGE_CASES))
def test_wooldridge_matches_r(crosscheck, load_wooldridge, case):
    fitted, variables = _fit_wooldridge_case(case, load_wooldridge)
    _check_against_r(fitted, variables, crosscheck["wooldridge"][case], case)
