"""OLS の White 検定（`OLSResults.white_test()`）のテスト。

役割分担（`test_ols_{api,validation,reference,crosscheck}.py`の4分割と同じ区分を、
診断検定1つを1ファイルにまとめて持つ）:

- 構造・オプション反映（`statistic`の選択・`aux_terms`/`dropped_terms`・
  `include_intercept=False`・`cov_type`非依存・スケール不変・結果型）
- `ValidationError`/`ComputationError`パス
- 主リファレンス（statsmodels `het_white`、`ols_white.json`）との数値照合
- 独立実装（R `lmtest::bptest`＋同じ補助回帰のlm、`ols_white_crosscheck.json`）とのクロスチェック

補助回帰の項（`x`・`x^2`・`x1:x2`）から定数列・重複列（ダミーの二乗等）を除いてランクに
基づく自由度を使う挙動は、実データ（`wage1_dummies`）でstatsmodels・Rの両方と照合する。
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
from _error_messages import escaped
from _helpers import (
    ROW_TIME,
    with_cluster_groups,
    with_row_time,
    wooldridge_loader,
)
from _tolerances import TOLERANCES
from econometricsmodels import (
    OLS,
    ComputationError,
    DiagnosticResult,
    IVOptions,
    OLSOptions,
    ValidationError,
    WhiteTestResult,
)
from statsmodels.stats.diagnostic import het_white

from benchmark.linear.constants import (
    WHITE_SYNTHETIC_SCENARIOS,
    WHITE_WOOLDRIDGE_CASES,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"

REF_RTOL = TOLERANCES["ols_white_reference"]["rtol"]
REF_ATOL = TOLERANCES["ols_white_reference"]["atol"]
REF_ATOL_P = TOLERANCES["ols_white_reference"]["atol_p_value"]
CC_RTOL = TOLERANCES["ols_white_crosscheck"]["rtol_strict"]
CC_ATOL = TOLERANCES["ols_white_crosscheck"]["atol"]
CC_ATOL_P = TOLERANCES["ols_white_crosscheck"]["atol_p_value"]

# 統計量は絶対誤差フロアを併用し、p値は相対誤差のみで比較する（裾のp値を検証するため）。
_ref_close = partial(assert_close, rtol=REF_RTOL, atol=REF_ATOL)
_ref_close_p = partial(assert_close, rtol=REF_RTOL, atol=REF_ATOL_P)
_cc_close = partial(assert_close, rtol=CC_RTOL, atol=CC_ATOL)
_cc_close_p = partial(assert_close, rtol=CC_RTOL, atol=CC_ATOL_P)

# 実データのうち、補助回帰で除かれる項が決まっているケース（除外ルールの実データでの確認）。
WOOLDRIDGE_EXPECTED_DROPPED = {
    "wage1": [],
    "gpa2": [],
    "wage1_dummies": ["female^2", "married^2"],
    "wage1_polynomial": ["exper^2", "tenure^2"],
    "wage1_region": [
        "northcen^2",
        "south^2",
        "west^2",
        "northcen:south",
        "northcen:west",
        "south:west",
    ],
}


def _synthetic(scenario: str) -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    return df, [c for c in df.columns if c not in ("y", "weight")]


def _expected_df(p: int) -> int:
    """重複・定数が無いとき、補助回帰の定数以外の列数`p(p+3)/2`。"""
    return p * (p + 3) // 2


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    lhs, rhs = formula.split("~")
    return lhs.strip(), [t.strip() for t in rhs.split("+")]


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads((FIXTURES_DIR / "ols_white.json").read_text())


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    return json.loads((FIXTURES_DIR / "ols_white_crosscheck.json").read_text())


@pytest.fixture(scope="module")
def load_wooldridge():
    return wooldridge_loader()


@pytest.fixture
def baseline() -> OLS:
    df, x_cols = _synthetic("baseline")
    return OLS(df, y="y", x=x_cols)


# ── 構造・オプション反映 ────────────────────────────────────────────


def test_default_returns_lm_version_with_chi2_distribution(baseline):
    res = baseline.fit().white_test()

    assert isinstance(res, WhiteTestResult)
    assert isinstance(res, DiagnosticResult)
    assert res.distribution == "chi2"
    assert res.df_denom is None
    assert res.df == _expected_df(3)
    assert 0.0 <= res.p_value <= 1.0


def test_f_version_has_f_distribution_and_denominator_degrees_of_freedom(
    baseline,
):
    fitted = baseline.fit()
    res = fitted.white_test("f")

    assert res.distribution == "f"
    assert res.df == _expected_df(3)
    assert res.df_denom == fitted.n_obs - res.df - 1
    assert 0.0 <= res.p_value <= 1.0


def test_statistic_argument_is_case_insensitive(baseline):
    fitted = baseline.fit()
    assert fitted.white_test("LM") == fitted.white_test("lm")
    assert fitted.white_test("F") == fitted.white_test("f")


def test_aux_terms_lists_constant_then_variables_squares_and_products(
    baseline,
):
    res = baseline.fit().white_test()

    assert res.aux_terms == [
        "const",
        "x1",
        "x2",
        "x3",
        "x1^2",
        "x2^2",
        "x3^2",
        "x1:x2",
        "x1:x3",
        "x2:x3",
    ]
    assert res.dropped_terms == []
    assert res.df == len(res.aux_terms) - 1


def test_lm_and_f_versions_report_the_same_terms(baseline):
    fitted = baseline.fit()
    lm, f = fitted.white_test("lm"), fitted.white_test("f")
    assert lm.aux_terms == f.aux_terms
    assert lm.dropped_terms == f.dropped_terms
    assert lm.df == f.df


def test_aux_regression_has_a_constant_without_an_intercept_in_the_model():
    """`include_intercept=False`でも補助回帰には定数を含める（`aux_terms`の
    先頭は常に`"const"`）。"""
    df, x_cols = _synthetic("baseline")
    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=x_cols, options=options).fit().white_test()

    assert res.aux_terms[0] == "const"
    assert res.aux_terms[1:4] == ["x1", "x2", "x3"]
    assert res.df == _expected_df(3)


def test_result_does_not_depend_on_cov_type(baseline):
    """White検定は古典的な等分散を仮定する補助回帰のLM検定で、`cov_type`に依存しない。"""
    df, x_cols = _synthetic("baseline")
    classical = baseline.fit().white_test()
    robust = (
        OLS(df, y="y", x=x_cols, options=OLSOptions(cov_type="hc1"))
        .fit()
        .white_test()
    )
    assert robust == classical


def test_squared_dummy_is_dropped_and_degrees_of_freedom_count_the_rest():
    df, _ = _synthetic("baseline")
    df = df.with_columns((pl.col("x3") > 0).cast(pl.Float64).alias("d"))
    res = OLS(df, y="y", x=["x1", "d"]).fit().white_test()

    assert res.dropped_terms == ["d^2"]
    assert res.aux_terms == ["const", "x1", "d", "x1^2", "x1:d"]
    assert res.df == 4


def test_constant_regressor_terms_are_dropped_in_a_model_without_intercept():
    """`include_intercept=False`のモデルに定数列（1ではない値）を入れた場合、
    その列を含む項（二乗・交差項）は補助回帰の定数と冗長なので全て除かれる。"""
    df, _ = _synthetic("baseline")
    df = df.with_columns(pl.lit(2.0).alias("c"))
    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=["c", "x1"], options=options).fit().white_test()

    assert res.dropped_terms == ["c", "c^2", "c:x1"]
    assert res.aux_terms == ["const", "x1", "x1^2"]


def test_result_is_invariant_to_scale_and_location_of_the_variables():
    """補助回帰の`R²`はスケール・平行移動で変わらない。平均が標準偏差より桁違いに
    大きい変数（賃金・人口等）でも失敗せず同じ結果になる。"""
    df, x_cols = _synthetic("baseline")
    base = OLS(df, y="y", x=x_cols).fit().white_test()

    shifted = df.with_columns(
        (pl.col("x1") * 1.0e3 + 5.0e3).alias("x1"),
        (pl.col("x2") * 10.0 + 4.0e4).alias("x2"),
    )
    res = OLS(shifted, y="y", x=x_cols).fit().white_test()

    assert res.df == base.df
    _cc_close(res.statistic, base.statistic, "statistic")
    # 残差自体が丸め誤差の分だけ変わるため、p値は少し緩く比較する。
    assert res.p_value == pytest.approx(base.p_value, rel=1e-6)


def test_to_dict_is_json_ready(baseline):
    res = baseline.fit().white_test("f")
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
    res = baseline.fit().white_test()
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.p_value = 0.0  # type: ignore[misc]


def test_white_test_is_not_computed_by_fit(baseline):
    """事後診断は`fit()`では計算しない。結果の公開属性に検定の値（プロパティ）は無く、
    `white_test`というメソッドだけがある。"""
    fitted = baseline.fit()
    assert [n for n in dir(fitted) if "white" in n] == ["white_test"]
    assert callable(fitted.white_test)


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

    classical = OLS(df, y="y", x=x_cols).fit().white_test()
    res = (
        OLS(
            df,
            y="y",
            x=x_cols,
            options=OLSOptions(cov_type=cov_type, **kwargs),
        )
        .fit()
        .white_test()
    )
    assert res == classical


def test_result_is_invariant_to_variable_order_and_scale_of_y():
    df, x_cols = _synthetic("baseline")
    base = OLS(df, y="y", x=x_cols).fit().white_test()

    reordered = OLS(df, y="y", x=list(reversed(x_cols))).fit().white_test()
    assert reordered.df == base.df
    _cc_close(reordered.statistic, base.statistic, "reordered/statistic")
    # ラベルの並びは列順に従い、項の集合は変わらない。
    assert reordered.aux_terms[1:4] == list(reversed(x_cols))

    scaled_y = df.with_columns((pl.col("y") * 1.0e4).alias("y"))
    res = OLS(scaled_y, y="y", x=x_cols).fit().white_test()
    _cc_close(res.statistic, base.statistic, "scaled_y/statistic")


def test_regressor_named_const_gets_an_unambiguous_first_constant():
    """`include_intercept=False`で`x`に`"const"`という名前の列（定数ではない）を入れると
    `aux_terms`に`"const"`が2つ現れるが、先頭が補助回帰の定数である。"""
    df, _ = _synthetic("baseline")
    df = df.rename({"x1": "const"})
    options = OLSOptions(include_intercept=False)
    res = OLS(df, y="y", x=["const", "x2"], options=options).fit().white_test()

    assert res.aux_terms == [
        "const",
        "const",
        "x2",
        "const^2",
        "x2^2",
        "const:x2",
    ]


def test_non_string_statistic_raises_type_error(baseline):
    """型違いは`ValidationError`ではなく組み込みの`TypeError`
    （`docs/guide/validation.md`の分担）。"""
    with pytest.raises(TypeError):
        baseline.fit().white_test(1)  # type: ignore[arg-type]


# ── ValidationError / ComputationError ──────────────────────────────


def test_unknown_statistic_raises_validation_error(baseline):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_DIAGNOSTIC_STATISTIC, other="chi2"),
    ):
        baseline.fit().white_test("chi2")


def test_too_few_observations_for_the_auxiliary_regression_raises():
    """`baseline_df1`（n=5）は補助回帰の定数込み10列に対して観測数が足りない。
    元の`fit()`は通る。"""
    df, x_cols = _synthetic("baseline_df1")
    fitted = OLS(df, y="y", x=x_cols).fit()

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_OBSERVATIONS_AUX_REGRESSION, n=5, k=10
        ),
    ):
        fitted.white_test()


def test_observation_count_equal_to_the_auxiliary_columns_raises():
    """`n = k`（2変数の補助回帰は定数込み6列、n=6）は拒否される。`n = k + 1`は
    成功する（`test_smallest_sample_matches_statsmodels`）。境界の`<=`/`<`の取り違えを
    検出するための組。"""
    rng = np.random.default_rng(11)
    n = 6
    x1, x2 = rng.normal(size=(2, n))
    y = 1.0 + x1 - 0.5 * x2 + rng.normal(size=n)
    fitted = OLS(
        pl.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y", x=["x1", "x2"]
    ).fit()

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_OBSERVATIONS_AUX_REGRESSION, n=6, k=6),
    ):
        fitted.white_test()


def test_sample_size_is_checked_against_the_terms_that_remain_after_dropping():
    """名目の補助回帰は定数込み6列（n=6では足りない）だが、ダミーの二乗を除くと5列で
    n=6は足りる。5観測では除外後の列数（k=5）でも足りず、メッセージの`k`も除外後の列数。"""
    x1 = [0.5, 1.2, -0.3, 2.1, 0.9, -1.4]
    d = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    y = [1.1, 2.9, 0.4, 5.2, 3.3, -0.7]
    df = pl.DataFrame({"y": y, "x1": x1, "d": d})

    ok = OLS(df, y="y", x=["x1", "d"]).fit().white_test("f")
    assert ok.df == 4
    assert ok.df_denom == 1

    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_OBSERVATIONS_AUX_REGRESSION, n=5, k=5),
    ):
        OLS(df.head(5), y="y", x=["x1", "d"]).fit().white_test()


def test_result_without_training_data_raises_validation_error():
    """`IVResults.first_stage()`が返す`OLSResults`は単一のソースDataFrameを持たず、
    説明変数を再抽出できない（`augment(new_data=None)`と同じ）。"""
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
        ValidationError, match=escaped(msgs.WHITE_NO_TRAINING_DATA)
    ):
        first_stage.white_test()


def test_all_constant_regressors_raise_computation_error():
    df, _ = _synthetic("baseline")
    df = df.with_columns(pl.lit(3.0).alias("c"))
    options = OLSOptions(include_intercept=False)
    fitted = OLS(df, y="y", x=["c"], options=options).fit()

    with pytest.raises(ComputationError, match="White test"):
        fitted.white_test()


def test_full_set_of_dummies_without_intercept_raises_computation_error():
    """全カテゴリのダミー（和が定数列）を`include_intercept=False`で入れたモデルは
    `fit()`は通るが、補助回帰の定数と共線になる。重複・定数列の除外では取れないため
    `ComputationError`にする（仕様上の制限、`ols-spec.md`「White検定」参照）。"""
    rng = np.random.default_rng(5)
    n = 90
    group = np.arange(n) % 3
    df = pl.DataFrame(
        {
            "y": rng.normal(size=n) + group,
            "d1": (group == 0).astype(float),
            "d2": (group == 1).astype(float),
            "d3": (group == 2).astype(float),
        }
    )
    options = OLSOptions(include_intercept=False)
    fitted = OLS(df, y="y", x=["d1", "d2", "d3"], options=options).fit()

    with pytest.raises(
        ComputationError, match=escaped(msgs.WHITE_AUX_REGRESSION_PREFIX)
    ):
        fitted.white_test()


# ── 主リファレンス（statsmodels `het_white`）との数値照合 ───────────────


def _check_against(fitted, ref: dict, label: str) -> None:
    lm = fitted.white_test("lm")
    f = fitted.white_test("f")

    _ref_close(lm.statistic, ref["lm"], f"{label}/lm")
    _ref_close_p(lm.p_value, ref["lm_p_value"], f"{label}/lm_p_value")
    _ref_close(f.statistic, ref["f"], f"{label}/f")
    _ref_close_p(f.p_value, ref["f_p_value"], f"{label}/f_p_value")
    assert fitted.n_obs == ref["n_obs"], f"{label}/n_obs"


@pytest.mark.parametrize("scenario", WHITE_SYNTHETIC_SCENARIOS)
def test_synthetic_matches_statsmodels(reference, scenario):
    df, x_cols = _synthetic(scenario)
    fitted = OLS(df, y="y", x=x_cols).fit()

    _check_against(fitted, reference["synthetic"][scenario], scenario)
    assert fitted.white_test().df == _expected_df(len(x_cols))


def test_no_intercept_matches_statsmodels(reference):
    """`include_intercept=False`のモデルの残差でも、補助回帰に定数を含めた
    statsmodelsの`het_white`と一致する。"""
    df, x_cols = _synthetic("baseline")
    options = OLSOptions(include_intercept=False)
    fitted = OLS(df, y="y", x=x_cols, options=options).fit()

    _check_against(
        fitted, reference["synthetic"]["baseline_no_intercept"], "no_intercept"
    )


@pytest.mark.parametrize("case", list(WHITE_WOOLDRIDGE_CASES))
def test_wooldridge_matches_statsmodels(reference, load_wooldridge, case):
    """`het_white`は重複列を除かないが補助回帰のランクで自由度を数えるため、
    ダミー・多項式・排他的ダミーのように補助回帰の列が重複・定数になるケース
    （`WOOLDRIDGE_EXPECTED_DROPPED`）でも、重複列を除く本実装と一致する。"""
    dataset, formula = WHITE_WOOLDRIDGE_CASES[case]
    y, x_cols = _parse_formula(formula)
    fitted = OLS(load_wooldridge(dataset), y=y, x=x_cols).fit()

    _check_against(fitted, reference["wooldridge"][case], case)
    assert (
        fitted.white_test().dropped_terms == WOOLDRIDGE_EXPECTED_DROPPED[case]
    )


def test_smallest_sample_matches_statsmodels():
    """観測数が補助回帰の列数（定数込み）より1つだけ多い境界（`n = k + 1`、
    `df_denom = 1`）の成功パスを、ライブのstatsmodelsと照合する。"""
    rng = np.random.default_rng(11)
    n = 7  # 2変数の補助回帰は定数込み6列
    x1, x2 = rng.normal(size=(2, n))
    y = 1.0 + x1 - 0.5 * x2 + rng.normal(size=n)
    fitted = OLS(
        pl.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y", x=["x1", "x2"]
    ).fit()

    lm, f = fitted.white_test("lm"), fitted.white_test("f")
    assert lm.df == 5
    assert f.df_denom == 1

    exog = sm.add_constant(np.column_stack([x1, x2]))
    resid = sm.OLS(y, exog).fit().resid
    sm_lm, sm_lm_p, sm_f, sm_f_p = het_white(resid, exog)
    _ref_close(lm.statistic, sm_lm, "n=k+1/lm")
    _ref_close_p(lm.p_value, sm_lm_p, "n=k+1/lm_p_value")
    _ref_close(f.statistic, sm_f, "n=k+1/f")
    _ref_close_p(f.p_value, sm_f_p, "n=k+1/f_p_value")


# ── 独立実装（R）とのクロスチェック ─────────────────────────────────


def _check_against_r(fitted, ref: dict, label: str) -> None:
    lm = fitted.white_test("lm")
    f = fitted.white_test("f")

    _cc_close(lm.statistic, ref["lm"], f"{label}/lm")
    _cc_close_p(lm.p_value, ref["lm_p_value"], f"{label}/lm_p_value")
    _cc_close(f.statistic, ref["f"], f"{label}/f")
    _cc_close_p(f.p_value, ref["f_p_value"], f"{label}/f_p_value")
    # 自由度: Rのbptestのparameter（ランクに基づく）と補助回帰のlmのF検定の自由度。
    assert lm.df == ref["df"], f"{label}/df"
    assert f.df == ref["f_df_num"], f"{label}/f_df_num"
    assert f.df_denom == ref["f_df_denom"], f"{label}/f_df_denom"
    assert fitted.n_obs == ref["n_obs"], f"{label}/n_obs"


@pytest.mark.parametrize("scenario", WHITE_SYNTHETIC_SCENARIOS)
def test_synthetic_matches_r(crosscheck, scenario):
    df, x_cols = _synthetic(scenario)
    fitted = OLS(df, y="y", x=x_cols).fit()
    _check_against_r(fitted, crosscheck["synthetic"][scenario], scenario)


def test_no_intercept_matches_r(crosscheck):
    df, x_cols = _synthetic("baseline")
    options = OLSOptions(include_intercept=False)
    fitted = OLS(df, y="y", x=x_cols, options=options).fit()
    _check_against_r(
        fitted,
        crosscheck["synthetic"]["baseline_no_intercept"],
        "no_intercept",
    )


@pytest.mark.parametrize("case", list(WHITE_WOOLDRIDGE_CASES))
def test_wooldridge_matches_r(crosscheck, load_wooldridge, case):
    """ダミー・多項式・排他的ダミーでは補助回帰の列が重複・定数になる。Rの`lm`は
    重複をエイリアスとして扱いランクに基づく自由度を使う。本実装が重複・定数列を
    除いて数えた自由度（`aux_terms`の定数以外の数）がそれと一致する。"""
    dataset, formula = WHITE_WOOLDRIDGE_CASES[case]
    y, x_cols = _parse_formula(formula)
    fitted = OLS(load_wooldridge(dataset), y=y, x=x_cols).fit()

    _check_against_r(fitted, crosscheck["wooldridge"][case], case)
    assert (
        fitted.white_test().dropped_terms == WOOLDRIDGE_EXPECTED_DROPPED[case]
    )
