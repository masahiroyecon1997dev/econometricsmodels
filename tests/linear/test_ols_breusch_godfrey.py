"""OLS の Breusch-Godfrey 検定（`OLSResults.breusch_godfrey_test()`）のテスト。

`test_ols_white.py`と同じ構成で、診断検定1つを1ファイルにまとめる:

- 構造・オプション反映（`statistic`の選択・`nlags`・時間列による並べ替え・
  `include_intercept=False`・`cov_type`非依存・時間列のdtype）
- `ValidationError`/`ComputationError`/`TypeError`パス
- 主リファレンス（statsmodels `acorr_breusch_godfrey`、`ols_breusch_godfrey.json`）との数値照合
- 独立実装（R `lmtest::bgtest`、`ols_breusch_godfrey_crosscheck.json`）とのクロスチェック

サンプル前期間のラグは0で埋める（statsmodels・R `bgtest`の既定と同じ）。補助回帰は元のモデルの
説明変数をそのまま使い、`include_intercept=False`でも定数を足さない（R・Greeneの定義）。
statsmodelsは切片なしのモデルでだけ補助回帰に定数を足すため、切片なしはRのみで照合する。
合成データには時間列が無いので、行番号の列（`ROW_TIME`）を時間列として足す。
"""

from __future__ import annotations

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
    IV,
    OLS,
    BreuschGodfreyTestResult,
    DiagnosticResult,
    IVOptions,
    OLSOptions,
    ValidationError,
)
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

from benchmark.linear.constants import (
    BG_NLAGS_BOUNDARY,
    BG_NO_INTERCEPT_SCENARIOS,
    BG_SYNTHETIC_SCENARIOS,
    BG_WOOLDRIDGE_CASES,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"

REF_RTOL = TOLERANCES["ols_breusch_godfrey_reference"]["rtol"]
REF_ATOL = TOLERANCES["ols_breusch_godfrey_reference"]["atol"]
REF_ATOL_P = TOLERANCES["ols_breusch_godfrey_reference"]["atol_p_value"]
CC_RTOL = TOLERANCES["ols_breusch_godfrey_crosscheck"]["rtol_strict"]
CC_ATOL = TOLERANCES["ols_breusch_godfrey_crosscheck"]["atol"]
CC_ATOL_P = TOLERANCES["ols_breusch_godfrey_crosscheck"]["atol_p_value"]

# 統計量は絶対誤差フロアを併用し、p値は相対誤差のみで比較する（裾のp値を検証するため）。
_ref_close = partial(assert_close, rtol=REF_RTOL, atol=REF_ATOL)
_ref_close_p = partial(assert_close, rtol=REF_RTOL, atol=REF_ATOL_P)
_cc_close = partial(assert_close, rtol=CC_RTOL, atol=CC_ATOL)
_cc_close_p = partial(assert_close, rtol=CC_RTOL, atol=CC_ATOL_P)


def _synthetic(scenario: str) -> tuple[pl.DataFrame, list[str]]:
    """合成データに時間列`ROW_TIME`（行番号）を足して返す。"""
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    return with_row_time(df), [
        c for c in df.columns if c not in ("y", "weight")
    ]


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    lhs, rhs = formula.split("~")
    return lhs.strip(), [t.strip() for t in rhs.split("+")]


def _ar1_dataset(
    n: int = 120, rho: float = 0.6, seed: int = 0
) -> pl.DataFrame:
    """AR(1)誤差を持つデータ（時間順）。時間列は`t`。"""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(n):
        u[t] = rho * (u[t - 1] if t else 0.0) + e[t]
    x = rng.normal(size=n)
    return pl.DataFrame({"y": 1.0 + x + u, "x": x, "t": np.arange(n)})


@pytest.fixture(scope="module")
def reference() -> dict:
    path = FIXTURES_DIR / "ols_breusch_godfrey.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def crosscheck() -> dict:
    path = FIXTURES_DIR / "ols_breusch_godfrey_crosscheck.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def load_wooldridge():
    return wooldridge_loader()


@pytest.fixture
def ar1() -> pl.DataFrame:
    return _ar1_dataset()


# ── 構造・オプション反映 ────────────────────────────────────────────


def test_default_returns_lm_version_with_chi2_distribution(ar1):
    res = OLS(ar1, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)

    assert isinstance(res, BreuschGodfreyTestResult)
    assert isinstance(res, DiagnosticResult)
    assert res.distribution == "chi2"
    assert res.df_denom is None
    assert res.df == res.nlags == 2
    assert 0.0 <= res.p_value <= 1.0


def test_f_version_has_f_distribution_and_denominator_degrees_of_freedom(ar1):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    res = fitted.breusch_godfrey_test("t", 3, "f")

    assert res.distribution == "f"
    assert res.df == res.nlags == 3
    # k = 切片 + x = 2
    assert res.df_denom == fitted.n_obs - 2 - 3


def test_detects_serial_correlation_in_ar1_errors(ar1):
    """AR(1)誤差のデータでは帰無仮説（系列相関なし）を強く棄却する。"""
    res = OLS(ar1, y="y", x=["x"]).fit().breusch_godfrey_test("t", 1)
    assert res.p_value < 1e-3


def test_statistic_argument_is_case_insensitive(ar1):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    assert fitted.breusch_godfrey_test("t", 2, "LM") == (
        fitted.breusch_godfrey_test("t", 2, "lm")
    )
    assert fitted.breusch_godfrey_test("t", 2, "F") == (
        fitted.breusch_godfrey_test("t", 2, "f")
    )


def test_rows_are_ordered_by_the_time_column_not_by_row_position(ar1):
    """行を並べ替えても、時間列で並べ直した結果と一致する。時間列を無視して行順のまま
    扱った場合（時間列を行番号にした場合）とは値が変わる。"""
    ordered = OLS(ar1, y="y", x=["x"]).fit().breusch_godfrey_test("t", 3)
    shuffled = ar1.sample(fraction=1.0, shuffle=True, seed=7)

    res = OLS(shuffled, y="y", x=["x"]).fit().breusch_godfrey_test("t", 3)
    # 行を並べ替えると和の順序が変わり最後の桁だけ変わりうるため、機械精度で比較する。
    assert (res.df, res.df_denom, res.distribution, res.nlags) == (
        ordered.df,
        ordered.df_denom,
        ordered.distribution,
        ordered.nlags,
    )
    _cc_close(res.statistic, ordered.statistic, "statistic")
    _cc_close_p(res.p_value, ordered.p_value, "p_value")

    by_row = with_row_time(shuffled)
    ignored = (
        OLS(by_row, y="y", x=["x"]).fit().breusch_godfrey_test(ROW_TIME, 3)
    )
    assert abs(ignored.statistic - ordered.statistic) > 1e-6


@pytest.mark.parametrize(
    "time_expr",
    [
        pl.col("t").cast(pl.Float64),
        pl.col("t").cast(pl.Int32).cast(pl.Date),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("t").cast(pl.Datetime("ms")),
    ],
    ids=["float", "date", "datetime_us", "datetime_ms"],
)
def test_time_column_dtypes_give_the_same_order(ar1, time_expr):
    base = OLS(ar1, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)
    df = ar1.with_columns(time_expr.alias("t"))
    res = OLS(df, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)
    assert res == base


def test_only_the_order_of_the_time_values_matters(ar1):
    """時間列は値の大小だけが使われる（等間隔でなくてもよい）。同じ順序を保つ変換
    （非等間隔な単調増加）では結果が変わらず、順序を逆にすると変わる。"""
    base = OLS(ar1, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)

    monotone = ar1.with_columns((pl.col("t") ** 2 + 5).alias("t"))
    assert (
        OLS(monotone, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)
        == base
    )

    reversed_time = ar1.with_columns((-pl.col("t")).alias("t"))
    res = OLS(reversed_time, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)
    assert res.statistic != base.statistic


def test_time_column_may_be_unrelated_to_the_regressors(ar1):
    """時間列は説明変数でなくてよく、データに他の列があっても結果は変わらない。"""
    with_extra = ar1.with_columns(pl.lit("a").alias("note"))
    base = OLS(ar1, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)
    res = OLS(with_extra, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)
    assert res == base


def test_no_constant_is_added_for_a_model_without_an_intercept(ar1):
    """`include_intercept=False`でも補助回帰に定数を足さない（R・Greeneの定義）。
    定数を足すstatsmodelsの値とは異なる（Rとの一致は`test_*_matches_r`）。"""
    options = OLSOptions(include_intercept=False)
    fitted = OLS(ar1, y="y", x=["x"], options=options).fit()
    res = fitted.breusch_godfrey_test("t", 2)

    resid = np.asarray(fitted.residuals)
    x = ar1["x"].to_numpy()
    with_constant = sm.OLS(
        resid,
        np.column_stack(
            [
                np.ones(len(resid)),
                x,
                np.concatenate([[0.0], resid[:-1]]),
                np.concatenate([[0.0, 0.0], resid[:-2]]),
            ]
        ),
    ).fit()
    statsmodels_like = len(resid) * with_constant.rsquared
    assert abs(res.statistic - statsmodels_like) > 1e-6


@pytest.mark.parametrize(
    "cov_type", ["hc0", "hc1", "hc2", "hc3", "hac", "cluster"]
)
def test_result_does_not_depend_on_any_cov_type(ar1, cov_type):
    """`hac`（時間列）・`cluster`（クラスター列）のように追加の列引数を持つ`cov_type`でも、
    説明変数・時間列の再抽出が崩れず結果が変わらない。"""
    df = with_cluster_groups(with_row_time(ar1), 10)
    kwargs = {
        "hac": {"hac_lags": 1, "hac_time": ROW_TIME},
        "cluster": {"cluster": "cluster_group"},
    }.get(cov_type, {})

    classical = OLS(df, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)
    res = (
        OLS(
            df,
            y="y",
            x=["x"],
            options=OLSOptions(cov_type=cov_type, **kwargs),
        )
        .fit()
        .breusch_godfrey_test("t", 2)
    )
    assert res == classical


def test_result_is_invariant_to_the_scale_of_the_regressors_and_of_y(ar1):
    base = OLS(ar1, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)
    scaled = ar1.with_columns(
        (pl.col("x") * 1.0e5 + 2.0e3).alias("x"),
        (pl.col("y") * 1.0e3).alias("y"),
    )
    res = OLS(scaled, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2)

    assert res.df == base.df
    _cc_close(res.statistic, base.statistic, "statistic")
    assert res.p_value == pytest.approx(base.p_value, rel=1e-6)


def test_to_dict_is_json_ready(ar1):
    res = OLS(ar1, y="y", x=["x"]).fit().breusch_godfrey_test("t", 2, "f")
    as_dict = res.to_dict()

    assert set(as_dict) == {
        "statistic",
        "p_value",
        "df",
        "df_denom",
        "distribution",
        "nlags",
    }
    assert json.loads(json.dumps(as_dict)) == as_dict


def test_breusch_godfrey_test_is_not_computed_by_fit(ar1):
    """事後診断は`fit()`では計算しない。"""
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    assert [n for n in dir(fitted) if "godfrey" in n] == [
        "breusch_godfrey_test"
    ]


# ── ValidationError / ComputationError / TypeError ──────────────────


@pytest.mark.parametrize("nlags", [0, -1, -(10**30)])
def test_nlags_below_one_raises_validation_error(ar1, nlags):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    # 巨大な負の整数は`i64::MIN`に飽和して報告される。
    expected = max(nlags, -(2**63))
    with pytest.raises(
        ValidationError, match=escaped(msgs.INVALID_NLAGS, nlags=expected)
    ):
        fitted.breusch_godfrey_test("t", nlags)


@pytest.mark.parametrize(
    "nlags", [True, 2.0, "2", None], ids=["bool", "float", "str", "none"]
)
def test_nlags_that_is_not_an_int_raises_type_error(ar1, nlags):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    with pytest.raises(TypeError, match="nlags"):
        fitted.breusch_godfrey_test("t", nlags)


@pytest.mark.parametrize("time", [1, None, ["t"]])
def test_time_that_is_not_a_str_raises_type_error(ar1, time):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    with pytest.raises(TypeError, match="time"):
        fitted.breusch_godfrey_test(time, 1)


def test_statistic_that_is_not_a_str_raises_type_error(ar1):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    with pytest.raises(TypeError, match="statistic"):
        fitted.breusch_godfrey_test("t", 1, 1)


def test_time_and_nlags_are_required(ar1):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    with pytest.raises(TypeError):
        fitted.breusch_godfrey_test()
    with pytest.raises(TypeError):
        fitted.breusch_godfrey_test("t")


def test_unknown_statistic_raises_validation_error(ar1):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_DIAGNOSTIC_STATISTIC, other="chi2"),
    ):
        fitted.breusch_godfrey_test("t", 1, "chi2")


def test_missing_time_column_raises_validation_error(ar1):
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nope"),
    ):
        fitted.breusch_godfrey_test("nope", 1)


def test_time_column_with_ties_raises_validation_error(ar1):
    """同値があると時間順が定まらない（`hac_time`と同じ扱い）。行順で黙って並べない。"""
    df = ar1.with_columns((pl.col("t") // 2).alias("t"))
    fitted = OLS(df, y="y", x=["x"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_TIED_TIME_ORDER, name="t", first=0, second=1
        ),
    ):
        fitted.breusch_godfrey_test("t", 1)


@pytest.mark.parametrize(
    "dtype", [pl.Int64, pl.Float64, pl.Date, pl.Datetime("us")], ids=str
)
def test_time_column_with_missing_values_raises_validation_error(ar1, dtype):
    """null は dtype ごとに別の抽出経路を通る（整数・`Date`・`Datetime`は物理表現、
    浮動小数は値として）。いずれも`ValidationError`。"""
    t = ar1["t"].cast(pl.Int32).cast(dtype).to_list()
    t[5] = None
    df = ar1.with_columns(pl.Series("t", t, dtype=dtype))
    fitted = OLS(df, y="y", x=["x"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="t", count=1),
    ):
        fitted.breusch_godfrey_test("t", 1)


@pytest.mark.parametrize(
    ("bad", "shown"),
    [(float("nan"), "NaN"), (float("inf"), "inf"), (float("-inf"), "-inf")],
)
def test_float_time_column_with_non_finite_values_raises_validation_error(
    ar1, bad, shown
):
    """NaN・無限大は null とは別の検査（`hac_time`と同じ）。順序が定まらないため拒否する。"""
    t = ar1["t"].cast(pl.Float64).to_list()
    t[3] = bad
    df = ar1.with_columns(pl.Series("t", t, dtype=pl.Float64))
    fitted = OLS(df, y="y", x=["x"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="t", value=shown, row=3
        ),
    ):
        fitted.breusch_godfrey_test("t", 1)


def test_time_column_may_also_be_a_regressor(ar1):
    """時間列を説明変数にも入れたトレンド回帰（実務で典型的）。説明変数と時間列を同じ
    列から再抽出しても、statsmodelsと一致する。"""
    fitted = OLS(ar1, y="y", x=["x", "t"]).fit()
    res = fitted.breusch_godfrey_test("t", 3)
    f_res = fitted.breusch_godfrey_test("t", 3, "f")

    exog = sm.add_constant(
        np.column_stack([ar1["x"].to_numpy(), ar1["t"].to_numpy()])
    )
    sm_res = sm.OLS(ar1["y"].to_numpy(), exog).fit()
    sm_lm, sm_lm_p, sm_f, sm_f_p = acorr_breusch_godfrey(
        sm_res, nlags=3, result_object=False
    )
    _ref_close(res.statistic, sm_lm, "trend/lm")
    _ref_close_p(res.p_value, sm_lm_p, "trend/lm_p_value")
    _ref_close(f_res.statistic, sm_f, "trend/f")
    _ref_close_p(f_res.p_value, sm_f_p, "trend/f_p_value")


def test_time_column_with_an_unsupported_dtype_raises_validation_error(ar1):
    df = ar1.with_columns(pl.col("t").cast(pl.String).alias("label"))
    fitted = OLS(df, y="y", x=["x"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_ORDER_DTYPE, name="label", dtype="String"
        ),
    ):
        fitted.breusch_godfrey_test("label", 1)


def test_observation_count_at_the_boundary():
    """補助回帰の列数は`k + nlags`（n=10、k=2、nlags=7なら9列でn=10>9で成功し
    `df_denom = 1`）。nlags=8は`n = k + nlags`で拒否される。成功側はstatsmodelsと照合する。"""
    df = _ar1_dataset(n=10, seed=2)
    fitted = OLS(df, y="y", x=["x"]).fit()

    ok = fitted.breusch_godfrey_test("t", 7, "f")
    assert ok.df == 7
    assert ok.df_denom == 1

    exog = sm.add_constant(df["x"].to_numpy())
    sm_res = sm.OLS(df["y"].to_numpy(), exog).fit()
    sm_lm, sm_lm_p, sm_f, sm_f_p = acorr_breusch_godfrey(
        sm_res, nlags=7, result_object=False
    )
    _ref_close(ok.statistic, sm_f, "n=k+m+1/f")
    _ref_close_p(ok.p_value, sm_f_p, "n=k+m+1/f_p_value")
    lm = fitted.breusch_godfrey_test("t", 7)
    _ref_close(lm.statistic, sm_lm, "n=k+m+1/lm")
    _ref_close_p(lm.p_value, sm_lm_p, "n=k+m+1/lm_p_value")

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_OBSERVATIONS_AUX_REGRESSION, n=10, k=10
        ),
    ):
        fitted.breusch_godfrey_test("t", 8)


def test_huge_nlags_is_reported_as_too_few_observations(ar1):
    """巨大な`nlags`でもオーバーフローせず観測数不足として報告される。"""
    fitted = OLS(ar1, y="y", x=["x"]).fit()
    with pytest.raises(ValidationError, match="insufficient observations"):
        fitted.breusch_godfrey_test("t", 2**62)


def test_result_without_training_data_raises_validation_error():
    """`IVResults.first_stage()`が返す`OLSResults`は単一のソースDataFrameを持たず、
    説明変数・時間列を再抽出できない（`augment(new_data=None)`・`white_test()`と同じ）。"""
    rng = np.random.default_rng(3)
    n = 120
    z1, z2, x1 = rng.normal(size=(3, n))
    endog = 0.7 * z1 + 0.4 * z2 + 0.3 * x1 + rng.normal(size=n)
    y = 1.0 + 0.5 * endog + 0.2 * x1 + rng.normal(size=n)
    df = pl.DataFrame(
        {"y": y, "endog1": endog, "x1": x1, "z1": z1, "z2": z2}
    ).with_columns(pl.int_range(pl.len()).alias("t"))
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
        ValidationError, match=escaped(msgs.BG_NO_TRAINING_DATA)
    ):
        first_stage.breusch_godfrey_test("t", 1)


# `ComputationError`パス: 補助回帰の特異・残差が全て0・補助回帰が残差を完全に説明する、は
# Python経由では到達できない（残差が全て0・完全適合になる入力は、元の`fit()`が
# 分散共分散行列の特異で先に`ComputationError`にする。補助回帰が特異になる条件は、
# 説明変数が残差のラグそのものという、`fit()`の結果に依存する構成のため合成できない）。
# これらはRustの単体テスト（`engine/src/linear/diagnostics.rs`の`mod tests`:
# `breusch_godfrey_reports_a_singular_auxiliary_regression_with_its_context`・
# `breusch_godfrey_fails_for_all_zero_or_non_finite_residuals`・
# `breusch_godfrey_fails_when_the_auxiliary_regression_fits_the_residuals_exactly`）で担保する。


# ── 主リファレンス（statsmodels `acorr_breusch_godfrey`）との数値照合 ───


def _check_against(fitted, time: str, ref: dict, label: str) -> None:
    for m_key, expected in ref["nlags"].items():
        m = int(m_key)
        lm = fitted.breusch_godfrey_test(time, m, "lm")
        f = fitted.breusch_godfrey_test(time, m, "f")

        _ref_close(lm.statistic, expected["lm"], f"{label}/m={m}/lm")
        _ref_close_p(
            lm.p_value, expected["lm_p_value"], f"{label}/m={m}/lm_p_value"
        )
        _ref_close(f.statistic, expected["f"], f"{label}/m={m}/f")
        _ref_close_p(
            f.p_value, expected["f_p_value"], f"{label}/m={m}/f_p_value"
        )
        assert lm.df == f.df == lm.nlags == m
    assert fitted.n_obs == ref["n_obs"], f"{label}/n_obs"


@pytest.mark.parametrize("scenario", BG_SYNTHETIC_SCENARIOS)
def test_synthetic_matches_statsmodels(reference, scenario):
    df, x_cols = _synthetic(scenario)
    fitted = OLS(df, y="y", x=x_cols).fit()
    _check_against(
        fitted, ROW_TIME, reference["synthetic"][scenario], scenario
    )


@pytest.mark.parametrize("case", list(BG_WOOLDRIDGE_CASES))
def test_wooldridge_matches_statsmodels(reference, load_wooldridge, case):
    """時間列で並べ替えたデータ（行を無作為に並べ替えて時間列を渡す）でも、時間順に
    並べたstatsmodelsと一致する。"""
    dataset, formula, time = BG_WOOLDRIDGE_CASES[case]
    y, x_cols = _parse_formula(formula)
    df = load_wooldridge(dataset).sample(fraction=1.0, shuffle=True, seed=3)
    fitted = OLS(df, y=y, x=x_cols).fit()
    _check_against(fitted, time, reference["wooldridge"][case], case)


# ── 独立実装（R）とのクロスチェック ─────────────────────────────────


def _check_against_r(fitted, time: str, ref: dict, label: str) -> None:
    for m_key, expected in ref["nlags"].items():
        m = int(m_key)
        lm = fitted.breusch_godfrey_test(time, m, "lm")
        f = fitted.breusch_godfrey_test(time, m, "f")

        _cc_close(lm.statistic, expected["lm"], f"{label}/m={m}/lm")
        _cc_close_p(
            lm.p_value, expected["lm_p_value"], f"{label}/m={m}/lm_p_value"
        )
        _cc_close(f.statistic, expected["f"], f"{label}/m={m}/f")
        _cc_close_p(
            f.p_value, expected["f_p_value"], f"{label}/m={m}/f_p_value"
        )
        assert lm.df == expected["df"], f"{label}/m={m}/df"
        assert f.df == expected["f_df_num"], f"{label}/m={m}/f_df_num"
        assert f.df_denom == expected["f_df_denom"], (
            f"{label}/m={m}/f_df_denom"
        )
    assert fitted.n_obs == ref["n_obs"], f"{label}/n_obs"


@pytest.mark.parametrize("scenario", BG_SYNTHETIC_SCENARIOS)
def test_synthetic_matches_r(crosscheck, scenario):
    df, x_cols = _synthetic(scenario)
    fitted = OLS(df, y="y", x=x_cols).fit()
    _check_against_r(
        fitted, ROW_TIME, crosscheck["synthetic"][scenario], scenario
    )


@pytest.mark.parametrize("scenario", BG_NO_INTERCEPT_SCENARIOS)
def test_no_intercept_matches_r(crosscheck, scenario):
    """切片なしのモデルは補助回帰に定数を足さない（R `bgtest`・Greeneの定義）。"""
    df, x_cols = _synthetic(scenario)
    options = OLSOptions(include_intercept=False)
    fitted = OLS(df, y="y", x=x_cols, options=options).fit()
    _check_against_r(
        fitted,
        ROW_TIME,
        crosscheck["synthetic"][f"{scenario}_no_intercept"],
        f"{scenario}_no_intercept",
    )


@pytest.mark.parametrize("case", list(BG_WOOLDRIDGE_CASES))
def test_wooldridge_matches_r(crosscheck, load_wooldridge, case):
    dataset, formula, time = BG_WOOLDRIDGE_CASES[case]
    y, x_cols = _parse_formula(formula)
    df = load_wooldridge(dataset).sample(fraction=1.0, shuffle=True, seed=3)
    fitted = OLS(df, y=y, x=x_cols).fit()
    _check_against_r(fitted, time, crosscheck["wooldridge"][case], case)


def test_boundary_nlags_is_covered_by_both_fixtures(reference, crosscheck):
    """`n = k + nlags + 1`（F検定の`df_denom = 1`）の成功パスが、凍結フィクスチャとして
    statsmodels・Rの両方に入っている（`BG_NLAGS_BOUNDARY`）。"""
    for scenario, m in BG_NLAGS_BOUNDARY.items():
        assert str(m) in reference["synthetic"][scenario]["nlags"]
        r_entry = crosscheck["synthetic"][scenario]["nlags"][str(m)]
        assert r_entry["f_df_denom"] == 1
