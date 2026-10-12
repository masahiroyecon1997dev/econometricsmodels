"""Tobit の `ValidationError`/`ComputationError` パスの検証。

入力データ・`predict()`の`new_data`・`augment()`・オプション・
`marginal_effects()`の不正値が想定した例外クラスとメッセージで弾かれること、
完全な多重共線性・準分離・未収束といった計算過程の問題が `ComputationError` に
なる（またはならない）ことを確認する。成功パスの構造・API・オプション反映は
`test_tobit_api.py`、主リファレンス（R `AER::tobit`）との数値照合は
`test_tobit_reference.py`、R クロスチェック（`censReg`）は
`test_tobit_crosscheck.py`（`test_logit_validation.py` と同じ分割）。
"""

from __future__ import annotations

import json
import math

import _error_messages as msgs
import numpy as np
import pandas as pd
import polars as pl
import pytest
from _constants import DATA_DIR
from _error_messages import escaped, rust_f64
from _helpers import (
    censored_separation_dataset,
    heavily_censored_tobit_dataset,
    name_collision_frame,
    with_cluster_groups,
)
from econometricsmodels import (
    ComputationError,
    Tobit,
    TobitOptions,
    ValidationError,
)

from benchmark.nonlinear.constants import MLE_COV_TYPES, NON_CLUSTER_COV_TYPES

_TOBIT_CENSORING_BOUNDS = json.loads(
    (DATA_DIR / "tobit_censoring_bounds.json").read_text()
)

# censored_datasetフィクスチャ（`dataset`のyを0.0で左打ち切り、打ち切り率21%）は
# conftest.pyで定義。


# ── ValidationError（入力データ） ─────────────────────────────────────


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
    df = name_collision_frame([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(
        ValidationError, match=escaped(msgs.CONST_COLLISION, role="x")
    ):
        Tobit(df, y="y", x=["const"]).fit()


def test_sigma_collision_raises():
    """`x`に`"sigma"`という列名があると、`TobitResult`が`param_names`の末尾に
    追加する合成パラメータ名`"sigma"`と衝突する（`"const"`列衝突と同型、
    `engine_pybind`の`validate_no_sigma_collision`のPython API境界での確認）。
    """
    df = name_collision_frame([0.0, 1.0, 0.0, 1.0], "sigma")
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


def test_data_not_polars_raises(censored_dataset):
    """`data`/`new_data`にpolars以外のDataFrame（pandas等）を渡すと、内部実装
    （`pyo3-polars`の`get_columns`呼び出し）が漏れた`AttributeError`ではなく
    `ValidationError`になること（`test_ols_validation.py`と同じ検証）。
    """
    bad = pd.DataFrame({"y": [0.0, 1.0, 2.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="data",
            type_name=msgs.fully_qualified_type_name(bad),
        ),
    ):
        Tobit(bad, y="y", x=["x1"]).fit()

    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    bad_new_data = pd.DataFrame({"x1": [1.0], "x2": [0.5]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="new_data",
            type_name=msgs.fully_qualified_type_name(bad_new_data),
        ),
    ):
        res.predict(new_data=bad_new_data)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="new_data",
            type_name=msgs.fully_qualified_type_name(bad_new_data),
        ),
    ):
        res.augment(new_data=bad_new_data)


def test_null_values_raise():
    df = pl.DataFrame({"y": [0.0, None, 1.0], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="y", count=1),
    ):
        Tobit(df, y="y", x=["x1"]).fit()


def test_null_values_in_x_raise():
    """`x` 列に null が含まれる場合も `ValidationError`（`y` だけでなく `x` も
    欠損チェックの対象）。"""
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
    `test_non_finite_values_raise` と同じ）。
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
    """文字列列は、dtypeの時点で`ValidationError`
    （`test_ols_validation.py::test_non_numeric_dtype_raises`参照）。
    """
    df = pl.DataFrame({"y": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="y", dtype="String"
        ),
    ):
        Tobit(df, y="y", x=["x1"]).fit()


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
    （engine側の`NoUncensoredObservations`）。
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


# ── ValidationError（predict()のnew_data） ────────────────────────────


def test_predict_unknown_target_raises(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_MARGINAL_EFFECTS_TARGET, other="bogus"),
    ):
        res.predict(target="bogus")


def test_predict_missing_column_raises(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0]})  # x2が無い

    with pytest.raises(
        ValidationError, match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="x2")
    ):
        res.predict(new_data=new_data)


def test_predict_non_numeric_dtype_raises(censored_dataset):
    """`test_non_numeric_dtype_raises`と同じdtypeのメッセージになる。"""
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": ["a", "b"], "x2": [1.0, 2.0]})

    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_UNSUPPORTED_NUMERIC_DTYPE, name="x1", dtype="String"
        ),
    ):
        res.predict(new_data=new_data)


def test_predict_null_or_non_finite_values_raise(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()

    new_data_null = pl.DataFrame({"x1": [1.0, None], "x2": [1.0, 2.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="x1", count=1),
    ):
        res.predict(new_data=new_data_null)

    new_data_inf = pl.DataFrame({"x1": [1.0, float("inf")], "x2": [1.0, 2.0]})
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE, name="x1", value="inf", row=1
        ),
    ):
        res.predict(new_data=new_data_inf)


# ── ValidationError（augment()） ──────────────────────────────────────


def test_augment_unknown_target_raises(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_MARGINAL_EFFECTS_TARGET, other="bogus"),
    ):
        res.augment(target="bogus")


def test_augment_column_collision_raises(censored_dataset):
    """元データ（`new_data=None`）・`new_data`のいずれかに既に
    `"predicted_expected_observed"`列がある場合`ValidationError`
    （黙って上書きしない）。
    """
    df_with_predicted = censored_dataset.with_columns(
        pl.lit(0.0).alias("predicted_expected_observed")
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.EXISTING_COLUMN_COLLISION, name="predicted_expected_observed"
        ),
    ):
        Tobit(df_with_predicted, y="y", x=["x1", "x2"]).fit().augment()

    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame(
        {"x1": [1.0], "x2": [0.5], "predicted_expected_observed": [0.0]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.EXISTING_COLUMN_COLLISION, name="predicted_expected_observed"
        ),
    ):
        res.augment(new_data=new_data)


def test_augment_missing_column_raises(censored_dataset):
    res = Tobit(censored_dataset, y="y", x=["x1", "x2"]).fit()
    new_data = pl.DataFrame({"x1": [1.0, 2.0]})  # x2が無い

    with pytest.raises(
        ValidationError, match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="x2")
    ):
        res.augment(new_data=new_data)


# ── ValidationError（オプション） ─────────────────────────────────────


@pytest.mark.parametrize("cov_type", ["bogus", ""])
def test_unknown_cov_type_raises(censored_dataset, cov_type):
    """未知の`cov_type`（空文字列を含む）は`ValidationError`。"""
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_COV_TYPE_NONLINEAR, other=cov_type),
    ):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(cov_type=cov_type),
        ).fit()


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_rejected(censored_dataset, cov_type):
    """`"nonrobust"`（旧別名）は受け付けない（概念ごとに文字列を1つに絞る）。"""
    with pytest.raises(ValidationError, match="unknown cov_type: 'nonrobust'"):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(cov_type=cov_type),
        ).fit()


@pytest.mark.parametrize("solver", ["bogus", ""])
def test_unknown_solver_raises(censored_dataset, solver):
    """未知の`solver`（空文字列を含む）は`ValidationError`。"""
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_METHOD_NONLINEAR, other=solver),
    ):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(solver=solver),
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
            options=TobitOptions(cov_type="cluster", cluster="cluster"),
        ).fit()


def test_cluster_count_at_most_slopes_raises_validation_error(
    censored_dataset,
):
    """クラスター数G≤傾き係数の数q（ここで`G=2 == q=2`、x1/x2）は`ValidationError`
    （engine側の`CommonError::InsufficientClustersForInference`）。

    `rank(Ŝ)≤G-1`のため全体Wald検定のq×q部分行列がG≤qで構造的に特異になる。
    GもqもR行列計算なしで即座に判定できるため、`wald_chi2_test`内の
    `ComputationError`ではなく`fit()`冒頭の`ValidationError`で弾く（mroz
    `hours`クラスターケースがこの経路。`G<q`側はOLSの同名テストで確認）。
    """
    df = with_cluster_groups(censored_dataset, 2, col="cluster")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=2, q=2),
    ):
        Tobit(
            df,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(cov_type="cluster", cluster="cluster"),
        ).fit()


def test_mroz_hours_cluster_cov_type_raises_validation_error():
    """実データでの`G <= q`境界の顕在化ケース。

    Wooldridge mroz `hours` Tobit（Wooldridge Example 17.2、RHS 7変数 → q=7）を
    `cluster="city"`（都市部居住ダミー、G=2）で推定すると`G=2 <= q=7`。
    `rank(Ŝ) <= G-1 = 1`のため全体Wald検定の`7×7`部分行列が構造的に特異になり、
    `fit()`冒頭のバリデーションが`ValidationError`
    （`CommonError::InsufficientClustersForInference`）で弾く
    （参照実装Rの`linearHypothesis`相当も同データで計算不能）。
    """
    from _constants import MROZ_X
    from _helpers import load_wooldridge_dataset

    mroz = load_wooldridge_dataset("mroz")
    options = TobitOptions(cov_type="cluster", cluster="city", lower=0.0)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=2, q=len(MROZ_X)
        ),
    ):
        Tobit(mroz, y="hours", x=MROZ_X, options=options).fit()


def test_cluster_without_col_raises(censored_dataset):
    """`cov_type="cluster"`なのに`cluster`未指定の場合`ValidationError`
    （OLS/WLS/IV/Logit/Probitと同じ検証、共通化された経路）。
    """
    options = TobitOptions(cov_type="cluster")
    with pytest.raises(
        ValidationError, match=escaped(msgs.MISSING_CLUSTER_COLUMN)
    ):
        Tobit(censored_dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_cluster_nonexistent_column_raises(censored_dataset):
    options = TobitOptions(cov_type="cluster", cluster="does_not_exist")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="does_not_exist"),
    ):
        Tobit(censored_dataset, y="y", x=["x1", "x2"], options=options).fit()


def test_cluster_with_null_raises(censored_dataset):
    """`cluster` に null（欠損）が含まれる場合 `ValidationError`
    （`extract_group_key_column` の null チェック）。"""
    n = censored_dataset.height
    groups = [None] + [str(i % 5) for i in range(n - 1)]
    df = censored_dataset.with_columns(pl.Series("grp", groups, dtype=pl.Utf8))
    options = TobitOptions(cov_type="cluster", cluster="grp")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name="grp", count=1),
    ):
        Tobit(df, y="y", x=["x1", "x2"], options=options).fit()


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
def test_cluster_unused_by_cov_type_raises(censored_dataset, cov_type):
    """`cov_type="cluster"`以外で`cluster`が指定されたら黙って無視せず
    `ValidationError`。
    """
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.UNUSED_OPTION,
            option="cluster",
            condition='cov_type="cluster"',
        ),
    ):
        Tobit(
            censored_dataset,
            y="y",
            x=["x1", "x2"],
            options=TobitOptions(cov_type=cov_type, cluster="cluster"),
        ).fit()


# ── ValidationError（marginal_effects()） ─────────────────────────────


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


# ── ComputationError ──────────────────────────────────────────────────


def _narrow_interval_dataset(n_interior: int) -> pl.DataFrame:
    """ほぼ全観測が下限0・上限1に打ち切られたデータ（内側は`n_interior`件だけ）。
    `[0, 1]`で両側打ち切りにすると`sigma`が区間幅の100倍近く以上に推定され、
    非打ち切り区間が`sigma`に比べて極端に狭くなる。"""
    n = 2000
    x = [((i * 7919) % 1009) / 1009 for i in range(n)]
    y = [
        0.5 if i < n_interior else (0.0 if i % 2 == 0 else 1.0)
        for i in range(n)
    ]
    return pl.DataFrame({"y": y, "x1": x})


def test_expected_uncensored_raises_computation_error_for_narrow_interval():
    """非打ち切り区間が`sigma`の1%未満（`sigma`が区間幅の100倍超）だと
    `expected_uncensored`は数値的に信頼できない（`W`で割る量が桁落ち）ため、
    `predict`・`augment`・`marginal_effects`が`ComputationError`になる。他の
    3つのtargetは同じ当てはめで通常どおり計算できる（エラーは固有）。
    `new_data`付きの`predict`も同様。"""
    data = _narrow_interval_dataset(n_interior=5)
    res = Tobit(
        data, y="y", x=["x1"], options=TobitOptions(lower=0.0, upper=1.0)
    ).fit()
    assert 1.0 / res.sigma < 0.01
    message = escaped(
        msgs.UNCENSORED_INTERVAL_TOO_NARROW,
        lower=rust_f64(0.0),
        upper=rust_f64(1.0),
        sigma=rust_f64(res.sigma),
    )
    new_data = pl.DataFrame({"x1": [0.2, 0.8]})

    with pytest.raises(ComputationError, match=message):
        res.predict(target="expected_uncensored")
    with pytest.raises(ComputationError, match=message):
        res.predict(target="expected_uncensored", new_data=new_data)
    with pytest.raises(ComputationError, match=message):
        res.augment(target="expected_uncensored")
    with pytest.raises(ComputationError, match=message):
        res.marginal_effects(target="expected_uncensored")

    for target in ["expected_latent", "expected_observed", "prob_uncensored"]:
        assert len(res.predict(target=target)) == data.height
        assert len(res.predict(target=target, new_data=new_data)) == 2
        assert res.augment(target=target).height == data.height
        assert len(res.marginal_effects(target=target)) == 1


def test_expected_uncensored_is_computed_when_the_interval_is_wide_enough():
    """境界の内側（`sigma`が区間幅の100倍未満）では同じ構成でも`expected_uncensored`
    が計算でき、値は非打ち切り区間`[0, 1]`に収まる。"""
    data = _narrow_interval_dataset(n_interior=20)
    res = Tobit(
        data, y="y", x=["x1"], options=TobitOptions(lower=0.0, upper=1.0)
    ).fit()
    assert 1.0 / res.sigma >= 0.01
    predicted = res.predict(target="expected_uncensored")
    assert all(0.0 <= value <= 1.0 for value in predicted)
    assert len(res.marginal_effects(target="expected_uncensored")) == 1


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。
    想定エラー（`ComputationError`、engine 側の `SingularDesignMatrix`）が発生する
    ことのみを確認する。OLS/WLS/Logit と同じく凍結 CSV
    （`tobit_perfect_multicollinearity.csv`、`x3 = 2·x1 + 3·x2`）を使う。

    Logit とは異なり、Tobit は `ols_initial_params` の QR 検証が `solver` に
    関わらず常に最初に実行されるため、完全な多重共線性は常にこの経路で検出される
    （`solver` を parametrize する必要が無い、`docs/spec/tobit-spec.md` 3.2節参照）。
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


@pytest.mark.parametrize("cov_type", NON_CLUSTER_COV_TYPES)
def test_scale_variance_raises_computation_error(cov_type):
    """変数間のスケールが極端に異なる設計行列（x1 を `*1e6`、x2 を `*1e-3`）は、
    傾き係数の同時共分散部分行列がスケール比の 2 乗（≈1e18）相当の条件数を持ち
    倍精度浮動小数点の限界を超えて数値的に特異になる（OLS/WLS と同じ理由・同じ
    凍結 CSV パターン、`test_ols_validation.py` 参照）。

    Tobit の全体 Wald 検定が OLS の F 検定と同型でこの部分行列の反転を要求するため、
    classical を含む全 cov_type で `ComputationError` になる。`scale_variance_mild`
    （スケール比 1e3）が数値リグレッション検知用の成功パス
    （`test_tobit_reference.py`）。数値比較はせずエラーパスのみ確認する
    （`TOBIT_ERROR_PATH_SCENARIOS`）。
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


@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_large_true_coefficient_dgp_converges_and_recovers_truth(solver):
    """大きい真の係数（`x1`の係数=100）でもノイズがあれば識別可能で、真値を回復する
    回帰テスト。

    DGPは`y* = 100·x1 + 0.5·x2 + N(0,1)`（n=200、左打ち切り約52%）。`y`の標準偏差が
    約65と大きく、`y`のスケールだけを根拠にした分離判定だと真の分離と取り違える。
    Newton/BFGS/LBFGSのいずれでも`x1≈100`・`σ≈1`（真値`β1=100`, `σ=1`）で収束する。

    真の（準完全）分離が`ComputationError`になることは
    `test_true_separation_noise_free_dgp_raises_computation_error`で検証する。
    """
    df = censored_separation_dataset(noise_sd=1.0)

    res = Tobit(
        df, y="y", x=["x1", "x2"], options=TobitOptions(solver=solver)
    ).fit()

    # これはリファレンス数値照合ではなく分離誤検知に対する回帰テスト。許容幅は「真値を回復し、
    # かつσ→0退化に倒れていない」ことだけを担保する緩いバンド。実測は（既定solverで）
    # x1≈99.91・x2≈0.52・const≈0.15・σ≈0.94だが、将来の
    # ソルバー変更でのメソッド間変動を吸収するためマージンを広く取る。σの下限0.5は
    # 「(準)分離で σ→0 へ退化していない」ことの実質的なガード。
    assert res.converged
    assert abs(res.params["x1"] - 100.0) < 2.0
    assert abs(res.params["x2"] - 0.5) < 0.5
    assert abs(res.params["const"]) < 5.0
    assert 0.5 < res.sigma < 2.0


@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
def test_true_separation_noise_free_dgp_raises_computation_error(solver):
    """ノイズを除いた完全分離DGP（`y* = 100·x1 + 0.5·x2`）は`ComputationError`。

    Tobitの「真の」分離は、Logit/Probitのように係数が±∞へ発散するのではなく
    `σ→0`退化として現れる。実測ではNewtonは`NonConvergence`（`max_iter`到達）、
    BFGS/L-BFGSも`ComputationError`になる（変種は問わない）。`max_iter`を
    35→2000に増やしてもNewton/BFGSは`NonConvergence`のまま。Logit/Probitの分離検出
    （`SeparationDetected`）はTobitでは呼ばない。

    **全件打ち切りとの棲み分け**: 非打ち切り観測が1件も無い（全観測が境界値
    ちょうど）ケースは`fit()`冒頭の`validate_has_uncensored_observations`が
    `ValidationError`（`NoUncensoredObservations`）で先に弾く
    （`test_no_uncensored_observations_raises`）。本ケースは
    非打ち切り観測が存在するため、そのバリデーションは通過し、最適化の
    非収束＝`ComputationError`（`ValueError`系ではない）として現れる。
    """
    df = censored_separation_dataset(noise_sd=0.0)

    with pytest.raises(ComputationError):
        Tobit(
            df, y="y", x=["x1", "x2"], options=TobitOptions(solver=solver)
        ).fit()


def test_quasi_separation_tiny_noise_converges_to_true_values():
    """境界レジーム（軽度の準完全分離＋ごく小さいノイズ）が、
    `raise_on_non_convergence`の値によらず`converged=True`で真値にほぼ完全一致
    することを固定する（`y* = 100·x1 + 0.5·x2 + N(0, 0.001)`に対し
    `x1≈99.9999`, `x2≈0.4998`, `σ≈0.00094`）。

    回帰ガード: `censored_contribution`（`engine/src/nonlinear/tobit.rs`）の
    Hessian項`A(u)=λ(u+λ)`の計算でクランプ済み`λ`と生の`zeta`を混在させると、
    `|zeta|>U_CLAMP`の領域で`A(u)`が負になってHessianの正定値性が崩れ、
    Newtonが`max_iter`まで収束しない（Probitの`ProbitProblem`と同型）。
    打ち切り境界付近の`zeta`は`U_CLAMP`を超えやすいため、この境界レジームが
    その影響を受ける。
    """
    df = censored_separation_dataset(noise_sd=0.001, seed=7)

    res = Tobit(df, y="y", x=["x1", "x2"]).fit()

    assert res.converged is True
    assert abs(res.params["x1"] - 100.0) < 0.01
    assert abs(res.params["x2"] - 0.5) < 0.01
    assert abs(res.sigma - 0.001) < 0.001


def test_many_regressors_no_false_separation():
    """説明変数を15本に増やしても、健全なDGPで偽の分離判定無しに収束する。

    当時あったノルム基準の分離ヒューリスティックは、多変量モデルで標準化パラメータの
    ノルムが`√k`オーダーで増え偽陽性が再発しうる懸念があった（Tobitでは係数由来でも
    ノルムが増える）。Tobitは分離検出を呼ばないため、ここで`ComputationError`に
    なってはいけない。`TobitScaling`が設計行列を列標準化・平均センタリングして
    ノルムを抑える回帰ガードでもある。
    """
    rng = np.random.default_rng(3)
    n = 400
    k = 15
    x = rng.normal(0.0, 1.0, size=(n, k))
    betas = np.where(np.arange(k) % 2 == 0, 1.0, -0.7)
    y_star = 2.0 + x @ betas + rng.normal(0.0, 1.5, size=n)
    df = pl.DataFrame(
        {"y": np.maximum(y_star, 0.0)} | {f"x{j}": x[:, j] for j in range(k)}
    )

    res = Tobit(df, y="y", x=[f"x{j}" for j in range(k)]).fit()

    assert res.converged
    # 代表的な係数が真値近傍（厳密照合は数値テストの領分）。
    assert abs(res.params["x0"] - 1.0) < 0.5
    assert abs(res.params["x1"] - (-0.7)) < 0.5
    assert 1.0 < res.sigma < 2.0


def test_mroz_hours_raw_scale_converges_without_false_separation():
    """実データ（Wooldridge mroz `hours`、生スケール）で偽の分離判定無しに
    収束する（当時あったノルム基準の分離ヒューリスティック誤発火の実データ回帰）。

    `hours`（0〜4950、左打ち切り約43%）を Example 17.2 の RHS 7変数で推定する。
    `y`の標準偏差が大きく（σ̂≈1122）、修正前は標準化パラメータノルムが
    閾値100を超え`ComputationError`になっていた。R `AER::tobit`（survreg）との
    厳密な数値照合は別テストの領分。ここでは「生スケールでも
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


@pytest.mark.parametrize("solver", ["newton", "bfgs", "lbfgs"])
@pytest.mark.parametrize("cov_type", MLE_COV_TYPES)
def test_truncated_point_returns_unconverged_result_for_every_solver(
    solver, cov_type
):
    """`max_iter=1`・`raise_on_non_convergence=False`で、重く打ち切られた
    データ（85%が左打ち切り）の打ち切り点でも、全solver・全`cov_type`で未収束の結果を返し、
    標準誤差は有限値になる。newton（LM減衰で凹な領域に留まる）・bfgs・lbfgs（初期点のHessianが
    正定値なら厳密な逆行列から出発し、1手目がNewtonステップと一致）が同じ振る舞いになる。
    かつてlbfgsは初期点のHessianを使わず1手目が勾配方向だったため、凹でない領域に着地して
    Hessianの逆行列を使う`cov_type`が`ComputationError`（`SingularHessian`）になっていた。
    """
    kwargs = {"cluster": "g"} if cov_type == "cluster" else {}
    res = Tobit(
        heavily_censored_tobit_dataset(),
        y="y",
        x=["x1", "x2"],
        options=TobitOptions(
            solver=solver,
            cov_type=cov_type,
            lower=0.0,
            max_iter=1,
            tol=1e-12,
            raise_on_non_convergence=False,
            **kwargs,
        ),
    ).fit()
    assert res.converged is False
    for name, se in res.std_errors.items():
        assert math.isfinite(se), f"std_errors[{name}]={se} is not finite"
