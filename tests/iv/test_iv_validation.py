"""IV の入力・変数指定・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」）。成功パスの構造・API・
オプション反映は `test_iv_api.py`、主リファレンス（linearmodels）との数値照合は
`test_iv_reference.py`（2SLS）・`test_iv_gmm_reference.py`（GMM）、R クロスチェックは
`test_iv_crosscheck.py`（2SLS）・`test_iv_gmm_crosscheck.py`（GMM）（OLS/WLS/Logit/Probit の `test_<手法>_validation.py` 等と
同じ4分割）。

`iv_dataset` フィクスチャと `our_fit` ヘルパーは `tests/iv/conftest.py`／
`tests/support/_iv_helpers.py`。
"""

from __future__ import annotations

import math

import _error_expectations as msgs
import pandas as pd
import polars as pl
import pytest
from _constants import DATA_DIR
from _error_expectations import escaped
from _errors import raises_error
from _helpers import (
    ROW_TIME,
    TIED_TIME_COLUMNS,
    hac_time_for,
    with_cluster_groups,
    with_row_time,
)
from _iv_helpers import iv_name_collision_frame, our_fit
from econometricsmodels import (
    IV,
    ComputationError,
    ErrorCode,
    IVOptions,
    ValidationError,
)

from benchmark.iv.constants import LINEARMODELS_COV_TYPES

# ── ValidationError（入力データ） ─────────────────────────────────────


_DEFAULT_ROLE_KWARGS = {
    "y": "y",
    "x_exog": ["x1"],
    "x_endog": ["endog1"],
    "instruments": ["z1", "z2"],
}


@pytest.mark.parametrize(
    "field, value, params",
    [
        (
            "x_exog",
            ["y", "x1"],
            {
                "column": "y",
                "argument": "y",
                "other_argument": "x_exog",
            },
        ),
        (
            "x_endog",
            ["y"],
            {
                "column": "y",
                "argument": "y",
                "other_argument": "x_endog",
            },
        ),
        (
            "instruments",
            ["y", "z1"],
            {
                "column": "y",
                "argument": "y",
                "other_argument": "instruments",
            },
        ),
        (
            "x_exog",
            ["x1", "endog1"],
            {
                "column": "endog1",
                "argument": "x_endog",
                "other_argument": "x_exog",
            },
        ),
        (
            "instruments",
            ["x1", "z2"],
            {
                "column": "x1",
                "argument": "instruments",
                "other_argument": "x_exog",
            },
        ),
        (
            "instruments",
            ["endog1", "z2"],
            {
                "column": "endog1",
                "argument": "instruments",
                "other_argument": "x_endog",
            },
        ),
    ],
    ids=[
        "y_in_x_exog",
        "y_in_x_endog",
        "y_in_instruments",
        "x_exog_overlaps_x_endog",
        "instruments_overlaps_x_exog",
        "x_endog_overlaps_instruments",
    ],
)
def test_variable_role_overlap_raises(iv_dataset, field, value, params):
    """同じ列名が2つの異なるロール（`y`/`x_exog`/`x_endog`/`instruments`）に
    渡された場合、`ValidationError`になること。
    """
    with raises_error(ErrorCode.ROLE_COLUMN_OVERLAP, **params):
        IV(iv_dataset, **{**_DEFAULT_ROLE_KWARGS, field: value}).fit()


@pytest.mark.parametrize(
    "field, value, name, role",
    [
        ("instruments", ["z1", "z1"], "z1", "instruments"),
        ("x_exog", ["x1", "x1"], "x1", "x_exog"),
        ("x_endog", ["endog1", "endog1"], "endog1", "x_endog"),
    ],
    ids=[
        "duplicate_instruments_column",
        "duplicate_x_exog_column",
        "duplicate_x_endog_column",
    ],
)
def test_duplicate_column_within_role_raises(
    iv_dataset, field, value, name, role
):
    """同じロール内で同じ列名を2回渡した場合、`ValidationError`になること。"""
    with raises_error(
        ErrorCode.ROLE_DUPLICATE_COLUMN, column=name, argument=role
    ):
        IV(iv_dataset, **{**_DEFAULT_ROLE_KWARGS, field: value}).fit()


def test_const_collision_with_include_intercept_raises():
    """`include_intercept=True`のとき`x_exog`に`"const"`という列名を含めると
    自動追加される定数項と衝突し`ValidationError`になること。
    """
    df = iv_name_collision_frame()
    with raises_error(ErrorCode.INTERCEPT_NAME_COLLISION, argument="x_exog"):
        IV(
            df, y="y", x_exog=["const"], x_endog=["endog1"], instruments=["z1"]
        ).fit()


def test_const_collision_in_x_endog_with_include_intercept_raises():
    """`x_exog`だけでなく`x_endog`に`"const"`という列名を含めた
    場合も、自動追加される定数項と衝突し`ValidationError`になること。

    修正前は`fit()`自体は成功していたが、構造方程式本体の`param_names`が
    `['const', 'x1', 'const']`という重複を持つことになり、`res.params`辞書
    （`dict(zip(param_names, params))`）構築時の後勝ちにより真の切片係数が
    サイレントに失われていた（前項のテストの背景参照）。
    """
    df = iv_name_collision_frame()
    with raises_error(ErrorCode.INTERCEPT_NAME_COLLISION, argument="x_endog"):
        IV(
            df, y="y", x_exog=["x1"], x_endog=["const"], instruments=["z1"]
        ).fit()


def test_const_collision_in_instruments_with_include_intercept_raises():
    """`instruments`に`"const"`という列名を含めた場合も、自動
    追加される定数項と衝突し`ValidationError`になること。

    修正前は`fit()`自体は成功していたが、`first_stage()[endog名].param_names`
    に`"const"`が2回出現し、`OlsResults.params`構築時の後勝ちにより真の切片
    係数が操作変数の係数でサイレントに上書きされていた（前々項のテストの背景参照）。
    """
    df = iv_name_collision_frame()
    with raises_error(
        ErrorCode.INTERCEPT_NAME_COLLISION, argument="instruments"
    ):
        IV(
            df,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["const", "z1"],
        ).fit()


@pytest.mark.parametrize("role", ["y", "x_exog", "x_endog", "instruments"])
def test_missing_column_raises(iv_dataset, role):
    """`y`・`x_exog`・`x_endog`・`instruments`のどの役割でも、存在しない列名は
    `ValidationError`になる。
    """
    args = {
        "y": "y",
        "x_exog": ["x1"],
        "x_endog": ["endog1"],
        "instruments": ["z1", "z2"],
    }
    args[role] = "nonexistent" if role == "y" else ["nonexistent"]
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument=role,
        data_argument="data",
    ):
        IV(iv_dataset, **args).fit()


def test_data_not_polars_raises():
    """`data`にpolars以外のDataFrame（pandas等）を渡すと、内部実装
    （`pyo3-polars`の`get_columns`呼び出し）が漏れた`AttributeError`ではなく
    `TypeError`になること（`test_ols_validation.py`と同じ検証。
    `IVResult`自体には`predict()`/`augment()`が無く、`first_stage()`が
    返す`OLSResults`側の検証は`test_ols_validation.py`でカバー済みのため
    `data`のみ確認する）。
    """
    bad = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x1": [1.0, 2.0, 3.0],
            "endog1": [1.0, 0.5, 1.5],
            "z1": [0.1, 0.2, 0.3],
            "z2": [0.3, 0.1, 0.2],
        }
    )
    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME_PANDAS,
            param_name="data",
        ),
    ):
        IV(
            bad,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_y_empty_string_raises(iv_dataset):
    """`y`に空文字列を渡した場合`ValidationError`
    （`test_ols_validation.py::test_y_empty_string_raises`参照）。
    """
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="",
        argument="y",
        data_argument="data",
    ):
        IV(
            iv_dataset,
            y="",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


_ROLE_COLUMNS = ["y", "x1", "x2", "endog1", "endog2", "z1", "z2"]

# `_fit_all_roles`で各列を渡す引数。
_ROLE_ARGUMENT = {
    "y": "y",
    "x1": "x_exog",
    "x2": "x_exog",
    "endog1": "x_endog",
    "endog2": "x_endog",
    "z1": "instruments",
    "z2": "instruments",
}


_ROLE_VALUES: dict[str, list[float]] = {
    "y": [1.0, 2.0, 3.0, 4.0],
    "x1": [0.5, 1.5, 2.5, 3.5],
    "x2": [3.0, 1.0, 4.0, 2.0],
    "endog1": [2.0, 1.0, 4.0, 3.0],
    "endog2": [1.0, 4.0, 2.0, 3.0],
    "z1": [1.0, 3.0, 2.0, 4.0],
    "z2": [4.0, 2.0, 3.0, 1.0],
}


def _fit_all_roles(df: pl.DataFrame) -> None:
    """`x_exog`・`x_endog`・`instruments`のいずれも2本以上持つ構成でfitする
    （リスト引数の先頭だけでなく、2本目以降の列でも検証するため）。
    """
    IV(
        df,
        y="y",
        x_exog=["x1", "x2"],
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2"],
    ).fit()


@pytest.mark.parametrize("bad_col", _ROLE_COLUMNS)
def test_null_values_raise(bad_col):
    """欠損値は`column_extraction`の責務で`ValidationError`。`y`列だけでなく
    `x_exog`/`x_endog`/`instruments`側の列でも検証する。
    """
    values: dict[str, list[float | None]] = dict(_ROLE_VALUES)
    values[bad_col] = [values[bad_col][0], None, *values[bad_col][2:]]
    df = pl.DataFrame(values)
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL,
        column=bad_col,
        count=1,
        argument=_ROLE_ARGUMENT[bad_col],
        data_argument="data",
    ):
        _fit_all_roles(df)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf")],
    ids=["nan", "inf"],
)
@pytest.mark.parametrize("bad_col", _ROLE_COLUMNS)
def test_non_finite_values_raise(bad_col, value):
    """NaN・無限大は`column_extraction.rs`内でnull（`test_null_values_raise`）
    とは別ロジックのため個別に確認する。`y`列だけでなく`x_exog`/`x_endog`/
    `instruments`側の列でも、リスト引数の2本目以降も含めて検証する
    （`test_null_values_raise`と同じ理由）。
    """
    values: dict[str, list[float]] = dict(_ROLE_VALUES)
    values[bad_col] = [values[bad_col][0], value, *values[bad_col][2:]]
    df = pl.DataFrame(values)
    with raises_error(
        ErrorCode.COLUMN_NON_FINITE,
        column=bad_col,
        value=pytest.approx(value, nan_ok=True),
        row=1,
        argument=_ROLE_ARGUMENT[bad_col],
        data_argument="data",
    ):
        _fit_all_roles(df)


@pytest.mark.parametrize("bad_col", _ROLE_COLUMNS)
def test_non_numeric_dtype_raises(bad_col):
    """文字列列は、dtypeの時点で`ValidationError`。`y`列だけでなく
    `x_exog`/`x_endog`/`instruments`側の列でも検証する（`test_null_values_raise`
    と同じ理由。`test_ols_validation.py::test_non_numeric_dtype_raises`参照）。
    """
    values: dict[str, list] = dict(_ROLE_VALUES)
    values[bad_col] = ["a", "b", "c", "d"]
    df = pl.DataFrame(values)
    with raises_error(
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        column=bad_col,
        dtype="String",
        argument=_ROLE_ARGUMENT[bad_col],
        data_argument="data",
    ):
        _fit_all_roles(df)


_OBSERVATION_COUNT_CASES = [
    # (include_intercept, n, outcome)。構造方程式は x1・endog1（切片ありk=3）、
    # 第一段階は x1・z1・z2（切片ありl=4）。切片なしはそれぞれk=2、l=3。
    pytest.param(True, 2, "structural", id="intercept-n2"),
    pytest.param(True, 3, "structural", id="intercept-n3"),
    pytest.param(True, 4, "first_stage", id="intercept-n4"),
    pytest.param(True, 5, "ok", id="intercept-n5"),
    pytest.param(False, 2, "structural", id="no_intercept-n2"),
    pytest.param(False, 3, "first_stage", id="no_intercept-n3"),
    pytest.param(False, 4, "ok", id="no_intercept-n4"),
]


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
@pytest.mark.parametrize(
    ("include_intercept", "n", "outcome"), _OBSERVATION_COUNT_CASES
)
def test_observation_count_boundaries(
    iv_dataset, estimator, include_intercept, n, outcome
):
    """観測数の境界。構造方程式の`n <= k`は構造方程式の`k`で、`n > k`でも
    第一段階の回帰変数の数`l`以下（`n <= l`）なら第一段階の観測数不足として
    `ValidationError`。`n = l + 1`から成功する。構造方程式のチェックが先に走る
    ため、`n <= k`は第一段階の`l`に隠れない。`estimator`によらず同じ検証。
    """
    k = 3 if include_intercept else 2
    l_first = k + 1
    options = IVOptions(
        estimator=estimator, include_intercept=include_intercept
    )
    df = iv_dataset.head(n)
    if outcome == "structural":
        with raises_error(
            ErrorCode.INSUFFICIENT_OBSERVATIONS,
            n_obs=n,
            n_params=k,
            min_obs=k + 1,
        ):
            our_fit(df, options=options)
    elif outcome == "first_stage":
        with raises_error(
            ErrorCode.INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE,
            n_obs=n,
            n_params=l_first,
            min_obs=l_first + 1,
        ):
            our_fit(df, options=options)
    else:
        result = our_fit(df, options=options)
        assert result.n_obs == n


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_observation_count_is_checked_before_cluster_count(
    iv_dataset, estimator
):
    """観測数の検証は、クラスター数の検証より先に走る。クラスターが1つだけで
    `n <= l`の入力は、クラスター数ではなく第一段階の観測数不足で拒否される。
    """
    df = iv_dataset.head(4).with_columns(pl.lit("a").alias("g"))
    with raises_error(
        ErrorCode.INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE,
        n_obs=4,
        n_params=4,
        min_obs=5,
    ):
        our_fit(
            df,
            options=IVOptions(
                estimator=estimator, cov_type="cluster", cluster="g"
            ),
        )


def test_insufficient_instruments_raises(iv_dataset):
    """識別の順序条件`len(instruments) >= len(x_endog)`を満たさない場合
    `ValidationError`（`IvError::InsufficientInstruments`）。

    `x_endog`・`instruments`のどちらも1要素以上（`test_x_endog_empty_raises`・
    `test_instruments_empty_raises`の空リスト検証には引っかからない）だが、
    `instruments`の数が`x_endog`に足りない組み合わせにする必要がある
    （空リスト検証対応時に必須の修正）。
    """
    with raises_error(
        ErrorCode.INSUFFICIENT_INSTRUMENTS, n_instruments=1, n_endog=2
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1", "z2"],
            instruments=["z1"],
        ).fit()


@pytest.mark.parametrize(
    "x_endog, instruments",
    [
        pytest.param([], [], id="both_empty"),
        pytest.param([], ["z1"], id="x_endog_empty_only"),
    ],
)
def test_x_endog_empty_raises(iv_dataset, x_endog, instruments):
    """`x_endog`が空リストの場合`ValidationError`。

    `x_endog=[]`・`instruments=[]`は実質OLSになるが、そもそもIVを使用すること自体が
    誤りなので弾く。`x_endog=[]`・`instruments`非空（対応する内生変数の無い操作変数
    だけを指定する誤用）も同様に弾く。
    """
    with raises_error(ErrorCode.ROLE_EMPTY, argument="x_endog"):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=x_endog,
            instruments=instruments,
        ).fit()


def test_instruments_empty_raises(iv_dataset):
    """`instruments`が空リストの場合`ValidationError`。

    `x_endog`は非空にする（空だと`test_x_endog_empty_raises`の`x_endog`側の
    バリデーションが先に発火してしまうため）。
    """
    with raises_error(ErrorCode.ROLE_EMPTY, argument="instruments"):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=[],
        ).fit()


# ── ValidationError（入力データ: hac_time列） ─────────────────────────


@pytest.mark.parametrize(
    "options_kwargs",
    [
        pytest.param({"cov_type": "hac"}, id="2sls_cov_type"),
        pytest.param(
            {"estimator": "gmm", "cov_type": "hac"}, id="gmm_cov_type"
        ),
        pytest.param(
            {"estimator": "gmm", "gmm_weight_type": "hac"},
            id="gmm_weight_type",
        ),
        # `gmm_type`ごとに重み行列を組み立てる箇所が異なる
        pytest.param(
            {
                "estimator": "gmm",
                "gmm_type": "iterated",
                "gmm_weight_type": "hac",
            },
            id="gmm_iterated_weight_type",
        ),
        pytest.param(
            {"estimator": "gmm", "gmm_type": "one_step", "cov_type": "hac"},
            id="gmm_one_step_cov_type",
        ),
        # 同じ列が`cov_type`と`gmm_weight_type`の両方から読まれる
        pytest.param(
            {
                "estimator": "gmm",
                "cov_type": "hac",
                "gmm_weight_type": "hac",
            },
            id="gmm_cov_type_and_weight_type",
        ),
    ],
)
@pytest.mark.parametrize(("time_expr", "rows"), TIED_TIME_COLUMNS)
def test_hac_time_with_tied_values_raises(
    iv_dataset, options_kwargs, time_expr, rows
):
    """`hac_time`に同じ値が1組でもあれば`ValidationError`にする。2SLS・GMMの
    `cov_type="hac"`と、GMMの重み行列（`gmm_weight_type="hac"`、別の抽出経路）の
    どちらでも、時点の順序が定まらないまま行順にフォールバックしない。
    """
    df = iv_dataset.with_columns(time_expr.alias("t"))
    options = IVOptions(hac_lags=2, hac_time="t", **options_kwargs)
    with raises_error(
        ErrorCode.COLUMN_VALUES_TIED,
        column="t",
        first_row=rows[0],
        second_row=rows[1],
        argument="hac_time",
        data_argument="data",
    ):
        our_fit(df, options=options)


# ── ValidationError（augment()） ──────────────────────────────────────


def test_first_stage_augment_none_raises_validation_error(iv_dataset):
    """`first_stage()`が返す`OLSResults`は、各内生変数の第一段階回帰専用に
    構築され単一のソースDataFrameを持たないため、`augment(new_data=None)`は
    `ValidationError`（`new_data`を指定した呼び出しは通常どおり動作する、
    `docs/spec/ols-spec.md`「augment()」参照）。
    """
    res = our_fit(iv_dataset)
    first_stage = res.first_stage()["endog1"]

    with raises_error(ErrorCode.AUGMENT_REQUIRES_NEW_DATA):
        first_stage.augment()

    new_data = pl.DataFrame({"x1": [1.0], "z1": [0.5], "z2": [0.2]})
    augmented = first_stage.augment(new_data)
    assert augmented.columns == ["x1", "z1", "z2", "predicted"]


# ── ValidationError（オプション） ─────────────────────────────────────


@pytest.mark.parametrize("estimator", ["invalid", ""])
def test_unknown_estimator_raises(iv_dataset, estimator):
    """未知の`estimator`（空文字列を含む）は`ValidationError`。"""
    options = IVOptions(estimator=estimator)
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="estimator",
        value=estimator,
        allowed=msgs.ALLOWED_ESTIMATORS,
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("cov_type", ["invalid", ""])
def test_unknown_cov_type_raises(iv_dataset, cov_type):
    """未知の`cov_type`（空文字列を含む）は`ValidationError`。"""
    options = IVOptions(cov_type=cov_type)
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="cov_type",
        value=cov_type,
        allowed=msgs.ALLOWED_COV_TYPES_LINEAR,
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("cov_type", ["nonrobust", "NONROBUST", "NonRobust"])
def test_nonrobust_is_rejected(iv_dataset, cov_type):
    """`"nonrobust"`は別名として受け付けない（概念ごとに文字列を1つに絞る）。"""
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="cov_type",
        value=cov_type,
        allowed=msgs.ALLOWED_COV_TYPES_LINEAR,
    ):
        our_fit(
            iv_dataset,
            options=IVOptions(cov_type=cov_type, **hac_time_for(cov_type)),
        )


@pytest.mark.parametrize("gmm_weight_type", ["invalid", ""])
def test_unknown_weight_type_raises(iv_dataset, gmm_weight_type):
    """未知の`gmm_weight_type`（空文字列を含む）は`ValidationError`。"""
    options = IVOptions(estimator="gmm", gmm_weight_type=gmm_weight_type)
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="gmm_weight_type",
        value=gmm_weight_type,
        allowed=msgs.ALLOWED_GMM_WEIGHT_TYPES,
    ):
        our_fit(iv_dataset, options=options)


def test_cluster_without_col_raises(iv_dataset):
    options = IVOptions(cov_type="cluster")
    with raises_error(ErrorCode.CLUSTER_REQUIRED, setting="cov_type"):
        our_fit(iv_dataset, options=options)


def test_cluster_nonexistent_column_raises(iv_dataset):
    """`cluster`が実在しない列名を指すと`ValidationError`（OLS/WLS/Logit/
    Probitと同じ理由）。
    """
    options = IVOptions(cov_type="cluster", cluster="does_not_exist")
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="does_not_exist",
        argument="cluster",
        data_argument="data",
    ):
        our_fit(iv_dataset, options=options)


def test_insufficient_clusters_raises(iv_dataset):
    """クラスターが1種類しかない場合`ValidationError`。"""
    df = iv_dataset.with_columns(pl.lit(0).alias("single_cluster"))
    options = IVOptions(cov_type="cluster", cluster="single_cluster")
    with raises_error(
        ErrorCode.CLUSTER_COUNT_TOO_SMALL,
        setting="cov_type",
        n_clusters=1,
        min_clusters=2,
    ):
        our_fit(df, options=options)


def test_gmm_weight_type_cluster_without_col_raises(iv_dataset):
    """`gmm_weight_type="cluster"`なのに`cluster`未指定なら、エラーは重み行列の
    設定を指す（`cov_type="cluster"`のときの`setting="cov_type"`とは区別される）。
    """
    options = IVOptions(estimator="gmm", gmm_weight_type="cluster")
    with raises_error(ErrorCode.CLUSTER_REQUIRED, setting="gmm_weight_type"):
        our_fit(iv_dataset, options=options)


def test_gmm_weight_type_cluster_with_single_cluster_raises(iv_dataset):
    """`cov_type`がclusterでなければ、クラスターが1種類しかない
    `gmm_weight_type="cluster"`のエラーは重み行列の設定を指す。
    """
    df = iv_dataset.with_columns(pl.lit(0).alias("single_cluster"))
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="cluster",
        cluster="single_cluster",
        cov_type="classical",
    )
    with raises_error(
        ErrorCode.CLUSTER_COUNT_TOO_SMALL,
        setting="gmm_weight_type",
        n_clusters=1,
        min_clusters=2,
    ):
        our_fit(df, options=options)


def test_cov_type_cluster_is_reported_before_gmm_weight_type(iv_dataset):
    """`cov_type="cluster"`と`gmm_weight_type="cluster"`の両方が`cluster`未指定
    なら、先に検証される`cov_type`が報告される。
    """
    options = IVOptions(
        estimator="gmm", cov_type="cluster", gmm_weight_type="cluster"
    )
    with raises_error(ErrorCode.CLUSTER_REQUIRED, setting="cov_type"):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_cluster_count_at_most_slopes_raises_validation_error(
    iv_dataset, estimator
):
    """`cov_type="cluster"`でクラスター数G≤構造方程式の傾き係数の数q
    （`our_fit`既定は`x_exog=["x1"]`・`x_endog=["endog1"]`で`q=2`、ここで
    `G=2 == q=2`）は`ValidationError`
    （`CommonError::InsufficientClustersForInference`）。

    `rank(Ŝ)≤G-1`のためG≤qでロバストWald/F（χ²）検定のq×q部分行列が構造的に
    特異になる。`engine_pybind::fit()`が`compute_first_stage`より前に構造方程式の
    qで判定するため、第一段階回帰（`endog1 ~ x_exog(x1) + instruments(z1, z2)`、
    q=3）の`FirstStageFailed`ではなく、構造方程式のq=2を含むメッセージが出る。
    `gmm_weight_type="cluster"`の重み行列`S`（l×l）が`G<l`で特異になる別軸の問題
    とは区別する。
    """
    df = with_cluster_groups(iv_dataset, 2)
    options = IVOptions(
        estimator=estimator, cov_type="cluster", cluster="cluster_group"
    )
    with raises_error(
        ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES,
        setting="cov_type",
        n_clusters=2,
        n_slopes=2,
        min_clusters=3,
    ):
        our_fit(df, options=options)


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_cluster_count_between_structural_and_first_stage_slopes_leaves_first_stage_f_undefined(
    iv_dataset, estimator
):
    """構造方程式ではG>q（G=3、q=2）だが第一段階ではG≤q（q=3）の場合、
    `fit()`は成功し、第一段階の全体F検定（`f_statistic`/`f_p_value`/`f_df_num`/
    `f_df_denom`）だけが`None`になる。

    第一段階のクラスターロバストな共分散は`rank(Ŝ)≤G-1<q`で傾き部分が特異なため
    全体F検定は計算できないが、係数・標準誤差は有効で、弱操作変数F統計量は`cov_type`に
    依存しない古典的なネストF検定なので`cov_type="classical"`と同じ値になる。
    """
    df = with_cluster_groups(iv_dataset, 3)
    res = our_fit(
        df,
        options=IVOptions(
            estimator=estimator, cov_type="cluster", cluster="cluster_group"
        ),
    )
    first_stage = res.first_stage()["endog1"]
    assert first_stage.f_statistic is None
    assert first_stage.f_p_value is None
    assert first_stage.f_df_num is None
    assert first_stage.f_df_denom is None
    assert all(
        math.isfinite(se) and se > 0 for se in first_stage.std_errors.values()
    )
    assert math.isfinite(res.wald_statistic)

    classical = our_fit(df, options=IVOptions(estimator=estimator))
    assert (
        res.weak_instrument_f_statistics
        == classical.weak_instrument_f_statistics
    )


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
@pytest.mark.parametrize("include_intercept", [True, False])
@pytest.mark.parametrize(
    ("n_groups", "outcome"),
    [(2, "structural"), (3, "first_stage_f_undefined"), (4, "ok")],
)
def test_cluster_count_boundaries(
    iv_dataset, estimator, include_intercept, n_groups, outcome
):
    """クラスター数の境界を、切片の有無の両方で確認する。構造方程式の傾き係数の数は
    `x1`・`endog1`の2個、第一段階は`x1`・`z1`・`z2`の3個で、切片の有無で傾き係数の数は
    変わらない（切片ありは`const`が傾きから除かれ、切片なしでは定数項が無い分、全係数が
    傾きになるため`q = k`）。G=2は構造方程式のG≤qで`ValidationError`、G=3は構造方程式では
    G>qだが第一段階でG≤qのため成功して第一段階の全体F検定だけが`None`
    （`test_cluster_count_between_structural_and_first_stage_slopes_leaves_first_
    stage_f_undefined`）、G=4は第一段階の全体F検定も計算できる（分母自由度`G-1=3`）。
    """
    df = with_cluster_groups(iv_dataset, n_groups)
    options = IVOptions(
        estimator=estimator,
        include_intercept=include_intercept,
        cov_type="cluster",
        cluster="cluster_group",
    )
    if outcome == "structural":
        with raises_error(
            ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES,
            setting="cov_type",
            n_clusters=2,
            n_slopes=2,
            min_clusters=3,
        ):
            our_fit(df, options=options)
        return

    res = our_fit(df, options=options)
    expected = ["x1", "endog1"]
    assert res.param_names == (
        ["const", *expected] if include_intercept else expected
    )
    assert res.n_obs == df.height
    first_stage = res.first_stage()["endog1"]
    if outcome == "first_stage_f_undefined":
        assert first_stage.f_statistic is None
        assert first_stage.f_p_value is None
        assert (first_stage.f_df_num, first_stage.f_df_denom) == (None, None)
    else:
        assert first_stage.f_statistic is not None
        assert first_stage.f_p_value is not None
        assert (first_stage.f_df_num, first_stage.f_df_denom) == (3, 3)


@pytest.mark.parametrize("gmm_type", ["two_step", "iterated"])
def test_gmm_cluster_weight_type_raises_validation_error_when_cluster_count_is_less_than_instrument_count(
    iv_dataset, gmm_type
):
    """`estimator="gmm"`固有のValidationErrorパス。`gmm_weight_type="cluster"`の重み行列`S`
    （l×l、`l`は全操作変数の数）はG個のランク1行列の和のため`rank(S)≤G`
    （`engine/src/iv/CLAUDE.md`「クラスター数Gと操作変数の数lの関係」参照）。
    `G=2 < l=3`（`x_exog=[]`・`instruments=["z1","z2"]`で`l=const+z1+z2=3`、過剰識別）
    だと`S`が構造的に特異になるため、`fit()`冒頭で`ValidationError`
    （`IvError::InsufficientClustersForWeightMatrix`）として弾く。`gmm_type="one_step"`
    は`gmm_weight_type`を使わないため検証しない。
    """
    n = iv_dataset.height
    df = iv_dataset.with_columns(
        (pl.int_range(pl.len()) < n // 2).cast(pl.Int64).alias("cluster_group")
    )
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="cluster",
        cluster="cluster_group",
        cov_type="classical",
        gmm_type=gmm_type,
    )
    with raises_error(
        ErrorCode.GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL,
        n_clusters=2,
        n_instruments=3,
        min_clusters=3,
    ):
        IV(
            df,
            y="y",
            x_exog=[],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
            options=options,
        ).fit()


@pytest.mark.parametrize(
    ("gmm_type", "instruments", "n_groups", "l", "outcome"),
    [
        # 過剰識別（l=3, k=2）: rank(S)≤Gなので、G=l=3ちょうどで成功する。
        ("two_step", ["z1", "z2"], 3, 3, "ok"),
        # 丁度識別（l=k=2）: Z'ê=0によりrank(S)≤G-1なので、G=l=2は失敗し、
        # G=l+1=3で成功する。
        ("two_step", ["z1"], 2, 2, "raises"),
        ("iterated", ["z1"], 2, 2, "raises"),
        ("two_step", ["z1"], 3, 2, "ok"),
        ("iterated", ["z1"], 3, 2, "ok"),
    ],
    ids=[
        "over_identified_g_equals_l-two_step",
        "just_identified_g_equals_l-two_step",
        "just_identified_g_equals_l-iterated",
        "just_identified_g_exceeds_l-two_step",
        "just_identified_g_exceeds_l-iterated",
    ],
)
def test_gmm_cluster_weight_type_cluster_count_boundaries(
    iv_dataset, gmm_type, instruments, n_groups, l, outcome
):
    """`gmm_weight_type="cluster"`のクラスター数の境界の、失敗の反対側（成功）。
    `test_gmm_cluster_weight_type_raises_validation_error_when_cluster_count_is_less_
    than_instrument_count`（過剰識別でG<l）と対にする（閾値は識別状態で異なる、
    `engine/src/iv/CLAUDE.md`「クラスター数Gと操作変数の数lの関係」参照）。
    `cov_type="classical"`のため、この検証だけが効く。

    過剰識別でG=lちょうどは`two_step`のみ確認する。`iterated`では、重みを残差から
    作り直すたびにJ統計量が常にl（`S`が`G×G`の正方行列`ĜĜ'`になり
    `Ĝ'S⁻¹Ĝ=I`）となって目的関数が平坦になり、反復推定が退化して
    `ComputationError`になる。
    """
    df = with_cluster_groups(iv_dataset, n_groups)
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="cluster",
        cluster="cluster_group",
        cov_type="classical",
        gmm_type=gmm_type,
    )

    def fit():
        return IV(
            df,
            y="y",
            x_exog=[],
            x_endog=["endog1"],
            instruments=instruments,
            options=options,
        ).fit()

    if outcome == "raises":
        # 丁度識別（`instruments`が1つ）は`l + 1`、過剰識別は`l`が下限。
        with raises_error(
            ErrorCode.GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL,
            n_clusters=n_groups,
            n_instruments=l,
            min_clusters=l + 1,
        ):
            fit()
    else:
        res = fit()
        assert res.param_names == ["const", "endog1"]
        assert res.n_obs == df.height


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(iv_dataset, confidence_level):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合
    `ValidationError`。
    """
    options = IVOptions(confidence_level=confidence_level)
    with raises_error(
        ErrorCode.CONFIDENCE_LEVEL_OUT_OF_RANGE, value=confidence_level
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("hac_lags", [-1, 500])  # 500 == iv_dataset の n_obs
def test_invalid_hac_lags_raises(iv_dataset, hac_lags):
    """`hac_lags`が`[0, n)`の範囲外の場合`ValidationError`。"""
    options = IVOptions(cov_type="hac", hac_lags=hac_lags, hac_time=ROW_TIME)
    with raises_error(
        ErrorCode.HAC_LAGS_OUT_OF_RANGE, value=hac_lags, n_obs=500, max=499
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("hac_lags", [-1, 500])  # 500 == iv_dataset の n_obs
def test_invalid_gmm_weight_hac_lags_raises(iv_dataset, hac_lags):
    """`gmm_weight_type="hac"`の`hac_lags`が`[0, n)`の範囲外の場合も
    `cov_type="hac"`と同じ`HAC_LAGS_OUT_OF_RANGE`の`ValidationError`。
    """
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="hac",
        hac_lags=hac_lags,
        hac_time=ROW_TIME,
    )
    with raises_error(
        ErrorCode.HAC_LAGS_OUT_OF_RANGE, value=hac_lags, n_obs=500, max=499
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize(
    "gmm_weight_type",
    ["unadjusted", "homoskedastic", "heteroskedastic", "kernel", "invalid"],
)
def test_unknown_gmm_weight_type_raises(iv_dataset, gmm_weight_type):
    """別名（`unadjusted`/`kernel`/`homoskedastic`/`heteroskedastic`）は
    受け付けない（`cov_type`と同じ語彙`classical`/`hac`に絞る）。
    """
    options = IVOptions(estimator="gmm", gmm_weight_type=gmm_weight_type)
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="gmm_weight_type",
        value=gmm_weight_type,
        allowed=msgs.ALLOWED_GMM_WEIGHT_TYPES,
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("gmm_max_iter", [-1, 0, 1, 2])
def test_invalid_gmm_max_iter_raises(iv_dataset, gmm_max_iter):
    options = IVOptions(
        estimator="gmm", gmm_type="iterated", gmm_max_iter=gmm_max_iter
    )
    with raises_error(
        ErrorCode.GMM_MAX_ITER_OUT_OF_RANGE,
        value=gmm_max_iter,
        min=3,
        max=10_000,
    ):
        our_fit(iv_dataset, options=options)


def test_unknown_gmm_type_raises(iv_dataset):
    options = IVOptions(estimator="gmm", gmm_type="three_step")
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="gmm_type",
        value="three_step",
        allowed=msgs.ALLOWED_GMM_TYPES,
    ):
        our_fit(iv_dataset, options=options)


_GMM = 'estimator="gmm"'


_GMM_WEIGHTED = 'estimator="gmm" and gmm_type in ("two_step", "iterated")'


_GMM_ITERATED = 'estimator="gmm" and gmm_type="iterated"'


_CLUSTER_CONDITION = (
    'cov_type="cluster" or (estimator="gmm" and '
    'gmm_type in ("two_step", "iterated") and gmm_weight_type="cluster")'
)


_HAC_CONDITION = (
    'cov_type="hac" or (estimator="gmm" and '
    'gmm_type in ("two_step", "iterated") and gmm_weight_type="hac")'
)


@pytest.mark.parametrize(
    ("estimator", "gmm_type", "option", "value", "condition"),
    [
        # estimator="2sls"ではGMM専用オプションは全て使われない
        ("2sls", None, "gmm_type", "two_step", _GMM),
        ("2sls", None, "gmm_weight_type", "robust", _GMM_WEIGHTED),
        ("2sls", None, "gmm_max_iter", 10, _GMM_ITERATED),
        ("2sls", None, "gmm_tol", 1e-6, _GMM_ITERATED),
        ("2sls", None, "raise_on_non_convergence", False, _GMM_ITERATED),
        # gmm_type="one_step"は重み行列を使わない
        ("gmm", "one_step", "gmm_weight_type", "robust", _GMM_WEIGHTED),
        # gmm_type="one_step"/"two_step"は反復しない
        ("gmm", "one_step", "gmm_max_iter", 10, _GMM_ITERATED),
        ("gmm", "two_step", "gmm_max_iter", 10, _GMM_ITERATED),
        ("gmm", "one_step", "gmm_tol", 1e-6, _GMM_ITERATED),
        ("gmm", "two_step", "gmm_tol", 1e-6, _GMM_ITERATED),
        ("gmm", "one_step", "raise_on_non_convergence", False, _GMM_ITERATED),
        ("gmm", "two_step", "raise_on_non_convergence", False, _GMM_ITERATED),
    ],
)
def test_gmm_option_unused_by_mode_raises(
    iv_dataset, estimator, gmm_type, option, value, condition
):
    """選んだ`estimator`/`gmm_type`で使われないGMM専用オプションが指定
    されたら黙って無視せず`ValidationError`。
    """
    kwargs = {} if gmm_type is None else {"gmm_type": gmm_type}
    options = IVOptions(estimator=estimator, **kwargs, **{option: value})
    with raises_error(
        ErrorCode.OPTION_UNUSED, option=option, condition=condition
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize(
    ("options_kwargs", "option", "condition"),
    [
        ({"cov_type": "classical"}, "cluster", _CLUSTER_CONDITION),
        ({"cov_type": "hc1"}, "cluster", _CLUSTER_CONDITION),
        # estimator="2sls"はgmm_weight_typeを使えず、cov_typeだけで判定される
        (
            {"estimator": "2sls", "cov_type": "classical"},
            "cluster",
            _CLUSTER_CONDITION,
        ),
        # gmm_type="one_step"はgmm_weight_typeを使わない
        (
            {"estimator": "gmm", "gmm_type": "one_step"},
            "cluster",
            _CLUSTER_CONDITION,
        ),
        # gmm_weight_typeが別の値なら使われない
        (
            {"estimator": "gmm", "gmm_weight_type": "robust"},
            "cluster",
            _CLUSTER_CONDITION,
        ),
        ({"cov_type": "cluster"}, "hac_lags", _HAC_CONDITION),
        ({"cov_type": "cluster"}, "hac_time", _HAC_CONDITION),
        # gmm_type="one_step"はgmm_weight_typeを使わないため、hacも使われない
        (
            {"estimator": "gmm", "gmm_type": "one_step"},
            "hac_lags",
            _HAC_CONDITION,
        ),
        (
            {"estimator": "gmm", "gmm_type": "one_step"},
            "hac_time",
            _HAC_CONDITION,
        ),
    ],
)
def test_cov_option_unused_by_mode_raises(
    iv_dataset, options_kwargs, option, condition
):
    """`cluster`/`hac_lags`/`hac_time`は`cov_type`と`gmm_weight_type`の
    どちらからも使われないときだけ`ValidationError`。
    """
    value = 2 if option == "hac_lags" else "x1"
    options = IVOptions(**options_kwargs, **{option: value})
    with raises_error(
        ErrorCode.OPTION_UNUSED, option=option, condition=condition
    ):
        our_fit(iv_dataset, options=options)


def test_cluster_used_only_by_gmm_weight_type_is_accepted(clustered_dataset):
    """`cov_type`が`cluster`でなくても`gmm_weight_type="cluster"`が使う
    ため、`cluster`の指定は有効（どちらか一方でも使えば有効）。
    """
    options = IVOptions(
        estimator="gmm",
        cov_type="classical",
        gmm_weight_type="cluster",
        cluster="cluster_group",
    )
    our_fit(clustered_dataset, options=options)


@pytest.mark.parametrize("gmm_tol", [0.0, -1.0])
def test_invalid_gmm_tol_raises(iv_dataset, gmm_tol):
    options = IVOptions(estimator="gmm", gmm_type="iterated", gmm_tol=gmm_tol)
    with raises_error(
        ErrorCode.TOLERANCE_NOT_POSITIVE_FINITE,
        option="gmm_tol",
        value=gmm_tol,
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("gmm_tol", [float("nan"), float("inf")])
def test_non_finite_gmm_tol_raises(iv_dataset, gmm_tol):
    """NaN/infの`gmm_tol`は収束判定として無意味なため、常に未収束になる前に
    `ValidationError`で弾く。
    """
    options = IVOptions(estimator="gmm", gmm_type="iterated", gmm_tol=gmm_tol)
    with raises_error(
        ErrorCode.TOLERANCE_NOT_POSITIVE_FINITE,
        option="gmm_tol",
        value=pytest.approx(gmm_tol, nan_ok=True),
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize(
    "options_kwargs",
    [
        # gmm_weight_type="hac"だけがhac_*を使う（大文字小文字を問わない）
        {
            "estimator": "GMM",
            "gmm_weight_type": "HAC",
            "hac_lags": 2,
            "hac_time": "x1",
        },
        # gmm_type="iterated"は重みと反復オプションの全てを使う
        {
            "estimator": "gmm",
            "gmm_type": "ITERATED",
            "gmm_weight_type": "robust",
            "gmm_max_iter": 10,
            "gmm_tol": 1e-3,
            "raise_on_non_convergence": False,
        },
        # cov_typeだけがhac_*を使う
        {
            "estimator": "2sls",
            "cov_type": "HAC",
            "hac_lags": 2,
            "hac_time": "x1",
        },
    ],
)
def test_options_used_by_mode_are_accepted(iv_dataset, options_kwargs):
    our_fit(iv_dataset, options=IVOptions(**options_kwargs))


@pytest.mark.parametrize(
    ("options_kwargs", "option", "value"),
    [
        ({"cov_type": "clusterr", "cluster": "x1"}, "cov_type", "clusterr"),
        (
            {"estimator": "gmm", "gmm_type": "bogus", "gmm_max_iter": 10},
            "gmm_type",
            "bogus",
        ),
    ],
)
def test_unknown_value_is_reported_before_unused_option(
    iv_dataset, options_kwargs, option, value
):
    """未知の値のエラーを優先し、未使用オプションの指摘で埋もれさせない。"""
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option=option, value=value
    ):
        our_fit(iv_dataset, options=IVOptions(**options_kwargs))


@pytest.mark.parametrize(
    ("options_kwargs", "setting"),
    [
        pytest.param({"cov_type": "hac"}, "cov_type", id="2sls_cov_type"),
        pytest.param(
            {"estimator": "gmm", "cov_type": "hac"}, "cov_type", id="gmm_cov"
        ),
        pytest.param(
            {"cov_type": "HAC"}, "cov_type", id="2sls_cov_type_uppercase"
        ),
        pytest.param(
            {"estimator": "gmm", "cov_type": "hac", "gmm_weight_type": "hac"},
            "cov_type",
            id="both_missing_reports_cov_type_first",
        ),
        pytest.param(
            {"estimator": "gmm", "gmm_weight_type": "hac"},
            "gmm_weight_type",
            id="gmm_weight_type",
        ),
        pytest.param(
            {
                "estimator": "gmm",
                "gmm_type": "iterated",
                "gmm_weight_type": "HAC",
            },
            "gmm_weight_type",
            id="gmm_iterated_weight_type",
        ),
    ],
)
def test_hac_requires_hac_time(iv_dataset, options_kwargs, setting):
    """HAC（`cov_type="hac"`、GMMの`gmm_weight_type="hac"`）で`hac_time`が未指定だと
    `ValidationError`。`hac_time`は`cov_type`と`gmm_weight_type`で共用するが、
    どちらが要求しているかをメッセージで示す。
    """
    options = IVOptions(hac_lags=2, **options_kwargs)
    with raises_error(ErrorCode.HAC_TIME_REQUIRED, setting=setting):
        our_fit(iv_dataset, options=options)


def test_gmm_one_step_with_hac_cov_type_requires_hac_time(iv_dataset):
    """`gmm_type="one_step"`は重み行列を使わないが、`cov_type="hac"`が`hac_time`を
    要求する。
    """
    options = IVOptions(estimator="gmm", gmm_type="one_step", cov_type="hac")
    with raises_error(ErrorCode.HAC_TIME_REQUIRED, setting="cov_type"):
        our_fit(iv_dataset, options=options)


# ── ComputationError ──────────────────────────────────────────────────


def test_perfect_multicollinearity_raises_computation_error():
    """`x_exog`が完全な多重共線性を持つ場合、第一段階回帰
    （`x_endog[j] ~ x_exog + instruments`）の設計行列が特異になり
    `ComputationError`（`DESIGN_MATRIX_SINGULAR`、`stage="FIRST_STAGE"`）。
    完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）で、
    想定エラーの送出のみ確認する。
    """
    df = pl.read_csv(DATA_DIR / "iv_perfect_multicollinearity.csv")
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR,
        stage="FIRST_STAGE",
        endog_name="endog1",
    ):
        IV(
            df,
            y="y",
            x_exog=["x1", "x2", "x3"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def _singular_first_stage_case(
    label: str, column: str, expr: pl.Expr, include_intercept: bool
):
    return pytest.param(
        column,
        expr,
        include_intercept,
        id=f"{label}-intercept={include_intercept}",
    )


# 切片に依存する共線性（`const`と比例する）は切片ありのみ。それ以外
# （操作変数同士の比例、操作変数と`x_exog`の共線）は切片の有無によらず特異になる。
_SINGULAR_FIRST_STAGE_CASES = [
    _singular_first_stage_case(
        "instruments_collinear_with_constant",
        "z2",
        2 * pl.col("z1") + 1,
        True,
    ),
    _singular_first_stage_case("instrument_constant", "z1", pl.lit(5.0), True),
    *(
        _singular_first_stage_case(label, "z2", expr, include_intercept)
        for include_intercept in (True, False)
        for label, expr in (
            ("instruments_proportional", 2 * pl.col("z1")),
            (
                "instrument_collinear_with_x_exog",
                pl.col("x1") + pl.col("z1"),
            ),
        )
    ),
]


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
@pytest.mark.parametrize(
    ("column", "expr", "include_intercept"), _SINGULAR_FIRST_STAGE_CASES
)
def test_collinear_instruments_raise_computation_error(
    iv_dataset, estimator, column, expr, include_intercept
):
    """`instruments`同士の共線、`instruments`と`x_exog`の共線（`Z=[const, x_exog,
    instruments]`が列ランク落ち）、定数の操作変数（`const`と共線）は、いずれも第一段階
    回帰の設計行列が特異になり`ComputationError`（`DESIGN_MATRIX_SINGULAR`、
    `stage="FIRST_STAGE"`）。第一段階は`estimator`によらず先に走るため、2SLSと
    GMMで同じ経路・同じエラーになる。切片に依存しない共線性は
    `include_intercept=False`でも同じ。
    """
    df = iv_dataset.with_columns(expr.alias(column))
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR,
        stage="FIRST_STAGE",
        endog_name="endog1",
    ):
        our_fit(
            df,
            options=IVOptions(
                estimator=estimator, include_intercept=include_intercept
            ),
        )


def _second_stage_singular_frame(label: str):
    """第一段階は成功し、構造方程式の説明変数が共線になる入力
    `(df, x_exog, x_endog, instruments)`。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")
    if label == "x_endog_collinear":
        return (
            df.with_columns((2 * pl.col("endog1")).alias("endog2")),
            ["x1"],
            ["endog1", "endog2"],
            ["z1", "z2", "z3"],
        )
    if label == "x_endog_with_x_exog":
        return (
            df.with_columns((3 * pl.col("x1")).alias("endog1")),
            ["x1"],
            ["endog1"],
            ["z1", "z2"],
        )
    # 操作変数`z1`が内生変数と（中心化後に）厳密に無相関: 第一段階の予測値が定数に
    # なり、第二段階で`const`と共線になる（階数条件の違反）。
    assert label == "instrument_irrelevant"
    small = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 5.0],
            "endog1": [1.0, 4.0, 4.0, 1.0],
            "z1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return small, [], ["endog1"], ["z1"]


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
@pytest.mark.parametrize(
    ("label", "include_intercept"),
    [
        ("x_endog_collinear", True),
        ("x_endog_collinear", False),
        # 厳密に無相関な操作変数は`const`との共線性（切片あり）でのみ特異になる。
        ("instrument_irrelevant", True),
    ],
)
def test_second_stage_singular_raises_computation_error(
    estimator, label, include_intercept
):
    """第一段階は成功するが構造方程式の説明変数が共線になるケース。第一段階は
    内生変数ごとに独立に走り、各回帰の設計行列`[const, x_exog, instruments]`は
    特異にならない。`x_endog`同士が共線（`endog2 = 2*endog1`）、または操作変数が
    内生変数と厳密に無相関（階数条件の違反）だと、第二段階の設計行列`[const, x_exog, x̂_endog]`が特異になる
    （`second_stage_input`経路）。2SLSは第二段階回帰の`ComputationError`
    （`DESIGN_MATRIX_SINGULAR`、`stage="SECOND_STAGE"`）、GMMは第二段階を持たず
    `X'ZWZ'X`の逆行列が取れずに`ComputationError`
    （`MATRIX_NEAR_SINGULAR`、`matrix="GMM_BREAD"`）になる。
    """
    df, x_exog, x_endog, instruments = _second_stage_singular_frame(label)
    expected = (
        {
            "code": ErrorCode.DESIGN_MATRIX_SINGULAR,
            "stage": "SECOND_STAGE",
        }
        if estimator == "2sls"
        else {
            "code": ErrorCode.MATRIX_NEAR_SINGULAR,
            "matrix": "GMM_BREAD",
        }
    )
    with raises_error(**expected):
        our_fit(
            df,
            x_exog=x_exog,
            x_endog=x_endog,
            instruments=instruments,
            options=IVOptions(
                estimator=estimator, include_intercept=include_intercept
            ),
        )


@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
@pytest.mark.parametrize("include_intercept", [True, False])
def test_endogenous_regressor_that_is_a_function_of_exogenous_has_a_perfect_first_stage(
    estimator, include_intercept
):
    """内生変数が外生変数の厳密な関数（`endog1 = 3*x1`）だと、その第一段階
    `endog1 ~ x1 + z1 + z2`は残差が0の完全フィットになり、標準誤差が定義できない。
    構造方程式の第二段階が共線になる前に、第一段階の文脈つきの`PERFECT_FIT`で止まる。"""
    df, x_exog, x_endog, instruments = _second_stage_singular_frame(
        "x_endog_with_x_exog"
    )
    with raises_error(
        ErrorCode.PERFECT_FIT, stage="FIRST_STAGE", endog_name="endog1"
    ):
        our_fit(
            df,
            x_exog=x_exog,
            x_endog=x_endog,
            instruments=instruments,
            options=IVOptions(
                estimator=estimator, include_intercept=include_intercept
            ),
        )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
def test_scale_variance_raises_computation_error(cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    第一段階回帰の傾き係数の同時共分散部分行列がスケール比の2乗相当の
    条件数を持ち倍精度浮動小数点の限界を超えて数値的に特異になり、
    第一段階の`ComputationError`（`MATRIX_NEAR_SINGULAR`、
    `matrix="COEFFICIENT_COVARIANCE"`、`stage="FIRST_STAGE"`）になる
    （実測確認済み、OLSの同名テストと同じ原理）。
    `test_perfect_multicollinearity_raises_computation_error`と同様、数値比較は
    せずエラーパスのみ確認する。`cov_type`は`LINEARMODELS_COV_TYPES`（hc2/hc3は
    エラーパスの網羅としては不要）で、cov_typeによらず第一段階回帰が特異になる。
    """
    df = pl.read_csv(DATA_DIR / "iv_scale_variance.csv")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR,
        matrix="COEFFICIENT_COVARIANCE",
        stage="FIRST_STAGE",
        endog_name="endog1",
    ):
        IV(
            with_row_time(df),
            y="y",
            x_exog=["x1", "x2"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
            options=options,
        ).fit()


def test_scale_variance_cluster_raises_computation_error():
    """`cluster`も上記`test_scale_variance_raises_computation_error`と同じ
    backstopの対象。`cov_type="cluster"`は`cluster`列の指定が別途必要なため
    `LINEARMODELS_COV_TYPES`のパラメトライズには含められず、専用テストとして確認する
    （OLS`test_ols_validation.py::test_scale_variance_cluster_raises_
    computation_error`・WLS`test_wls_validation.py`の同名テストと同じ理由）。
    均等な疑似グループ（行番号%10、
    `G=10`、各グループ50件）を使う——第一段階回帰の`q`（`x_exog`2列+
    `instruments`2列=4）より十分大きく、
    `test_cluster_count_at_most_slopes_raises_validation_error`が
    確認するクラスター数不足の`ValidationError`（第一段階の`q=3`〔`x_exog`
    〔x1〕+`instruments`〔z1,z2〕〕相当）ではなく、傾き係数の共分散部分行列の
    条件数超過による`ComputationError`（`MATRIX_NEAR_SINGULAR`、
    `matrix="COEFFICIENT_COVARIANCE"`、`stage="FIRST_STAGE"`）が発生することを
    確認する。
    """
    df = pl.read_csv(DATA_DIR / "iv_scale_variance.csv")
    df = with_cluster_groups(df, 10)
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR,
        matrix="COEFFICIENT_COVARIANCE",
        stage="FIRST_STAGE",
        endog_name="endog1",
    ):
        IV(
            df,
            y="y",
            x_exog=["x1", "x2"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
            options=options,
        ).fit()


def test_gmm_raise_on_non_convergence_true_raises_computation_error(
    iv_dataset,
):
    """厳しすぎる`gmm_tol`で最大反復回数内に収束しない場合、既定
    （`raise_on_non_convergence=True`）では`ComputationError`（`MleError.
    NonConvergence`と同じ分類、`engine_pybind/src/iv/common.rs`参照）。
    """
    options = IVOptions(
        estimator="gmm",
        gmm_type="iterated",
        gmm_weight_type="robust",
        gmm_tol=1e-300,
        gmm_max_iter=3,
    )
    with raises_error(
        ErrorCode.NOT_CONVERGED, n_iter=3, iter_option="gmm_max_iter"
    ):
        our_fit(iv_dataset, options=options)


# ── 例外クラスの継承 ──────────────────────────────────────────────────


def test_validation_error_is_value_error():
    """ValidationErrorがValueErrorのサブクラスであること。

    素の`except ValueError`でも捕まえられる
    （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
    """
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    """ComputationErrorがRuntimeErrorのサブクラスであること。"""
    assert issubclass(ComputationError, RuntimeError)
