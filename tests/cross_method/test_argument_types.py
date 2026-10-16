"""全手法共通の、列名引数（`y`・`x`・`entity`等）の型検証のテスト。

型の誤り（`x`が`list`でない、列名が`str`でない）は、`ValidationError`ではなく
Pythonの慣習どおり組み込みの`TypeError`になり、メッセージに引数名と実際の型が含まれる。
値の誤り（存在しない列名等）は各手法の`test_<手法>_validation.py`が`ValidationError`で扱う。
"""

from __future__ import annotations

import re

import _error_messages as msgs
import numpy as np
import polars as pl
import pytest
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    FEOptions,
    Logit,
    Probit,
    Tobit,
    ValidationError,
)

N = 60


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    x1 = rng.normal(0.0, 1.0, N)
    x2 = rng.normal(0.0, 1.0, N)
    y = 1.0 + x1 - 0.5 * x2 + rng.normal(0.0, 1.0, N)
    return pl.DataFrame(
        {
            "y": y,
            "y01": (y > np.median(y)).astype(np.float64),
            "x1": x1,
            "x2": x2,
            "z": x1 + rng.normal(0.0, 1.0, N),
            "w": np.ones(N),
            "entity": np.arange(N) // 6,
            "t": np.arange(N) % 6,
        }
    )


# 引数の既定値を使うことを表す番兵（`None`も「`None`を渡す」テスト値として使うため）。
_UNSET = object()


def _or(value, default):
    return default if value is _UNSET else value


# 手法ごとに、引数を差し替えて推定量を構築する関数。`x`は手法の説明変数引数
# （IVでは`x_exog`）、`y`・`entity`・`weight`等を上書きできる。
def _build_ols(df, y="y", x=_UNSET):
    return OLS(df, y=y, x=_or(x, ["x1"]))


def _build_wls(df, y="y", x=_UNSET, weight="w"):
    return WLS(df, y=y, x=_or(x, ["x1"]), weight=weight)


def _build_logit(df, y="y01", x=_UNSET):
    return Logit(df, y=y, x=_or(x, ["x1"]))


def _build_probit(df, y="y01", x=_UNSET):
    return Probit(df, y=y, x=_or(x, ["x1"]))


def _build_tobit(df, y="y", x=_UNSET):
    return Tobit(df, y=y, x=_or(x, ["x1"]))


def _build_fe(df, y="y", x=_UNSET, entity="entity"):
    return FE(df, y=y, x=_or(x, ["x1"]), entity=entity)


def _build_re(df, y="y", x=_UNSET, entity="entity"):
    return RE(df, y=y, x=_or(x, ["x1"]), entity=entity)


def _build_iv(df, y="y", x=_UNSET, x_endog=_UNSET, instruments=_UNSET):
    return IV(
        df,
        y=y,
        x_exog=_or(x, ["x2"]),
        x_endog=_or(x_endog, ["x1"]),
        instruments=_or(instruments, ["z"]),
    )


# 手法と、エラーメッセージ上の説明変数の引数名（IVは`x_exog`）。
X_METHODS = {
    "OLS": (_build_ols, "x"),
    "WLS": (_build_wls, "x"),
    "Logit": (_build_logit, "x"),
    "Probit": (_build_probit, "x"),
    "Tobit": (_build_tobit, "x"),
    "FE": (_build_fe, "x"),
    "RE": (_build_re, "x"),
    "IV": (_build_iv, "x_exog"),
}


@pytest.fixture(params=list(X_METHODS), ids=list(X_METHODS))
def x_method(request):
    return X_METHODS[request.param]


def _msg_list(name: str, got: str) -> str:
    return re.escape(
        f"'{name}' must be a list of column names (e.g. {name}=[\"x1\"]), "
        f"got {got}"
    )


def _msg_name(name: str, got: str) -> str:
    return re.escape(f"'{name}' must be a str column name, got {got}")


def _assert_type_error_at_fit(build, match):
    """構築は成功し（何も検査しない）、`fit()`が`TypeError`を送出する。

    値の検証と同じく、列名引数の型の検証も`fit()`で行う。`ValidationError`
    ではなく組み込みの`TypeError`であることも確認する。
    """
    model = build()  # 構築時には例外にならない
    with pytest.raises(TypeError, match=match) as info:
        model.fit()
    assert not isinstance(info.value, ValidationError)


SERIES = pl.Series(["x1"])


@pytest.mark.parametrize(
    ("value", "got"),
    [
        ("x1", "str"),
        (("x1",), "tuple"),
        (SERIES, msgs.fully_qualified_type_name(SERIES)),
        (None, "NoneType"),
        ({"x1"}, "set"),
    ],
    ids=["str", "tuple", "polars_series", "none", "set"],
)
def test_x_must_be_a_list(df, x_method, value, got):
    """`x`が`list`でないと、引数名・期待する形・実際の型を含む`TypeError`になる。"""
    builder, arg_name = x_method

    _assert_type_error_at_fit(
        lambda: builder(df, x=value), _msg_list(arg_name, got)
    )


@pytest.mark.parametrize(
    ("value", "position", "got"),
    [
        ([1], 0, "int"),
        (["x1", 2.5], 1, "float"),
        ([None], 0, "NoneType"),
        ([b"x1"], 0, "bytes"),
    ],
    ids=["int", "float_second", "none", "bytes"],
)
def test_x_elements_must_be_str(df, x_method, value, position, got):
    """`x`の要素が`str`でないと、要素の位置を含む`TypeError`になる。"""
    builder, arg_name = x_method

    _assert_type_error_at_fit(
        lambda: builder(df, x=value),
        re.escape(
            f"'{arg_name}[{position}]' must be a str column name, got {got}"
        ),
    )


# 列名を受け取る単一の`str`引数: `y`（全手法）、`entity`（FE・RE）、`weight`（WLS）。
BAD_NAMES = [
    (["y"], "list"),
    (None, "NoneType"),
    (1, "int"),
    (b"y", "bytes"),
]


@pytest.mark.parametrize(
    ("value", "got"), BAD_NAMES, ids=[x[1] for x in BAD_NAMES]
)
def test_y_must_be_a_str(df, x_method, value, got):
    """`y`が`str`でないと、引数名と実際の型を含む`TypeError`になる。"""
    builder, _ = x_method

    _assert_type_error_at_fit(
        lambda: builder(df, y=value), _msg_name("y", got)
    )


@pytest.mark.parametrize(
    ("value", "got"), BAD_NAMES, ids=[x[1] for x in BAD_NAMES]
)
@pytest.mark.parametrize("builder", [_build_fe, _build_re], ids=["FE", "RE"])
def test_entity_must_be_a_str(df, builder, value, got):
    _assert_type_error_at_fit(
        lambda: builder(df, entity=value), _msg_name("entity", got)
    )


@pytest.mark.parametrize(
    ("value", "got"), BAD_NAMES, ids=[x[1] for x in BAD_NAMES]
)
def test_weight_must_be_a_str(df, value, got):
    _assert_type_error_at_fit(
        lambda: _build_wls(df, weight=value), _msg_name("weight", got)
    )


@pytest.mark.parametrize("name", ["x_endog", "instruments"])
@pytest.mark.parametrize(
    ("value", "got"),
    [
        ("z", "str"),
        (("z",), "tuple"),
        (SERIES, msgs.fully_qualified_type_name(SERIES)),
        (None, "NoneType"),
        ({"z"}, "set"),
    ],
    ids=["str", "tuple", "polars_series", "none", "set"],
)
def test_iv_column_lists_must_be_lists(df, name, value, got):
    """IVの`x_endog`・`instruments`も`list`のみを受け付ける。"""
    _assert_type_error_at_fit(
        lambda: _build_iv(df, **{name: value}), _msg_list(name, got)
    )


@pytest.mark.parametrize("name", ["x_endog", "instruments"])
@pytest.mark.parametrize(
    ("value", "position", "got"),
    [([1], 0, "int"), (["z", 2.5], 1, "float"), ([b"z"], 0, "bytes")],
    ids=["int", "float_second", "bytes"],
)
def test_iv_column_list_elements_must_be_str(df, name, value, position, got):
    _assert_type_error_at_fit(
        lambda: _build_iv(df, **{name: value}),
        re.escape(
            f"'{name}[{position}]' must be a str column name, got {got}"
        ),
    )


def test_list_arguments_still_work(df):
    """正しい`list`・`str`の引数は従来どおり推定できる（回帰確認）。"""
    assert _build_ols(df, x=["x1", "x2"]).fit().n_obs == N
    assert FE(df, y="y", x=["x1"], entity="entity").fit().n_obs == N
    assert (
        FE(
            df,
            y="y",
            x=["x1"],
            entity="entity",
            options=FEOptions(time="t"),
        )
        .fit()
        .n_obs
        == N
    )
    assert (
        IV(df, y="y", x_exog=["x2"], x_endog=["x1"], instruments=["z"])
        .fit()
        .n_obs
        == N
    )
    assert RE(df, y="y", x=["x1"], entity="entity").fit().n_obs == N
