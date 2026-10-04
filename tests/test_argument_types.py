"""全手法共通の、列名引数（`y`・`x`・`entity`等）の型検証のテスト。

型の誤り（`x`が`list`でない、列名が`str`でない）は、`ValidationError`ではなく
Pythonの慣習どおり組み込みの`TypeError`になり、メッセージに引数名と実際の型が含まれる。
値の誤り（存在しない列名等）は各手法の`test_<手法>_validation.py`が`ValidationError`で扱う。
"""

from __future__ import annotations

import re

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


@pytest.mark.parametrize(
    ("value", "got"),
    [
        ("x1", "str"),
        (("x1",), "tuple"),
        (pl.Series(["x1"]), "polars.series.series.Series"),
        (None, "NoneType"),
        ({"x1"}, "set"),
    ],
    ids=["str", "tuple", "polars_series", "none", "set"],
)
def test_x_must_be_a_list(df, x_method, value, got):
    """`x`が`list`でないと、引数名・期待する形・実際の型を含む`TypeError`になる。"""
    builder, arg_name = x_method

    with pytest.raises(TypeError, match=_msg_list(arg_name, got)) as info:
        builder(df, x=value).fit()

    assert not isinstance(info.value, ValidationError)


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

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"'{arg_name}[{position}]' must be a str column name, got {got}"
        ),
    ):
        builder(df, x=value).fit()


@pytest.mark.parametrize(
    ("value", "got"),
    [(["y"], "list"), (None, "NoneType"), (1, "int"), (b"y", "bytes")],
    ids=["list", "none", "int", "bytes"],
)
def test_y_must_be_a_str(df, x_method, value, got):
    """`y`が`str`でないと、引数名と実際の型を含む`TypeError`になる。"""
    builder, _ = x_method

    with pytest.raises(
        TypeError,
        match=re.escape(f"'y' must be a str column name, got {got}"),
    ):
        builder(df, y=value).fit()


@pytest.mark.parametrize("builder", [_build_fe, _build_re], ids=["FE", "RE"])
def test_entity_must_be_a_str(df, builder):
    with pytest.raises(
        TypeError,
        match=re.escape("'entity' must be a str column name, got list"),
    ):
        builder(df, entity=["entity"]).fit()


def test_weight_must_be_a_str(df):
    with pytest.raises(
        TypeError,
        match=re.escape("'weight' must be a str column name, got list"),
    ):
        _build_wls(df, weight=["w"]).fit()


@pytest.mark.parametrize(
    "name", ["x_endog", "instruments"], ids=["x_endog", "instruments"]
)
def test_iv_column_lists_must_be_lists(df, name):
    """IVの`x_endog`・`instruments`も`list`のみを受け付ける。"""
    with pytest.raises(TypeError, match=_msg_list(name, "str")):
        _build_iv(df, **{name: "z"}).fit()


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
