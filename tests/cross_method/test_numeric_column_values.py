"""全手法共通の、数値列の欠損値（null）・NaN・無限大のテスト。

数値として使う列は手法・ロールごとに別の抽出箇所を通り、複数列を順に処理する
ループもある。先頭の列だけを調べるループや、特定の列だけ検査を飛ばす経路を見逃さない
ように、各手法が使う数値列の全てに、nullとNaNと±infを入れて確認する（`fit()`の
`y`・`x`・IVの各変数、`predict()`・`augment()`の`new_data`の全ての`x`列）。
キー列（`cluster`・`entity`・`time`）は`test_key_columns.py`が担う。
"""

from __future__ import annotations

import _error_messages as msgs
import polars as pl
import pytest
from _dtype_helpers import make_frame
from _error_messages import escaped
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    Logit,
    Probit,
    Tobit,
    ValidationError,
)

# 壊す行（`fit()`の入力、`new_data`のどちらでも同じ位置で報告される）。
ROW = 1


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return make_frame()


# 壊し方 → メッセージ。nullは件数、NaN・無限大は値と行番号を報告する。
BAD_VALUES = [
    pytest.param(None, None, id="null"),
    pytest.param(float("nan"), "NaN", id="NaN"),
    pytest.param(float("inf"), "inf", id="inf"),
    pytest.param(float("-inf"), "-inf", id="-inf"),
]


def _break(frame: pl.DataFrame, column: str, bad: float | None, row: int):
    """`column`の`row`行目だけを`bad`（`None`ならnull）に置き換える。"""
    return frame.with_columns(
        frame[column].cast(pl.Float64).scatter(row, bad).alias(column)
    )


def _assert_rejected(call, column: str, shown: str | None, row: int) -> None:
    if shown is None:
        message = escaped(msgs.COLUMN_HAS_MISSING_VALUES, name=column, count=1)
    else:
        message = escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE,
            name=column,
            value=shown,
            row=row,
        )
    with pytest.raises(ValidationError, match=message):
        call()


# ── fit(): 使う数値列の全て ───────────────────────────────────────────

# 手法 → (`fit()`、使う数値列)。複数の`x`を渡し、後ろの列も確かめる。重み（WLS）は
# 0以下・null・NaNを重み専用のメッセージで拒否するため、ここには含めない
# （`test_wls_validation.py`の`test_non_positive_weight_raises`・`test_nan_weight_raises`・
# `test_null_weight_raises`が担う）。
FIT_CASES = {
    "OLS": (
        lambda d: OLS(d, "y", ["x1", "x2"]).fit(),
        ["y", "x1", "x2"],
    ),
    "WLS": (
        lambda d: WLS(d, "y", ["x1", "x2"], "w").fit(),
        ["y", "x1", "x2"],
    ),
    "Logit": (
        lambda d: Logit(d, "y01", ["x1", "x2"]).fit(),
        ["y01", "x1", "x2"],
    ),
    "Probit": (
        lambda d: Probit(d, "y01", ["x1", "x2"]).fit(),
        ["y01", "x1", "x2"],
    ),
    "Tobit": (
        lambda d: Tobit(d, "yc", ["x1", "x2"]).fit(),
        ["yc", "x1", "x2"],
    ),
    # IVは役割ごとに別の抽出箇所を通る（被説明変数・内生・外生・操作変数）。
    "IV": (
        lambda d: IV(d, "y", ["x2"], ["x1"], ["z", "z2"]).fit(),
        ["y", "x2", "x1", "z", "z2"],
    ),
    "FE": (
        lambda d: FE(d, "y", ["x1", "x2"], "entity").fit(),
        ["y", "x1", "x2"],
    ),
    "RE": (
        lambda d: RE(d, "y", ["x1", "x2"], "entity").fit(),
        ["y", "x1", "x2"],
    ),
}

FIT_COLUMN_CASES = [
    (method, column)
    for method, (_, columns) in FIT_CASES.items()
    for column in columns
]


@pytest.mark.parametrize(("bad", "shown"), BAD_VALUES)
@pytest.mark.parametrize(
    ("method", "column"),
    FIT_COLUMN_CASES,
    ids=[f"{method}-{column}" for method, column in FIT_COLUMN_CASES],
)
def test_bad_value_in_any_used_column_raises_in_fit(
    frame, method, column, bad, shown
):
    """使う数値列のどれにnull・NaN・無限大があっても、その列名で拒否する。"""
    fit, _ = FIT_CASES[method]
    df = _break(frame, column, bad, ROW)

    _assert_rejected(lambda: fit(df), column, shown, ROW)


@pytest.mark.parametrize("method", list(FIT_CASES))
def test_fit_succeeds_when_no_value_is_broken(frame, method):
    """対照: 同じ`fit()`でも壊さなければ成功する（上のテストの失敗は壊した値による）。"""
    fit, _ = FIT_CASES[method]

    fit(frame)


# ── predict()・augment(): new_dataの全てのx列 ─────────────────────────

# 手法 → `fit()`済みの結果（`x=["x1", "x2"]`）。`predict()`・`augment()`を持つ手法。
FITTED = {
    "OLS": lambda d: OLS(d, "y", ["x1", "x2"]).fit(),
    "WLS": lambda d: WLS(d, "y", ["x1", "x2"], "w").fit(),
    "Logit": lambda d: Logit(d, "y01", ["x1", "x2"]).fit(),
    "Probit": lambda d: Probit(d, "y01", ["x1", "x2"]).fit(),
    "Tobit": lambda d: Tobit(d, "yc", ["x1", "x2"]).fit(),
}

NEW_DATA_CASES = [
    (method, column, call)
    for method in FITTED
    for column in ["x1", "x2"]
    for call in ["predict", "augment"]
]


@pytest.mark.parametrize(("bad", "shown"), BAD_VALUES)
@pytest.mark.parametrize(
    ("method", "column", "call"),
    NEW_DATA_CASES,
    ids=[
        f"{method}-{column}-{call}" for method, column, call in NEW_DATA_CASES
    ],
)
def test_bad_value_in_any_new_data_column_raises(
    frame, method, column, call, bad, shown
):
    """`new_data`のどの`x`列にnull・NaN・無限大があっても、その列名で拒否する。

    先頭の`x1`だけを調べるループは、`x2`の値を見逃す。
    """
    res = FITTED[method](frame)
    new_data = _break(frame.select("x1", "x2").head(5), column, bad, ROW)

    _assert_rejected(
        lambda: getattr(res, call)(new_data=new_data), column, shown, ROW
    )


def test_new_data_with_valid_values_is_accepted(frame):
    """対照: 同じ`new_data`でも壊さなければ、全手法の`predict()`・`augment()`が通る。"""
    new_data = frame.select("x1", "x2").head(5)

    for fit in FITTED.values():
        res = fit(frame)
        assert len(res.predict(new_data=new_data)) == 5
        assert res.augment(new_data=new_data).height == 5
