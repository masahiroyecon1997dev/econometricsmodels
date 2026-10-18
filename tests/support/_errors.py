"""構造化例外（`ValidationError`/`ComputationError`）を`code`と`params`で検査するヘルパー。

新規・更新するテストは、文面ではなく`code`と`params`で照合する（`message`/`hint`の
言い回しを直してもテストが壊れないようにするため）。文面そのものはRust側の単体テストと
`tests/cross_method/test_error_catalog.py`が固定する。

`raises_error`は検査した`code`を記録し、`--check-error-coverage`を付けた全体実行の終了時に
`MIGRATED_CODES`の全`code`が1度は送出されたことを`conftest.py`が検査する
（一度も送出されない`code`は、死んだ定義かテストの抜け）。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from econometricsmodels import (
    ComputationError,
    ErrorCode,
    ValidationError,
    _lib,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_CATALOG = {c["code"]: c for c in _lib.error_catalog()["codes"]}
_STAGES = {s["name"]: s for s in _lib.error_catalog()["stages"]}
_CLASSES = {
    "ValidationError": ValidationError,
    "ComputationError": ComputationError,
}

# `raises_error`が検査して通った`code`（セッション中に積み上がる）。
RAISED_CODES: set[str] = set()

# 実際のエラー経路から送出されることをテストで保証済みの`code`。エラーの移行が進んだ
# 段階で、その`code`をここに足す（全体実行の終了時に、足した全`code`が1度は
# `raises_error`で検査されたかを調べる）。全`code`の移行が終わったら、カタログの
# 全`code`を対象にする。
MIGRATED_CODES: frozenset[str] = frozenset()


def unraised_migrated_codes() -> list[str]:
    """`MIGRATED_CODES`のうち、まだ`raises_error`で検査されていない`code`。"""
    return sorted(MIGRATED_CODES - RAISED_CODES)


def _expected_param_names(code: str, params: dict[str, object]) -> set[str]:
    """`params`のキーとして期待される集合（`stage`があればその補助値も）。"""
    entry = _CATALOG[code]
    names = {p["name"] for p in entry["params"]}
    if "stage" in params:
        assert entry["stage_allowed"], f"{code} cannot carry a stage"
        stage = _STAGES.get(params["stage"])
        assert stage is not None, f"unknown stage {params['stage']!r}"
        names |= {"stage", *stage["params"]}
    return names


def _type_matches(type_name: str, value: object) -> bool:
    match type_name:
        case "int":
            return isinstance(value, int) and not isinstance(value, bool)
        case "float":
            return isinstance(value, float)
        case "float_or_none":
            return value is None or isinstance(value, float)
        case "str":
            return isinstance(value, str)
        case "str_list" | "column_list":
            return isinstance(value, list) and all(
                isinstance(v, str) for v in value
            )
        case "bool":
            return isinstance(value, bool)
    return False


@contextmanager
def raises_error(
    code: ErrorCode | str, **params: object
) -> Iterator[pytest.ExceptionInfo[ValidationError | ComputationError]]:
    """ブロック内で`code`の例外が送出され、`params`が一致することを検査する。

    `code`のカタログ上のクラス（`ValidationError`/`ComputationError`）で
    あること、`params`のキー集合（`stage`があればその補助値も）と各値の型がカタログと
    一致すること、`str(e)`が
    `message`と`hint`の連結で`args`が`(str(e),)`であることも毎回検査する。
    `params`に渡したキーだけ値を比較する（`pytest.approx`も渡せる）。

    Args:
        code: 期待する`code`。
        **params: 期待する`params`の値（一部のキーだけでよい）。

    Yields:
        送出された例外の`pytest.ExceptionInfo`。
    """
    expected = ErrorCode(code)
    with pytest.raises((ValidationError, ComputationError)) as info:
        yield info
    err = info.value
    assert err.code == expected, (
        f"expected code {expected}, got {err.code}: {err}"
    )
    expected_class = _CLASSES[_CATALOG[expected]["class"]]
    assert type(err) is expected_class, (
        f"{expected} should be {expected_class.__name__}, "
        f"got {type(err).__name__}"
    )
    names = _expected_param_names(expected, err.params)
    assert set(err.params) == names, (
        f"{expected}: params keys {sorted(err.params)} do not match the "
        f"catalog (expected {sorted(names)})"
    )
    for spec in _CATALOG[expected]["params"]:
        value = err.params[spec["name"]]
        assert _type_matches(spec["type"], value), (
            f"{expected}: param {spec['name']!r} should be {spec['type']}, "
            f"got {value!r}"
        )
    for name, value in params.items():
        assert name in err.params, f"{expected}: no param {name!r}"
        assert err.params[name] == value, (
            f"{expected}: param {name!r} is {err.params[name]!r}, "
            f"expected {value!r}"
        )
    full = err.message + (f" {err.hint}" if err.hint else "")
    assert str(err) == full
    assert err.args == (full,)
    RAISED_CODES.add(str(expected))
