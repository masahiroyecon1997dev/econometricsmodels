"""Rustのpanicを境界で`INTERNAL_ERROR`（`ComputationError`）にするテスト。

panicはそのままだとPyO3が`PanicException`（`BaseException`の子）にして、`except Exception`で
捕まえられない。境界の関数は`guard`で包んであり、アプリが通常のエラーとして扱える。
`_lib.panic_for_test`は本番の境界と同じ`guard`の中でpanicを起こす非公開の入口で、`guard`自体の
動作（panicの変換・`detail`・他のエラーの素通し）を確かめる。個々の境界の関数が`guard`で
包まれていること自体は、このテストでは保証しない（`docs/spec/error-handling.md`
「panicの境界変換」）。panicの場所（ファイルと行）は標準エラー出力に出るが、
捕捉されたテストでは通常見えない。
"""

from __future__ import annotations

import pytest
from _errors import raises_error
from econometricsmodels import ComputationError, ErrorCode, _lib

KINDS = ["str", "string", "other", "long"]

EXPECTED_DETAILS = {
    "str": "panic: deliberate panic for a test",
    "string": "panic: deliberate panic for a test: 6",
    "other": "panic: a panic with a payload that is not text",
    "long": f"panic: {'x' * 200} (truncated)",
}


# ── guard（panic_for_test） ──────────────────────────────────────────


@pytest.mark.parametrize("kind", KINDS)
def test_a_panic_becomes_an_internal_error(kind):
    """`RuntimeError`の系統なので、アプリは`except RuntimeError`でも捕まえられる。"""
    with raises_error(ErrorCode.INTERNAL_ERROR) as info:
        _lib.panic_for_test(kind)
    assert isinstance(info.value, ComputationError)
    assert isinstance(info.value, RuntimeError)


@pytest.mark.parametrize("kind", KINDS)
def test_the_panic_does_not_escape_as_a_base_exception(kind):
    """`PanicException`は`BaseException`の子で`except Exception`をすり抜ける。
    すり抜けず、`Exception`の子として捕まることを直接確かめる。"""
    with pytest.raises(BaseException) as info:
        _lib.panic_for_test(kind)
    assert isinstance(info.value, Exception)
    assert type(info.value).__name__ != "PanicException"
    assert info.value.__cause__ is None


@pytest.mark.parametrize("kind", KINDS)
def test_the_detail_carries_the_panic_message(kind):
    with raises_error(ErrorCode.INTERNAL_ERROR) as info:
        _lib.panic_for_test(kind)
    assert info.value.params["detail"] == EXPECTED_DETAILS[kind]


def test_the_truncated_detail_is_part_of_the_message():
    with raises_error(ErrorCode.INTERNAL_ERROR) as info:
        _lib.panic_for_test("long")
    assert f"{'x' * 200} (truncated). This should not happen." in str(
        info.value
    )


def test_guard_passes_a_non_panic_error_through():
    """panicではないエラー（ここでは未知の`kind`への`ValueError`）は`INTERNAL_ERROR`に
    変換されず、そのまま送出される。"""
    with pytest.raises(ValueError, match="unknown kind"):
        _lib.panic_for_test("no such kind")
