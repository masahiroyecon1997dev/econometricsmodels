"""Resultsクラスのdocstringが、公開プロパティ・メソッドを漏れなく挙げることのテスト。

各`*Results`クラスのdocstringには「Fields by purpose」として、目的別に
プロパティ名とメソッド名を並べている（APIページで全項目を読まずに目的の
項目を探せるようにするため）。フィールドを足したのにdocstringへ足し忘れる
こと、消したのに残すことを検出する。
"""

from __future__ import annotations

import inspect
import re

import econometricsmodels
import pytest

RESULTS = [
    name for name in econometricsmodels.__all__ if name.endswith("Results")
]


# 項目名ではなく、説明文の中で名前を挙げているだけの語。
NOT_MEMBERS = {"const", "sigma", "fit", "target"}


def _public_members(cls) -> set[str]:
    return {
        name for name, _ in inspect.getmembers(cls) if not name.startswith("_")
    }


def _mentioned(doc: str) -> set[str]:
    return set(re.findall(r"`([a-z_0-9]+)(?:\(\))?`", doc))


@pytest.mark.parametrize("name", RESULTS)
def test_every_public_member_is_listed_in_the_class_docstring(name):
    cls = getattr(econometricsmodels, name)
    doc = cls.__doc__ or ""
    assert "Fields by purpose" in doc
    missing = _public_members(cls) - _mentioned(doc)
    assert not missing, f"{name} docstring does not list: {sorted(missing)}"


@pytest.mark.parametrize("name", RESULTS)
def test_class_docstring_lists_only_existing_members(name):
    cls = getattr(econometricsmodels, name)
    section = (cls.__doc__ or "").split("Fields by purpose")[1]
    section = section.split("Args:")[0]
    unknown = {
        m
        for m in _mentioned(section)
        if m not in _public_members(cls) and m not in NOT_MEMBERS
    }
    assert not unknown, f"{name} docstring lists unknown: {sorted(unknown)}"
