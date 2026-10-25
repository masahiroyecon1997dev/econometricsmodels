"""公開サイト（mkdocs）に出る文章が英語で、内部ドキュメントを指していないことのテスト。

公開サイトに載るのは英語のページ（`docs/`直下の`index.md`・`getting-started.md`と
`usage/`・`api/`・`guide/`・`methods/`）と、APIページがdocstringから描画するpublicなクラス・メソッドの
docstring。日本語の内部ドキュメント（`docs/spec/`等）は`exclude_docs`で公開から外しており、
そこへの参照や日本語が公開側に混ざると、公開サイトで読めない参照や日本語が出る。
定義式の正本は`docs/methods/`で、`docs/spec/`には重複させない（CLAUDE.md 13章）。
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import econometricsmodels
import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
JAPANESE = re.compile(r"[぀-ヿ一-鿿＀-￯]")
# 公開側から内部ドキュメント・開発者向けファイルを指す参照。
INTERNAL_REF = re.compile(
    r"docs/(?:spec|planning|performance)|\.claude/|CLAUDE\.md|"
    r"engine_pybind/|engine/src"
)

PUBLIC_PAGES = sorted(
    [DOCS / "index.md", DOCS / "getting-started.md"]
    + [
        p
        for d in ("usage", "api", "guide", "methods")
        for p in (DOCS / d).glob("*.md")
    ]
)


def _public_docstrings():
    """公開API（`__all__`）のクラスと、そのpublicメンバーのdocstring。"""
    for name in econometricsmodels.__all__:
        obj = getattr(econometricsmodels, name)
        if not inspect.isclass(obj):
            continue
        yield name, obj.__doc__
        for member, value in inspect.getmembers(obj):
            if not member.startswith("_"):
                yield f"{name}.{member}", getattr(value, "__doc__", None)


@pytest.mark.parametrize("page", PUBLIC_PAGES, ids=lambda p: p.name)
def test_public_page_is_english(page):
    text = page.read_text(encoding="utf-8")
    assert not JAPANESE.search(text), f"{page.name} contains Japanese text"
    assert not re.search(r"docs/(?:spec|planning)", text), (
        f"{page.name} points to an internal document"
    )


def test_internal_documents_are_not_published():
    config = (DOCS / "mkdocs.yml").read_text(encoding="utf-8")
    block = config.split("exclude_docs:")[1].split("\n\n")[0]
    for excluded in ("spec/", "performance/", "planning/"):
        assert excluded in block


def test_methods_pages_are_in_nav():
    config = (DOCS / "mkdocs.yml").read_text(encoding="utf-8")
    for page in (DOCS / "methods").glob("*.md"):
        assert f"methods/{page.name}" in config


def test_public_docstrings_are_english_without_internal_refs():
    problems = []
    for qualname, doc in _public_docstrings():
        if not doc:
            continue
        if JAPANESE.search(doc) or INTERNAL_REF.search(doc):
            problems.append(qualname)
    assert not problems, f"internal refs or Japanese in: {problems}"
