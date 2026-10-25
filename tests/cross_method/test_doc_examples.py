"""公開ドキュメントのPythonコード例が、そのまま実行できることのテスト。

`getting-started.md`・`index.md`・`usage/`の各ページは、ページごとに
上から順にコードブロックを実行する（ブロック間で名前空間を共有する）。
利用者がコピーして試す例なので、データの不備（例: Logitで完全分離になる
データ）や、書き換え漏れで動かない例を検出する。他のページに置いた列を
前提にする例は書かず、各ページで自己完結させる。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
PAGES = sorted(
    [DOCS / "index.md", DOCS / "getting-started.md"]
    + list((DOCS / "usage").glob("*.md"))
)
CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_python_examples_run(page, capsys):
    namespace: dict = {}
    for index, block in enumerate(
        CODE_BLOCK.findall(page.read_text(encoding="utf-8"))
    ):
        try:
            code = compile(block, f"{page.name}[{index}]", "exec")
            exec(code, namespace)  # noqa: S102
        except Exception as error:  # noqa: BLE001
            pytest.fail(
                f"{page.name}: code block {index} failed: "
                f"{type(error).__name__}: {error}"
            )
    capsys.readouterr()
