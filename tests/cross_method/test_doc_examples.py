"""公開ドキュメントのPythonコード例が、そのまま実行できることのテスト。

`getting-started.md`・`index.md`・`usage/`の各ページと`guide/pickling.md`は、ページごとに
上から順にコードブロックを実行する（ブロック間で名前空間を共有する）。
利用者がコピーして試す例なので、データの不備（例: Logitで完全分離になる
データ）や、書き換え漏れで動かない例を検出する。他のページに置いた列を
前提にする例は書かず、各ページで自己完結させる。

ワーカープロセスを起動する例は、`exec`では動かせない（`__main__`のガードと、ワーカーが
関数をimportできることが要る）。そのブロックの直前の行に`<!-- doc-test: subprocess -->`
（HTMLコメント。公開ページには表示されない）を置くと、そのブロックだけで完結した
スクリプトとして別のPythonプロセスで実行する（前のブロックの名前は使えない）。
実行しないブロックは作らない（テストはスキップしない方針）。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
PAGES = sorted(
    [DOCS / "index.md", DOCS / "getting-started.md"]
    + [DOCS / "guide" / "pickling.md"]
    + list((DOCS / "usage").glob("*.md"))
)
# 直前の行にこのコメントを置いたブロックは、別プロセスで実行する。
CODE_BLOCK = re.compile(
    r"(?P<subprocess><!-- doc-test: subprocess -->\n)?```python\n(?P<code>.*?)```",
    re.DOTALL,
)
SUBPROCESS_TIMEOUT_SECONDS = 120


def _run_in_subprocess(page: Path, index: int, code: str, workdir: Path):
    script = workdir / f"{page.stem}_{index}.py"
    script.write_text(code, encoding="utf-8")
    try:
        done = subprocess.run(
            [sys.executable, "-I", str(script)],
            cwd=workdir,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        pytest.fail(
            f"{page.name}: code block {index} did not finish within "
            f"{SUBPROCESS_TIMEOUT_SECONDS}s in a subprocess:\n{stderr[-2000:]}"
        )
    if done.returncode != 0:
        pytest.fail(
            f"{page.name}: code block {index} failed in a subprocess "
            f"(exit code {done.returncode}):\n{done.stderr[-2000:]}"
        )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_python_examples_run(page, capsys, tmp_path):
    namespace: dict = {}
    text = page.read_text(encoding="utf-8")
    in_subprocess = 0
    for index, match in enumerate(CODE_BLOCK.finditer(text)):
        block = match.group("code")
        if match.group("subprocess"):
            in_subprocess += 1
            _run_in_subprocess(page, index, block, tmp_path)
            continue
        try:
            code = compile(block, f"{page.name}[{index}]", "exec")
            exec(code, namespace)  # noqa: S102
        except Exception as error:  # noqa: BLE001
            pytest.fail(
                f"{page.name}: code block {index} failed: "
                f"{type(error).__name__}: {error}"
            )
    # コメントの直後に空行を入れる、綴りを間違える等で、印がどのブロックにも結び付かないと、
    # そのブロックは黙ってこのプロセスで`exec`される。印の数と別プロセスで走らせた数を揃える。
    assert text.count("doc-test: subprocess") == in_subprocess, (
        f"{page.name}: a doc-test marker is not directly followed by a code block"
    )
    capsys.readouterr()
