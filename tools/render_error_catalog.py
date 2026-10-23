r"""エラーカタログから、コミットする生成物を作る（`--check`で差分を検出する）。

カタログの正本はRustの`engine/src/shared/error_catalog.rs`で、ビルドした拡張の
`econometricsmodels._lib.error_catalog()`から読む。生成物は2つ。

1. `python_package/econometricsmodels/error_codes.py`: `ErrorCode`（`StrEnum`）。全体が生成物。
2. `docs/guide/error-codes.md`の生成ブロック（`code`ごとのクラス・`params`・`message`/`hint`の
   テンプレートの表）。ブロックの外の説明文は手書き。

使用例（リポジトリルートから。先に`maturin develop`で拡張をビルドしておく）:
    python tools/render_error_catalog.py            # 生成物を書き換える
    python tools/render_error_catalog.py --check    # 差分があれば終了コード1
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
ERROR_CODES_PATH = (
    REPO_ROOT / "python_package" / "econometricsmodels" / "error_codes.py"
)
DOC_PATH = REPO_ROOT / "docs" / "guide" / "error-codes.md"

BLOCK_BEGIN = (
    "<!-- BEGIN GENERATED CATALOG (tools/render_error_catalog.py) -->"
)
BLOCK_END = "<!-- END GENERATED CATALOG -->"

PACKAGE_DIR = REPO_ROOT / "python_package" / "econometricsmodels"

_LINE_LENGTH = 79

_TYPE_LABELS = {
    "int": "int",
    "float": "float",
    "float_or_none": "float or None",
    "str": "str",
    "str_list": "list of str",
    "column_list": "list of str (column names)",
    "bool": "bool",
}


def _load_lib() -> ModuleType:
    """ビルド済みの拡張`_lib`を、パッケージの`__init__`を経由せずに読む。

    `__init__`は生成物の`error_codes.py`を読み込むため、その生成物が無い・壊れて
    いる状態でもこのスクリプトで復旧できるようにする。すでにimport済みならそれを使う。
    """
    loaded = sys.modules.get("econometricsmodels._lib")
    if loaded is not None:
        return loaded
    candidates = sorted(PACKAGE_DIR.glob("_lib.*"))
    for path in candidates:
        if path.suffix in {".so", ".pyd"}:
            spec = importlib.util.spec_from_file_location(
                "econometricsmodels._lib", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit(
        f"built extension not found in {PACKAGE_DIR}; run `maturin develop`"
    )


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_error_codes(catalog: dict[str, Any]) -> str:
    """`error_codes.py`の全文を作る。"""
    lines = [
        '"""Error codes of `ValidationError` and `ComputationError`.',
        "",
        "Generated from the error catalog by `tools/render_error_catalog.py`;",
        "do not edit by hand.",
        '"""',
        "",
        "from enum import StrEnum",
        "",
        "",
        "class ErrorCode(StrEnum):",
        '    """Every `code` an exception of this package can carry.',
        "",
        "    Members are strings, so `err.code == ErrorCode.COLUMN_NOT_FOUND` and",
        '    `err.code == "COLUMN_NOT_FOUND"` are both true.',
        '    """',
        "",
    ]
    for c in catalog["codes"]:
        code = c["code"]
        line = f'    {code} = "{code}"'
        if len(line) > _LINE_LENGTH:
            # ruff formatの出力と一致させる（生成物がformat検査で差分にならないように）。
            lines += [f"    {code} = (", f'        "{code}"', "    )"]
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    """Markdownの表のセルに入れる文字列にする。"""
    for ch in ("\\", "|", "<", "*"):
        text = text.replace(ch, "\\" + ch)
    return text


def _params_cell(params: list[dict], *, stage_allowed: bool) -> str:
    items = []
    for p in params:
        label = _TYPE_LABELS[p["type"]]
        note = "" if p["in_text"] else ", not shown in the text"
        items.append(f"`{p['name']}` ({label}{note})")
    if stage_allowed:
        items.append("`stage` (optional, see Stages)")
    return "<br>".join(items) if items else "-"


def _class_table(catalog: dict[str, Any], class_name: str) -> list[str]:
    lines = [
        "| Code | Parameters | Message | Hint |",
        "|---|---|---|---|",
    ]
    for c in catalog["codes"]:
        if c["class"] != class_name:
            continue
        hint = _cell(c["hint"]) if c["hint"] else "-"
        params = _params_cell(c["params"], stage_allowed=c["stage_allowed"])
        lines.append(
            f"| `{c['code']}` | {params} | {_cell(c['message'])} | {hint} |"
        )
    return lines


def render_doc_block(catalog: dict[str, Any]) -> str:
    """`error-codes.md`の生成ブロック（開始・終了の印を含む）を作る。"""
    lines = [BLOCK_BEGIN, ""]
    lines += ["### ValidationError", ""]
    lines += _class_table(catalog, "ValidationError")
    lines += ["", "### ComputationError", ""]
    lines += _class_table(catalog, "ComputationError")
    lines += [
        "",
        "### Stages",
        "",
        (
            "When an error happens inside a nested regression, `params` also "
            "holds `stage` (one of the identifiers below, plus the extra "
            "parameters listed) and the message starts with the phrase shown."
        ),
        "",
        "| Stage | Extra parameters | Phrase |",
        "|---|---|---|",
    ]
    for s in catalog["stages"]:
        extra = (
            "<br>".join(f"`{p}`" for p in s["params"]) if s["params"] else "-"
        )
        lines.append(
            f"| `{s['name']}` | {extra} | {_cell(s['prefix'].removesuffix(': '))} |"
        )
    lines += [
        "",
        "### Test names",
        "",
        "| `test` | Name used in the message |",
        "|---|---|",
    ]
    lines += [
        f"| `{k}` | {_cell(v)} |" for k, v in catalog["test_labels"].items()
    ]
    lines += ["", BLOCK_END]
    return "\n".join(lines)


def splice_block(doc: str, block: str) -> str:
    """`doc`の生成ブロックを`block`に差し替える。"""
    begin = doc.find(BLOCK_BEGIN)
    end = doc.find(BLOCK_END)
    if begin == -1 or end == -1 or end < begin:
        raise ValueError("generated block markers not found")
    return doc[:begin] + block + doc[end + len(BLOCK_END) :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="書き換えず、生成物が最新でなければ終了コード1にする",
    )
    args = parser.parse_args(argv)

    catalog = _load_lib().error_catalog()
    try:
        doc = splice_block(
            DOC_PATH.read_text(encoding="utf-8"), render_doc_block(catalog)
        )
    except (OSError, ValueError) as exc:
        print(f"cannot update {_display(DOC_PATH)}: {exc}", file=sys.stderr)
        return 1
    targets = {
        ERROR_CODES_PATH: render_error_codes(catalog),
        DOC_PATH: doc,
    }
    stale = [
        path
        for path, text in targets.items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    if args.check:
        if stale:
            names = ", ".join(_display(p) for p in stale)
            print(
                f"out of date: {names}\n"
                "run `python tools/render_error_catalog.py` and commit",
                file=sys.stderr,
            )
            return 1
        print("error catalog outputs are up to date")
        return 0
    for path in stale:
        path.write_text(targets[path], encoding="utf-8", newline="\n")
        print(f"wrote {_display(path)}")
    if not stale:
        print("already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
