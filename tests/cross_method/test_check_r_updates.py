"""R更新確認スクリプト（`.github/scripts/check_r_updates.py`）のR本体の版比較。

フィクスチャの`_meta.r_version`は生成スクリプトにより書式が混在する
（`getRversion()`由来の`4.5.3`と、`R.version.string`由来の
`R version 4.5.3 (2026-03-11)`）。書式の違いで同じ版が別の版と判定され、
更新が無いのに「更新あり」のIssueが毎週起票されないことを固定する。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    path = REPO_ROOT / ".github" / "scripts" / "check_r_updates.py"
    spec = importlib.util.spec_from_file_location("check_r_updates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_r_updates = _load_script()


def _write_fixture(directory: Path, name: str, r_version: object) -> None:
    payload = {"_meta": {"r_version": r_version}}
    (directory / f"{name}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4.5.3", "4.5.3"),
        ("R version 4.5.3 (2026-03-11)", "4.5.3"),
        ("R version 4.6.0 (2026-04-24)", "4.6.0"),
        ("not a version", None),
    ],
)
def test_extract_r_version(raw: str, expected: str | None) -> None:
    assert check_r_updates._extract_r_version(raw) == expected


def test_mixed_formats_are_the_same_version(tmp_path: Path) -> None:
    _write_fixture(tmp_path, "a", "4.5.3")
    _write_fixture(tmp_path, "b", "R version 4.5.3 (2026-03-11)")

    version = check_r_updates.fixture_r_version(tmp_path)

    assert version == "4.5.3"
    # 最新のAPT版と同じなら「更新あり」にならない（誤検知の再発防止）。
    assert check_r_updates._version_key(
        version
    ) == check_r_updates._version_key("4.5.3")


def test_fixture_r_version_picks_the_largest(tmp_path: Path) -> None:
    _write_fixture(tmp_path, "old", "R version 4.5.3 (2026-03-11)")
    _write_fixture(tmp_path, "new", "4.6.0")
    _write_fixture(tmp_path, "unparsable", "unknown")
    _write_fixture(tmp_path, "not_a_string", 4)

    assert check_r_updates.fixture_r_version(tmp_path) == "4.6.0"


def test_fixture_r_version_without_any_value(tmp_path: Path) -> None:
    assert check_r_updates.fixture_r_version(tmp_path) is None


def test_committed_fixtures_match_current_r() -> None:
    """コミット済みフィクスチャの最大のRの版が、版番号として読み取れる。"""
    version = check_r_updates.fixture_r_version(
        REPO_ROOT / "tests" / "fixtures" / "benchmarks"
    )

    assert version is not None
    assert all(part.isdigit() for part in version.split("."))
