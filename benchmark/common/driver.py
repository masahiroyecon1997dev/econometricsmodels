"""フィクスチャ生成スクリプトの共通ドライバ。

現時点では `run_fixture_cli`（`generate_*_fixtures.py` 11ファイルで一字一句同じ
だった `__main__` ブロック。`docs/planning/specs/refactoring-candidates.md` 項目19）
のみ。`MethodBenchmarkSpec` / `build_fixture_json`（メインループのデータ駆動化）は、
OLS/WLS を最初の消費者として形が固まってから追加する
（`docs/planning/specs/benchmark-restructure-design.md` 8章、rule of three）。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path


def run_fixture_cli(
    build_fixtures_fn: Callable[[], dict],
    default_output: str | Path,
    *,
    description: str | None = None,
) -> None:
    """`generate_<手法>_fixtures.py` 共通の `__main__` 処理。

    `--output` をパースし、`build_fixtures_fn()` の結果を JSON として書き出して
    バイト数を表示する。`--output` の既定値以外は全手法で完全に同一だった。

    Args:
        build_fixtures_fn: 引数なしで手法1つ分の fixtures dict を返す関数。
        default_output: `--output` の既定パス（通常
            `benchmark.common.BENCHMARKS_DIR / "<手法>.json"`）。
        description: argparse の description（通常は呼び出し元モジュールの
            `__doc__`）。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", default=str(default_output))
    args = parser.parse_args()

    fixtures = build_fixtures_fn()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
