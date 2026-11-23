"""係数の分散共分散行列（`cov_params`）を別のフィクスチャJSONに分けるための共通処理。

リファレンスのアダプタ（`statsmodels_ref.py`・`r.py`等）は、各ケースの結果に
`cov_params`を常に含めて返す。手法ごとの生成スクリプトは、シナリオ×`cov_type`のツリーを
1か所の`build_fixtures()`で組み立て、

- 既存の統計量のフィクスチャ（`ols.json`等）には`strip_cov_params`で`cov_params`を除いて書く
- 行列のフィクスチャ（`ols_cov.json`等）には`only_cov_params`で`cov_params`だけを残して書く

ように2つのファイルへ分ける。同じツリーから作るので、2つのファイルのシナリオ・`cov_type`の
キーは構造的にずれない。行列だけのファイルを別にしてあるのは、既存のフィクスチャJSONの
中身を変えずに済ませ、行列が大きくなっても統計量の差分と混ざらないようにするため。

使用例（`generate_ols_cov_fixtures.py`参照）:
    python -m benchmark.linear.fixtures.generate_ols_cov_fixtures
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COV_KEY = "cov_params"

# 行列のフィクスチャの`_meta`に、元の統計量のフィクスチャの`_meta`から写すキー
# （リファレンスのバージョンを再取得せずに済ませる）。
_COPIED_META_KEYS = ("method", "primary_reference")


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and COV_KEY in node


def strip_cov_params(tree: Any) -> Any:
    """ツリーの葉（`cov_params`を持つdict）から`cov_params`を除く。

    それ以外のキーとその並びは変えない。葉でないdictは再帰し、他の値はそのまま返す。
    """
    if _is_leaf(tree):
        return {k: v for k, v in tree.items() if k != COV_KEY}
    if isinstance(tree, dict):
        return {k: strip_cov_params(v) for k, v in tree.items()}
    return tree


def only_cov_params(tree: Any) -> Any:
    """ツリーから葉の`cov_params`だけを残す（`{"cov_params": 行列}`）。

    `cov_params`を含まない部分（`predict`の葉・`_meta`等）は落とす。残るものが無い
    部分木は`None`を返し、親が取り除く。
    """
    if _is_leaf(tree):
        return {COV_KEY: tree[COV_KEY]}
    if isinstance(tree, dict):
        kept = {k: only_cov_params(v) for k, v in tree.items()}
        kept = {k: v for k, v in kept.items() if v is not None}
        return kept or None
    return None


def build_cov_fixtures(
    build_fixtures_fn: Callable[[], dict],
    *,
    source_fixture: str,
    purpose: str,
) -> dict:
    """`build_fixtures_fn()`のツリーから、行列だけのフィクスチャを組み立てる。

    Args:
        build_fixtures_fn: 統計量のフィクスチャの`build_fixtures()`。
        source_fixture: 同じツリーから作る統計量のフィクスチャのファイル名
            （`_meta.source_fixture`に残す）。
        purpose: `_meta.purpose`に残す説明。
    """
    full = build_fixtures_fn()
    cov = only_cov_params(full)
    source_meta = full.get("_meta", {})
    version_keys = [k for k in source_meta if k.endswith("_version")]
    cov["_meta"] = {
        **{k: source_meta[k] for k in _COPIED_META_KEYS if k in source_meta},
        "purpose": purpose,
        "source_fixture": source_fixture,
        "generated_at": datetime.now(UTC).isoformat(),
        **{k: source_meta[k] for k in version_keys},
    }
    return cov


def run_cov_fixture_cli(
    build_fixtures_fn: Callable[[], dict],
    default_output: str | Path,
    *,
    source_fixture: str,
    purpose: str,
    description: str | None = None,
) -> None:
    """行列のフィクスチャを書き出す`__main__`の共通処理（`run_fixture_cli`の行列版）。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", default=str(default_output))
    args = parser.parse_args()

    fixtures = build_cov_fixtures(
        build_fixtures_fn, source_fixture=source_fixture, purpose=purpose
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
