"""係数の分散共分散行列のフィクスチャ（`<手法>_cov.json`）を統計量のフィクスチャへ合流させる。

統計量のフィクスチャ（`ols.json`等）と行列のフィクスチャ（`ols_cov.json`等）は、同じ
`build_fixtures()`のツリーから分けて書かれる（`benchmark/common/cov_fixtures.py`）。テストは
`load_fixtures_with_cov`で2つを読み、葉のdictに`cov_params`を足した1つのツリーとして
使う。各テストの呼び出し箇所を変えずに、`assert_inference_close`が葉の`cov_params`を
比較できる。

合流の時点で、ファイルの食い違いを検出する。

- 行列のフィクスチャの葉は、統計量のフィクスチャに同じ位置の葉がある
- 統計量のフィクスチャの葉（`params`と`std_errors`を持つもの）は、すべて行列を持つ
- 行列の対角成分の平方根が`std_errors`と一致する（どちらかだけを再生成した場合など）
"""

from __future__ import annotations

import json
import math
from pathlib import Path

COV_KEY = "cov_params"

# 行列の対角成分の平方根と`std_errors`の一致。同じ計算から取り出した値なので機械精度で一致する
# （実測は丸め誤差のみ）。
_DIAG_RTOL = 1e-10


def cov_fixture_path(path: Path) -> Path:
    """統計量のフィクスチャのパスから、対応する行列のフィクスチャのパスを返す。

    `ols.json` → `ols_cov.json`、`ols_crosscheck.json` → `ols_cov_crosscheck.json`。
    """
    stem = path.stem
    if stem.endswith("_crosscheck"):
        stem = stem.removesuffix("_crosscheck") + "_cov_crosscheck"
    else:
        stem = stem + "_cov"
    return path.with_name(stem + ".json")


def _is_stats_leaf(node: object) -> bool:
    return isinstance(node, dict) and "params" in node and "std_errors" in node


def _is_cov_leaf(node: object) -> bool:
    return isinstance(node, dict) and COV_KEY in node


def _merge(stats: dict, cov: dict, path: str) -> None:
    for key, cov_node in cov.items():
        where = f"{path}/{key}"
        assert key in stats, f"{where}: 行列のフィクスチャにだけあるキー"
        stats_node = stats[key]
        if _is_cov_leaf(cov_node):
            assert _is_stats_leaf(stats_node), (
                f"{where}: 統計量のフィクスチャに対応する葉がない"
            )
            _check_diagonal(stats_node["std_errors"], cov_node[COV_KEY], where)
            stats_node[COV_KEY] = cov_node[COV_KEY]
        else:
            assert isinstance(stats_node, dict), where
            _merge(stats_node, cov_node, where)


def _check_diagonal(std_errors: dict, cov: dict, where: str) -> None:
    assert list(cov) == list(std_errors), f"{where}: 行列の名前の並びが違う"
    for name, se in std_errors.items():
        assert list(cov[name]) == list(std_errors), f"{where}/{name}: 列"
        from_cov = math.sqrt(cov[name][name])
        assert abs(from_cov - se) <= _DIAG_RTOL * abs(se), (
            f"{where}/{name}: sqrt(diag)={from_cov!r} != std_errors={se!r}"
        )


def _uncovered_leaves(node: object, path: str) -> list[str]:
    if _is_stats_leaf(node):
        return [] if COV_KEY in node else [path]
    if not isinstance(node, dict):
        return []
    found: list[str] = []
    for key, child in node.items():
        found += _uncovered_leaves(child, f"{path}/{key}")
    return found


def load_fixtures_with_cov(path: Path) -> dict:
    """統計量のフィクスチャを読み、行列のフィクスチャを葉へ合流させて返す。"""
    stats = json.loads(path.read_text())
    cov = json.loads(cov_fixture_path(path).read_text())
    cov.pop("_meta", None)
    _merge(stats, cov, path.stem)
    uncovered = _uncovered_leaves(stats, path.stem)
    assert not uncovered, f"行列のフィクスチャにない葉: {uncovered[:5]}"
    return stats
