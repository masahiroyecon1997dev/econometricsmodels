"""係数の分散共分散行列のフィクスチャ（`<手法>_cov.json`）を統計量のフィクスチャへ合流させる。

統計量のフィクスチャ（`ols.json`等）と行列のフィクスチャ（`ols_cov.json`等）は、同じ
`build_fixtures()`のツリーから分けて書かれる（`benchmark/common/cov_fixtures.py`）。テストは
`load_fixtures_with_cov`で2つを読み、葉のdictに`cov_params`を足した1つのツリーとして
使う。各テストの呼び出し箇所を変えずに、`assert_inference_close`が葉の`cov_params`を
比較できる。

合流の時点で、ファイルの食い違いを検出する。

- 行列のフィクスチャの葉は、統計量のフィクスチャに同じ位置の葉がある
- 統計量のフィクスチャの葉（`params`と`std_errors`を持つもの）は、すべて行列を持つ。
  意図的に持たない葉は`NO_COV_LEAVES`に理由つきで登録する（登録した葉は行列を持たず、
  登録していない葉は必ず持つ）
- 行列の対角成分の平方根が`std_errors`と一致する（どちらかだけを再生成した場合など）

読み込んだ行列がテストで比較されたことも検出する。合流させた行列は`TrackedCov`で、
`assert_cov_params_close`が比較した行列を`CONSUMED_COV_LEAVES`に記録する。`--check-cov-coverage`を
付けた全体実行の終了時に、`conftest.py`が`REGISTERED_COV_LEAVES`のうち比較されなかった葉を
失敗として報告する（読み込んだだけで比較しない行列は、検証した気になるだけで何も検証しないため）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

COV_KEY = "cov_params"

_REAL_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"
)

# 統計量はあるが、意図的に行列を比べない葉（`<フィクスチャ名>/<葉のパス>` → 理由）。
# 登録した葉は行列のフィクスチャに入れない（生成側で落とす）。リファレンスの行列が
# 本実装の定義と異なる、あるいは独立なリファレンスがない葉だけを、理由つきで登録する。
# `docs/guide/verification.md`の「行列を比べない葉」に同じ一覧を載せる。
_PLM_TRACE_FORM = (
    "全個体の観測数が同じで時点がずれたパネルでは、plmは分散成分にトレース形を使い、"
    "本実装の閉形式と値が異なる。plmの行列はリファレンスにならず、手計算の閉形式の行列は"
    "本実装と計算方法が同じで独立なリファレンスにならない"
)
NO_COV_LEAVES: dict[str, str] = {
    f"re_variance_method/synthetic/ow_equal_counts_shifted_time/{method}/{cov_type}": _PLM_TRACE_FORM
    for method in ("wallace_hussain", "amemiya")
    for cov_type in ("classical", "cluster")
}

# 読み込んだ行列の葉（`<フィクスチャ名>/<葉のパス>`）と、テストで比較された葉。
# 実際のフィクスチャのディレクトリから読んだものだけを記録する。
REGISTERED_COV_LEAVES: set[str] = set()
CONSUMED_COV_LEAVES: set[str] = set()


class TrackedCov(dict):
    """合流させた行列。どの葉の行列かを`leaf_id`に持つ（比較の記録に使う）。"""

    leaf_id: str


def mark_cov_consumed(cov: object) -> None:
    """`cov`が合流させた行列なら、比較されたものとして記録する。"""
    leaf_id = getattr(cov, "leaf_id", None)
    if leaf_id is not None:
        CONSUMED_COV_LEAVES.add(leaf_id)


def unconsumed_cov_leaves() -> list[str]:
    """読み込んだが、どのテストも比較しなかった行列の葉。"""
    return sorted(REGISTERED_COV_LEAVES - CONSUMED_COV_LEAVES)


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


def _merge(stats: dict, cov: dict, path: str, tracked: bool) -> None:
    for key, cov_node in cov.items():
        where = f"{path}/{key}"
        assert key in stats, f"{where}: 行列のフィクスチャにだけあるキー"
        stats_node = stats[key]
        if _is_cov_leaf(cov_node):
            assert _is_stats_leaf(stats_node), (
                f"{where}: 統計量のフィクスチャに対応する葉がない"
            )
            assert where not in NO_COV_LEAVES, (
                f"{where}: 行列を持たない葉（NO_COV_LEAVES）に行列がある"
            )
            _check_diagonal(stats_node["std_errors"], cov_node[COV_KEY], where)
            merged = TrackedCov(cov_node[COV_KEY])
            merged.leaf_id = where
            stats_node[COV_KEY] = merged
            if tracked:
                REGISTERED_COV_LEAVES.add(where)
        else:
            assert isinstance(stats_node, dict), where
            _merge(stats_node, cov_node, where, tracked)


def _check_diagonal(std_errors: dict, cov: dict, where: str) -> None:
    assert list(cov) == list(std_errors), f"{where}: 行列の名前の並びが違う"
    for name, se in std_errors.items():
        assert list(cov[name]) == list(std_errors), f"{where}/{name}: 列"
        from_cov = math.sqrt(cov[name][name])
        assert abs(from_cov - se) <= _DIAG_RTOL * abs(se), (
            f"{where}/{name}: sqrt(diag)={from_cov!r} != std_errors={se!r}"
        )


def _stats_leaves(node: object, path: str) -> dict[str, bool]:
    """統計量の葉（`<パス>` → 行列を持つか）。"""
    if _is_stats_leaf(node):
        return {path: COV_KEY in node}
    if not isinstance(node, dict):
        return {}
    found: dict[str, bool] = {}
    for key, child in node.items():
        found |= _stats_leaves(child, f"{path}/{key}")
    return found


def load_fixtures_with_cov(path: Path) -> dict:
    """統計量のフィクスチャを読み、行列のフィクスチャを葉へ合流させて返す。"""
    stats = json.loads(path.read_text())
    cov = json.loads(cov_fixture_path(path).read_text())
    cov.pop("_meta", None)
    tracked = path.resolve().parent == _REAL_FIXTURE_DIR
    _merge(stats, cov, path.stem, tracked)
    leaves = _stats_leaves(stats, path.stem)
    uncovered = [
        leaf
        for leaf, has_cov in leaves.items()
        if not has_cov and leaf not in NO_COV_LEAVES
    ]
    assert not uncovered, f"行列のフィクスチャにない葉: {uncovered[:5]}"
    prefix = f"{path.stem}/"
    stale = [
        leaf
        for leaf in NO_COV_LEAVES
        if leaf.startswith(prefix) and leaf not in leaves
    ]
    assert not stale, (
        f"NO_COV_LEAVESにあるが統計量のフィクスチャに無い葉: {stale[:5]}"
    )
    return stats
