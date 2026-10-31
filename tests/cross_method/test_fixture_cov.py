"""係数の分散共分散行列のフィクスチャ（`<手法>_cov.json`）の読み込みと比較ヘルパーのテスト。

数値の照合そのもの（本実装とリファレンスの行列の比較）は各手法のreference/crosscheckテストが担う。
ここでは、その土台が不整合を見逃さないことを確かめる。

- `assert_cov_params_close`: 非対角成分のずれ・名前の並びの違いで失敗する
- `load_fixtures_with_cov`: 統計量のフィクスチャと行列のフィクスチャのキーや対角成分が食い違うと失敗する
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import _fixture_cov as fixture_cov
import pytest
from _assertions import assert_cov_params_close
from _fixture_cov import cov_fixture_path, load_fixtures_with_cov

REF = {
    "const": {"const": 4.0, "x1": -1.0},
    "x1": {"x1": 1.0, "const": -1.0},
}
RTOL = 1e-8


def _ours(**changes: float) -> dict:
    ours = copy.deepcopy(REF)
    for key, delta in changes.items():
        row, col = key.split("__")
        ours[row][col] += delta
    return ours


def test_cov_params_close_accepts_values_within_tolerance():
    assert_cov_params_close(_ours(const__x1=1e-10), REF, "case", rtol=RTOL)


def test_cov_params_close_rejects_a_shifted_off_diagonal_entry():
    # 非対角成分だけがずれる。対角成分（標準誤差）の比較では見つからない。
    with pytest.raises(AssertionError, match="cov_params/const/x1"):
        assert_cov_params_close(_ours(const__x1=1e-6), REF, "case", rtol=RTOL)


def test_cov_params_close_scales_the_tolerance_by_both_variances():
    # 同じ絶対差でも、分散が小さい係数どうしの共分散では許される差が小さい
    ref = {"a": {"a": 1e-8, "b": 0.0}, "b": {"a": 0.0, "b": 1e-8}}
    ours = copy.deepcopy(ref)
    ours["a"]["b"] = 1e-12
    with pytest.raises(AssertionError, match="cov_params/a/b"):
        assert_cov_params_close(ours, ref, "case", rtol=RTOL)


def test_cov_params_close_rejects_a_different_order_of_names():
    reordered = {"x1": REF["x1"], "const": REF["const"]}
    with pytest.raises(AssertionError, match="並び"):
        assert_cov_params_close(reordered, REF, "case", rtol=RTOL)


def test_cov_params_close_rejects_a_missing_coefficient():
    with pytest.raises(AssertionError, match="キー集合"):
        assert_cov_params_close({"const": {"const": 4.0}}, REF, "c", rtol=RTOL)


@pytest.mark.parametrize(
    ("stats_name", "cov_name"),
    [
        ("ols.json", "ols_cov.json"),
        ("ols_crosscheck.json", "ols_cov_crosscheck.json"),
        ("fe_crosscheck.json", "fe_cov_crosscheck.json"),
    ],
)
def test_cov_fixture_path_follows_the_naming_convention(stats_name, cov_name):
    assert cov_fixture_path(Path(stats_name)).name == cov_name


def _write_pair(tmp_path: Path, stats: dict, cov: dict) -> Path:
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(stats))
    cov_fixture_path(path).write_text(json.dumps(cov))
    return path


def _leaf(se_a: float = 2.0, se_b: float = 1.0) -> dict:
    return {
        "params": {"a": 0.1, "b": 0.2},
        "std_errors": {"a": se_a, "b": se_b},
    }


def _cov_leaf(var_a: float = 4.0, var_b: float = 1.0) -> dict:
    return {
        "cov_params": {
            "a": {"a": var_a, "b": 0.5},
            "b": {"a": 0.5, "b": var_b},
        }
    }


def test_load_merges_the_matrix_into_the_matching_leaf(tmp_path):
    stats = {"synthetic": {"baseline": {"classical": _leaf()}}, "_meta": {}}
    cov = {"synthetic": {"baseline": {"classical": _cov_leaf()}}, "_meta": {}}
    loaded = load_fixtures_with_cov(_write_pair(tmp_path, stats, cov))
    leaf = loaded["synthetic"]["baseline"]["classical"]
    assert leaf["cov_params"]["a"]["b"] == 0.5
    assert math.sqrt(leaf["cov_params"]["a"]["a"]) == leaf["std_errors"]["a"]


def test_load_rejects_a_stats_leaf_without_a_matrix(tmp_path):
    stats = {"baseline": {"classical": _leaf(), "hc1": _leaf()}}
    cov = {"baseline": {"classical": _cov_leaf()}}
    path = _write_pair(tmp_path, stats, cov)
    with pytest.raises(AssertionError, match="行列のフィクスチャにない葉"):
        load_fixtures_with_cov(path)


def test_load_rejects_a_matrix_without_a_stats_leaf(tmp_path):
    stats = {"baseline": {"classical": _leaf()}}
    cov = {"baseline": {"classical": _cov_leaf(), "hc1": _cov_leaf()}}
    path = _write_pair(tmp_path, stats, cov)
    with pytest.raises(
        AssertionError, match="行列のフィクスチャにだけあるキー"
    ):
        load_fixtures_with_cov(path)


def test_load_rejects_a_diagonal_that_disagrees_with_the_std_errors(tmp_path):
    # 片方のファイルだけを再生成した場合の食い違い
    stats = {"baseline": {"classical": _leaf(se_a=2.0)}}
    cov = {"baseline": {"classical": _cov_leaf(var_a=9.0)}}
    path = _write_pair(tmp_path, stats, cov)
    with pytest.raises(AssertionError, match="sqrt"):
        load_fixtures_with_cov(path)


# ── 行列を持たない葉の登録と、比較の記録 ───────────────────────────


def test_load_accepts_a_registered_leaf_without_a_matrix(
    tmp_path, monkeypatch
):
    stats = {"baseline": {"classical": _leaf(), "shifted": _leaf()}}
    cov = {"baseline": {"classical": _cov_leaf()}}
    path = _write_pair(tmp_path, stats, cov)
    monkeypatch.setitem(
        fixture_cov.NO_COV_LEAVES, "toy/baseline/shifted", "reason"
    )
    loaded = load_fixtures_with_cov(path)
    assert "cov_params" not in loaded["baseline"]["shifted"]
    assert "cov_params" in loaded["baseline"]["classical"]


def test_load_rejects_a_registered_leaf_that_has_a_matrix(
    tmp_path, monkeypatch
):
    stats = {"baseline": {"classical": _leaf()}}
    cov = {"baseline": {"classical": _cov_leaf()}}
    path = _write_pair(tmp_path, stats, cov)
    monkeypatch.setitem(
        fixture_cov.NO_COV_LEAVES, "toy/baseline/classical", "reason"
    )
    with pytest.raises(AssertionError, match="NO_COV_LEAVES"):
        load_fixtures_with_cov(path)


def test_load_rejects_a_registered_leaf_that_no_longer_exists(
    tmp_path, monkeypatch
):
    stats = {"baseline": {"classical": _leaf()}}
    cov = {"baseline": {"classical": _cov_leaf()}}
    path = _write_pair(tmp_path, stats, cov)
    monkeypatch.setitem(
        fixture_cov.NO_COV_LEAVES, "toy/baseline/gone", "reason"
    )
    with pytest.raises(AssertionError, match="統計量のフィクスチャに無い葉"):
        load_fixtures_with_cov(path)


def test_comparing_a_loaded_matrix_marks_it_consumed(tmp_path, monkeypatch):
    stats = {"baseline": {"classical": _leaf()}}
    cov = {"baseline": {"classical": _cov_leaf()}}
    path = _write_pair(tmp_path, stats, cov)
    loaded = load_fixtures_with_cov(path)
    ref = loaded["baseline"]["classical"]["cov_params"]
    monkeypatch.setattr(fixture_cov, "CONSUMED_COV_LEAVES", set())
    assert_cov_params_close(copy.deepcopy(ref), ref, "case", rtol=RTOL)
    assert fixture_cov.CONSUMED_COV_LEAVES == {"toy/baseline/classical"}


def test_unconsumed_leaves_are_the_loaded_ones_that_were_never_compared(
    monkeypatch,
):
    monkeypatch.setattr(
        fixture_cov, "REGISTERED_COV_LEAVES", {"a/x", "a/y", "b/z"}
    )
    monkeypatch.setattr(fixture_cov, "CONSUMED_COV_LEAVES", {"a/x"})
    assert fixture_cov.unconsumed_cov_leaves() == ["a/y", "b/z"]


def test_the_registered_leaves_have_a_reason():
    # 行列を比べない葉は、理由なしに増やさない。
    assert fixture_cov.NO_COV_LEAVES
    assert all(
        len(reason) > 20 for reason in fixture_cov.NO_COV_LEAVES.values()
    )
