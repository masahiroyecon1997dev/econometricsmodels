"""フィクスチャ生成スクリプトの共通ドライバ。

現時点では `run_fixture_cli`（`generate_*_fixtures.py` 11ファイルで一字一句同じ
だった `__main__` ブロックを1関数に集約したもの）のみ。当初構想していた
`MethodBenchmarkSpec` / `build_fixture_json`（メインループのデータ駆動化）は
見送った——OLS〜IV/IV-GMM の6手法すべて軽量な共有ヘルパー（本関数・
`benchmark.linear.references.r.run_lm_r`・`benchmark.common.reference.r` の
`run_r`/`normalize_names`・`extract_params_std_errors`）で移行でき、dataclass Spec 層を
足す必要が生じなかったため（rule of three 未達）。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path


def group_by_dataset_source(
    flat: dict, wooldridge_datasets: Iterable[str]
) -> dict:
    """平坦なフィクスチャdictを`synthetic`/`wooldridge`の二層に振り分ける。

    全フィクスチャJSONの最上位を`{"synthetic": ..., "wooldridge": ...,
    "_meta": ...}`に揃えるための共通処理。`wooldridge_datasets`に挙げた
    キー（Wooldridge実データのケース）を`wooldridge`へ、それ以外
    （合成データのシナリオと、それを元にした`solver`・`no_intercept`等の
    特殊ケース）を`synthetic`へ入れる。各内側の並び順は`flat`の挿入順を保つ。
    `_meta`は最後に置く。

    Args:
        flat: 最上位にシナリオ名・実データ名・`_meta`が並ぶdict。
        wooldridge_datasets: `wooldridge`側に入れるキーの集合。`flat`に
            存在しないキーを渡すと`KeyError`にする（取りこぼし防止）。

    Returns:
        `{"synthetic": {...}, "wooldridge": {...}, "_meta": ...}`。
    """
    real = list(wooldridge_datasets)
    missing = [name for name in real if name not in flat]
    if missing:
        raise KeyError(f"wooldridge datasets not in fixtures: {missing}")
    grouped: dict = {
        "synthetic": {
            k: v for k, v in flat.items() if k not in real and k != "_meta"
        },
        "wooldridge": {name: flat[name] for name in real},
    }
    if "_meta" in flat:
        grouped["_meta"] = flat["_meta"]
    return grouped


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
