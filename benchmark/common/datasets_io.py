"""固定済みデータセットの読み込みと凍結（freeze）まわりの共通ヘルパー。

- `DATA_DIR`: 固定済み合成データセットCSVの置き場所。
- `load_frozen_dataset`: `{prefix}_{scenario}.csv`＋`{prefix}_true_beta.json`を読む。
- `freeze_scenarios`: シナリオでループしてCSV＋true_beta辞書を積み上げる。
- `run_freeze_cli`: 各系統の `benchmark/<系統>/freeze.py` 共通の `__main__`。

旧 `benchmark/_common.py`（→ `benchmark/common/helpers.py`）から Initiative A で
関心事ごとに分割した（`docs/planning/specs/benchmark-restructure-design.md` 4章）。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# このファイルは benchmark/common/ 配下。parents[2] がリポジトリルート。
# BENCHMARKS_DIR: リファレンス JSON（<手法>.json / <手法>_crosscheck.json）の置き場所。
# DATA_DIR: 固定済み合成データセット CSV の置き場所（その下の data/）。
BENCHMARKS_DIR = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "benchmarks"
)
DATA_DIR = BENCHMARKS_DIR / "data"


def load_frozen_dataset(
    prefix: str, scenario: str
) -> tuple[pl.DataFrame, list[float] | None]:
    """固定済みの合成データセットCSV＋true_beta JSONを読む。

    `freeze_datasets.py`が書き出した`{prefix}_{scenario}.csv`と
    `{prefix}_true_beta.json`を`DATA_DIR`から読む。

    Args:
        prefix: データセットのファイル名prefix（例: "synthetic", "logit",
            "probit", "iv"）。
        scenario: シナリオ名（例: "baseline"）。

    Returns:
        (df, true_beta)のタプル。true_betaはJSON側にエントリが無い場合
        （クラスター確認用の一時CSV等）は`None`。
    """
    df = pl.read_csv(DATA_DIR / f"{prefix}_{scenario}.csv")
    true_betas = json.loads(
        (DATA_DIR / f"{prefix}_true_beta.json").read_text()
    )
    return df, true_betas.get(scenario)


def freeze_scenarios(
    output_dir: Path,
    generator_fn: Callable[..., tuple[pl.DataFrame, np.ndarray]],
    scenarios: list[str],
    prefix: str,
    true_betas: dict[str, list[float]],
    *,
    filename_suffix: str = "",
    key_suffix: str = "",
    **generator_kwargs: Any,
) -> None:
    """シナリオでループしてCSVを書き出し、true_beta辞書に積み上げる。

    `freeze_datasets.py`（旧・単一ファイル）の各系統ブロックが繰り返していた
    「シナリオ生成→CSV書き出し→true_beta収集」パターンの共通部分。
    書き出したtrue_betas辞書のJSON化・書き出しは呼び出し側が行う
    （系統によって1系統内で複数回に分けて書き出すケースがあるため）。

    Args:
        output_dir: CSV出力先ディレクトリ。
        generator_fn: `(scenario, **kwargs) -> (df, true_beta)`を返す生成関数。
        scenarios: 対象シナリオ名のリスト。
        prefix: 出力ファイル名prefix（例: "synthetic", "iv"）。
        true_betas: 結果を積み上げる辞書（呼び出し側が保持、副作用で更新）。
        filename_suffix: 出力CSVファイル名に付与するsuffix（例: "_k1"）。
        key_suffix: true_betas辞書のキーに付与するsuffix。
        **generator_kwargs: `generator_fn`にそのまま渡す追加引数。
    """
    for scenario in scenarios:
        df, true_beta = generator_fn(scenario, **generator_kwargs)
        df.write_csv(output_dir / f"{prefix}_{scenario}{filename_suffix}.csv")
        true_betas[f"{scenario}{key_suffix}"] = true_beta.tolist()


def run_freeze_cli(
    freeze_fn: Callable[[Path], None],
    default_output_dir: str,
    success_message: str,
    *,
    description: str | None = None,
) -> None:
    """`freeze_datasets.py`・`freeze_<系統>_datasets.py`共通の`__main__`処理。

    引数パース→出力先ディレクトリ作成→`freeze_fn`呼び出し→完了printまでを
    まとめる。`freeze_fn`自体はディレクトリ作成・printを行わない前提
    （出力先ディレクトリは呼び出し側が用意済みとして渡す）。

    Args:
        freeze_fn: `(output_dir) -> None`。実際の凍結処理本体。
        default_output_dir: `--output-dir`の既定値。
        success_message: 完了printのメッセージ本文（`" to {output_dir}"`が
            続く、例: "wrote frozen linear datasets"）。
        description: argparseのdescription（通常は呼び出し元モジュールの`__doc__`）。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", default=default_output_dir)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_fn(output_dir)
    print(f"{success_message} to {output_dir}")
