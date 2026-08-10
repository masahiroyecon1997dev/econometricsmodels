"""系統（linear/nonlinear/iv）をまたいで使う共通ヘルパー。

Issue #231（リファクタリング）で、複数の`benchmark/<系統>/`スクリプトに
コピペされていた以下を集約した。

- `imbalanced_cluster_groups`: 全系統のfixture生成・テストから使われる
  クラスターラベル生成（旧`generate_synthetic_datasets.py`、系統非依存部分）。
- `hac_auto_lag`: HACの自動ラグ選択式（旧`compare_performance.py`・
  `generate_ols_crosscheck_fixtures.py`等5箇所に同一実装が重複していた）。
- `DATA_DIR`: 固定済み合成データセットCSVの置き場所（`freeze_datasets.py`参照）。
- `load_frozen_dataset`: `{prefix}_{scenario}.csv`＋`{prefix}_true_beta.json`を
  読む処理（旧`_load_synthetic`/`_load_iv_dataset`、3箇所に類似実装が重複していた）。
- `freeze_scenarios`: シナリオでループしてCSV＋true_beta辞書を積み上げる処理
  （旧`freeze_datasets.py`の各系統ブロックに繰り返されていたパターン）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

DATA_DIR = (
    Path(__file__).resolve().parent
    / ".."
    / "tests"
    / "api_tests"
    / "fixtures"
    / "benchmarks"
    / "data"
).resolve()

# [2, 3, 5, 10, 30, 50]（合計100）を1タイルとして繰り返す不均衡なクラスタサイズ
# パターン（testing-policy.md「テスト用データセット」3.参照）。
_IMBALANCED_CLUSTER_TILE = [2, 3, 5, 10, 30, 50]


def imbalanced_cluster_groups(n: int) -> list[str]:
    """不均衡なクラスタグループ（グループ数・サイズが偏ったラベル列）を生成する。

    `_IMBALANCED_CLUSTER_TILE`（サイズ合計100）をnに応じてタイル状に繰り返す。
    均等サイズの疑似グループ（行番号%10等）だけでは見逃す、実務的に起こりやすい
    グループサイズの偏りを再現する。

    Args:
        n: 観測数。100の倍数である必要がある（タイルが端数なく割り切れるように）。

    Returns:
        長さnのグループラベル（"g0", "g1", ...）のリスト。

    Raises:
        ValueError: nが100の倍数でない場合。
    """
    if n % 100 != 0:
        raise ValueError(
            f"n must be a multiple of 100 to tile the imbalanced cluster "
            f"pattern exactly, got n={n}"
        )
    n_tiles = n // 100
    labels: list[str] = []
    group_idx = 0
    for _ in range(n_tiles):
        for size in _IMBALANCED_CLUSTER_TILE:
            labels.extend([f"g{group_idx}"] * size)
            group_idx += 1
    return labels


def hac_auto_lag(n: int) -> int:
    """engineのHAC自動ラグ選択式（`resolve_hac_lags`）と同じ計算式。

    Rクロスチェック・性能比較の各スクリプトに同じ明示ラグを渡すことで、
    自動ラグ選択式自体の実装差を比較対象から除外するために使う。
    """
    return int(4 * (n / 100) ** (2 / 9))


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
