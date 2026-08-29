"""テストデータ生成・ロードの共通ヘルパー。

複数のテストファイルに重複していた以下を集約する。

- `with_cluster_groups`: 「行番号%N」の疑似クラスターラベル付与。
  `benchmark/_common.py`の`imbalanced_cluster_groups`（不均衡クラスタ版）とは
  役割が近いが、これは均等サイズ版でテスト専用のロジックのため、`benchmark/`とは
  ライフサイクルが異なる`tests/`側に置く（`.claude/rules/testing-policy.md`
  「テストの分離」参照。ユーザー確認済み）。
- `separation_suspected_dataset`: 準完全分離データのDGP（`test_logit.py`/
  `test_probit.py`で完全に同一実装だった）。
- `MROZ_X`: Wooldridge mrozデータセットの説明変数リスト。
- `load_wooldridge_dataset`: Wooldridgeデータセットのロード（`wooldridge`
  パッケージが無い環境ではskip）。`benchmark/load_wooldridge.py`の`load`を
  呼ぶだけの`wooldridge.data(name)`→`pl.from_pandas`実装が、複数ファイルに
  微妙に異なる書き方（直接呼び出し／`load_wooldridge.py`経由）で重複していた。
- `DATA_DIR`: 固定済み合成データセットCSV（`tests/fixtures/benchmarks/data/`）の
  置き場所。全テストファイルが`tests/`直下にあるため値は常に同じで、
  `Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"`という
  同一の組み立て方が複数ファイルに重複していた。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"


def with_cluster_groups(
    df: pl.DataFrame, n_groups: int, col: str = "cluster_group"
) -> pl.DataFrame:
    """行番号を`n_groups`で割った余りを疑似クラスターラベルとして付与する。

    統計的な意味はなく、クラスターロバストSEの実装の動作確認用
    （`.claude/rules/testing-policy.md`「テスト用データセット」3.）。
    """
    return (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % n_groups).alias(col))
        .drop("_row")
    )


def separation_suspected_dataset() -> pl.DataFrame:
    """准完全分離データ（`x1`の真の係数を極端に大きくし、ほぼ全観測がx1の符号だけで
    完全に分類できるようにしたDGP）を生成する。

    `Logit`/`Probit`いずれのComputationError（`SeparationSuspected`）テストにも
    使う（Probit側もsigmoidベースのDGPをそのまま流用する。`test_logit.py`の
    Logit版のProbit版という位置づけ、DGP自体の正確なProbitリンクである必要はない）。
    """
    random.seed(42)
    n = 200
    beta = (0.0, 100.0, 0.5)
    x1 = [random.uniform(-2.0, 2.0) for _ in range(n)]
    x2 = [random.uniform(-1.0, 1.0) for _ in range(n)]
    y = []
    for i in range(n):
        z = beta[0] + beta[1] * x1[i] + beta[2] * x2[i]
        p = 1.0 / (1.0 + math.exp(-z))
        y.append(1.0 if random.random() < p else 0.0)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


MROZ_X = ["nwifeinc", "educ", "exper", "expersq", "age", "kidslt6", "kidsge6"]


def wooldridge_loader() -> Callable[[str], pl.DataFrame]:
    """`wooldridge`パッケージ（benchmark依存グループ）が無い環境ではskipする。

    tests本体はtest依存グループのみで完結させる方針（testing-policy.md、
    CLAUDE.md 3章「benchmark/はtests/とは別ライフサイクル」）のため、実データ
    クロスチェックのみ任意扱いにする。Wooldridgeデータはデータの再配布ライセンスが
    未確認のためCSVとして固定せず（`benchmark/freeze_datasets.py`のdocstring
    参照）、都度ロードする。

    複数のデータセット名を扱うテスト（`pytest.mark.parametrize`でデータセット名を
    振る等）向けにロード関数自体を返す。1件だけロードする場合は
    `load_wooldridge_dataset`を使う方が簡潔。
    """
    pytest.importorskip("wooldridge")
    from benchmark.common.load_wooldridge import load

    return load


def load_wooldridge_dataset(name: str) -> pl.DataFrame:
    """指定したWooldridgeデータセット1件をpolars DataFrameとしてロードする。"""
    return wooldridge_loader()(name)
