"""系統によらず全テストが使う定数の集約。

`_helpers.py`/`_assertions.py`はファイル名が関数（ヘルパー・アサーション）を
示唆するため、定数はこのファイルに置く（`_tolerances.py`は`TOLERANCES`辞書専用）。

- `DATA_DIR`: 固定済み合成データセットCSV（`tests/fixtures/benchmarks/data/`）
  の置き場所。
- `MROZ_X`: Wooldridge mrozデータセットの説明変数リスト。
- `APPLE_X`/`APPLE_CLUSTER`: Wooldridge appleデータセットの説明変数・クラスター列。
- `MARGEFF_AT`: `marginal_effects(at=...)`で検証する`at`の全パターン
  （`check_margeff`が使う）。

`DATA_DIR`・`APPLE_X`・`APPLE_CLUSTER`は、フィクスチャ生成側と値がずれないよう
`benchmark.common`が定義する値をそのまま公開する（テストはこのファイルから
import する）。
"""

from __future__ import annotations

from benchmark.common import APPLE_CLUSTER, APPLE_X, DATA_DIR

__all__ = ["APPLE_CLUSTER", "APPLE_X", "DATA_DIR", "MARGEFF_AT", "MROZ_X"]

MROZ_X = ["nwifeinc", "educ", "exper", "expersq", "age", "kidslt6", "kidsge6"]

MARGEFF_AT = ["overall", "mean", "median"]
