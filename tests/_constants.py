"""系統によらず全テストが使う定数の集約。

`_helpers.py`/`_assertions.py`はファイル名が関数（ヘルパー・アサーション）を
示唆するが、実際には以下の定数も同居していた
（`refactoring-candidates-2.md`項目46）。`_tolerances.py`（`TOLERANCES`辞書
専用）と同じく、定数は関数と分離しこのファイルに集約する。

- `DATA_DIR`: 固定済み合成データセットCSV（`tests/fixtures/benchmarks/data/`）
  の置き場所（旧`_helpers.py`）。
- `MROZ_X`: Wooldridge mrozデータセットの説明変数リスト（旧`_helpers.py`）。
- `MARGEFF_AT`: `marginal_effects(at=...)`で検証する`at`の全パターン
  （旧`_assertions.py`、`check_margeff`が使う）。
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"

MROZ_X = ["nwifeinc", "educ", "exper", "expersq", "age", "kidslt6", "kidsge6"]

MARGEFF_AT = ["overall", "mean", "median"]
