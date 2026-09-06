"""IV 系統のテスト共通フィクスチャ。

`iv_dataset`/`clustered_dataset` は `test_iv_api.py`/`test_iv_validation.py` の
両方が使う（関心事分割前は `test_iv.py` 内で定義されていた、
`refactoring-candidates-2.md` 項目68）。`tests/conftest.py` の `dataset` 等と
違い IV 固有のため、系統ディレクトリ側の conftest に置く。
"""

from __future__ import annotations

import polars as pl
import pytest
from _constants import DATA_DIR
from _helpers import with_cluster_groups


@pytest.fixture(scope="module")
def iv_dataset() -> pl.DataFrame:
    """`test_iv_reference.py` と同じ固定済み CSV（内生変数 `endog1`・操作変数
    `z1`/`z2` を持つ、n=500 の合成データセット）を再利用する。
    """
    return pl.read_csv(DATA_DIR / "iv_baseline.csv")


@pytest.fixture(scope="module")
def clustered_dataset(iv_dataset: pl.DataFrame) -> pl.DataFrame:
    """`iv_dataset` に10グループの疑似クラスター列を付与したもの。"""
    return with_cluster_groups(iv_dataset, 10)
