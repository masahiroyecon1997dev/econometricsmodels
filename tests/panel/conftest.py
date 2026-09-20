"""panel系統（FE）のテスト共通フィクスチャ。

`fe_dataset`は`test_fe_api.py`/`test_fe_validation.py`の両方が使う。
`tests/conftest.py`の`dataset`等と違いFE固有のため、系統ディレクトリ側の
conftestに置く（`tests/iv/conftest.py`と同じ方針）。
"""

from __future__ import annotations

import polars as pl
import pytest
from _constants import DATA_DIR


@pytest.fixture(scope="module")
def fe_dataset() -> pl.DataFrame:
    """`test_fe_reference.py`と同じ固定済みCSV（baselineシナリオ、entity/time
    列を持つバランスパネル、n_entities=40 x n_periods=6）を再利用する。
    """
    return pl.read_csv(DATA_DIR / "fe_baseline.csv")
