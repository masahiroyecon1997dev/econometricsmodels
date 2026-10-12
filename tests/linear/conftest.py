"""`tests/linear/`共通のフィクスチャ。"""

import polars as pl
import pytest


@pytest.fixture(scope="module")
def unit_weight_dataset(dataset: pl.DataFrame) -> pl.DataFrame:
    """共有`dataset`フィクスチャに、全て1の重み列`weight`を足したもの（WLS用）。

    重み=1のWLSはOLSと同じ推定結果になるため、WLS固有の重みの扱いを調べない
    テスト（バリデーション・オプションの反映・OLSとの不変条件）が使う。
    """
    return dataset.with_columns(pl.lit(1.0).alias("weight"))
