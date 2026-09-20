"""`test_fe_*.py`共通の小さなヘルパー（engineラッパー）。

`tests/iv/_iv_helpers.py`と同じ仕組みの系統ディレクトリ版。
"""

from __future__ import annotations

import polars as pl
from econometricsmodels import FE, FeOptions, FeResults


def our_fit(
    df: pl.DataFrame,
    *,
    x: list[str] | None = None,
    entity: str = "entity",
    options: FeOptions | None = None,
) -> FeResults:
    """既定は`x=["x1", "x2"], entity="entity"`（FEテストの大半が使う共通
    パターン、`benchmark/panel/datasets.py`が生成する列構成）。異なる変数
    構成が必要なテストのみ明示的に上書きする。
    """
    kwargs = {}
    if options is not None:
        kwargs["options"] = options
    return FE(
        df,
        y="y",
        x=["x1", "x2"] if x is None else x,
        entity=entity,
        **kwargs,
    ).fit()
