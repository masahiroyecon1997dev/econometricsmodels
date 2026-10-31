"""`test_fe_*.py`共通の小さなヘルパー（engineラッパー）。

`tests/support/_iv_helpers.py`と同じ位置づけの、FE用ヘルパー。
"""

from __future__ import annotations

import polars as pl
from econometricsmodels import FE, FEOptions, FEResults


def our_fit(
    df: pl.DataFrame,
    *,
    x: list[str] | None = None,
    effects: list[str] | None = None,
    options: FEOptions | None = None,
) -> FEResults:
    """既定は`x=["x1", "x2"], effects=["entity"]`（FEテストの大半が使う共通
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
        effects=["entity"] if effects is None else effects,
        **kwargs,
    ).fit()
