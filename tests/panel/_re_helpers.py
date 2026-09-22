"""`test_re_*.py`共通の小さなヘルパー（engineラッパー）。

`tests/panel/_fe_helpers.py`と同じ仕組みのRE版。
"""

from __future__ import annotations

import polars as pl
from econometricsmodels import RE, REOptions, REResults


def our_fit_re(
    df: pl.DataFrame,
    *,
    x: list[str] | None = None,
    entity: str = "entity",
    options: REOptions | None = None,
) -> REResults:
    """既定は`x=["x1", "x2"], entity="entity"`（`_fe_helpers.our_fit`と同じ既定、
    `benchmark/panel/datasets.py`が生成する列構成。RE専用の合成データセットは
    無く`fe_*.csv`を再利用するため列構成もFEと共通、`benchmark/panel/
    references/linearmodels_ref.py`モジュールdoc参照）。
    """
    kwargs = {}
    if options is not None:
        kwargs["options"] = options
    return RE(
        df,
        y="y",
        x=["x1", "x2"] if x is None else x,
        entity=entity,
        **kwargs,
    ).fit()
