"""`tests/panel/`（FE/RE）が共有するテストデータ。

- `hac_lag_panel_frame`: Driscoll-Kraayバンド幅の自動選択（`dk_bandwidth_used`）を
  検証する、時点数を指定できるバランスパネル。
"""

from __future__ import annotations

import numpy as np
import polars as pl


def hac_lag_panel_frame(n_periods: int, n_entities: int = 5) -> pl.DataFrame:
    """`n_entities`個体 × `n_periods`時点のバランスパネル（DKのバンド幅を
    時点数`t`ごとに確かめる用）。列は`entity`/`time`/`y`/`x1`/`x2`。
    """
    n = n_entities * n_periods
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pl.DataFrame(
        {
            "entity": np.repeat(np.arange(n_entities), n_periods),
            "time": np.tile(np.arange(n_periods), n_entities),
            "y": 1.0 + x1 - 0.5 * x2 + rng.normal(size=n),
            "x1": x1,
            "x2": x2,
        }
    )
