"""`tests/panel/`（FE/RE）が共有するテストデータ。

- `hac_lag_panel_frame`: Driscoll-Kraayバンド幅の自動選択（`dk_bandwidth_used`）を
  検証する、時点数を指定できるバランスパネル。
- `x_columns`: パネルのシナリオCSV（`entity`/`time`/`y`と説明変数）の説明変数列。
- `bandwidth_cases`: シナリオごとのDriscoll-Kraayバンド幅のリストを
  `(シナリオ, バンド幅)`の組に展開する（`pytest.mark.parametrize`用）。
"""

from __future__ import annotations

import numpy as np
import polars as pl


def x_columns(df: pl.DataFrame) -> list[str]:
    """`entity`/`time`/`y`以外の列（説明変数）を列順のまま返す。"""
    return [c for c in df.columns if c not in ("y", "entity", "time")]


def bandwidth_cases(
    bandwidths_by_scenario: dict[str, list[int]],
) -> list[tuple[str, int]]:
    """`{シナリオ: [バンド幅, ...]}`を`(シナリオ, バンド幅)`の組に展開する。"""
    return [
        (scenario, bandwidth)
        for scenario, bandwidths in bandwidths_by_scenario.items()
        for bandwidth in bandwidths
    ]


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
