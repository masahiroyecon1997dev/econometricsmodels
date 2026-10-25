"""`tests/cross_method/`共通のフィクスチャ。

`panel_df`・`panel_df_unbalanced`は、FE/REの自由度・分布を調べるテスト
（`test_stat_dist.py`・`test_test_dfs.py`）が使う。小さな60行のデータ
（`_dtype_helpers.make_frame`）を使うテストとは別のフィクスチャ名にしてある。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _cross_method_helpers import (
    N_ENTITIES,
    N_GROUPS,
    N_PERIODS,
    N,
)


@pytest.fixture(scope="module")
def panel_df() -> pl.DataFrame:
    """20 entity × 10 時点の均衡パネル（200行）。全手法で使える列を持つ。

    `y`（連続）・`y_bin`（二値）・`y_cens`（0.5で左打ち切り）・`x1`・`x2`・
    操作変数`z1`/`z2`/`z3`・内生変数`endog`・重み`w`・`entity`・`time`と、
    クラスター列`grp`・`coarse`・粗い時点列`time5`。
    """
    rng = np.random.default_rng(425)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    z1 = rng.normal(size=N)
    z2 = rng.normal(size=N)
    z3 = rng.normal(size=N)
    endog = 0.5 * z1 + 0.5 * z2 + rng.normal(size=N)
    latent = 1.0 + x1 - 0.5 * x2 + 0.5 * endog + rng.normal(size=N)
    return pl.DataFrame(
        {
            "y": latent,
            "y_bin": (latent > 1.0).astype(int),
            "y_cens": np.maximum(latent, 0.5),
            "x1": x1,
            "x2": x2,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "endog": endog,
            "w": rng.uniform(0.5, 2.0, size=N),
            "entity": entity,
            "time": np.tile(np.arange(N_PERIODS), N_ENTITIES),
            # entityとは別軸のクラスター列。`grp`はentityと入れ子にならない
            # （G=7、entity内で値が変わる）、`coarse`はentityを束ねる入れ子
            # （G=5）。どちらもGがentity数・時点数・`df_resid`のいずれとも
            # 異なる値になるようにして、自由度の取り違えを区別できる。
            "grp": np.arange(N) % N_GROUPS,
            "coarse": entity // 4,
            # 時点を2つずつ束ねた列（T=5）。`time`（T=10）と別のTを持つので、
            # dkの時点数`T-1`と`cluster="time"`の`G-1`（どちらも`time`だと9）を
            # 区別できる。
            "time5": np.tile(np.arange(N_PERIODS), N_ENTITIES) // 2,
        }
    )


@pytest.fixture(scope="module")
def panel_df_unbalanced(panel_df: pl.DataFrame) -> pl.DataFrame:
    """`panel_df`から時点を欠かせた不均衡パネル。クラスターのサイズも偏らせる。

    時点0は全entityで欠測し、entityごとに観測数が異なる。`grp_u`は非入れ子で
    偏ったクラスター（6群、サイズは概ね3%〜50%）、`coarse_u`はentityを束ねる
    入れ子で偏ったクラスター（サイズ10/4/3/2/1 entity）。
    """
    rng = np.random.default_rng(426)
    sub = panel_df.filter(
        (pl.col("time") >= 1) & (pl.col("time") < 4 + (pl.col("entity") % 6))
    )
    grp = rng.choice(
        6, size=sub.height, p=[0.03, 0.05, 0.07, 0.10, 0.25, 0.50]
    )
    assert len(set(grp.tolist())) == 6
    entity = sub["entity"].to_numpy()
    coarse = np.select(
        [entity < 10, entity < 14, entity < 17, entity < 19], [0, 1, 2, 3], 4
    )
    return sub.with_columns(
        pl.Series("grp_u", grp), pl.Series("coarse_u", coarse)
    )
