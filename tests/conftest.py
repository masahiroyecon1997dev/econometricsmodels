"""共有フィクスチャ。n=100, seed=42, 10 クラスターのデータセットを提供する。"""

import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="session")
def dataset() -> pl.DataFrame:
    """再現可能な OLS テスト用データセット。

    y = 1.5 + 2.0*x1 - 0.5*x2 + ε, ε ~ N(0, 1)
    cluster: 0..9 を 10 周
    """
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    eps = rng.normal(0.0, 1.0, n)
    y = 1.5 + 2.0 * x1 - 0.5 * x2 + eps
    cluster = (np.arange(n) % 10).astype(np.int64)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "cluster": cluster})
