"""共有フィクスチャ。n=100, seed=42, 10 クラスターのデータセットを提供する。"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# 各テストファイルが個別に`sys.path.insert`していた`benchmark/`直下は、conftest.py
# （pytest起動時に最初に読み込まれる）で一度だけ挿入する。系統別サブディレクトリ
# （`benchmark/linear/fixtures`等）の挿入は手法ごとに異なるため各ファイルに残す。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))


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


@pytest.fixture(scope="module")
def binary_dataset(dataset: pl.DataFrame) -> pl.DataFrame:
    """共有`dataset`フィクスチャの`y`を中央値で0/1化した二値分類用データセット
    （Logit/Probit共通）。
    """
    median = dataset["y"].median()
    y_binary = (dataset["y"] > median).cast(pl.Float64)
    return dataset.with_columns(y_binary.alias("y"))
