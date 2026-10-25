"""クラスターを動的に付与する凍結CSVとクラスター列の共有定義（生成側・テスト側）。

クラスター列はCSVに含めず（entityとは無関係な専用列のため）、フィクスチャ生成・
テスト時にここの関数で都度付与する。CSV名は`freeze.py`が凍結するときの
`filename_suffix`と同じ接尾辞の定数から作り、`freeze.py`もその定数を使う
（名前が二重にならないようにする）。
"""

from __future__ import annotations

import polars as pl

from benchmark.common import DATA_DIR, imbalanced_cluster_groups

# クラスター不均衡シナリオ（baselineをentity=20×period=10のn=200にしたもの）。
CLUSTER_IMBALANCED_SUFFIX = "_cluster_imbalanced"
CLUSTER_IMBALANCED_CSV = f"fe_baseline{CLUSTER_IMBALANCED_SUFFIX}.csv"

# baselineを説明変数1個（k=1）に絞ったシナリオ。クラスター数の境界
# （G=2、G=3）の成功パスに使う。
K1_SUFFIX = "_k1"
K1_CSV = f"fe_baseline{K1_SUFFIX}.csv"


def cluster_imbalanced_groups() -> list[str]:
    """`CLUSTER_IMBALANCED_CSV`の行順に並んだ、サイズ[2,3,5,10,30,50]のタイルの
    クラスター列（entityとは無関係）。"""
    n = pl.read_csv(DATA_DIR / CLUSTER_IMBALANCED_CSV).height
    return imbalanced_cluster_groups(n)


def cluster_imbalanced_frame(
    cluster_col: str = "cluster_group",
) -> pl.DataFrame:
    """`CLUSTER_IMBALANCED_CSV`に、`cluster_imbalanced_groups()`のクラスター列
    （列名`cluster_col`）を付けたデータ。"""
    return pl.read_csv(DATA_DIR / CLUSTER_IMBALANCED_CSV).with_columns(
        pl.Series(cluster_col, cluster_imbalanced_groups())
    )


def cluster_g3_groups() -> list[str]:
    """`K1_CSV`の行順に並んだ3グループ（行番号%3）のクラスター列。"""
    n = pl.read_csv(DATA_DIR / K1_CSV).height
    return [str(i % 3) for i in range(n)]
