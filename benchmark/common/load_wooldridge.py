"""Wooldridge教科書データセットのロードヘルパー。

`wooldridge` PyPIパッケージ（pip install wooldridge）を使用する。同パッケージは
pandas DataFrameを返すため、ここでpolarsに変換する。
※ この変換はテストデータの読み込み時のみに使うものであり、econometricsmodels
本体のAPI（データ入力はpolars限定、CLAUDE.md 2章）とは無関係。

使用例:
    from benchmark.common.load_wooldridge import load

    df = load("wage1")
    print(df.columns)
"""

from __future__ import annotations

import polars as pl


def load(dataset_name: str) -> pl.DataFrame:
    """Wooldridgeデータセットをpolars DataFrameとして読み込む。

    Args:
        dataset_name: wooldridgeパッケージ内のデータセット名（例: "wage1"）。
            利用可能な一覧は `wooldridge.dataWoo("<name>")` や
            https://justinmshea.github.io/wooldridge/ を参照。

    Returns:
        polars DataFrame。
    """
    import wooldridge  # pip install wooldridge

    pandas_df = wooldridge.data(dataset_name)
    return pl.from_pandas(pandas_df)


def load_apple_adoption() -> pl.DataFrame:
    """Wooldridge `apple`に、`ecolbs > 0`の2値`y`（0.0/1.0）を足して返す。

    Logit/Probitの多クラスター実データケース用（州`state`でG=49）。Wooldridgeデータは
    CSVとして固定しないため、`y`はロードの都度ここで導出する。
    """
    return load("apple").with_columns(
        (pl.col("ecolbs") > 0).cast(pl.Float64).alias("y")
    )


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "wage1"
    print(load(name).head())
