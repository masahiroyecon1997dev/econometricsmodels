"""Wooldridge教科書データセットのロードヘルパー。

`wooldridge` PyPIパッケージ（pip install wooldridge）を使用する。同パッケージは
pandas DataFrameを返すため、ここでpolarsに変換する。
※ この変換はテストデータの読み込み時のみに使うものであり、econometricsmodels
本体のAPI（データ入力はpolars限定、CLAUDE.md 2章）とは無関係。

使用例:
    from load_wooldridge import load

    df = load("wage1")
    print(df.columns)
"""

from __future__ import annotations

import polars as pl

# 手法ごとに適切なデータセットの候補（要検討・要確定）。
# 実際に採用するデータセットはモデル実装時に個別に確認する。
SUGGESTED_DATASETS: dict[str, list[str]] = {
    "ols": ["wage1", "gpa2"],
    "wls": ["hprice1"],
    "iv": ["mroz", "card"],
    "probit_logit": ["mroz"],
}


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


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "wage1"
    print(load(name).head())
