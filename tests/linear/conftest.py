"""`tests/linear/`共通のフィクスチャ。"""

import polars as pl
import pytest
from _constants import DATA_DIR
from _helpers import with_cluster_groups


@pytest.fixture(scope="module")
def unit_weight_dataset(dataset: pl.DataFrame) -> pl.DataFrame:
    """共有`dataset`フィクスチャに、全て1の重み列`weight`を足したもの（WLS用）。

    重み=1のWLSはOLSと同じ推定結果になるため、WLS固有の重みの扱いを調べない
    テスト（バリデーション・オプションの反映・OLSとの不変条件）が使う。
    """
    return dataset.with_columns(pl.lit(1.0).alias("weight"))


@pytest.fixture(scope="session")
def wide_dataset() -> pl.DataFrame:
    """説明変数8列（x1..x8）の凍結済み合成データ（n=100、seed=42）。

    列数・列順に依存するテスト（`param_names`の並び、`params_table`/`conf_int`の
    キー、`predict()`/`augment()`の列の扱い等）で、共有`dataset`（2列）では
    1通りの入れ替えしか起こり得ず検出できない取り違えを拾うために使う。
    切片を含む真の係数は全て0.75以上離してある（`benchmark.linear.constants.WIDE_BETA`、
    切片＋x1..x8の順）。列は`y`/`x1..x8`/`weight`（WLS用の0.5〜1.5の一様乱数）に、
    `dataset`と同じ行番号%10の疑似クラスター列`cluster`を足したもの。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_wide.csv")
    return with_cluster_groups(df, 10, col="cluster")
