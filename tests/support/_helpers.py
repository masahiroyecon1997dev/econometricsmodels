"""テストデータ生成・ロードの共通ヘルパー。

複数のテストファイルが共有する以下を置く。

- `with_cluster_groups`: 「行番号%N」の疑似クラスターラベル付与。
  `benchmark/common/dgp.py`の`imbalanced_cluster_groups`（不均衡クラスタ版）とは
  役割が近いが、これは均等サイズ版でテスト専用のロジックのため、`benchmark/`とは
  ライフサイクルが異なる`tests/`側に置く（`.claude/rules/testing-policy.md`
  「テストの分離」参照）。
- `load_wooldridge_dataset`: Wooldridgeデータセットのロード。`benchmark/
  load_wooldridge.py`の`load`を呼ぶだけの`wooldridge.data(name)`→
  `pl.from_pandas`を1箇所にまとめたもの。
- `TIED_TIME_COLUMNS`: `hac_time`に同値を含む時点列（全値同一・各値2回・
  離れた1組のみ同値）とその最初の同値の行の組。OLS/WLS/IVのバリデーション
  テストで共有する。

- `CASE_SPELLINGS` / `respell`: 文字列オプションの大文字小文字の綴りバリアント
  （`test_*_is_case_insensitive`が全手法で同じ網羅性になるよう共有する）。
- `ROW_TIME` / `with_row_time` / `hac_time_for`: HACの時間順序列`hac_time`は必須（行順を
  暗黙に使わない）なので、行順をそのまま時間順として使うテストが、行番号の列を明示的に
  足して`hac_time`に渡すための共通部品。

- `name_collision_frame`: 列名`const`・`sigma`が自動追加される合成パラメータ名と
  衝突する`ValidationError`テスト用の、`y`と衝突列だけの小さなDataFrame
  （IVは列構成が違うため`tests/support/_iv_helpers.py`の`iv_name_collision_frame`）。
- `reorder_rows`: `hac_time`の行順入れ替えテスト（OLS/WLS/IV）で、行の並べ替えだけを
  施したDataFrameを作る（時間順のDataFrame側は線形系統用の`_ols_helpers.py`の
  `hac_reorder_frame`）。

このファイルは全系統が使うものだけを置く。系統専用のものは`_nonlinear_helpers.py`・
`_panel_helpers.py`・`_ols_helpers.py`・`_iv_helpers.py`・`_fe_helpers.py`・
`_re_helpers.py`に置く。定数（`DATA_DIR`・`MROZ_X`）は`_constants.py`に置く。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl
import pytest

from benchmark.common import balanced_cluster_groups

# 行順をそのまま時間順として使うHACテスト用の、行番号の列名。
ROW_TIME = "row_time"


def with_row_time(df: pl.DataFrame) -> pl.DataFrame:
    """行番号（0始まり）の列`ROW_TIME`を足す。`hac_time=ROW_TIME`と組み合わせて、
    データの行順を時間順として使う（statsmodels等の行順HACとの照合用）。
    """
    return df.with_columns(pl.int_range(pl.len()).alias(ROW_TIME))


def hac_time_for(*settings: str) -> dict[str, str]:
    """`settings`（`cov_type`や`gmm_weight_type`の値）のどれかがHACなら
    `{"hac_time": ROW_TIME}`、そうでなければ空dict（`hac_time`は使われないと
    `ValidationError`になるため、HACのときだけ渡す）。
    """
    if any(setting.lower() == "hac" for setting in settings):
        return {"hac_time": ROW_TIME}
    return {}


def reorder_rows(df: pl.DataFrame, order: Sequence[int]) -> pl.DataFrame:
    """`df`の行を`order`（行番号の置換）の順に並べ替えた、中身の同じDataFrame。

    `hac_time`の行順入れ替えテストで「並べ替えただけ」を構成上保証するために使う。
    `order`が恒等置換（並べ替えにならない）または置換でない場合は`ValueError`。
    """
    n = df.height
    if sorted(order) != list(range(n)):
        raise ValueError(f"order is not a permutation of range({n}): {order}")
    if list(order) == list(range(n)):
        raise ValueError("order is the identity and does not reorder rows")
    return df[list(order)]


def name_collision_frame(
    y: Sequence[float], collided: str = "const"
) -> pl.DataFrame:
    """`y`と、自動追加される合成パラメータ名（`"const"`・`"sigma"`）と同名の列
    `collided`だけを持つ、列名衝突の`ValidationError`テスト用のDataFrame。

    衝突列は`y`と同じ行数で、最後の値だけ0.5ずらした1, 2, 3, ...（定数列にならず、
    `y`とも一致しない）。衝突は入力の列名だけで判定されるため、値は問わない。
    """
    n = len(y)
    values = [float(i + 1) for i in range(n)]
    values[-1] += 0.5
    return pl.DataFrame({"y": list(y), collided: values})


def _alternating_case(label: str) -> str:
    return "".join(
        char.upper() if i % 2 else char.lower() for i, char in enumerate(label)
    )


# 文字列オプション（`cov_type`・`gmm_weight_type`等）の大文字小文字の綴り方。
# 正規形（小文字）・全大文字・先頭のみ大文字・大小交互の4通りで、
# `test_*_is_case_insensitive`が全ての値に同じ網羅性で適用する。
CASE_SPELLINGS = {
    "lower": str.lower,
    "upper": str.upper,
    "title": str.title,
    "alternating": _alternating_case,
}


def respell(label: str, spelling: str) -> str:
    """`label`を`CASE_SPELLINGS[spelling]`の綴りに変換する。"""
    return CASE_SPELLINGS[spelling](label)


# `hac_lags_used`/`dk_bandwidth_used`の自動選択ラグ（`floor(4*(n/100)^(2/9))`）を
# `benchmark.common.hac_auto_lag`と直接突き合わせる標本サイズ。式の値が切り替わる前後
# （n=100は丁度4、n=8・20・500・1000は値が異なる）と、厳密値が整数（16）になる
# 境界n=51200（浮動小数点では15.999…になり、RustとPythonで同じく15に床される）を含む。
HAC_AUTO_LAG_SAMPLE_SIZES = [8, 20, 100, 500, 1000, 51200]


def hac_lag_frame(n: int, seed: int = 0) -> pl.DataFrame:
    """`n`行の合成データ（HACの自動選択ラグを標本サイズごとに確かめる用）。

    列は`y`/`x1`/`x2`/`endog1`/`z1`/`z2`/`weight`（OLS/WLS/IVの各既定の
    列構成）と、行番号の`ROW_TIME`。統計的な性質は問わない（推定が特異に
    ならず走ればよい）。
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    endog1 = z1 + 0.5 * z2 + rng.normal(size=n)
    y = 1.0 + x1 - 0.5 * x2 + endog1 + rng.normal(size=n)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "endog1": endog1,
            "z1": z1,
            "z2": z2,
            "weight": rng.uniform(0.5, 2.0, size=n),
        }
    ).pipe(with_row_time)


# 同値を含む時点列を作るpolars式と、最初に報告される同値の行の組（0始まり）。
# 行順に黙ってフォールバックして時系列順のHACに見えてしまう入力の代表。
TIED_TIME_COLUMNS = [
    pytest.param(pl.lit(1), (0, 1), id="all_identical"),
    pytest.param(pl.int_range(pl.len()) // 2, (0, 1), id="each_twice"),
    # 行0と行2だけが同値。間に別の値を挟んでも、行順ではなく値で重複を見つける。
    pytest.param(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(0)
        .otherwise(pl.int_range(pl.len())),
        (0, 2),
        id="one_pair_not_adjacent",
    ),
]


def with_cluster_groups(
    df: pl.DataFrame, n_groups: int, col: str = "cluster_group"
) -> pl.DataFrame:
    """行番号を`n_groups`で割った余りを疑似クラスターラベル（整数）として付与する
    （`benchmark.common.balanced_cluster_groups`と同じラベル）。

    統計的な意味はなく、クラスターロバストSEの実装の動作確認用
    （`.claude/rules/testing-policy.md`「テスト用データセット」3.）。
    """
    return df.with_columns(
        pl.Series(
            col, balanced_cluster_groups(df.height, n_groups), dtype=pl.Int64
        )
    )


def wooldridge_loader() -> Callable[[str], pl.DataFrame]:
    """`wooldridge`パッケージ（test依存グループ）を使ってロード関数を返す。

    `wooldridge`はtest依存グループに含まれ標準CIで常にインストールされるため、
    通常はskipされない。`pytest.importorskip`は、想定外の理由でインストールが
    欠けた環境（test依存グループを経由しないpytest実行等）向けの防御的フォール
    バックとして残している。Wooldridgeデータはデータの再配布ライセンスが未確認
    のためCSVとして固定せず（`benchmark/linear/freeze.py`のdocstring参照）、
    都度ロードする。

    複数のデータセット名を扱うテスト（`pytest.mark.parametrize`でデータセット名を
    振る等）向けにロード関数自体を返す。1件だけロードする場合は
    `load_wooldridge_dataset`を使う方が簡潔。
    """
    pytest.importorskip("wooldridge")
    from benchmark.common.load_wooldridge import load

    return load


def load_wooldridge_dataset(name: str) -> pl.DataFrame:
    """指定したWooldridgeデータセット1件をpolars DataFrameとしてロードする。"""
    return wooldridge_loader()(name)
