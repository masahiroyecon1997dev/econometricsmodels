"""テストデータ生成・ロードの共通ヘルパー。

複数のテストファイルに重複していた以下を集約する。

- `with_cluster_groups`: 「行番号%N」の疑似クラスターラベル付与。
  `benchmark/common/dgp.py`の`imbalanced_cluster_groups`（不均衡クラスタ版）とは
  役割が近いが、これは均等サイズ版でテスト専用のロジックのため、`benchmark/`とは
  ライフサイクルが異なる`tests/`側に置く（`.claude/rules/testing-policy.md`
  「テストの分離」参照。ユーザー確認済み）。
- `separation_suspected_dataset`: 準完全分離データのDGP
  （`test_logit_validation.py`/`test_probit_validation.py`で完全に同一実装だった）。
- `load_wooldridge_dataset`: Wooldridgeデータセットのロード。`benchmark/
  load_wooldridge.py`の`load`を呼ぶだけの`wooldridge.data(name)`→
  `pl.from_pandas`実装が、複数ファイルに微妙に異なる書き方（直接呼び出し／
  `load_wooldridge.py`経由）で重複していた。
- `TIED_TIME_COLUMNS`: `hac_time`に同値を含む時点列（全値同一・各値2回・
  離れた1組のみ同値）とその最初の同値の行の組。OLS/WLS/IVのバリデーション
  テストで共有する。

- `ROW_TIME` / `with_row_time` / `hac_time_for`: HACの時間順序列`hac_time`は必須（行順を
  暗黙に使わない）なので、行順をそのまま時間順として使うテストが、行番号の列を明示的に
  足して`hac_time`に渡すための共通部品。

定数（`DATA_DIR`・`MROZ_X`）は`_constants.py`に分離済み
（ファイル名が関数を示唆するのに定数も同居していたための整理）。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

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
    """行番号を`n_groups`で割った余りを疑似クラスターラベルとして付与する。

    統計的な意味はなく、クラスターロバストSEの実装の動作確認用
    （`.claude/rules/testing-policy.md`「テスト用データセット」3.）。
    """
    return (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % n_groups).alias(col))
        .drop("_row")
    )


def separation_suspected_dataset() -> pl.DataFrame:
    """准完全分離データ（`x1`の真の係数を極端に大きくし、ほぼ全観測がx1の符号だけで
    完全に分類できるようにしたDGP）を生成する。

    `Logit`/`Probit`いずれのComputationError（`SeparationSuspected`）テストにも
    使う（Probit側もsigmoidベースのDGPをそのまま流用する。
    `test_logit_validation.py`のLogit版のProbit版という位置づけ、DGP自体の
    正確なProbitリンクである必要はない）。

    `benchmark/`側のDGP（`benchmark/nonlinear/datasets.py`等）と同じ
    `numpy`（`np.random.default_rng`）ベースのベクトル化演算で書く
    （以前は標準ライブラリ`random`＋素朴な`for`ループだった）。
    """
    rng = np.random.default_rng(42)
    n = 200
    beta = (0.0, 100.0, 0.5)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    z = beta[0] + beta[1] * x1 + beta[2] * x2
    p = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, p).astype(np.float64)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


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
