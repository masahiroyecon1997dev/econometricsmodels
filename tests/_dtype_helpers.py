"""dtype・入力型のテスト（`test_input_dtypes.py`・`test_key_columns.py`・
`test_data_input_types.py`）が共有するヘルパー。
"""

from __future__ import annotations

import numpy as np
import polars as pl

N = 60

# 整数値の列として作ってから各dtypeへキャストする。値は0..99で、Int8/UInt8の範囲
# （最大127）とFloat16で正確に表せる範囲（整数は2048まで）に収まる。
NUMERIC_DTYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float16,
    pl.Float32,
    pl.Float64,
    pl.Decimal(18, 0),
]

# 推定に使わない列に置いても`fit()`を壊してはいけないdtype。
UNUSED_COLUMN_DTYPES = [
    *NUMERIC_DTYPES,
    pl.Boolean,
    pl.String,
    pl.Categorical,
    pl.Enum(["a", "b"]),
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
    pl.Binary,
    pl.List(pl.Int64),
    pl.Array(pl.Int64, 2),
    pl.Struct({"a": pl.Int64}),
    pl.Null,
]

# 数値として使えないdtypeと、メッセージに出るdtype名。
NON_NUMERIC_DTYPES = [
    (pl.String, "String"),
    (pl.Categorical, "Categorical"),
    (pl.Enum(["a", "b"]), "Enum"),
    (pl.Date, "Date"),
    (pl.Datetime, "Datetime"),
    (pl.Duration, "Duration"),
    (pl.Time, "Time"),
    (pl.Binary, "Binary"),
    (pl.List(pl.Int64), "List"),
    (pl.Array(pl.Int64, 2), "Array"),
    (pl.Struct({"a": pl.Int64}), "Struct"),
]


def dtype_id(dtype: pl.DataType) -> str:
    return str(dtype)


def unused_column(dtype: pl.DataType) -> pl.Series:
    """推定に使わない列（`n`行）をdtypeごとに作る。"""
    if dtype == pl.Null:
        return pl.Series("unused", [None] * N, dtype=pl.Null)
    if dtype == pl.Boolean:
        return pl.Series("unused", [True, False] * (N // 2))
    if dtype == pl.String:
        return pl.Series("unused", ["a", "b"] * (N // 2))
    if dtype == pl.Enum(["a", "b"]):
        return pl.Series("unused", ["a", "b"] * (N // 2), dtype=dtype)
    if dtype == pl.Categorical:
        return pl.Series("unused", ["a", "b"] * (N // 2), dtype=pl.Categorical)
    if dtype == pl.Date:
        return pl.Series("unused", [0] * N, dtype=pl.Int32).cast(pl.Date)
    if dtype == pl.Datetime:
        return pl.Series("unused", [0] * N, dtype=pl.Int64).cast(pl.Datetime)
    if dtype == pl.Duration:
        return pl.Series("unused", [0] * N, dtype=pl.Int64).cast(pl.Duration)
    if dtype == pl.Time:
        return pl.Series("unused", [0] * N, dtype=pl.Int64).cast(pl.Time)
    if dtype == pl.Binary:
        return pl.Series("unused", [b"a", b"b"] * (N // 2))
    if dtype == pl.List(pl.Int64):
        return pl.Series("unused", [[1, 2]] * N)
    if dtype == pl.Array(pl.Int64, 2):
        return pl.Series("unused", [[1, 2]] * N, dtype=dtype)
    if dtype == pl.Struct({"a": pl.Int64}):
        return pl.Series("unused", [{"a": 1}] * N)
    return pl.Series("unused", np.arange(N) % 100, dtype=pl.Int64).cast(dtype)


def make_frame(seed: int = 7) -> pl.DataFrame:
    """全手法で使える、10エンティティ×6期の均衡パネル（`n=60`）。

    `y`（連続）・`y01`（二値）・`yc`（0で左打ち切り）・`x1`・`x2`・操作変数`z`/`z2`・
    重み`w`・`entity`・`t`（時点0..5）・`cluster`（10グループ）を含む。
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, N)
    x2 = rng.normal(0.0, 1.0, N)
    y = 1.0 + x1 - 0.5 * x2 + rng.normal(0.0, 1.0, N)
    return pl.DataFrame(
        {
            "y": y,
            "y01": (y > np.median(y)).astype(np.float64),
            "yc": np.clip(y, 0.0, None),
            "x1": x1,
            "x2": x2,
            "z": x1 + rng.normal(0.0, 1.0, N),
            "z2": x1 + rng.normal(0.0, 1.0, N),
            "w": np.ones(N),
            "entity": np.arange(N) // 6,
            "t": np.arange(N) % 6,
            "cluster": np.arange(N) % 10,
        }
    )
