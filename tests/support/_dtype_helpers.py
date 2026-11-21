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

# 中に`UInt128`を持つネスト型と、メッセージに出るdtype名（最外の型名）。`List`・`Array`の
# 入れ子、複数フィールド・入れ子の`Struct`、多次元の`Array`を含め、判定の再帰をすべて通す。
NESTED_UINT128_DTYPES = [
    (pl.List(pl.UInt128), "List"),
    (pl.Array(pl.UInt128, 2), "Array"),
    (pl.Struct({"a": pl.UInt128}), "Struct"),
    (pl.List(pl.List(pl.UInt128)), "List"),
    (pl.Array(pl.List(pl.UInt128), 2), "Array"),
    (pl.List(pl.Array(pl.UInt128, 2)), "List"),
    (pl.Array(pl.UInt128, (2, 2)), "Array"),
    (pl.Struct({"a": pl.Int64, "b": pl.UInt128}), "Struct"),
    (pl.Struct({"a": pl.Struct({"b": pl.UInt128})}), "Struct"),
    (pl.List(pl.Struct({"a": pl.UInt128})), "List"),
    (pl.Struct({"a": pl.List(pl.UInt128)}), "Struct"),
    (pl.Struct({"UInt128": pl.UInt128}), "Struct"),
]

# 中に`UInt128`という名前の`Enum`・フィールドを持つが、`UInt128`そのものは含まないネスト型。
# 取り込めるdtypeなので、`UInt128`と取り違えてはいけない（ネスト型なのでどの役割にも使えず、
# 呼び名は最外の型名）。
NESTED_LOOKALIKE_DTYPES = [
    (pl.List(pl.Enum(["UInt128", "b"])), "List"),
    (pl.Struct({"a": pl.Enum(["UInt128", "b"])}), "Struct"),
]


class RegisteredExt(pl.BaseExtension):
    """`register_extension_type`で登録する拡張型。登録後は、`Extension`の列のdtypeが
    この利用者のクラスのインスタンスになり、クラス名が`Extension`ではなくなる。
    登録はプロセスの間有効（このモジュールをimportしたときに1度だけ行う）。
    """

    def __init__(self) -> None:
        super().__init__("registered_ext", pl.Int64)


pl.register_extension_type("registered_ext", RegisteredExt)

# `Extension`（ストレージ型によらず取り込めない）と、それを中に持つネスト型。メッセージに出る
# dtype名は、`Extension`そのものは`Extension`、ネスト型は最外の型名。登録した拡張型も
# `Extension`として拒否する。
EXTENSION_DTYPES = [
    (pl.Extension("my_ext", pl.Int64), "Extension"),
    (RegisteredExt(), "Extension"),
    (pl.List(RegisteredExt()), "List"),
    (pl.Extension("my_ext", pl.UInt128), "Extension"),
    (pl.List(pl.Extension("my_ext", pl.Int64)), "List"),
    (pl.Array(pl.Extension("my_ext", pl.Int64), 2), "Array"),
    (pl.Struct({"a": pl.Extension("my_ext", pl.Int64)}), "Struct"),
    (pl.List(pl.Struct({"a": pl.Extension("my_ext", pl.Int64)})), "List"),
]

# 推定に使わない列に置いても`fit()`を壊してはいけないdtype。
UNUSED_COLUMN_DTYPES = [
    *NUMERIC_DTYPES,
    pl.UInt128,
    pl.Boolean,
    pl.String,
    pl.Categorical,
    pl.Enum(["a", "b"]),
    # 名前が`UInt128`の`Enum`のカテゴリ・`Struct`のフィールド。ここへの追加は挙動を変えず、
    # 判定を変えたときの退行を見る確認用（誤拒否の再発を防ぐのは、`Enum`を受け付ける
    # `test_key_columns.py`のテスト）。
    pl.Enum(["UInt128", "b"]),
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
    pl.Binary,
    pl.List(pl.Int64),
    pl.Array(pl.Int64, 2),
    pl.Struct({"a": pl.Int64}),
    pl.Struct({"UInt128": pl.Int64}),
    *(dtype for dtype, _ in NESTED_UINT128_DTYPES),
    *(dtype for dtype, _ in NESTED_LOOKALIKE_DTYPES),
    *(dtype for dtype, _ in EXTENSION_DTYPES),
    pl.Null,
]

# 数値として使えないdtypeと、メッセージに出るdtype名。
NON_NUMERIC_DTYPES = [
    (pl.UInt128, "UInt128"),
    (pl.String, "String"),
    (pl.Categorical, "Categorical"),
    (pl.Enum(["a", "b"]), "Enum"),
    (pl.Enum(["UInt128", "b"]), "Enum"),
    (pl.Date, "Date"),
    (pl.Datetime, "Datetime"),
    (pl.Duration, "Duration"),
    (pl.Time, "Time"),
    (pl.Binary, "Binary"),
    (pl.List(pl.Int64), "List"),
    (pl.Array(pl.Int64, 2), "Array"),
    (pl.Struct({"a": pl.Int64}), "Struct"),
    (pl.Struct({"UInt128": pl.Int64}), "Struct"),
    *NESTED_UINT128_DTYPES,
    *NESTED_LOOKALIKE_DTYPES,
    *EXTENSION_DTYPES,
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
    if isinstance(dtype, pl.Enum):
        return pl.Series(
            "unused", dtype.categories.to_list()[:2] * (N // 2), dtype=dtype
        )
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
    if dtype == pl.Struct({"UInt128": pl.Int64}):
        return pl.Series("unused", [{"UInt128": 1}] * N)
    if dtype == pl.List(pl.UInt128):
        return pl.Series("unused", [[1, 2]] * N, dtype=dtype)
    if dtype == pl.Array(pl.UInt128, 2):
        return pl.Series("unused", [[1, 2]] * N, dtype=dtype)
    if dtype == pl.Struct({"a": pl.UInt128}):
        return pl.Series("unused", [{"a": 1}] * N, dtype=dtype)
    if any(
        dtype == nested
        for nested, _ in (
            *NESTED_UINT128_DTYPES,
            *NESTED_LOOKALIKE_DTYPES,
            *EXTENSION_DTYPES,
        )
    ):
        # `UInt128`・`Extension`を中に持つ値は作れる形が限られるので、全行が欠損の列にする
        # （dtypeの検査は欠損値の検査より先に行われる）。
        return pl.Series("unused", [None] * N, dtype=dtype)
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
