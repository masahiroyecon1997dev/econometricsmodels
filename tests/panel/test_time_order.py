"""FE/REのDriscoll-Kraay HAC（`cov_type="dk"`）が、時点の順序を列の値の順序で決めること。

時点の順序は、ラベルを文字列にした辞書順ではなく、列のdtypeの値の順序で決める
（整数・浮動小数は数値順、`Date`・`Datetime`は時系列順、`Enum`はカテゴリの定義順、
文字列・`Categorical`はラベルの辞書順）。ラベルが固定幅でない整数（`1, 10, 11, 2, ...`）や
負の数・小数を使っても、同じ時間順のラベルなら標準誤差は変わらない。

主リファレンスは、`test_fe_crosscheck.py`のfixest（FE・1-way/2-way）と、
`test_re_crosscheck.py`のplmで検証済みのゼロ埋め文字列ラベルの結果（RE）。既存のフィクスチャは
固定幅のラベル（`t00`〜`t24`）だけなので、同じ観測の時点列を固定幅でない形に付け替えて
同じ参照値と照合する。
"""

from __future__ import annotations

import datetime as dt
import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_dict_close
from _constants import DATA_DIR
from _tolerances import TOLERANCES
from econometricsmodels import FE, RE, FEOptions, REOptions

SCENARIO = "cross_sectionally_correlated"
N_PERIODS = 25

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "fe_crosscheck.json"
)
RTOL = TOLERANCES["fe_crosscheck"]["rtol"]
ATOL = TOLERANCES["fe_crosscheck"]["atol"]

_assert_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)

# 同じ時間順になる、固定幅でないラベルへの付け替え。`period`は`0..24`の整数。
_MONTH_STARTS = [
    dt.date(2019 + (8 + p) // 12, (8 + p) % 12 + 1, 1)
    for p in range(N_PERIODS)
]
_ENUM = pl.Enum([f"Q{p + 1}" for p in range(N_PERIODS)])

RELABELINGS = {
    # 10期以上の、ゼロ埋めのない整数（辞書順では 0, 1, 10, 11, ..., 2, ...）。
    "integer": pl.col("period"),
    "negative_integer": pl.col("period") - 30,
    "mixed_width_integer": pl.col("period") * 7 - 50,
    "unsigned_integer": pl.col("period").cast(pl.UInt8),
    "float": pl.col("period").cast(pl.Float64) + 0.5,
    "date": pl.col("period").replace_strict(
        range(N_PERIODS), _MONTH_STARTS, return_dtype=pl.Date
    ),
    "datetime": pl.col("period")
    .replace_strict(range(N_PERIODS), _MONTH_STARTS, return_dtype=pl.Date)
    .cast(pl.Datetime("ns")),
    # `Q1, ..., Q25`。辞書順なら `Q1, Q10, ...` だが、`Enum`はカテゴリの定義順。
    "enum": pl.concat_str(
        pl.lit("Q"), (pl.col("period") + 1).cast(pl.String)
    ).cast(_ENUM),
}


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    """`cross_sectionally_correlated`の固定済みCSV（15エンティティ×25時点）。

    `time`は`t00`〜`t24`の固定幅ラベルで、`period`にその番号（`0..24`）を持たせる。
    """
    df = pl.read_csv(DATA_DIR / f"fe_{SCENARIO}.csv")
    return df.with_columns(
        pl.col("time").str.slice(1).cast(pl.Int64).alias("period")
    )


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE_PATH.read_text())[SCENARIO]


def _x_columns(df: pl.DataFrame) -> list[str]:
    """回帰変数は、元のCSVの`x*`列だけ（付け替えた時点列は含めない）。"""
    return [c for c in df.columns if c.startswith("x")]


def _fit_fe(df: pl.DataFrame, column: str, *, two_way: bool):
    options = (
        FEOptions(cov_type="dk", time=column)
        if two_way
        else FEOptions(cov_type="dk", dk_time=column)
    )
    return FE(df, "y", _x_columns(df), "entity", options).fit()


def _fit_re(df: pl.DataFrame, column: str):
    options = REOptions(cov_type="dk", time=column)
    return RE(df, "y", _x_columns(df), "entity", options).fit()


@pytest.mark.parametrize("two_way", [False, True], ids=["one_way", "two_way"])
@pytest.mark.parametrize("relabeling", list(RELABELINGS))
def test_fe_dk_matches_fixest_for_any_labels_in_time_order(
    frame, reference, relabeling, two_way
):
    """固定幅でないラベルでも、FEのDKはfixestの参照値と一致する。"""
    df = frame.with_columns(RELABELINGS[relabeling].alias("label"))

    res = _fit_fe(df, "label", two_way=two_way)

    ref = reference["two_way" if two_way else "one_way"]["dk"]
    _assert_close(res.std_errors, ref["se"], f"{relabeling}/se")
    _assert_close(res.test_stats, ref["test_stats"], f"{relabeling}/t")
    _assert_close(res.p_values, ref["p_values"], f"{relabeling}/p")


@pytest.mark.parametrize("relabeling", list(RELABELINGS))
def test_re_dk_matches_the_zero_padded_labels(frame, relabeling):
    """REのDKは、plmで検証済みのゼロ埋め文字列ラベルの結果と一致する。"""
    df = frame.with_columns(RELABELINGS[relabeling].alias("label"))

    got = _fit_re(df, "label")
    expected = _fit_re(df, "time")

    for name in expected.std_errors:
        assert got.std_errors[name] == pytest.approx(
            expected.std_errors[name], rel=1e-12
        ), name
    assert got.f_statistic == pytest.approx(expected.f_statistic, rel=1e-12)
    assert got.hausman_statistic == pytest.approx(
        expected.hausman_statistic, rel=1e-12
    )


def test_unpadded_labels_sorted_as_text_would_give_a_different_result(frame):
    """文字列のラベルは辞書順のまま使うため、ゼロ埋めのない数字の文字列は時間順にならない。

    整数列と数字の文字列列は同じ観測でも結果が異なる（文字列は`1, 10, 11, ..., 2`の順）。
    整数列の結果が正しい順序によるものであることは、上のfixestとの一致が示している。
    """
    df = frame.with_columns(pl.col("period").cast(pl.String).alias("text"))

    as_integer = _fit_fe(frame, "period", two_way=False)
    as_text = _fit_fe(df, "text", two_way=False)

    differences = [
        abs(as_integer.std_errors[name] - as_text.std_errors[name])
        for name in as_integer.std_errors
    ]
    assert max(differences) > 1e-4


def test_categorical_labels_are_ordered_alphabetically(frame):
    """`Categorical`は順序を持たない型で、ラベルの辞書順になる（同じ値の文字列列と同じ）。"""
    labels = pl.concat_str(pl.lit("Q"), pl.col("period").cast(pl.String))
    df = frame.with_columns(
        labels.alias("text"), labels.cast(pl.Categorical).alias("cat")
    )

    as_text = _fit_fe(df, "text", two_way=False)
    as_categorical = _fit_fe(df, "cat", two_way=False)

    for name in as_text.std_errors:
        assert as_categorical.std_errors[name] == pytest.approx(
            as_text.std_errors[name], rel=1e-12
        )


@pytest.mark.parametrize("two_way", [False, True], ids=["one_way", "two_way"])
def test_dk_does_not_depend_on_row_order(frame, two_way):
    df = frame.with_columns(pl.col("period").alias("label"))
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=11)

    expected = _fit_fe(df, "label", two_way=two_way)
    got = _fit_fe(shuffled, "label", two_way=two_way)

    for name in expected.std_errors:
        assert got.std_errors[name] == pytest.approx(
            expected.std_errors[name], rel=1e-10
        ), name


# ── two-way FEの固定効果: 基準時点とキーの順序 ───────────────────────


def _time_effects(df: pl.DataFrame, column: str) -> dict:
    res = FE(
        df,
        "y",
        _x_columns(df),
        "entity",
        FEOptions(time=column),
    ).fit()
    return res.fixed_effects()["time"]


def test_two_way_time_effects_are_listed_in_time_order(frame):
    effects = _time_effects(frame, "period")

    assert list(effects) == [str(p) for p in range(N_PERIODS)]
    assert effects["0"] == 0.0  # 最初の時点が基準


def test_two_way_time_effects_follow_the_order_of_negative_and_float_labels(
    frame,
):
    negative = _time_effects(
        frame.with_columns((pl.col("period") - 30).alias("label")), "label"
    )
    floats = _time_effects(
        frame.with_columns(
            (pl.col("period").cast(pl.Float64) + 0.5).alias("label")
        ),
        "label",
    )

    assert list(negative) == [str(p - 30) for p in range(N_PERIODS)]
    assert negative["-30"] == 0.0
    assert list(floats) == [f"{p + 0.5}" for p in range(N_PERIODS)]
    assert floats["0.5"] == 0.0


def test_two_way_time_effects_values_do_not_depend_on_how_labels_are_written(
    frame,
):
    """同じ時間順なら、ラベルの書き方によらず時点効果・entity効果は同じ値になる。"""
    padded = frame.with_columns(
        pl.col("period").cast(pl.String).str.zfill(3).alias("label")
    )
    plain = frame.with_columns(pl.col("period").alias("label"))

    expected = FE(
        padded,
        "y",
        _x_columns(padded),
        "entity",
        FEOptions(time="label"),
    ).fit()
    got = FE(
        plain, "y", _x_columns(plain), "entity", FEOptions(time="label")
    ).fit()

    expected_time = list(expected.fixed_effects()["time"].values())
    got_time = list(got.fixed_effects()["time"].values())
    assert got_time == pytest.approx(expected_time, rel=1e-10, abs=1e-10)
    assert got.fixed_effects()["entity"] == pytest.approx(
        expected.fixed_effects()["entity"], rel=1e-10, abs=1e-10
    )
