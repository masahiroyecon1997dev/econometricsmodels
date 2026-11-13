"""`FEIV`の入力・オプションの検証（`ValidationError`）と計算上のエラー（`ComputationError`）。

エラーは`code`と`params`で照合する（文面は照合しない。`testing-policy.md`「テストの3系統」）。
FEIVはFEと同じ固定効果の検証（singleton・自由度・クラスター数・DK）を持つうえに、操作変数
モデル固有の検証（第一段階の失敗に`stage`と`endog_name`が付く、ロールごとの`role`、操作変数の数、
Kleibergen-Paap・有効Fの未定義）を持つ。境界の失敗側のテストは、反対側の成功パスと対にして
同じ見出しに置く。

役割分担:
    - 入力の検証・エラーの`code`と`params`: このファイル
    - 横断的な検査（列の型・欠損値・列の順序）: `tests/cross_method/`
    - `_lib`の配線・オプション・pickle・コピー: `test_fe_iv_binding.py`
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _constants import DATA_DIR
from _errors import raises_error
from econometricsmodels import (
    FEIV,
    ComputationError,
    ErrorCode,
    FEIVOptions,
    ValidationError,
)

N_ENTITIES = 12
N_PERIODS = 5
N = N_ENTITIES * N_PERIODS


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    """12個体×5時点の均衡パネル（内生変数1個`a`、操作変数`z1`・`z2`・`z3`、外生変数`x1`・`x2`）。
    時点の別名`dkt`はDKの時点列用（`time`と同じ値）。"""
    rng = np.random.default_rng(5)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    time = np.tile(np.arange(N_PERIODS), N_ENTITIES)
    alpha = rng.normal(size=N_ENTITIES)[entity]
    z1, z2, z3, x1, x2, u = (rng.normal(size=N) for _ in range(6))
    a = 0.8 * z1 + 0.4 * z2 + 0.5 * u + alpha + rng.normal(size=N)
    y = a + x1 + u + alpha
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "a": a,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "entity": entity,
            "time": time,
            "dkt": time,
        }
    )


def _fit(
    data: pl.DataFrame,
    *,
    y: str = "y",
    x_exog: list[str] | None = None,
    x_endog: list[str] | None = None,
    instruments: list[str] | None = None,
    effects: list[str] | None = None,
    **options,
):
    return FEIV(
        data,
        y=y,
        x_exog=["x1"] if x_exog is None else x_exog,
        x_endog=["a"] if x_endog is None else x_endog,
        instruments=["z1", "z2"] if instruments is None else instruments,
        effects=["entity"] if effects is None else effects,
        options=FEIVOptions(**options),
    ).fit()


def _panel(n_entities: int, n_periods: int, seed: int = 9) -> pl.DataFrame:
    """小さな均衡パネル（`y`・`x1`・`a`・`z1`・`z2`・`entity`・`time`）。"""
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    z1, z2, x1, u = (rng.normal(size=n) for _ in range(4))
    a = 0.8 * z1 + 0.4 * z2 + 0.5 * u + rng.normal(size=n)
    y = a + x1 + u + rng.normal(size=n_entities)[entity]
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "a": a,
            "z1": z1,
            "z2": z2,
            "entity": entity,
            "time": time,
        }
    )


# ── ValidationError（入力データ: 変数ロール） ─────────────────────────


@pytest.mark.parametrize("role", ["x_endog", "instruments"])
def test_empty_x_endog_or_instruments_raises(df, role):
    with raises_error(ErrorCode.ROLE_EMPTY, argument=role):
        _fit(df, **{role: []})


def test_x_exog_may_be_empty(df):
    assert _fit(df, x_exog=[]).param_names == ["a"]


def test_too_few_instruments_raises_and_equal_count_is_accepted(df):
    with raises_error(ErrorCode.INSUFFICIENT_INSTRUMENTS):
        _fit(df, x_endog=["a", "x2"], instruments=["z1"])
    # 操作変数が内生変数と同数（丁度識別）なら成功する。
    assert (
        _fit(df, x_endog=["a", "x2"], instruments=["z1", "z2"]).overid_df
        is None
    )


def test_a_column_in_two_roles_raises(df):
    with raises_error(ErrorCode.ROLE_COLUMN_OVERLAP):
        _fit(df, instruments=["z1", "x1"])


def test_duplicate_column_within_a_role_raises(df):
    with raises_error(ErrorCode.ROLE_DUPLICATE_COLUMN):
        _fit(df, instruments=["z1", "z1"])


def test_effects_limit_names_the_model():
    """`effects`は5列まで。水準が3つの次元を足す（次元どうしが同じ分け方にならない）。"""
    data = _panel(12, 6).with_columns(
        *[
            pl.Series(
                f"e{k}",
                [
                    (i * (k + 1) + t * (k + 2)) % 3
                    for i in range(12)
                    for t in range(6)
                ],
            )
            for k in range(4)
        ]
    )
    five = ["entity", "time", "e0", "e1", "e2"]
    assert _fit(data, effects=five).n_groups.keys() == set(five)
    with raises_error(ErrorCode.EFFECTS_TOO_MANY, model="FEIV", max_columns=5):
        _fit(data, effects=[*five, "e3"])


# ── ValidationError（入力データ: 列の存在・欠損値） ─────────────────────


@pytest.mark.parametrize(
    ("argument", "kwargs"),
    [
        ("y", {"y": "nope"}),
        ("x_exog", {"x_exog": ["x1", "nope"]}),
        ("x_endog", {"x_endog": ["nope"]}),
        ("instruments", {"instruments": ["z1", "nope"]}),
        ("effects", {"effects": ["nope"]}),
        ("effects", {"effects": ["entity", "time", "nope"]}),
    ],
    ids=[
        "y",
        "x_exog",
        "x_endog",
        "instruments",
        "effects_first",
        "effects_third",
    ],
)
def test_a_missing_column_names_its_role(df, argument, kwargs):
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND, column="nope", argument=argument
    ):
        _fit(df, **kwargs)


def test_missing_cluster_and_dk_time_columns_raise(df):
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND, column="nope", argument="cluster"
    ):
        _fit(df, cov_type="cluster", cluster="nope")
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND, column="nope", argument="dk_time"
    ):
        _fit(df, cov_type="dk", dk_time="nope")


def test_a_null_in_a_later_effect_column_raises(df):
    third = pl.Series("sector", ["s"] * N).scatter(0, None)
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL, column="sector", argument="effects"
    ):
        _fit(df.with_columns(third), effects=["entity", "time", "sector"])


# ── ValidationError（入力データ: 定数の列・固定効果を除くと一定の列） ────


# 列`extra`を足した（`x_endog`は置き換えた）ときの、ロールごとの引数。
_WITH_EXTRA_COLUMN = {
    "x_exog": {"x_exog": ["x1", "extra"]},
    "x_endog": {"x_endog": ["extra"]},
    "instruments": {"instruments": ["z1", "extra"]},
}


@pytest.mark.parametrize("role", list(_WITH_EXTRA_COLUMN))
def test_a_column_with_one_value_in_every_row_raises(df, role):
    """生の値が全て同じ列は、固定効果を除く前に`REGRESSOR_CONSTANT`（`role`付き）。"""
    with raises_error(ErrorCode.REGRESSOR_CONSTANT, role=role, column="extra"):
        _fit(df.with_columns(extra=pl.lit(1.5)), **_WITH_EXTRA_COLUMN[role])


@pytest.mark.parametrize("role", list(_WITH_EXTRA_COLUMN))
def test_a_column_constant_within_entities_raises(df, role):
    """個体ごとに一定の列は、固定効果を除くと分散が無いため`REGRESSOR_NO_WITHIN_VARIATION`。"""
    level = np.random.default_rng(2).normal(size=N_ENTITIES)
    data = df.with_columns(extra=pl.Series(np.repeat(level, N_PERIODS)))
    with raises_error(
        ErrorCode.REGRESSOR_NO_WITHIN_VARIATION, role=role, column="extra"
    ):
        _fit(data, **_WITH_EXTRA_COLUMN[role])


def test_a_time_invariant_column_is_accepted_with_a_time_only_effect(df):
    """時点の固定効果だけを入れても、個体ごとに一定の列は個体内の変動が残らないが、
    時点を除いた後の変動は残る（個体の効果を入れた1-wayとは別）。"""
    level = np.random.default_rng(2).normal(size=N_ENTITIES)
    data = df.with_columns(extra=pl.Series(np.repeat(level, N_PERIODS)))
    res = _fit(data, x_exog=["x1", "extra"], effects=["time"])
    assert "extra" in res.params


def test_a_constant_outcome_raises(df):
    with raises_error(ErrorCode.OUTCOME_CONSTANT):
        _fit(df.with_columns(y=pl.lit(2.0)))


def test_an_outcome_constant_within_entities_raises(df):
    level = np.random.default_rng(3).normal(size=N_ENTITIES)
    with raises_error(ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY):
        _fit(df.with_columns(y=pl.Series(np.repeat(level, N_PERIODS))))


# ── ValidationError（入力データ: singleton・自由度） ────────────────────


def test_a_singleton_entity_raises(df):
    data = df.filter((pl.col("entity") != 0) | (pl.col("time") == 0))
    with raises_error(
        ErrorCode.SINGLETON_GROUP, argument="effects", column="entity"
    ):
        _fit(data)


def test_insufficient_degrees_of_freedom_names_the_first_stage_and_boundary():
    """3個体×2時点（n=6）は、第一段階（`x1`・`z1`・`z2`と3個体の効果）の自由度が足りず失敗する。
    4個体×2時点（n=8）は成功する。第一段階の設計は構造式より列が多いので、この境界を決める
    のは第一段階。"""
    with raises_error(
        ErrorCode.INSUFFICIENT_DEGREES_OF_FREEDOM,
        n_obs=6,
        n_params=3,
        n_effects=3,
        min_obs=7,
        stage="FIRST_STAGE",
        endog_name="a",
    ):
        _fit(_panel(3, 2))
    assert _fit(_panel(4, 2)).df_resid == 8 - 4 - 2


def test_leverage_of_an_unbalanced_two_way_model_has_a_size_limit():
    """組合せが欠ける2-wayのHC2/HC3は、水準の少ない方の効果の水準数の2乗の密な行列を解くので、
    水準数に上限（3000）がある。成功側（3000水準ちょうど）は、密な行列の求解が重く、環状の組合せは
    交互射影も収束しにくいため置かない。"""
    n = 3001
    rng = np.random.default_rng(1)
    v = rng.normal(size=3 * n)
    data = pl.DataFrame(
        {
            "entity": [i for i in range(n) for _ in range(3)],
            "time": [(i + k) % n for i in range(n) for k in range(3)],
            "y": v + rng.normal(size=3 * n),
            "x1": rng.normal(size=3 * n),
            "a": v + rng.normal(size=3 * n),
            "z1": rng.normal(size=3 * n),
            "z2": rng.normal(size=3 * n),
        }
    )
    with raises_error(
        ErrorCode.LEVERAGE_DIMENSION_TOO_LARGE, n_levels=n, max_levels=3000
    ):
        _fit(data, effects=["entity", "time"], cov_type="hc2")


# ── ValidationError（入力データ: クラスター） ───────────────────────────


def test_cluster_count_at_most_slopes_raises_and_one_more_is_accepted():
    """クラスター数`G`が傾き係数の数`q`（`x1`・`a`で2）以下は`ValidationError`、`G = q + 1`は成功。"""
    with raises_error(
        ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES,
        setting="cov_type",
        n_clusters=2,
        n_slopes=2,
        min_clusters=3,
    ):
        _fit(_panel(2, 5), cov_type="cluster")
    assert _fit(_panel(3, 5), cov_type="cluster").stat_df == 3 - 1


def test_the_cluster_count_bound_follows_the_number_of_endogenous_variables(
    df,
):
    """内生変数が2個だと傾き係数は3個（`x1`・`a`・`x2`）。3クラスターは失敗し、4クラスターは成功する。"""
    kwargs = {"x_endog": ["a", "x2"], "instruments": ["z1", "z2", "z3"]}
    with raises_error(
        ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES, n_clusters=3, n_slopes=3
    ):
        _fit(df.filter(pl.col("entity") < 3), cov_type="cluster", **kwargs)
    res = _fit(df.filter(pl.col("entity") < 4), cov_type="cluster", **kwargs)
    assert res.stat_df == 4 - 1


def test_a_single_cluster_raises(df):
    data = df.filter(pl.col("entity") == 0)
    with raises_error(ErrorCode.CLUSTER_COUNT_TOO_SMALL, n_clusters=1):
        _fit(data, cov_type="cluster")


# ── ValidationError（入力データ: Driscoll-Kraay） ───────────────────────


def test_dk_periods_at_most_slopes_raises_and_one_more_is_accepted(df):
    """時点数`T`が傾き係数の数以下は`ValidationError`。傾き係数は`x1`・`a`・`x2`の3個で、
    3時点は失敗し、4時点は成功する。"""
    kwargs = {
        "x_endog": ["a", "x2"],
        "instruments": ["z1", "z2", "z3"],
        "cov_type": "dk",
        "dk_time": "dkt",
        "dk_bandwidth": 0,
    }
    with raises_error(
        ErrorCode.DK_PERIODS_NOT_ABOVE_COEFFICIENTS,
        n_periods=3,
        n_coefficients=3,
        min_periods=4,
    ):
        _fit(df.filter(pl.col("time") < 3), **kwargs)
    assert _fit(df.filter(pl.col("time") < 4), **kwargs).dk_bandwidth_used == 0


def test_dk_bandwidth_must_be_below_the_number_of_periods(df):
    """バンド幅は`0..T-1`（`T=5`なら4まで）。範囲外は`ValidationError`、上限ちょうどは成功する。"""
    for value in (-1, N_PERIODS):
        with raises_error(
            ErrorCode.DK_BANDWIDTH_OUT_OF_RANGE,
            value=value,
            n_periods=N_PERIODS,
            max=N_PERIODS - 1,
        ):
            _fit(df, cov_type="dk", dk_time="dkt", dk_bandwidth=value)
    res = _fit(df, cov_type="dk", dk_time="dkt", dk_bandwidth=N_PERIODS - 1)
    assert res.dk_bandwidth_used == N_PERIODS - 1


_TWO_PERIODS = pl.DataFrame(
    {
        "y": [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0],
        "a": [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0],
        "z1": [0.5, 2.5, 2.5, 4.0, 4.5, 5.0, 0.5, 1.0],
        "z2": [1.0, 0.0, 3.0, 1.0, 2.0, 4.0, 3.0, 1.5],
        "entity": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "time": ["1", "2"] * 4,
    }
)


@pytest.mark.parametrize(
    ("effects", "options", "code"),
    [
        (
            ["entity"],
            {"cov_type": "dk", "dk_time": "time", "dk_bandwidth": 0},
            ErrorCode.DK_TWO_PERIODS_DEGENERATE,
        ),
        (
            ["entity", "time"],
            {"cov_type": "dk", "dk_time": "time", "dk_bandwidth": 0},
            ErrorCode.DK_TWO_PERIODS_DEGENERATE,
        ),
        (
            ["entity"],
            {"cov_type": "cluster", "cluster": "time"},
            ErrorCode.CLUSTER_TWO_GROUPS_DEGENERATE,
        ),
    ],
    ids=["dk-one_way", "dk-two_way", "cluster_by_time"],
)
def test_two_group_split_degeneracy_raises(effects, options, code):
    """2時点で全個体が各時点に1観測ずつだと、within変換で時点（クラスター）スコアが恒等的にゼロに
    なり、共分散がゼロに退化する。傾き係数が`a`の1個だけなら`T > q`・`G > q`の検証は通る。"""
    with raises_error(code):
        _fit(
            _TWO_PERIODS,
            x_exog=[],
            x_endog=["a"],
            instruments=["z1", "z2"],
            effects=effects,
            **options,
        )


# ── ValidationError（オプション） ───────────────────────────────────────


def test_hc0_is_not_available_and_names_the_model(df):
    with raises_error(
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL, model="FEIV"
    ):
        _fit(df, cov_type="hc0")


def test_an_unknown_cov_type_raises(df):
    with raises_error(ErrorCode.OPTION_UNKNOWN_VALUE):
        _fit(df, cov_type="nonrobust")


def test_dk_requires_dk_time(df):
    with raises_error(ErrorCode.DK_TIME_REQUIRED):
        _fit(df, cov_type="dk")


@pytest.mark.parametrize(
    ("cov_type", "option", "value"),
    [
        ("classical", "cluster", "entity"),
        ("hc1", "cluster", "time"),
        ("dk", "cluster", "entity"),
        ("hc1", "dk_time", "time"),
        ("cluster", "dk_bandwidth", 2),
    ],
)
def test_an_option_the_cov_type_does_not_use_raises(
    df, cov_type, option, value
):
    kwargs = {"dk_time": "dkt"} if cov_type == "dk" else {}
    with raises_error(ErrorCode.OPTION_UNUSED):
        _fit(df, cov_type=cov_type, **{option: value}, **kwargs)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, 1.5, -0.1])
def test_confidence_level_must_be_inside_zero_and_one(df, confidence_level):
    with raises_error(ErrorCode.CONFIDENCE_LEVEL_OUT_OF_RANGE):
        _fit(df, confidence_level=confidence_level)


@pytest.mark.parametrize("tol", [0.0, -1e-9, float("inf"), float("nan")])
def test_absorption_tol_must_be_positive_and_finite(df, tol):
    with raises_error(ErrorCode.TOLERANCE_NOT_POSITIVE_FINITE, option="tol"):
        _fit(df, tol=tol)


@pytest.mark.parametrize("max_iter", [0, -1, 10_001])
def test_absorption_max_iter_must_be_in_range(df, max_iter):
    with raises_error(
        ErrorCode.MAX_ITER_OUT_OF_RANGE, value=max_iter, min=1, max=10_000
    ):
        _fit(df, max_iter=max_iter)


# ── ValidationError（診断メソッドの引数） ────────────────────────────────


def test_effective_f_needs_one_endogenous_variable_and_a_valid_alpha(df):
    res = _fit(df, x_endog=["a", "x2"], instruments=["z1", "z2", "z3"])
    with raises_error(ErrorCode.EFFECTIVE_F_ENDOGENOUS_COUNT, n_endog=2):
        res.effective_f()
    one = _fit(df)
    with raises_error(ErrorCode.ALPHA_OUT_OF_RANGE, value=2.0):
        one.effective_f(alpha=2.0)
    assert one.effective_f(alpha=0.10).alpha == 0.10


# ── ComputationError ────────────────────────────────────────────────────


def test_collinear_exogenous_columns_name_the_first_stage(df):
    """外生変数の共線性は、まず第一段階の設計行列で見つかる（`stage`と`endog_name`が付く）。
    完全な多重共線性は数値比較の対象外で、想定したエラーの送出だけを確認する。"""
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR,
        columns=["x1", "x1c"],
        stage="FIRST_STAGE",
        endog_name="a",
    ):
        _fit(df.with_columns(x1c=pl.col("x1")), x_exog=["x1", "x1c"])


def test_collinear_instruments_report_the_pair(df):
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR,
        columns=["z1", "z1c"],
        stage="FIRST_STAGE",
        endog_name="a",
    ):
        _fit(df.with_columns(z1c=pl.col("z1")), instruments=["z1", "z1c"])


def test_an_instrument_that_copies_an_exogenous_column_reports_the_pair(df):
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR,
        columns=["x1", "xc"],
        stage="FIRST_STAGE",
        endog_name="a",
    ):
        _fit(df.with_columns(xc=pl.col("x1")), instruments=["xc", "z1"])


def test_an_exogenous_copy_of_the_endogenous_variable_is_a_perfect_first_stage(
    df,
):
    """内生変数と同じ列を外生変数に入れると、第一段階が完全に当てはまる。"""
    with raises_error(
        ErrorCode.PERFECT_FIT, stage="FIRST_STAGE", endog_name="a"
    ):
        _fit(df.with_columns(ac=pl.col("a")), x_exog=["x1", "ac"])


@pytest.mark.parametrize(
    "cov_type", ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
)
def test_a_perfect_second_stage_raises_for_every_cov_type(df, cov_type):
    """構造式が完全に当てはまる（`y`が`x1`・`a`と個体の効果の厳密な関数）と、標準誤差とF検定が
    定義できず、`cov_type`によらず同じ`code`で`ComputationError`。"""
    alpha = np.random.default_rng(5).normal(size=N_ENTITIES)
    # 個体の効果を足した`y`。個体内の変動は`x1 + 0.5 * a`だけになる。
    perfect = df["x1"] + 0.5 * df["a"] + pl.Series(np.repeat(alpha, N_PERIODS))
    options = {"cov_type": cov_type}
    if cov_type == "dk":
        options["dk_time"] = "dkt"
    with raises_error(ErrorCode.PERFECT_FIT, stage="SECOND_STAGE"):
        _fit(df.with_columns(y=perfect), **options)


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
def test_a_scale_difference_makes_the_covariance_numerically_singular(
    cov_type,
):
    """`x1`を`1e6`倍、内生変数を`1e-3`倍した設計は、傾き係数の共分散の条件数が倍精度の限界を超え、
    全体F検定が検出して`MATRIX_NEAR_SINGULAR`（`testing-policy.md`の3系統の`ComputationError`）。"""
    data = pl.read_csv(DATA_DIR / "fe_iv_unbalanced.csv").with_columns(
        (pl.col("x1") * 1e6).alias("x1"),
        (pl.col("endog1") * 1e-3).alias("endog1"),
    )
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR,
        matrix="COEFFICIENT_COVARIANCE",
        stage="OVERALL_F_TEST",
    ):
        FEIV(
            data,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
            effects=["entity"],
            options=FEIVOptions(cov_type=cov_type),
        ).fit()


def test_absorption_not_converged_raises():
    """不均衡な2-wayで交互射影が`max_iter`までに収束しないと`ABSORPTION_NOT_CONVERGED`。"""
    data = pl.read_csv(DATA_DIR / "fe_iv_unbalanced.csv")
    with raises_error(ErrorCode.ABSORPTION_NOT_CONVERGED, n_sweeps=1):
        FEIV(
            data,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
            effects=["entity", "time"],
            options=FEIVOptions(max_iter=1),
        ).fit()


def test_weak_instrument_statistics_are_undefined_when_dk_periods_are_too_few(
    df,
):
    """DKで時点数`T`が`d`（除外操作変数の数 − 内生変数の数 + 1）以下だと、Kleibergen-Paapと
    有効Fは定義できず`ComputationError`（`d = 3`・`T = 3`）。Cragg-Donaldは`cov_type`によらず
    計算できる。`T = 4`は成功する。"""
    options = {"cov_type": "dk", "dk_time": "dkt", "dk_bandwidth": 0}
    kwargs = {"instruments": ["z1", "z2", "z3"]}
    short = _fit(df.filter(pl.col("time") < 3), **kwargs, **options)
    assert short.cragg_donald().statistic > 0
    with raises_error(
        ErrorCode.WEAK_INSTRUMENT_STATISTIC_UNDEFINED,
        statistic="Kleibergen-Paap",
    ):
        short.kleibergen_paap()
    with raises_error(
        ErrorCode.WEAK_INSTRUMENT_STATISTIC_UNDEFINED, statistic="effective F"
    ):
        short.effective_f()
    enough = _fit(df.filter(pl.col("time") < 4), **kwargs, **options)
    assert enough.kleibergen_paap().rk_wald_f > 0
    assert enough.effective_f().statistic > 0


# ── 例外クラスの継承 ────────────────────────────────────────────────────


def test_validation_error_is_value_error(df):
    with pytest.raises(ValueError) as info:
        _fit(df, cov_type="hc0")
    assert isinstance(info.value, ValidationError)


def test_computation_error_is_runtime_error(df):
    with pytest.raises(RuntimeError) as info:
        _fit(df.with_columns(z1c=pl.col("z1")), instruments=["z1", "z1c"])
    assert isinstance(info.value, ComputationError)
