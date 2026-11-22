"""FEIVの入力・オプションの検証（`ValidationError`）と計算上のエラー（`ComputationError`）。

`_lib.fit_fe_iv`を直接呼ぶ（Pythonのラッパーは別）。エラーは`code`と`params`で照合する（文面は照合しない。
`testing-policy.md`「テストの3系統」）。FEIVはFEIVと同じ固定効果・ロールの検証を持つうえに、GMM固有の
オプション（`gmm_type`・`gmm_weight_type`・`gmm_max_iter`・`gmm_tol`・`raise_on_non_convergence`）の値・組合せ、
重み行列のクラスター数、事後診断`c_stat`の検証を持つ。境界の失敗側のテストは、反対側の成功パスと対にして
同じ見出しに置く。

役割分担:
    - 入力の検証・エラーの`code`と`params`: このファイル
    - 横断的な検査（列の型・欠損値・列の順序）: `tests/cross_method/`
    - 配線・数値・pickle: `test_fe_iv_gmm_binding.py`
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _errors import raises_error
from econometricsmodels import ErrorCode, _lib

N_ENTITIES = 12
N_PERIODS = 5
N = N_ENTITIES * N_PERIODS


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    """12個体×5時点の均衡パネル（内生変数`a`、操作変数`z1`〜`z3`、外生変数`x1`・`x2`）。
    粗いクラスター列`half`（2群）・`quarter`（4群）、時点の別名`dkt`を持つ。"""
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
            "half": entity % 2,
            "quarter": entity % 4,
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
    **options: object,
):
    return _lib.fit_fe_iv(
        data,
        y,
        ["x1"] if x_exog is None else x_exog,
        ["a"] if x_endog is None else x_endog,
        ["z1", "z2"] if instruments is None else instruments,
        ["entity"] if effects is None else effects,
        _lib.FEIVOptions(estimator="gmm", **options),
    )


# ── ValidationError（入力データ: 変数ロール） ─────────────────────────


@pytest.mark.parametrize("role", ["x_endog", "instruments"])
def test_empty_x_endog_or_instruments_raises(df, role):
    with raises_error(ErrorCode.ROLE_EMPTY, argument=role):
        _fit(df, **{role: []})


def test_too_few_instruments_raises_and_equal_count_is_accepted(df):
    with raises_error(
        ErrorCode.INSUFFICIENT_INSTRUMENTS, n_instruments=1, n_endog=2
    ):
        _fit(df, x_endog=["a", "x2"], instruments=["z1"])
    # 操作変数が内生変数と同数（丁度識別）なら成功し、Hansen Jは無い。
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


def test_effects_limit_names_the_model(df):
    data = df.with_columns(
        e3=pl.col("entity") % 3,
        e4=pl.col("time") % 3,
        e5=(pl.col("entity") + pl.col("time")) % 3,
        e6=(pl.col("entity") * 2 + pl.col("time")) % 3,
    )
    effects = ["entity", "time", "e3", "e4", "e5", "e6"]
    with raises_error(
        ErrorCode.EFFECTS_TOO_MANY, model="FEIV", n_columns=6, max_columns=5
    ):
        _fit(data, effects=effects)


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
        _fit(df, cluster="nope")
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
    with raises_error(
        ErrorCode.REGRESSOR_CONSTANT, role=role, columns=["extra"]
    ):
        _fit(df.with_columns(extra=pl.lit(1.5)), **_WITH_EXTRA_COLUMN[role])


@pytest.mark.parametrize("role", list(_WITH_EXTRA_COLUMN))
def test_a_column_constant_within_entities_raises(df, role):
    """個体ごとに一定の列は、固定効果を除くと分散が無いため`REGRESSOR_NO_WITHIN_VARIATION`。"""
    level = np.random.default_rng(2).normal(size=N_ENTITIES)
    data = df.with_columns(extra=pl.Series(np.repeat(level, N_PERIODS)))
    with raises_error(
        ErrorCode.REGRESSOR_NO_WITHIN_VARIATION, role=role, columns=["extra"]
    ):
        _fit(data, **_WITH_EXTRA_COLUMN[role])


# 同じロールに、名前の昇順にならない2列（`extra2`が先）を入れた引数。並び順が引数どおり
# （名前順ではない）であることまで確かめる。
_WITH_TWO_EXTRA_COLUMNS = {
    "x_exog": {"x_exog": ["extra2", "x1", "extra"]},
    "x_endog": {
        "x_endog": ["extra2", "a", "extra"],
        "instruments": ["z1", "z2", "z3"],
    },
    "instruments": {"instruments": ["extra2", "z1", "z2", "extra"]},
}


def _with_extra_columns(data: pl.DataFrame) -> pl.DataFrame:
    """生の値が定数の列`extra`・`extra2`と、個体ごとに一定の列`inv`・`inv2`を足す。"""
    level = np.random.default_rng(2).normal(size=N_ENTITIES)
    return data.with_columns(
        extra=pl.lit(1.5),
        extra2=pl.lit(2.5),
        inv=pl.Series(np.repeat(level, N_PERIODS)),
        inv2=pl.Series(np.repeat(level**2 + 1.0, N_PERIODS)),
    )


@pytest.mark.parametrize("role", list(_WITH_TWO_EXTRA_COLUMNS))
def test_every_constant_column_of_a_role_is_reported(df, role):
    """同じロールの定数列は、最初の1列で止めず全て、引数の並び順で報告する。"""
    with raises_error(
        ErrorCode.REGRESSOR_CONSTANT, role=role, columns=["extra2", "extra"]
    ):
        _fit(_with_extra_columns(df), **_WITH_TWO_EXTRA_COLUMNS[role])


@pytest.mark.parametrize("role", list(_WITH_TWO_EXTRA_COLUMNS))
def test_every_column_constant_within_entities_is_reported(df, role):
    """個体ごとに一定の列も、同じロールの全てを引数の並び順で報告する。"""
    data = _with_extra_columns(df).with_columns(
        extra=pl.col("inv"), extra2=pl.col("inv2")
    )
    with raises_error(
        ErrorCode.REGRESSOR_NO_WITHIN_VARIATION,
        role=role,
        columns=["extra2", "extra"],
    ):
        _fit(data, **_WITH_TWO_EXTRA_COLUMNS[role])


def test_the_constant_columns_are_only_those_with_one_value_in_every_row(df):
    """個体ごとに一定の列は、生の値が定数ではないので`REGRESSOR_CONSTANT`の列に入らない。"""
    with raises_error(
        ErrorCode.REGRESSOR_CONSTANT, role="x_exog", columns=["extra"]
    ):
        _fit(_with_extra_columns(df), x_exog=["inv", "extra", "x1"])


def test_every_regressor_of_a_role_can_be_reported(df):
    """ロールの全ての列が該当するときは、全列を報告する。"""
    with raises_error(
        ErrorCode.REGRESSOR_CONSTANT,
        role="x_exog",
        columns=["extra2", "extra"],
    ):
        _fit(_with_extra_columns(df), x_exog=["extra2", "extra"])


# 役割は`x_exog`、`x_endog`、`instruments`の順に調べ、該当列のある最初の役割だけを報告する。
# 生の値が定数の列（`REGRESSOR_CONSTANT`）は、固定効果を除いた後の検査より先に調べる。
_FIRST_ROLE_CASES = [
    pytest.param(
        {
            "x_exog": ["extra"],
            "x_endog": ["extra2"],
            "instruments": ["z1", "inv"],
        },
        (ErrorCode.REGRESSOR_CONSTANT, "x_exog", ["extra"]),
        id="constant-in-every-role",
    ),
    pytest.param(
        {"x_endog": ["extra"], "instruments": ["z1", "extra2"]},
        (ErrorCode.REGRESSOR_CONSTANT, "x_endog", ["extra"]),
        id="constant-in-endog-and-instruments",
    ),
    pytest.param(
        {"x_exog": ["inv"], "x_endog": ["inv2"]},
        (ErrorCode.REGRESSOR_NO_WITHIN_VARIATION, "x_exog", ["inv"]),
        id="invariant-in-exog-and-endog",
    ),
    pytest.param(
        {"x_endog": ["inv"], "instruments": ["z1", "inv2"]},
        (ErrorCode.REGRESSOR_NO_WITHIN_VARIATION, "x_endog", ["inv"]),
        id="invariant-in-endog-and-instruments",
    ),
    pytest.param(
        {"x_exog": ["inv"], "instruments": ["z1", "extra"]},
        (ErrorCode.REGRESSOR_CONSTANT, "instruments", ["extra"]),
        id="constant-instrument-before-invariant-exog",
    ),
]


@pytest.mark.parametrize(("kwargs", "expected"), _FIRST_ROLE_CASES)
def test_only_the_first_role_with_an_offending_column_is_reported(
    df, kwargs, expected
):
    code, role, columns = expected
    with raises_error(code, role=role, columns=columns):
        _fit(_with_extra_columns(df), **kwargs)


def test_a_time_invariant_column_is_accepted_with_a_time_only_effect(df):
    """時点の固定効果だけなら、個体ごとに一定の列にも変動が残る（個体の効果の1-wayとは別）。"""
    level = np.random.default_rng(2).normal(size=N_ENTITIES)
    data = df.with_columns(extra=pl.Series(np.repeat(level, N_PERIODS)))
    res = _fit(data, x_exog=["x1", "extra"], effects=["time"])
    assert "extra" in res.param_names


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


def test_insufficient_degrees_of_freedom_has_a_boundary(df):
    """GMMに第一段階は無い。構造式（`x1`・`a`と個体の効果）と操作変数（`x1`・`z1`・`z2`と個体の効果）の
    うち大きい方が境界を決める。観測数が`n_params + n_effects`以下だと失敗し、1つ多いと成功する。"""
    # 3個体×2時点（n=6）: 操作変数の設計（3列＋3個体の効果＝6）に足りない。
    small = df.filter((pl.col("entity") < 3) & (pl.col("time") < 2))
    with raises_error(ErrorCode.INSUFFICIENT_DEGREES_OF_FREEDOM, n_obs=6):
        _fit(small, gmm_type="one_step")
    # 4個体×2時点（n=8）は成功する。
    ok = df.filter((pl.col("entity") < 4) & (pl.col("time") < 2))
    assert _fit(ok, gmm_type="one_step").df_resid == 8 - 4 - 2


# ── ValidationError（入力データ: クラスター） ───────────────────────────


def test_cluster_count_at_most_slopes_raises_and_one_more_is_accepted(df):
    """`cov_type="cluster"`のクラスター数`G`が傾き係数の数`q`（`x1`・`a`で2）以下は`ValidationError`、
    `G = q + 1`は成功する（重みの検証は`one_step`で外す）。"""
    two = df.filter(pl.col("entity") < 2)
    with raises_error(
        ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES,
        setting="cov_type",
        n_clusters=2,
        n_slopes=2,
        min_clusters=3,
    ):
        _fit(two, gmm_type="one_step", cov_type="cluster")
    three = df.filter(pl.col("entity") < 3)
    assert _fit(three, gmm_type="one_step", cov_type="cluster").n_obs == 15


def test_a_single_cluster_raises(df):
    data = df.filter(pl.col("entity") == 0)
    with raises_error(ErrorCode.CLUSTER_COUNT_TOO_SMALL, n_clusters=1):
        _fit(data, gmm_type="one_step", cov_type="cluster")


# ── ValidationError（入力データ: Driscoll-Kraay） ───────────────────────


def test_dk_periods_at_most_slopes_raises_and_one_more_is_accepted(df):
    """時点数`T`が傾き係数の数以下は`ValidationError`。傾き係数は`x1`・`a`・`x2`の3個で、
    3時点は失敗し、4時点は成功する。"""
    kwargs = {
        "x_endog": ["a", "x2"],
        "x_exog": ["x1"],
        "instruments": ["z1", "z2", "z3"],
        "gmm_type": "one_step",
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


def test_confidence_level_must_be_inside_zero_and_one(df):
    for value in (0.0, 1.0, -0.1, 1.5):
        with raises_error(ErrorCode.CONFIDENCE_LEVEL_OUT_OF_RANGE):
            _fit(df, confidence_level=value)


# ── ValidationError（オプション: 値） ────────────────────────────────


def test_unknown_gmm_type_raises_with_the_allowed_values(df):
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="gmm_type",
        value="three_step",
        allowed=["one_step", "two_step", "iterated"],
    ):
        _fit(df, gmm_type="three_step")


@pytest.mark.parametrize("weight", ["hac", "nope", ""])
def test_unknown_weight_type_raises_and_hac_is_not_a_weight(df, weight):
    """重みの語彙は`classical`・`robust`・`cluster`の3つ（`hac`は無い）。"""
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="gmm_weight_type",
        value=weight,
        allowed=["classical", "robust", "cluster"],
    ):
        _fit(df, gmm_weight_type=weight)


def test_a_misspelled_weight_type_is_reported_even_for_one_step(df):
    """未知の値の検査は、組合せ（使われないオプション）の検査より先。"""
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="gmm_weight_type",
        value="hac",
        allowed=["classical", "robust", "cluster"],
    ):
        _fit(df, gmm_type="one_step", gmm_weight_type="hac")


def test_hc0_is_not_available_for_feiv_gmm(df):
    with raises_error(
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL,
        value="hc0",
        model="FEIV",
        allowed=["classical", "hc1", "hc2", "hc3", "cluster", "dk"],
    ):
        _fit(df, cov_type="hc0")


@pytest.mark.parametrize("max_iter", [-1, 0, 2, 10001])
def test_gmm_max_iter_must_be_between_3_and_10000(df, max_iter):
    with raises_error(
        ErrorCode.GMM_MAX_ITER_OUT_OF_RANGE, value=max_iter, min=3, max=10000
    ):
        _fit(df, gmm_type="iterated", gmm_max_iter=max_iter)


@pytest.mark.parametrize("max_iter", [3, 10000])
def test_gmm_max_iter_bounds_are_accepted(df, max_iter):
    result = _fit(
        df,
        gmm_type="iterated",
        gmm_max_iter=max_iter,
        raise_on_non_convergence=False,
    )
    assert result.n_iter <= max_iter


@pytest.mark.parametrize("tol", [0.0, -1e-6, float("nan"), float("inf")])
def test_gmm_tol_must_be_positive_and_finite(df, tol):
    with raises_error(
        ErrorCode.TOLERANCE_NOT_POSITIVE_FINITE, option="gmm_tol"
    ):
        _fit(df, gmm_type="iterated", gmm_tol=tol)


# ── ValidationError（オプション: 使われないオプション） ─────────────────

_ITERATED = 'estimator="gmm" and gmm_type="iterated"'
_WEIGHTED = 'estimator="gmm" and gmm_type in ("two_step", "iterated")'
_CLUSTER = (
    'cov_type="cluster" or (estimator="gmm" and '
    'gmm_type in ("two_step", "iterated") and gmm_weight_type="cluster")'
)


@pytest.mark.parametrize(
    ("options", "option", "condition"),
    [
        (
            {"gmm_type": "one_step", "gmm_weight_type": "robust"},
            "gmm_weight_type",
            _WEIGHTED,
        ),
        ({"gmm_max_iter": 10}, "gmm_max_iter", _ITERATED),
        ({"gmm_type": "two_step", "gmm_tol": 1e-4}, "gmm_tol", _ITERATED),
        (
            {"gmm_type": "one_step", "raise_on_non_convergence": False},
            "raise_on_non_convergence",
            _ITERATED,
        ),
        (
            {
                "cov_type": "hc1",
                "gmm_weight_type": "robust",
                "cluster": "entity",
            },
            "cluster",
            _CLUSTER,
        ),
        (
            {"cov_type": "hc1", "gmm_type": "one_step", "cluster": "entity"},
            "cluster",
            _CLUSTER,
        ),
        ({"dk_time": "dkt"}, "dk_time", 'cov_type="dk"'),
        ({"dk_bandwidth": 1}, "dk_bandwidth", 'cov_type="dk"'),
    ],
)
def test_an_option_that_the_mode_does_not_use_raises(
    df, options, option, condition
):
    with raises_error(
        ErrorCode.OPTION_UNUSED, option=option, condition=condition
    ):
        _fit(df, **options)


@pytest.mark.parametrize(
    "options",
    [
        # `cluster`は`cov_type`と重みのどちらか一方が使えば有効。
        {
            "cov_type": "hc1",
            "gmm_weight_type": "cluster",
            "cluster": "quarter",
        },
        {
            "cov_type": "cluster",
            "gmm_weight_type": "robust",
            "cluster": "quarter",
        },
        {"cov_type": "cluster", "gmm_type": "one_step", "cluster": "quarter"},
        {"gmm_type": "iterated", "gmm_max_iter": 20, "gmm_tol": 1e-4},
    ],
)
def test_an_option_that_one_of_the_modes_uses_is_accepted(df, options):
    assert _fit(df, **options).converged


def test_dk_requires_dk_time_and_does_not_fall_back_to_the_effects(df):
    with raises_error(ErrorCode.DK_TIME_REQUIRED):
        _fit(df, cov_type="dk")
    assert _fit(df, cov_type="dk", dk_time="dkt").cov_type == "dk"


# ── ValidationError（重み行列のクラスター数） ─────────────────────────


def test_cluster_weight_needs_as_many_clusters_as_instruments(df):
    """過剰識別（`l = 1 + 2 = 3`）では、クラスター数が`l`以上でなければならない。"""
    with raises_error(
        ErrorCode.GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL,
        n_clusters=2,
        n_instruments=3,
        min_clusters=3,
    ):
        _fit(df, cluster="half")


def test_cluster_weight_with_exactly_enough_clusters_is_accepted(df):
    """`l = 3`でクラスターが4つ。SEのクラスターが傾き係数の数（2）を超えるので成功する。"""
    assert _fit(df, cluster="quarter").gmm_weight_type == "cluster"


# ── ComputationError ───────────────────────────────────────────────


def test_a_perfect_structural_fit_is_a_computation_error(df):
    data = df.with_columns(y=1.5 * pl.col("x1") - 0.7 * pl.col("a"))
    with raises_error(ErrorCode.PERFECT_FIT):
        _fit(data)


def test_non_convergence_raises_by_default_and_is_reported_on_request(df):
    options = {
        "gmm_type": "iterated",
        "gmm_weight_type": "robust",
        "gmm_max_iter": 3,
        "gmm_tol": 1e-15,
    }
    with raises_error(
        ErrorCode.NOT_CONVERGED, n_iter=3, iter_option="gmm_max_iter"
    ):
        _fit(df, **options)
    assert not _fit(df, **options, raise_on_non_convergence=False).converged


def test_collinear_exogenous_columns_are_a_singular_matrix(df):
    """外生変数の共線性は、完全な多重共線性として`ComputationError`（数値比較の対象外で、想定した
    エラーの送出だけを確認する）。GMMには第一段階が無く、操作変数の設計行列`Z'Z`の特異として
    `MATRIX_NEAR_SINGULAR`になる（列の名前は付かない）。"""
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR):
        _fit(df.with_columns(x1c=pl.col("x1")), x_exog=["x1", "x1c"])


def test_collinear_instruments_are_a_singular_matrix(df):
    with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR):
        _fit(df.with_columns(z1c=pl.col("z1")), instruments=["z1", "z1c"])


def test_the_cov_type_does_not_change_the_error_of_a_singular_matrix(df):
    data = df.with_columns(z1c=pl.col("z1"))
    for cov_type in ("classical", "hc1", "cluster"):
        with raises_error(ErrorCode.MATRIX_NEAR_SINGULAR):
            _fit(data, instruments=["z1", "z1c"], cov_type=cov_type)


# ── c_stat ─────────────────────────────────────────────────────────


def _result(df):
    return _fit(df, instruments=["z1", "z2", "z3"])


def test_c_stat_needs_something_to_test(df):
    with raises_error(ErrorCode.C_STATISTIC_NOTHING_TO_TEST):
        _result(df).c_stat()


@pytest.mark.parametrize("argument", ["variables", "instruments"])
def test_c_stat_rejects_an_empty_list_and_a_repeated_name(df, argument):
    result = _result(df)
    name = "a" if argument == "variables" else "z3"
    with raises_error(ErrorCode.ROLE_EMPTY, argument=argument):
        result.c_stat(**{argument: []})
    with raises_error(ErrorCode.ROLE_DUPLICATE_COLUMN):
        result.c_stat(**{argument: [name, name]})


@pytest.mark.parametrize(
    ("argument", "name", "role"),
    [
        ("variables", "x1", "x_endog"),
        ("variables", "z1", "x_endog"),
        ("variables", "nope", "x_endog"),
        ("instruments", "a", "instruments"),
        ("instruments", "x1", "instruments"),
    ],
)
def test_c_stat_rejects_a_name_that_is_not_in_the_model(
    df, argument, name, role
):
    with raises_error(
        ErrorCode.C_STATISTIC_UNKNOWN_VARIABLE,
        name=name,
        argument=argument,
        role=role,
    ):
        _result(df).c_stat(**{argument: [name]})


def test_c_stat_rejects_testing_too_many_instruments(df):
    expected = {"n_tested": 3, "n_instruments": 3, "n_endog": 1}
    with raises_error(
        ErrorCode.C_STATISTIC_TOO_MANY_TESTED_INSTRUMENTS, **expected
    ):
        _result(df).c_stat(instruments=["z1", "z2", "z3"])
    # 1つ残れば識別できる（外した後の操作変数が内生変数の数に足りる）。
    assert _result(df).c_stat(instruments=["z2", "z3"]).df == 2


def test_c_stat_argument_types_are_checked(df):
    result = _result(df)
    for bad in ("a", 1, ["a", 1]):
        with pytest.raises(TypeError):
            result.c_stat(variables=bad)
        with pytest.raises(TypeError):
            result.c_stat(instruments=bad)


def test_c_stat_is_undefined_with_fewer_clusters_than_moment_conditions(df):
    """`fit`は通る（`l = 1 + 3 = 4`でクラスターが4つ）が、`Z_full`は`x_exog`・内生変数・操作変数3つの5列で、
    クラスター数が足りない。検定する内生変数を加えないときは`Z_full`が4列で足りる。"""
    result = _fit(df, instruments=["z1", "z2", "z3"], cluster="quarter")
    with raises_error(ErrorCode.C_STATISTIC_UNDEFINED):
        result.c_stat(variables=["a"])
    assert result.c_stat(instruments=["z3"]).df == 1


def test_c_stat_is_repeatable(df):
    """`c_stat`は`fit`に渡したデータから毎回再計算する。同じ呼び出しは同じ値を返す。"""
    result = _result(df)
    first = result.c_stat(instruments=["z3"])
    second = result.c_stat(instruments=["z3"])
    assert first.statistic == second.statistic
