"""FEGMMの入力・オプションの検証（`ValidationError`）と計算上のエラー（`ComputationError`）。

`_lib.fit_fe_gmm`を直接呼ぶ（Pythonのラッパーは別）。エラーは`code`と`params`で照合する（文面は照合しない。
`testing-policy.md`「テストの3系統」）。FEGMMはFEIVと同じ固定効果・ロールの検証を持つうえに、GMM固有の
オプション（`gmm_type`・`gmm_weight_type`・`gmm_max_iter`・`gmm_tol`・`raise_on_non_convergence`）の値・組合せ、
重み行列のクラスター数、事後診断`c_stat`の検証を持つ。境界の失敗側のテストは、反対側の成功パスと対にして
同じ見出しに置く。

役割分担:
    - 入力の検証・エラーの`code`と`params`: このファイル
    - 横断的な検査（列の型・欠損値・列の順序）: `tests/cross_method/`
    - 配線・数値・pickle: `test_fe_gmm_binding.py`
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
    return _lib.fit_fe_gmm(
        data,
        y,
        ["x1"] if x_exog is None else x_exog,
        ["a"] if x_endog is None else x_endog,
        ["z1", "z2"] if instruments is None else instruments,
        ["entity"] if effects is None else effects,
        _lib.FEGMMOptions(**options),
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
        ErrorCode.EFFECTS_TOO_MANY, model="FEGMM", n_columns=6, max_columns=5
    ):
        _fit(data, effects=effects)


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


def test_hc0_is_not_available_for_fegmm(df):
    with raises_error(
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL,
        value="hc0",
        model="FEGMM",
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

_ITERATED = 'gmm_type="iterated"'
_WEIGHTED = 'gmm_type in ("two_step", "iterated")'
_CLUSTER = (
    'cov_type="cluster" or (gmm_type in ("two_step", "iterated") '
    'and gmm_weight_type="cluster")'
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
