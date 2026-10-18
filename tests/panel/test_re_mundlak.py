"""RE: 個体平均・時点平均を説明変数に加える相関ランダム効果（Mundlak型）の検証。

`x`とその個体平均`x_bar`を同時に入れると、個体平均の回帰（between回帰）では2列が同じ列になり、
ハウスマン補助回帰では`x̃`が`x*`と`x_bar*`の差で表せる別名になる。どちらも列を落として進める
（Rの`lm`と同じ規則: 前の列の線形結合になる列。最終回帰は全列を残すので、完全共線は
`ComputationError`のまま）。

plmとの数値照合・手計算・statsmodelsとの照合は`test_re_variance_method.py`
（`re_variance_method.json`の`*_mundlak_*`ケース）が担う。ここは、そこで固定しない
性質（列順・別名の数・事前検証・厳格さが残る入力）を確かめる。
"""

from __future__ import annotations

import polars as pl
import pytest
from _assertions import assert_close
from _constants import DATA_DIR
from _errors import raises_error
from econometricsmodels import (
    FE,
    RE,
    ErrorCode,
    FEOptions,
    REOptions,
    ValidationError,
)

RTOL = 1e-9
ATOL = 1e-12


def _frame() -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / "fe_baseline.csv").with_columns(
        x1_bar=pl.col("x1").mean().over("entity"),
        x2_bar=pl.col("x2").mean().over("entity"),
        x1_tbar=pl.col("x1").mean().over("time"),
    )


def _fit(x, options=None, df=None):
    return RE(
        _frame() if df is None else df,
        y="y",
        x=x,
        entity="entity",
        options=options or REOptions(cov_type="classical"),
    ).fit()


def test_mundlak_specification_is_estimated():
    """個体平均の列を`x`に含むMundlak指定が推定でき、個体平均の係数も推定される。"""
    res = _fit(["x1", "x2", "x1_bar"])
    assert list(res.params) == ["const", "x1", "x2", "x1_bar"]
    assert res.df_model == 4
    assert res.sigma2_u > 0.0


@pytest.mark.parametrize("variance_method", ["swamy_arora", "wallace_hussain"])
def test_results_do_not_depend_on_the_column_order(variance_method):
    """列順を変えても、別名として落ちる列が変わるだけで、分散成分・係数・ハウスマン検定は
    同じ（落ちるのは検定対象の`x̃`側）。"""
    options = REOptions(cov_type="classical", variance_method=variance_method)
    first = _fit(["x1", "x2", "x1_bar"], options)
    second = _fit(["x1_bar", "x1", "x2"], options)
    assert_close(
        first.sigma2_u, second.sigma2_u, "sigma2_u", rtol=RTOL, atol=ATOL
    )
    for name in ("const", "x1", "x2", "x1_bar"):
        assert_close(
            first.params[name],
            second.params[name],
            name,
            rtol=RTOL,
            atol=ATOL,
        )
    assert first.hausman_df == second.hausman_df == 1
    assert_close(
        first.hausman_statistic,
        second.hausman_statistic,
        "hausman_statistic",
        rtol=RTOL,
        atol=ATOL,
    )


def test_variance_components_equal_those_without_the_mean_column():
    """個体平均の列はbetween回帰で元の列と同じ列になって落ちるため、`σ_u²`は加えない場合と
    同じ（時間不変の列は内部のFEにも入らないので`σ_ε²`も同じ）。"""
    with_bar = _fit(["x1", "x2", "x1_bar"])
    without = _fit(["x1", "x2"])
    assert_close(
        with_bar.sigma2_u, without.sigma2_u, "sigma2_u", rtol=RTOL, atol=ATOL
    )
    assert_close(
        with_bar.sigma2_eps,
        without.sigma2_eps,
        "sigma2_eps",
        rtol=RTOL,
        atol=ATOL,
    )


def test_hausman_degrees_of_freedom_count_the_identified_columns():
    """検定する`x̃`のうち別名になった列は数えない。全ての時間変化列の個体平均を加えると、
    検定できる列が残らず`hausman_*`は`None`になる。"""
    assert _fit(["x1", "x2"]).hausman_df == 2
    assert _fit(["x1", "x2", "x1_bar"]).hausman_df == 1
    all_means = _fit(["x1", "x2", "x1_bar", "x2_bar"])
    assert all_means.hausman_df is None
    assert all_means.hausman_statistic is None
    assert all_means.hausman_p_value is None


def test_two_way_means_in_both_directions():
    """2-wayでは個体平均の列は個体平均の回帰で、時点平均の列は時点平均の回帰で別名になる。
    個体平均だけなら`x̃`に別名は無く（`hausman_df = 2`）、両方を加えると`x̃1`が別名になる
    （`hausman_df = 1`）。分散成分は平均を加えない場合と同じ。"""
    options = REOptions(cov_type="classical", time="time")
    plain = _fit(["x1", "x2"], options)
    entity_mean = _fit(["x1", "x2", "x1_bar"], options)
    both = _fit(["x1", "x2", "x1_bar", "x1_tbar"], options)
    assert entity_mean.hausman_df == 2
    assert both.hausman_df == 1
    for res in (entity_mean, both):
        assert_close(
            res.sigma2_u, plain.sigma2_u, "sigma2_u", rtol=RTOL, atol=ATOL
        )
        assert_close(
            res.sigma2_lambda,
            plain.sigma2_lambda,
            "sigma2_lambda",
            rtol=RTOL,
            atol=ATOL,
        )


def test_two_way_means_of_every_regressor_leave_nothing_to_test():
    """2-wayで全ての時間変化列の個体平均・時点平均を加えると、ハウスマン検定できる列が残らず
    `hausman_*`は`None`（推定は成功する）。"""
    df = _frame().with_columns(
        x2_tbar=pl.col("x2").mean().over("time"),
    )
    res = _fit(
        ["x1", "x2", "x1_bar", "x2_bar", "x1_tbar", "x2_tbar"],
        REOptions(cov_type="classical", time="time"),
        df,
    )
    assert res.hausman_df is None
    assert res.hausman_statistic is None
    assert res.hausman_p_value is None


def test_cluster_validation_counts_the_identified_columns():
    """クラスターロバストなハウスマン検定は、補助回帰の傾き係数の数（別名を除く）より多い
    クラスターを要する。`x1`・`x2`・`x1_bar`では`x*`の3個と`x̃2`の1個で4個、
    クラスター数が5なら成功し、4なら`ValidationError`。"""
    df = _frame().with_columns(
        g5=pl.col("entity").rank("dense") % 5,
        g4=pl.col("entity").rank("dense") % 4,
    )
    ok = _fit(
        ["x1", "x2", "x1_bar"],
        REOptions(cov_type="cluster", cluster="g5"),
        df,
    )
    assert ok.hausman_df == 1
    with pytest.raises(ValidationError):
        _fit(
            ["x1", "x2", "x1_bar"],
            REOptions(cov_type="cluster", cluster="g4"),
            df,
        )


@pytest.mark.parametrize(
    ("x", "stage"),
    [
        pytest.param(
            ["x1", "x2", "x3", "x1_bar"],
            "WITHIN_REGRESSION",
            id="within_collinear",
        ),
        pytest.param(
            ["x1", "x2", "x1_bar", "x1_bar2"],
            "QUASI_DEMEANED_REGRESSION",
            id="repeated_mean",
        ),
    ],
)
def test_collinear_designs_are_still_computation_errors(x, stage):
    """別名を落とすのはbetween回帰とハウスマン補助回帰だけで、内部のwithin回帰と最終回帰は
    全列を残すため、完全共線は`ComputationError`のまま。失敗するのは、within変動のある列同士
    なら内部のwithin回帰、個体平均とその倍数なら最終回帰（準偏差変換後）。"""
    df = _frame().with_columns(
        x3=2.0 * pl.col("x1") + 3.0 * pl.col("x2"),
        x1_bar2=2.0 * pl.col("x1_bar"),
    )
    with raises_error(ErrorCode.DESIGN_MATRIX_SINGULAR, stage=stage):
        _fit(x, df=df)


def test_all_means_add_up_to_the_fixed_effects_slopes():
    """全ての時間変化列の個体平均を加えたRE（Mundlak）の時間変化列の係数は、FEの係数と一致する
    （個体平均の係数がbetweenとwithinの差を吸収する）。plm・手計算に依存しない性質で、
    不均衡パネル・既定のSwamy-Aroraで確かめる。"""
    df = pl.read_csv(DATA_DIR / "fe_unbalanced.csv").with_columns(
        x1_bar=pl.col("x1").mean().over("entity"),
        x2_bar=pl.col("x2").mean().over("entity"),
    )
    re_res = _fit(["x1", "x2", "x1_bar", "x2_bar"], df=df)
    fe_res = FE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=FEOptions(cov_type="classical"),
    ).fit()
    for name in ("x1", "x2"):
        assert_close(
            re_res.params[name],
            fe_res.params[name],
            name,
            rtol=1e-8,
            atol=1e-10,
        )


def test_entity_count_boundary_uses_the_identified_columns():
    """個体数が傾き係数の数+1（自由度ちょうど0のbetween回帰）では、別名を落とした後の
    列数で判定する。`x1`・`x2`・`x1_bar`は別名を落とすと3列で`ValidationError`、
    `x1`・`x1_bar`は2列（`hausman_*`は`None`）で推定できる。"""
    df = pl.DataFrame(
        {
            "entity": [0] * 5 + [1] * 5 + [2] * 5,
            "x1": [
                0.3, -1.2, 0.8, 1.9, -0.4,
                1.1, 0.2, -0.7, 0.5, -1.6,
                -0.9, 1.4, 0.6, -0.2, 0.1,
            ],
            "x2": [
                1.5, 0.1, -0.8, 0.9, 0.4,
                -1.1, 0.7, 0.3, -0.5, 1.2,
                0.2, -0.6, 1.3, -1.0, 0.8,
            ],
            "y": [
                1.0, -0.5, 2.1, 3.0, 0.2,
                0.4, 1.7, -0.3, 1.1, -1.4,
                -0.8, 2.2, 0.9, 0.3, 0.6,
            ],
        }
    ).with_columns(x1_bar=pl.col("x1").mean().over("entity"))  # fmt: skip
    with raises_error(
        ErrorCode.INSUFFICIENT_OBSERVATIONS,
        stage="BETWEEN_REGRESSION_ENTITY",
        n_obs=3,
        n_params=3,
        min_obs=4,
    ):
        _fit(["x1", "x2", "x1_bar"], df=df)
    res = _fit(["x1", "x1_bar"], df=df)
    assert res.hausman_df is None
    assert res.hausman_statistic is None


@pytest.mark.parametrize("variance_method", ["swamy_arora", "wallace_hussain"])
def test_mean_columns_work_with_every_cov_type(variance_method):
    """`cov_type`によらず推定できる（クラスターの既定・DKの時点列を含む）。"""
    for cov_type in ("classical", "hc1", "hc2", "hc3", "cluster", "dk"):
        options = REOptions(
            cov_type=cov_type,
            dk_time="time" if cov_type == "dk" else None,
            variance_method=variance_method,
        )
        res = _fit(["x1", "x2", "x1_bar"], options)
        assert res.hausman_df == 1, cov_type
