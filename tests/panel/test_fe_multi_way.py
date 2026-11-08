"""FE の3つ以上の効果（`effects`が3〜5列）の構造・API・基本的な数値の検証。

3つ以上の効果は、水準ダミーを明示した最小二乗（LSDV）を独立なオラクルとして、
係数・古典的な標準誤差・HC1/HC2/HC3を照合する。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _errors import raises_error
from econometricsmodels import FE, FEOptions


def _panel(seed: int = 3, duplicate: bool = False) -> pl.DataFrame:
    """企業30×年8×産業4の不均衡パネル（産業は行ごとに無作為）。"""
    rng = np.random.default_rng(seed)
    firm_effect = rng.normal(size=30)
    year_effect = rng.normal(size=8)
    industry_effect = rng.normal(size=4)
    rows = []
    for firm in range(30):
        for year in range(8):
            if year >= 2 and firm >= 2 and rng.random() > 0.7:
                continue
            industry = int(rng.integers(4))
            x1, x2 = rng.normal(size=2)
            y = (
                1.5 * x1
                - 0.7 * x2
                + firm_effect[firm]
                + year_effect[year]
                + industry_effect[industry]
                + 0.5 * rng.normal()
            )
            rows.append((f"f{firm:02d}", year, f"i{industry}", x1, x2, y))
    df = pl.DataFrame(
        rows,
        schema=["firm", "year", "industry", "x1", "x2", "y"],
        orient="row",
    )
    if duplicate:
        df = pl.concat([df, df.head(5)])
    return df


def _fit(df: pl.DataFrame, effects: list[str], cov_type: str = "classical"):
    return FE(
        df,
        y="y",
        x=["x1", "x2"],
        effects=effects,
        options=FEOptions(cov_type=cov_type),
    ).fit()


def _lsdv(df: pl.DataFrame, effects: list[str]):
    """水準ダミー（2つ目以降は最初の水準を落とす）を並べた最小二乗。"""
    columns = [df["x1"].to_numpy(), df["x2"].to_numpy()]
    for position, name in enumerate(effects):
        levels = sorted(df[name].unique().to_list())
        for level in levels[1 if position else 0 :]:
            columns.append((df[name] == level).cast(pl.Float64).to_numpy())
    z = np.column_stack(columns)
    y = df["y"].to_numpy()
    bread = np.linalg.inv(z.T @ z)
    beta = bread @ z.T @ y
    residuals = y - z @ beta
    n, rank = z.shape
    leverage = np.einsum("ij,jk,ik->i", z, bread, z)
    covariances = {"classical": residuals @ residuals / (n - rank) * bread}
    for name, power in [("hc1", 0), ("hc2", 1), ("hc3", 2)]:
        weight = residuals**2 / (1 - leverage) ** power
        meat = (z * weight[:, None]).T @ z
        scale = n / (n - rank) if name == "hc1" else 1.0
        covariances[name] = scale * bread @ meat @ bread
    return beta[:2], covariances, rank


@pytest.mark.parametrize("duplicate", [False, True], ids=["plain", "repeated"])
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_three_effects_match_the_dummy_variable_regression(
    cov_type, duplicate
):
    df = _panel(duplicate=duplicate)
    effects = ["firm", "year", "industry"]
    ours = _fit(df, effects, cov_type)
    beta, covariances, rank = _lsdv(df, effects)
    assert ours.df_model == rank
    assert list(ours.n_groups) == effects
    assert [ours.params[name] for name in ("x1", "x2")] == pytest.approx(
        beta, rel=1e-8
    )
    std_errors = np.sqrt(np.diag(covariances[cov_type])[:2])
    assert [ours.std_errors[name] for name in ("x1", "x2")] == pytest.approx(
        std_errors, rel=1e-6
    )


def test_fixed_effects_have_one_entry_per_column_and_rebuild_the_outcome():
    df = _panel()
    effects = ["firm", "year", "industry"]
    ours = _fit(df, effects)
    fixed = ours.fixed_effects()
    assert list(fixed) == effects
    assert list(fixed["year"]) == [str(year) for year in range(8)]
    total = sum(
        df[name]
        .cast(pl.Utf8)
        .replace_strict(fixed[name], return_dtype=pl.Float64)
        for name in effects
    )
    fitted = total + sum(ours.params[name] * df[name] for name in ("x1", "x2"))
    assert (df["y"] - fitted).to_list() == pytest.approx(
        ours.residuals, abs=1e-8
    )
    # 基準の水準（連結なら2列目以降の最初の水準）は0。
    assert fixed["year"]["0"] == 0.0
    assert fixed["industry"]["i0"] == 0.0


def test_four_and_five_effects_are_accepted():
    df = _panel().with_columns(
        region=pl.col("firm").str.slice(1).cast(pl.Int64) % 3,
        group=pl.col("year") % 2,
    )
    four = _fit(df, ["firm", "year", "industry", "region"])
    assert list(four.n_groups) == ["firm", "year", "industry", "region"]
    five = _fit(df, ["firm", "year", "industry", "region", "group"])
    assert len(five.n_groups) == 5
    assert five.df_model == 2 + sum(five.n_groups.values()) - 4


def test_the_estimates_do_not_depend_on_the_order_of_the_effects():
    df = _panel()
    a = _fit(df, ["firm", "year", "industry"])
    b = _fit(df, ["industry", "firm", "year"])
    assert a.params == pytest.approx(b.params, rel=1e-8)
    assert a.df_model == b.df_model


def test_a_third_column_may_hold_any_key_type():
    df = _panel().with_columns(
        code=pl.col("industry").str.slice(1).cast(pl.Int64)
    )
    by_string = _fit(df, ["firm", "year", "industry"])
    by_integer = _fit(df, ["firm", "year", "code"])
    assert by_string.params == pytest.approx(by_integer.params, rel=1e-10)


def test_a_singleton_in_the_third_column_is_reported_with_its_column():
    df = _panel()
    df = df.with_columns(
        industry=pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("only"))
        .otherwise(pl.col("industry"))
    )
    with raises_error(
        "SINGLETON_GROUP", argument="effects", column="industry", group="only"
    ):
        _fit(df, ["firm", "year", "industry"])


def test_cluster_defaults_to_the_first_column():
    df = _panel()
    effects = ["firm", "year", "industry"]
    default = FE(df, y="y", x=["x1", "x2"], effects=effects).fit()
    explicit = FE(
        df,
        y="y",
        x=["x1", "x2"],
        effects=effects,
        options=FEOptions(cov_type="cluster", cluster="firm"),
    ).fit()
    assert default.std_errors == explicit.std_errors


def test_a_regressor_absorbed_by_the_third_effect_is_rejected():
    df = _panel().with_columns(
        x2=pl.col("industry").str.slice(1).cast(pl.Float64)
    )
    with raises_error("REGRESSOR_NO_WITHIN_VARIATION"):
        _fit(df, ["firm", "year", "industry"])


def test_a_boolean_third_column_is_an_identity_key():
    """3つ目以降の列は個体の列と同じ同一性だけのキーで、`Boolean`も使える
    （2つ目の時点の列は`Boolean`を拒否する）。"""
    df = _panel().with_columns(
        flag=pl.col("firm").str.slice(1).cast(pl.Int64) % 2 == 0
    )
    ours = _fit(df, ["firm", "year", "flag"])
    assert list(ours.n_groups) == ["firm", "year", "flag"]


def test_a_third_column_nested_in_the_first_keeps_the_row_effects():
    """3つ目の列が1つ目に入れ子のとき、効果は個別には識別されず、行ごとの効果の和だけが
    識別される。それでも傾きは2つの効果だけのモデルと同じで、和は残差を再現する。"""
    df = _panel().with_columns(
        region=pl.col("firm").str.slice(1).cast(pl.Int64) % 5
    )
    nested = _fit(df, ["firm", "year", "region"])
    two_way = _fit(df, ["firm", "year"])
    assert nested.params == pytest.approx(two_way.params, rel=1e-8)
    fixed = nested.fixed_effects()
    total = sum(
        df[name]
        .cast(pl.Utf8)
        .replace_strict(fixed[name], return_dtype=pl.Float64)
        for name in ("firm", "year", "region")
    )
    fitted = total + sum(
        nested.params[name] * df[name] for name in ("x1", "x2")
    )
    assert (df["y"] - fitted).to_list() == pytest.approx(
        nested.residuals, abs=1e-8
    )


# ── 入力のエラー（3つ目以降の効果） ────────────────────────────────────


def test_a_missing_value_in_the_third_column_is_rejected():
    df = _panel().with_columns(
        industry=pl.when(pl.int_range(pl.len()) == 3)
        .then(None)
        .otherwise(pl.col("industry"))
    )
    with raises_error("COLUMN_CONTAINS_NULL", column="industry"):
        _fit(df, ["firm", "year", "industry"])


def test_a_missing_third_column_is_reported():
    with raises_error("COLUMN_NOT_FOUND", column="nowhere"):
        _fit(_panel(), ["firm", "year", "nowhere"])


@pytest.mark.parametrize("column", ["y", "x1"])
def test_a_third_column_that_is_also_y_or_x_is_rejected(column):
    with raises_error("ROLE_COLUMN_OVERLAP"):
        _fit(_panel(), ["firm", "year", column])


def test_too_few_observations_for_the_absorbed_effects_are_rejected():
    # 8行・個体4×時点2×産業2: df_model = 2 + 4 + 2 + 2 - 2 = 8で残差の自由度が0。
    df = pl.DataFrame(
        {
            "firm": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "year": [0, 1, 0, 1, 0, 1, 0, 1],
            "industry": ["p", "q", "p", "q", "q", "p", "q", "p"],
            "x1": [0.1, 0.7, 0.3, 0.2, 0.9, 0.4, 0.6, 0.5],
            "x2": [0.5, 0.1, 0.9, 0.3, 0.2, 0.8, 0.4, 0.6],
            "y": [1.0, 0.2, 0.7, 0.9, 0.3, 0.8, 0.5, 0.4],
        }
    )
    with raises_error("INSUFFICIENT_DEGREES_OF_FREEDOM"):
        _fit(df, ["firm", "year", "industry"])


def test_the_sweep_limit_is_enforced_for_three_effects():
    with raises_error("ABSORPTION_NOT_CONVERGED"):
        FE(
            _panel(),
            y="y",
            x=["x1", "x2"],
            effects=["firm", "year", "industry"],
            options=FEOptions(max_iter=1, tol=1e-14),
        ).fit()


def test_a_dense_leverage_matrix_above_the_limit_is_rejected():
    # 最大の次元（2500水準）を除いた水準数の合計（2000 + 1500）が上限（3000）を超える。
    rows = 5000
    df = pl.DataFrame(
        {
            "a": [f"a{r % 2500:05d}" for r in range(rows)],
            "b": [f"b{r % 2000:05d}" for r in range(rows)],
            "c": [f"c{r % 1500:05d}" for r in range(rows)],
            "x": [float((r * 104729) % 97) for r in range(rows)],
            "y": [float((r * 7919) % 101) for r in range(rows)],
        }
    )
    with raises_error(
        "LEVERAGE_DIMENSION_TOO_LARGE", n_levels=3500, max_levels=3000
    ):
        FE(
            df,
            y="y",
            x=["x"],
            effects=["a", "b", "c"],
            options=FEOptions(cov_type="hc3"),
        ).fit()


def test_the_confidence_level_applies_to_three_effects():
    df = _panel()
    effects = ["firm", "year", "industry"]

    def fit(level):
        return FE(
            df,
            y="y",
            x=["x1", "x2"],
            effects=effects,
            options=FEOptions(cov_type="classical", confidence_level=level),
        ).fit()

    narrow, wide = fit(0.8), fit(0.95)
    for name in ("x1", "x2"):
        assert narrow.params[name] == wide.params[name]
        assert narrow.conf_int[name][1] - narrow.conf_int[name][0] < (
            wide.conf_int[name][1] - wide.conf_int[name][0]
        )
