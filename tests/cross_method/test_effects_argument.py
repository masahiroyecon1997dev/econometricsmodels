"""FE/REの必須引数`effects`（個体・時点の列のリスト）の検証と意味のテスト。

`effects`は1〜2要素のリストで、1要素が1-way、2要素が2-way（`effects[0]`が個体、
`effects[1]`が時点）。先頭の列がデフォルトのクラスターキーになる。2要素の順序は
バランスパネルではモデルが対称で検証では検出できないため、入れ替えたときの結果の
対応をここで固定する。列の型の規則（`effects[0]`は同一性だけのキー、`effects[1]`は
時点のキー）は`test_key_columns.py`・`test_input_dtypes.py`が見る。
"""

from __future__ import annotations

import random

import polars as pl
import pytest
from _errors import raises_error
from econometricsmodels import FE, RE, FEOptions, REOptions

N_ENTITIES = 8
N_PERIODS = 5


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    """個体8×時点5のバランスパネル（個体効果・時点効果あり）。"""
    rng = random.Random(7)
    entity_effect = [rng.gauss(0, 1) for _ in range(N_ENTITIES)]
    period_effect = [rng.gauss(0, 0.5) for _ in range(N_PERIODS)]
    rows = []
    for i in range(N_ENTITIES):
        for t in range(N_PERIODS):
            x1 = rng.gauss(0, 1)
            x2 = rng.gauss(0, 1)
            y = (
                1.0
                + 2.0 * x1
                - 1.0 * x2
                + entity_effect[i]
                + period_effect[t]
                + rng.gauss(0, 0.3)
            )
            rows.append((f"e{i}", t, f"g{i % 3}", x1, x2, y))
    return pl.DataFrame(
        rows,
        schema=["firm", "year", "group", "x1", "x2", "y"],
        orient="row",
    )


MODELS = [("FE", FE), ("RE", RE)]


def _fit(model, df, effects, **options):
    options_cls = FEOptions if model is FE else REOptions
    return model(
        df,
        y="y",
        x=["x1", "x2"],
        effects=effects,
        options=options_cls(**options),
    ).fit()


# ── 型 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("name", "model"), MODELS)
def test_effects_must_be_a_list_of_column_names(name, model, panel):
    with pytest.raises(TypeError, match="effects must be a list"):
        model(panel, y="y", x=["x1"], effects="firm").fit()  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="effects"):
        model(panel, y="y", x=["x1"], effects=["firm", 1]).fit()  # type: ignore[list-item]


# ── 列数・重複・他の役割との重複 ─────────────────────────────────────────


@pytest.mark.parametrize(("name", "model"), MODELS)
def test_empty_effects_raises(name, model, panel):
    with raises_error("ROLE_EMPTY", argument="effects"):
        _fit(model, panel, [])


@pytest.mark.parametrize(("name", "model"), MODELS)
def test_three_effects_raise_effects_too_many(name, model, panel):
    with raises_error(
        "EFFECTS_TOO_MANY", model=name, n_effects=3, max_effects=2
    ):
        _fit(model, panel, ["firm", "year", "group"])


@pytest.mark.parametrize(("name", "model"), MODELS)
def test_too_many_effects_is_reported_before_other_checks(name, model, panel):
    """列数の検査は列の役割の検査より先（重複があっても列数のエラー）。"""
    with raises_error("EFFECTS_TOO_MANY"):
        _fit(model, panel, ["firm", "firm", "year"])


@pytest.mark.parametrize(("name", "model"), MODELS)
def test_duplicate_effects_raise(name, model, panel):
    with raises_error(
        "ROLE_DUPLICATE_COLUMN", column="firm", argument="effects"
    ):
        _fit(model, panel, ["firm", "firm"])


@pytest.mark.parametrize(("name", "model"), MODELS)
@pytest.mark.parametrize("column", ["y", "x1"])
def test_effects_overlapping_y_or_x_raise(name, model, panel, column):
    # 先に宣言された役割（`y`、次いで`x`の側）が`argument`、`effects`が相手になる。
    with raises_error(
        "ROLE_COLUMN_OVERLAP",
        column=column,
        argument="y" if column == "y" else "x",
        other_argument="effects",
    ):
        _fit(model, panel, ["firm", column])


# ── 1要素と2要素 ────────────────────────────────────────────────────────


@pytest.mark.parametrize(("name", "model"), MODELS)
def test_n_groups_has_one_entry_per_effect_in_order(name, model, panel):
    one_way = _fit(model, panel, ["firm"])
    two_way = _fit(model, panel, ["firm", "year"])
    swapped = _fit(model, panel, ["year", "firm"])

    assert one_way.n_groups == {"firm": N_ENTITIES}
    assert two_way.n_groups == {"firm": N_ENTITIES, "year": N_PERIODS}
    assert list(swapped.n_groups) == ["year", "firm"]


def test_fixed_effects_are_keyed_by_the_effects_columns(panel):
    one_way = _fit(FE, panel, ["firm"]).fixed_effects()
    two_way = _fit(FE, panel, ["firm", "year"]).fixed_effects()

    assert list(one_way) == ["firm"]
    assert len(one_way["firm"]) == N_ENTITIES
    assert list(two_way) == ["firm", "year"]
    assert len(two_way["firm"]) == N_ENTITIES
    assert len(two_way["year"]) == N_PERIODS


# ── 先頭の列がデフォルトのクラスターキー ─────────────────────────────────


@pytest.mark.parametrize(("name", "model"), MODELS)
def test_first_effect_is_the_default_cluster_key(name, model, panel):
    default = _fit(model, panel, ["firm", "year"])
    on_first = _fit(
        model, panel, ["firm", "year"], cov_type="cluster", cluster="firm"
    )
    on_second = _fit(
        model, panel, ["firm", "year"], cov_type="cluster", cluster="year"
    )

    assert default.std_errors == pytest.approx(on_first.std_errors, rel=1e-12)
    assert default.std_errors != pytest.approx(on_second.std_errors, rel=1e-6)


# ── 2要素の順序 ─────────────────────────────────────────────────────────


def test_fe_two_way_coefficients_do_not_depend_on_the_order(panel):
    forward = _fit(FE, panel, ["firm", "year"], cov_type="classical")
    reverse = _fit(FE, panel, ["year", "firm"], cov_type="classical")

    assert forward.params == pytest.approx(reverse.params, rel=1e-10)
    assert forward.std_errors == pytest.approx(reverse.std_errors, rel=1e-10)
    assert forward.df_resid == reverse.df_resid


def test_fe_two_way_order_changes_the_default_cluster_key(panel):
    forward = _fit(FE, panel, ["firm", "year"])
    reverse = _fit(FE, panel, ["year", "firm"])

    assert forward.params == pytest.approx(reverse.params, rel=1e-10)
    assert forward.std_errors != pytest.approx(reverse.std_errors, rel=1e-6)


def test_re_two_way_order_swaps_the_entity_and_time_components(panel):
    """REの2要素は順序に意味がある: 入れ替えると個体側と時点側の成分が入れ替わる。"""
    forward = _fit(RE, panel, ["firm", "year"], cov_type="classical")
    reverse = _fit(RE, panel, ["year", "firm"], cov_type="classical")

    assert forward.sigma2_u == pytest.approx(reverse.sigma2_lambda, rel=1e-10)
    assert forward.sigma2_lambda == pytest.approx(reverse.sigma2_u, rel=1e-10)
    assert forward.theta_entity == pytest.approx(reverse.theta_time, rel=1e-10)
    assert forward.theta_time == pytest.approx(reverse.theta_entity, rel=1e-10)
    assert forward.theta_total == pytest.approx(reverse.theta_total, rel=1e-10)
    assert forward.params == pytest.approx(reverse.params, rel=1e-10)
