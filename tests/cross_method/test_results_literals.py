"""`IVResults`・`FEIVResults`の`Literal`型の戻り値が、実際に返る値と一致することのテスト。

`estimator`・`gmm_type`・`gmm_weight_type`は、Rust側（`engine_pybind`）が正規形の文字列を
返し、Python側が`Literal[...]`で型を付けている。Rust側の名前が増減しても型が気づかれずに
ずれないよう、`Literal`の値の集合が、取りうる設定すべてで得られる値の集合と一致する
（過不足がない）ことを確かめる。
"""

from __future__ import annotations

import types
import typing

import polars as pl
import pytest
from econometricsmodels import FEIV, IV, FEIVOptions, IVOptions

PROPERTIES = ("estimator", "gmm_type", "gmm_weight_type")


def _literal_values(result_class: type, name: str) -> set[str]:
    """プロパティの戻り値の型に含まれる`Literal`の値の集合（`None`は除く）。"""
    hint = typing.get_type_hints(getattr(result_class, name).fget)["return"]
    members = (
        typing.get_args(hint)
        if typing.get_origin(hint) in (typing.Union, types.UnionType)
        else (hint,)
    )
    values: set[str] = set()
    for member in members:
        if member is type(None):
            continue
        assert typing.get_origin(member) is typing.Literal, (name, member)
        values.update(typing.get_args(member))
    return values


@pytest.fixture(scope="module")
def df(panel_df: pl.DataFrame) -> pl.DataFrame:
    # `hac`の重みは時点の順位に同順位が無いこと（行ごとに異なる値）が要る。
    return panel_df.with_row_index("row")


def _iv_options() -> list[IVOptions]:
    options = [IVOptions(estimator="2sls")]
    options.append(IVOptions(estimator="gmm", gmm_type="one_step"))
    for gmm_type in ("two_step", "iterated"):
        for weight in ("classical", "robust", "cluster", "hac"):
            options.append(
                IVOptions(
                    estimator="gmm",
                    gmm_type=gmm_type,
                    gmm_weight_type=weight,
                    cluster="entity" if weight == "cluster" else None,
                    hac_time="row" if weight == "hac" else None,
                    cov_type="classical",
                )
            )
    return options


def _fe_iv_options() -> list[FEIVOptions]:
    options = [FEIVOptions(estimator="2sls")]
    options.append(FEIVOptions(estimator="gmm", gmm_type="one_step"))
    for gmm_type in ("two_step", "iterated"):
        for weight in ("classical", "robust", "cluster"):
            options.append(
                FEIVOptions(
                    estimator="gmm",
                    gmm_type=gmm_type,
                    gmm_weight_type=weight,
                )
            )
    return options


def _observed(results: list) -> dict[str, set[str]]:
    observed: dict[str, set[str]] = {name: set() for name in PROPERTIES}
    for result in results:
        for name in PROPERTIES:
            value = getattr(result, name)
            if value is not None:
                observed[name].add(value)
    return observed


def test_iv_results_literals_match_the_values_returned(df):
    results = [
        IV(
            df,
            y="y",
            x_exog=["x1"],
            x_endog=["endog"],
            instruments=["z1", "z2"],
            options=options,
        ).fit()
        for options in _iv_options()
    ]
    observed = _observed(results)
    # 2SLSと`one_step`は`gmm_weight_type`が`None`になる（型に`None`を含む）。
    assert any(r.gmm_type is None for r in results)
    assert any(r.gmm_weight_type is None for r in results)
    for name in PROPERTIES:
        assert observed[name] == _literal_values(type(results[0]), name), name


def test_fe_iv_results_literals_match_the_values_returned(df):
    results = [
        FEIV(
            df,
            y="y",
            x_exog=["x1"],
            x_endog=["endog"],
            instruments=["z1", "z2"],
            effects=["entity"],
            options=options,
        ).fit()
        for options in _fe_iv_options()
    ]
    observed = _observed(results)
    assert any(r.gmm_type is None for r in results)
    assert any(r.gmm_weight_type is None for r in results)
    for name in PROPERTIES:
        assert observed[name] == _literal_values(type(results[0]), name), name
