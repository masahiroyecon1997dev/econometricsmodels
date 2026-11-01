"""結果クラスの`repr()`のテスト。

`repr(result)`は、被説明変数・観測数・`cov_type`と、係数ごとの`param`・`std_error`を
数行で見せる（デバッグやノートブックでの確認用）。書式は公開APIではないので、
テストは列幅などの細部ではなく、含まれる情報と形（1行目＋表）だけを確かめる。
値の整形（`%.6g`）の精度は仕様として固定する。機械可読な出力は`params_table()`が担う。
係数名に空白を含まない入力では、行は空白で分割して読める（表の読み取りに使う）。
"""

from __future__ import annotations

import copy
import pickle
from functools import partial

import econometricsmodels
import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _tolerances import TOLERANCES
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    FEOptions,
    IVOptions,
    Logit,
    LogitOptions,
    OLSOptions,
    OLSResults,
    Probit,
    ProbitOptions,
    REOptions,
    Tobit,
    TobitOptions,
    WLSOptions,
)
from econometricsmodels._results_utils import _REPR_MAX_ROWS, results_repr

assert_repr_close = partial(assert_close, **TOLERANCES["results_repr"])

X = ["x1", "x2"]


class _FakeRaw:
    """`_lib`の結果を真似た最小の入れ物。名前と値を自由に決められる。"""

    dep_var_name = "y"
    n_obs = 5
    cov_type = "classical"

    def __init__(
        self,
        names: list[str],
        params: list[float] | None = None,
        std_errors: list[float] | None = None,
    ) -> None:
        self.param_names = names
        self.params = params or [float(i) for i in range(len(names))]
        self.std_errors = std_errors or [0.5] * len(names)


def _fake(k: int) -> _FakeRaw:
    return _FakeRaw([f"x{i}" for i in range(k)])


def _ols(df, **options):
    return OLS(df, y="y", x=X, options=OLSOptions(**options)).fit()


def _wls(df, **options):
    return WLS(df, y="y", x=X, weight="w", options=WLSOptions(**options)).fit()


def _logit(df, **options):
    return Logit(df, y="y_bin", x=X, options=LogitOptions(**options)).fit()


def _probit(df, **options):
    return Probit(df, y="y_bin", x=X, options=ProbitOptions(**options)).fit()


def _tobit(df, *, x=X, **options):
    options = {"lower": 0.5, **options}
    return Tobit(df, y="y_cens", x=x, options=TobitOptions(**options)).fit()


def _iv(df, **options):
    return IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog"],
        instruments=["z1", "z2"],
        options=IVOptions(**options),
    ).fit()


def _fe(df, **options):
    return FE(
        df, y="y", x=X, effects=["entity"], options=FEOptions(**options)
    ).fit()


def _re(df, **options):
    return RE(
        df, y="y", x=X, effects=["entity"], options=REOptions(**options)
    ).fit()


# 手法の順は linear → nonlinear → iv → panel。
FITS = {
    "OLSResults": _ols,
    "WLSResults": _wls,
    "LogitResults": _logit,
    "ProbitResults": _probit,
    "TobitResults": _tobit,
    "IVResults": _iv,
    "FEResults": _fe,
    "REResults": _re,
}

RESULTS = [
    name for name in econometricsmodels.__all__ if name.endswith("Results")
]


def _frame(panel_df: pl.DataFrame) -> pl.DataFrame:
    """HAC用の重複しない順序列`row`を足したデータ。"""
    return panel_df.with_row_index("row")


def _wide_frame(panel_df: pl.DataFrame, k: int) -> pl.DataFrame:
    """独立な説明変数`v0`..`v{k-1}`を足したデータ。"""
    rng = np.random.default_rng(11)
    return panel_df.with_columns(
        [
            pl.Series(f"v{i}", rng.normal(size=panel_df.height))
            for i in range(k)
        ]
    )


def _table(result) -> list[list[str]]:
    """表の行を`[名前, param, std_error]`に分ける（1行目と列見出しは除く）。"""
    return [line.split() for line in repr(result).splitlines()[2:]]


def test_every_results_class_is_covered_here():
    assert set(FITS) == set(RESULTS)


# ── 全手法の形 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", FITS)
def test_repr_shows_header_and_one_row_per_coefficient(name, panel_df):
    result = FITS[name](panel_df)
    lines = repr(result).splitlines()

    assert type(result).__name__ == name
    assert lines[0] == (
        f"{name}(dep_var={result.dep_var_name!r}, "
        f"n_obs={result.n_obs}, cov_type={result.cov_type!r})"
    )
    assert lines[1].split() == ["param", "std_error"]
    rows = _table(result)
    assert [row[0] for row in rows] == result.param_names
    for row_name, param, std_error in rows:
        assert_repr_close(
            float(param), result.params[row_name], f"{name} {row_name} param"
        )
        assert_repr_close(
            float(std_error),
            result.std_errors[row_name],
            f"{name} {row_name} std_error",
        )


# 手法ごとに、既定とは違う`cov_type`（綴りの大文字小文字も含む）で推定する。
COV_TYPE_CASES = {
    "ols-hc1": (_ols, {"cov_type": "hc1"}, "hc1"),
    "ols-upper-case": (_ols, {"cov_type": "HC1"}, "hc1"),
    "ols-hac": (_ols, {"cov_type": "hac", "hac_time": "row"}, "hac"),
    "wls-cluster": (
        _wls,
        {"cov_type": "cluster", "cluster": "entity"},
        "cluster",
    ),
    "logit-cluster": (
        _logit,
        {"cov_type": "cluster", "cluster": "entity"},
        "cluster",
    ),
    "probit-hc1": (_probit, {"cov_type": "hc1"}, "hc1"),
    "tobit-cluster": (
        _tobit,
        {"cov_type": "cluster", "cluster": "entity"},
        "cluster",
    ),
    "iv-hc1": (_iv, {"cov_type": "hc1"}, "hc1"),
    "fe-dk": (_fe, {"cov_type": "dk", "dk_time": "time"}, "dk"),
    "re-hc1": (_re, {"cov_type": "hc1"}, "hc1"),
}


@pytest.mark.parametrize("case", COV_TYPE_CASES)
def test_header_shows_the_cov_type_of_the_fit(case, panel_df):
    fit, options, expected = COV_TYPE_CASES[case]
    result = fit(_frame(panel_df), **options)

    assert repr(result).splitlines()[0].endswith(f"cov_type={expected!r})")


@pytest.mark.parametrize(
    "fit", [_ols, _wls, _logit, _probit, _tobit, _iv], ids=lambda f: f.__name__
)
def test_include_intercept_false_has_no_const_row(fit, panel_df):
    result = fit(panel_df, include_intercept=False)

    assert "const" not in result.param_names
    assert [row[0] for row in _table(result)] == result.param_names


def test_repr_of_tobit_ends_with_the_sigma_row(panel_df):
    for options in ({}, {"include_intercept": False}):
        result = _tobit(panel_df, **options)

        assert _table(result)[-1][0] == "sigma"


@pytest.mark.parametrize("name", FITS)
@pytest.mark.parametrize(
    "roundtrip",
    [
        lambda r: pickle.loads(pickle.dumps(r)),
        copy.copy,
        copy.deepcopy,
    ],
    ids=["pickle", "copy", "deepcopy"],
)
def test_repr_survives_pickle_and_copy(name, roundtrip, panel_df):
    result = FITS[name](panel_df)

    assert repr(roundtrip(result)) == repr(result)


def test_repr_names_the_subclass():
    class MyResults(OLSResults):
        pass

    raw = (
        OLS(
            pl.DataFrame(
                {"y": [1.0, 2.0, 4.0, 3.0], "x1": [1.0, 2.0, 3.0, 5.0]}
            ),
            y="y",
            x=["x1"],
        )
        .fit()
        ._raw
    )

    assert repr(MyResults(raw)).startswith("MyResults(")


# ── 係数が多いときの省略 ──────────────────────────────────────────────


def test_repr_does_not_truncate_up_to_the_limit():
    lines = results_repr("OLSResults", _fake(_REPR_MAX_ROWS)).splitlines()

    assert len(lines) == 2 + _REPR_MAX_ROWS
    assert not any(line.startswith("...") for line in lines)


@pytest.mark.parametrize("extra", [1, 5])
def test_repr_keeps_the_first_rows_and_the_last_one_when_truncated(extra):
    count = _REPR_MAX_ROWS + extra
    lines = results_repr("OLSResults", _fake(count)).splitlines()

    assert len(lines) == 2 + _REPR_MAX_ROWS + 1
    shown = [line.split()[0] for line in lines[2:]]
    assert shown[:-2] == [f"x{i}" for i in range(_REPR_MAX_ROWS - 1)]
    assert shown[-2] == "..."
    assert lines[-2] == f"... ({extra} more)"
    assert shown[-1] == f"x{count - 1}"
    omitted = {f"x{i}" for i in range(_REPR_MAX_ROWS - 1, count - 1)}
    assert not omitted & set(shown)


def test_repr_truncates_a_real_model_with_many_coefficients(panel_df):
    k = _REPR_MAX_ROWS + 2
    names = [f"v{i}" for i in range(k)]
    result = OLS(_wide_frame(panel_df, k), y="y", x=names).fit()

    omitted = len(result.param_names) - _REPR_MAX_ROWS
    lines = repr(result).splitlines()
    assert omitted > 0
    assert len(lines) == 2 + _REPR_MAX_ROWS + 1
    assert lines[-2] == f"... ({omitted} more)"
    assert lines[-1].split()[0] == result.param_names[-1]


def test_repr_of_a_truncated_tobit_still_shows_sigma(panel_df):
    k = _REPR_MAX_ROWS + 1
    names = [f"v{i}" for i in range(k)]
    result = _tobit(_wide_frame(panel_df, k), x=names)

    lines = repr(result).splitlines()
    assert any(line.startswith("...") for line in lines)
    assert lines[-1].split()[0] == "sigma"


# ── 整形 ──────────────────────────────────────────────────────────────


def test_repr_escapes_and_cuts_coefficient_names():
    raw = _FakeRaw(["b\nc", "a" * 200, "x"])

    lines = results_repr("OLSResults", raw).splitlines()

    assert len(lines) == 2 + 3
    assert lines[2].startswith("b\\nc")
    assert lines[3].startswith("a" * 29 + "\u2026 ")
    assert lines[4].startswith("x ")


def test_repr_of_a_model_without_coefficients_is_the_header_only():
    lines = results_repr("OLSResults", _FakeRaw([])).splitlines()

    assert len(lines) == 2
    assert lines[1].split() == ["param", "std_error"]


def test_repr_does_not_fail_on_non_finite_values():
    raw = _FakeRaw(
        ["a", "b"],
        params=[float("nan"), float("inf")],
        std_errors=[float("-inf"), 1e300],
    )

    text = results_repr("OLSResults", raw)

    for expected in ("nan", "inf", "-inf", "1e+300"):
        assert expected in text.split()


def test_repr_keeps_tiny_and_huge_values_readable():
    raw = _FakeRaw(
        ["a", "b"], params=[2.5e-9, 1.25e12], std_errors=[1e-10, -3.0]
    )

    rows = [
        line.split()
        for line in results_repr("OLSResults", raw).splitlines()[2:]
    ]

    for (_, param, std_error), want_param, want_se in zip(
        rows, raw.params, raw.std_errors
    ):
        assert_repr_close(float(param), want_param, "param")
        assert_repr_close(float(std_error), want_se, "std_error")
