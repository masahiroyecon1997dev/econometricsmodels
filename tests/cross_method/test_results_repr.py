"""結果クラスの`repr()`のテスト。

`repr(result)`は、被説明変数・観測数・`cov_type`と、係数ごとの`param`・`std_error`を
数行で見せる（デバッグやノートブックでの確認用）。書式は公開APIではないので、
テストは書式の細部ではなく、含まれる情報と形（1行目＋表）だけを確かめる。
機械可読な出力は`params_table()`が担う。
"""

from __future__ import annotations

import polars as pl
import pytest
from _cross_method_helpers import fit_fe, fit_re
from econometricsmodels import (
    IV,
    OLS,
    WLS,
    Logit,
    Probit,
    Tobit,
    TobitOptions,
)
from econometricsmodels._results_utils import results_repr


def _fit_ols(df):
    return OLS(df, y="y", x=["x1", "x2"]).fit()


def _fit_wls(df):
    return WLS(df, y="y", x=["x1", "x2"], weight="w").fit()


def _fit_iv(df):
    return IV(
        df,
        y="y",
        x_endog=["endog"],
        x_exog=["x1"],
        instruments=["z1", "z2"],
    ).fit()


def _fit_logit(df):
    return Logit(df, y="y_bin", x=["x1", "x2"]).fit()


def _fit_probit(df):
    return Probit(df, y="y_bin", x=["x1", "x2"]).fit()


def _fit_tobit(df):
    return Tobit(
        df, y="y_cens", x=["x1", "x2"], options=TobitOptions(lower=0.5)
    ).fit()


FITS = {
    "OLSResults": _fit_ols,
    "WLSResults": _fit_wls,
    "IVResults": _fit_iv,
    "FEResults": fit_fe,
    "REResults": fit_re,
    "LogitResults": _fit_logit,
    "ProbitResults": _fit_probit,
    "TobitResults": _fit_tobit,
}


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
    rows = lines[2:]
    assert [row.split()[0] for row in rows] == result.param_names
    for row, param_name in zip(rows, result.param_names):
        _, param, std_error = row.split()
        assert float(param) == pytest.approx(
            result.params[param_name], rel=1e-5
        )
        assert float(std_error) == pytest.approx(
            result.std_errors[param_name], rel=1e-5
        )


def test_repr_of_tobit_includes_the_sigma_row(panel_df):
    result = _fit_tobit(panel_df)

    assert "sigma" in result.param_names
    assert repr(result).splitlines()[-1].split()[0] == "sigma"


def test_repr_does_not_depend_on_str_or_print_hooks(panel_df):
    result = _fit_ols(panel_df)

    assert str(result) == repr(result)
    assert "object at 0x" not in repr(result)


class _FakeRaw:
    """係数が多いときの省略を確かめるための、`_lib`の結果を真似た最小の入れ物。"""

    dep_var_name = "y"
    n_obs = 5
    cov_type = "classical"

    def __init__(self, k: int) -> None:
        self.param_names = [f"x{i}" for i in range(k)]
        self.params = [float(i) for i in range(k)]
        self.std_errors = [0.5] * k


def test_repr_truncates_after_twenty_coefficients():
    lines = results_repr("OLSResults", _FakeRaw(25)).splitlines()

    assert len(lines) == 1 + 1 + 20 + 1
    assert lines[-1] == "... (5 more)"
    assert lines[-2].split()[0] == "x19"


def test_repr_does_not_truncate_up_to_twenty_coefficients():
    lines = results_repr("OLSResults", _FakeRaw(20)).splitlines()

    assert len(lines) == 1 + 1 + 20
    assert not lines[-1].startswith("...")


def test_repr_keeps_tiny_and_huge_values_readable():
    raw = _FakeRaw(2)
    raw.params = [2.5e-9, 1.25e12]
    raw.std_errors = [1e-10, 3.0]

    text = results_repr("OLSResults", raw)

    assert "2.5e-09" in text
    assert "1.25e+12" in text


def test_repr_matches_between_fits_of_the_same_data(panel_df):
    df: pl.DataFrame = panel_df

    assert repr(_fit_ols(df)) == repr(_fit_ols(df))
