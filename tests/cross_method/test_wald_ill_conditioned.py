"""傾き係数が準共線のとき、全体Wald/F統計量が結果の`cov_params`から厳密に求めた値に一致すること。

係数の共分散行列は`(X'X)⁻¹ Ŝ (X'X)⁻¹`のような行列積で組むため、丸め誤差で左右がわずかに
非対称になる。傾き係数が準共線だと、その非対称が条件数倍に増幅される。全体Wald/Fの二次形式は
`cov_params`の傾き部分行列を対称化してから反転する（`shared::wald`）。ここでは、返された
`cov_params`と係数から有理数で厳密に求めた二次形式を真値とし、2SLS・GMM・Tobitが一致することを
確かめる（対称化が無いと、このデータでは相対誤差が数%から数百%になる）。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import polars as pl
import pytest
from econometricsmodels import IV, IVOptions, Tobit, TobitOptions

# 2つの傾き（a, b）の相関が1 - 5e-9程度になる強い準共線。
EPS = 1e-4
N = 600
# 厳密解は有理数なので、残る誤差は共分散行列を引数にした二次形式の条件数による分（実測1e-8程度）。
RTOL = 1e-6


def _data() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    a = rng.normal(size=N)
    b = a + EPS * rng.normal(size=N)
    z = rng.normal(size=(N, 3))
    d = z @ np.array([1.0, 0.4, 0.2]) + rng.normal(size=N)
    y = 0.5 * a + 0.4 * b + 0.7 * d + rng.normal(size=N)
    return pl.DataFrame(
        {
            "y": y,
            "ty": np.maximum(y, 0.0),
            "a": a,
            "b": b,
            "d": d,
            "z1": z[:, 0],
            "z2": z[:, 1],
            "z3": z[:, 2],
            "g": (np.arange(N) % 31).astype(str),
        }
    )


def _solve(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction]:
    n = len(rhs)
    rows = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = next(r for r in range(i, n) if rows[r][i] != 0)
        rows[i], rows[pivot] = rows[pivot], rows[i]
        for r in range(n):
            if r != i and rows[r][i] != 0:
                factor = rows[r][i] / rows[i][i]
                rows[r] = [x - factor * p for x, p in zip(rows[r], rows[i])]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def _exact_wald_quadratic_form(res) -> float:
    """`cov_params`の傾き部分行列（対称化済み）による`β'V⁻¹β`を有理数で厳密に求める。"""
    names = [n for n in res.param_names if n not in ("const", "sigma")]
    cov = res.cov_params
    v = [
        [(Fraction(cov[i][j]) + Fraction(cov[j][i])) / 2 for j in names]
        for i in names
    ]
    beta = [Fraction(res.params[n]) for n in names]
    solved = _solve(v, beta)
    return float(sum(b * s for b, s in zip(beta, solved)))


def _assert_relative_close(got: float, expected: float) -> None:
    assert abs(got - expected) <= RTOL * abs(expected), (got, expected)


@pytest.mark.parametrize("cov_type", ["hc1", "cluster", "hac"])
def test_iv_2sls_wald_f_matches_exact_quadratic_form(cov_type):
    kwargs = {"cluster": "g"} if cov_type == "cluster" else {}
    if cov_type == "hac":
        kwargs["hac_time"] = "t"
    df = _data().with_columns(pl.int_range(N).cast(pl.Float64).alias("t"))
    res = IV(
        df,
        "y",
        ["a", "b"],
        ["d"],
        ["z1", "z2", "z3"],
        options=IVOptions(estimator="2sls", cov_type=cov_type, **kwargs),
    ).fit()

    n_slopes = len(res.param_names) - 1
    # 2SLSの統計量はF = W / q。
    _assert_relative_close(
        res.wald_statistic, _exact_wald_quadratic_form(res) / n_slopes
    )


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
def test_iv_gmm_wald_chi2_matches_exact_quadratic_form(cov_type):
    kwargs = {"cluster": "g"} if cov_type == "cluster" else {}
    res = IV(
        _data(),
        "y",
        ["a", "b"],
        ["d"],
        ["z1", "z2", "z3"],
        options=IVOptions(
            estimator="gmm",
            gmm_type="two_step",
            gmm_weight_type="classical",
            cov_type=cov_type,
            **kwargs,
        ),
    ).fit()

    # GMMの統計量はχ²（q で割らない生の二次形式）。
    _assert_relative_close(res.wald_statistic, _exact_wald_quadratic_form(res))


@pytest.mark.parametrize("cov_type", ["hc0", "hc1", "cluster"])
def test_tobit_wald_chi2_matches_exact_quadratic_form(cov_type):
    kwargs = {"cluster": "g"} if cov_type == "cluster" else {}
    res = Tobit(
        _data(),
        "ty",
        ["a", "b", "d"],
        options=TobitOptions(cov_type=cov_type, lower=0.0, **kwargs),
    ).fit()

    _assert_relative_close(res.wald_statistic, _exact_wald_quadratic_form(res))
