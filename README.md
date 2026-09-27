# econometricsmodels

[![CI (engine)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml)
[![CI (python)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml)
[![Docs](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/cd_docs.yml/badge.svg)](https://masahiroyecon1997dev.github.io/econometricsmodels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is econometricsmodels?

**econometricsmodels** provides econometric methods through a consistent, programmatic Python API, with a growing focus on causal inference. The API covers the workflow from model specification and validation to structured results.

## Key Design Principles

- **Consistent API** — Models follow a consistent interface: a DataFrame, a
  `y` column, a list of `x` columns, and an options object in; a `.fit()`
  call out. Moving from OLS to Logit, IV, or panel models doesn't mean
  learning a new interface.
- **Programmatic specification and results** — Models are defined with
  explicit arguments and an options object, not a formula string. Results
  come back as Python objects (`.params`, `.std_errors`, `.p_values`,
  `.conf_int`), not formatted text, so they are easy to build, inspect, and
  feed into other code.
- **Explicit validation** — Problematic inputs are rejected rather than
  silently converted, dropped, or ignored. Missing values, degenerate
  groups, and other invalid inputs raise an error rather than being quietly
  dropped or coerced.
- **Modern dataframe workflow** — Built for [polars](https://pola.rs/),
  with data passed to the Rust core through Arrow, avoiding unnecessary
  conversions and copies.
- **Toward accessible causal inference** — Econometric and causal methods
  made available to a wider range of developers, under the MIT license so
  it is easy to embed in your own tools.

## Is this for you?

**Good fit:**

- Calling estimation methods from scripts, pipelines, or apps — the API is
  built for programmatic construction, not interactive formula-writing.
- Wanting explicit validation on top of your estimates, without giving up
  broad econometric coverage (see [Numerical verification](#numerical-verification)).
- Projects that use more than one method over time — the same interface
  covers linear, discrete-choice, IV, and panel models today, with more on
  the way (see [Implemented models](#implemented-models)).
- Already working in polars, where handing data to the Rust core through
  Arrow avoids unnecessary conversion overhead.

**Probably not a good fit:**

- You prefer R-style formula syntax (`y ~ x1 + x2`).
- You need causal designs such as difference-in-differences or regression
  discontinuity, which are still on the roadmap.
- You need the broader diagnostics and edge-case coverage of
  long-established tools like statsmodels and the R ecosystem.

## Quickstart

```python
import polars as pl
from econometricsmodels import OLS, Logit, Probit

df = pl.DataFrame({"y": [...], "x1": [...], "x2": [...]})

dependent = "y"
independent = ["x1", "x2"]

# Assume y is binary here; OLS is therefore a linear probability model.
ols = OLS(df, y=dependent, x=independent).fit()
logit = Logit(df, y=dependent, x=independent).fit()
probit = Probit(df, y=dependent, x=independent).fit()

for name, result in [
    ("OLS", ols),
    ("Logit", logit),
    ("Probit", probit),
]:
    print(name)
    print(result.params)
    print(result.std_errors)
```

## Programmatic Results & Reporting

Results expose their values as ordinary Python data structures, including
dictionaries for parameter values and row-oriented data from
`coef_table()`:

```python
result.params
# {"const": 0.42, "x1": 1.87}

result.coef_table()
# [
#     {"param": "const", "coef": 0.42, "std_err": 0.11,
#      "test_stat": 3.8, "p_value": 0.0002, ...},
#     {"param": "x1", "coef": 1.87, "std_err": 0.23, ...},
# ]
```

There is no formatted text to parse. `coef_table()` is shaped so it can
be passed directly into a Polars DataFrame, a JSON API response, or an
automated reporting workflow:

```python
pl.DataFrame(result.coef_table())

json.dumps(result.coef_table())
```

## Implemented Models

Currently implemented:

- OLS, WLS
- Logit, Probit, Tobit
- IV (2SLS, GMM)
- FE (fixed effects), RE (random effects)

The roadmap focuses primarily on econometric and causal inference methods,
with additional structural microeconometric methods planned as the project
expands.

See the [documentation](https://masahiroyecon1997dev.github.io/econometricsmodels/)
for usage details on each method, and
[GitHub Issues](https://github.com/masahiroyecon1997dev/econometricsmodels/issues)
for the current status of planned and in-progress work.

## Numerical Verification

Every estimator is numerically checked against established reference
implementations before it is considered done. We use the reference
implementation that is most appropriate and widely used for each method
(statsmodels for many regression-family estimators, R packages elsewhere),
with an independent implementation used as a cross-check where available.

We aim for close numerical agreement and investigate discrepancies rather
than treating them as acceptable by default. Where implementations use
different statistical conventions, such as small-sample corrections, the
differences are documented and the comparison criteria are adjusted
accordingly.

See the [inference conventions guide](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/inference-conventions/)
for method-by-method details.

## Performance

The computational core is written in Rust, so the aim is that the
extensive validation above doesn't come at the cost of speed in practice —
calling `fit()` should never be the bottleneck. Detailed benchmarks against
statsmodels/linearmodels, and known performance limitations, are tracked in
[`docs/performance/`](docs/performance/).

## Installation

```bash
pip install econometricsmodels
```

Requires Python 3.12 or later. The computational core is pure Rust
(no system BLAS/LAPACK dependency), with prebuilt wheels for Linux,
macOS, and Windows.

## Limitations / Disclaimer

This is a solo-maintained, pre-1.0 project. It has not yet undergone
the years of community scrutiny and extensive edge-case testing that
established tools such as statsmodels and the R ecosystem have. See
[Numerical verification](#numerical-verification) for how estimates are
checked. For published research or other important analyses, we recommend
cross-checking results against a trusted, established package.

Before 1.0, breaking changes may be introduced in minor version releases,
not just major ones.

## Documentation / Links

- [Documentation](https://masahiroyecon1997dev.github.io/econometricsmodels/) — usage guides and full API reference
- [PyPI](https://pypi.org/project/econometricsmodels/)
- [GitHub Issues](https://github.com/masahiroyecon1997dev/econometricsmodels/issues) — bug reports, feature requests, roadmap status
- [Changelog](CHANGELOG.md)
- [License](LICENSE) — MIT
