# econometricsmodels

[![CI (engine)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml)
[![CI (python)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml)
[![Docs](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/cd_docs.yml/badge.svg)](https://masahiroyecon1997dev.github.io/econometricsmodels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/blob/main/LICENSE)

## What is econometricsmodels?

**econometricsmodels** is a Python package of econometric estimators built for embedding in scripts, pipelines, and apps. It rejects invalid input rather than silently dropping or repairing it, and returns result objects that are easy to work with directly and can also be exported as JSON-ready data. Every estimator is numerically checked against established reference implementations such as statsmodels, linearmodels, and R, and the comparisons are published.

## Key Design Principles

- **Consistent API** — Every model follows the same conventions: a polars
  DataFrame, column names as strings, and an options object in; a `.fit()`
  call out, returning a result with the same accessors. Methods with extra
  inputs (instruments for IV, an entity column for panel models) add
  arguments, but moving from OLS to Logit, IV, or panel models doesn't mean
  learning a new interface design.
- **Programmatic specification and results** — Models are defined with
  explicit arguments and an options object, not a formula string. Results
  come back as Python objects (`.params`, `.std_errors`, `.p_values`,
  `.conf_int`), not formatted text, so they are easy to build, inspect, and
  feed into other code. Coefficient tables (`coef_table()`) are plain lists
  of dicts that can be passed straight to polars or `json.dumps`.
- **Explicit validation** — Invalid input is rejected rather than silently
  dropped or repaired. Missing values, perfectly collinear columns,
  degenerate groups, and options that would have no effect raise an error
  instead of being quietly handled, and the error message states the cause
  (for example "perfect multicollinearity detected") rather than a bare
  "singular matrix" (see
  [Validation and errors](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/validation/)).
- **Modern dataframe workflow** — Built for [polars](https://pola.rs/),
  with data passed to the Rust core through Arrow, avoiding unnecessary
  conversions and copies.
- **Verified against established packages** — statsmodels, linearmodels,
  and the R ecosystem are mature yardsticks, and building after them means
  every estimator can be tested against them from the start. Each estimator
  is compared against at least one reference implementation, the
  comparisons run in CI on every pull request, and differences in
  conventions (such as small-sample corrections) are documented (see
  [Numerical verification](#numerical-verification)).
- **Performance as a design goal** — Strict validation should not come at
  the cost of speed. The Rust core is designed to be competitive in speed
  and memory with the packages it is verified against, and representative
  cases of each implemented method are benchmarked (against those packages
  wherever a comparable implementation exists), with results published as
  measured, including cases where it is not faster (see
  [Performance](#performance)).

## Is this for you?

**Good fit:**

- Calling estimation methods from scripts, pipelines, or apps — the API is
  built for programmatic construction, not interactive formula-writing.
- Wanting invalid input to fail loudly instead of being silently dropped or
  repaired — missing values, perfectly collinear columns, and unusable
  options raise an error that says why (see
  [Validation and errors](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/validation/)).
- Needing to confirm that results match statsmodels, R, or linearmodels —
  for example when migrating an existing analysis, or when you have to show
  how your numbers were checked (see
  [Numerical verification](#numerical-verification)).
- Projects that use more than one method over time — the same interface
  covers linear, discrete-choice, IV, and panel models today, with more on
  the way (see [Implemented models](#implemented-models)).
- Already working in polars, where handing data to the Rust core through
  Arrow avoids unnecessary conversion overhead.

**Probably not a good fit:**

- You prefer R-style formula syntax (`y ~ x1 + x2`).
- You need the broader diagnostics and edge-case coverage of
  long-established tools like statsmodels and the R ecosystem.

## Installation

```bash
pip install econometricsmodels
```

Requires Python 3.12 or later. The computational core is pure Rust
(no system BLAS/LAPACK dependency), with prebuilt wheels for Linux,
macOS, and Windows.

## Quickstart

```python
import random

import polars as pl
from econometricsmodels import OLS, Logit, Probit

# Simulated data with a binary outcome.
random.seed(0)
n = 500
x1 = [random.gauss(0, 1) for _ in range(n)]
x2 = [random.gauss(0, 1) for _ in range(n)]
y = [int(0.3 + a - 0.8 * b + random.gauss(0, 1) > 0) for a, b in zip(x1, x2)]
df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

# The same call shape for every model. Since y is binary, OLS here is a
# linear probability model.
for name, model in [("OLS", OLS), ("Logit", Logit), ("Probit", Probit)]:
    result = model(df, y="y", x=["x1", "x2"]).fit()
    print(name)
    print(result.params)
    print(result.std_errors)
```

Estimation options are passed as an options object. For example,
cluster-robust standard errors:

```python
from econometricsmodels import OLS, OLSOptions

df_grouped = df.with_columns((pl.int_range(pl.len()) % 50).alias("group"))
options = OLSOptions(cov_type="cluster", cluster="group")
clustered = OLS(df_grouped, y="y", x=["x1", "x2"], options=options).fit()
print(clustered.std_errors)
```

## Programmatic Results & Reporting

Results expose their values as ordinary Python data structures, including
dictionaries for parameter values and row-oriented data from
`coef_table()`. The output below is for the last model in the Quickstart
(Probit), rounded:

```python
result.params
# {"const": 0.32, "x1": 1.04, "x2": -0.87}

result.coef_table()
# [
#     {"param": "const", "coef": 0.32, "std_err": 0.07,
#      "test_stat": 4.46, "p_value": 8e-06,
#      "conf_lower": 0.18, "conf_upper": 0.46},
#     {"param": "x1", "coef": 1.04, "std_err": 0.09, ...},
#     {"param": "x2", "coef": -0.87, "std_err": 0.09, ...},
# ]
```

There is no formatted text to parse. `coef_table()` is shaped so it can
be passed directly into a Polars DataFrame, a JSON API response, or an
automated reporting workflow:

```python
import json

import polars as pl

pl.DataFrame(result.coef_table())

json.dumps(result.coef_table())
```

## Input Validation

Invalid input is rejected instead of being silently dropped or repaired.
A `ValidationError` means the input or an option cannot be used as given;
a `ComputationError` means the inputs are acceptable on their face but the
numerical computation fails. Two examples, using the `df` from the
Quickstart:

**Missing values are never dropped.** A null, NaN, or infinite value in any
column the model uses raises an error, so the number of observations never
changes without you choosing it.

```python
df_missing = df.with_columns(
    pl.when(pl.int_range(pl.len()) == 3)
    .then(None)
    .otherwise(pl.col("x1"))
    .alias("x1")
)
OLS(df_missing, y="y", x=["x1", "x2"]).fit()
# ValidationError: column 'x1' contains 1 missing value(s). Missing values are
# not handled automatically; please impute or remove them before calling this
# function
```

**Perfectly collinear columns are never removed.** A perfectly collinear
design raises an error that names the cause, instead of being estimated on a
reduced set of regressors or failing with a bare "singular matrix". A design
that is only numerically near-singular (for example, regressors on extremely
different scales) raises a separate error that says so.

```python
df_collinear = df.with_columns((pl.col("x1") * 2).alias("x1_double"))
OLS(df_collinear, y="y", x=["x1", "x1_double"]).fit()
# ComputationError: design matrix is singular (perfect multicollinearity
# detected)
```

Options are checked in the same spirit: for example, passing `cluster`
while `cov_type` is not `"cluster"` raises an error instead of being
ignored. The exception class is the stable interface; the message text may
change between versions. See
[Validation and errors](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/validation/)
for what raises which error and what is deliberately not checked, and
[Accepted data](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/accepted-data/)
for the accepted input and column types.

## Implemented Models

Currently implemented:

- OLS, WLS
- Logit, Probit, Tobit
- IV (2SLS, GMM)
- FE (fixed effects), RE (random effects)

More methods are planned, including further discrete-choice models and
causal inference designs. What is planned or in progress, and in what
order, is tracked in
[GitHub Issues](https://github.com/masahiroyecon1997dev/econometricsmodels/issues).

See the [documentation](https://masahiroyecon1997dev.github.io/econometricsmodels/)
for usage details on each method.

## Numerical Verification

Every estimator is compared numerically against established reference
implementations before it is considered done. For each method, a primary
reference whose statistical conventions match is chosen, and where one
exists an independent implementation (mostly R) is used as a cross-check.
The comparison covers the main reported statistics — coefficients, standard
errors, test statistics, p-values, confidence intervals, fit statistics, and
model tests — for each supported covariance type.

| Method | Primary reference | Independent cross-check | Relative tolerance (primary / cross-check) |
|---|---|---|---|
| OLS | statsmodels `OLS` | R `lm` + `sandwich`/`lmtest` | 1e-8 / 1e-8 (HAC: 1e-2) |
| WLS | statsmodels `WLS` | R `lm(weights=)` + `sandwich`/`lmtest` | 1e-8 / 1e-8 (HAC: 5e-2) |
| Logit | statsmodels `Logit` | R `glm` + `sandwich`, `marginaleffects` | 1e-8 / 1e-6 |
| Probit | statsmodels `Probit` | R `glm` + `sandwich`, `marginaleffects` | 1e-8 / 1e-6 |
| Tobit<sup>2</sup> | R `AER::tobit` | R `censReg` | 1e-8 / 1e-8 |
| IV (2SLS) | linearmodels `IV2SLS` | R `ivreg` + `sandwich`/`lmtest` | 1e-8 / 1e-8 (HAC: 1e-2) |
| IV (GMM) | linearmodels `IVGMM` | R `momentfit` | 1e-8 / 1e-8 |
| FE<sup>1</sup> | linearmodels `PanelOLS`, R `fixest` | R `fixest`, R `plm` | 1e-8 / 1e-8 |
| RE<sup>1</sup> | linearmodels `RandomEffects`, R `plm` | R `plm`, statsmodels `OLS` | 1e-8 / 1e-8 (unbalanced panels: 5e-3 to 5e-2) |

<sup>1</sup> For FE and RE, linearmodels covers only classical and HC1
standard errors. Its cluster and Driscoll–Kraay standard errors use
different small-sample corrections, so R (`fixest`, `plm`) is the primary
reference for those. Which implementation checks which statistic is
detailed on the verification page.

<sup>2</sup> There is no Python Tobit implementation, so both references are
R packages. They are different implementations (`survreg` and `maxLik`
based), so the second one is a genuine cross-check, but there is no
reference from outside R. Where no Python implementation exists, two
different R packages are used.

- **Tolerance.** The target relative tolerance against the primary
  reference is 1e-8 (with a tiny absolute floor for values near zero); for
  closed-form estimators the measured agreement is about 1e-14. Where
  implementations legitimately differ, looser documented values are used
  instead: the figures in parentheses and the Logit/Probit cross-check, as
  well as some solver options and real-data cases listed on the
  verification page. Typical reasons are HAC small-sample conventions, both
  sides being iterative optimizers, or plm and linearmodels estimating the
  Swamy–Arora variance components slightly differently on unbalanced
  panels. The full list of tolerances and the reasons are on the
  verification page.
- **Single-reference cases.** Some statistics have only one reference. For
  example, IV (2SLS) with HC2/HC3 standard errors is checked against `ivreg`
  only, because linearmodels has no equivalent. These cases are listed on the
  verification page.
- **Reproducible and continuous.** Reference values are generated once from
  pinned package versions and stored in the repository, and the comparisons
  run in CI on every pull request on Python 3.12, 3.13 and 3.14. The Rust
  engine also has its own unit and property-based tests.

For the statistics compared, the test data, tolerances, and the cases
without a second reference, see the
[verification page](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/verification/).
Test distributions and degrees of freedom for each method are in the
[inference conventions guide](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/inference-conventions/).

## Performance

The computational core is written in Rust. Speed and memory are measured for
each implemented method at representative settings (sample size, number of
regressors, and one or two standard-error types), not for every option
combination.

**Single-threaded.** The core currently runs its linear algebra on a single
thread, and the reference packages were pinned to one thread as well, so the
comparison measures the efficiency of the computation itself. Speed on many
cores, for example for very wide models, is not covered by these numbers.

Time for one `fit()` call at 1,000,000 observations, with 5 regressors and
classical standard errors:

| Method | `fit()` time |
|---|---|
| OLS | 0.07 s |
| WLS | 0.10 s |
| Logit | 0.70 s |
| Probit | 0.81 s |
| Tobit | 1.3 s |
| IV (2SLS) | 0.49 s |
| FE | 0.20 s |
| RE | 0.62 s |

Execution time and memory are compared against statsmodels and linearmodels
wherever a comparable Python implementation exists. Some examples at the
same sample size (reference time divided by ours):

- OLS: about 3.0x faster than statsmodels.
- IV (2SLS): about 30x faster than linearmodels.
- FE: about 11x faster than linearmodels.
- Logit: about 1.4x faster than statsmodels, and about 1.1x with clustered
  standard errors.
- Tobit has no Python implementation to compare against, so only its own
  time is shown.

Peak memory is lower in the largest cases measured too: IV (2SLS) peaks at
about 1.1 GB against about 4.0 GB for linearmodels, and OLS at about 0.32 GB
against about 0.57 GB for statsmodels (whole-process peak, including the
Python interpreter).

Times are end-to-end measurements from CI on shared runners, so absolute
values vary from run to run; the comparison between two packages within a
run is the meaningful part. See the
[performance page](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/performance/)
for the current numbers, measurement conditions, and known limitations, and
the
[full results](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/performance-results/)
per method.

## Limitations / Disclaimer

This is a personal, pre-1.0 project maintained by one person. It has not
yet undergone the years of community scrutiny and extensive edge-case
testing that established tools such as statsmodels and the R ecosystem
have. See
[Numerical verification](#numerical-verification) for how estimates are
checked. For published research or other important analyses, we recommend
cross-checking results against a trusted, established package.

Before 1.0, breaking changes may be introduced in minor version releases,
not just major ones.

## Documentation / Links

- [Documentation](https://masahiroyecon1997dev.github.io/econometricsmodels/) — usage guides and full API reference
- [PyPI](https://pypi.org/project/econometricsmodels/)
- [GitHub Issues](https://github.com/masahiroyecon1997dev/econometricsmodels/issues) — bug reports, feature requests, roadmap status
- [Changelog](https://github.com/masahiroyecon1997dev/econometricsmodels/blob/main/CHANGELOG.md)
- [License](https://github.com/masahiroyecon1997dev/econometricsmodels/blob/main/LICENSE) — MIT
