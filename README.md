# econometricsmodels

[![CI (engine)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml)
[![CI (python)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml)
[![Docs](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/cd_docs.yml/badge.svg)](https://masahiroyecon1997dev.github.io/econometricsmodels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is econometricsmodels?

**econometricsmodels** is a Python package of econometric estimators built for embedding in scripts, pipelines, and apps. It rejects invalid input rather than silently dropping or repairing it, and returns result objects that are easy to work with directly and can also be exported as JSON-ready data. Every estimator is numerically checked against established reference implementations such as statsmodels, linearmodels, and R, and the comparisons are published.

## Key Design Principles

- **Consistent API** — Models follow a consistent interface: a DataFrame, a
  `y` column, a list of `x` columns, and an options object in; a `.fit()`
  call out. Moving from OLS to Logit, IV, or panel models doesn't mean
  learning a new interface.
- **Programmatic specification and results** — Models are defined with
  explicit arguments and an options object, not a formula string. Results
  come back as Python objects (`.params`, `.std_errors`, `.p_values`,
  `.conf_int`), not formatted text, so they are easy to build, inspect, and
  feed into other code. Coefficient tables (`coef_table()`) are plain lists
  of dicts that can be passed straight to polars or `json.dumps`.
- **Explicit validation** — Invalid input is rejected rather than silently
  dropped or repaired. Missing values, collinear columns, degenerate
  groups, and options that would have no effect raise an error instead of
  being quietly handled (see
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
  and memory with the packages it is verified against, and every
  implemented method is benchmarked (against those packages wherever a
  comparable implementation exists), with results published as measured,
  including cases where it is not faster (see
  [Performance](#performance)).

## Is this for you?

**Good fit:**

- Calling estimation methods from scripts, pipelines, or apps — the API is
  built for programmatic construction, not interactive formula-writing.
- Wanting invalid input to fail loudly instead of being silently dropped or
  repaired — missing values, collinear columns, and unusable options raise
  an error (see
  [Validation and errors](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/validation/)).
- Wanting estimates you can trace to established implementations — every
  estimator is compared against statsmodels, linearmodels, or R, with the
  comparisons published (see [Numerical verification](#numerical-verification)).
- Projects that use more than one method over time — the same interface
  covers linear, discrete-choice, IV, and panel models today, with more on
  the way (see [Implemented models](#implemented-models)).
- Already working in polars, where handing data to the Rust core through
  Arrow avoids unnecessary conversion overhead.

**Probably not a good fit:**

- You prefer R-style formula syntax (`y ~ x1 + x2`).
- You need the broader diagnostics and edge-case coverage of
  long-established tools like statsmodels and the R ecosystem.

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

**Collinear columns are never removed.** A perfectly collinear design raises
an error instead of being estimated on a reduced set of regressors.

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

Planned next are causal inference designs such as difference-in-differences
and regression discontinuity, followed by structural microeconometric
methods as the project expands. The order may change.

See the [documentation](https://masahiroyecon1997dev.github.io/econometricsmodels/)
for usage details on each method, and
[GitHub Issues](https://github.com/masahiroyecon1997dev/econometricsmodels/issues)
for the current status of planned and in-progress work.

## Numerical Verification

Every estimator is compared numerically against established reference
implementations before it is considered done. For each method, a primary
reference whose statistical conventions match is chosen, and where one
exists an independent implementation (mostly R) is used as a cross-check.
The comparison covers every reported statistic, not only coefficients and
standard errors: test statistics, p-values, confidence intervals, fit
statistics (R², log-likelihood, AIC/BIC), model tests, and marginal effects
where applicable, for each supported covariance type.

| Method | Primary reference | Independent cross-check | Relative tolerance (primary / cross-check) |
|---|---|---|---|
| OLS | statsmodels `OLS` | R `lm` + `sandwich`/`lmtest` | 1e-8 / 1e-8 (HAC: 1e-2) |
| WLS | statsmodels `WLS` | R `lm(weights=)` + `sandwich`/`lmtest` | 1e-8 / 1e-8 (HAC: 5e-2) |
| Logit | statsmodels `Logit` | R `glm` + `sandwich`, `marginaleffects` | 1e-8 / 2e-4 |
| Probit | statsmodels `Probit` | R `glm` + `sandwich`, `marginaleffects` | 1e-8 / 2e-4 |
| Tobit | R `AER::tobit` | R `censReg` | 1e-8 / 1e-8 |
| IV (2SLS) | linearmodels `IV2SLS` | R `ivreg` + `sandwich`/`lmtest` | 1e-8 / 1e-8 (HAC: 1e-2) |
| IV (GMM) | linearmodels `IVGMM` | none (`ivreg` has no GMM) | 1e-8 / none |
| FE | linearmodels `PanelOLS`, R `fixest` | R `fixest`, R `plm` | 1e-8 / 1e-8 |
| RE | linearmodels `RandomEffects`, R `plm` | R `plm`, statsmodels `OLS` | 1e-8 / 1e-8 (unbalanced panels: 5e-3 to 5e-2) |

- **Primary tolerance.** The relative tolerance is 1e-8 for every method,
  with a tiny absolute floor for values near zero. For closed-form
  estimators (OLS, WLS, IV, FE, RE) the measured agreement is about 1e-14.
- **Looser tolerances have a reason.** Values in parentheses and the
  Logit/Probit cross-check are looser because the reference packages use
  different small-sample or HAC conventions, or because both sides are
  iterative optimizers. Each is documented and applied to that statistic
  only; tolerances are never loosened across the board.
- **Discrepancies are investigated, not accepted.** Where implementations
  use different statistical conventions, the difference is documented and
  the comparison is adjusted for that statistic only.
- **Single-reference cases are stated.** Where only one reference exists
  (for example Tobit has no non-R reference), the verification page says so.
- **Reproducible and continuous.** Reference values are generated once from
  pinned package versions and stored in the repository, and the comparisons
  run in CI on every pull request on Python 3.12, 3.13 and 3.14. The Rust
  engine also has its own unit and property-based tests.

See the [verification page](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/verification/)
for the full tolerances, the statistics compared, the test data, and the
cases without a second reference, and the
[inference conventions guide](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/inference-conventions/)
for method-by-method details on test distributions and degrees of freedom.

## Performance

The computational core is written in Rust, with the goal of keeping
extensive validation from imposing an unacceptable performance cost.
Speed remains an important design goal, especially for scripted workflows
and larger datasets.

Execution time and memory are compared against statsmodels and linearmodels
wherever a comparable Python implementation exists, and the results are
tabulated per method. See the
[performance page](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/performance/)
for a summary and known limitations, and the
[full results](https://masahiroyecon1997dev.github.io/econometricsmodels/guide/performance-results/)
per method.

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
