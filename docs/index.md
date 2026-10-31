# econometricsmodels

**econometricsmodels** is a Python package of econometric estimators built for embedding in scripts, pipelines, and apps. It rejects invalid input rather than silently dropping or repairing it, and returns result objects that are easy to work with directly and can also be exported as JSON-ready data. Every estimator is numerically checked against established reference implementations such as statsmodels, linearmodels, and R, and the comparisons are published.

## Key design principles

- **Consistent API.** Every model follows the same conventions: a polars DataFrame, column names as strings, and an options object in; a `.fit()` call out, returning a result with the same accessors. Methods with extra inputs (instruments for IV, an entity column for panel models) add arguments, but moving from OLS to Logit, IV, or panel models doesn't mean learning a new interface design.
- **Programmatic specification and results.** Models are defined with explicit arguments and an options object, not a formula string (`y ~ x1 + x2` is not supported). Results come back as Python objects (`.params`, `.std_errors`, `.p_values`, `.conf_int`), not formatted text, and parameter tables (`params_table()`) are plain lists of dicts that can be passed straight to polars or `json.dumps`.
- **Explicit validation.** Invalid input is rejected rather than silently dropped or repaired. Missing values, perfectly collinear columns, degenerate groups, and options that would have no effect raise an error that states the cause (see [Validation and errors](guide/validation.md)).
- **Modern dataframe workflow.** Built for [polars](https://pola.rs/), with data passed to the Rust core through Arrow, avoiding unnecessary conversions and copies. Other dataframe types are not accepted (see [Accepted data](guide/accepted-data.md)).
- **Verified against established packages.** Each estimator is compared against at least one reference implementation, and differences in conventions (such as small-sample corrections) are documented (see [Verification](guide/verification.md)).
- **Performance as a design goal.** The computational core is written in Rust, with no system BLAS/LAPACK dependency. Representative cases of each method are timed, against the packages it is verified against where there is a comparable one (Tobit has none), and the results are published as measured, including cases where it is not faster (see [Performance](guide/performance.md)).

## Installation

```bash
pip install econometricsmodels
```

It needs Python 3.12 or later. The data library [polars](https://pola.rs/) (1.38.1 or later) is installed with it.

## A first example

```python
import polars as pl
from econometricsmodels import OLS, OLSOptions

df = pl.DataFrame(
    {
        "y": [2.1, 3.9, 6.2, 8.1, 9.8],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)

result = OLS(df, y="y", x=["x1"], options=OLSOptions(cov_type="hc1")).fit()

print(result.params)  # {"const": ..., "x1": ...}
print(result.params_table())  # a list of dicts, ready for polars or json.dumps
```

[Getting Started](getting-started.md) walks through this example, how to read the results and how errors are reported. Invalid input raises an exception that carries a stable `code` and the values behind the message, so a program can react to it without parsing text (see [Validation and errors](guide/validation.md#reading-an-error)).

## Supported methods

| Method | Use it when | Usage | Options and results | Definitions |
|---|---|---|---|---|
| OLS | The outcome is continuous and the regressors can be treated as exogenous | [Usage](usage/ols.md) | [Options and results](api/ols.md) | [Definitions](methods/ols.md) |
| WLS | As OLS, with known weights for the observations (analytic weights, inverse error variance) | [Usage](usage/wls.md) | [Options and results](api/wls.md) | [Definitions](methods/wls.md) |
| Logit | The outcome is binary (0/1), with a logistic link | [Usage](usage/logit.md) | [Options and results](api/logit.md) | [Definitions](methods/logit.md) |
| Probit | The outcome is binary (0/1), with a normal link | [Usage](usage/probit.md) | [Options and results](api/probit.md) | [Definitions](methods/probit.md) |
| Tobit | The outcome is censored at a lower bound, an upper bound or both (for example, zero spending) | [Usage](usage/tobit.md) | [Options and results](api/tobit.md) | [Definitions](methods/tobit.md) |
| IV | A regressor is endogenous and you have excluded instruments (2SLS or GMM) | [Usage](usage/iv.md) | [Options and results](api/iv.md) | [Definitions](methods/iv.md) |
| FE | Panel data, with entity (and optionally time) effects that may be correlated with the regressors | [Usage](usage/fe.md) | [Options and results](api/fe.md) | [Definitions](methods/fe.md) |
| RE | Panel data, with entity (and optionally time) effects assumed uncorrelated with the regressors; time-invariant regressors are allowed | [Usage](usage/re.md) | [Options and results](api/re.md) | [Definitions](methods/re.md) |

Each method has three pages: **Usage** (examples), **Options and results** (the arguments, options and result fields) and **Definitions** (the estimator, the covariance types, the test distributions, the fit statistics and the diagnostics). [Getting Started](getting-started.md) introduces the conventions all of them share.

More methods are planned as the project expands; see [GitHub Issues](https://github.com/masahiroyecon1997dev/econometricsmodels/issues) for what is planned or in progress.

## Guides

- [Arguments and options](guide/api-conventions.md): which inputs are arguments and which are options, how arguments are named, and the arguments of every model.
- [Accepted data](guide/accepted-data.md): the input type and the column types each argument accepts.
- [Validation and errors](guide/validation.md): how to read and handle an error, why invalid input is rejected instead of repaired (for example, missing values are never dropped), and when each error is raised. [Error codes](guide/error-codes.md) lists every `code`.
- [Inference conventions](guide/inference-conventions.md): which test distributions and degrees of freedom each method uses, and where they differ from R, statsmodels and linearmodels.
- [Pickling and copying](guide/pickling.md): sending options and results through `multiprocessing`, `joblib` and caches, what a pickle contains and costs, and the version it needs.

## Verification and performance

- [Verification](guide/verification.md): the reference implementations each method is checked against, and the tolerances used.
- [Performance](guide/performance.md): how execution time and memory compare with statsmodels and linearmodels, with [full results](guide/performance-results.md) per method.
