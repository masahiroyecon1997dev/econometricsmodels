# econometricsmodels

**econometricsmodels** is a Python package of econometric estimators built for embedding in scripts, pipelines, and apps. It rejects invalid input rather than silently dropping or repairing it, and returns result objects that are easy to work with directly and can also be exported as JSON-ready data. Every estimator is numerically checked against established reference implementations such as statsmodels, linearmodels, and R, and the comparisons are published.

## Key design principles

- **Consistent API.** Every model follows the same conventions: a polars DataFrame, column names as strings, and an options object in; a `.fit()` call out, returning a result with the same accessors. Methods with extra inputs (instruments for IV, an entity column for panel models) add arguments, but moving from OLS to Logit, IV, or panel models doesn't mean learning a new interface design.
- **Programmatic specification and results.** Models are defined with explicit arguments and an options object, not a formula string (`y ~ x1 + x2` is not supported). Results come back as Python objects (`.params`, `.std_errors`, `.p_values`, `.conf_int`), not formatted text, and parameter tables (`params_table()`) are plain lists of dicts that can be passed straight to polars or `json.dumps`.
- **Explicit validation.** Invalid input is rejected rather than silently dropped or repaired. Missing values, perfectly collinear columns, degenerate groups, and options that would have no effect raise an error that states the cause (see [Validation and errors](guide/validation.md)).
- **Modern dataframe workflow.** Built for [polars](https://pola.rs/), with data passed to the Rust core through Arrow, avoiding unnecessary conversions and copies. Other dataframe types are not accepted (see [Accepted data](guide/accepted-data.md)).
- **Verified against established packages.** Each estimator is compared against at least one reference implementation, and differences in conventions (such as small-sample corrections) are documented (see [Verification](guide/verification.md)).
- **Performance as a design goal.** The computational core is written in Rust, with no system BLAS/LAPACK dependency. Representative cases of each method are benchmarked against the packages it is verified against, and the results are published as measured, including cases where it is not faster (see [Performance](guide/performance.md)).

## Installation

```bash
pip install econometricsmodels
```

## Supported methods

Currently implemented: OLS (Ordinary Least Squares), WLS (Weighted Least Squares), Logit (binary logistic regression), Probit (binary probit regression), Tobit (censored regression), IV (instrumental variables, 2SLS/GMM), FE (fixed effects panel regression), and RE (random effects panel regression). See [Getting Started](getting-started.md) for usage, and the API Reference ([OLS](api/ols.md) / [WLS](api/wls.md) / [Logit](api/logit.md) / [Probit](api/probit.md) / [Tobit](api/tobit.md) / [IV](api/iv.md) / [FE](api/fe.md) / [RE](api/re.md)) for detailed options and return values. The [Methods](methods/ols.md) section defines the estimators, covariance types and diagnostics ([OLS](methods/ols.md), [WLS](methods/wls.md)).

More methods are planned as the project expands; see [GitHub Issues](https://github.com/masahiroyecon1997dev/econometricsmodels/issues) for what is planned or in progress.

## Guides

- [Verification](guide/verification.md): the reference implementations each method is checked against, and the tolerances used.
- [Validation and errors](guide/validation.md): why invalid input is rejected instead of repaired (for example, missing values are never dropped), and when each error is raised.
- [Inference conventions](guide/inference-conventions.md): which test distributions and degrees of freedom each method uses, and where they differ from R, statsmodels and linearmodels.
- [Performance](guide/performance.md): how execution time and memory compare with statsmodels and linearmodels, with [full results](guide/performance-results.md) per method.
