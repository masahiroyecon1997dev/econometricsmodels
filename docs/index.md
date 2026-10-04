# econometricsmodels

A Python API providing statistical and econometric analysis methods, designed for ease of embedding from scripts and programs (type completion, validation, dynamic construction).

- The computational core is implemented in **Rust** and thinly bound to Python via **PyO3**.
- Data input is restricted to **polars** DataFrames only, passed to the Rust side via **Arrow zero-copy**.
- Formula-string parsing (e.g. `y ~ x1 + x2`) is not used. The dependent variable is passed as a single column name (`str`), independent variables as a list of column names (`list[str]`), and estimation options as an instance of a dedicated class.

## Installation

```bash
pip install econometricsmodels
```

## Supported methods

Currently implemented: OLS (Ordinary Least Squares), WLS (Weighted Least Squares), Logit (binary logistic regression), Probit (binary probit regression), Tobit (censored regression), IV (instrumental variables, 2SLS/GMM), FE (fixed effects panel regression), and RE (random effects panel regression). See [Getting Started](getting-started.md) for usage, and the API Reference ([OLS](api/ols.md) / [WLS](api/wls.md) / [Logit](api/logit.md) / [Probit](api/probit.md) / [Tobit](api/tobit.md) / [IV](api/iv.md) / [FE](api/fe.md) / [RE](api/re.md)) for detailed options and return values.

More methods are planned as the project expands; see [GitHub Issues](https://github.com/masahiroyecon1997dev/econometricsmodels/issues) for what is planned or in progress.

## Guides

- [Verification](guide/verification.md): the reference implementations each method is checked against, and the tolerances used.
- [Validation and errors](guide/validation.md): why invalid input is rejected instead of repaired (for example, missing values are never dropped), and when each error is raised.
- [Inference conventions](guide/inference-conventions.md): which test distributions and degrees of freedom each method uses, and where they differ from R, statsmodels and linearmodels.
- [Performance](guide/performance.md): how execution time and memory compare with statsmodels and linearmodels, with [full results](guide/performance-results.md) per method.
