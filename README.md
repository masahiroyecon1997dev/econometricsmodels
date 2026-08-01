# econometricsmodels

[![CI (engine)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml)
[![CI (python)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml)
[![Docs](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/cd_docs.yml/badge.svg)](https://masahiroyecon1997dev.github.io/econometricsmodels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python API providing statistical and econometric analysis methods. It is primarily intended for use as the analysis engine for [economicon](https://github.com/masahiroyecon1997dev/economicon), a GUI application for data analysis, and its design prioritizes ease of embedding from scripts and programs (type completion, validation, dynamic construction).

- The computational core is implemented in **Rust** and thinly bound to Python via **PyO3**.
- Data input is restricted to **polars** DataFrames only, passed to the Rust side via **Arrow zero-copy**.
- Formula-string parsing (e.g. `y ~ x1 + x2`) is not used. The dependent variable is passed as a single column name (`str`), independent variables as a list of column names (`list[str]`), and estimation options as an instance of a dedicated class.

## Installation

```bash
pip install econometricsmodels
```

Requires Python 3.12 or later.

## Quickstart

```python
import polars as pl
from econometricsmodels import OLS

df = pl.DataFrame(
    {
        "y": [2.1, 3.9, 6.2, 8.1, 9.8],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)

result = OLS(df, y="y", x=["x1"]).fit()

print(result.params)       # {"const": ..., "x1": ...}
print(result.std_errors)   # {"const": ..., "x1": ...}
print(result.r_squared)
```

For more details — including how to switch to heteroskedasticity-robust standard errors (HC0-HC3), cluster-robust standard errors, and HAC (Newey-West) standard errors — see the [documentation site](https://masahiroyecon1997dev.github.io/econometricsmodels/getting-started/). The full documentation, including the API reference, is published at [https://masahiroyecon1997dev.github.io/econometricsmodels/](https://masahiroyecon1997dev.github.io/econometricsmodels/).

## Implementation status

Implemented: **OLS** (Ordinary Least Squares), **WLS** (Weighted Least Squares).

In progress: **Logit**.

Planned next, in this order: **Probit → Tobit → IV (2SLS/GMM) → FE (Fixed Effects) → RE (Random Effects) → GLS**.

During the `0.x.x` pre-release period, breaking changes may occur even in minor version bumps.

## Verification accuracy

Estimates for each implemented method (coefficients, standard errors, confidence intervals, R², adjusted R², AIC, BIC, log-likelihood, F-statistic, F-test p-value) are verified by numerical comparison against reference implementations. In addition to a primary reference, an independent implementation is used as a cross-check.

| `cov_type` | Primary reference | Independent cross-check |
|---|---|---|
| classical / HC0-3 / cluster | statsmodels (relative tolerance 1e-8) | R (`lm`/`lm(weights=)` + `sandwich`/`lmtest`, relative tolerance 1e-8) |
| HAC (Newey-West) | statsmodels (relative tolerance 1e-8) | R (relative tolerance 1e-2 for OLS, 5e-2 for WLS — looser due to differing small-sample correction conventions) |

## License

[MIT License](LICENSE)
