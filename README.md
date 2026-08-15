# econometricsmodels

[![CI (engine)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml)
[![CI (python)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml)
[![Docs](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/cd_docs.yml/badge.svg)](https://masahiroyecon1997dev.github.io/econometricsmodels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python API providing statistical and econometric analysis methods. It is primarily intended for use as the analysis engine for [economicon](https://github.com/masahiroyecon1997dev/economicon), a GUI application for data analysis, and its design prioritizes ease of embedding from scripts and programs (type completion, validation, dynamic construction).

- The computational core is implemented in **Rust** and thinly bound to Python via **PyO3**.
- Data input is restricted to **polars** DataFrames only, passed to the Rust side via **Arrow zero-copy**.
- Formula-string parsing (e.g. `y ~ x1 + x2`) is not used. The dependent variable is passed as a single column name (`str`), independent variables as a list of column names (`list[str]`), and estimation options as an instance of a dedicated class.

## Is this for you?

**Good fit:**

- Calling estimation methods from scripts, pipelines, or GUI apps — the `str`/`list[str]` + options-object API (see above) is built for programmatic construction, not interactive formula-writing.
- Exploratory or experimental analysis, where you value having validation (see [Verification accuracy](#verification-accuracy) below) but don't need the full breadth of diagnostics a mature package offers.
- Environments where a pure-Rust computational core (no system BLAS/LAPACK dependency) simplifies installation.
- Working with large datasets, where the **polars + Arrow zero-copy** handoff to the Rust core (see above) avoids the extra data copy that pandas/numpy-conversion-based wrappers typically incur.
- Projects that will lean on more than one econometric method over time — the roadmap covers a broad range (see [Implementation status](#implementation-status)), all under one consistent API.

**Probably not a good fit:**

- Published research or other work where correctness has to be beyond question — see the disclaimer below.
- Workflows built around R-style formula syntax (`y ~ x1 + x2`); this is a deliberate design choice (see above), not a missing feature, and isn't planned.
- Use cases that need the long tail of diagnostics, edge-case handling, and model types that statsmodels/R packages have accumulated over years of real-world use.
- Cases where you only ever need a single method, or where install footprint is tight — this package bundles everything into one Rust-compiled extension by design, so it's not the leanest choice if you don't need the breadth.

## Disclaimer

This is a solo-maintained, pre-1.0 project. Estimates are checked against statsmodels/R reference implementations (see [Verification accuracy](#verification-accuracy)), but it has not had the years of community scrutiny and edge-case hardening that established packages have. **If you're using this for an important decision — an academic paper or otherwise — we strongly recommend double-checking against a trusted, established package such as [statsmodels](https://www.statsmodels.org/) or an R equivalent.**

## Installation

```bash
pip install econometricsmodels
```

Requires Python 3.12 or later.

## Quickstart

### OLS

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

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)  # {"const": ..., "x1": ...}
print(result.r_squared)

# Row-oriented parameter table (param/coef/std_err/t_stat/p_value/conf_lower/conf_upper).
print(result.coef_table())

# Overall-fit statistics.
print(result.f_statistic, result.f_p_value)
print(result.aic, result.bic)
```

### Logit

```python
from econometricsmodels import Logit

df = pl.DataFrame(
    {
        "y": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        "x1": [1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.8, 4.0, 4.5, 5.0],
    }
)

result = Logit(df, y="y", x=["x1"]).fit()

print(result.coef_table())  # same shape as OLS, but z_stat instead of t_stat
print(result.aic, result.bic)

# Likelihood-ratio test for overall significance (the Logit/Probit analogue
# of OLS's F-statistic), plus McFadden pseudo R-squared.
print(result.lr_statistic, result.lr_p_value)
print(result.pseudo_r_squared)
```

`Probit` has the same API shape as `Logit` (drop-in replacement).

For more details — including how to switch to heteroskedasticity-robust standard errors (HC0-HC3), cluster-robust standard errors, HAC (Newey-West) standard errors, and Logit/Probit-specific features (marginal effects, classification tables) — see the [documentation site](https://masahiroyecon1997dev.github.io/econometricsmodels/getting-started/). The full documentation, including the API reference, is published at [https://masahiroyecon1997dev.github.io/econometricsmodels/](https://masahiroyecon1997dev.github.io/econometricsmodels/).

## Implementation status

Implemented: **OLS** (Ordinary Least Squares), **WLS** (Weighted Least Squares), **Logit**, **Probit**, **IV** (2SLS/GMM).

Planned next, in this order: **Tobit → FE (Fixed Effects) → RE (Random Effects) → GLS**.

During the `0.x.x` pre-release period, breaking changes may occur even in minor version bumps.

## Verification accuracy

Estimates for each implemented method (coefficients, standard errors, confidence intervals, AIC, BIC, log-likelihood, and the model's overall-significance test — F-statistic/p-value for OLS/WLS, likelihood-ratio statistic/p-value for Logit/Probit — plus R²/adjusted R² for OLS/WLS and McFadden pseudo R² for Logit/Probit) are verified by numerical comparison against reference implementations. In addition to a primary reference, an independent implementation is used as a cross-check.

| Method | `cov_type` | Primary reference | Independent cross-check |
|---|---|---|---|
| OLS / WLS | classical / HC0-3 / cluster | statsmodels (relative tolerance 1e-8) | R (`lm`/`lm(weights=)` + `sandwich`/`lmtest`, relative tolerance 1e-8) |
| OLS / WLS | HAC (Newey-West) | statsmodels (relative tolerance 1e-8) | R (relative tolerance 1e-2 for OLS, 5e-2 for WLS — looser due to differing small-sample correction conventions) |
| Logit / Probit | classical / OPG / HC0 / cluster | statsmodels (relative tolerance 1e-8) | R (`glm` + `sandwich`/`marginaleffects`, relative tolerance ~2e-4 — looser due to differing optimizer convergence) |
| Logit / Probit | HC1 | R (`glm` + `sandwich`, relative tolerance ~2e-4) — used as the primary reference here, since statsmodels' discrete-choice models omit the `n/(n-k)` small-sample correction for HC1 | — |
| IV (2SLS) | classical / HC0-HC1 / cluster | linearmodels (relative tolerance 1e-8) | R (`ivreg` + `sandwich`/`lmtest`, relative tolerance 1e-8) |
| IV (2SLS) | HAC (Newey-West) | linearmodels (relative tolerance 1e-8) | R (relative tolerance 1e-2, loosened to 1e-1 for a small-`n` scenario) |
| IV (2SLS / GMM) | HC2/HC3 | Verified against a manually-derived sandwich formula only (no cross-implementation reference: linearmodels has no HC2/HC3 for IV, and `ivreg` has no established leverage formula for it) | — |
| IV (GMM) | classical / HC0-HC1 / cluster / HAC | linearmodels `IVGMM` (relative tolerance 1e-8) | — (`ivreg` does not support GMM) |

## License

[MIT License](LICENSE)
