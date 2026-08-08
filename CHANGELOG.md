# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/) (during the `0.x.x` pre-release period, breaking changes may occur even in minor version bumps — see CLAUDE.md section 9).

## [Unreleased]

## [0.4.0] - 2026-08-08

Added Probit (binary probit regression) to Phase 2 (generalized and discrete choice models).

### Added

- Probit estimation (`Probit` / `ProbitOptions` / `ProbitResults`), estimated by maximum likelihood
- Solver options: Newton-Raphson (default), BFGS, L-BFGS (`method`)
- Standard error options: classical (observed information), OPG (BHHH), HC0/HC1, cluster-robust
- Goodness-of-fit statistics: log-likelihood, likelihood-ratio test, McFadden pseudo R², AIC, BIC
- `predict()` and `pred_table()` (classification table)
- `marginal_effects()` (average marginal effects, and at-mean / at-median), with delta-method standard errors
- Probit API reference and usage examples in mkdocs

### Changed

- `OlsResults`/`WlsResults`: renamed the `nobs` property to `n_obs`, for naming consistency with Logit/Probit/FE/RE/IV (breaking change, permitted during the `0.x.x` pre-release period)

### Fixed

- OLS: a NaN diagonal in the QR decomposition (produced when the design matrix is all-zero, e.g. `include_intercept=False` with all-zero explanatory columns) could evade `ensure_full_rank`'s singularity check, instead of raising a `SingularMatrix` error

## [0.3.0] - 2026-08-01

Added Logit (binary logistic regression) to Phase 2 (generalized and discrete choice models).

### Added

- Logit estimation (`Logit` / `LogitOptions` / `LogitResults`), estimated by maximum likelihood
- Solver options: Newton-Raphson (default), BFGS, L-BFGS (`method`)
- Standard error options: classical (observed information), OPG (BHHH), HC0/HC1, cluster-robust
- Goodness-of-fit statistics: log-likelihood, likelihood-ratio test, McFadden pseudo R², AIC, BIC
- `predict()` and `pred_table()` (classification table)
- `marginal_effects()` (average marginal effects, and at-mean / at-median), with delta-method standard errors
- Logit API reference and usage examples in mkdocs
- OLS: added `fitted_values` and `predict()` (in-sample and out-of-sample)

### Fixed

- OLS: the robust Wald F-test could become numerically unstable when explanatory variables had extreme differences in scale
- A gap in singular-matrix detection for non-pivoted Cholesky decompositions could miss near-singular covariance matrices (affects OLS and Logit)
- Logit: `y` values outside {0.0, 1.0} were silently accepted instead of raising an error
- Logit: a non-positive `tol` was silently accepted instead of raising an error
- Logit: a degenerate input (no intercept and no explanatory variables) caused an internal panic instead of a graceful error
- Logit: under (quasi-)complete separation, the solver could falsely report convergence due to floating-point underflow in the gradient norm

## [0.2.0] - 2026-07-25

Added WLS (Weighted Least Squares) to Phase 1 (basic regression).

### Added

- WLS estimation (`WLS` / `WlsResults`). The weight column is specified via the top-level `weight` argument, alongside `y`/`x` (an analytic weight; no normalization required)
- WLS supports the same standard error options as OLS (classical / HC0-HC3 / cluster / HAC)
- Added WLS API reference and usage examples to mkdocs

### Changed

- OLS's coefficient of determination, log-likelihood, AIC, BIC, F-statistic, and F-test p-value are now also cross-checked against an independent R implementation, in addition to the primary reference (statsmodels) (previously only coefficients and standard errors were cross-checked)

### Fixed

- Fixed a bug where WLS's coefficient of determination (R² / adjusted R²), log-likelihood, AIC, and BIC were systematically incorrect when weights were non-uniform
- Fixed a bug where cluster-robust standard error computation was non-deterministic across runs (internal group aggregation depended on `HashMap` iteration order) (affected both OLS and WLS)

## [0.1.0] - 2026-07-24

Initial release. Only OLS (Ordinary Least Squares) from Phase 1 (basic regression) is implemented.

### Added

- OLS estimation (classical / HC0-HC3 robust standard errors / cluster-robust standard errors / HAC (Newey-West) standard errors)
- Coefficient of determination (R² / adjusted R²), log-likelihood, AIC, BIC, Wald F-test
- Python API taking a polars DataFrame as input (`OLS` / `OLSOptions` / `OlsResults`)
- Rust computational core (`engine`) and PyO3 bindings (`engine_pybind`)

[Unreleased]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/releases/tag/v0.1.0
