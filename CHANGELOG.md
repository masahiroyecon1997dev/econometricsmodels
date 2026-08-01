# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/) (during the `0.x.x` pre-release period, breaking changes may occur even in minor version bumps — see CLAUDE.md section 9).

## [Unreleased]

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

[Unreleased]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/releases/tag/v0.1.0
