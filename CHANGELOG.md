# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/) (during the `0.x.x` pre-release period, breaking changes may occur even in minor version bumps — see CLAUDE.md section 9).

## [Unreleased]

### Added

- OLS: `OLSResults.white_test(statistic="lm" | "f")`, the White test for heteroskedasticity, returning a `WhiteTestResult` (`statistic`, `p_value`, `df`, `df_denom`, `distribution`, the terms of the auxiliary regression in `aux_terms` and the dropped ones in `dropped_terms`, and `to_dict()`). The auxiliary regression always includes a constant, drops constant and duplicate terms (such as the square of a dummy) and counts the degrees of freedom from the rank. Verified against statsmodels `het_white` and R `lmtest::bptest`
- `DiagnosticResult` and `WhiteTestResult` (also exported at the top level), the shared result type of post-estimation diagnostic tests

### Changed

- **Breaking**: OLS / WLS / Logit / Probit / Tobit: `predict()` now returns a plain `list[float]` (one value per observation, like `residuals`) instead of a list of single-key dicts (`{"predicted": ...}` / `{"probability": ...}`). Intervals, if added later, will be a separate method rather than extra keys. Migration: replace `[row["predicted"] for row in res.predict()]` with `res.predict()`. `augment()` is unchanged

## [0.8.0] - 2026-10-04

Input-validation and API-consistency release. No new estimation method is added: accepted input types and option values are now checked strictly, names are unified across methods, and FE / RE inference follows fixest / plm conventions. Many changes are breaking (permitted during the `0.x.x` pre-release period).

### Added

- OLS / WLS / IV results expose `hac_lags_used`, the lag count actually used by `cov_type="hac"` (automatic or explicit); FE / RE results expose `dk_bandwidth_used` for `cov_type="dk"`. For IV GMM, the value is set when either `cov_type` or `gmm_weight_type` is `"hac"` (`cov_type` wins when both are and they differ)
- `FEResults.n_periods`
- polars columns of dtype Int8 / Int16 / UInt8 / UInt16 / Float16 / Int128 / Decimal / Struct / Array can now be passed in a `DataFrame` (before, such a column made the conversion fail even when unused, and Decimal / Int128 panicked); `to_dummies()` output (UInt8) is accepted. The release-build extension module grows from 34.6 MB to 43.5 MB
- Timezone-naive `Datetime` is accepted for identity keys (`entity`, `cluster`)
- Linux aarch64 and musllinux (x86_64 / aarch64) wheels
- Public docs pages: accepted data, validation, verification, and performance

### Changed

- **Breaking**: columns used as numbers (`y`, `x`, weights, instruments) must be integer, float, Boolean or Decimal. Before, every column was cast to Float64 unconditionally, so strings that look numeric were silently converted, unreadable strings were reported as missing values, and Date / Duration passed as days / microseconds. Other dtypes now raise `ValidationError` naming the dtype (`hac_time` additionally accepts Date / Datetime, since it only orders rows)
- **Breaking**: key columns are checked by dtype: `entity` / `cluster` accept integer, float, string, Categorical / Enum, Boolean and Date; `time` / `dk_time` accept integer, float, string, Categorical / Enum, Date and Datetime. Others (Time, Duration, Decimal, List, Struct, ...) raise `ValidationError`. NaN / infinity in a float key column is rejected (it used to form one `"NaN"` group). Timezone-aware `Datetime` key columns are rejected with a hint (`replace_time_zone(None)`)
- **Breaking**: column-name arguments (`y`, `x`, `entity`, `weight`, `x_exog`, `x_endog`, `instruments`) raise `TypeError` naming the argument, the expected form and the actual type when malformed; lists must be `list` (`str`, `tuple` and `polars.Series` are rejected). String options (`cluster`, `hac_time`, `time`, `dk_time`, `cov_type`, `solver`, `estimator`, `gmm_weight_type`, `gmm_type`) raise `TypeError` the same way, both in the constructor and on attribute assignment
- **Breaking**: numeric options (`confidence_level`, `hac_lags`, `max_iter`, `tol`, `lower`, `upper`, `gmm_max_iter`, `gmm_tol`, `dk_bandwidth`) raise `TypeError` for `bool`, strings, `None` where not allowed, and floats given to integer options (before, `hac_lags=True` was accepted as 1); integers beyond the i64 range raise `ValidationError` instead of `OverflowError`. Logit / Probit / Tobit `tol` of NaN / infinity is now `ValidationError`
- **Breaking**: `max_iter` and `gmm_max_iter` have an upper limit of 10,000 (`max_iter` 1–10,000, `gmm_max_iter` 3–10,000); larger values raise `ValidationError`
- **Breaking**: FE / RE: the Driscoll-Kraay time order is now the order of the column's values (integers, Date, Datetime and floats by value, Enum by category order), not the lexicographic order of the labels. Before, un-padded integer periods (`1, 10, 11, 2, ...`), negative numbers and decimals paired the wrong periods in the lag terms and changed the standard errors without an error. Two-way FE `fixed_effects()` now returns the time effects in time order with the first period as the baseline (`γ = 0`), so with 10 or more integer periods the baseline and the entity-effect constant change (predictions are unchanged)
- **Breaking**: FE / RE: the small-sample correction of cluster and Driscoll-Kraay standard errors now follows fixest / Stata (`G/(G-1) × (n-1)/(n-K)`) instead of linearmodels (no `G/(G-1)`), and the inference degrees of freedom follow `cov_type` (`G - 1` for cluster, `T - 1` for dk). Standard errors, test statistics and p-values change for these `cov_type`s
- **Breaking**: RE: `f_statistic` is now a Wald test that follows the fit's `cov_type`, like FE (denominator degrees of freedom `G - 1` for cluster, `T - 1` for dk, `df_resid` otherwise), verified against `plm::pwaldtest`
- **Breaking**: RE: the Hausman test's Wald test on the auxiliary regression now follows the RE fit's `cov_type` (classical / `hc1`–`hc3` / `cluster` / `dk`) instead of always using classical covariance. With the default `cov_type="cluster"` it is now the cluster-robust Hausman test; use `cov_type="classical"` for the previous values. Verified against `plm::phtest(method = "aux", vcov = ...)`. The auxiliary regression uses OLS-style (Stata/R) small-sample corrections, which differ from RE's linearmodels-style standard errors. When the auxiliary regression cannot be computed (e.g. `cov_type="cluster"` with no more clusters than the `2k` auxiliary slopes, or `"dk"` with too few periods), `fit()` now raises `ValidationError` / `ComputationError` instead of the Hausman fields being `None` (they are `None` only when there are no slope coefficients)
- **Breaking**: RE: the Hausman test (`hausman_statistic` / `hausman_p_value` / `hausman_df`) is now the regression-based (auxiliary regression) version, equivalent to `plm::phtest(method = "aux", effect = "individual")`, replacing the quadratic form whose sign was masked with `abs()`. The statistic is non-negative by construction, and the comparison is always against one-way FE: `REOptions.time` no longer switches it to two-way. Values change (identical to the classical Hausman test on balanced panels with a common σ²)
- **Breaking**: RE: `REOptions.time` is renamed to `REOptions.dk_time`, matching `FEOptions.dk_time`. RE has no two-way structure, so a `time` option had nothing to mean there; it was only the Driscoll-Kraay time ordering. It is used only with `cov_type="dk"` (where it is required), so specifying it with another `cov_type` raises `ValidationError` (like `cluster` / `dk_bandwidth`). A `time` option will be introduced again, with the same meaning as in FE, when two-way RE is implemented
- **Breaking**: Logit / Probit / Tobit: the confidence-interval keys of `marginal_effects()` rows are renamed from `conf_low` / `conf_high` to `conf_lower` / `conf_upper`, matching `coef_table()`
- **Breaking**: IV: renamed the result property `IVResults.n_iterations` to `n_iter` and the option `IVOptions.gmm_convergence` to `gmm_tol` (it is a convergence tolerance), matching Logit / Probit / Tobit's `n_iter` / `tol`
- **Breaking**: IV: renamed the GMM-only option `IVOptions.weight_type` to `gmm_weight_type` (and the result property `IVResults.weight_type` likewise), matching the `gmm_` prefix of the other GMM-only options
- **Breaking**: Tobit: `predict()`/`augment()` now take `new_data` as the first argument and `target` second (`predict(new_data=None, target="expected_observed")`), matching the other methods' `predict(new_data)`
- **Breaking**: OLS / WLS / IV: renamed `r_squared_adj` to `adj_r_squared` (result property, and the corresponding engine/pybind fields), matching the word order of `pseudo_r_squared`
- **Breaking**: Logit / Probit / Tobit: the estimate key of `marginal_effects()` renamed from `dydx` to `effect` (and `MarginalEffectsResult.dydx` likewise); it corresponds to `dy/dx` in Stata's `margins, dydx(*)` and statsmodels
- **Breaking**: IV: the GMM estimation type is now chosen by name instead of an iteration count. `IVOptions.gmm_iterations` is removed in favor of `gmm_type` (`"one_step"` / `"two_step"` (default) / `"iterated"`); `gmm_max_iter` (at least 3, counting the initial estimate; effective default 100) and `gmm_tol` (effective default 1e-6) apply only to `"iterated"` and raise `ValidationError` with the other types. `"one_step"` no longer accepts/validates `gmm_weight_type`. `IVResults` gains `gmm_type`
- **Breaking**: the `method` option/property no longer exists anywhere, since it meant two different things: IV's `IVOptions.method` / `IVResults.method` is now `estimator` (values `"2sls"` / `"gmm"` unchanged), and Logit / Probit / Tobit's `method` on the options and results is now `solver` (values `"newton"` / `"bfgs"` / `"lbfgs"` unchanged). Error messages change accordingly (`unknown estimator: ...`, `unknown solver: ...`)
- **Breaking**: the test-statistic name is unified across all methods. `t_stats` (OLS / WLS / FE / RE), `z_stats` (Logit / Probit / Tobit) and `stats` (IV) are now `test_stats`, and the `coef_table()` keys `t_stat` / `z_stat` / `stat` are now `test_stat`. `marginal_effects()` rows use `test_stat` too. No aliases are kept
- All Results classes gain `stat_dist` (`"t"` or `"normal"`) and `stat_df` (degrees of freedom of the t distribution, `None` for `"normal"`). `stat_df` is the value actually used, which can differ from `df_resid` (e.g. OLS / WLS / IV 2SLS with `cov_type="cluster"` use `G - 1`), so p-values can be recomputed from `test_stats`
- **Breaking**: FE / RE: `cov_type="hac"` is renamed to `"dk"` (Driscoll-Kraay). FE / RE's `"hac"` was a different estimator from OLS / WLS / IV's Newey-West `"hac"`, so the same string pointed at two estimators. The value matches fixest's `"DK"` (case-insensitive) and pairs with `dk_bandwidth`. The engine enum variants `FeCovType::Hac` / `ReCovType::Hac` become `Dk`
- **Breaking**: IV: `gmm_weight_type` values now use the same vocabulary as `cov_type`: `"unadjusted"` is now `"classical"` and `"kernel"` is now `"hac"` (`"robust"` and `"cluster"` are unchanged). The aliases `"homoskedastic"` / `"heteroskedastic"` are removed, and the default is `"classical"`. The engine `WeightType::Unadjusted` / `Kernel` variants become `Classical` / `Hac`
- **Breaking**: OLS / WLS / IV / Logit / Probit / Tobit: the `cov_type="nonrobust"` alias of `"classical"` is removed (it was accepted by these but not by FE / RE). Every concept now has exactly one string
- **Breaking**: Logit / Probit: the column appended by `augment()` is renamed from `"probability"` to `"predicted_probability"`, matching the `predicted_` prefix of OLS / WLS (`"predicted"`) and Tobit (`"predicted_<target>"`). The `"probability"` key returned by `predict()` is unchanged
- **Breaking**: IV: the overall-fit test is renamed `f_statistic` / `f_p_value` -> `wald_statistic` / `wald_p_value` (2SLS: F-type, GMM: chi-squared type; `wald_dist` is `"f"` or `"chi2"`), with `wald_df_num` / `wald_df_denom` (`None` for chi-squared). OLS / WLS / FE / RE keep `f_statistic`
- Test-statistic degrees of freedom are now exposed so p-values can be recomputed: `f_df_num` / `f_df_denom` (OLS / WLS / FE / RE), `lr_df` (Logit / Probit), `wald_dist` / `wald_df` (Tobit, always `"chi2"`), `overid_df`, `wu_hausman_df_num` / `wu_hausman_df_denom`, `weak_instrument_f_df_num` / `weak_instrument_f_df_denom` (IV). They are `None` when the statistic is `None` or NaN
- OLS / WLS Results gain `df_resid` and `df_model`
- Logit / Probit / Tobit Results gain `dep_var_name`, which OLS / WLS / IV / FE / RE already had
- **Breaking**: standard-error column options lose the `_col` suffix, matching `y` / `x` / `entity` / `time` / `weight`. `cluster_col` is now `cluster` (OLS / WLS / IV / FE / RE / Logit / Probit / Tobit). `time_col` (the time-ordering column for HAC) is now `hac_time` for OLS / WLS / IV and `dk_time` for FE (Driscoll-Kraay ordering, overriding `time`), named after the `cov_type` values `"hac"` / `"dk"`. `hac_lags` and `dk_bandwidth` are unchanged; no aliases are kept
- **Breaking**: an option that the chosen mode does not use now raises `ValidationError` instead of being silently ignored, so a forgotten `cov_type="cluster"` no longer returns classical standard errors. Covers `cluster` / `hac_lags` / `hac_time` (OLS / WLS / IV, and `cluster` for Logit / Probit / Tobit) with a `cov_type` that does not use them, `cluster` / `dk_time` / `dk_bandwidth` for FE and `cluster` / `dk_bandwidth` for RE, and IV's GMM-only options with `estimator="2sls"` or a `gmm_type` that does not use them. The error message says which setting to change. IV's `gmm_type`, `gmm_weight_type` and `raise_on_non_convergence` now default to `None` (effective defaults `"two_step"` / `"classical"` / `True`) so that specifying them can be detected. For IV, `cluster` / `hac_lags` / `hac_time` stay valid when either `cov_type` or `gmm_weight_type` uses them
- **Breaking**: OLS / WLS / IV: `hac_time` must now give every observation a distinct value. A column with any repeated value (including a constant column) raises `ValidationError` naming the first two rows that share a value, instead of silently falling back to the row order for the tied rows and returning what looked like a time-ordered HAC estimate. The column is now ordered in its own dtype rather than after a conversion to a float, so integers beyond 2^53 and nanosecond `Datetime` values that differ by less than a float can resolve keep their order instead of being treated as tied. A `Boolean` `hac_time` column is rejected by dtype. Results for columns that were already distinct are unchanged
- **Breaking**: FE: `cov_type="dk"` now requires `dk_time`, in one-way and two-way models alike. Before, an unset `dk_time` silently fell back to `time` (the two-way fixed-effects column), so the user never stated which column defines the Driscoll-Kraay periods. `time` now only sets the fixed-effects structure. The engine's `FeCovType::Dk.time` is a required `TimeKeys` instead of an `Option`. The error for a missing `dk_time` also now names `dk_time`, where it used to say the `time` option was required. In FE and RE alike, `dk_time` may not be the same column as `y` or `entity` (`ValidationError`); it may be the same as `x` (RE used to reject a `time` that was also in `x`) and, in FE, the same as `time`
- **Breaking**: OLS / WLS / IV: `hac_time` is now required for `cov_type="hac"` (and IV's `gmm_weight_type="hac"`); leaving it out raises `ValidationError`. Before, the row order of the data was silently taken as the time order, so data not sorted by time gave a wrong estimate that looked time-ordered. If the rows are already in time order, add an index column (for example `df.with_row_index("t")`) and pass its name. statsmodels and R's `sandwich` use the row order and take no time column, so this differs from them on purpose

### Fixed

- p-values are computed from the survival function instead of `1 - cdf` (t / normal, Wald F, 2SLS / GMM Wald / Sargan / J, Tobit Wald, Logit / Probit LR), so tail p-values no longer collapse to 0
- FE / RE: `cov_type="dk"` with a number of periods not larger than the number of jointly tested coefficients (FE slope F-test, RE Hausman test), and FE with two periods (dk) or two clusters (cluster) in which every level of an absorbed dimension has one observation per group, now raise `ValidationError`. Before, these returned a near-zero standard error and a huge F statistic, or `ComputationError`
- FE: the length of the `dk_time` override column is validated against the number of observations
- IV: GMM with `gmm_weight_type="cluster"` and fewer clusters than instruments is now `ValidationError` up front instead of `ComputationError`
- Passing a non-polars object (pandas `DataFrame`, `LazyFrame`, `Series`) now raises `ValidationError` with a clear message (with a `.collect()` hint for `LazyFrame`) instead of an `AttributeError` that leaked internals
- Logit / Probit / Tobit: large-sample Newton no longer stalls near the optimum because of rounding error in the log-likelihood sum (compensated summation; Probit n=1e6 from 10 iterations / 6.7 s to 5 iterations / 0.9 s), and BFGS / L-BFGS terminate when the line search stagnates near the optimum (Probit n=1e6 BFGS 43 s to 2.1 s, L-BFGS 47 s to 1.9 s)

### Performance

- FE / RE: entity and time identifiers are converted to integer codes once instead of grouping by string keys in every step. At n=1,000,000, k=5 (local A/B): RE about 3.7–4.4 s to 0.9–1.1 s, FE about 1.5–2.2 s to 0.25–0.30 s. RE's within R² and Hausman test also reuse the inner FE's within transformation. Results are bit-identical

## [0.7.0] - 2026-09-22

Added FE (Fixed Effects) and RE (Random Effects) to Phase 4 (panel data models).

### Added

- FE (Fixed Effects) estimation (`FE` / `FEOptions` / `FEResults`), estimated via within transformation delegated to the OLS estimator
- 1-way and 2-way within (demeaning) transformation, with singleton-group detection and validation of zero-variance regressors that arise after transformation
- Fixed effects themselves (α_i) can be recovered from the fit
- Standard error options: classical, HC1-3, cluster-robust, and Driscoll-Kraay panel HAC
- Panel-specific R² (within / between / overall), degrees-of-freedom adjustment (`df_resid` / `df_model`)
- FE API reference and usage examples in mkdocs
- RE (Random Effects) estimation (`RE` / `REOptions` / `REResults`), estimated by Swamy-Arora variance component estimation (σ_ε², σ_u²) followed by GLS via quasi-demeaning (θ transformation), delegated to the OLS estimator
- Standard error options: classical, HC1-3, cluster-robust
- Hausman test (fixed vs. random effects); the reported statistic is non-negative by construction (`hausman_statistic` uses `abs()`, matching `plm::phtest`'s sign convention)
- Panel-specific R² (within / between / overall), `f_statistic` / `f_p_value`, degrees-of-freedom adjustment
- RE API reference and usage examples in mkdocs
- `augment()` extended to Logit/Probit/Tobit (previously OLS/WLS only), and newly added to OLS/WLS itself — returns the source (or `new_data`) DataFrame with a predicted-value column appended
- `predict()` out-of-sample support (`new_data`) added to Logit/Probit/Tobit
- WLS: added `predict()`
- Logit/Probit/Tobit/IV results now expose the `method` actually used for estimation (IV additionally exposes `weight_type`, set only when `method="gmm"`)
- WLS: the weight column may now also be included in `x` (previously rejected; only `weight == y` remains disallowed)

### Changed

- WLS now has its own dedicated `WLSOptions` class instead of reusing `OLSOptions` (breaking change, permitted during the `0.x.x` pre-release period; #308)
- Renamed `IvOptions`/`IvResults` → `IVOptions`/`IVResults`, `FeOptions`/`FeResults` → `FEOptions`/`FEResults`, `ReOptions`/`ReResults` → `REOptions`/`REResults`, `OlsResults` → `OLSResults`, `WlsResults` → `WLSResults`, for naming consistency with the estimator classes (`IV`/`FE`/`RE`/`OLS`/`WLS`), which already used fully-uppercase acronyms (breaking change, permitted during the `0.x.x` pre-release period; #310)
- OLS's `predict()` return key renamed from `"fitted"` to `"predicted"`, for consistency between in-sample and out-of-sample (`new_data`) predictions (breaking change, permitted during the `0.x.x` pre-release period; #309)
- Logit/Probit/Tobit's BFGS/L-BFGS solvers reimplemented in-house (`FaerBfgs`/`FaerLbfgs`), replacing the `argmin` crate dependency
- faer's global parallelism pinned to single-threaded (`Par::Seq`), for reproducible results across runs

### Fixed

- IV: the `"const"` column-name collision check (previously applied only to `x_exog`) now also covers `x_endog`/`instruments`, preventing a true intercept coefficient from being silently overwritten by dictionary key collision
- IV: `x_endog`/`instruments` given as empty lists now raise `ValidationError`, instead of silently falling back to plain OLS
- OLS/WLS: `x=[]` (zero regressors) is now rejected with a `NoRegressors` error (except when used internally by FE's delegation to the OLS estimator)
- Logit/Probit/Tobit: BFGS/L-BFGS line search now has an evaluation-count budget, preventing an infinite loop in degenerate cases
- Tobit: fixed a Hessian weight computation bug that mixed the clamped λ with the raw z, producing incorrect standard errors in some regions
- Tobit: large-sample inputs could converge to a point where the Hessian becomes singular; convergence detection now includes a second-order condition guard

## [0.6.0] - 2026-09-06

Added Tobit (censored regression) to Phase 2 (generalized and discrete choice models).

### Added

- Tobit estimation (`Tobit` / `TobitOptions` / `TobitResults`), estimated by maximum likelihood
- Censoring bounds set via `TobitOptions.lower` (default 0.0) and `upper`; either may be `None`, giving left-, right-, or two-sided censoring
- Solver options: Newton-Raphson (default), BFGS, L-BFGS (`method`)
- Standard error options: classical (observed information), OPG (BHHH), HC0, HC1, cluster-robust
- The error scale `sigma` is reported as an estimated parameter alongside the regression coefficients (in `params`, `std_errors`, `coef_table()`, etc.)
- Goodness-of-fit statistics: log-likelihood, Wald test for overall significance (the Tobit analogue of OLS's F-test and Logit/Probit's likelihood-ratio test), AIC, BIC
- `predict()` with selectable target: `"expected_latent"` (`x'β`), `"expected_observed"` (censoring-adjusted `E[y|x]`), or `"prob_uncensored"`
- `marginal_effects()` (average, and at-mean / at-median), per target, with delta-method standard errors
- `censoring_fit_check()`: observed vs model-implied rates at each censoring boundary (Tobit's counterpart to Logit/Probit's `pred_table()`)
- Tobit API reference and usage examples in mkdocs

### Changed

- `cov_type="cluster"`: when the number of clusters does not exceed the number of parameters, all methods (OLS/WLS, Logit/Probit, IV) now raise `ValidationError` consistently (previously the behaviour differed across methods)

## [0.5.0] - 2026-08-15

Added IV (instrumental variables) to Phase 3 (2SLS/GMM).

### Added

- IV estimation (`IV` / `IvOptions` / `IvResults`), by two-stage least squares (2SLS) or generalized method of moments (GMM) (`method`)
- Independent variables split into `x_exog` (exogenous) and `x_endog` (endogenous), plus `instruments` (excluded instruments), consistent with this project's list-based API design
- Standard error options: classical, HC0-HC3, cluster-robust, HAC (Newey-West) — the same range as OLS/WLS
- GMM-specific options: `weight_type` (unadjusted/robust/cluster/kernel) for the point-estimation weight matrix, independent of `cov_type`; `gmm_iterations` (fixed iteration count) or `gmm_convergence` (convergence-based stopping)
- Diagnostics: `weak_instrument_f_statistics` (partial F-statistics per endogenous variable), `overid_statistic`/`overid_p_value` (Sargan for 2SLS, Hansen J for GMM), `wu_hausman_statistic`/`wu_hausman_p_value` (regression-based endogeneity test, 2SLS only)
- `first_stage()`: per-endogenous-variable first-stage `OlsResults`
- IV API reference and usage examples in mkdocs

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

[Unreleased]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/releases/tag/v0.1.0
