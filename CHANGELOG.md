# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/) (during the `0.x.x` pre-release period, breaking changes may occur even in minor version bumps — see CLAUDE.md section 9).

## [Unreleased]

### Added

- Every results class (`OLSResults`, `WLSResults`, `IVResults`, `LogitResults`, `ProbitResults`, `TobitResults`, `FEResults`, `REResults`) has a `cov_params` property: the covariance matrix of the coefficients under the `cov_type` used, as a dictionary of dictionaries keyed by coefficient name on both axes (`result.cov_params["x1"]["x2"]`, in `param_names` order). `std_errors` are the square roots of its diagonal. It makes joint tests, standard errors of linear combinations such as `b1 + b2`, and the delta method possible without a second fit. For Tobit it includes `"sigma"`. It is pickled and copied with the result
- Options and results can be pickled and copied (`pickle`, `copy.copy`, `copy.deepcopy`), so they work with `multiprocessing`, `concurrent.futures`, `joblib`, caches and Celery. It covers every `*Options` class, every result (`OLSResults`, `WLSResults`, `IVResults`, `LogitResults`, `ProbitResults`, `TobitResults`, `FEResults`, `REResults`) and the objects they return (`IVResults.first_stage()`, `marginal_effects()`, `censoring_fit_check()`, the diagnostic test results). A restored result gives the same values and the same answers from `predict()`, `augment()`, `marginal_effects()`, `pred_table()` and the diagnostic tests, because the pickle includes the `polars.DataFrame` passed to `fit()` (all its columns), so **a pickle is at least as large as that DataFrame** (about the same for Logit, Probit and Tobit, which rebuild their design matrix from it when restored; about 1.4 times for OLS, which also keeps the residuals and fitted values); take the values you need out of the result (`params_table()`, `params`) when only numbers have to cross a process boundary. `copy.copy` shares the DataFrame with the original and `copy.deepcopy` copies it. FE computes its fixed effects when pickled or copied. A pickle can only be read by the same version of econometricsmodels: restoring one written by another version, or an altered one, raises `ValidationError` with the new code `PICKLE_FORMAT_MISMATCH`. See the new guide "Pickling and copying"

### Changed

- **Breaking**: optional arguments are keyword-only everywhere, so adding or reordering an argument can no longer make a positional call silently set another one. (1) Every `*Options` class (`OLSOptions`, `WLSOptions`, `IVOptions`, `LogitOptions`, `ProbitOptions`, `TobitOptions`, `FEOptions`, `REOptions`) takes all its arguments by keyword: `OLSOptions(cov_type="hc1")`; `OLSOptions("hc1")` raises `TypeError`. (2) The `options` argument of `OLS`, `WLS`, `IV`, `Logit`, `Probit`, `Tobit`, `FE` and `RE` is keyword-only: `OLS(df, "y", ["x1"], options=opts)`. The columns (`data`, `y`, `x`, `weight`, `effects`, `x_exog`, `x_endog`, `instruments`) stay positional. (3) The optional arguments of the result methods are keyword-only: `predict()` / `augment()` (`new_data`, `target`), `pred_table()` (`threshold`), `marginal_effects()` (`at`, `target`, `confidence_level`), `white_test()` (`statistic`), `breusch_pagan_test()` (`variables`, `statistic`) and `breusch_godfrey_test()` (`statistic`; `time` and `nlags` stay positional). Migration: write the argument name, for example `result.predict(new_data=df)` and `result.breusch_pagan_test(variables=["x1"], statistic="f")`. A positional call fails with `TypeError` instead of changing meaning
- **Breaking**: `FE` and `RE` take the columns that define the effects as a required argument `effects` (`list[str]`, one or two columns), replacing the `entity` argument and `FEOptions.time` / `REOptions.time`. One column gives one-way effects; two columns give two-way effects on a balanced panel, with the entity first and the time period second (`FE(df, y="y", x=["x1"], effects=["id", "year"])`). The first column is the default clustering column for `cov_type="cluster"`. In `RE` the order of a two-way `effects` matters (`sigma2_u` and `theta_entity` belong to the first column, `sigma2_lambda` and `theta_time` to the second); in `FE` swapping the columns leaves the coefficients unchanged but changes the default cluster, the order of the keys of `fixed_effects()` and which column holds the reference period. `dk_time` is unchanged: it is still required with `cov_type="dk"` and is never taken from `effects`. Passing a `str` as `effects` raises `TypeError`, and `time=` is no longer an option. The results have `n_groups` (a `dict` from the column name to the number of groups, in the order of `effects`) instead of `n_entities` and `n_periods`, and `FEResults.fixed_effects()` always returns `{column: {level: effect}}`, also for one-way effects. A third column in `effects` raises the new `ValidationError` code `EFFECTS_TOO_MANY` (at most two columns are supported); an empty list raises `ROLE_EMPTY`. The params and message of `SINGLETON_GROUP` now name the column and the argument (`effects`). Migration: `FE(df, y="y", x=["x1"], entity="id")` becomes `FE(df, y="y", x=["x1"], effects=["id"])`, and `options=FEOptions(time="year")` with `entity="id"` becomes `effects=["id", "year"]`. A pickle of an FE or RE result or options object written before this change cannot be read.

## [0.9.0] - 2026-10-25

### Added

- OLS: `OLSResults.white_test(statistic="lm" | "f")`, the White test for heteroskedasticity, returning a `WhiteTestResult` (`statistic`, `p_value`, `df`, `df_denom`, `distribution`, the terms of the auxiliary regression in `aux_terms` and the dropped ones in `dropped_terms`, and `to_dict()`). The auxiliary regression always includes a constant, drops constant and duplicate terms (such as the square of a dummy) and counts the degrees of freedom from the rank. Verified against statsmodels `het_white` and R `lmtest::bptest`
- OLS: `OLSResults.breusch_pagan_test(variables=None, statistic="lm" | "f")`, the Breusch-Pagan test for heteroskedasticity (Koenker's studentized version, which does not assume normal errors), returning a `BreuschPaganTestResult` (`statistic`, `p_value`, `df`, `df_denom`, `distribution`, `aux_terms`, `dropped_terms`, `to_dict()`). `variables` lists the columns that may drive the error variance; it defaults to the model's regressors and may be any numeric columns, including ones that are not in the model. The auxiliary regression always includes a constant and drops constant and duplicate columns, counting the degrees of freedom from the rank. Verified against statsmodels `het_breuschpagan` and R `lmtest::bptest`
- OLS: `OLSResults.breusch_godfrey_test(time, nlags, statistic="lm" | "f")`, the Breusch-Godfrey test for serial correlation, returning a `BreuschGodfreyTestResult` (`statistic`, `p_value`, `df`, `df_denom`, `distribution`, `nlags`, `to_dict()`). `time` (the column giving the time order) and `nlags` are required; lags before the first observation are zero (as in R `bgtest` and statsmodels), and a model without an intercept gets no constant in the auxiliary regression (R and Greene; statsmodels differs there). Verified against statsmodels `acorr_breusch_godfrey` and R `lmtest::bgtest`
- `DiagnosticResult`, `WhiteTestResult`, `BreuschPaganTestResult` and `BreuschGodfreyTestResult` (also exported at the top level), the result types of post-estimation diagnostic tests

- RE: `REOptions.variance_method` selects the estimator of the variance components for one-way and two-way effects: `"swamy_arora"` (default, unchanged), `"wallace_hussain"` or `"amemiya"` (case-insensitive; R `plm`'s `random.method` `swar`, `walhus`, `amemiya`). Wallace-Hussain and Amemiya follow `plm`: closed forms on balanced panels and `plm`'s trace-term forms on unbalanced one-way panels (a one-way panel counts as balanced when every entity has the same number of observations; unbalanced two-way panels are still rejected). Amemiya raises `ValidationError` for a regressor without within variation (time-invariant; for two-way effects, varying in at most one dimension), which `plm` accepts on balanced panels. `REResults.variance_method` reports the estimator used. `variance_method` is the last positional parameter of `REOptions`, so the first six stay in the order of `FEOptions`.
- RE: `REResults` now exposes the estimated variance components and the weights of the quasi-demeaning: `sigma2_eps` (σ_ε²) and `sigma2_u` (σ_u², a negative estimate is raised to 0) for one-way and two-way effects, `sigma2_lambda` (σ_λ²) for two-way effects, `theta` (entity id to θ_i, one-way; θ_i differs across entities only on an unbalanced panel) and `theta_entity` / `theta_time` / `theta_total` (the three weights of the two-way GLS transformation; `theta_total` is not the product of the other two). The fields that do not apply are `None`. Verified against `linearmodels` `variance_decomposition` and `theta` (one-way, including unbalanced panels) and `plm::ercomp` (balanced panels, one-way and two-way)
- RE: two-way random effects (entity + time) on balanced panels, through `REOptions.time` (same meaning as `FEOptions.time`; `dk_time` stays separate and `cov_type="dk"` still requires it). The variance components σ_ε², σ_u² and σ_λ² follow the Swamy-Arora estimator of `plm` (`ercomp(effect = "twoways", method = "swar")`), the GLS transformation uses three weights `y - θ₁ȳᵢ. - θ₂ȳ.ₜ + θ₃ȳ..`, and the F statistic, `cov_type` and degrees of freedom follow one-way RE. A regressor that varies in only one dimension (between entities only, or over time only) is allowed. An unbalanced panel raises `ValidationError`. `REResults.n_periods` gives the number of periods (`None` for one-way). The Hausman test (the auxiliary-regression version, comparing against two-way fixed effects, with `cov_type` following the main model) and the within / between / overall R² are computed for two-way effects too: within is taken after removing both the entity and the time means, between on entity means and overall on the raw data, both with the intercept. `hausman_df` is the number of regressors that still vary after the two-way within transformation, and the statistics are `None` when there is none. Coefficients, standard errors, variance components and the Hausman statistic (for regressors that vary in both dimensions, for every `cov_type`) agree with `plm`; with regressors that vary in only one dimension the Hausman statistic is checked against a hand-computed auxiliary regression, because `plm::phtest` tests the wrong coefficients there, and the R² against a computation from the definitions, because `plm` has no counterpart

- Tobit: a fourth `target`, `"expected_uncensored"`, for `predict()`, `augment()` and `marginal_effects()`: the expected value among uncensored observations, `E[y|lower<y<upper,x]` (`E[y|y>0,x]` for a single lower bound of 0), for left-, right- and two-sided censoring. It completes the McDonald-Moffitt decomposition `E[y|x] = lower·P(lower) + upper·P(upper) + P(uncensored|x)·E[y|lower<y<upper,x]` with the existing targets. The marginal effect is `w·β_j` with `w` the variance of the truncated normal divided by `σ²`, with delta-method standard errors; it stays accurate where the uncensored probability is nearly 0 (observations far from the censoring bound), where the plain formula divides by zero. With two-sided censoring, an uncensored interval narrower than 1% of the estimated `sigma` raises `ComputationError` (the quantity cannot be computed reliably there; use another `target`) Verified against R (the weight and its Jacobian against `numDeriv`, and the mean against numerical integration, the plain `φ/W` formula and the decomposition above, for `AER::tobit` and `censReg` fits) and, for predictions and marginal effects, against `scipy.stats.truncnorm`
- Errors: when the design matrix is singular (`DESIGN_MATRIX_SINGULAR`), `params["columns"]` and the message now name the columns that take part in one linear dependency, for example `the columns 'x1', 'x1_copy' are perfectly collinear` (the intercept is `const`). If there are several dependencies, the first one is reported. It applies to OLS, WLS, Logit, Probit, Tobit, IV (first and second stage), FE and RE
- Errors: `ValidationError` and `ComputationError` are now structured. Every exception has a stable `code` (for example `COLUMN_NOT_FOUND`), `params` (the values that appear in the text, such as the column name, the counts and the limits), `message` (what is wrong) and `hint` (what to do next); `str(e)` is `message` and `hint` joined by a space. Applications can branch on `code` and show their own text, for example in another language. `ErrorCode` (a `StrEnum` of all codes, also exported at the top level) lists the codes; the exceptions can be constructed, copied and pickled. An error that happens inside a nested regression (the first stage of IV, the within regression used to estimate a variance component of RE, an auxiliary regression of a diagnostic test, and so on) keeps the cause as its `code` and adds a `stage` to `params` and a phrase at the start of the message. The new page "Error codes" in the guide lists every code with its parameters and the English templates of `message` and `hint`, and shows how to render the templates in your own language

### Changed

- The requirement on polars is relaxed from `polars==1.44.2` (v0.8.0) to `polars>=1.38.1`, with no upper limit, so installing the package no longer replaces the polars you already have. 1.38.1 is the oldest version for which the whole test suite passes (older versions either cannot be read at all or, from 1.28 to 1.31, read an `Enum` column as different values). The tests run against 1.38.1 and the version in the lockfile
- The source distribution and the wheels no longer contain the development notes (`CLAUDE.md`), `.gitkeep` files and the test-only `error_catalog_golden.txt`. The test-only functions `raise_error_for_test` and `panic_for_test` of the private module `_lib` exist only in debug builds, so they are no longer in the published wheels
- A panic in the Rust code no longer reaches Python as `pyo3_runtime.PanicException`, which derives from `BaseException` and is not caught by `except Exception`. It is raised as a `ComputationError` with the code `INTERNAL_ERROR`, and `detail` is `panic: ` followed by the panic message. It covers the `fit()` calls and the result methods (`predict()`, `augment()`, the diagnostic tests and so on). Normal calls are not slower
- **Breaking**: a perfect fit and a constant outcome are now reported by their own errors instead of by a misleading one or by a meaningless result. (1) `PERFECT_FIT` (`ComputationError`): when the residuals are zero up to rounding, that is the residual sum of squares is at most `k` times the machine epsilon times the total sum of squares (equivalent to `1 - R² ≤ k·ε`, which covers a residual standard deviation below about `1e-8` of the standard deviation of `y`), the standard errors and the F statistic are undefined, so OLS, WLS, FE, RE and IV (2SLS and GMM, the structural residuals, and each first stage) raise it before the covariance is computed. Before, such data gave an F statistic of order `1e35` for some `cov_type` and a `MATRIX_NEAR_SINGULAR` ("rescale the variables") for others, depending on rounding errors in the residuals. In RE it also replaces `DESIGN_MATRIX_SINGULAR` for noise-free data. Data in which `y` is an exact function of the regressors, such as an accounting identity, can no longer be estimated. (2) `OUTCOME_CONSTANT` (`ValidationError`): with an intercept, `y` has the same value in every row, up to rounding. Before, OLS, WLS and IV succeeded with `r_squared = -inf` and FE and RE raised `MATRIX_NEAR_SINGULAR`. (3) `OUTCOME_CONSTANT_WITHIN_ENTITY` (`ValidationError`): in FE and RE, `y` is constant within every entity (or, in a two-way FE, explained by the entity and time effects alone), so there is no within variation left. The three codes are in the error code table.
- **Breaking**: the text of every `ValidationError` and `ComputationError` message changed. Each message now states the problem with your values and the limit, in plain words without symbols such as `n`, `k` or `X'X`, and ends with a one-sentence `hint` that names a mechanical fix. The exception classes and the situations that raise them are unchanged, except for the items below. Code that matched on the message text (for example with `pytest.raises(match=...)` or `"..." in str(e)`) must switch to `e.code` and `e.params`
- **Breaking**: `data` and `new_data` that are not a `polars.DataFrame` (a pandas DataFrame, a `polars.LazyFrame`, a `polars.Series`, a dictionary, and so on) now raise `TypeError` instead of `ValidationError`, like every other argument of the wrong type. The message names the argument and the type it got, and for a pandas DataFrame and a `LazyFrame` it says how to convert it (`polars.from_pandas(df)`, `.collect()`; an `options` of the wrong class says `options must be an instance of OLSOptions, got WLSOptions.`; other polars types such as `Series` are shown by their class name). The text of the other `TypeError`s for the column arguments, `options` and the numeric and string options changed in the same way (`x must be a list of column names, got str. Pass a list, for example x=["x1"].`)
- Error messages: several errors that were a generic computation failure now have their own `code` and a hint that matches the cause (a singular design matrix, a near-singular covariance matrix, non-convergence, detected separation, an unusable weight, a degenerate diagnostic test). The `ComputationError` for a failing regression inside RE (the within regression that estimates σ_ε²) now says which step failed. The `ValidationError` for too few time periods with `cov_type="dk"` in RE, for the Hausman test and for the overall F test, names the number of periods needed. A count of an item that can be 1 is written with `(s)` (`1 coefficient(s)`)
- Error messages: a tied time-order column (`hac_time`, or `time` of `breusch_godfrey_test()`) no longer suggests omitting the option to use the row order, which is never used; it suggests adding an index column instead. A separated Logit/Probit fit no longer suggests a penalized estimator (the package has none); it suggests removing or combining the regressors, or adding observations. The `dk_bandwidth` error names the option and says what `t` is. Tobit's censoring-bound errors print `None` / `0.0` instead of Rust's `None` / `Some(0.0)`. The exception classes and the situations that raise them are unchanged
- Panel / clustered estimators: the `ValidationError` for a null in an `entity`, `cluster` or `time` column now reports how many values are missing and what to do about it ("column 'x' contains N missing value(s). Missing values are not handled automatically; please impute or remove them before calling this function"), the same message as for numeric columns. Before, it only said that the column contained missing values. The exception class and the situations that raise it are unchanged
- **Breaking**: `coef_table()` is renamed `params_table()` (all methods; there is no alias, the old name raises `AttributeError`), and the row keys of `params_table()` and `marginal_effects()` now follow one rule: each key is the singular of the matching result attribute (`param_name` / `param_names`, `param` / `params`, `std_error` / `std_errors`, `test_stat` / `test_stats`, `p_value` / `p_values`; `conf_lower` and `conf_upper` are the two ends of `conf_int`; the rows of `marginal_effects()` have `effect` in place of `param`). Renamed keys: `param` (the name) -> `param_name`, `coef` -> `param`, `std_err` -> `std_error`. **`row["param"]` still exists but now holds the value, not the name, so code that only read the name does not fail: it silently gets a float. Replace every `row["param"]` with `row["param_name"]` first, then `row["coef"]` with `row["param"]`, and `row["std_err"]` with `row["std_error"]`.** To migrate the method, replace `.coef_table()` with `.params_table()`. Neither the method name nor the value key says "coefficient" any more, which was inaccurate for the `sigma` row of Tobit.
- **Breaking**: `IVOptions` positional order is now `(estimator, cov_type, include_intercept, confidence_level, cluster, hac_lags, hac_time, gmm_type, gmm_weight_type, gmm_max_iter, gmm_tol, raise_on_non_convergence)`: `gmm_type` comes before `gmm_weight_type`, the order in which they are validated and the order of dependence (`gmm_type` decides whether `gmm_weight_type`, `gmm_max_iter` and `gmm_tol` are used). Passing these two by keyword is unaffected
- Errors are now reported in the same order for every method, and an input with several problems reports the first one in that order: option values (`estimator`, `cov_type`, `gmm_type`, `gmm_weight_type`, `solver`, `variance_method`), then option combinations (an option that is not used, a missing `hac_time` / `dk_time`), then the roles of the columns, then the `y` / `x` columns, then the columns named by options, then the estimation itself. See "Order in which errors are reported" in the validation guide. In particular, an unknown `gmm_type` or `gmm_weight_type` is now reported as an unknown value even when the option would not be used (for example with `estimator="2sls"`), instead of as an unused option
- **Breaking**: `REOptions` positional order is now `(cov_type, confidence_level, time, cluster, dk_time, dk_bandwidth)`, the same as `FEOptions`. In v0.8.0 it was `(cov_type, confidence_level, dk_time, cluster, dk_bandwidth)`; pass `dk_time` and `cluster` by keyword
- RE: one-way estimation now accepts a regressor that varies only over time (a year trend, for example) on a balanced panel. Its entity means are the same for every entity and collinear with the intercept, so the between regression that estimates σ_u² drops it and does not count it in the degrees of freedom, as `plm` does (before, the fit failed with a `ComputationError` about the between regression); the final regression keeps every column. The Hausman test leaves out the within-transformed copy of such a regressor, whose coefficient is not identified, so `hausman_df` is smaller by that many (and the statistics are `None` when no testable regressor is left). Results of inputs without such a regressor are unchanged. Coefficients, standard errors and variance components agree with `plm`
- FE / RE: the `ValidationError` message for an unbalanced panel in a two-way model now says "two-way estimation (fixed or random effects)" instead of "two-way fixed effects". For RE with `cov_type="dk"`, a missing `dk_time` now gives the same message as FE (it names the `dk_time` option)
- RE: the error message of a failing between regression (the regression of the entity means, or for two-way effects also of the time means, that estimates a variance component) now says which one failed: "between-regression on entity means ..." / "between-regression on time means ...".
- RE: a regressor together with its entity mean (and, for two-way effects, its time mean), as in a correlated-random-effects (Mundlak) specification, can now be estimated instead of failing with `ComputationError`. The between regression and the Hausman auxiliary regression now drop aliased columns by rank, like R `lm` (a column that is a linear combination of earlier columns, tolerance `1e-7`; the later column is dropped), in addition to the existing constant-column rules. The variance components match `plm`; `hausman_df` counts the identified `x̃` columns (`None` when none is left) and the number of clusters required by `cov_type="cluster"` follows the identified columns. The final regression and the internal within regression still keep every column, so a perfectly collinear design still raises `ComputationError` (now from the final regression instead of the between regression, for example a time-invariant regressor and a multiple of it). `plm::phtest(method = "aux")` selects the tested coefficients by position and gives column-order-dependent statistics on these inputs, so it is not a reference. `variance_method="amemiya"` rejects the time-invariant mean columns. The tolerance is relative to the uncentered norm of a column, so a column whose variation is below `1e-7` of its mean now counts as aliased with the constant in the between regressions (it is kept in the final regression), which can change `σ_u²` for such extreme inputs.
- RE: a time-invariant regressor (constant over time within every entity) is now allowed instead of being rejected with a `ValidationError`, as in R `plm`. The internal one-way FE regression that estimates σ_ε² uses only the regressors with within-variation (`k_tv` of them), so its degrees of freedom are `n - k_tv - n_entities`; the final regression, F statistic, `cov_type` and between / overall R² use every column, and within R² uses the time-varying coefficients only. The Hausman test compares only the time-varying regressors (`hausman_df` is `k_tv`, and the statistics are `None` when every regressor is time-invariant). A regressor that is constant across entities or collinear with another time-invariant regressor still fails with a `ComputationError`. Verified against R `plm` (balanced panels to machine precision) and, for the Hausman test, an auxiliary regression that selects the tested coefficients by name, because `plm::phtest(method = "aux")` tests the wrong coefficients when a regressor is time-invariant
- **Breaking**: OLS / WLS / Logit / Probit / Tobit: `predict()` now returns a plain `list[float]` (one value per observation, like `residuals`) instead of a list of single-key dicts (`{"predicted": ...}` / `{"probability": ...}`). Intervals, if added later, will be a separate method rather than extra keys. Migration: replace `[row["predicted"] for row in res.predict()]` with `res.predict()`. `augment()` is unchanged
- Logit / Probit: with an intercept, regressors are now centered as well as scaled before optimization (the result is mapped back to the original scale; without an intercept nothing changes). `solver="lbfgs"` needed about 95 iterations on Wooldridge `apple` and about 90 on `mroz`, so it failed at the default `max_iter=35`; it now converges well within the default. `newton` and `bfgs` are unchanged in practice, and estimates agree with the references as before
- Logit / Probit / Tobit: `solver="lbfgs"` now starts from the inverse Hessian at the starting point (when it is positive definite, otherwise as before), so it no longer needs many more iterations than `newton` and `bfgs` when the regressors are strongly correlated. For example, with 20 equicorrelated regressors (correlation 0.9) it needed 44 iterations and failed at the default `max_iter=35`; it now converges in 4. Estimates are unchanged within the convergence tolerance. For Tobit, `lbfgs` with `max_iter=1` and `raise_on_non_convergence=False` no longer raises a singular-Hessian `ComputationError` for the Hessian-based `cov_type`s on a heavily censored sample; like `newton` and `bfgs` it returns the unconverged result
- FE: for extreme scale differences between slope coefficients, the F-test failure is now reported by FE's own F-test (still `ComputationError`, with a different message) instead of by the internal OLS fit. With a robust `cov_type` an input that failed before can now succeed. Results of every other input are bit-for-bit unchanged
- Logit / Probit: separation is now detected exactly. After convergence, complete and quasi-complete separation are decided by a linear program (the largest margin of the separating direction is above `1e-6`) instead of the old check that the norm of the standardized parameters exceeds 100, which measured where the optimizer stopped rather than whether the data are separated. Before, a Logit with `n = k + 1` missed about 83% of separated samples (415 of 500 reported `converged=True` with standard errors in the thousands), quasi-complete separation by a dummy was missed even at `n = 1,000`, and strongly collinear regressors with a finite MLE were rejected by mistake (8 of 20 Probit samples). Data that used to return an unstable "converged" fit now raise `ComputationError` (`SEPARATION_DETECTED`) or, with `raise_on_non_convergence=False`, return `converged=False`; data that was wrongly rejected now fits. If the linear program cannot be completed numerically, the fit raises `ComputationError` (`SEPARATION_CHECK_FAILED`) or, with `raise_on_non_convergence=False`, returns `converged=False`. The check adds about 68 ms at `n = 1,000,000`, `k = 6` (about 11% of a Newton fit)
- IV: when `cov_type="cluster"` leaves too few clusters for the overall F test of a first stage (fewer than the number of its slope coefficients plus one), the fit no longer fails: only that first-stage F test is `None`, and the coefficients, standard errors and the weak-instrument F are returned. The structural equation still needs more clusters than coefficients and raises `ValidationError` otherwise
- OLS / WLS / FE / RE: `f_statistic`, `f_p_value` and `f_df` are `None`, not `NaN`, when the model has no slope coefficient to test
- IV: the number of observations is checked for the structural equation first and then for the first stage, and a first stage with too few observations has its own error. GMM with exactly as many observations as first-stage regressors (where 2SLS equals OLS) is now rejected like 2SLS instead of silently reducing to OLS

### Fixed

- A column of dtype `UInt128` that the model does not use no longer makes `fit()`, `predict()` or `augment()` fail (it used to end in a Rust panic). The data is now read column by column: only the columns you name are converted, as `Accepted data` already said, and `augment()` returns your data with the prediction column added, so unused columns are kept. A `UInt128` column that you do name (as `y`, `x`, a weight, a cluster, an entity, a time or an instrument) raises `ValidationError` with the code `COLUMN_UNSUPPORTED_DTYPE`
- The `__repr__` of every `Options` class prints Python literals (`True`, `"g"`, `0.0`, `None`) instead of Rust's (`true`, `Some("g")`, `Some(0.0)`), so `eval(repr(options))` rebuilds the same options
- The Wald F tests (the RE Hausman test and the joint tests of FE and IV) symmetrize the covariance submatrix before inverting it. With two almost collinear tested coefficients and `cov_type="dk"`, the RE Hausman statistic had a relative error of `1.2e-7`; it is now at the rounding level of the input
- IV: a structural equation with as many clusters as coefficients, or fewer, now reports the structural equation's own cluster count in a `ValidationError`, not the first stage's count wrapped in a first-stage error
- Clustered fits: a cluster-code length that differs from the number of observations raises `ValidationError` instead of a panic or a standard error computed from the first rows only

### Performance

- Cluster-robust covariance (all methods): one pass over the rows instead of one per cluster. The engine step at `n = 1,000,000` takes 0.003–0.02 s instead of 0.05–0.13 s, and the cluster column is turned into integer codes without a `String` per row (about 60 MB less peak memory). Results are bit-for-bit unchanged
- Logit / Probit / Tobit: `scores()` computes each per-observation quantity once instead of once per column (`k = 5`, `n = 1,000,000`: Logit 0.085 → 0.031 s, Probit 0.185 → 0.048 s, Tobit 0.157 → 0.045 s). Results are bit-for-bit unchanged
- Logit / Probit / Tobit: `bfgs` starts from the inverse of the exact Hessian at the starting point (when it is positive definite), which cuts the iterations (Logit `n = 100,000`: 13 → 6, Probit `n = 1,000,000`: 10 → 7, Tobit `n = 100,000`: 15 → 9)

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

[Unreleased]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/releases/tag/v0.1.0
