# Verification

Every estimator is compared numerically against established reference implementations before it is considered done. This page lists which implementations are used for each method, how closely the results must agree, and what is deliberately *not* compared against a reference.

## How verification works

- **Primary reference.** For each statistic a primary reference implementation is chosen, the one whose conventions match the ones used here. Results are compared at a relative tolerance of `1e-8` unless a documented reason requires a looser value.
- **Independent cross-check.** Where available, a second, independent implementation (mostly R) is compared as well. It covers the main reported statistics, not only coefficients and standard errors, so a bug in the primary reference or a shared misunderstanding is not silently inherited. Statistics without an independent reference are listed under [Not compared against a second reference](verification-details.md#not-compared-against-a-second-reference).
- **Frozen reference values.** Reference values are generated once from the pinned reference packages and stored as JSON files in the repository. The test suite compares against these files, so R and the reference packages are not needed to run the tests, and a new release of a reference package cannot change a result unnoticed.
- **Convention differences are documented.** When two implementations use different statistical conventions, such as small-sample corrections, the difference is documented (see [Inference conventions](inference-conventions.md)) and the comparison is adjusted for that statistic only.
- **Continuous integration.** The comparisons run on every pull request in the *CI (python)* workflow on Python 3.12, 3.13 and 3.14, and the Rust engine's own unit and property-based tests run in *CI (engine)*.

## Reference implementations

| Method | Primary reference | Independent cross-check |
|---|---|---|
| OLS | statsmodels `OLS` | R `lm` + `sandwich` / `lmtest` |
| OLS White test (`white_test()`) | statsmodels `het_white` | R `lmtest::bptest` (Koenker's studentized version) and `lm` for the F version |
| OLS Breusch-Pagan test (`breusch_pagan_test()`) | statsmodels `het_breuschpagan(robust=True)` | R `lmtest::bptest(studentize = TRUE)` and `lm` for the F version |
| OLS Breusch-Godfrey test (`breusch_godfrey_test()`) | statsmodels `acorr_breusch_godfrey` (models with an intercept) | R `lmtest::bgtest(fill = 0)`, the only reference for models without an intercept |
| WLS | statsmodels `WLS` | R `lm(weights=)` + `sandwich` / `lmtest` |
| Logit | statsmodels `Logit` | R `glm` + `sandwich`, `marginaleffects` |
| Probit | statsmodels `Probit` | R `glm` + `sandwich`, `marginaleffects` (observed-information covariance, see below) |
| Tobit | R `AER::tobit` | R `censReg` |
| IV, 2SLS | linearmodels `IV2SLS` | R `ivreg` + `sandwich` / `lmtest` |
| IV, GMM | linearmodels `IVGMM` | R `momentfit` (`ivreg` has no GMM) |
| FE | linearmodels `PanelOLS` (classical, HC1); R `fixest` (cluster, Driscoll-Kraay, and the F statistic for every `cov_type`) | R `fixest`; R `plm` + `sandwich` (one-way cluster and Driscoll-Kraay only, see the section on cluster and Driscoll-Kraay standard errors below) |
| RE | linearmodels `RandomEffects` (classical, HC1); R `plm` (every `cov_type`, and the only reference for HC2, HC3, cluster, Driscoll-Kraay) | R `plm`; statsmodels OLS on plm's quasi-demeaned data (cluster inference: standard errors, degrees of freedom, F statistic) |
| RE with a time-invariant regressor | R `plm` (balanced panels, and unbalanced panels through plm's variance components); the Hausman test against plm's auxiliary regression rebuilt with the tested coefficients selected by name | A hand-written implementation of the specified procedure (statsmodels `OLS` covariances, numpy Driscoll-Kraay); see below |
| RE, two-way (`REOptions.time`, balanced panels) | R `plm` (`model = "random"`, `effect = "twoways"`; the only reference, because linearmodels `RandomEffects` is one-way only) | statsmodels `OLS` on plm's quasi-demeaned data (log-likelihood, AIC, BIC); a hand-written implementation of the specified procedure (statsmodels `OLS` covariances, numpy Driscoll-Kraay); see below |
| RE, `variance_method` (Wallace-Hussain and Amemiya; one-way and two-way), and correlated random effects (a regressor together with its entity or time mean) | R `plm` (`random.method = "walhus"` and `"amemiya"`, and `"swar"` for the balanced correlated-random-effects cases; the only reference, because linearmodels `RandomEffects` has Swamy-Arora only and cannot estimate a regressor together with its entity mean) | A hand-written implementation of plm's formulas with observation-level matrices (the closed forms on balanced panels and the trace-term forms on unbalanced one-way panels), plus statsmodels `OLS` on plm's quasi-demeaned data (log-likelihood, AIC, BIC and cluster inference: standard errors, t statistics, p-values, confidence intervals, F statistic, degrees of freedom `G-1`); see below |

Versions used to generate the reference values: statsmodels 0.15.0, linearmodels 7.1, R 4.5.3 with sandwich 3.1.3, lmtest 0.9.40, fixest 0.14.2, plm 2.6.7, ivreg 0.6.8, momentfit 1.0, AER 1.2-17, censReg 0.5-40 and marginaleffects 1.0.0. Statements on this site about the behavior of these packages describe these versions.

pyfixest is not used as a reference. In a comparison with pyfixest 0.60.0, its HC2 and HC3 standard errors differed systematically from those of fixest, in the way expected if the HC1 factor `n/(n-k)` were applied to them as well. Later versions were not checked.

## What is compared

| Method | Statistics |
|---|---|
| OLS, WLS | coefficients, standard errors, the covariance matrix of the coefficients (`cov_params`), t statistics, p-values, confidence intervals, R², adjusted R², F statistic and p-value, log-likelihood, AIC, BIC, predictions |
| OLS White test | LM and F statistics with their p-values, and the degrees of freedom of the auxiliary regression (cross-check) |
| OLS Breusch-Pagan test | LM and F statistics with their p-values, and the degrees of freedom of the auxiliary regression (cross-check) |
| OLS Breusch-Godfrey test | LM and F statistics with their p-values for several lag orders, and the degrees of freedom (cross-check) |
| Logit, Probit | coefficients, standard errors, the covariance matrix of the coefficients (`cov_params`), z statistics, p-values, confidence intervals (also at a non-default confidence level, for the coefficients and the marginal effects), log-likelihood, null log-likelihood, likelihood-ratio statistic and p-value, McFadden R², AIC, BIC, marginal effects with standard errors, z statistics, p-values and confidence intervals, the classification table, predicted probabilities (in sample and for new data) |
| Tobit | coefficients and σ with their standard errors, the covariance matrix of the coefficients and σ (`cov_params`), z statistics, p-values and confidence intervals, log-likelihood, AIC, BIC, Wald statistic and p-value, marginal effects, predictions |
| IV | coefficients, standard errors, the covariance matrix of the coefficients (`cov_params`), test statistics, p-values, confidence intervals (also at a non-default confidence level, with cluster standard errors on the `G - 1` degrees of freedom), R², adjusted R², robust Wald statistic, weak-instrument F, Wu-Hausman (2SLS), Sargan (2SLS) or Hansen J (GMM); `first_stage()` against a direct OLS fit (see below) |
| FE, RE | coefficients, standard errors, the covariance matrix of the coefficients (`cov_params`), t statistics, p-values, confidence intervals, F statistic, within / between / overall R², Hausman test (RE), variance components and quasi-demeaning weights (RE) |
| RE with a time-invariant regressor | the same statistics, plus the degrees of freedom of the tests, log-likelihood, AIC and BIC (against the hand-written implementation) |
| RE, two-way | the same statistics for every `cov_type`, plus the three variance components (σ_ε², σ_u², σ_λ²) and the three quasi-demeaning weights, the degrees of freedom of the tests, log-likelihood, AIC and BIC; the Hausman test and the log-likelihood / information criteria / R² are hand-written from plm's output (see below) |
| RE, `variance_method` and correlated random effects | variance components (and, for two-way effects, the three weights), coefficients, standard errors, t statistics, p-values, confidence intervals, F statistic and Hausman test against plm, for every `cov_type` on the representative scenarios (the others: `classical` and `cluster`); every statistic for every `cov_type` on every scenario against the hand-written implementation; log-likelihood, AIC, BIC and cluster inference against statsmodels on every scenario (correlated random effects with Swamy-Arora on an unbalanced panel has only the hand-written reference, because plm estimates `σ_u²` differently there) |

Standard-error types are covered per method: classical, HC0–HC3, HAC, cluster, Driscoll-Kraay (panel) and OPG (maximum likelihood), wherever the method supports them. Each reference package covers a subset, so some combinations have only one reference (listed under [Not compared against a second reference](verification-details.md#not-compared-against-a-second-reference)).

## Tolerances

Agreement is checked as `|ours - reference| <= max(rtol · |reference|, atol)`.

The covariance matrix of the coefficients is compared entry by entry as `|ours - reference| <= rtol · sqrt(V_ii · V_jj)`, with no absolute floor. `V_ii` and `V_jj` are the reference variances of the two coefficients. An off-diagonal covariance can be close to zero, so a relative error of the entry itself would not be meaningful; scaling by both variances keeps every entry at the same precision as the standard errors. The matrix is compared at the tolerance of the other statistics of the same method, given in the tables below. Measured agreement on this scaled measure, against the primary reference and the cross-check:

- OLS, WLS: up to about `1e-11`.
- Logit, Probit: up to about `5e-10` (Logit) and `7e-9` (Probit), against both statsmodels and R. The `bfgs` / `lbfgs` solvers are compared with statsmodels' own `bfgs` / `lbfgs` and differ by up to about `2e-5`.
- Tobit: the matrix covers σ as well (the reference `(β, log σ)` matrix is mapped to `(β, σ)` with the Jacobian `diag(1, …, 1, σ)`). Up to about `5e-9` against `AER::tobit` and up to about `1e-8` against `censReg` on the synthetic scenarios, and up to about `3e-7` against `censReg` on the real-data example, where `censReg` and `AER::tobit` themselves differ by more than elsewhere (see the cross-check table). The `bfgs` / `lbfgs` solvers are compared at their own tolerances and differ by up to about `1e-7`.
- IV: up to about `1e-10` for 2SLS and `3e-10` for GMM, against both linearmodels and R (every `cov_type`, HAC and cluster included).
- FE, RE: up to about `2e-12` against linearmodels, `fixest`, `plm` and statsmodels (the RE constant is part of the matrix). The RE variants compared with `plm` (a time-invariant regressor, two-way effects, `variance_method`) use the same tolerance for the matrix as for their standard errors. A design that contains an absolute-year column or a column very close to its entity mean loosens both, and the matrix agrees up to `1e-8` to `1e-7` there.

For the FE and RE cross-checks (coefficient and F-statistic p-values) the absolute floor `atol` is not applied: with a floor, a tail p-value below it (for example `1e-40`) would pass even if it were returned as `0`. FE p-values are compared by relative error only (`1e-6`, measured up to `3e-8` because the tail amplifies rounding errors in the statistic). RE p-values are compared the same way (`1e-6`), on balanced and unbalanced panels alike.

### Against the primary reference

| Method | rtol | atol | Notes |
|---|---|---|---|
| OLS, WLS, IV (2SLS and GMM), FE, RE | `1e-8` | `1e-10` | Closed-form estimators. Measured agreement is about `1e-14` (RE: `1e-9` to `1e-14`; IV: `1e-10` or better, except the Wald statistic with cluster standard errors on the real-data example, see the cross-check table). Two-way RE: measured up to `4e-12`, except for a column holding an absolute year (see the tolerance table below). |
| OLS White test | `1e-8` | `1e-10` (statistics only) | Closed form from the auxiliary regression's R². Measured agreement is about `2e-12`. p-values are compared by relative error only (no absolute floor), because some are as small as `1e-39` and a floor would hide a tail that collapsed to `0`. |
| OLS Breusch-Pagan test | `1e-8` | `1e-10` (statistics only) | Closed form from the auxiliary regression's R². Measured agreement is about `2e-12`. p-values are compared by relative error only, for the same reason. |
| OLS Breusch-Godfrey test | `1e-8` | `1e-10` (statistics only) | Closed form from the auxiliary regression's residual sum of squares. Measured agreement is about `5e-12`. p-values are compared by relative error only, for the same reason (the smallest is about `1e-108`). |
| Logit, Probit | `1e-8` | `1e-9` | Iterative optimization leaves a little more noise near zero. `bfgs` / `lbfgs` solvers use `1e-3` against statsmodels' own `bfgs` / `lbfgs` (measured at most about `8e-5`), since a different optimization path stops at a slightly different point. The harder `bfgs` / `lbfgs` combinations (the near-separation scenario with classical, OPG and HC0; cluster standard errors; the real-data cluster example) are compared with statsmodels' `newton` result for the same scenario, using `2e-5` (measured at most about `3.6e-6`): statsmodels' own `bfgs` / `lbfgs` stop earlier on near-separation data and differ from this package by up to `7e-3`, which is the reference's error, not ours. |
| Tobit | `1e-8` | `1e-9` | Real-data confidence intervals use `3e-8`; `bfgs` / `lbfgs` solvers use `2e-7`. The harder `bfgs` / `lbfgs` combinations (heavy and interval censoring, OPG / HC0 / HC1, cluster) use `2e-6` (measured at most `5e-7`); on the real-data example they use `3e-6` for point estimates (measured up to `1e-6`) and `3e-4` for standard errors and the statistics built on them (measured up to `1e-4` for HC0 with `lbfgs`), because those solvers stop on a line-search stall before reaching `newton`'s precision, even with a tighter `tol`. |

### Against the independent cross-check

| Method | Tolerance | Looser items and why |
|---|---|---|
| OLS | `1e-8` (classical, HC0–HC3, cluster, HAC; measured about `1e-14`) | none. The R reference uses `NeweyWest(prewhite = FALSE, adjust = FALSE)`. With `adjust = TRUE` (the default of `kernHAC()` and `vcovHAC()`, not of `NeweyWest()`) R multiplies the covariance by `n/(n-k)`, which statsmodels and this package do not apply, and that scalar was the whole difference (see [Inference conventions](inference-conventions.md#differences-from-other-packages)). |
| WLS | `1e-8` (every `cov_type`, HAC included) | none. Same `adjust = FALSE` as OLS. |
| OLS White test | `1e-8` (measured about `1e-12`) | Statistics use an absolute floor of `1e-8`; p-values are compared by relative error only, for the same reason as above. The degrees of freedom are compared exactly. |
| OLS Breusch-Pagan test | `1e-8` (measured about `2e-12`) | Same as the White test. |
| OLS Breusch-Godfrey test | `1e-8` (measured about `6e-12`) | Same as the White test. |
| Logit | `2e-7` (measured up to about `3.5e-8`, including marginal effects with their standard errors, z statistics and confidence intervals, p-values, and the cluster cases; the absolute floor is `1e-12`, so small values are compared relatively too) | Both sides are iterative optimizers. A p-value in the far tail (below about `5e-6`) is compared under the absolute floor only, so the z statistic of the same coefficient, compared at `2e-7`, is what guards it. |
| Probit | `2e-7` (measured up to about `2.8e-8`, including marginal effects with their standard errors, z statistics and confidence intervals, p-values, the cluster cases and the real-data cluster example `apple`, which is about `2e-9` apart from the p-values; the absolute floor is `1e-12`, and tail p-values below about `5e-6` are guarded by the z statistics) | R's `glm` stops on the change in deviance alone, which leaves the coefficients about `1e-7` off for a non-canonical link (about `6e-8` on `apple`) and the robust covariance amplifies that to about `1e-6`; the reference values are therefore refitted from the converged coefficients until they stop moving. |
| Tobit | `1e-8` (measured about `2e-9`) | `5e-8` for HC0 / HC1 on a badly conditioned design; `1e-4` for standard errors on the real-data example, where `censReg` and `AER::tobit` differ from each other by more than on the synthetic data. The strict comparison on that data is against `AER::tobit`. |
| IV (2SLS) | `1e-8` (every `cov_type`, HAC included, also the Wu-Hausman test and the 40-observation and 3-observation samples), except the Wald statistic and its p-value with cluster standard errors on the Wooldridge `card` example (`5e-8`, measured `1.6e-8`; see [the notes](verification-details.md#iv-wald-statistic-with-cluster-standard-errors-on-card)); the absolute floor is `1e-8`, except for the Sargan and Wu-Hausman p-values, which are compared by relative error only (the rejecting scenario's Sargan p-value is about `6e-8`, and a floor would hide a wrong value) | none. The R reference uses `NeweyWest(prewhite = FALSE, adjust = FALSE)`; with `adjust = TRUE` R multiplies the covariance by `n/(n-k)`, and by `n/(n-k_aug)` for the Wu-Hausman auxiliary regression, which has one more regressor per endogenous variable. With cluster standard errors `ivreg` takes the F denominator degrees of freedom of the Wu-Hausman test from the observations rather than the clusters, so its p-value is not compared; it is recomputed from `ivreg`'s statistic with `G - 1` degrees of freedom and compared with this package's. |
| IV (GMM) | `1e-8`, absolute floor `1e-8` (the Wald statistic and its p-value with cluster standard errors on the Wooldridge `card` example: `5e-8`, measured `1.1e-8`; see [the notes](verification-details.md#iv-wald-statistic-with-cluster-standard-errors-on-card)) (measured at most about `9e-11` for standard errors, z statistics and the Wald statistic, `2e-11` for coefficients and `2e-12` for Hansen J) | none. The R script has to reproduce linearmodels' small-sample corrections and weight handling by hand; the reasons are in the notes on the [GMM cross-check](verification-details.md#iv-gmm-cross-check-with-momentfit) below. |
| FE | `1e-8` (every `cov_type`, including cluster and Driscoll-Kraay; measured about `1e-14`) | none. The plm second reference for one-way cluster and Driscoll-Kraay uses the same `1e-8` (measured up to `5e-11` for the F statistic and `8e-13` for coefficients and standard errors); p-values `1e-6` relative, measured up to `9e-10` in the far tail. |
| RE | `1e-8` for the coefficients, standard errors, test statistics, p-values (`1e-6` relative), confidence intervals, F statistic and Hausman test against plm on every scenario, the unbalanced panel included, where the reference is plm applied to data quasi-demeaned with linearmodels' variance components (see [the notes](verification-details.md#fe-and-re-cluster-and-driscoll-kraay-standard-errors)); the statsmodels cluster check uses `1e-8` for coefficients, standard errors, confidence intervals and the F statistic, and `1e-6` relative for p-values (measured up to `4e-13` and `6e-12`) | The Hausman test on the ill-conditioned scenario (two regressors correlated at 0.999, so the tested covariance is nearly singular and rounding errors are amplified): `5e-8` (measured up to `7.6e-9`). |
| RE with a time-invariant regressor | `1e-8` (p-values `1e-6` relative; measured about `1e-12`). Inputs with a column holding an absolute year (see the two-way row below) loosen only the statistics whose rounding error grows (standard errors and what follows from them, F statistic, Hausman test) to `1e-6`; measured up to `8e-8`. | none. Unbalanced panels are not compared with plm directly (see the section on a time-invariant regressor below). |
| RE, two-way | `1e-8` (p-values `1e-8` relative, widened in the tail by `max(1, abs(ln p))` because `ln p` is about minus half the statistic, so the statistic's rounding error is multiplied by the statistic's size; measured up to `4e-12`). Inputs with a column holding an absolute year (1980s–1990s values next to regressors of order one, so the scales differ by three orders of magnitude): the standard errors, t statistics, p-values, confidence intervals, F statistic and Hausman test only are loosened to `1e-6` (measured up to `1.8e-7`); coefficients, variance components, weights, log-likelihood, information criteria and R² keep `1e-8`. | none. The same synthetic regression with the year centred on its mean agrees with plm at `1e-8` like the other scenarios (it is a scenario of its own), so the looser tolerance reflects the conditioning of the input, not a difference between the implementations; this package also moves by about `1e-9` between the two (measured on `wagepan`, where the centred check was done by hand). |
| RE, `variance_method` and correlated random effects | `1e-8` (p-values `1e-8` relative, widened in the tail like two-way RE; measured up to about `1e-10`). Inputs with a column holding an absolute year, or a column within a relative `1e-3` of an entity mean (nearly aliased): standard errors and what follows from them, F statistic and Hausman test only, `1e-6` (measured up to `6e-8`). | none |

## Details of the cross-checks

How particular comparisons are set up, and where a reference needed a workaround or differs from this package, is on [Verification details](verification-details.md):

- [Notes on the cross-checks](verification-details.md#notes-on-the-cross-checks): IV (the Wald statistic with cluster standard errors, the GMM cross-check with momentfit, models without an intercept, the first stage), FE and RE cluster and Driscoll-Kraay standard errors, and the F statistic.
- [Not compared against a second reference](verification-details.md#not-compared-against-a-second-reference): the statistics that have only one reference or a narrower comparison, by method.

## polars versions

The package accepts `polars>=1.38.1` with no upper limit, so installing it never changes the polars you already have. Every pull request runs the whole test suite twice:

| polars | Where |
|---|---|
| 1.38.1 (the oldest supported) | A second `pytest` run on Python 3.12, after the normal run |
| the version in the lockfile (currently 2.0.0) | The normal run on Python 3.12, 3.13 and 3.14. Dependabot updates the lockfile, so new polars releases are tested through its pull requests |

The lower limit is the oldest version for which the whole suite passes. Older versions are known to fail: 1.27 and below cannot hand a DataFrame to the Rust core, 1.28 to 1.31 read an `Enum` time column as different values (so the estimates change without an error), and 1.36 and below cannot read `Float16` columns. Versions between the two tested ones are not run, which is the usual practice of testing both ends of a range. Because the compiled core reads a polars DataFrame through polars' own foreign function interface, a future polars release can break it; if that happens, the test run on the lockfile version fails first.

## Test data

All synthetic datasets are generated once, frozen as CSV files in the repository and read from there, so a later change to a data generator cannot silently invalidate stored reference values.

**Synthetic scenarios** (not every scenario applies to every method):

- small samples, including the boundary of exactly one residual degree of freedom
- large error variance, heteroskedasticity, autocorrelation (HAC)
- moderate multicollinearity, ill-conditioned designs (variables on very different scales, correlations above 0.99)
- many regressors (`k = 20`), regressors with heavy-tailed outliers (for IV only the exogenous regressors are widened or contaminated, so the degree of over-identification does not change)
- cluster structures with equal group sizes, strongly unequal group sizes (for example 2, 3, 5, 10, 30, 50) and the minimum number of clusters
- Tobit: censoring patterns (light, moderate, heavy, right-censored, interval-censored, and interval-censored with a narrow uncensored interval so that observations fall in both tails);
- IV: weak instruments; multiple endogenous variables (over-identified, just-identified, weak instruments, heteroskedastic errors, autocorrelated errors with HAC, a 40-observation sample, and three endogenous variables); and an invalid instrument that is correlated with the structural error, so that the Sargan / Hansen J test rejects (p-values between about `1e-8` and `1e-5`, also with the GMM robust and HAC weights);
- FE / RE: unbalanced panels and cross-sectional dependence
- time-invariant regressors for RE: one, or several (a continuous and a dummy), on a balanced panel and on unbalanced panels, including panel lengths from 2 to 10 observations per entity
- regressors that vary in only one dimension for RE:
    - a regressor that varies over time only (a trend or an absolute year, which one-way RE drops from the between regression) and one that varies across entities only;
    - balanced panels with five periods, with the column in any position, a nearly balanced panel (two rows removed) and every regressor one-dimensional;
    - two-way RE on panels from 5×4 to 15×25, with unequal variance, serial correlation and strongly unequal scales

**Real data** (textbook datasets from the `wooldridge` package, loaded at test time and compared with the references only):

| Method | Dataset |
|---|---|
| OLS | `wage1`, `gpa2`, `mroz` (a linear probability model with the 0/1 `inlf` as the response; classical and HC0-HC3 only, as there is no natural cluster column and no time order); the time series `prminwge` for HAC, with the automatic lag, against both statsmodels and R (the White test also uses `wage1` with the dummies `female` and `married`, with the squared terms `expersq` and `tenursq`, and with the mutually exclusive region dummies `northcen`, `south` and `west`, so that terms of its auxiliary regression are duplicated or constant; the Breusch-Pagan test uses `hprice1` (its LM and F statistics are also compared with the values printed in Wooldridge's Example 8.4), `wage1`, and `wage1` with columns that are not in the model as the variables; the Breusch-Godfrey test also uses the time series `phillips`, with the rows shuffled and ordered by `year`) |
| WLS | `401ksubs` (cross-sectional, so HAC is not checked on real data) |
| Logit, Probit, Tobit | `mroz` |
| Logit, Probit (cluster standard errors with many clusters) | `apple` (`y = ecolbs > 0`, clustered by `state`: 49 clusters of 1 to 66 observations) |
| IV | `card` (cluster standard errors also by region: nine clusters built from the region dummies `reg661`–`reg669`, sizes 85 to 627) |
| FE, RE | `wagepan` (RE also with the time-invariant `educ`, `black` and `hisp` added, and with a year trend `yr` added; two-way RE with `educ` and `yr`) |

**Error paths.** Invalid input (missing values, duplicate columns, too few observations, too few clusters, out-of-range options) must raise a validation error, and numerical failure (for example perfect multicollinearity, complete or quasi-complete separation) must raise a computation error. Only the exception type is checked, with no reference values: most reference packages silently drop the offending columns and continue, which this package deliberately does not do.

## Rust engine tests

The computational core has its own unit tests (`cargo test`) and property-based tests (`proptest`) for the estimators. The property tests check invariants such as scale and permutation invariance across randomly generated inputs, which a fixed set of scenarios cannot cover.

## Reproducing

The reference values are produced by the scripts under `benchmark/` and regenerated with `benchmark/regenerate_all.py`; the generating R and Python packages are pinned. See the [repository](https://github.com/masahiroyecon1997dev/econometricsmodels) for the scripts and the stored reference values under `tests/fixtures/benchmarks/`.

## Where to look next

- [Inference conventions](inference-conventions.md): the distributions, degrees of freedom and defaults that differ from other packages.
- [Validation and errors](validation.md): the error paths that the tests check.
- [Performance](performance.md): how execution time and memory compare with the same reference packages.
