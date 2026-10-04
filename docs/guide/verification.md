# Verification

Every estimator is compared numerically against established reference implementations before it is considered done. This page lists which implementations are used for each method, how closely the results must agree, and what is deliberately *not* compared against a reference.

## How verification works

- **Primary reference.** For each statistic a primary reference implementation is chosen, the one whose conventions match the ones used here. Results are compared at a relative tolerance of `1e-8` unless a documented reason requires a looser value.
- **Independent cross-check.** Where available, a second, independent implementation (mostly R) is compared as well. It covers the main reported statistics, not only coefficients and standard errors, so a bug in the primary reference or a shared misunderstanding is not silently inherited. Statistics without an independent reference are listed under [Not compared against a second reference](#not-compared-against-a-second-reference).
- **Frozen reference values.** Reference values are generated once from the pinned reference packages and stored as JSON files in the repository. The test suite compares against these files, so R and the reference packages are not needed to run the tests, and a new release of a reference package cannot change a result unnoticed.
- **Convention differences are documented.** When two implementations use different statistical conventions, such as small-sample corrections, the difference is documented (see [Inference conventions](inference-conventions.md)) and the comparison is adjusted for that statistic only.
- **Continuous integration.** The comparisons run on every pull request in the *CI (python)* workflow on Python 3.12, 3.13 and 3.14, and the Rust engine's own unit and property-based tests run in *CI (engine)*.

## Reference implementations

| Method | Primary reference | Independent cross-check |
|---|---|---|
| OLS | statsmodels `OLS` | R `lm` + `sandwich` / `lmtest` |
| WLS | statsmodels `WLS` | R `lm(weights=)` + `sandwich` / `lmtest` |
| Logit | statsmodels `Logit` | R `glm` + `sandwich`, `marginaleffects` |
| Probit | statsmodels `Probit` | R `glm` + `sandwich`, `marginaleffects` (observed-information covariance, see below) |
| Tobit | R `AER::tobit` | R `censReg` |
| IV, 2SLS | linearmodels `IV2SLS` | R `ivreg` + `sandwich` / `lmtest` |
| IV, GMM | linearmodels `IVGMM` | none (`ivreg` has no GMM) |
| FE | linearmodels `PanelOLS` (classical, HC1); R `fixest` (cluster, Driscoll–Kraay, and the F statistic for every `cov_type`) | R `fixest`; R `plm` + `sandwich` (one-way cluster and Driscoll–Kraay only, see the section on cluster and Driscoll–Kraay standard errors below) |
| RE | linearmodels `RandomEffects` (classical, HC1); R `plm` (every `cov_type`, and the only reference for HC2, HC3, cluster, Driscoll–Kraay) | R `plm`; statsmodels OLS on plm's quasi-demeaned data (cluster inference: standard errors, degrees of freedom, F statistic) |

Versions used to generate the reference values: statsmodels 0.15.0, linearmodels 7.0, R 4.5.3 with sandwich 3.1.3, lmtest 0.9.40, fixest 0.14.2, plm 2.6.7, ivreg 0.6.8, AER 1.2-17, censReg 0.5-40 and marginaleffects 1.0.0.

pyfixest is not used for accuracy checks. Its HC2/HC3 standard errors apply the HC1 degrees-of-freedom correction by mistake, so it is only used for [performance comparisons](performance.md).

## What is compared

| Method | Statistics |
|---|---|
| OLS, WLS | coefficients, standard errors, t statistics, p-values, confidence intervals, R², adjusted R², F statistic and p-value, log-likelihood, AIC, BIC, predictions |
| Logit, Probit | coefficients, standard errors, z statistics, p-values, confidence intervals, log-likelihood, null log-likelihood, likelihood-ratio statistic and p-value, McFadden R², AIC, BIC, marginal effects with standard errors |
| Tobit | coefficients and σ with their standard errors, z statistics, p-values and confidence intervals, log-likelihood, AIC, BIC, Wald statistic and p-value, marginal effects, predictions |
| IV | coefficients, standard errors, test statistics, p-values, confidence intervals, R², robust Wald statistic, weak-instrument F, Wu-Hausman (2SLS), Sargan (2SLS) or Hansen J (GMM) |
| FE, RE | coefficients, standard errors, t statistics, p-values, confidence intervals, F statistic, within / between / overall R², Hausman test (RE) |

Standard-error types are covered per method: classical, HC0–HC3, HAC, cluster, Driscoll–Kraay (panel) and OPG (maximum likelihood), wherever the method supports them. Each reference package covers a subset, so some combinations have only one reference (listed under [Not compared against a second reference](#not-compared-against-a-second-reference)).

## Tolerances

Agreement is checked as `|ours - reference| <= max(rtol · |reference|, atol)`.

For the FE and RE cross-checks (coefficient and F-statistic p-values) the absolute floor `atol` is not applied: with a floor, a tail p-value below it (for example `1e-40`) would pass even if it were returned as `0`. FE p-values are compared by relative error only (`1e-6`, measured up to `3e-8` because the tail amplifies rounding errors in the statistic). RE p-values are compared the same way on balanced panels; on an unbalanced panel, where the statistics themselves differ by up to 0.7 % (see below), they are compared on the base-10 logarithm (within `0.1`, measured below `0.03`).

### Against the primary reference

| Method | rtol | atol | Notes |
|---|---|---|---|
| OLS, WLS, IV (2SLS and GMM), FE, RE | `1e-8` | `1e-10` | Closed-form estimators. Measured agreement is about `1e-14` (RE: `1e-9` to `1e-14`). |
| Logit, Probit | `1e-8` | `1e-9` | Iterative optimization leaves a little more noise near zero. `bfgs` / `lbfgs` solvers use `1e-3` (measured at most about `8e-5`), since a different optimization path stops at a slightly different point. |
| Tobit | `1e-8` | `1e-9` | Real-data confidence intervals use `3e-8`; `bfgs` / `lbfgs` solvers use `2e-7`. |

### Against the independent cross-check

| Method | Tolerance | Looser items and why |
|---|---|---|
| OLS | `1e-8` (classical, HC0–HC3, cluster; measured about `1e-14`) | HAC `1e-2` (measured about 0.4 %): R's Newey–West small-sample, pre-whitening and adjustment conventions differ. p-values use an absolute tolerance of `1e-6`. |
| WLS | `1e-8` | HAC `5e-2` (measured at most about 4.3 %). p-values use an absolute tolerance of `1e-6`. |
| Logit | `2e-4` (measured about `1.4e-4`) | Both sides are iterative optimizers. Marginal-effect standard errors `2e-4` (delta method, measured `8.3e-5`). |
| Probit | `2e-4` (measured about `9e-5`) | Marginal-effect standard errors `2e-4` (delta method, measured `5.5e-5`). |
| Tobit | `1e-8` (measured about `2e-9`) | `5e-8` for HC0 / HC1 on a badly conditioned design; `1e-4` for standard errors on the real-data example, limited by `censReg`'s convergence. The strict comparison on that data is against `AER::tobit`. |
| IV (2SLS) | `1e-8` | HAC `1e-2` (`0.1` for a 40-observation sample); Wu-Hausman under HAC `2e-2`. These are small-sample convention differences. |
| FE | `1e-8` (every `cov_type`, including cluster and Driscoll–Kraay; measured about `1e-14`) | none. The plm second reference for one-way cluster and Driscoll–Kraay uses the same `1e-8` (measured up to `5e-11` for the F statistic and `8e-13` for coefficients and standard errors); p-values `1e-6` relative, measured up to `9e-10` in the far tail. |
| RE | `1e-8` on balanced panels (the statsmodels cluster check uses `1e-8` for coefficients, standard errors, confidence intervals and the F statistic, and `1e-6` relative for p-values; measured up to `4e-13` and `6e-12`) | Unbalanced panel, because plm and linearmodels estimate the Swamy–Arora variance components slightly differently: coefficients `5e-3`; F statistic `1e-3` for classical, `2e-3` for HC1 / HC2 / HC3, `4e-3` for cluster and `2e-2` for Driscoll–Kraay (measured up to about 0.03 %, 0.09 %, 0.13 % and 0.7 %); standard errors, test statistics, p-values and confidence intervals `5e-3` for cluster, `2e-2` for classical/HC1/HC2/HC3 and `5e-2` for Driscoll–Kraay (measured up to about 0.3 %, 1.1 % and 3.7 %). The tolerance is set per statistic so that a missing `G/(G-1)` correction (about 1.3 % in the cluster standard error) is still detected. |

### FE and RE: cluster and Driscoll–Kraay standard errors

Point estimates and the classical and HC1 standard errors agree with linearmodels to machine precision. For `cov_type="cluster"` and `cov_type="dk"`, FE and RE deliberately use the small-sample corrections and degrees of freedom of fixest, Stata and plm rather than those of linearmodels (see [Inference conventions](inference-conventions.md#differences-from-other-packages)), so linearmodels is not a numerical reference for these two types. They are compared with R instead: FE against `fixest` with its default `ssc()` (every statistic, including p-values and confidence intervals, agrees to machine precision), RE against `plm::vcovHC(method = "arellano", type = "sss")` for cluster and `plm::vcovSCC(type = "sss")` for Driscoll–Kraay. RE with a cluster column other than the entity is compared against `lm` plus `sandwich::vcovCL` on plm's quasi-demeaned data, because plm can only cluster by entity or time. The Driscoll–Kraay bandwidth is passed to the reference explicitly, because neither fixest nor plm uses the same default rule. plm reports z statistics, so its t statistics, p-values and confidence intervals are recomputed from its standard errors with the t distribution (`G - 1` degrees of freedom for cluster, `t_periods - 1` for Driscoll–Kraay).

**Second reference for FE.** FE cluster and Driscoll–Kraay have a second, independent reference built from `plm` (`model = "within"`) and `sandwich`, for one-way FE only (plm has no covariance matrix for two-way within models and can only cluster by entity or time). It agrees with the implementation and with fixest to machine precision on balanced and unbalanced panels, for every statistic including the F statistic (`plm::pwaldtest`). It is less independent than the table suggests:

- **Cluster:** plm supplies the within-transformed data and the coefficients; the standard error comes from `sandwich::vcovCL` on that data, so it is not a plm covariance estimator. The small-sample factor is computed by `sandwich` rather than written by hand, but the count `K = k + 1` (the absorbed entity effect counted once, as in fixest's `K.fixef = "nonnested"`) is chosen in the reference script by adding an intercept, so it is the same convention as fixest and the implementation, not an independent one. The `t` degrees of freedom (`G - 1`) are the same assumption.
- **Driscoll–Kraay:** plm's own `type = "sss"` correction counts the fixed effects differently from fixest, so it is not used. The raw Bartlett sandwich `plm::vcovSCC(type = "HC0")` is multiplied by fixest's factor `T / (T - 1) · (n - 1) / (n - k - G)`, written by hand in the reference script. plm therefore checks the kernel and bandwidth convention, not the correction factor, which only fixest verifies independently. The `t_periods - 1` degrees of freedom are again the same assumption.
- **Bandwidth equal to `t_periods - 1`:** fixest drops the last lag term at this boundary and does not match the implementation, while plm uses the standard Bartlett kernel and does. This boundary is checked against plm only.
- **Boundary cases:** the second reference also covers the cases fixest is checked on where a small count of clusters or periods makes the denominator degrees of freedom equal to the number of slopes (the case of fewer degrees of freedom than slopes is checked against fixest only): three clusters with two slopes, two clusters with one slope, and three periods with two slopes for Driscoll–Kraay. Because plm clusters only by entity, the cluster counts are reached by keeping the first three or two entities rather than by an unrelated cluster column, and unequal group sizes are represented, more mildly than for RE, by unequal numbers of observations per entity (2, 3, 5 and 10 periods in turn). Driscoll–Kraay is also checked with bandwidths 0 and 1 and, for the boundary above, on balanced, unbalanced and four-period panels. Within R² is compared with plm as well.

**Degrees of freedom for RE.** plm reports z statistics, so the `G - 1` degrees of freedom for cluster are written by hand in the plm reference script. As an independent check, plm's quasi-demeaned data is passed to statsmodels `OLS(...).fit(cov_type="cluster", use_t=True)`, which computes the cluster standard errors, t statistics, p-values, confidence intervals, F statistic and the `G - 1` degrees of freedom natively. It agrees with the implementation to machine precision on balanced panels, including clusters defined by a column other than the entity (strongly unequal cluster sizes of 2 to 50 observations, and the minimal case of three clusters with one slope). statsmodels cannot do the same for Driscoll–Kraay (its `hac-groupsum` inference degrees of freedom are not `t_periods - 1`), so the `t_periods - 1` degrees of freedom of RE Driscoll–Kraay have no second reference: plm verifies the standard error only.

### FE and RE: F statistic

The F statistic tests that all slope coefficients are jointly zero (the constant and the fixed effects are not tested). FE and RE are checked against R for every `cov_type`, and the two methods are different:

- **FE** is a Wald test that follows `cov_type`, with `G - 1` denominator degrees of freedom for cluster and `t_periods - 1` for Driscoll–Kraay. It is compared with `fixest::wald(keep = <slopes>, vcov = <same vcov>)` for classical, HC1, HC2, HC3, cluster and Driscoll–Kraay, one-way and two-way, and agrees to machine precision. `fixest::fitstat(m, "f")` is not used because it also tests the fixed-effect dummies. The p-value is recomputed from the fixest statistic and the degrees of freedom of the t tests, because `wald()` raises the denominator degrees of freedom to at least the numerator degrees of freedom plus one (this only matters when the denominator degrees of freedom do not exceed the number of slope coefficients: `G - 1`, `t_periods - 1` or `df_resid` ≤ the number of slopes, e.g. two clusters with one slope, or one residual degree of freedom). The boundaries `G = q + 1` and `t_periods = q + 1` with two slopes, and one residual degree of freedom with HC1–HC3, are included. classical and HC1 are also compared with linearmodels. For one-way FE with cluster and Driscoll–Kraay it is also compared with `plm::pwaldtest(test = "F", vcov = ...)` on the plm-based covariance matrix of the second reference (see above); the p-value uses the same hand-written denominator degrees of freedom (`G - 1`, `t_periods - 1`) as the t tests, and for Driscoll–Kraay the covariance matrix contains the hand-written correction factor.
- **RE** is the same Wald test as FE and also follows `cov_type` (denominator degrees of freedom `G - 1` for cluster, `t_periods - 1` for Driscoll–Kraay, `df_resid` otherwise). Every `cov_type` is compared with the statistic of `plm::pwaldtest(test = "F", vcov = ...)`, and cluster on balanced panels also with statsmodels `f_test` on plm's quasi-demeaned data (its `G - 1` denominator degrees of freedom are computed natively); classical and HC1 are also compared with linearmodels' `f_statistic_robust`. Both agree to machine precision on balanced panels, and linearmodels also on the unbalanced panel because the point estimates agree. plm differs on the unbalanced panel because of the Swamy–Arora variance components (see above): by up to about 0.03 % for classical, 0.09 % for HC1 / HC2 / HC3, 0.13 % for cluster and 0.7 % for Driscoll–Kraay. The p-value is recomputed from the plm statistic and the degrees of freedom of the t tests, because `pwaldtest` only adjusts its denominator degrees of freedom when the covariance matrix carries a `cluster` attribute. linearmodels' `f_statistic` is not used: it is based on the sums of squares of the transformed data, ignores `cov_type` and can be **negative** for very unbalanced panels, while the Wald form cannot.

## Not compared against a second reference

- **Tobit: both references are R packages.** Primary and cross-check are both R implementations (`survreg` and `maxLik`). The hand-written parts of the cross-check script (the Jacobian from log σ to σ, the HC1 correction, the marginal effects) are verified against numerical derivatives and an independent implementation of the closed-form expressions.
- **Probit and R's `glm`.** For a non-canonical link, R's `glm` covariance uses the expected information matrix, which differs from the observed information used here by 2–3 % (classical) and up to 8 % (HC0 / HC1). The cross-check therefore builds the observed-information covariance explicitly, checked against `numDeriv::hessian`. Logit is unaffected.
- **statsmodels gaps for Logit and Probit.** statsmodels returns HC0 for `hc1` (no `n/(n-k)` correction) and cannot compute OPG marginal effects, so R is the only reference for those two.
- **IV GMM** is checked against linearmodels only. **IV HC2/HC3** is checked against `ivreg` only, because linearmodels has no equivalent.
- **FE:** AIC, BIC and log-likelihood are checked against fixest only (linearmodels does not provide them), and so are HC2 / HC3, two-way cluster and Driscoll–Kraay, cluster by a column other than the entity, and the two-way within R². One-way cluster and Driscoll–Kraay with entity clustering also have the plm reference described above, which is not a fully independent check of the small-sample correction for Driscoll–Kraay (linearmodels has no HC2 / HC3, uses different small-sample corrections for cluster and Driscoll–Kraay, and uses a different within R² definition for two-way FE). Between / overall R² are checked against linearmodels only.
- **RE:** AIC, BIC and log-likelihood have no independent reference and rely on the OLS tests of the underlying computation. HC2 / HC3, cluster, Driscoll–Kraay and the Hausman test (the regression-based version, always compared with one-way FE) are checked against plm only; the cluster `G - 1` degrees of freedom are additionally checked against statsmodels, but the Driscoll–Kraay `t_periods - 1` degrees of freedom have no second reference. Between / overall R² are checked against linearmodels only (plm has no equivalent).

## Test data

All synthetic datasets are generated once, frozen as CSV files in the repository and read from there, so a later change to a data generator cannot silently invalidate stored reference values.

**Synthetic scenarios** (not every scenario applies to every method):

- small samples, including the boundary of exactly one residual degree of freedom
- large error variance, heteroskedasticity, autocorrelation (HAC)
- moderate multicollinearity, ill-conditioned designs (variables on very different scales, correlations above 0.99)
- many regressors (`k = 20`), regressors with heavy-tailed outliers
- cluster structures with equal group sizes, strongly unequal group sizes (for example 2, 3, 5, 10, 30, 50) and the minimum number of clusters
- censoring patterns (light, moderate, heavy, right-censored, interval-censored) for Tobit; weak and multiple instruments for IV; unbalanced panels and cross-sectional dependence for FE / RE

**Real data** (textbook datasets from the `wooldridge` package, loaded at test time and compared with the references only):

| Method | Dataset |
|---|---|
| OLS | `wage1`, `gpa2` |
| WLS | `401ksubs` |
| Logit, Probit, Tobit | `mroz` |
| IV | `card` |
| FE, RE | `wagepan` |

**Error paths.** Invalid input (missing values, duplicate columns, too few observations, too few clusters, out-of-range options) must raise a validation error, and numerical failure (for example perfect multicollinearity, complete separation) must raise a computation error. Only the exception type is checked, with no reference values: most reference packages silently drop the offending columns and continue, which this package deliberately does not do.

## Rust engine tests

The computational core has its own unit tests (`cargo test`) and property-based tests (`proptest`) for the estimators. The property tests check invariants such as scale and permutation invariance across randomly generated inputs, which a fixed set of scenarios cannot cover.

## Reproducing

The reference values are produced by the scripts under `benchmark/` and regenerated with `benchmark/regenerate_all.py`; the generating R and Python packages are pinned. See the [repository](https://github.com/masahiroyecon1997dev/econometricsmodels) for the scripts and the stored reference values under `tests/fixtures/benchmarks/`.
