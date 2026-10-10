# Inference conventions

This page summarizes which reference distribution each method uses for its test statistics, p-values and confidence intervals, how to read the diagnostic statistics, and where the defaults differ from R, statsmodels and linearmodels. It matters when you compare output against another package.

## Reference distributions by method

| Method | Coefficient statistics (`test_stats`) | Overall test | Degrees of freedom |
|---|---|---|---|
| OLS | t | F (`f_statistic`) | `n - k`; `G - 1` with `cov_type="cluster"` |
| WLS | t | F | same as OLS |
| FE | t | F | `df_resid` (`n - n_entities - k` for one-way) for `classical`/`hc1`-`hc3`; `G - 1` with `cov_type="cluster"`; `t_periods - 1` with `cov_type="dk"` |
| RE | t | F | `df_resid` (`n - k`) for `classical`/`hc1`-`hc3`; `G - 1` with `cov_type="cluster"`; `t_periods - 1` with `cov_type="dk"`. The F statistic is the same Wald test as FE (it follows `cov_type`), so `f_df_denom` equals the degrees of freedom of the t tests |
| IV, `estimator="2sls"` | t | F (`wald_dist="f"`) | `df_resid`; `G - 1` with `cov_type="cluster"` |
| IV, `estimator="gmm"` | normal | chi-squared (`wald_dist="chi2"`) | none |
| Logit / Probit | normal | likelihood-ratio chi-squared (`lr_statistic`, `lr_df`) | none |
| Tobit | normal | Wald chi-squared (`wald_dist="chi2"`, `wald_df`) | none |

- Every results object reports the distribution actually used in `stat_dist` (`"t"` or `"normal"`) and the t degrees of freedom in `stat_df` (`None` for the normal distribution). `stat_df` is the value used for the p-values and confidence intervals, so it can differ from `df_resid` (for example with `cov_type="cluster"` in OLS, WLS, FE, RE and 2SLS, or `cov_type="dk"` in FE and RE).
- The degrees of freedom of every test statistic are exposed so you can recompute a p-value yourself: `f_df_num` / `f_df_denom` (OLS, WLS, FE, RE), `wald_df_num` / `wald_df_denom` (IV), `wu_hausman_df_num` / `wu_hausman_df_denom`, `weak_instrument_f_df_num` / `weak_instrument_f_df_denom`, `overid_df` (IV), `lr_df` (Logit, Probit), `wald_df` (Tobit), `hausman_df` (RE). A degrees-of-freedom field is `None` when its statistic is not available.
- `wald_dist` (IV and Tobit) tells whether the overall Wald statistic is F or chi-squared, since the name alone does not.
- `marginal_effects()` always uses the normal distribution.

### Why the choice differs between methods

- **t / F (OLS, WLS, FE, RE, 2SLS).** Under the classical assumptions the standardized coefficient follows a t distribution exactly in finite samples. The same t distribution is used for every `cov_type` (classical, HC0-HC3, cluster, HAC) so that switching the standard error type does not silently change the reference distribution.
- **Normal / chi-squared (GMM, Logit, Probit, Tobit).** The theory behind GMM and maximum likelihood is purely asymptotic. There is no finite-sample degrees-of-freedom argument, so a t distribution would claim a justification that does not exist.

## Differences from other packages

The defaults below were checked against R 4.5.3 (sandwich 3.1.3, lmtest 0.9.40, fixest 0.14.2, plm 2.6.7, ivreg 0.6.8, AER 1.2.17), statsmodels 0.15.0 and linearmodels 7.0.

### R

| R | Reference distribution | Difference from econometricsmodels |
|---|---|---|
| `lm` + `summary` | t, `n - k` | same as OLS |
| `lm` + `lmtest::coeftest` with `sandwich::vcovHC` / `vcovCL` | t, `n - k` | same as OLS for HC; with clustering, `cov_type="cluster"` uses `G - 1` degrees of freedom instead of `n - k` |
| `glm(family = binomial)` (logit / probit) | z | same as Logit / Probit |
| `glm(family = gaussian)` | t | not applicable (use OLS) |
| `fixest::feols` | t, `n - k`; with `cluster`, t with `G - 1` | OLS matches; FE's small-sample corrections (`cov_type="cluster"`/`"dk"`) and their degrees of freedom (`G - 1` / `t_periods - 1`) are aligned with fixest's `ssc()` defaults (`K.fixef="nonnested"` for cluster, `"full"` for `hc1`-`hc3`/`dk`). One exception: with `dk_bandwidth = t_periods - 1` fixest's `vcov = DK(...)` silently drops the last lag term (an off-by-one in its compiled code), so its standard errors differ from this package, which keeps the standard Bartlett kernel; smaller bandwidths agree |
| `fixest::feglm` | z | same as Logit / Probit |
| `plm` (`within`, `random`) | p-values from the normal distribution | FE and RE always use t (never normal), so p-values and confidence intervals differ; RE's `cov_type="cluster"` small-sample correction matches `plm::vcovHC(type="sss")` (Stata/R-style `G/(G-1)·(n-1)/(n-K)`) |
| `ivreg::ivreg` | t | same as 2SLS |
| `AER::tobit` | z | same as Tobit |

### statsmodels

- With the default `cov_type="nonrobust"`, `OLS` uses the t distribution. With any other `cov_type` (HC0-HC3, `"cluster"`, `"HAC"`) it switches to the normal distribution (`use_t=False`). econometricsmodels uses t for every `cov_type`, so pass `use_t=True` to statsmodels when comparing.
- `Logit` and `Probit` use the normal distribution, as here.

### linearmodels

- **IV (`IV2SLS`, `IVGMM`)**: `fit(debiased=False)` is the default, which reports normal and chi-squared statistics. econometricsmodels 2SLS always reports t and F (equivalent to `debiased=True`); GMM always reports normal and chi-squared (equivalent to `debiased=False`).
- **Panel (`PanelOLS`, `RandomEffects`)**: `fit(debiased=True)` is the default, which reports t and F, consistent with FE and RE here. Point estimates and the `classical`/`hc1`-`hc3` standard errors still match linearmodels' `PanelOLS`/`RandomEffects`. The `cluster` and `dk` standard errors and their degrees of freedom no longer match linearmodels: FE/RE switched to fixest (R) / Stata-style small-sample corrections (`G/(G-1)·(n-1)/(n-K)`) instead of linearmodels' `n/(n-extra_df-k)`, because that is what fixest, `xtreg` and `reghdfe` users expect. See `docs/spec/fe-spec.md` / `docs/spec/re-spec.md` for the exact formulas. The RE F statistic is the Wald quadratic form (the same as `plm::pwaldtest(test = "F", vcov = ...)` and linearmodels' `f_statistic_robust`), not linearmodels' `f_statistic`, which is a sum-of-squares version that ignores `cov_type`: the two agree only for the classical covariance on balanced panels, and linearmodels' value can even be negative on very unbalanced panels.

## Weak instruments

`weak_instrument_f_statistics` returns, for each endogenous variable, the raw partial F statistic of the excluded instruments in the first stage. It always assumes homoskedasticity and does not depend on `cov_type`.

The result is not compared with the Stock-Yogo critical values, so no weak / strong verdict is reported. A commonly quoted rule of thumb is an F statistic above about 10; whether that is adequate for your application is left to you. Joint diagnostics for several endogenous variables (Cragg-Donald statistic) are not provided.

## Overidentification tests

`overid_statistic` and `overid_p_value` (chi-squared with `overid_df` degrees of freedom, `None` when just-identified) are computed differently for the two estimators.

- **2SLS reports the Sargan test.** It is always computed under homoskedasticity and does not depend on `cov_type`.
- **GMM reports Hansen's J test.** It uses the weight matrix of the point estimate and therefore follows `gmm_weight_type`.

If you need an overidentification test that is robust to heteroskedasticity or clustering, use `estimator="gmm"` with a robust or cluster `gmm_weight_type` and read Hansen's J.

## Other diagnostics

- **White test (OLS, `white_test()`)**: a post-estimation diagnostic, not computed by `fit()`. `statistic="lm"` is `n * R²` of the auxiliary regression of the squared residuals on the regressors, their squares and cross products (chi-squared with `df` degrees of freedom); `statistic="f"` is the F version (`F(df, df_denom)`). The auxiliary regression always has a constant and does not depend on `cov_type`. Constant terms and terms numerically identical to an earlier one (such as the square of a 0/1 dummy) are dropped, and `df` counts the rank (`aux_terms` and `dropped_terms` show which terms). statsmodels `het_white` and R's `lmtest::bptest(studentize = TRUE)` give the same values.
- **Hausman test (RE)**: `hausman_statistic` / `hausman_p_value` / `hausman_df` are the regression-based (auxiliary regression) version, equivalent to `plm::phtest(method = "aux", effect = "individual")`. The comparison is always against one-way FE. The Wald test on the auxiliary regression follows the `cov_type` of the RE fit (the default `"cluster"` gives the cluster-robust Hausman test; `"classical"` gives the classical version, which assumes RE is fully efficient under the null). The auxiliary regression uses the small-sample corrections of `OLS` (Stata/R-style, e.g. `G/(G-1)·(n-1)/(n-k)` for cluster), which now use the same style of correction as the RE standard errors themselves (both fixest/Stata-style since FE/RE moved off the linearmodels-style corrections). The statistic is non-negative by construction. If the robust covariance is structurally singular (e.g. the number of clusters does not exceed the `2k` auxiliary slopes, or there are too few periods for `"dk"`), `fit()` raises an error instead of returning `None`.
- **Wu-Hausman test (IV)**: follows the `cov_type` passed to `fit()`.
