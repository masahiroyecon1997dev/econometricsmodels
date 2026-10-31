# Inference conventions

This page summarizes which reference distribution each method uses for its test statistics, p-values and confidence intervals, how to read the diagnostic statistics, and where the defaults differ from R, statsmodels and linearmodels. It matters when you compare output against another package.

## Reference distributions by method

Notation: `n` is the number of observations, `k` the number of coefficients, `G` the number of clusters, `n_entities` the number of panel entities, and `t_periods` the number of distinct periods of the `dk_time` column (the periods of the Driscoll-Kraay kernel).

| Method | Coefficient statistics (`test_stats`) | Overall test | Degrees of freedom |
|---|---|---|---|
| OLS | t | F (`f_statistic`) | `n - k`; `G - 1` with `cov_type="cluster"` |
| WLS | t | F | same as OLS |
| FE | t | F | `df_resid` (`n - n_entities - k` for one-way) for `classical`/`hc1`-`hc3`; `G - 1` with `cov_type="cluster"`; `t_periods - 1` with `cov_type="dk"` |
| RE (one-way and two-way) | t | F | `df_resid` (`n - k`, with `k` all coefficients including the constant, for `classical`/`hc1`-`hc3`); `G - 1` with `cov_type="cluster"`; `t_periods - 1` with `cov_type="dk"`. The F statistic follows `cov_type`, so `f_df_denom` equals the degrees of freedom of the t tests. See [Definitions: RE](../methods/re.md#degrees-of-freedom) |
| IV, `estimator="2sls"` | t | F (`wald_dist="f"`) | `df_resid`; `G - 1` with `cov_type="cluster"` |
| IV, `estimator="gmm"` | normal | chi-squared (`wald_dist="chi2"`) | none |
| Logit / Probit | normal | likelihood-ratio chi-squared (`lr_statistic`, `lr_df`) | none |
| Tobit | normal | Wald chi-squared (`wald_dist="chi2"`, `wald_df`) | none |

- Every results object reports the distribution actually used in `stat_dist` (`"t"` or `"normal"`) and the t degrees of freedom in `stat_df` (`None` for the normal distribution). `stat_df` is the value used for the p-values and confidence intervals, so it can differ from `df_resid` (for example with `cov_type="cluster"` in OLS, WLS, FE, RE and 2SLS, or `cov_type="dk"` in FE and RE).
- The degrees of freedom of every test statistic are exposed so you can recompute a p-value yourself: `f_df_num` / `f_df_denom` (OLS, WLS, FE, RE), `wald_df_num` / `wald_df_denom` (IV), `wu_hausman_df_num` / `wu_hausman_df_denom`, `weak_instrument_f_df_num` / `weak_instrument_f_df_denom`, `overid_df` (IV), `lr_df` (Logit, Probit), `wald_df` (Tobit), `hausman_df` (RE). A degrees-of-freedom field is `None` when its statistic is not available. The overall F statistic (`f_statistic`, `f_p_value`) is also `None` when it is not defined, for example for a first-stage regression with `cov_type="cluster"` and no more clusters than slope coefficients (see [Weak instruments](#weak-instruments)).
- `REOptions.variance_method` (`"swamy_arora"`, `"wallace_hussain"`, `"amemiya"`) selects how the RE variance components are estimated. Wallace-Hussain and Amemiya follow `plm`; Swamy-Arora on an unbalanced one-way panel is the `linearmodels` formula, not `plm`'s, so the two give different numbers there. The formulas and the cases that are rejected are in [Definitions: RE](../methods/re.md#variance-components-variance_method).
- Two-way RE (`REOptions.time`, balanced panels) reports `n_periods`. `time` only defines the time effect: the Driscoll-Kraay time column is the separate `dk_time` option.
- `wald_dist` names the distribution of the overall Wald statistic: `"f"` or `"chi2"` for IV, depending on the estimator, and always `"chi2"` for Tobit.
- `marginal_effects()` always uses the normal distribution.

### Why the choice differs between methods

- **t / F (OLS, WLS, FE, RE, 2SLS).** Under the classical assumptions the standardized coefficient follows a t distribution exactly in finite samples. The same t distribution is used for every `cov_type` (classical, HC0-HC3, cluster, HAC) so that switching the standard error type does not silently change the reference distribution.
- **Normal / chi-squared (GMM, Logit, Probit, Tobit).** The theory behind GMM and maximum likelihood is purely asymptotic. There is no finite-sample degrees-of-freedom argument, so a t distribution would claim a justification that does not exist.

## Differences from other packages

The defaults below were checked against the package versions listed under [Verification](verification.md#reference-implementations). Statements about a package apply to those versions.

### R

| R | Reference distribution | Difference from econometricsmodels |
|---|---|---|
| `lm` + `summary` | t, `n - k` | same as OLS |
| `lm` + `lmtest::coeftest` with `sandwich::vcovHC` / `vcovCL` | t, `n - k` | same as OLS for HC; with clustering, `cov_type="cluster"` uses `G - 1` degrees of freedom instead of `n - k` |
| `lm` + `sandwich::NeweyWest` | t, `n - k` | the standard errors agree only with `prewhite = FALSE` (the default of `NeweyWest()` is `TRUE`) and `adjust = FALSE` (the default of `NeweyWest()`). `adjust = TRUE`, which is the default of `kernHAC()` and `vcovHAC()`, multiplies the covariance by `n/(n-k)` (standard errors by `sqrt(n/(n-k))`; 0.4 % for n = 500, k = 4, 4 % for n = 40, k = 3), which OLS, WLS and IV (HAC) here, statsmodels and linearmodels do not apply. For the 2SLS Wu-Hausman test the factor is `n/(n-k_aug)`, with one more regressor per endogenous variable |
| `glm(family = binomial)` (logit / probit) | z | same as Logit / Probit |
| `glm(family = gaussian)` | t | not applicable (use OLS) |
| `fixest::feols` | t, `n - k`; with `cluster`, t with `G - 1` | OLS matches; FE's small-sample corrections (`cov_type="cluster"`/`"dk"`) and their degrees of freedom (`G - 1` / `t_periods - 1`) are aligned with fixest's `ssc()` defaults (`K.fixef="nonnested"` for cluster, `"full"` for `hc1`-`hc3`/`dk`). One exception: with `dk_bandwidth = t_periods - 1`, `vcov = DK(lag = t_periods - 1)` gave standard errors equal to the computation with lags 1 to `t_periods - 2` only, so they differ from this package, which includes the last lag of the standard Bartlett kernel; smaller bandwidths agree. The fixest documentation does not state the maximum lag, so the cause was not determined |
| `fixest::feglm` | z | same as Logit / Probit |
| `plm` (`within`) | t, `df.residual` (`n - n_entities - k`) | same as FE for `classical` and `hc1`-`hc3`; with `cov_type="cluster"` or `"dk"` this package uses `G - 1` or `t_periods - 1` degrees of freedom |
| `plm` (`random`) | z (normal) | RE uses t, so p-values and confidence intervals differ; RE's `cov_type="cluster"` small-sample correction matches `plm::vcovHC(type="sss")` (Stata/R-style `G/(G-1)·(n-1)/(n-K)`) |
| `ivreg::ivreg` | t | same as 2SLS, except with cluster standard errors: `ivreg` takes the Wu-Hausman F denominator degrees of freedom from the observations (`n - k_aug`), while this package uses `G - 1` |
| `AER::tobit` | z | same as Tobit |

### statsmodels

- With the default `cov_type="nonrobust"`, `OLS` uses the t distribution. With any other `cov_type` (HC0-HC3, `"cluster"`, `"HAC"`) it switches to the normal distribution (`use_t=False`). econometricsmodels uses t for every `cov_type`, so pass `use_t=True` to statsmodels when comparing.
- `Logit` and `Probit` use the normal distribution, as here.

### linearmodels

- **IV (`IV2SLS`, `IVGMM`)**: `fit(debiased=False)` is the default, which reports normal and chi-squared statistics. econometricsmodels 2SLS always reports t and F (equivalent to `debiased=True`, except with `cov_type="cluster"`, where the degrees of freedom are `G - 1` and linearmodels uses `n - k`); GMM always reports normal and chi-squared (equivalent to `debiased=False`).
- **Panel (`PanelOLS`, `RandomEffects`)**: `fit(debiased=True)` is the default, which reports t and F, consistent with FE and RE here. Point estimates and the `classical`/`hc1`-`hc3` standard errors match linearmodels' `PanelOLS`/`RandomEffects`.
    - The `cluster` and `dk` standard errors and their degrees of freedom differ from linearmodels. FE and RE use the fixest (R) and Stata small-sample corrections (`G/(G-1)·(n-1)/(n-K)`), which is what `fixest`, `xtreg` and `reghdfe` users expect, whereas linearmodels uses `n/(n-extra_df-k)`. The exact formulas are in [Definitions: FE](../methods/fe.md#covariance-of-the-coefficients-cov_type) and [Definitions: RE](../methods/re.md#covariance-of-the-coefficients-cov_type).
    - The RE F statistic is the Wald quadratic form (the same as `plm::pwaldtest(test = "F", vcov = ...)` and linearmodels' `f_statistic_robust`), not linearmodels' `f_statistic`. That one is a sum-of-squares version that ignores `cov_type`; the two agree only for the classical covariance on balanced panels.
    - RE accepts a time-invariant regressor (FE rejects it). The variance component σ_ε² is then estimated from the within regression of the time-varying regressors only, with `n - N - k_tv` degrees of freedom (as in `plm`), whereas linearmodels also counts the time-invariant columns, so its σ_ε² and therefore its coefficients differ from the fourth digit.

## Weak instruments

`weak_instrument_f_statistics` returns, for each endogenous variable, the raw partial F statistic of the excluded instruments in the first stage. It always assumes homoskedasticity and does not depend on `cov_type`.

Because it is the classical (homoskedastic) statistic, it is not cluster-robust evidence of relevance: with `cov_type="cluster"` and few clusters it can look strong while the cluster-robust first stage is imprecise. This limitation exists regardless of the number of clusters.

With `cov_type="cluster"`, the structural equation needs more clusters than its slope coefficients. A first-stage regression returned by `first_stage()` can have more slope coefficients (the exogenous regressors plus all excluded instruments). When the number of clusters is larger than the structural slopes but not larger than the slopes of a first-stage regression, `fit()` still succeeds: that regression's overall F statistic (`f_statistic`, `f_p_value`, `f_df_num`, `f_df_denom`) is `None`, because the cluster-robust covariance of its slopes is singular. Its coefficients, standard errors and the weak-instrument F statistic are unaffected.

The result is not compared with the Stock-Yogo critical values, so no weak / strong verdict is reported. A commonly quoted rule of thumb is an F statistic above about 10; whether that is adequate for your application is left to you. Joint diagnostics for several endogenous variables (Cragg-Donald statistic) are not provided.

## Overidentification tests

`overid_statistic` and `overid_p_value` (chi-squared with `overid_df` degrees of freedom, `None` when just-identified) are computed differently for the two estimators.

- **2SLS reports the Sargan test.** It is always computed under homoskedasticity and does not depend on `cov_type`.
- **GMM reports Hansen's J test.** It uses the weight matrix of the point estimate and therefore follows `gmm_weight_type`.

If you need an overidentification test that is robust to heteroskedasticity or clustering, use `estimator="gmm"` with a robust or cluster `gmm_weight_type` and read Hansen's J.

## Endogeneity test (Wu-Hausman)

`wu_hausman_statistic` and `wu_hausman_p_value` (IV, `estimator="2sls"` only) are the regression-based test described in [Definitions: IV](../methods/iv.md#endogeneity-wu_hausman_statistic). Unlike the weak-instrument statistic, it follows the `cov_type` passed to `fit()`; the F test has `wu_hausman_df_num` and `wu_hausman_df_denom` degrees of freedom.

## Other diagnostics

- **White test (OLS, `white_test()`)**: a post-estimation diagnostic, not computed by `fit()`. `statistic="lm"` is `n * R²` of the auxiliary regression of the squared residuals on the regressors, their squares and cross products (chi-squared with `df` degrees of freedom); `statistic="f"` is the F version (`F(df, df_denom)`). The auxiliary regression always has a constant and does not depend on `cov_type`. Constant terms and terms numerically identical to an earlier one (such as the square of a 0/1 dummy) are dropped, and `df` counts the rank (`aux_terms` and `dropped_terms` show which terms). statsmodels `het_white` and R's `lmtest::bptest(studentize = TRUE)` give the same values.
- **Breusch-Pagan test (OLS, `breusch_pagan_test(variables=None)`)**: a post-estimation diagnostic, not computed by `fit()`. It is Koenker's studentized version, which does not assume normal errors: `statistic="lm"` is `n * R²` of the auxiliary regression of the squared residuals on a constant and `variables` (chi-squared with `df` degrees of freedom) and `statistic="f"` is the F version (`F(df, df_denom)`). `variables` is a list of column names that may drive the error variance; it defaults to the model's regressors but can be any numeric columns, including ones that are not in the model. The auxiliary regression always has a constant and does not depend on `cov_type`. Constant columns and columns numerically identical to an earlier one are dropped, and `df` counts the rank (`aux_terms` and `dropped_terms` show which). R's `lmtest::bptest(studentize = TRUE)` and statsmodels `het_breuschpagan(robust=True)` give the same values (statsmodels counts the chi-squared degrees of freedom from the number of columns, so it differs only when constant or duplicate columns are passed).
- **Breusch-Godfrey test (OLS, `breusch_godfrey_test(time, nlags)`)**: a post-estimation diagnostic for serial correlation of the errors, not computed by `fit()`. The residuals are regressed on the model's regressors and their own lags 1 to `nlags`; `statistic="lm"` is `n * R²` (chi-squared with `nlags` degrees of freedom) and `statistic="f"` is the F version (`F(nlags, n - k - nlags)`). `time` and `nlags` have no defaults: the row order is never used as the time order. Lags before the first observation are filled with zero, as in R's `lmtest::bgtest` and statsmodels. The auxiliary regression uses the model's regressors as they are, so a model fitted with `include_intercept=False` gets no constant (R and Greene); statsmodels adds a constant in that case, so its value differs there. It does not depend on `cov_type`.
- **Hausman test (RE)**: `hausman_statistic` / `hausman_p_value` / `hausman_df` are the regression-based (auxiliary regression) version, equivalent to `plm::phtest(method = "aux")` with `effect = "individual"` (one-way) or `"twoways"` (when `REOptions.time` is set). The comparison is against FE with the same structure as the RE model.
    - The Wald test follows the `cov_type` of the RE fit: the default `"cluster"` gives the cluster-robust Hausman test, and `"classical"` gives the classical version, which assumes RE is fully efficient under the null.
    - If the robust covariance is structurally singular (for example, no more clusters than the auxiliary slopes), `fit()` raises an error instead of returning `None`. The three values are `None` only when no slope coefficient can be tested.
    - Regressors that cannot be identified in the auxiliary regression are left out and `hausman_df` counts the rest: a regressor that varies only over time on a balanced panel, or one whose entity mean is also a regressor (correlated random effects).
    - For those regressors `plm::phtest(method = "aux")` (plm 2.6.7) is not equivalent: in a check with a time-invariant regressor its statistic changed with the order of the regressors in the formula, whereas the statistic here does not depend on the column order.

    The definition and the exact rules are in [Definitions: RE](../methods/re.md#hausman-test).
