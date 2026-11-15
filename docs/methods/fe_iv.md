# FEIV: definitions

**FEIV**: [Usage](../usage/fe_iv.md) · [Options and results](../api/fe_iv.md) · **Definitions**

This page defines what `FEIV` computes: fixed effects two-stage least squares (FE-2SLS, `estimator="2sls"`) and fixed effects GMM (`estimator="gmm"`), that is, the [FE](fe.md) within transformation followed by the [IV](iv.md) two-stage least squares or GMM estimator. The sections up to the first-stage regressions and the fixed effects define two-stage least squares; [GMM estimator](#gmm-estimator-estimatorgmm) gives what is different for GMM. Where a definition is the same as in `FE` or `IV`, this page links to it and gives only what is different: the degrees of freedom that the absorbed fixed effects take away, the leverage of HC2 and HC3, the diagnostics, the fit statistics, the first-stage regressions, the recovery of the fixed effects and the differences from `fixest`. For the arguments and return values see the [FEIV API reference](../api/fe_iv.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md).

## Model and estimator

The structural equation is, with \(D\) columns in `effects` as in [FE](fe.md#model-and-estimator),

\[
y_{it} = x_{1,it}'\beta_1 + x_{2,it}'\beta_2 + \sum_{d=1}^{D} \alpha^{(d)}_{g_d(it)} + \varepsilon_{it}
\]

where \(x_1\) are the exogenous regressors (`x_exog`, \(k_1\) columns) and \(x_2\) the endogenous regressors (`x_endog`, \(k_2\) columns), \(k = k_1 + k_2\), and the excluded instruments (`instruments`) are \(z\) (\(m \ge k_2\) columns). There is no intercept. \(n\) is the number of observations.

Every variable (\(y\), \(x_1\), \(x_2\) and \(z\)) is transformed once with the within transformation of [FE](fe.md#model-and-estimator) (closed form for one-way effects and for a balanced two-way panel, alternating projections otherwise, controlled by `tol` and `max_iter`); a tilde marks a transformed variable. Then:

1. **First stage**, for each endogenous variable \(j\): the regression of \(\tilde x_{2j}\) on \([\tilde x_1, \tilde z]\), with the `cov_type` of the fit. The fitted values are \(\hat{\tilde x}_{2j}\), and \(\hat X = [\tilde x_1, \hat{\tilde x}_2]\).
2. **Second stage**: \(\hat\beta = (\hat X'\hat X)^{-1}\hat X'\tilde y\).

This equals two-stage least squares with a dummy variable for every group of every column of `effects` (Frisch-Waugh-Lovell). The residuals are the **structural residuals** \(\hat\varepsilon = \tilde y - \tilde X\hat\beta\) of the within-transformed data, computed with the actual endogenous variables \(\tilde X = [\tilde x_1, \tilde x_2]\) rather than the fitted values, as in [IV](iv.md#model-and-notation). `residuals` returns them.

A group with a single observation in any column of `effects`, a variable with no variation left after the transformation, an outcome with no variation left, and too few instruments (\(m < k_2\)) are rejected rather than handled; see [Validation and errors](../guide/validation.md).

## Degrees of freedom

As in [FE](fe.md#degrees-of-freedom). With \(n_{\text{abs}} = \sum_{d=1}^{D} G_d - (D-1)\) the number of absorbed fixed-effect parameters (\(G_d\) is the value of column \(d\) in `n_groups`),

\[
\mathrm{df}_{\text{model}} = k + n_{\text{abs}}, \qquad
\mathrm{df}_{\text{resid}} = n - \mathrm{df}_{\text{model}}
\]

`fit()` raises a `ValidationError` when \(n \le k + n_{\text{abs}}\) (the structural equation) or \(n \le l + n_{\text{abs}}\), with \(l = k_1 + m\) (the first stage). \(n_{\text{abs}}\) is counted as `fixest` counts it: disconnected sets of groups and effects nested in one another do not reduce it, although the rank of the dummy-variable matrix is then smaller.

## Covariance of the coefficients: `cov_type`

The same estimators as in [FE](fe.md#covariance-of-the-coefficients-cov_type) (`"classical"`, `"hc1"` to `"hc3"`, `"cluster"`, `"dk"`; the default is `"cluster"` on the first column of `effects`, and `"hc0"` is not available), computed as in [IV](iv.md#covariance-of-the-coefficients-cov_type) for 2SLS: the regressor matrix of the sandwich is \(\hat X\), the bread is \((\hat X'\hat X)^{-1}\), and the residuals are the structural residuals \(\hat\varepsilon\). The small-sample factors, the clusters, the Driscoll-Kraay kernel and its bandwidth are those of FE, with \(\mathrm{df}_{\text{resid}}\) and \(\mathrm{df}_{\text{model}}\) from above. Below, \(G\) is the number of clusters and \(T_d\) the number of periods of `dk_time`.

- `"hc2"`, `"hc3"`: the leverage is \(h_{it} = h^{\text{dummy}}_{it} + \hat h_{it}\), the leverage of the effects alone plus the leverage \(\hat X(\hat X'\hat X)^{-1}\hat X'\) of the second stage. It equals the diagonal of the hat matrix of the second stage with a dummy variable for every group. The same limit on the number of groups as in FE applies.
- `"cluster"`: with an effect nested in the clusters, the small-sample correction treats it as not estimated, as in [FE](fe.md#cluster-robust-small-sample-factor).

## Test distributions and confidence intervals

As in [FE](fe.md#test-distributions-and-confidence-intervals): the t distribution for every `cov_type`, with \(\mathrm{df}_{\text{resid}}\) (`"classical"`, `"hc1"` to `"hc3"`), \(G-1\) (`"cluster"`) or \(T_d-1\) (`"dk"`) degrees of freedom, returned as `stat_df`. `stat_dist` is `"t"`.

## Overall test and fit statistics

### Overall test

`wald_statistic` is the Wald test that all \(k\) slope coefficients are zero with the covariance chosen by `cov_type`, divided by \(k\), as in [IV](iv.md#overall-test): \(F = \hat\beta'\hat\Sigma^{-1}\hat\beta / k \sim F(k, \mathrm{df})\), with the same degrees of freedom as the t tests (`wald_df_denom` equals `stat_df`). `wald_dist` is `"f"`.

### Fit statistics

`r_squared_within`, `r_squared_between` and `r_squared_overall` are defined as in [FE](fe.md#fit-statistics), with \(\hat\beta\) the structural coefficients and \(\tilde X\), \(X\) the actual endogenous variables, so the residuals are the structural residuals. Because those are not least-squares residuals, an R² can be low or negative, and it is not the share of the variance explained by the instruments. The log-likelihood, AIC and BIC are not reported.

## Diagnostics

All the diagnostics use the within-transformed variables and count the absorbed fixed-effect parameters in their degrees of freedom. The statistics follow the definitions of [IV](iv.md#diagnostics), with the changes below.

### Degrees of freedom of the diagnostics

The number of absorbed parameters \(n_{\text{abs}}\) that a diagnostic subtracts depends on the diagnostic and, for the covariance-based ones, on `cov_type`:

| Diagnostic | Absorbed parameters subtracted |
|---|---|
| Sargan, weak-instrument partial F, Cragg-Donald | \(n_{\text{abs}}\) (all the fixed effects) |
| Kleibergen-Paap, effective F, with `"classical"`, `"hc1"` to `"hc3"`, `"dk"` | \(n_{\text{abs}}\) |
| Kleibergen-Paap, effective F, with `"cluster"` | the \(K\) of the cluster small-sample correction of [FE](fe.md#cluster-robust-small-sample-factor) for the first stage, less its \(l\) slope coefficients; effects nested in the clusters are not counted |
| Wu-Hausman | the degrees of freedom of its own auxiliary fixed-effects regression (below) |

### Weak instruments: `weak_instrument_f_statistics`

For each endogenous variable, the partial F test that the coefficients of the \(m\) excluded instruments are jointly zero in its first stage, after partialling out \(\tilde x_1\), always under homoskedasticity (it does not depend on `cov_type`). \(\mathrm{SSR}_u\) is the residual sum of squares of the first-stage regression of \(\tilde x_{2j}\) on \([\tilde x_1, \tilde z]\), and \(\mathrm{SSR}_r\) that of the regression on \(\tilde x_1\) alone:

\[
F_j = \frac{(\mathrm{SSR}_r - \mathrm{SSR}_u)/m}{\mathrm{SSR}_u/(n-l-n_{\text{abs}})} \sim F(m, n-l-n_{\text{abs}})
\]

`weak_instrument_f_df_num` is \(m\) and `weak_instrument_f_df_denom` is \(n-l-n_{\text{abs}}\). Without the absorbed parameters in the denominator the statistic would be too large.

### Weak instruments, post-estimation

`cragg_donald()`, `kleibergen_paap()` and `effective_f()` are defined as in [IV](iv.md#weak-instruments-post-estimation), on the first-stage design \([\tilde x_1, \tilde z]\), with \(n_{\text{abs}}\) from the table above, and are never computed by `fit()`. Each statistic equals the corresponding [IV](iv.md#weak-instruments-post-estimation) statistic with a dummy variable for every group as an exogenous regressor, so the Stock-Yogo and effective F critical values are applied as tabulated. This package has not checked those critical values against simulations with fixed effects.

- `kleibergen_paap()` and `effective_f()` follow `cov_type` and its options (`cluster`, `dk_time`, `dk_bandwidth`). With `"dk"` the scores are summed over each period of `dk_time` and the Bartlett kernel is applied to those sums, as in the covariance of the coefficients, with the small-sample factor of the Driscoll-Kraay covariance. With `"hc2"` and `"hc3"` the scores are weighted by the leverage that includes the absorbed fixed effects, as above.
- With `"dk"`, `kleibergen_paap()` and `effective_f()` need more periods than \(r = m - k_2 + 1\), that is \(T_d > r\). With \(T_d \le r\) the period scores do not span the \(r\) dimensions that the statistics need: the rk LM statistic is a constant that does not depend on the data (with \(T_d = r\) and a bandwidth of 0 it equals \(T_d\)), and the period scores of the rk Wald F, which sum to zero, have rank at most \(T_d - 1 < r\), so their covariance is singular. Neither Kleibergen-Paap statistic is defined, and the method raises a `ComputationError`.
- A statistic that cannot be computed raises `ComputationError`; see [When a diagnostic cannot be computed](iv.md#when-a-diagnostic-cannot-be-computed).

### Overidentification: `overid_statistic`

The Sargan statistic of the within-transformed data, chi-squared with \(m - k_2\) degrees of freedom (`overid_df`); `None` when just identified:

\[
S = \frac{\hat\varepsilon' \tilde Z(\tilde Z'\tilde Z)^{-1}\tilde Z'\hat\varepsilon}{\hat\sigma^2}, \qquad
\hat\sigma^2 = \frac{\hat\varepsilon'\hat\varepsilon}{n - n_{\text{abs}}}, \qquad \tilde Z = [\tilde x_1, \tilde z]
\]

It assumes homoskedasticity and does not depend on `cov_type`. The variance uses \(n - n_{\text{abs}}\), not \(n\): the within residuals have an expected sum of squares of about \(\sigma^2(n - n_{\text{abs}})\) (the estimated slopes are not subtracted, as in [IV](iv.md#overidentification-overid_statistic)), and dividing by \(n\) makes the statistic larger by the factor \(n/(n - n_{\text{abs}})\), which can reject too often when there are few periods. The numerator is the same as in the Sargan statistic of 2SLS with a dummy variable for every group (Frisch-Waugh-Lovell); the variance estimate uses the degrees of freedom that the absorbed fixed effects use up, whereas [IV](iv.md#overidentification-overid_statistic) divides by \(n\). FEIV currently has no overidentification test that is robust to heteroskedasticity or clustering.

### Endogeneity: `wu_hausman_statistic`

The regression-based version of [IV](iv.md#endogeneity-wu_hausman_statistic), as a fixed-effects regression: the first-stage residuals \(\hat v\) (which are within-transformed already) are added, \(\tilde y \sim [\tilde x_1, \tilde x_2, \hat v]\), with the covariance given by the `cov_type` of the fit, and the F test that the coefficients of \(\hat v\) are jointly zero is reported, \(F(k_2, \mathrm{df})\) with \(\mathrm{df}\) the inference degrees of freedom of that regression: \(G-1\) (`"cluster"`), \(T_d-1\) (`"dk"`), and \(n - k - k_2 - n_{\text{abs}}\) otherwise. `wu_hausman_df_num` and `wu_hausman_df_denom` are these degrees of freedom. The statistic is `None` when the auxiliary regression cannot be estimated: too few observations, a perfect fit, a singular design, or a covariance that cannot be inverted.

Unlike the structural equation, the auxiliary regression does not need more clusters (or periods) than its slope coefficients, because only the \(k_2 \times k_2\) block of \(\hat v\) enters the test.

## First stage: `first_stage()`

Returns, per endogenous variable, the fixed-effects regression of `x_endog[i]` on `x_exog` and `instruments` with the same `effects` and `cov_type` as the structural equation, as an [`FEResults`](../api/fe.md). Its overall F statistic (`f_statistic`, `f_p_value`, `f_df_num`, `f_df_denom`) is `None`, since it is not computed for a first stage; its standard errors are computed even when the number of clusters (or periods) is at most the number of its slope coefficients. It is not the partial F statistic above. Its `fixed_effects()` returns the fixed effects of the first stage.

## Fixed effects: `fixed_effects()`

The effects are recovered after estimation as in [FE](fe.md#fixed-effects-fixed_effects), with the structural equation: for one-way effects \(\hat\alpha_i = \bar y_{i\cdot} - \bar x_{i\cdot}'\hat\beta\), where \(x\) holds the actual endogenous variables (not their fitted values). The reference groups are those of FE. The method re-reads the data passed to `fit()` and estimates the model again, so it costs about as much as `fit()`.

## Differences from `fixest`

Measured with `fixest` 0.14.2 (`feols` with an instrumental-variable part):

- The coefficients, standard errors (classical, HC1, cluster) and the Wu-Hausman auxiliary regression agree. `fitstat(model, "wh")` is the Wald F statistic that the coefficients of \(\hat v\) in the auxiliary regression are jointly zero under iid errors (the squared t statistic when there is one endogenous variable); the other `cov_type` values are compared with the same auxiliary regression under the corresponding `vcov`.
- `fitstat(model, "sargan")` corresponds to the variance \(\hat\varepsilon'\hat\varepsilon/n\) (checked numerically: its value is larger than this package's by the factor below). This package uses \(n - n_{\text{abs}}\) (see above), which accounts for the degrees of freedom used by the fixed effects, so the two statistics differ by the factor \(n/(n - n_{\text{abs}})\); the difference matters most when there are few periods.
- `fitstat(model, "wr2")` is based on the residuals of the second-stage regression on the fitted values, whereas `r_squared_within` uses the structural residuals, so they differ.
- When the groups of the effects form several disconnected sets, the number of absorbed parameters in the partial F is one larger than the rank of the dummy-variable matrix, as in `fixest`. This was checked on one example and is not part of the automated tests.

## GMM estimator (`estimator="gmm"`)

With `estimator="gmm"` the within-transformed variables enter the generalized method of moments instead of two-stage least squares: the [IV](iv.md) GMM estimator after the [FE](fe.md) within transformation. The sections above define the model, the degrees of freedom, the diagnostics of two-stage least squares and the fixed effects; this part gives what is different for GMM: the moment conditions in the within space, the weight matrix, the covariance of the coefficients, the Hansen J and C statistics and their small-sample treatment of the absorbed effects, and the limits of the supported weights and diagnostics. GMM has no first-stage regressions, weak-instrument diagnostics, Wu-Hausman test or `fixed_effects()`; the exogeneity of a regressor is tested by `c_stat()`.

### Model and weight matrix

The structural equation and the notation are those of [the model above](#model-and-estimator): \(x_1\) (`x_exog`, \(k_1\) columns) and \(x_2\) (`x_endog`, \(k_2\) columns), \(k = k_1 + k_2\), the excluded instruments \(z\) (`instruments`, \(m \ge k_2\) columns), and \(D\) columns in `effects`. There is no intercept. \(n\) is the number of observations.

Every variable is transformed once with the within transformation of [FE](fe.md#model-and-estimator) (a tilde marks a transformed variable). With the regressors \(\tilde X = [\tilde x_1, \tilde x_2]\) and the instruments \(\tilde Z = [\tilde x_1, \tilde z]\) (\(l = k_1 + m\) columns), with \(\tilde w_{it}'\) the row of \(\tilde Z\) for observation \((i,t)\), the moment conditions are \(E[\tilde w_{it}\varepsilon_{it}] = 0\), and for a weight matrix \(W\)

\[
\hat\beta(W) = (\tilde X'\tilde Z\, W\, \tilde Z'\tilde X)^{-1}\, \tilde X'\tilde Z\, W\, \tilde Z'\tilde y
\]

No first-stage regression is run. `gmm_type` selects how \(W\) is obtained:

| `gmm_type` | Procedure |
|---|---|
| `"one_step"` | \(W = (\tilde Z'\tilde Z)^{-1}\) only. The point estimate is the two-stage least squares estimate of FEIV with `estimator="2sls"`. |
| `"two_step"` (default) | Estimate with \(W_0 = (\tilde Z'\tilde Z)^{-1}\), build \(S\) from its residuals, and re-estimate with \(W = S^{-1}\). |
| `"iterated"` | Repeat the second step, rebuilding \(S\) from the latest residuals, until every coefficient satisfies \(|\hat\beta_j - \hat\beta_j^{\text{prev}}| \le \max\big(\texttt{gmm\_tol}\cdot|\hat\beta_j^{\text{prev}}|,\ 10^{-8}\big)\), or until `gmm_max_iter` estimations (counting the first) have been run. |

\(S\) is the estimated covariance of the moments \(\sum_{it}\tilde w_{it}\hat\varepsilon_{it}\), chosen by `gmm_weight_type`, with \(\hat\varepsilon\) the structural residuals of the previous estimate:

| `gmm_weight_type` | \(S\) |
|---|---|
| `"classical"` | \(\hat\sigma^2\tilde Z'\tilde Z\), with \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon / n\). With `"two_step"` this reproduces the two-stage least squares estimate. |
| `"robust"` | \(\sum_{it} \hat\varepsilon_{it}^2 \tilde w_{it}\tilde w_{it}'\) |
| `"cluster"` (default) | \(\sum_g \big(\sum_{(i,t)\in g}\tilde w_{it}\hat\varepsilon_{it}\big)\big(\sum_{(i,t)\in g}\tilde w_{it}\hat\varepsilon_{it}\big)'\), over the \(G\) groups of `cluster` (by default the first column of `effects`) |

There is no `"hac"` weight. The weight type only affects the point estimate; the reported standard errors follow `cov_type`, which is chosen independently. `gmm_type="one_step"` does not use a weight type. A positive scalar multiple of \(S\) does not change \(\hat\beta\), so no small-sample factor is applied to the \(S\) of the point estimate.

The within transformation makes the errors of an entity sum to zero over its periods, so the transformed errors are negatively correlated over time and more so the shorter the panel. The `"robust"` weight, which treats them as uncorrelated, is therefore a poor estimate of the covariance of the moments when the number of periods is small, and `"cluster"` (the default) does not rely on that assumption.

`converged` and `n_iter` report the outcome: `"one_step"` and `"two_step"` are always converged with 1 and 2 estimations, and only `"iterated"` can fail to converge. In that case `fit()` raises a `ComputationError` unless `raise_on_non_convergence=False`.

### Clusters needed for the cluster weight

For `gmm_weight_type="cluster"` the weight matrix \(S\) is \(l \times l\) and has rank at most \(G\) (at most \(G-1\) when just identified, because the moments then sum to zero), so `fit()` raises a `ValidationError` unless \(G \ge l\) when overidentified, or \(G > l\) when just identified.

### Degrees of freedom of the GMM fit

As in [the definition above](#degrees-of-freedom): with \(n_{\text{abs}}\) the number of absorbed fixed-effect parameters, \(\mathrm{df}_{\text{model}} = k + n_{\text{abs}}\) and \(\mathrm{df}_{\text{resid}} = n - \mathrm{df}_{\text{model}}\). `fit()` raises a `ValidationError` when \(n \le k + n_{\text{abs}}\) or \(n \le l + n_{\text{abs}}\).

### GMM covariance of the coefficients

The covariance is always the general sandwich

\[
\widehat{\mathrm{Var}}(\hat\beta) = B^{-1}\big(\tilde X'\tilde Z\,W\,\hat\Omega\,W\,\tilde Z'\tilde X\big)B^{-1}, \qquad B = \tilde X'\tilde Z\,W\,\tilde Z'\tilde X
\]

where \(W = S^{-1}\) is the weight matrix actually used for the point estimate (for `"one_step"`, \((\tilde Z'\tilde Z)^{-1}\)) and \(\hat\Omega\) is the covariance of the moments \(\tilde w_{it}\hat\varepsilon_{it}\) for `cov_type`, with the small-sample factors of [FE](fe.md#covariance-of-the-coefficients-cov_type). The simplification to \(B^{-1}\) for an efficient weight matrix is not taken, so `gmm_weight_type` and `cov_type` can differ freely. The default is `"cluster"` on the first column of `effects`, and `"hc0"` is not available.

| `cov_type` | \(\hat\Omega\) |
|---|---|
| `"classical"` | \(\hat\sigma^2\tilde Z'\tilde Z\), with \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon/\mathrm{df}_{\text{resid}}\) |
| `"hc1"` | \(\dfrac{n}{\mathrm{df}_{\text{resid}}}\sum_{it}\hat\varepsilon_{it}^2\tilde w_{it}\tilde w_{it}'\) |
| `"hc2"`, `"hc3"` | \(\sum_{it} c_{it}\hat\varepsilon_{it}^2\tilde w_{it}\tilde w_{it}'\) with \(c_{it} = 1/(1-h_{it})\) or \(1/(1-h_{it})^2\) |
| `"cluster"` | the cluster form of \(S\) above, multiplied by the factor \(f\) of [FE](fe.md#cluster-robust-small-sample-factor), \(f = \frac{G}{G-1}\cdot\frac{n-1}{n-K}\), where \(K\) is the effective number of parameters defined there |
| `"dk"` | the Driscoll-Kraay form of [FE](fe.md#driscoll-kraay-dk) on the period sums of \(\tilde w_{it}\hat\varepsilon_{it}\), with the factor \(\frac{T_d}{T_d-1}\cdot\frac{n-1}{n-\mathrm{df}_{\text{model}}}\) |

For `"hc2"` and `"hc3"` the leverage is \(h_{it} = h^{\text{dummy}}_{it} + \tilde w_{it}'(\tilde Z'\tilde Z)^{-1}\tilde w_{it}\), the leverage of the effects alone plus the leverage of the instruments. It equals the diagonal of the hat matrix of the instruments with a dummy variable for every group, the same construction as in [the covariance above](#covariance-of-the-coefficients-cov_type), with the same limit as in FE: the groups of all columns of `effects` except the one with the most groups may number at most 3,000 in total (an error is raised above that; use another `cov_type`). The cluster correction \(K\), the Driscoll-Kraay bandwidth and the time periods are those of FE. The clusters of `cov_type="cluster"` and of the weight are both set by `cluster`.

The sandwich is symmetrized, so `cov_params` is symmetric up to rounding error.

With `cov_type="cluster"`, `fit()` raises a `ValidationError` when there are fewer than two clusters, and also unless there are more clusters than slope coefficients (`"dk"`: more time periods than slope coefficients), because the overall Wald test is then not defined.

Two things are not compared with an independent implementation. `"hc2"` and `"hc3"` with GMM are checked only against a direct computation of the formulas above (the instrument-based leverage is this package's construction; the `"hc2"` and `"hc3"` of `plm` 2.6.7 for within IV do not include the leverage of the fixed effects and so differ from this definition). And the cluster factor uses the convention for \(K\) of `fixest` 0.14.2 (see [FE](fe.md#cluster-robust-small-sample-factor)), whereas [IV](iv.md#covariance-of-the-coefficients-cov_type) uses \(K = k\), and `linearmodels` 7.1 applies no \(G/(G-1)\) factor in `IVGMM`. The cluster factor is compared with `fixest` through the reduction to FE-2SLS, but not for an overidentified robust or cluster weight.

### GMM test distributions and confidence intervals

GMM is an asymptotic estimator with no finite-sample degrees of freedom: the coefficient statistics are standard normal (`test_stats` are z statistics, `stat_dist` is `"normal"`, `stat_df` is `None`) and the overall test is chi-squared. This differs from [the test distributions above](#test-distributions-and-confidence-intervals), which uses the t and F distributions.

### GMM overall test and fit statistics

#### GMM overall test

`wald_statistic` is the Wald statistic that all \(k\) slope coefficients are zero, \(\hat\beta'\hat\Sigma^{-1}\hat\beta\), with \(\hat\Sigma\) the covariance of the coefficients for `cov_type`, not divided by \(k\). It is chi-squared with \(k\) degrees of freedom (`wald_dist` is `"chi2"`, `wald_df_num` is \(k\), `wald_df_denom` is `None`).

#### GMM fit statistics

`r_squared_within`, `r_squared_between` and `r_squared_overall` are defined as in [the fit statistics above](#fit-statistics), from the structural residuals, so an R² can be low or negative. The log-likelihood, AIC and BIC are not reported.

### GMM diagnostics

All the GMM diagnostics use the within-transformed variables and count the absorbed fixed-effect parameters.

#### Hansen J statistic

Hansen's J, computed by `fit()`: \(J = (\tilde Z'\hat\varepsilon)'S^{-1}(\tilde Z'\hat\varepsilon)\), with \(\hat\varepsilon\) the final structural residuals and \(S\) the covariance of the moments of the weight actually used for the point estimate (so J follows `gmm_weight_type`; for `"one_step"` it is the classical \(S\)). It is chi-squared with \(m - k_2\) degrees of freedom (`overid_df`), and `None` when just identified.

\(S\) is rescaled for the absorbed effects only where it depends on \(\hat\sigma^2\): for `"classical"` and `"one_step"`, \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon/(n - n_{\text{abs}})\) (that is, \(S\) is multiplied by \(n/(n - n_{\text{abs}})\)). The within residuals have an expected sum of squares of about \(\sigma^2(n - n_{\text{abs}})\), and dividing by \(n\) would make J too large when there are few periods. With this rescaling the J of a `"classical"` weight equals the Sargan statistic of [2SLS](#overidentification-overid_statistic).

The `"robust"` and `"cluster"` weights are not rescaled, because there is no established correction for them. For `"cluster"` no rescaling is needed when every column of `effects` is nested in the clusters (for example one-way effects clustered by the entity, or by a coarser column): the transformed moments of a cluster then equal \(\sum_{(i,t)\in g}\tilde w_{it}\varepsilon_{it}\), the moments of the untransformed errors. With an effect that crosses the clusters, such as a time effect in a two-way model, this holds only approximately (the difference vanishes as the number of clusters grows), and it does not hold for clusters that cross the entities, such as the periods. For `"robust"`, \(\sum \hat\varepsilon^2\tilde w\tilde w'\) is biased downwards when there are few periods even for homoskedastic errors, because the within residuals have the variance \(\sigma^2(1 - 1/T)\) (in a balanced one-way model; \(\sigma^2(1 - 1/T_i)\) for entity \(i\) in an unbalanced one, and \(\sigma^2(1 - 1/T - 1/N + 1/n)\) in a balanced two-way model), so J over-rejects there, and a rescaling by \(n/(n - n_{\text{abs}})\) would not fix it under heteroskedasticity. Prefer the cluster weight when the panel is short.

#### Exogeneity and instrument validity, post-estimation: `c_stat()`

`c_stat()` is a method, not a field: it is never computed by `fit()`. It re-reads the variables from the data passed to `fit()`, transforms them again, and tests the validity of a set of moment conditions with the C statistic (difference-in-Hansen test) exactly as [IV](iv.md#exogeneity-and-instrument-validity-post-estimation-c_stat) does, with the within-transformed \(\tilde Z_{\mathrm{full}} = [\tilde x_1,\ \tilde x_{2B},\ \tilde z]\) and \(\tilde Z_c = [\tilde x_1,\ \tilde z \setminus \tilde z_B]\), which leaves out the tested moment conditions, in place of \(Z_{\mathrm{full}}\) and \(Z_c\). Here \(\tilde x_{2B}\) are the \(k_{2B}\) tested endogenous variables and \(\tilde z_B\) the \(m_B\) tested instruments:

\[
C = J_e - J_c \sim \chi^2(k_{2B} + m_B)
\]

where \(J_e\) uses \(S_e\), the covariance of the moments of \(\tilde Z_{\mathrm{full}}\) built from the first-step residuals, and \(J_c\) uses the sub-matrix \(S_c\) of \(S_e\) that belongs to \(\tilde Z_c\). Both use the same \(S_e\), so \(C\) is never negative, and \(J_c\) is exactly 0 when \(\tilde Z_c\) is just identified.

The kind of \(S_e\) follows the fit: `gmm_weight_type` (with `cluster`) for `gmm_type` `"two_step"` and `"iterated"`, and the classical one for `"one_step"`, which has no weight matrix. It does not depend on `cov_type`, `gmm_max_iter` or `gmm_tol`, and a model fitted with `gmm_type="one_step"` or `"iterated"` gets the two-step statistic. As for the Hansen J statistic, the classical \(S_e\) uses \(\hat\sigma^2 = \hat\varepsilon^{(1)\prime}\hat\varepsilon^{(1)}/(n - n_{\text{abs}})\), so \(C\) is \((n - n_{\text{abs}})/n\) times the value without the correction, and the robust and cluster weights are not adjusted. As for J, the cluster weight needs no adjustment when every column of `effects` is nested in the clusters, only approximately when an effect crosses the clusters, and not at all for clusters that cross the entities (for example the periods), and with a robust weight on a short panel \(C\) over-rejects. \(J_e\) of a test of instruments only equals the Hansen J of a fit with the same weight, and only for `"two_step"` (and `"one_step"` with the classical weight): for `"iterated"` the fit's J uses the weight of the last iteration, whereas \(J_e\) always uses the first-step residuals.

The tested moment conditions must leave the model identified: at least as many excluded instruments as endogenous variables must remain. The statistic is not defined when a matrix it needs is singular: the instruments are collinear, the structural equation fits perfectly, or \(S_e\) has too little rank. With the cluster weight that last case occurs with fewer clusters than the \(k_1 + k_{2B} + m\) columns of \(\tilde Z_{\mathrm{full}}\), which can happen although the fit itself succeeds, because the weight of the fit only needs \(G \ge l\) clusters. `fit()` reports too few clusters for its weight as a `ValidationError`, whereas `c_stat()` can only detect it while computing, and raises a `ComputationError`.
