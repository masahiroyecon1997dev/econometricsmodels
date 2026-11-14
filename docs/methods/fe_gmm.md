# FEGMM: definitions

**FEGMM**: [Usage](../usage/fe_gmm.md) · [Options and results](../api/fe_gmm.md) · **Definitions**

This page defines what `FEGMM` computes: fixed effects generalized method of moments, that is, the [FE](fe.md) within transformation followed by the [IV](iv.md) GMM estimator. Where a definition is the same as in `FE`, `FEIV` or `IV`, this page links to it and gives only what is different: the moment conditions in the within space, the weight matrix, the covariance of the coefficients, the Hansen J and C statistics and their small-sample treatment of the absorbed effects, and the limits of the supported weights and diagnostics. For the arguments and return values see the [FEGMM API reference](../api/fe_gmm.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md).

## Model and estimator

The structural equation and the notation are those of [FEIV](fe_iv.md#model-and-estimator): \(x_1\) (`x_exog`, \(k_1\) columns) and \(x_2\) (`x_endog`, \(k_2\) columns), \(k = k_1 + k_2\), the excluded instruments \(z\) (`instruments`, \(m \ge k_2\) columns), and \(D\) columns in `effects`. There is no intercept. \(n\) is the number of observations.

Every variable is transformed once with the within transformation of [FE](fe.md#model-and-estimator) (a tilde marks a transformed variable). With the regressors \(\tilde X = [\tilde x_1, \tilde x_2]\) and the instruments \(\tilde Z = [\tilde x_1, \tilde z]\) (\(l = k_1 + m\) columns), the moment conditions are \(E[\tilde z_{it}\varepsilon_{it}] = 0\), and for a weight matrix \(W\)

\[
\hat\beta(W) = (\tilde X'\tilde Z\, W\, \tilde Z'\tilde X)^{-1}\, \tilde X'\tilde Z\, W\, \tilde Z'\tilde y
\]

No first-stage regression is run. `gmm_type` selects how \(W\) is obtained:

| `gmm_type` | Procedure |
|---|---|
| `"one_step"` | \(W = (\tilde Z'\tilde Z)^{-1}\) only. The point estimate is the two-stage least squares estimate of [FEIV](fe_iv.md). |
| `"two_step"` (default) | Estimate with \(W_0 = (\tilde Z'\tilde Z)^{-1}\), build \(S\) from its residuals, and re-estimate with \(W = S^{-1}\). |
| `"iterated"` | Repeat the second step, rebuilding \(S\) from the latest residuals, until every coefficient satisfies \(|\hat\beta_j - \hat\beta_j^{\text{prev}}| \le \max\big(\texttt{gmm\_tol}\cdot|\hat\beta_j^{\text{prev}}|,\ 10^{-8}\big)\), or until `gmm_max_iter` estimations (counting the first) have been run. |

\(S\) is the estimated covariance of the moments \(\sum_{it}\tilde z_{it}\hat\varepsilon_{it}\), chosen by `gmm_weight_type`, with \(\hat\varepsilon\) the structural residuals of the previous estimate:

| `gmm_weight_type` | \(S\) |
|---|---|
| `"classical"` | \(\hat\sigma^2\tilde Z'\tilde Z\), with \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon / n\). With `"two_step"` this reproduces the two-stage least squares estimate. |
| `"robust"` | \(\sum_{it} \hat\varepsilon_{it}^2 \tilde z_{it}\tilde z_{it}'\) |
| `"cluster"` (default) | \(\sum_g \big(\sum_{(i,t)\in g}\tilde z_{it}\hat\varepsilon_{it}\big)\big(\sum_{(i,t)\in g}\tilde z_{it}\hat\varepsilon_{it}\big)'\), over the \(G\) groups of `cluster` (by default the first column of `effects`) |

There is no `"hac"` weight. The weight type only affects the point estimate; the reported standard errors follow `cov_type`, which is chosen independently. `gmm_type="one_step"` does not use a weight type. A positive scalar multiple of \(S\) does not change \(\hat\beta\), so no small-sample factor is applied to the \(S\) of the point estimate.

The within transformation makes the errors of an entity sum to zero over its periods, so the transformed errors are negatively correlated over time and more so the shorter the panel. The `"robust"` weight, which treats them as uncorrelated, is therefore a poor estimate of the covariance of the moments when the number of periods is small, and `"cluster"` (the default) does not rely on that assumption.

`converged` and `n_iter` report the outcome: `"one_step"` and `"two_step"` are always converged with 1 and 2 estimations, and only `"iterated"` can fail to converge. In that case `fit()` raises an error unless `raise_on_non_convergence=False`.

### Clusters

For `gmm_weight_type="cluster"` the weight matrix \(S\) is \(l \times l\) and has rank at most \(G\) (at most \(G-1\) when just identified, because the moments then sum to zero), so `fit()` raises a `ValidationError` unless \(G \ge l\) when overidentified, or \(G > l\) when just identified.

## Degrees of freedom

As in [FEIV](fe_iv.md#degrees-of-freedom): with \(n_{\text{abs}}\) the number of absorbed fixed-effect parameters, \(\mathrm{df}_{\text{model}} = k + n_{\text{abs}}\) and \(\mathrm{df}_{\text{resid}} = n - \mathrm{df}_{\text{model}}\). `fit()` raises a `ValidationError` when \(n \le k + n_{\text{abs}}\) or \(n \le l + n_{\text{abs}}\).

## Covariance of the coefficients: `cov_type`

The covariance is always the general sandwich

\[
\widehat{\mathrm{Var}}(\hat\beta) = B^{-1}\big(\tilde X'\tilde Z\,W\,\hat\Omega\,W\,\tilde Z'\tilde X\big)B^{-1}, \qquad B = \tilde X'\tilde Z\,W\,\tilde Z'\tilde X
\]

where \(W = S^{-1}\) is the weight matrix actually used for the point estimate (for `"one_step"`, \((\tilde Z'\tilde Z)^{-1}\)) and \(\hat\Omega\) is the covariance of the moments \(\tilde z_{it}\hat\varepsilon_{it}\) for `cov_type`, with the small-sample factors of [FE](fe.md#covariance-of-the-coefficients-cov_type). The simplification to \(B^{-1}\) for an efficient weight matrix is not taken, so `gmm_weight_type` and `cov_type` can differ freely. The default is `"cluster"` on the first column of `effects`, and `"hc0"` is not available.

| `cov_type` | \(\hat\Omega\) |
|---|---|
| `"classical"` | \(\hat\sigma^2\tilde Z'\tilde Z\), with \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon/\mathrm{df}_{\text{resid}}\) |
| `"hc1"` | \(\dfrac{n}{\mathrm{df}_{\text{resid}}}\sum_{it}\hat\varepsilon_{it}^2\tilde z_{it}\tilde z_{it}'\) |
| `"hc2"`, `"hc3"` | \(\sum_{it} c_{it}\hat\varepsilon_{it}^2\tilde z_{it}\tilde z_{it}'\) with \(c_{it} = 1/(1-h_{it})\) or \(1/(1-h_{it})^2\) |
| `"cluster"` | the cluster form of \(S\) above, multiplied by the factor \(f\) of [FE](fe.md#cluster-robust-small-sample-factor), \(f = \frac{G}{G-1}\cdot\frac{n-1}{n-K}\) |
| `"dk"` | the Driscoll-Kraay form of [FE](fe.md#driscoll-kraay-dk) on the period sums of \(\tilde z_{it}\hat\varepsilon_{it}\), with the factor \(\frac{T_d}{T_d-1}\cdot\frac{n-1}{n-\mathrm{df}_{\text{model}}}\) |

For `"hc2"` and `"hc3"` the leverage is \(h_{it} = h^{\text{dummy}}_{it} + \tilde z_{it}'(\tilde Z'\tilde Z)^{-1}\tilde z_{it}\), the leverage of the effects alone plus the leverage of the instruments. It equals the diagonal of the hat matrix of the instruments with a dummy variable for every group, the same construction as in [FEIV](fe_iv.md#covariance-of-the-coefficients-cov_type), with the same limit on the number of groups as in FE. The cluster correction \(K\), the Driscoll-Kraay bandwidth and the time periods are those of FE. The clusters of `cov_type="cluster"` and of the weight are both set by `cluster`.

The sandwich is symmetrized, so `cov_params` is symmetric up to rounding error.

`fit()` raises an error with fewer than two clusters with `cov_type="cluster"`, and also unless there are more clusters than slope coefficients (`"dk"`: more time periods than slope coefficients), because the overall Wald test is then not defined.

Two things are not checked against another implementation. `"hc2"` and `"hc3"` with GMM have no reference implementation (the instrument-based leverage is this package's construction). And the cluster factor follows `fixest`'s convention for \(K\), which differs from the convention of [IV](iv.md#gmm) and from `linearmodels`.

## Test distributions and confidence intervals

GMM is an asymptotic estimator with no finite-sample degrees of freedom: the coefficient statistics are standard normal (`test_stats` are z statistics, `stat_dist` is `"normal"`, `stat_df` is `None`) and the overall test is chi-squared. This differs from [FEIV](fe_iv.md#test-distributions-and-confidence-intervals), which uses the t and F distributions.

## Overall test and fit statistics

### Overall test

`wald_statistic` is the Wald statistic that all \(k\) slope coefficients are zero with the covariance chosen by `cov_type`, \(\hat\beta'\hat\Sigma^{-1}\hat\beta\), not divided by \(k\). It is chi-squared with \(k\) degrees of freedom (`wald_dist` is `"chi2"`, `wald_df_num` is \(k\), `wald_df_denom` is `None`).

### Fit statistics

`r_squared_within`, `r_squared_between` and `r_squared_overall` are defined as in [FEIV](fe_iv.md#fit-statistics), from the structural residuals, so an R² can be low or negative. The log-likelihood, AIC and BIC are not reported.

## Diagnostics

All the diagnostics use the within-transformed variables and count the absorbed fixed-effect parameters. There are no first-stage regressions, weak-instrument diagnostics, Wu-Hausman test or `fixed_effects()`; the exogeneity of a regressor is tested by `c_stat()`.

### Overidentification: `overid_statistic`

Hansen's J, computed by `fit()`: \(J = (\tilde Z'\hat\varepsilon)'S^{-1}(\tilde Z'\hat\varepsilon)\), with \(\hat\varepsilon\) the final structural residuals and \(S\) the covariance of the moments of the weight actually used for the point estimate (so J follows `gmm_weight_type`; for `"one_step"` it is the classical \(S\)). It is chi-squared with \(m - k_2\) degrees of freedom (`overid_df`), and `None` when just identified.

\(S\) is rescaled for the absorbed effects only where it depends on \(\hat\sigma^2\): for `"classical"` and `"one_step"`, \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon/(n - n_{\text{abs}})\) (that is, \(S\) is multiplied by \(n/(n - n_{\text{abs}})\)). The within residuals have an expected sum of squares of about \(\sigma^2(n - n_{\text{abs}})\), and dividing by \(n\) would make J too large when there are few periods. With this rescaling the J of a `"classical"` weight equals the Sargan statistic of [FEIV](fe_iv.md#overidentification-overid_statistic).

The `"robust"` and `"cluster"` weights are not rescaled, because there is no established correction for them. For `"cluster"` with clusters that contain the entities (the entities themselves, or any coarser column in which the entities are nested) none is needed: the transformed moments of a cluster equal the moments of its untransformed errors. With clusters that cross the entities, such as the periods, that does not hold. For `"robust"`, \(\sum \hat\varepsilon^2\tilde z\tilde z'\) is biased downwards when there are few periods even for homoskedastic errors, because the within residuals have the variance \(\sigma^2(1 - 1/T)\), so J over-rejects there, and a rescaling by \(n/(n - n_{\text{abs}})\) would not fix it under heteroskedasticity. Prefer the cluster weight when the panel is short.

### Exogeneity and instrument validity, post-estimation: `c_stat()`

`c_stat()` is a method, not a field: it is never computed by `fit()`. It re-reads the variables from the data passed to `fit()`, transforms them again, and tests the validity of a set of moment conditions with the C statistic (difference-in-Hansen test) exactly as [IV](iv.md#exogeneity-and-instrument-validity-post-estimation-c_stat) does, with the within-transformed \(\tilde Z_{\mathrm{full}} = [\tilde x_1,\ \tilde x_{2B},\ \tilde z]\) and \(\tilde Z_c\), which leaves out the tested moment conditions, in place of \(Z_{\mathrm{full}}\) and \(Z_c\):

\[
C = J_e - J_c \sim \chi^2(k_{2B} + m_B)
\]

where \(J_e\) uses \(S_e\), the covariance of the moments of \(\tilde Z_{\mathrm{full}}\) built from the first-step residuals, and \(J_c\) uses the sub-matrix \(S_c\) of \(S_e\) that belongs to \(\tilde Z_c\). Both use the same \(S_e\), so \(C\) is never negative, and \(J_c\) is exactly 0 when \(\tilde Z_c\) is just identified.

The kind of \(S_e\) follows the fit: `gmm_weight_type` (with `cluster`) for `gmm_type` `"two_step"` and `"iterated"`, and the classical one for `"one_step"`, which has no weight matrix. It does not depend on `cov_type`, `gmm_max_iter` or `gmm_tol`, and a model fitted with `gmm_type="one_step"` or `"iterated"` gets the two-step statistic. As for the Hansen J statistic, the classical \(S_e\) uses \(\hat\sigma^2 = \hat\varepsilon^{(1)\prime}\hat\varepsilon^{(1)}/(n - n_{\text{abs}})\), so \(C\) is \((n - n_{\text{abs}})/n\) times the value without the correction, and the robust and cluster weights are not adjusted. As for J, the cluster weight needs no adjustment when the clusters contain the entities, but not when they cross the entities (for example the periods), and with a robust weight on a short panel \(C\) over-rejects. \(J_e\) of a test of instruments only equals the Hansen J of a fit with the same weight.

The tested moment conditions must leave the model identified: at least as many excluded instruments as endogenous variables must remain. The statistic is not defined when a matrix it needs is singular: the instruments are collinear, the structural equation fits perfectly, or \(S_e\) has too little rank. With the cluster weight that last case occurs with fewer clusters than the \(k_1 + k_{2B} + m\) columns of \(\tilde Z_{\mathrm{full}}\), which can happen although the fit itself succeeds, because the weight of the fit only needs \(G \ge l\) clusters. `fit()` reports too few clusters for its weight as a `ValidationError`, whereas `c_stat()` can only detect it while computing, and raises a `ComputationError`.
