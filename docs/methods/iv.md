# IV: definitions

**IV**: [Usage](../usage/iv.md) · [Options and results](../api/iv.md) · **Definitions**

This page defines what `IV` computes: the 2SLS and GMM estimators, the covariance estimators behind `cov_type`, the test distributions, and the diagnostics (weak instruments, overidentification, endogeneity). For the arguments and return values see the [IV API reference](../api/iv.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md).

## Model and notation

The structural equation is \(y = X\beta + \varepsilon\) with \(n\) observations and \(k\) coefficients. \(X = [X_1, X_2]\) holds the exogenous regressors \(X_1\) (`x_exog`, plus the constant `const` when `include_intercept=True`) and the endogenous regressors \(X_2\) (`x_endog`). The instruments are \(Z = [X_1, Z_2]\), where \(Z_2\) are the excluded instruments (`instruments`), so \(Z\) has \(l\) columns. Identification needs the order condition \(l \ge k\), that is, at least as many excluded instruments as endogenous regressors; `fit()` checks it, but not the rank condition (see [What is not checked](../guide/validation.md#what-is-not-checked)).

Residuals are always the **structural residuals** \(\hat\varepsilon = y - X\hat\beta\), computed with the actual endogenous variables rather than their first-stage fitted values. `residuals` returns them.

## Estimators: `estimator`

### `"2sls"` (default)

With the projection \(P_Z = Z(Z'Z)^{-1}Z'\) and \(\hat X = P_Z X\) (the fitted values of the first stages),

\[
\hat\beta = (X'P_Z X)^{-1} X'P_Z y
\]

### `"gmm"`

The moment conditions are \(E[z_i \varepsilon_i] = 0\). For a weight matrix \(W\),

\[
\hat\beta(W) = (X'Z W Z'X)^{-1} X'Z W Z'y
\]

`gmm_type` selects how \(W\) is obtained:

| `gmm_type` | Procedure |
|---|---|
| `"one_step"` | \(W = (Z'Z)^{-1}\) only. The point estimate is the 2SLS estimate. |
| `"two_step"` (default) | Estimate with \(W_0 = (Z'Z)^{-1}\), build \(S\) from its residuals, and re-estimate with \(W = S^{-1}\). |
| `"iterated"` | Repeat the second step, rebuilding \(S\) from the latest residuals, until every coefficient changes by less than the tolerance or `gmm_max_iter` estimations (counting the first) have been run. |

\(S\) is the estimated covariance of the moments \(\sum_i z_i\hat\varepsilon_i\), chosen by `gmm_weight_type`:

| `gmm_weight_type` | \(S\) |
|---|---|
| `"classical"` | \(\hat\sigma^2 Z'Z\), with \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon / n\). With `"two_step"` this reproduces 2SLS, and Hansen's J equals the Sargan statistic. |
| `"robust"` | \(\sum_i \hat\varepsilon_i^2 z_i z_i'\) |
| `"cluster"` | \(\sum_g \big(\sum_{i\in g} z_i\hat\varepsilon_i\big)\big(\sum_{i\in g} z_i\hat\varepsilon_i\big)'\) |
| `"hac"` | The Newey-West (Bartlett) form of the OLS HAC estimator, applied to \(z_i\hat\varepsilon_i\) |

The weight type only affects the point estimate; the reported standard errors follow `cov_type`, which is chosen independently. `gmm_type="one_step"` does not use a weight type. A positive scalar multiple of \(S\) does not change \(\hat\beta\), so no small-sample factor is applied to \(S\).

`converged` and `n_iter` report the outcome: 2SLS has `n_iter = 1`, `"one_step"` and `"two_step"` are always converged with 1 and 2 estimations, and only `"iterated"` can fail to converge. In that case `fit()` raises an error unless `raise_on_non_convergence=False`.

## Covariance of the coefficients: `cov_type`

The covariance matrix of the coefficients, for either estimator, is available as `cov_params` on the results; the standard errors are the square roots of its diagonal.

### 2SLS

Every option is a sandwich \((\hat X'\hat X)^{-1}\hat S(\hat X'\hat X)^{-1}\) built on \(\hat X\) and the structural residuals, except `"classical"`. Note that \(\hat X'\hat X = X'P_ZX\).

| `cov_type` | Covariance |
|---|---|
| `"classical"` (default) | \(\hat\sigma^2 (X'P_ZX)^{-1}\), \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon/(n-k)\) |
| `"hc0"` to `"hc3"` | \(\hat S = \sum_i c_i \hat\varepsilon_i^2 \hat x_i\hat x_i'\), with the same \(c_i\) as in [OLS](ols.md#hc0-to-hc3-heteroskedasticity-robust) and the leverage \(h_{ii} = \hat x_i'(X'P_ZX)^{-1}\hat x_i\) |
| `"hac"` | The Newey-West estimator of [OLS](ols.md#hac-newey-west) on \(\hat x_i\hat\varepsilon_i\); `hac_time` is required |
| `"cluster"` | The cluster estimator of [OLS](ols.md#cluster-cluster-robust) on \(\hat x_i\hat\varepsilon_i\), with the factor \(\frac{G}{G-1}\cdot\frac{n-1}{n-k}\) |

The small-sample factors use \(k\), the number of coefficients of the structural equation, not the number of instruments \(l\).

### GMM

The covariance is always the general sandwich

\[
\widehat{\mathrm{Var}}(\hat\beta) = B^{-1}\big(X'Z\,W\hat\Omega W\,Z'X\big)B^{-1}, \qquad B = X'ZWZ'X
\]

where \(W = S^{-1}\) is the weight matrix actually used for the point estimate and \(\hat\Omega\) is the moment covariance for `cov_type`: homoskedastic (`"classical"`, with centred residuals), heteroskedasticity-robust (`"hc0"` to `"hc3"`, with the leverage \(h_{ii} = z_i'(Z'Z)^{-1}z_i\) for HC2 and HC3), clustered or HAC. The simplification to \(B^{-1}\) for an efficient weight matrix is not taken, so `gmm_weight_type` and `cov_type` can differ freely.

### Clusters

With `cov_type="cluster"`, `fit()` raises an error with fewer than two clusters, and also when there are not more clusters than slope coefficients of the structural equation, because the overall test is then not defined. For `gmm_weight_type="cluster"` the weight matrix \(S\) is \(l \times l\) and has rank at most \(G\) (at most \(G-1\) when just identified, because the moments then sum to zero), so `fit()` raises an error unless \(G \ge l\) when over-identified, or \(G > l\) when just identified.

## Test distributions and confidence intervals

| | Coefficient statistics | Overall test | Degrees of freedom |
|---|---|---|---|
| `"2sls"` | t | F | \(n-k\); \(G-1\) with `cov_type="cluster"` |
| `"gmm"` | normal | chi-squared | none |

GMM is an asymptotic estimator with no finite-sample degrees of freedom, so it uses the normal distribution; 2SLS uses the t and F distributions in the same way as [OLS](ols.md#test-distributions-and-confidence-intervals). `stat_dist` and `wald_dist` report which was used, and `stat_df` is `None` for GMM.

The overall test, `wald_statistic`, is the Wald test that all slope coefficients are zero, using the covariance chosen by `cov_type`:

\[
W = \hat\beta_s'\hat\Sigma_s^{-1}\hat\beta_s, \qquad
\text{2SLS: } F = W/q \sim F(q, \mathrm{df}), \qquad \text{GMM: } W \sim \chi^2(q)
\]

## Fit statistics

\(R^2 = 1 - \mathrm{SSR}/\mathrm{SST}\) and the adjusted \(R^2\) are computed from the structural residuals, with \(\mathrm{SST}\) as in [OLS](ols.md#fit-statistics). Because the structural residuals are not least-squares residuals, \(R^2\) can be low or even negative, and it is not the share of variance explained by the instruments. A perfect fit (residuals zero up to rounding) is rejected, because the standard errors are then not defined.

## Diagnostics

### Weak instruments: `weak_instrument_f_statistics`

For each endogenous variable \(x_{2j}\), the first-stage regression is \(x_{2j} \sim X_1 + Z_2\). The statistic is the partial F test that the coefficients of the \(m\) excluded instruments are jointly zero after partialling out \(X_1\):

\[
F_j = \frac{(\mathrm{SSR}_r - \mathrm{SSR}_u)/m}{\mathrm{SSR}_u/(n-l)} \sim F(m, n-l)
\]

where \(\mathrm{SSR}_r\) is the residual sum of squares of the restricted regression on \(X_1\) alone and \(\mathrm{SSR}_u\) of the first stage. It is always computed under homoskedasticity and does not depend on `cov_type`. It is not compared with the Stock-Yogo critical values and no weak/strong verdict is reported; a commonly quoted rule of thumb is a value above about 10. Joint diagnostics for several endogenous variables (Cragg-Donald) are not provided.

### Overidentification: `overid_statistic`

Chi-squared with \(l - k\) degrees of freedom (`overid_df`); `None` when just identified.

| `estimator` | Statistic |
|---|---|
| `"2sls"` | Sargan: \(\hat\varepsilon' Z (Z'Z)^{-1} Z'\hat\varepsilon / \hat\sigma^2\), with \(\hat\sigma^2 = \hat\varepsilon'\hat\varepsilon/n\). Homoskedastic; does not depend on `cov_type`. |
| `"gmm"` | Hansen's J: \((Z'\hat\varepsilon)' S^{-1} (Z'\hat\varepsilon)\), with the weight matrix used for the point estimate, so it follows `gmm_weight_type`. |

For a test that is robust to heteroskedasticity or clustering, use `estimator="gmm"` with a robust or cluster `gmm_weight_type` and read Hansen's J.

### Endogeneity: `wu_hausman_statistic`

The regression-based (control function) version: the first-stage residuals \(\hat v\) are added to the structural equation, \(y \sim X + \hat V\), and the F test that the coefficients of \(\hat V\) are jointly zero is reported, with the covariance given by the `cov_type` of the fit (the formulation of linearmodels' `wooldridge_regression`). The degrees of freedom are the number of endogenous variables and the residual degrees of freedom of the augmented regression, \(n - k - k_{\text{endog}}\) (\(G - 1\) with `cov_type="cluster"`).

It is `None` for `estimator="gmm"`, and also when the augmented regression cannot be estimated: it has \(k_{\text{endog}}\) more columns than the structural equation, so there may be too few observations or clusters, or it may fit perfectly.

## First stage: `first_stage()`

Returns, per endogenous variable, the plain OLS regression of `x_endog[i]` on `x_exog` and `instruments` as an `OLSResults`. Its `f_statistic` tests all slope coefficients including those of `x_exog`, so it is not the partial F statistic above. With `cov_type="cluster"` and as many clusters as, or fewer than, the slope coefficients of a first stage (but more than the structural equation's), that regression's overall F statistic is `None`.
