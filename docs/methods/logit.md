# Logit: definitions

**Logit**: [Usage](../usage/logit.md) · [Options and results](../api/logit.md) · **Definitions**

This page defines what `Logit` computes: the estimator, the covariance estimators behind `cov_type`, the test distributions, the fit statistics, marginal effects and predictions. For the arguments and return values see the [Logit API reference](../api/logit.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md). [Probit](probit.md) differs only in the link function.

## Model and estimator

The dependent variable \(y_i\) is coded 0 or 1 (anything else raises an error), and

\[
P(y_i = 1 \mid x_i) = \Lambda(x_i'\theta) = \frac{1}{1 + e^{-x_i'\theta}}
\]

with \(n\) observations and \(k\) coefficients \(\theta\). \(X\) holds the columns listed in `x`; with `include_intercept=True` (the default) a constant column named `const` is added in front of them. The estimate \(\hat\theta\) maximizes the log-likelihood

\[
\ell(\theta) = \sum_i \big[\, y_i\, x_i'\theta - \ln\big(1 + e^{x_i'\theta}\big) \big]
\]

with score \(s_i = (y_i - p_i)\,x_i\) and Hessian \(H = -X'WX\), where \(p_i = \Lambda(x_i'\hat\theta)\) and \(W = \mathrm{diag}\big(p_i(1-p_i)\big)\). The log-likelihood is globally concave, and strictly concave when the columns of \(X\) are not perfectly collinear, so a maximum is unique if it exists; under separation (below) it does not exist. A design with perfectly collinear columns raises an error instead of being estimated on a reduced set of columns (see [Validation and errors](../guide/validation.md)).

### Solvers and convergence

`solver` selects the algorithm: `"newton"` (default, Newton-Raphson), `"bfgs"` or `"lbfgs"`. All three converge to the same maximum-likelihood estimate; they differ in the cost of an iteration and in how they behave on ill-conditioned problems. The optimization is run on standardized columns and the results are transformed back, so the scale of the regressors does not affect convergence.

`tol` is a threshold on the gradient norm. The gradient is that of the log-likelihood with respect to the coefficients of the standardized columns (with an intercept, each column is centered and divided by its standard deviation). For `"newton"` it applies to the total gradient; for `"bfgs"` and `"lbfgs"` it applies to the total gradient divided by `n`. The default depends on the solver (`1e-6` for `"newton"`, `1e-8` for the others), and an explicit `tol` is used as given. When the solver has not converged after `max_iter` iterations, `fit()` raises an error unless `raise_on_non_convergence=False`, in which case the last iterate is returned with `converged=False`. `n_iter` is the number of iterations used.

### Separation

If some combination of the regressors perfectly predicts the outcome, for all observations (complete separation) or for a subset (quasi-complete separation, for example a dummy variable that equals 1 only when \(y = 1\)), no finite maximum-likelihood estimate exists and the coefficients would diverge. This is detected exactly after the optimizer stops, whatever the sample size, and raises an error. With `raise_on_non_convergence=False` the result is returned with `converged=False` instead. Data that are close to, but not at, separation have a finite estimate and are estimated without an error, although the standard errors can be very large.

## Covariance of the coefficients: `cov_type`

Let \(s_i\) be the score of observation \(i\) and \(H\) the Hessian of \(\ell\), both at \(\hat\theta\). Standard errors are the square roots of the diagonal of the covariance matrix, and the covariance matrix of the coefficients is available as `cov_params` on the results.

| `cov_type` | Covariance |
|---|---|
| `"classical"` (default) | \(-H^{-1}\) (observed information) |
| `"opg"` | \(\big(\sum_i s_i s_i'\big)^{-1}\) (outer product of gradients) |
| `"hc0"` | \(H^{-1}\big(\sum_i s_i s_i'\big)H^{-1}\) |
| `"hc1"` | `"hc0"` multiplied by \(n/(n-k)\) |
| `"cluster"` | \(c\; H^{-1}\big(\sum_g S_g S_g'\big)H^{-1}\), with \(S_g = \sum_{i \in g} s_i\) and \(c = \dfrac{G}{G-1}\cdot\dfrac{n-1}{n-k}\) |

For `"cluster"`, observations are grouped by the column named in `cluster` (\(G\) groups) and the small-sample factor \(c\) is always applied. `fit()` raises an error with fewer than two clusters, and also when there are not more clusters than slope coefficients, because the cluster-robust covariance is then degenerate. HC2, HC3 and HAC are not available: the first two depend on the hat matrix of a linear regression, and HAC is not implemented for maximum-likelihood models.

## Test distributions and confidence intervals

Every `cov_type` uses the **standard normal distribution**. For coefficient \(j\), the z statistic is \(\hat\theta_j / \mathrm{se}(\hat\theta_j)\), the p-value is two-sided, and the confidence interval is \(\hat\theta_j \pm z_{1-\alpha/2}\,\mathrm{se}(\hat\theta_j)\) with \(1-\alpha =\) `confidence_level`. The theory behind maximum likelihood is asymptotic and has no finite-sample degrees of freedom, so `stat_dist` is `"normal"` and `stat_df` is `None`.

## Overall test and fit statistics

Let \(\ell_0\) be the log-likelihood of the model with only an intercept. It has a closed form that does not need a second optimization:

\[
\ell_0 = n_1 \ln\bar y + n_0 \ln(1-\bar y)
\]

where \(n_1\) and \(n_0\) are the numbers of observations with \(y=1\) and \(y=0\), and \(\bar y = n_1/n\). Let \(q = k-1\) be the number of slope coefficients.

\[
\mathrm{LR} = 2(\ell - \ell_0) \sim \chi^2(q), \qquad
R^2_{\mathrm{McFadden}} = 1 - \frac{\ell}{\ell_0}
\]

\[
\mathrm{AIC} = -2\ell + 2k, \qquad \mathrm{BIC} = -2\ell + k \ln n
\]

\(q\) is reported as `lr_df`. The overall test is the likelihood-ratio test whatever `cov_type` is used: it does not follow `cov_type`.

With `include_intercept=False`, \(\ell_0\) is still the log-likelihood of the intercept-only model, which is not nested in the fitted model, and \(q = k-1\) is used all the same (the convention of statsmodels). The LR statistic can then be negative and the test is not meaningful. If \(q = 0\) (`include_intercept=False` with a single regressor), `lr_df` is `None` and `lr_p_value` is `NaN`, while `lr_statistic` and `pseudo_r_squared` are still computed.

## Marginal effects

`marginal_effects(at=...)` returns \(\partial P(y=1)/\partial x_j\) for each explanatory variable (the constant is excluded), with delta-method standard errors and z statistics. For Logit

\[
\frac{\partial p_i}{\partial x_{ij}} = p_i(1-p_i)\,\hat\theta_j
\]

The point of evaluation is chosen by `at`:

| `at` | Meaning |
|---|---|
| `"overall"` (default) | The average of the effect over all observations (average marginal effects). |
| `"mean"` | The effect at the mean of each explanatory variable. |
| `"median"` | The effect at the median of each explanatory variable. |

The effect is the derivative for every column, including a 0/1 dummy variable; it is not the discrete change from 0 to 1. The standard error is \(\sqrt{J \hat\Sigma J'}\), where \(J\) is the gradient of the effect with respect to \(\theta\) and \(\hat\Sigma\) is the covariance matrix chosen by `cov_type`. The confidence level of `marginal_effects()` is its own argument and need not equal the one used in `fit()`.

## Predictions

`predict()` returns the fitted probabilities \(\hat p_i = \Lambda(x_i'\hat\theta)\), not a 0/1 class. Without `new_data` it is evaluated on the estimation data. With `new_data` the frame must contain the columns of `x` (matched by name, in any order); do not include `const`: it is added automatically when the model has an intercept. `augment()` returns the same values as a `"predicted_probability"` column attached to the source data.

`pred_table(threshold=0.5)` is a 2×2 classification table for the estimation data only. The rows are the actual outcome, and the columns count the observations whose predicted probability is above or not above `threshold`. The actual outcome is always the observed 0/1 value, whatever the threshold.
