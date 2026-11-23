# Probit: definitions

**Probit**: [Usage](../usage/probit.md) · [Options and results](../api/probit.md) · **Definitions**

This page defines what `Probit` computes. Probit is the same maximum-likelihood model as [Logit](logit.md) with the standard normal CDF in place of the logistic CDF, so the solvers, convergence rules, separation detection, covariance estimators (`cov_type`), test distribution, overall test and `pred_table()` are those defined on the [Logit page](logit.md). Only what the link function changes is described here. For the arguments and return values see the [Probit API reference](../api/probit.md).

## Model and estimator

The dependent variable \(y_i\) is coded 0 or 1, and

\[
P(y_i = 1 \mid x_i) = \Phi(x_i'\theta)
\]

where \(\Phi\) is the standard normal CDF and \(\phi\) its density. With \(a_i = 2y_i - 1 \in \{-1, +1\}\) and \(z_i = x_i'\theta\), the log-likelihood is

\[
\ell(\theta) = \sum_i \ln \Phi(a_i z_i)
\]

Let \(\lambda_i = a_i\,\phi(a_i z_i) / \Phi(a_i z_i)\) (the generalized residual). The score is \(s_i = \lambda_i x_i\) and the Hessian is \(H = -X'WX\) with \(W = \mathrm{diag}\big(\lambda_i(\lambda_i + z_i)\big)\). The log-likelihood is globally concave, and strictly concave when the columns of \(X\) are not perfectly collinear, so a maximum is unique if it exists; under separation (see [Logit](logit.md#separation)) it does not exist.

For extreme linear predictors the ratio \(\phi/\Phi\) underflows in floating point. The argument \(a_i z_i\) is therefore clamped to \([-8.1259, 8.1259]\) (that is, \(-\Phi^{-1}(\epsilon)\) with \(\epsilon\) the machine epsilon) before \(\phi\) and \(\Phi\) are evaluated, as R's `binomial(link = "probit")` does. The clamp is applied to the log-likelihood, the score and the Hessian alike, so they stay consistent with each other.

## Covariance of the coefficients: `cov_type`

As for [Logit](logit.md#covariance-of-the-coefficients-cov_type), with the Probit score and Hessian above: `cov_type` is `"classical"`, `"opg"`, `"hc0"`, `"hc1"` or `"cluster"`. The covariance matrix of the coefficients is available as `cov_params` on the results, as for Logit.

## Test distributions and confidence intervals

Tests use the standard normal distribution, as for [Logit](logit.md#test-distributions-and-confidence-intervals).

## Overall test and fit statistics

The overall test is the likelihood-ratio test \(2(\ell - \ell_0)\) with the closed-form intercept-only log-likelihood \(\ell_0 = n_1 \ln\bar y + n_0 \ln(1-\bar y)\), where \(\bar y = n_1/n\). The degrees of freedom, and what happens with `include_intercept=False`, are as for [Logit](logit.md#overall-test-and-fit-statistics). McFadden's pseudo R², AIC and BIC are defined as for Logit.

## Predictions

`predict()` returns the fitted probabilities \(\hat p_i = \Phi(x_i'\hat\theta)\); `augment()` attaches them as a `"predicted_probability"` column; `pred_table()` is as for [Logit](logit.md#predictions).

## Marginal effects

For Probit

\[
\frac{\partial p_i}{\partial x_{ij}} = \phi(x_i'\hat\theta)\,\hat\theta_j
\]

The choices of `at` (`"overall"`, `"mean"`, `"median"`), the delta-method standard errors and the exclusion of the constant are as for [Logit](logit.md#marginal-effects).
