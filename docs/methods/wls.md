# WLS: definitions

**WLS**: [Usage](../usage/wls.md) · [Options and results](../api/wls.md) · **Definitions**

This page defines what `WLS` computes. WLS is least squares on \(\sqrt{w}\)-scaled data, so the covariance estimators and test distributions are those defined on the [OLS page](ols.md), applied to the scaled data. Only what the weights change is described here. For the arguments and return values see the [WLS API reference](../api/wls.md).

## Model and estimator

The model is \(y = X\beta + \varepsilon\) with \(\mathrm{Var}(\varepsilon_i) = \sigma^2 / w_i\): the weights are **analytic weights**, proportional to the inverse of the error variance. They need not be normalized. Frequency weights and probability (sampling) weights are not supported.

\[
\hat\beta = (X'WX)^{-1} X'Wy, \qquad W = \mathrm{diag}(w_1, \dots, w_n)
\]

Equivalently, \(\hat\beta\) is the OLS estimator of \(\tilde y\) on \(\tilde X\), where \(\tilde y_i = \sqrt{w_i}\, y_i\) and \(\tilde x_i = \sqrt{w_i}\, x_i\). The constant column added by `include_intercept=True` is scaled too, so it becomes the column \(\sqrt{w_i}\).

The weight column is given by `weight`. Every weight must be finite and strictly positive; a zero, negative, missing or non-finite weight raises an error rather than dropping the observation. The weight column may not be the same column as `y`; it may also appear in `x`.

## Covariance of the coefficients: `cov_type`

Every `cov_type` is defined as on the OLS page, evaluated on the scaled quantities \(\tilde x_i\) and \(\tilde\varepsilon_i = \sqrt{w_i}\,\hat\varepsilon_i\). For `"classical"`:

\[
\hat\sigma^2 = \frac{\sum_i w_i \hat\varepsilon_i^2}{n-k}, \qquad
\widehat{\mathrm{Var}}(\hat\beta) = \hat\sigma^2 (X'WX)^{-1}
\]

In the sandwich estimators the weights therefore enter squared. For example, HC0 is

\[
\hat S = \sum_i \tilde\varepsilon_i^2\, \tilde x_i \tilde x_i' = \sum_i w_i^2\, \hat\varepsilon_i^2\, x_i x_i'
\]

The weights play no part in the choice of the HAC lag length, in the grouping by `cluster`, or in the small-sample factor and degrees of freedom: those depend only on \(n\), \(k\) and \(G\), exactly as for OLS.

## Test distributions and the overall F test

As for OLS, every `cov_type` uses the t distribution with \(n-k\) degrees of freedom (\(G-1\) for `"cluster"`), and the overall F test is the Wald form \(\hat\beta_s' \hat\Sigma_s^{-1} \hat\beta_s / q\) computed from the scaled data.

## Fit statistics

The sums of squares are weighted. \(\mathrm{SSR} = \sum_i w_i \hat\varepsilon_i^2\), and the total sum of squares uses the weighted mean \(\bar y_w = \sum_i w_i y_i / \sum_i w_i\):

\[
\mathrm{SST} = \begin{cases}
\sum_i w_i (y_i - \bar y_w)^2 & \text{with an intercept} \\
\sum_i w_i y_i^2 & \text{without}
\end{cases}
\qquad R^2 = 1 - \frac{\mathrm{SSR}}{\mathrm{SST}}
\]

Adjusted R² has the same form as for OLS. The log-likelihood of the weighted model includes the Jacobian of the scaling:

\[
\ell = -\frac{n}{2}\Big(\ln(2\pi) + \ln\frac{\mathrm{SSR}}{n} + 1\Big) + \frac12 \sum_i \ln w_i
\]

AIC and BIC are computed from it as for OLS.

## Residuals and predictions

`residuals` are on the **original scale**, \(y_i - x_i'\hat\beta\), not the weighted residuals \(\sqrt{w_i}\,\hat\varepsilon_i\) that the standard errors are built from. This matches `statsmodels`' `.resid` and keeps residual plots comparable with OLS.

`predict()` returns \(x_i'\hat\beta\) on the original scale. The weights play no role in it, with or without `new_data`: a new observation needs no weight, and none is applied to the estimation data either.
