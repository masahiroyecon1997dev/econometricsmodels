# Tobit: definitions

**Tobit**: [Usage](../usage/tobit.md) · [Options and results](../api/tobit.md) · **Definitions**

This page defines what `Tobit` computes: the censored-normal likelihood, the covariance estimators behind `cov_type`, the test distributions, the overall test, the four prediction targets and the censoring fit check. Tobit is estimated by maximum likelihood like [Logit](logit.md), and the solvers, `tol`, `max_iter` and `raise_on_non_convergence` work as described there. For the arguments and return values see the [Tobit API reference](../api/tobit.md).

## Model and estimator

The observed outcome \(y_i\) is a continuous latent regression, censored at a lower bound \(a\) (`lower`), an upper bound \(b\) (`upper`), or both. A side set to `None` is not censored.

\[
y_i^* = x_i'\beta + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2), \qquad
y_i = \min\big(\max(y_i^*, a),\, b\big)
\]

With \(z_i = (y_i - x_i'\beta)/\sigma\), \(\Phi\) the standard normal CDF and \(\phi\) its density, an observation contributes to the log-likelihood according to its censoring status:

| Status | Contribution |
|---|---|
| uncensored, \(a < y_i < b\) | \(-\ln\sigma + \ln\phi(z_i)\) |
| censored below, \(y_i = a\) | \(\ln\Phi\big((a - x_i'\beta)/\sigma\big)\) |
| censored above, \(y_i = b\) | \(\ln\big(1 - \Phi\big((b - x_i'\beta)/\sigma\big)\big)\) |

An observation is censored when \(y_i\) is exactly equal to a bound, so make sure that the bounds and \(y\) have the same floating-point values. The `lower` and `upper` bounds are echoed back as `TobitResults.lower` and `.upper`.

The error standard deviation \(\sigma\) is estimated together with \(\beta\) and is reported as the last parameter, named `sigma`, with its own standard error. Internally the optimizer works with \(\ln\sigma\), which keeps \(\sigma\) positive, and the covariance matrix is transformed to \((\beta, \sigma)\) by the delta method, including the covariances between \(\beta\) and \(\sigma\). A column of `x` may therefore not be named `"sigma"`.

Unlike the Logit and Probit likelihoods, the Tobit likelihood in \((\beta, \ln\sigma)\) is not globally concave. The default `"newton"` solver damps its step (Levenberg-Marquardt) where the Hessian is indefinite; the three solvers still converge to the same maximum-likelihood estimate.

### What `fit()` rejects

`fit()` raises an error if both bounds are `None`, if `lower >= upper`, if any \(y_i\) lies outside the bounds, and if no observation lies strictly inside the bounds (the model is then not identified).

## Covariance of the coefficients: `cov_type`

`"classical"`, `"opg"`, `"hc0"`, `"hc1"` and `"cluster"` are defined as for [Logit](logit.md#covariance-of-the-coefficients-cov_type), with the score and Hessian of the Tobit log-likelihood and \(k+1\) parameters (\(\beta\) and \(\sigma\)), so the factor of `"hc1"` is \(n/(n-k-1)\) and the small-sample factor of `"cluster"` uses \((n-1)/(n-k-1)\), where \(k\) is the number of coefficients without `sigma`. The covariance matrix includes \(\sigma\) as its last row and column, expressed in the scale of \(\sigma\) rather than \(\ln\sigma\), so `std_errors`, `test_stats`, `p_values` and `conf_int` have an entry for `sigma`. The matrix is available as `cov_params` on the results. HC2, HC3 and HAC are not available.

## Test distributions and confidence intervals

As for Logit, every `cov_type` uses the standard normal distribution (`stat_dist` is `"normal"`, `stat_df` is `None`). This includes the entry for `sigma`: the z test of `sigma = 0` is not a meaningful hypothesis test, because \(\sigma > 0\) by construction, so read it as a precision measure.

## Overall test and fit statistics

The overall test is a **Wald test** that all slope coefficients are zero, with \(\hat\beta_s\) the \(q\) slope coefficients and \(\hat\Sigma_s\) the corresponding block of the covariance matrix chosen by `cov_type`:

\[
W = \hat\beta_s' \hat\Sigma_s^{-1} \hat\beta_s \sim \chi^2(q)
\]

reported as `wald_statistic`, `wald_p_value` and `wald_df`. It follows `cov_type`; with `"classical"` it equals the Wald statistic of R's `AER::tobit`. `fit()` raises an error when there are not more clusters than slope coefficients with `cov_type="cluster"`, because the block is then singular.

There is no likelihood-ratio statistic and no pseudo R², because the intercept-only Tobit model has no closed-form log-likelihood. The information criteria count \(\sigma\) as a parameter:

\[
\mathrm{AIC} = -2\ell + 2(k+1), \qquad \mathrm{BIC} = -2\ell + (k+1)\ln n
\]

## Predictions and marginal effects: `target`

`predict()` and `marginal_effects()` take a `target`. Write \(\mu_i = x_i'\hat\beta\), \(z_a = (a-\mu_i)/\hat\sigma\) and \(z_b = (b-\mu_i)/\hat\sigma\); a side that is not censored has \(\Phi = 0\) (below) or \(1\) (above) and \(\phi = 0\).

| `target` | Quantity | Expression |
|---|---|---|
| `"expected_latent"` | \(E[y^* \mid x]\) | \(\mu_i\) |
| `"expected_observed"` (default) | \(E[y \mid x]\), the censoring-adjusted mean | \(a\,\Phi(z_a) + b\,(1-\Phi(z_b)) + (\Phi(z_b)-\Phi(z_a))\,\mu_i + \sigma\,(\phi(z_a)-\phi(z_b))\) |
| `"prob_uncensored"` | \(P(a < y < b \mid x)\) | \(\Phi(z_b) - \Phi(z_a)\) |
| `"expected_uncensored"` | \(E[y \mid a < y < b,\, x]\), the mean of a truncated normal | \(\mu_i + \sigma\,\dfrac{\phi(z_a) - \phi(z_b)}{\Phi(z_b) - \Phi(z_a)}\) |

The four are related by

\[
E[y \mid x] = a\,P(y = a \mid x) + b\,P(y = b \mid x) + P(a < y < b \mid x)\; E[y \mid a < y < b,\, x]
\]

With a single lower bound of 0 this reads \(E[y \mid x] = P(y > 0 \mid x)\, E[y \mid y > 0, x]\). `"expected_uncensored"` answers questions such as "among households with positive spending, how does average spending change with x?".

`marginal_effects()` returns \(\partial\,\text{target}/\partial x_j\) for each explanatory variable (the constant and `sigma` are excluded). For every target the effect has the form \(w \cdot \hat\beta_j\), where \(w\) depends on the target:

| `target` | \(w\) |
|---|---|
| `"expected_latent"` | \(1\) |
| `"expected_observed"` | \(\Phi(z_b) - \Phi(z_a)\) (McDonald-Moffitt) |
| `"prob_uncensored"` | \((\phi(z_a) - \phi(z_b))/\sigma\) |
| `"expected_uncensored"` | the variance of the truncated normal divided by \(\sigma^2\), which lies in \((0, 1)\) |

The point of evaluation is chosen by `at` (`"overall"` average effects, the default, `"mean"` or `"median"`) and the standard errors are delta-method standard errors from the covariance matrix of \((\hat\beta, \hat\sigma)\), as for [Logit](logit.md#marginal-effects). With a right bound only, the effect of `"prob_uncensored"` has the opposite sign.

With two-sided censoring, `"expected_uncensored"` needs the uncensored interval to be wide enough to compute the truncated mean reliably. When \(\hat\sigma\) is more than 100 times the interval width \(b - a\), `predict()`, `augment()` and `marginal_effects()` with this target raise a `ComputationError`; use another `target`. A one-sided bound and the other three targets are not affected.

`augment()` appends the predictions as a column named `"predicted_{target}"` (for example `"predicted_expected_observed"`), so calling it once per target on the same DataFrame does not collide.

## Censoring fit check

`censoring_fit_check()` compares, for each censored direction, the share of observations actually at the bound with the share the model implies. It returns one entry per category: `"lower"`, `"uncensored"` and `"upper"` (a direction that is not censored is left out).

| Field | Definition |
|---|---|
| `observed_rate` | The share of observations in the category (\(y_i = a\), \(a < y_i < b\), \(y_i = b\)). |
| `model_implied_rate` | The mean over observations of the fitted probability of the category: \(\Phi(z_a)\), \(\Phi(z_b) - \Phi(z_a)\), \(1 - \Phi(z_b)\). |

It is computed on the estimation data only. A large gap between the two rates suggests that the censored-normal model does not describe the data well.
