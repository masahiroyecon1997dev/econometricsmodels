# RE: definitions

**RE**: [Usage](../usage/re.md) · [Options and results](../api/re.md) · **Definitions**

This page defines what `RE` computes: the variance components, the quasi-demeaning (GLS) transformation, the covariance estimators behind `cov_type`, the test distributions, the panel R² statistics and the Hausman test. For the arguments and return values see the [RE API reference](../api/re.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md). The fixed effects counterpart is [FE](fe.md), and the notation is the same.

## Model and estimator

\[
y_{it} = \beta_0 + x_{it}'\beta + u_i + \varepsilon_{it} \quad \text{(one-way)}, \qquad
y_{it} = \beta_0 + x_{it}'\beta + u_i + \lambda_t + \varepsilon_{it} \quad \text{(two-way)}
\]

The effects are random: \(u_i\), \(\lambda_t\) and \(\varepsilon_{it}\) are uncorrelated with the regressors and with each other, with variances \(\sigma_u^2\), \(\sigma_\lambda^2\) and \(\sigma_\varepsilon^2\). Unlike FE, RE keeps the intercept (`const` is always the first parameter) and **allows time-invariant regressors**. With `REOptions.time` set it estimates entity and time effects on a balanced panel; otherwise only entity effects, and an unbalanced panel is allowed. An entity with a single observation is rejected, as in FE.

The estimator is feasible GLS, computed as OLS on quasi-demeaned data (all columns, including the constant, are transformed); \(y^*\) and \(X^*\) denote the quasi-demeaned outcome and regressors:

\[
y_{it} - \theta_i \bar y_{i\cdot}, \qquad \theta_i = 1 - \sqrt{\frac{\sigma_\varepsilon^2}{T_i\,\sigma_u^2 + \sigma_\varepsilon^2}}
\]

For the two-way model the transformation is \(y_{it} - \theta_1\bar y_{i\cdot} - \theta_2\bar y_{\cdot t} + \theta_3\bar y_{\cdot\cdot}\), with

\[
\theta_1 = 1 - \frac{\sigma_\varepsilon}{\sigma_1}, \quad
\theta_2 = 1 - \frac{\sigma_\varepsilon}{\sigma_2}, \quad
\theta_3 = \theta_1 + \theta_2 - 1 + \frac{\sigma_\varepsilon}{\sigma_4}
\]

and \(\sigma_1^2 = \sigma_\varepsilon^2 + T\sigma_u^2\), \(\sigma_2^2 = \sigma_\varepsilon^2 + N\sigma_\lambda^2\), \(\sigma_4^2 = \sigma_\varepsilon^2 + T\sigma_u^2 + N\sigma_\lambda^2\). With \(\theta = 0\) the estimator is pooled OLS; with \(\theta = 1\) it is the FE estimator. The weights are returned as `theta` (`theta_entity`, `theta_time` and `theta_total` for two-way). A design with perfectly collinear columns raises an error instead of being estimated on a reduced set of columns, with the exception described under [correlated random effects](#time-invariant-regressors-and-correlated-random-effects).

## Variance components: `variance_method`

All three methods estimate \(\sigma_\varepsilon^2\), \(\sigma_u^2\) (and \(\sigma_\lambda^2\) for two-way) by equating sums of squares of a first-stage regression's residuals to their expectations. Negative \(\sigma_u^2\) and \(\sigma_\lambda^2\) are set to 0. The method used is returned as `variance_method`; the components as `sigma2_eps`, `sigma2_u` and `sigma2_lambda`.

| `variance_method` | First-stage residuals |
|---|---|
| `"swamy_arora"` (default) | The within regression for \(\sigma_\varepsilon^2\), and the between regression on the entity means (the time means, for \(\sigma_\lambda^2\)) for the others |
| `"wallace_hussain"` | Pooled OLS of \(y\) on a constant and all regressors |
| `"amemiya"` | The within estimator |

**Swamy-Arora.** With \(\bar T\) the harmonic mean of the \(T_i\):

\[
\hat\sigma_\varepsilon^2 = \frac{\mathrm{SSR}_{\text{within}}}{n - N - k_{tv}}, \qquad
\hat\sigma_u^2 = \max\Big(0,\ \frac{\mathrm{SSR}_{\text{between}}}{N - K_b - 1} - \frac{\hat\sigma_\varepsilon^2}{\bar T}\Big)
\]

where \(k_{tv}\) counts the time-varying regressors (those with within variation) and \(K_b\) the regressors in the between regression. In the two-way model \(\hat\sigma_\varepsilon^2\) uses \(n - N - T + 1 - k_{tv}\), and \(\hat\sigma_u^2\), \(\hat\sigma_\lambda^2\) use the entity-mean and time-mean regressions with \(\sigma_\varepsilon^2/T\) and \(\sigma_\varepsilon^2/N\) subtracted. On an unbalanced one-way panel this is the `linearmodels` formula, which differs from the one in R's `plm` (plm 2.6.7; the difference in \(\sigma_u^2\) depends on how unbalanced the panel is, and was 2.6 % in one test panel).

**Wallace-Hussain and Amemiya** follow `plm` (`random.method = "walhus"` and `"amemiya"`). On a balanced panel (a one-way panel whose entities all have the same number of observations; two-way panels are always balanced) they use closed forms, and on an unbalanced one-way panel the trace-term forms. The closed form (balanced) and the trace-term form (unbalanced one-way) do not give the same value even on a balanced panel, so \(\hat\sigma_\varepsilon^2\) is not continuous when a row is dropped from a balanced panel. These methods are not available for unbalanced two-way panels (the panel is rejected). If the estimated \(\sigma_\varepsilon^2\) is not positive, `fit()` raises an error, whereas plm 2.6.7 sets it to 0 and continues (in a test case with unequal group sizes it then returned `theta = 1` for every entity). `"amemiya"` rejects a regressor without within variation (it would bias \(\sigma_u^2\)); `"wallace_hussain"` and `"swamy_arora"` accept it.

## Degrees of freedom

\(\mathrm{df}_{\text{model}} = k + 1\) counts all columns of the quasi-demeaned regression, and \(\mathrm{df}_{\text{resid}} = n - k - 1\) (with \(k\) slope coefficients). Nothing is subtracted for the effects in either the one-way or the two-way model, because the GLS transformation does not absorb entity or time dummies.

## Covariance of the coefficients: `cov_type`

The options are the same as for [FE](fe.md#covariance-of-the-coefficients-cov_type), computed on the quasi-demeaned data: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"` (the default, by entity) and `"dk"` (Driscoll-Kraay, requires `dk_time`, which is a different option from `time`). `"hc0"` is not available. The covariance matrix of the coefficients is available as `cov_params` on the results, with the constant first. Differences from FE:

- Because there are no absorbed dummies, the leverage in HC2 and HC3 is the plain \(h_{ii} = x_i^{*\prime}(X^{*\prime}X^*)^{-1}x_i^*\) of the quasi-demeaned design, and the small-sample factor uses \(K = \mathrm{df}_{\text{model}}\) in every case: \(\frac{G}{G-1}\cdot\frac{n-1}{n-K}\) for `"cluster"` (the form of `plm::vcovHC(type = "sss")`) and \(\frac{T_d}{T_d-1}\cdot\frac{n-1}{n-K}\) for `"dk"`.
- Cluster standard errors with `G` clusters need \(G\) greater than the number of slope coefficients, \(T_d\) greater than it for Driscoll-Kraay.

## Test distributions and confidence intervals

As for FE, every `cov_type` uses the **t distribution**, with \(\mathrm{df}_{\text{resid}}\) for `"classical"` and `"hc1"` to `"hc3"`, \(G-1\) for `"cluster"` and \(T_d-1\) for `"dk"`, returned as `stat_df`.

## Overall test and fit statistics

### Overall F test

The overall F test is the Wald test that the \(k\) slope coefficients are zero, \(F = \hat\beta_s'\hat\Sigma_s^{-1}\hat\beta_s/k\), with the covariance chosen by `cov_type` and the same degrees of freedom, so `f_df_denom` equals `stat_df`. The constant is not part of the tested hypothesis. This Wald form is used rather than the sum-of-squares form of `linearmodels`' `f_statistic`, which ignores `cov_type`.

### Fit statistics

The log-likelihood is computed from the residuals of the quasi-demeaned regression as for [OLS](ols.md#fit-statistics), and \(\mathrm{AIC} = -2\ell + 2(k+1)\), \(\mathrm{BIC} = -2\ell + (k+1)\ln n\).

The three R² values use \(\hat\beta_0\) and \(\hat\beta\) on the respective variables, with the usual centered total sum of squares (unlike FE, because the model has a constant):

| Statistic | Computed on |
|---|---|
| `r_squared_within` | The within-transformed variables (both effects removed in the two-way model); the slope coefficients only |
| `r_squared_between` | The entity means |
| `r_squared_overall` | The original variables |

With no slope coefficient the three values are 0. They agree with `linearmodels`' `rsquared_within`, `rsquared_between` and `rsquared_overall` for the one-way model with the classical covariance. `r_squared_between` can be negative.

## Time-invariant regressors and correlated random effects

A time-invariant regressor is allowed. So is a regressor together with its entity mean (and, in the two-way model, its time mean), the correlated random effects (Mundlak) specification. Such a design makes some auxiliary regressions rank deficient. The between regressions of the variance components and the Hausman auxiliary regression therefore drop aliased columns, the later of two dependent columns, using a relative tolerance of \(10^{-7}\), as R's `lm` does. The final regression keeps every column. A regressor that varies in only one dimension (for example a year trend on a balanced panel) is treated the same way.

## Hausman test

`hausman_statistic`, `hausman_p_value` and `hausman_df` are computed by `fit()`; they are `None` only when no slope coefficient can be tested. The test is the regression-based (auxiliary regression) version of Wooldridge (2010, section 10.7.3). It matches `plm::phtest(method = "aux")` when no regressor is time-invariant (see [Inference conventions](../guide/inference-conventions.md#other-diagnostics)). The quasi-demeaned \(y^*\) is regressed on an untransformed constant, the quasi-demeaned regressors \(X^*\) and the within-transformed regressors \(\tilde X\), and the Wald test that the coefficients of \(\tilde X\) are zero is reported:

\[
H = \hat\delta'\hat\Sigma_\delta^{-1}\hat\delta \sim \chi^2(\mathrm{df}), \qquad \mathrm{df} = \text{number of tested columns of } \tilde X
\]

where \(\hat\delta\) are the estimated coefficients of the tested columns of \(\tilde X\) and \(\hat\Sigma_\delta\) is their block of the covariance matrix chosen by `cov_type`.

- The statistic is non-negative by construction.
- The comparison is with the FE model of the same structure: entity effects, or entity and time effects when `time` is set.
- The covariance follows `cov_type`, so the default `"cluster"` gives the cluster-robust Hausman test, and `"classical"` gives the classical version, which assumes that RE is fully efficient under the null. The small-sample factors are those of the RE standard errors, with \(K\) the number of regressors of the auxiliary regression (the constant, \(X^*\) and the tested columns of \(\tilde X\), up to \(2k+1\)) instead of \(k+1\). `"dk"` rests on large-\(T\) asymptotics and may be distorted for a short panel.
- Only time-varying regressors are tested. A column of \(\tilde X\) is left out of the test, and not counted in `hausman_df`, when it cannot be identified. This happens in two cases: its quasi-demeaned counterpart in \(X^*\) differs from it only by a constant (for example a regressor that varies only over time on a balanced panel), or the entity mean (in the two-way model, the time mean) of the regressor is also a regressor.
- The auxiliary regression has up to \(2k\) slope coefficients, so `cov_type="cluster"` needs more than that many clusters, and `"dk"` needs more periods than the slope coefficients. If the regression cannot be computed, `fit()` raises an error instead of returning `None`; use `cov_type="classical"` or more clusters.
