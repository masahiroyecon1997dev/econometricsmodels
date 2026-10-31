# OLS: definitions

**OLS**: [Usage](../usage/ols.md) · [Options and results](../api/ols.md) · **Definitions**

This page defines what `OLS` computes: the estimator, the covariance estimators behind `cov_type`, the test distributions, the fit statistics and the post-estimation diagnostics. For the arguments and return values see the [OLS API reference](../api/ols.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md).

## Model and estimator

The model is \(y = X\beta + \varepsilon\), with \(n\) observations and \(k\) coefficients. \(X\) holds the columns listed in `x`. With `include_intercept=True` (the default) a constant column named `const` is added in front of them.

\[
\hat\beta = (X'X)^{-1} X'y, \qquad \hat\varepsilon = y - X\hat\beta, \qquad \hat\sigma^2 = \frac{\mathrm{SSR}}{n-k}
\]

where \(\mathrm{SSR} = \sum_i \hat\varepsilon_i^2\). A design with perfectly collinear columns has no unique solution, so it raises an error instead of being estimated on a reduced set of columns (see [Validation and errors](../guide/validation.md)).

## Covariance of the coefficients: `cov_type`

Every option below is a sandwich estimator \((X'X)^{-1} \hat S (X'X)^{-1}\) for the covariance of \(\hat\beta\), except `"classical"`. Standard errors are the square roots of its diagonal, and the covariance matrix of the coefficients is available as `cov_params` on the results.

### `"classical"` (default)

\[
\widehat{\mathrm{Var}}(\hat\beta) = \hat\sigma^2 (X'X)^{-1}
\]

This assumes homoskedastic, uncorrelated errors.

### `"hc0"` to `"hc3"`: heteroskedasticity-robust

\(\hat S = \sum_i c_i \hat\varepsilon_i^2 x_i x_i'\), where \(x_i\) is the \(i\)-th row of \(X\) and the weight \(c_i\) depends on the type:

| `cov_type` | \(c_i\) |
|---|---|
| `"hc0"` | \(1\) |
| `"hc1"` | \(n / (n - k)\) |
| `"hc2"` | \(1 / (1 - h_{ii})\) |
| `"hc3"` | \(1 / (1 - h_{ii})^2\) |

\(h_{ii} = x_i'(X'X)^{-1}x_i\) is the leverage of observation \(i\). HC2 and HC3 grow without bound as a leverage approaches 1, so a single extreme observation can dominate them.

### `"hac"`: Newey-West

Heteroskedasticity- and autocorrelation-consistent standard errors with Bartlett weights:

\[
\hat S = \hat S_0 + \sum_{l=1}^{L} w_l \big(\hat S_l + \hat S_l'\big), \qquad w_l = 1 - \frac{l}{L+1}, \qquad \hat S_l = \sum_{t=l+1}^{n} \hat\varepsilon_t \hat\varepsilon_{t-l}\, x_t x_{t-l}'
\]

The observations are put in time order by the column named in `hac_time` (required: row order is never assumed to be time order, and the values must all differ). The number of lags \(L\) is `hac_lags` when given, with \(0 \le L < n\). Otherwise it is chosen as

\[
L = \left\lfloor 4\,(n/100)^{2/9} \right\rfloor
\]

This rule depends only on `n`, not on the data. The value used is returned as `hac_lags_used`.

### `"cluster"`: cluster-robust

With observations grouped by the column named in `cluster` (\(G\) groups):

\[
\hat S = \sum_g s_g s_g', \qquad s_g = \sum_{i \in g} \hat\varepsilon_i x_i
\]

and the covariance is multiplied by the small-sample factor \(\dfrac{G}{G-1}\cdot\dfrac{n-1}{n-k}\). The factor is always applied. `fit()` raises an error with fewer than two clusters, and also when there are not more clusters than slope coefficients, because the overall F test (below) is then not defined.

## Test distributions and confidence intervals

Every `cov_type` uses the **t distribution**: OLS never switches to the normal distribution for robust standard errors. (The maximum-likelihood models Logit, Probit and Tobit use the normal distribution; see [Inference conventions](../guide/inference-conventions.md).)

| | degrees of freedom |
|---|---|
| all `cov_type` except `"cluster"` | \(n-k\) |
| `"cluster"` | \(G-1\) |

For coefficient \(j\), the t statistic is \(\hat\beta_j / \mathrm{se}(\hat\beta_j)\), the p-value is two-sided, and the confidence interval is

\[
\hat\beta_j \pm t_{1-\alpha/2,\,\mathrm{df}} \cdot \mathrm{se}(\hat\beta_j), \qquad 1-\alpha = \texttt{confidence\_level}
\]

The degrees of freedom actually used are returned as `stat_df`; use it, not `df_resid`, to recompute a p-value. `df_resid` is \(n-k\) regardless of `cov_type`.

## Overall test and fit statistics

### Overall F test

Let \(\hat\beta_s\) be the \(q\) estimated slope coefficients (all except the constant) and \(\hat\Sigma_s\) the corresponding block of the estimated covariance matrix chosen by `cov_type`:

\[
F = \frac{\hat\beta_s' \hat\Sigma_s^{-1} \hat\beta_s}{q} \sim F(q, \mathrm{df})
\]

\(\mathrm{df}\) is the t degrees of freedom above (\(G-1\) for `"cluster"`). With `"classical"` this is the textbook F test; with the other `cov_type` values it is a robust Wald test.

### Fit statistics

With an intercept, \(\mathrm{SST} = \sum_i (y_i - \bar y)^2\); without one, \(\mathrm{SST} = \sum_i y_i^2\).

\[
R^2 = 1 - \frac{\mathrm{SSR}}{\mathrm{SST}}, \qquad
\bar R^2 = 1 - \frac{n-d}{n-k}\,(1-R^2)
\]

where \(d = 1\) with an intercept and \(d = 0\) without.

\[
\ell = -\frac{n}{2}\Big(\ln(2\pi) + \ln\frac{\mathrm{SSR}}{n} + 1\Big), \qquad
\mathrm{AIC} = -2\ell + 2k, \qquad
\mathrm{BIC} = -2\ell + k\ln n
\]

The log-likelihood \(\ell\) uses the maximum-likelihood variance \(\mathrm{SSR}/n\), not \(\hat\sigma^2\).

Two situations in which these statistics are undefined are rejected rather than reported: an outcome that is constant in a model with an intercept (\(\mathrm{SST} = 0\)), and a perfect fit, where the residuals are zero up to rounding (\(\mathrm{SSR} \le k\,\epsilon_{\mathrm{mach}}\,\mathrm{SST}\), \(\epsilon_{\mathrm{mach}}\) the machine epsilon). See [Validation and errors](../guide/validation.md).

## Predictions

`predict()` returns \(\hat y = x'\hat\beta\). Without `new_data` it is evaluated on the estimation data. With `new_data` the frame must contain the columns of `x` (matched by name, in any order); do not include `const`: it is added automatically when the model has an intercept. `augment()` returns the same values as a `"predicted"` column attached to the source data.

## Post-estimation diagnostics

The diagnostics are never computed by `fit()`; call them on the results. None of them depends on `cov_type`, and each returns a result with `statistic`, `p_value`, `df`, `df_denom` and `distribution`.

Each test regresses a residual-based quantity on a set of auxiliary columns, and offers two versions chosen by `statistic`: `"lm"` (\(n R^2\) of the auxiliary regression, chi-squared) and `"f"` (the F test that the auxiliary coefficients are all zero). For the White and Breusch-Pagan tests the auxiliary regression always has a constant, \(q\) is the number of auxiliary columns excluding the constant, and the LM version has \(q\) degrees of freedom. The Breusch-Godfrey test differs in both respects: its degrees of freedom are \(m\), the number of lags, and its auxiliary regression adds no constant of its own (see below).

### `white_test()`

Tests the null of homoskedastic errors with the \(n R^2\) form of the statistic, which does not assume normal errors. The auxiliary regression is of \(\hat\varepsilon^2\) on the regressors, their squares and their pairwise products. The F version is \(F(q,\, n-q-1)\).

- The auxiliary regression always has a constant, even for a model fitted with `include_intercept=False`; `aux_terms` lists the terms, starting with `const`.
- Terms that are constant, or numerically identical to an earlier term (for example the square of a dummy variable), are dropped and listed in `dropped_terms`. `df` counts the terms that remain.

### `breusch_pagan_test(variables=None)`

The same Koenker version of the Breusch-Pagan test, but the auxiliary regression of \(\hat\varepsilon^2\) is on `variables` (a list of column names) instead of the regressors, their squares and products. By default `variables` is the model's regressors; it can also name columns that are not in the model. The constant is always included and constant or duplicate columns are dropped as above. The F version is \(F(q,\, n-q-1)\).

### `breusch_godfrey_test(time, nlags)`

Tests the null of no autocorrelation of order up to `nlags`. The residuals are put in the order of the column `time`, and the auxiliary regression is of \(\hat\varepsilon_t\) on the model's own \(X\) and its `nlags` lags (\(\hat\varepsilon_{t-1}, \dots, \hat\varepsilon_{t-m}\), \(m\) = `nlags`). Lags before the start of the sample are set to zero, so the auxiliary regression always uses all \(n\) observations.

\[
\mathrm{LM} = n R^2 \sim \chi^2(m), \quad R^2 = 1 - \frac{\mathrm{SSR}_{\mathrm{aux}}}{\sum_t \hat\varepsilon_t^2}, \qquad
F = \frac{\big(\sum_t \hat\varepsilon_t^2 - \mathrm{SSR}_{\mathrm{aux}}\big)/m}{\mathrm{SSR}_{\mathrm{aux}}/(n-k-m)} \sim F(m,\, n-k-m)
\]

`time` and `nlags` have no defaults. The auxiliary regression uses `X` exactly as in the model, so no constant is added when the model has none.

Here \(R^2\) is measured against \(\sum_t \hat\varepsilon_t^2\), not against the variation of \(\hat\varepsilon\) around its mean (an uncentered \(R^2\)). With an intercept the residuals have mean zero and the two coincide. Without one they differ, and the definition used here is that of R's `lmtest::bgtest` and of Greene; statsmodels adds a constant to the auxiliary regression in that case, so its value differs.
