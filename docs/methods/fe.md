# FE: definitions

**FE**: [Usage](../usage/fe.md) · [Options and results](../api/fe.md) · **Definitions**

This page defines what `FE` computes: the within estimator, the degrees of freedom, the covariance estimators behind `cov_type`, the test distributions, the panel R² statistics and the recovery of the fixed effects. For the arguments and return values see the [FE API reference](../api/fe.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md). The random effects counterpart is [RE](re.md).

## Model and estimator

The model for entity \(i\) at period \(t\) is

\[
y_{it} = x_{it}'\beta + \alpha_i + \varepsilon_{it} \quad \text{(one-way)}, \qquad
y_{it} = x_{it}'\beta + \alpha_i + \gamma_t + \varepsilon_{it} \quad \text{(two-way)}
\]

with \(n\) observations, \(N\) entities, \(T\) periods and \(k\) slope coefficients. The entities are identified by `entity`; setting `FEOptions.time` adds the time effects \(\gamma_t\). The effects are absorbed, so there is **no intercept** and `const` never appears.

\(\hat\beta\) is the OLS estimator on the within-transformed data, which equals the least-squares dummy-variable estimator:

\[
\tilde y_{it} = y_{it} - \bar y_{i\cdot}, \qquad
\tilde y_{it} = y_{it} - \bar y_{i\cdot} - \bar y_{\cdot t} + \bar y_{\cdot\cdot}\ \text{(two-way)}, \qquad
\hat\beta = (\tilde X'\tilde X)^{-1}\tilde X'\tilde y
\]

The two-way transformation requires a **balanced panel** (every entity observed in every period); a one-way model allows an unbalanced panel and repeated (entity, time) pairs are not checked. The following are rejected rather than handled: an entity (or, for two-way, a period) with a single observation, a regressor with no within-variation (constant over time within every entity), and an outcome with no variation left after the effects are removed. See [Validation and errors](../guide/validation.md).

## Degrees of freedom

\[
\mathrm{df}_{\text{model}} = \begin{cases} k + N & \text{one-way} \\ k + N + T - 1 & \text{two-way} \end{cases}, \qquad
\mathrm{df}_{\text{resid}} = n - \mathrm{df}_{\text{model}}
\]

The effects count as estimated parameters. `fit()` raises an error when \(n \le \mathrm{df}_{\text{model}}\).

## Covariance of the coefficients: `cov_type`

All options are sandwich estimators \((\tilde X'\tilde X)^{-1}\hat S(\tilde X'\tilde X)^{-1}\) on the within-transformed data, scaled as below. Standard errors are the square roots of the diagonal, and the covariance matrix of the coefficients is available as `cov_params` on the results (it holds the slope coefficients only, not the fixed effects). `cov_type` defaults to `"cluster"` on `entity`, following `fixest`: panel data almost always have within-entity serial correlation. `"hc0"` is not available.

| `cov_type` | Covariance |
|---|---|
| `"classical"` | \(\hat\sigma^2(\tilde X'\tilde X)^{-1}\), with \(\hat\sigma^2 = \mathrm{SSR}/\mathrm{df}_{\text{resid}}\) |
| `"hc1"` | \(\hat S = \sum_{it} \hat\varepsilon_{it}^2 \tilde x_{it}\tilde x_{it}'\) times \(n/\mathrm{df}_{\text{resid}}\) |
| `"hc2"`, `"hc3"` | \(\hat S = \sum_{it} c_{it}\hat\varepsilon_{it}^2 \tilde x_{it}\tilde x_{it}'\) with \(c_{it} = 1/(1-h_{it})\) or \(1/(1-h_{it})^2\), where \(h_{it}\) is the leverage of the full dummy-variable regression, not of the within-transformed data |
| `"cluster"` | \(f\,(\tilde X'\tilde X)^{-1}\big(\sum_g s_g s_g'\big)(\tilde X'\tilde X)^{-1}\), \(s_g = \sum_{(i,t)\in g}\tilde x_{it}\hat\varepsilon_{it}\) |
| `"dk"` | Driscoll-Kraay, below |

### Cluster-robust: small-sample factor

With \(G\) clusters (`cluster`, by default the entities), the small-sample factor \(f\) is

\[
f = \frac{G}{G-1}\cdot\frac{n-1}{n-K}
\]

following `fixest`'s default `ssc()`. The effective number of parameters \(K\) depends on whether the fixed effects are *nested* in the clusters (each level of the effect lies in a single cluster):

| Fixed effects | \(K\) |
|---|---|
| All nested in the clusters (the default: one-way, clustered by entity) | \(\mathrm{df}_{\text{model}}\) minus the number of nested effect levels, plus the number of effect dimensions |
| Some nested (typically two-way, clustered by entity) | \(\mathrm{df}_{\text{model}}\) minus the nested effect levels, plus the number of nested dimensions |
| None nested (Stata's `xtreg, fe` situation) | \(\mathrm{df}_{\text{model}}\) |

and \(K\) is never allowed to be below \(k+1\). `linearmodels` instead uses \(n/(n-\text{extra\_df}-k)\) with no \(G/(G-1)\) factor, so cluster standard errors differ from it.

`fit()` raises an error with fewer than two clusters or when there are not more clusters than slope coefficients (the joint test is then undefined). It also rejects the degenerate case of two clusters that contain exactly one observation of each absorbed effect level, where the clustered covariance is identically zero.

### Driscoll-Kraay: `"dk"`

A panel HAC estimator that is robust to correlation across entities as well as within them (`fixest`'s `vcov = "DK"`, Stata's `xtscc`). With the period scores \(h_t = \sum_i \tilde x_{it}\hat\varepsilon_{it}\) over \(T_d\) periods of the `dk_time` column:

\[
\hat S = \hat\Gamma_0 + \sum_{l=1}^{L} w_l\big(\hat\Gamma_l + \hat\Gamma_l'\big), \quad
\hat\Gamma_l = \sum_{t=l+1}^{T_d} h_t h_{t-l}', \quad w_l = 1 - \frac{l}{L+1}
\]

and the covariance is \(\frac{T_d}{T_d-1}\cdot\frac{n-1}{n-\mathrm{df}_{\text{model}}}\,(\tilde X'\tilde X)^{-1}\hat S(\tilde X'\tilde X)^{-1}\).

- `dk_time` is required and is never taken from `time`, so the fixed effects and the kernel may use different time granularities (for example quarterly effects with yearly periods for the kernel).
- The bandwidth \(L\) is `dk_bandwidth` when given (\(0 \le L < T_d\)), otherwise \(\lfloor 4\,(T_d/100)^{2/9}\rfloor\); the value used is `dk_bandwidth_used`. Note that the rule uses the number of periods \(T_d\), not \(n\).
- The periods are ordered by the **values** of `dk_time` (numerically, chronologically, by category order for an `Enum`, alphabetically for strings), never by the order of the rows. See [Accepted data](../guide/accepted-data.md#order-of-time-periods).
- It needs more periods than slope coefficients (\(T_d > k\)), because \(\hat S\) has rank at most \(T_d - 1\).
- With `dk_bandwidth` equal to \(T_d - 1\), `fixest` drops the last lag term because of an off-by-one in its compiled code. This package keeps the standard Bartlett kernel, so the two differ there only; smaller bandwidths agree.

## Test distributions and confidence intervals

Every `cov_type` uses the **t distribution**. The degrees of freedom are

| `cov_type` | degrees of freedom |
|---|---|
| `"classical"`, `"hc1"` to `"hc3"` | \(\mathrm{df}_{\text{resid}}\) |
| `"cluster"` | \(G-1\) |
| `"dk"` | \(T_d-1\) |

The value used is returned as `stat_df`. The overall F test (`f_statistic`) is the Wald test that all slope coefficients are zero, \(F = \hat\beta'\hat\Sigma^{-1}\hat\beta / k \sim F(k, \mathrm{df})\) with \(\hat\Sigma\) the covariance chosen by `cov_type` and the same degrees of freedom. The fixed effects themselves are not part of the tested hypothesis.

## Fit statistics

Three R² values are reported, because "the" R² is not well defined once effects are involved. With \(\hat\beta\) applied to the respective variables:

| Statistic | Definition |
|---|---|
| `r_squared_within` | \(1 - \sum \hat\varepsilon_{it}^2 / \sum \tilde y_{it}^2\) on the within-transformed variables, with the transformation that was used (entity only, or entity and time) |
| `r_squared_between` | \(1 - \sum_i (\bar y_{i\cdot} - \bar x_{i\cdot}'\hat\beta)^2 / \sum_i \bar y_{i\cdot}^2\) on the entity means |
| `r_squared_overall` | \(1 - \sum (y_{it} - x_{it}'\hat\beta)^2 / \sum y_{it}^2\) on the original variables |

The between and overall values use uncentered total sums of squares, because the model has no constant, and the between value is not weighted by the number of observations per entity; both agree with `linearmodels`. The within value for a two-way model uses both effects, whereas `linearmodels` always removes entity means only.

The log-likelihood is computed from the within residuals as for [OLS](ols.md#fit-statistics), and \(\mathrm{AIC} = -2\ell + 2\,\mathrm{df}_{\text{model}}\), \(\mathrm{BIC} = -2\ell + \mathrm{df}_{\text{model}}\ln n\).

## Fixed effects: `fixed_effects()`

The effects are recovered after estimation from the coefficients:

- **One-way:** \(\hat\alpha_i = \bar y_{i\cdot} - \bar x_{i\cdot}'\hat\beta\), returned as a `dict` keyed by entity.
- **Two-way:** the split of the overall level between \(\alpha_i\) and \(\gamma_t\) is not identified, so the first period in time order is the reference with \(\hat\gamma = 0\) and the entity effects absorb the level. The result has the keys `"entity"` and `"time"`, and the periods are listed in time order. `fixest::fixef()` uses the first period in the order of the rows, so its values agree only when that is also the first period in time order.

`FE` has no `predict()`, `augment()` or `summary()`.
