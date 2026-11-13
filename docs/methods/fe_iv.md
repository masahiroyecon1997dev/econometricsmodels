# FEIV: definitions

**FEIV**: [Usage](../usage/fe_iv.md) · [Options and results](../api/fe_iv.md) · **Definitions**

This page defines what `FEIV` computes: fixed effects two-stage least squares (FE-2SLS), that is, the [FE](fe.md) within transformation followed by the [IV](iv.md) two-stage least squares estimator. Where a definition is the same as in `FE` or `IV`, this page links to it and gives only what is different: the degrees of freedom that the absorbed fixed effects take away, the leverage of HC2 and HC3, the diagnostics, the fit statistics, the first-stage regressions, the recovery of the fixed effects and the differences from `fixest`. For the arguments and return values see the [FEIV API reference](../api/fe_iv.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md).

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
