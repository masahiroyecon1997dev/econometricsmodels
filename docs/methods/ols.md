# OLS: definitions

This page defines what `OLS` computes: the estimator, the covariance estimators behind `cov_type`, the test distributions, the fit statistics and the post-estimation diagnostics. For the arguments and return values see the [OLS API reference](../api/ols.md); for how the test distributions differ from R, statsmodels and linearmodels see [Inference conventions](../guide/inference-conventions.md).

## Model and estimator

The model is `y = Xβ + ε`, with `n` observations and `k` coefficients. `X` holds the columns listed in `x`. With `include_intercept=True` (the default) a constant column named `const` is added in front of them.

```
β̂ = (X'X)⁻¹ X'y          ε̂ = y − Xβ̂          σ̂² = SSR / (n − k)
```

`SSR = Σ ε̂ᵢ²`. A design with perfectly collinear columns has no unique solution, so it raises an error instead of being estimated on a reduced set of columns (see [Validation and errors](../guide/validation.md)).

## Covariance of the coefficients: `cov_type`

Every option below is a sandwich estimator `(X'X)⁻¹ Ŝ (X'X)⁻¹` for the covariance of `β̂`, except `"classical"`. Standard errors are the square roots of its diagonal.

### `"classical"` (default)

```
Var(β̂) = σ̂² (X'X)⁻¹
```

This assumes homoskedastic, uncorrelated errors.

### `"hc0"` to `"hc3"`: heteroskedasticity-robust

`Ŝ = Σᵢ cᵢ ε̂ᵢ² xᵢxᵢ'`, where `xᵢ` is the `i`-th row of `X` and the weight `cᵢ` depends on the type:

| `cov_type` | `cᵢ` |
|---|---|
| `"hc0"` | `1` |
| `"hc1"` | `n / (n − k)` |
| `"hc2"` | `1 / (1 − hᵢᵢ)` |
| `"hc3"` | `1 / (1 − hᵢᵢ)²` |

`hᵢᵢ = xᵢ'(X'X)⁻¹xᵢ` is the leverage of observation `i`. HC2 and HC3 grow without bound as a leverage approaches 1, so a single extreme observation can dominate them.

### `"hac"`: Newey–West

Heteroskedasticity- and autocorrelation-consistent standard errors with Bartlett weights:

```
Ŝ = Ŝ₀ + Σ_{l=1..L} wₗ (Ŝₗ + Ŝₗ')        wₗ = 1 − l / (L + 1)
Ŝₗ = Σ_{t=l+1..n} ε̂ₜ ε̂ₜ₋ₗ xₜxₜ₋ₗ'
```

The observations are put in time order by the column named in `hac_time` (required: row order is never assumed to be time order, and the values must all differ). The number of lags `L` is `hac_lags` when given, with `0 ≤ L < n`. Otherwise it is chosen as

```
L = floor(4 · (n / 100)^(2/9))
```

This rule depends only on `n`, not on the data. The value used is returned as `hac_lags_used`.

### `"cluster"`: cluster-robust

With observations grouped by the column named in `cluster` (`G` groups):

```
Ŝ = Σ_g s_g s_g'        s_g = Σ_{i∈g} ε̂ᵢ xᵢ
```

and the covariance is multiplied by the small-sample factor `G/(G − 1) · (n − 1)/(n − k)`. The factor is always applied. `fit()` raises an error with fewer than two clusters, and also when there are not more clusters than slope coefficients, because the overall F test (below) is then not defined.

## Test distributions and confidence intervals

Every `cov_type` uses the **t distribution**; the package never switches to the normal distribution for robust standard errors.

| | degrees of freedom |
|---|---|
| all `cov_type` except `"cluster"` | `n − k` |
| `"cluster"` | `G − 1` |

For coefficient `j`, the t statistic is `β̂ⱼ / se(β̂ⱼ)`, the p-value is two-sided, and the confidence interval is `β̂ⱼ ± t(1 − α/2, df) · se(β̂ⱼ)` with `1 − α = confidence_level`. The degrees of freedom actually used are returned as `stat_df`; use it, not `df_resid`, to recompute a p-value. `df_resid = n − k` regardless of `cov_type`.

## Overall F test

Let `β_s` be the `q` slope coefficients (all except the constant) and `Σ_s` the corresponding block of the covariance matrix chosen by `cov_type`:

```
F = β_s' Σ_s⁻¹ β_s / q          F ~ F(q, df)
```

`df` is the t degrees of freedom above (`G − 1` for `"cluster"`). With `"classical"` this is the textbook F test; with the other `cov_type` values it is a robust Wald test.

## Fit statistics

With an intercept, `SST = Σ(yᵢ − ȳ)²`; without one, `SST = Σ yᵢ²`.

```
R² = 1 − SSR / SST
adjusted R² = 1 − (n − c)/(n − k) · (1 − R²)        c = 1 with an intercept, 0 without
log-likelihood = −n/2 · (ln(2π) + ln(SSR / n) + 1)
AIC = −2·log-likelihood + 2k          BIC = −2·log-likelihood + k·ln(n)
```

The log-likelihood uses the maximum-likelihood variance `SSR / n`, not `σ̂²`.

Two situations in which these statistics are undefined are rejected rather than reported: an outcome that is constant in a model with an intercept (`SST = 0`), and a perfect fit, where the residuals are zero up to rounding (`SSR ≤ k · ε · SST`, `ε` the machine epsilon). See [Validation and errors](../guide/validation.md).

## Predictions

`predict()` returns `ŷ = x β̂`. Without `new_data` it is evaluated on the estimation data. With `new_data` the frame must contain the columns of `x` (matched by name, in any order); do not include `const`, it is added automatically when the model has an intercept. `augment()` returns the same values as a `"predicted"` column attached to the source data.

## Post-estimation diagnostics

The diagnostics are never computed by `fit()`; call them on the results. None of them depends on `cov_type`, and each returns a result with `statistic`, `p_value`, `df`, `df_denom` and `distribution`.

Each test regresses a residual-based quantity on a set of auxiliary columns, and offers two versions chosen by `statistic`: `"lm"` (`n · R²` of the auxiliary regression, chi-squared with `q` degrees of freedom) and `"f"` (the F test that the auxiliary slopes are all zero). `q` is the number of auxiliary columns excluding the constant.

### `white_test()`

Tests the null of homoskedastic errors, in the form that does not assume normal errors (Koenker's studentized version). The auxiliary regression is of `ε̂²` on the regressors, their squares and their pairwise products. The F version is `F(q, n − q − 1)`.

- The auxiliary regression always has a constant, even for a model fitted with `include_intercept=False`; `aux_terms` lists the terms, starting with `const`.
- Terms that are constant, or numerically identical to an earlier term (for example the square of a dummy variable), are dropped and listed in `dropped_terms`. `df` counts the terms that remain.

### `breusch_pagan_test(variables=None)`

The same Koenker version of the Breusch–Pagan test, but the auxiliary regression of `ε̂²` is on `variables` (a list of column names) instead of the regressors, their squares and products. By default `variables` is the model's regressors; it can also name columns that are not in the model. The constant is always included and constant or duplicate columns are dropped as above. The F version is `F(q, n − q − 1)`.

### `breusch_godfrey_test(time, nlags)`

Tests the null of no autocorrelation of order up to `nlags`. The residuals are put in the order of the column `time`, and the auxiliary regression is of `ε̂` on the model's own `X` and its `nlags` lags (`ε̂ₜ₋₁, …, ε̂ₜ₋ₘ`, `m = nlags`). Lags before the start of the sample are set to zero, so the auxiliary regression always uses all `n` observations.

```
LM = n · R²  ~  χ²(m)
F  = ((Σε̂² − SSR_aux) / m) / (SSR_aux / (n − k − m))  ~  F(m, n − k − m)
```

`time` and `nlags` have no defaults. The auxiliary regression uses `X` exactly as in the model, so no constant is added when the model has none.
