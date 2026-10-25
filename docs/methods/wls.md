# WLS: definitions

This page defines what `WLS` computes. WLS is least squares on `√w`-scaled data, so the covariance estimators and test distributions are those defined on the [OLS page](ols.md), applied to the scaled data. Only what the weights change is described here. For the arguments and return values see the [WLS API reference](../api/wls.md).

## Model and estimator

The model is `y = Xβ + ε` with `Var(εᵢ) = σ² / wᵢ`: the weights are **analytic weights**, proportional to the inverse of the error variance. They need not be normalized. Frequency weights and probability (sampling) weights are not supported.

```
β̂ = (X'WX)⁻¹ X'Wy          W = diag(w₁, …, wₙ)
```

Equivalently, `β̂` is the OLS estimator of `ỹ` on `X̃`, where `ỹᵢ = √wᵢ · yᵢ` and `x̃ᵢ = √wᵢ · xᵢ`. The constant column added by `include_intercept=True` is scaled too, so it becomes the column `√wᵢ`.

The weight column is given by `weight`. Every weight must be finite and strictly positive; a zero, negative, missing or non-finite weight raises an error rather than dropping the observation. The weight column may not be the same column as `y`; it may also appear in `x`.

## Covariance of the coefficients: `cov_type`

Every `cov_type` is defined as on the OLS page, evaluated on the scaled quantities `x̃ᵢ` and `ε̃ᵢ = √wᵢ · ε̂ᵢ`:

```
σ̂² = Σ wᵢ ε̂ᵢ² / (n − k)          Var(β̂) = σ̂² (X'WX)⁻¹        (classical)
```

In the sandwich estimators the weights therefore enter squared. For example, HC0 is

```
Ŝ = Σᵢ ε̃ᵢ² x̃ᵢx̃ᵢ' = Σᵢ wᵢ² ε̂ᵢ² xᵢxᵢ'
```

The weights play no part in the choice of the HAC lag length, in the grouping by `cluster`, or in the small-sample factor and degrees of freedom: those depend only on `n`, `k` and `G`, exactly as for OLS.

## Test distributions and the overall F test

As for OLS, every `cov_type` uses the t distribution with `n − k` degrees of freedom (`G − 1` for `"cluster"`), and the overall F test is the Wald form `β_s' Σ_s⁻¹ β_s / q` computed from the scaled data.

## Fit statistics

The sums of squares are weighted. `SSR = Σ wᵢ ε̂ᵢ²`, and the total sum of squares uses the weighted mean `ȳ_w = Σ wᵢ yᵢ / Σ wᵢ`:

```
SST = Σ wᵢ (yᵢ − ȳ_w)²   (with an intercept)        SST = Σ wᵢ yᵢ²   (without)
R² = 1 − SSR / SST
```

Adjusted R² has the same form as for OLS. The log-likelihood of the weighted model includes the Jacobian of the scaling:

```
log-likelihood = −n/2 · (ln(2π) + ln(SSR / n) + 1) + ½ Σ ln wᵢ
```

AIC and BIC are computed from it as for OLS.

## Residuals and predictions

`residuals` are on the **original scale**, `yᵢ − xᵢ'β̂`, not the weighted residuals `√wᵢ · ε̂ᵢ` that the standard errors are built from. This matches `statsmodels`' `.resid` and keeps residual plots comparable with OLS.

`predict()` returns `xᵢ'β̂` on the original scale. The weights play no role in it, with or without `new_data`: a new observation needs no weight, and none is applied to the estimation data either.
