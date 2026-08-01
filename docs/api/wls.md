# WLS

`WLS` is implemented by applying `OLS`'s solver directly to `sqrt(weight)`-transformed data, so the normal-equations solver, the general form of standard errors (HC0-3 / cluster / HAC), and the estimation options (`OLSOptions`) are shared with [OLS](ols.md). This page only covers the semantics of weights (analytic weights) and the API specific to WLS.

## Weights (analytic weight)

The `weight` argument specifies the column name of the weight column in `data`. Weights are treated as analytic weights proportional to the inverse of the variance, and do not need to be normalized (e.g. to sum to 1). Values less than or equal to 0, missing values, or NaN raise a `ValidationError`.

::: econometricsmodels.WLS
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.WlsResults
