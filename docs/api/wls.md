# WLS

`WLS` fits a weighted least squares regression: `OLS` with a `weight` column. The standard error types (`cov_type`: classical, HC0-3, cluster, HAC) are the ones available for [OLS](ols.md). For the definitions behind the estimates, standard errors and fit statistics, see [Methods: WLS](../methods/wls.md).

## Weights (analytic weight)

The `weight` argument specifies the column name of the weight column in `data`. Weights are treated as analytic weights proportional to the inverse of the variance, and do not need to be normalized (e.g. to sum to 1). Values less than or equal to 0, missing values, or NaN raise a `ValidationError`.

::: econometricsmodels.WLS
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.WLSOptions

::: econometricsmodels.WLSResults
