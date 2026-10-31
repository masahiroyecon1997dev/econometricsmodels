# WLS: usage

**WLS**: **Usage** · [Options and results](../api/wls.md) · [Definitions](../methods/wls.md)

`WLS` is `OLS` with an added `weight` argument (the column name of the weight column). Weights are treated as analytic weights proportional to the inverse of the variance, and do not need to be normalized. Every weight must be finite and strictly positive; a zero, negative, missing, NaN or infinite weight raises a `ValidationError`.

```python
import polars as pl
from econometricsmodels import WLS

df = pl.DataFrame(
    {
        "y": [2.1, 3.9, 6.2, 8.1, 9.8],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "w": [1.0, 2.0, 1.0, 2.0, 1.0],
    }
)

result = WLS(df, y="y", x=["x1"], weight="w").fit()

print(result.params)
print(result.std_errors)
```

Estimation options are set with `WLSOptions`: `cov_type`, `include_intercept`, `confidence_level`, `cluster`, `hac_lags` and `hac_time`. [Switching the type of standard error](ols.md#switching-the-type-of-standard-error) shows how to choose the standard errors; the options mean the same here. See [Options and results](../api/wls.md) for the `weight` argument and [Methods: WLS](../methods/wls.md) for how the weights enter the estimates and fit statistics.

The post-estimation tests of `OLSResults` (`white_test()`, `breusch_pagan_test()` and `breusch_godfrey_test()`) are not available for `WLSResults`. `WLSResults.predict()` and `WLSResults.augment()` work exactly like their `OLSResults` counterparts (see [Predicted values](ols.md#predicted-values)); weights play no role in either the training-data or out-of-sample case.

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name: the covariance of `x1` and `x2` is `result.cov_params["x1"]["x2"]`, and the standard errors are the square roots of its diagonal. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.
