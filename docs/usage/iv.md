# IV: usage

**IV**: **Usage** · [Options and results](../api/iv.md) · [Definitions](../methods/iv.md)

`IV` estimates a linear model with endogenous regressors. Independent variables are split into `x_exog` (exogenous) and `x_endog` (endogenous), plus `instruments` (excluded instruments — at least one per endogenous variable for identification).

```python
import polars as pl
from econometricsmodels import IV

df = pl.DataFrame(
    {
        "y": [1.0, 2.4, 2.9, 4.3, 5.1, 5.8, 7.2, 7.9],
        "endog1": [1.1, 1.9, 2.8, 3.6, 4.4, 5.3, 6.0, 6.9],
        "z1": [0.9, 2.1, 2.7, 3.9, 4.2, 5.6, 5.8, 7.1],
    }
)

result = IV(df, y="y", x_exog=[], x_endog=["endog1"], instruments=["z1"]).fit()

print(result.params)  # {"const": ..., "endog1": ...}
print(result.std_errors)  # {"const": ..., "endog1": ...}
print(result.r_squared)
```

`IVOptions.estimator` selects `"2sls"` (default) or `"gmm"`. `cov_type` supports the same range as [OLS](ols.md#switching-the-type-of-standard-error); for `estimator="gmm"`, a separate `gmm_weight_type` selects the weight matrix used for point estimation. See [Options and results](../api/iv.md) for the full list of options.

## Diagnostics and first-stage results

```python
print(result.weak_instrument_f_statistics)  # {"endog1": ...}
print(
    result.overid_statistic, result.overid_p_value
)  # None, None (just-identified)
print(
    result.wu_hausman_statistic, result.wu_hausman_p_value
)  # estimator="2sls" only

first_stage = result.first_stage()
print(
    first_stage["endog1"].params
)  # OLSResults for endog1 ~ x_exog + instruments
```

See [Methods: IV](../methods/iv.md#diagnostics) for what each diagnostic tests, and the [Options and results](../api/iv.md) for when it is `None`.

## Covariance matrix of the coefficients

The covariance matrix of the coefficients is available as `result.cov_params`, in the same form for every method. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients).
