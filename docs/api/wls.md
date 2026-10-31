# WLS: options and results

**WLS**: [Usage](../usage/wls.md) · **Options and results** · [Definitions](../methods/wls.md)

`WLS` fits a weighted least squares regression: `OLS` with a `weight` column. The standard error types (`cov_type`: classical, HC0-3, cluster, HAC) are the ones available for [OLS](ols.md). For the definitions behind the estimates, standard errors and fit statistics, see [Methods: WLS](../methods/wls.md).

## Weights (analytic weight)

The `weight` argument specifies the column name of the weight column in `data`. Weights are treated as analytic weights proportional to the inverse of the variance, and do not need to be normalized (e.g. to sum to 1). Values less than or equal to 0, missing values, or NaN raise a `ValidationError`.

## Options at a glance

`WLSOptions` has the same fields as `OLSOptions`; the weight column is the `weight` argument of `WLS`, not an option.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"hc0"` to `"hc3"`, `"hac"`, `"cluster"` | `"classical"` | See the table below |
| `cluster` | Column that defines the clusters | `None` | Only with `cov_type="cluster"` |
| `hac_time` | Column that gives the time order | `None` | Required with `cov_type="hac"`; the row order is never assumed to be the time order |
| `hac_lags` | Number of Newey-West lags | `None` (automatic) | Only with `cov_type="hac"` |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `include_intercept` | Add a constant column (`const`) | `True` | Set to `False` if `x` already contains a constant |

Which options go with which `cov_type` (an option set where it is not used raises a `ValidationError`):

| `cov_type` | Required | Optional |
|---|---|---|
| `"classical"`, `"hc0"` to `"hc3"` | | |
| `"hac"` | `hac_time` | `hac_lags` |
| `"cluster"` | `cluster` | |

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.WLS
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.WLSOptions

::: econometricsmodels.WLSResults
