# OLS: options and results

**OLS**: [Usage](../usage/ols.md) · **Options and results** · [Definitions](../methods/ols.md)

The formulas behind the estimates, the `cov_type` options, the test distributions and the diagnostics are defined in [Methods: OLS](../methods/ols.md).

## Options at a glance

Pass an `OLSOptions` as `options=`. Every field has a default, so `OLSOptions()` is the classical model with an intercept.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"hc0"` to `"hc3"`, `"hac"`, `"cluster"` | `"classical"` | See the table below |
| `cluster` | Column that defines the clusters | `None` | Only with `cov_type="cluster"` |
| `hac_time` | Column that gives the time order | `None` | Required with `cov_type="hac"`; the row order is never assumed to be the time order |
| `hac_lags` | Number of Newey–West lags | `None` (automatic) | Only with `cov_type="hac"` |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `include_intercept` | Add a constant column (`const`) | `True` | Set to `False` if `x` already contains a constant |

Which options go with which `cov_type` (an option set where it is not used raises a `ValidationError`):

| `cov_type` | Required | Optional |
|---|---|---|
| `"classical"`, `"hc0"` to `"hc3"` | | |
| `"hac"` | `hac_time` | `hac_lags` |
| `"cluster"` | `cluster` | |

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.OLS
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.OLSOptions

::: econometricsmodels.OLSResults

::: econometricsmodels.DiagnosticResult

::: econometricsmodels.WhiteTestResult

::: econometricsmodels.BreuschPaganTestResult

::: econometricsmodels.BreuschGodfreyTestResult
