# FE: options and results

**FE**: [Usage](../usage/fe.md) · **Options and results** · [Definitions](../methods/fe.md)

`FE` estimates a one-way (entity) or two-way (entity and time) fixed effects ("within") panel regression, with the same constructor and `.fit()` as [OLS](ols.md). `entity` is a required argument. There is no intercept, since the effects absorb it. The estimator, covariance estimators, R² statistics and the recovery of the fixed effects are defined in [Definitions: FE](../methods/fe.md).

## Options at a glance

Pass an `FEOptions` as `options=`. `entity` is an argument of `FE`, not an option, and there is no `include_intercept`.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, `"dk"` | `"cluster"` | `"hc0"` is not available |
| `cluster` | Column that defines the clusters | `None` (the `entity` column) | Only with `cov_type="cluster"` |
| `dk_time` | Column that defines the time periods of the Driscoll-Kraay kernel | `None` | Required with `cov_type="dk"`; never taken from `time` |
| `dk_bandwidth` | Kernel bandwidth | `None` (automatic) | Only with `cov_type="dk"` |
| `time` | Column of the time identifier; sets two-way effects | `None` (one-way) | Two-way needs a balanced panel |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |

Which options go with which `cov_type` (an option set where it is not used raises a `ValidationError`):

| `cov_type` | Required | Optional |
|---|---|---|
| `"classical"`, `"hc1"` to `"hc3"` | | |
| `"cluster"` (default) | | `cluster` (default: `entity`) |
| `"dk"` | `dk_time` | `dk_bandwidth` |

`time` is independent of `cov_type`: it sets the effects of the model, whereas `dk_time` sets the periods of the Driscoll-Kraay kernel, and the two may be different columns.

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.FE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.FEOptions

::: econometricsmodels.FEResults
