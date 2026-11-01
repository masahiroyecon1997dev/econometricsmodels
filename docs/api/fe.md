# FE: options and results

**FE**: [Usage](../usage/fe.md) · **Options and results** · [Definitions](../methods/fe.md)

`FE` estimates a one-way (entity) or two-way (entity and time) fixed effects ("within") panel regression, with the same constructor and `.fit()` as [OLS](ols.md). `effects` is a required argument: a list of one column (one-way, the entity) or two columns (two-way, the entity then the time period; the panel must be balanced). The number of groups of each column is in `FEResults.n_groups`, keyed by column name, and `fixed_effects()` returns the effects of each column in the same order. There is no intercept, since the effects absorb it. The estimator, covariance estimators, R² statistics and the recovery of the fixed effects are defined in [Definitions: FE](../methods/fe.md).

## Options at a glance

Pass an `FEOptions` as `options=`. `effects` is an argument of `FE`, not an option, and there is no `include_intercept`.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, `"dk"` | `"cluster"` | `"hc0"` is not available |
| `cluster` | Column that defines the clusters | `None` (the first column of `effects`) | Only with `cov_type="cluster"` |
| `dk_time` | Column that defines the time periods of the Driscoll-Kraay kernel | `None` | Required with `cov_type="dk"`; never taken from `effects` |
| `dk_bandwidth` | Kernel bandwidth | `None` (automatic) | Only with `cov_type="dk"` |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |

Which options go with which `cov_type` (an option set where it is not used raises a `ValidationError`):

| `cov_type` | Required | Optional |
|---|---|---|
| `"classical"`, `"hc1"` to `"hc3"` | | |
| `"cluster"` (default) | | `cluster` (default: the first column of `effects`) |
| `"dk"` | `dk_time` | `dk_bandwidth` |

`dk_time` is independent of `effects`: `effects` sets the effects of the model, whereas `dk_time` sets the periods of the Driscoll-Kraay kernel, and it may be the same column as the second column of `effects` or a different one. It cannot be the `y` column or the first column of `effects`.

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.FE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.FEOptions

::: econometricsmodels.FEResults
