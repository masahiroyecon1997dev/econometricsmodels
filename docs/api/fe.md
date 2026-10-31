# FE: options and results

**FE**: [Usage](../usage/fe.md) · **Options and results** · [Definitions](../methods/fe.md)

`FE` estimates a one-way (entity) or two-way (entity and time) fixed effects ("within") panel regression. It shares the API shape of [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), but `entity` is a required argument rather than an option, since a fixed effects model has no meaning without a panel structure. Two-way effects are selected with `FEOptions.time` and need a balanced panel.

There is no intercept: the within transformation removes it, so `FEOptions` has no `include_intercept` and `FEResults.param_names` never includes `"const"`. `cov_type` defaults to `"cluster"` on `entity` (not `"classical"`), and `"hc0"` is not supported. `"dk"` (Driscoll-Kraay) requires `FEOptions.dk_time`, which is independent of `time`.

The estimator, degrees of freedom, covariance estimators, test distributions, the three R² statistics and the recovery of the fixed effects (`FEResults.fixed_effects()`) are defined in [Methods: FE](../methods/fe.md).

## Options at a glance

`entity` is an argument of `FE`, not an option. There is no `include_intercept`: the effects absorb the constant.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, `"dk"` | `"cluster"` | `"hc0"` is not available |
| `cluster` | Column that defines the clusters | `None` (the `entity` column) | Only with `cov_type="cluster"` |
| `dk_time` | Column that defines the time periods of the Driscoll-Kraay kernel | `None` | Required with `cov_type="dk"`; never taken from `time` |
| `dk_bandwidth` | Kernel bandwidth | `None` (automatic) | Only with `cov_type="dk"` |
| `time` | Column of the time identifier; sets two-way effects | `None` (one-way) | Two-way needs a balanced panel |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.FE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.FEOptions

::: econometricsmodels.FEResults
