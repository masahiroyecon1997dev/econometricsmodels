# RE: options and results

**RE**: [Usage](../usage/re.md) · **Options and results** · [Definitions](../methods/re.md)

`RE` estimates a random effects (GLS) panel regression. It shares the API shape of [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), but — like [FE](fe.md) — `entity` is a required argument. Unlike FE, RE keeps an intercept (`REResults.param_names[0]` is always `"const"`) and accepts time-invariant regressors. It estimates entity random effects, or entity and time random effects on a balanced panel when `REOptions.time` is set.

`cov_type` defaults to `"cluster"` on `entity`, and `"hc0"` is not supported. `"dk"` requires `REOptions.dk_time`, which is a different option from `time`. `REOptions.variance_method` selects `"swamy_arora"` (default), `"wallace_hussain"` or `"amemiya"`. The Hausman test comparing RE with the equivalent FE model is computed by `fit()` and exposed as `hausman_statistic`, `hausman_p_value` and `hausman_df`.

The variance components, the GLS transformation, the covariance estimators, the test distributions, the R² statistics, correlated random effects and the Hausman test are defined in [Methods: RE](../methods/re.md).

## Options at a glance

`entity` is an argument of `RE`, not an option.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, `"dk"` | `"cluster"` | `"hc0"` is not available |
| `cluster` | Column that defines the clusters | `None` (the `entity` column) | Only with `cov_type="cluster"` |
| `dk_time` | Column that defines the time periods of the Driscoll–Kraay kernel | `None` | Required with `cov_type="dk"`; never taken from `time` |
| `dk_bandwidth` | Kernel bandwidth | `None` (automatic) | Only with `cov_type="dk"` |
| `time` | Column of the time identifier; sets two-way effects | `None` (one-way) | Two-way needs a balanced panel |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `variance_method` | `"swamy_arora"`, `"wallace_hussain"` or `"amemiya"` | `"swamy_arora"` | See [Definitions](../methods/re.md#variance-components-variance_method) |

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.RE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.REOptions

::: econometricsmodels.REResults
