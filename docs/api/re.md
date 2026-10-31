# RE: options and results

**RE**: [Usage](../usage/re.md) · **Options and results** · [Definitions](../methods/re.md)

`RE` estimates a random effects (GLS) panel regression, with the same constructor and `.fit()` as [OLS](ols.md). Like [FE](fe.md), `effects` is a required argument: a list of one column (entity effects) or two columns (entity then time effects, on a balanced panel), with at most two columns. In a two-way model the order matters: the first column gives `sigma2_u` and `theta_entity`, the second gives `sigma2_lambda` and `theta_time`. The first column is also the default clustering column, so swapping the columns changes the default standard errors and the Hausman test. The number of groups of each column is in `REResults.n_groups`, a dictionary keyed by column name in the order of `effects`. Unlike FE, RE keeps an intercept (`REResults.param_names[0]` is always `"const"`) and accepts time-invariant regressors. The variance components, the GLS transformation, the covariance estimators and the Hausman test are defined in [Definitions: RE](../methods/re.md). The Hausman test is computed by `fit()` and exposed as `hausman_statistic`, `hausman_p_value` and `hausman_df`.

## Options at a glance

Pass an `REOptions` as `options=`. `effects` is an argument of `RE`, not an option.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, `"dk"` | `"cluster"` | `"hc0"` is not available |
| `cluster` | Column that defines the clusters | `None` (the first column of `effects`) | Only with `cov_type="cluster"` |
| `dk_time` | Column that defines the time periods of the Driscoll-Kraay kernel | `None` | Required with `cov_type="dk"`; never taken from `effects` |
| `dk_bandwidth` | Kernel bandwidth | `None` (automatic) | Only with `cov_type="dk"` |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `variance_method` | `"swamy_arora"`, `"wallace_hussain"` or `"amemiya"` | `"swamy_arora"` | See [Definitions](../methods/re.md#variance-components-variance_method) |

Which options go with which `cov_type` (an option set where it is not used raises a `ValidationError`):

| `cov_type` | Required | Optional |
|---|---|---|
| `"classical"`, `"hc1"` to `"hc3"` | | |
| `"cluster"` (default) | | `cluster` (default: the first column of `effects`) |
| `"dk"` | `dk_time` | `dk_bandwidth` |

`variance_method` applies whatever the `cov_type`.

`dk_time` is independent of `effects`: `effects` sets the effects of the model, whereas `dk_time` sets the periods of the Driscoll-Kraay kernel. It may be the same column as the second column of `effects` or a different one. It cannot be the `y` column or the first column of `effects`.

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.RE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.REOptions

::: econometricsmodels.REResults
