# RE

`RE` estimates a random effects (GLS) panel regression. It shares the API shape of [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), but — like [FE](fe.md) — `entity` is a required argument. Unlike FE, RE keeps an intercept (`REResults.param_names[0]` is always `"const"`) and accepts time-invariant regressors. It estimates entity random effects, or entity and time random effects on a balanced panel when `REOptions.time` is set.

`cov_type` defaults to `"cluster"` on `entity`, and `"hc0"` is not supported. `"dk"` requires `REOptions.dk_time`, which is a different option from `time`. `REOptions.variance_method` selects `"swamy_arora"` (default), `"wallace_hussain"` or `"amemiya"`. The Hausman test comparing RE with the equivalent FE model is computed by `fit()` and exposed as `hausman_statistic`, `hausman_p_value` and `hausman_df`.

The variance components, the GLS transformation, the covariance estimators, the test distributions, the R² statistics, correlated random effects and the Hausman test are defined in [Methods: RE](../methods/re.md). Getting Started has [examples](../getting-started.md#re-random-effects-panel-regression).

::: econometricsmodels.RE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.REOptions

::: econometricsmodels.REResults
