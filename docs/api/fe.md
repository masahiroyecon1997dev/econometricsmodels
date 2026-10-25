# FE

`FE` estimates a one-way (entity) or two-way (entity and time) fixed effects ("within") panel regression. It shares the API shape of [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), but `entity` is a required argument rather than an option, since a fixed effects model has no meaning without a panel structure. Two-way effects are selected with `FEOptions.time` and need a balanced panel.

There is no intercept: the within transformation removes it, so `FEOptions` has no `include_intercept` and `FEResults.param_names` never includes `"const"`. `cov_type` defaults to `"cluster"` on `entity` (not `"classical"`), and `"hc0"` is not supported. `"dk"` (Driscoll–Kraay) requires `FEOptions.dk_time`, which is independent of `time`.

The estimator, degrees of freedom, covariance estimators, test distributions, the three R² statistics and the recovery of the fixed effects (`FEResults.fixed_effects()`) are defined in [Methods: FE](../methods/fe.md). Getting Started has [examples](../getting-started.md#fe-fixed-effects-panel-regression).

::: econometricsmodels.FE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.FEOptions

::: econometricsmodels.FEResults
