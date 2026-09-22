# RE

`RE` estimates a random effects (Swamy-Arora GLS) panel regression. It shares the general API
shape with [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), but — like
[FE](fe.md) — `entity` is a required argument, since the model has no meaning without a panel
structure. Unlike FE, RE keeps an intercept (`REResults.param_names[0]` is always `"const"`) and
supports entity-direction random effects only; two-way RE is not implemented.

## Standard error types

`REOptions.cov_type` defaults to `"cluster"` (clustered on `entity`), the same departure from
OLS's `"classical"` default that [FE](fe.md#standard-error-types) makes, and for the same
reason. Supported values are `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, and `"hac"` —
`"hc0"` is not supported. `"hac"` is the same Driscoll-Kraay panel estimator FE uses, ordered by
`REOptions.time`; unlike `FEOptions`, there is no separate `time_col` since RE has no two-way
structure to disambiguate from the HAC time granularity.

## The Hausman test

`REResults` exposes `hausman_statistic`, `hausman_p_value`, and `hausman_df` directly as
properties — computed automatically inside `fit()` by comparing RE against an internally
estimated, equivalent FE specification, rather than requiring a separate call (unlike FE's
[`fixed_effects()`](fe.md#recovering-the-fixed-effects)). The comparison always uses classical
standard errors regardless of `REOptions.cov_type`. All three are `None` when the internal FE
comparison is unavailable (e.g. `REOptions.time` requests a two-way FE comparison that itself
fails on an unbalanced panel, or the comparison is numerically degenerate) — RE's own result is
still returned normally in that case. See `REResults`'s docstring for the exact fallback
conditions.

`hausman_statistic` is always non-negative — matching R's `plm::phtest`, which takes the absolute
value of the underlying quadratic form unconditionally. In finite samples the difference matrix
being compared can be indefinite, which would otherwise make the raw quadratic form negative.

## Panel R² and `df_resid`

Like FE, `REResults` reports `r_squared_within`/`r_squared_between`/`r_squared_overall` instead
of a single R². `df_resid` follows the plain OLS formula `n - k`, **not** FE's
`n - n_entities - k` — RE is a GLS transform and does not consume entity-dummy degrees of
freedom the way FE's within transformation does.

::: econometricsmodels.RE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.REOptions

::: econometricsmodels.REResults
