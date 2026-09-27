# RE

`RE` estimates a random effects (Swamy-Arora GLS) panel regression. It shares the general API
shape with [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), but — like
[FE](fe.md) — `entity` is a required argument, since the model has no meaning without a panel
structure. Unlike FE, RE keeps an intercept (`REResults.param_names[0]` is always `"const"`) and
supports entity-direction random effects only; two-way RE is not implemented.

## Standard error types

`REOptions.cov_type` defaults to `"cluster"` (clustered on `entity`), the same departure from
OLS's `"classical"` default that [FE](fe.md#standard-error-types) makes, and for the same
reason. Supported values are `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, and `"dk"` —
`"hc0"` is not supported. `"dk"` is the same Driscoll-Kraay panel estimator FE uses, ordered by
`REOptions.time`; unlike `FEOptions`, there is no separate `dk_time` since RE has no two-way
structure to disambiguate from the HAC time granularity.

## The Hausman test

`REResults` exposes `hausman_statistic`, `hausman_p_value`, and `hausman_df` directly as
properties — computed automatically inside `fit()`, rather than requiring a separate call
(unlike FE's [`fixed_effects()`](fe.md#recovering-the-fixed-effects)).

The test is the regression-based (auxiliary regression) version (Wooldridge 2010, section
10.7.3; equivalent to R's `plm::phtest(method = "aux", effect = "individual")`): the
quasi-demeaned `y` is regressed on a constant, the quasi-demeaned regressors and the
within-transformed regressors, and the within-transformed coefficients are tested jointly for
zero. Its properties:

- The statistic is always non-negative (no indefinite variance-difference problem).
- The comparison is always against **one-way** (entity) fixed effects, the same structure as RE
  itself. `REOptions.time` is used only as the Driscoll-Kraay time ordering (`cov_type="dk"`)
  and does not affect the test; specifying it with any other `cov_type` raises
  `ValidationError`.
- It always uses classical (non-robust) covariance regardless of `REOptions.cov_type`.
- All three are `None` only when the auxiliary regression cannot be formed (no slope
  coefficients, rank-deficient design, singular Wald test); RE's own result is still returned
  normally. See `REResults`'s docstring.

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
