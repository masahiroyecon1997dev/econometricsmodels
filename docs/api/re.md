# RE

`RE` estimates a random effects (GLS) panel regression. The variance components are estimated
by Swamy-Arora (the default), Wallace-Hussain or Amemiya (see [Variance components](#variance-components)). It shares the general API
shape with [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), but — like
[FE](fe.md) — `entity` is a required argument, since the model has no meaning without a panel
structure. Unlike FE, RE keeps an intercept (`REResults.param_names[0]` is always `"const"`) and
estimates entity-direction random effects, or entity and time random effects on a balanced panel when
`REOptions.time` is set.

## Standard error types

`REOptions.cov_type` defaults to `"cluster"` (clustered on `entity`), the same departure from
OLS's `"classical"` default that [FE](fe.md#standard-error-types) makes, and for the same
reason. Supported values are `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, and `"dk"` —
`"hc0"` is not supported. `"dk"` is the same Driscoll-Kraay panel estimator FE uses, and it
requires `REOptions.dk_time`, the column that defines the time periods (the same option as
`FEOptions.dk_time`). The bandwidth actually used is reported as `REResults.dk_bandwidth_used`
(`None` unless `cov_type="dk"`). `REOptions.time` selects two-way random effects and is a different option from `dk_time`.

The small-sample corrections and the degrees of freedom of the t tests follow the same Stata/R
convention as FE (`fixest`'s `ssc()` defaults; `plm::vcovHC(type = "sss")` for `"cluster"`):
`"cluster"` scales by `G/(G-1) · (n-1)/(n-K)` and uses `G - 1` degrees of freedom, `"dk"` uses
`T/(T-1)` with `T` = number of time periods in place of `G` and `T - 1` degrees of freedom, and
the other types use `df_resid`. These differ from `linearmodels`, which does not apply the
`G/(G-1)` factor. The F statistic is the exception: it does not depend on `cov_type`, so `f_df_denom` is always `df_resid` even when `stat_df` is `G - 1` or `T - 1`. See
[Inference conventions](../guide/inference-conventions.md) for the full table.

## Variance components

`REOptions.variance_method` selects how the variance components `σ_ε²`, `σ_u²` (and `σ_λ²` for
two-way effects) are estimated, and therefore the quasi-demeaning weights and the GLS estimates:

- `"swamy_arora"` (default): the within and between regressions.
- `"wallace_hussain"`: the residuals of the pooled OLS of `y` on a constant and the regressors.
- `"amemiya"`: the residuals of the within estimator. A regressor without within variation (a
  time-invariant regressor; in a two-way model, one that varies in at most one dimension) raises
  `ValidationError`.

Wallace-Hussain and Amemiya follow R's `plm` (`random.method = "walhus"` and `"amemiya"`): closed
forms on a balanced panel (a one-way panel whose entities all have the same number of
observations; two-way panels are always balanced), and trace-term forms on an unbalanced one-way
panel. The two forms give different values on the same balanced panel, as in `plm`. If the
estimated `σ_ε²` is not positive, `fit()` raises `ComputationError` (`plm` sets it to 0 and
returns `NaN` weights). The method actually used is reported as `REResults.variance_method`. See
[Inference conventions](../guide/inference-conventions.md) for the differences from `plm` and
`linearmodels`.

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
- The comparison is always against fixed effects with the same structure as RE itself (one-way
  entity effects, or entity and time effects when `REOptions.time` is set). `REOptions.dk_time` is used only as the Driscoll-Kraay time periods (`cov_type="dk"`)
  and does not affect the test; specifying it with any other `cov_type` raises
  `ValidationError`.
- The Wald test uses the covariance matching `REOptions.cov_type` (classical, `hc1`–`hc3`,
  `cluster`, or Driscoll-Kraay `dk`). With the default `cov_type="cluster"` this is the
  cluster-robust Hausman test; use `cov_type="classical"` for the classical version, which
  assumes RE is fully efficient under the null. The auxiliary regression uses the same
  Stata/R-style small-sample corrections as RE's own standard errors.
- All three are `None` only when there are no slope coefficients to compare. If the auxiliary
  regression cannot be computed (rank-deficient design, or a robust covariance that is
  structurally singular such as `cov_type="cluster"` with no more clusters than the
  identified auxiliary slopes (`2k` at most), or `"dk"` with too few periods), `fit()` raises
  `ValidationError` / `ComputationError` instead, like a multicollinear design matrix. Use
  `cov_type="classical"` or more clusters in that case. See `REResults`'s docstring.
- Correlated random effects (Mundlak): a regressor may be included together with its entity
  mean (and, for two-way effects, its time mean). The between regressions and the Hausman
  auxiliary regression drop aliased columns by rank, like R's `lm` (the later column, relative
  tolerance `1e-7`), so the within-transformed copy of such a regressor is not tested and
  `hausman_df` counts only the identified copies; all three values are `None` when every
  time-varying regressor's mean is included. The final regression keeps every column, so a
  perfectly collinear design still raises `ComputationError`. `variance_method="amemiya"`
  rejects the time-invariant mean columns.

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
