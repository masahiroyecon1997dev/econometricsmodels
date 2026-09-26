# IV

`IV` estimates a linear model with endogenous regressors by two-stage least squares (2SLS) or generalized method of moments (GMM). Unlike [OLS](ols.md), independent variables are split into two lists — `x_exog` (exogenous) and `x_endog` (endogenous) — plus `instruments` (the excluded instruments, one per endogenous variable at minimum for identification). `IVOptions.estimator` (`"2sls"`, the default, or `"gmm"`) selects the estimator; a single `IV`/`IVResults` pair serves both.

## Standard error types

`IVOptions.cov_type` supports `"classical"`, `"hc0"` through `"hc3"`, `"hac"` (with `hac_lag`/`hac_time`), and `"cluster"` (with `cluster`) — the same range as [OLS](../getting-started.md#switching-the-type-of-standard-error). `stats` (test statistics) and `p_values` use a t-test for `estimator="2sls"` and a z-test for `estimator="gmm"`.

## GMM weight type and iteration

For `estimator="gmm"`, `IVOptions.gmm_weight_type` selects the weight matrix used for point estimation — `"classical"`, `"robust"`, `"cluster"`, or `"hac"` (the same vocabulary as `cov_type`) — independently of `cov_type` (which only affects the final reported standard errors). `gmm_weight_type="cluster"`/`"hac"` read `cluster`/`hac_lag`/`hac_time` from the same fields `cov_type` uses. `IVOptions.gmm_type` selects the GMM estimation type: `"two_step"` (default, efficient two-step GMM), `"one_step"` (weight matrix `(Z'Z)⁻¹` only; `gmm_weight_type` is not used), or `"iterated"` (repeat until convergence). `gmm_max_iter` (maximum number of estimations counting the initial one, at least `3`; effective default `100`) and `gmm_tol` (convergence tolerance; effective default `1e-6`) apply only to `"iterated"` and raise `ValidationError` when given with the other types; `IVResults.converged`/`n_iter` report the outcome (`converged` is always `True` for `"one_step"`/`"two_step"`). `estimator="2sls"` ignores all of these fields. The estimator, GMM type and weight matrix actually used are echoed back on the result as `IVResults.estimator`, `IVResults.gmm_type` and `IVResults.gmm_weight_type` (normalized to lowercase); `gmm_type` is `None` for `estimator="2sls"`, and `gmm_weight_type` is `None` for `estimator="2sls"` and for `gmm_type="one_step"`, which have no such concept.

## Diagnostics

`IVResults` exposes three diagnostics in addition to the coefficient table:

- `weak_instrument_f_statistics`: partial F-statistic per endogenous variable, testing the excluded instruments' joint significance after partialling out `x_exog` (always under the classical formula, regardless of `cov_type`).
- `overid_statistic`/`overid_p_value`: the overidentification test — Sargan (`estimator="2sls"`) or Hansen J (`estimator="gmm"`). `None` when just-identified (`len(instruments) == len(x_endog)`).
- `wu_hausman_statistic`/`wu_hausman_p_value`: regression-based endogeneity test (adds first-stage residuals to the structural equation). Only available for `estimator="2sls"`; always `None` for `estimator="gmm"`.

## First-stage results

`IVResults.first_stage()` returns a `dict[str, OLSResults]` keyed by endogenous variable name, one plain-OLS regression of `x_endog[i]` on `x_exog + instruments` per endogenous variable. Its `f_statistic` includes `x_exog`'s contribution and is not the same as `weak_instrument_f_statistics`.

::: econometricsmodels.IV
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.IVOptions

::: econometricsmodels.IVResults
