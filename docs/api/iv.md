# IV

`IV` estimates a linear model with endogenous regressors by two-stage least squares (2SLS) or generalized method of moments (GMM). Unlike [OLS](ols.md), independent variables are split into two lists — `x_exog` (exogenous) and `x_endog` (endogenous) — plus `instruments` (the excluded instruments, one per endogenous variable at minimum for identification). `IvOptions.method` (`"2sls"`, the default, or `"gmm"`) selects the estimator; a single `IV`/`IvResults` pair serves both.

## Standard error types

`IvOptions.cov_type` supports `"classical"`, `"hc0"` through `"hc3"`, `"hac"` (with `hac_lag`/`time_col`), and `"cluster"` (with `cluster_col`) — the same range as [OLS](../getting-started.md#switching-the-type-of-standard-error). `stats` (test statistics) and `p_values` use a t-test for `method="2sls"` and a z-test for `method="gmm"`.

## GMM weight type and iteration

For `method="gmm"`, `IvOptions.weight_type` selects the weight matrix used for point estimation — `"unadjusted"` (alias `"homoskedastic"`), `"robust"` (alias `"heteroskedastic"`), `"cluster"`, or `"kernel"` — independently of `cov_type` (which only affects the final reported standard errors). `weight_type="cluster"`/`"kernel"` read `cluster_col`/`hac_lag`/`time_col` from the same fields `cov_type` uses. `gmm_iterations` (default `2`, efficient two-step GMM) sets a fixed iteration count; `1` gives one-step GMM, `3+` gives iterated GMM. Set `gmm_convergence` to switch to convergence-based stopping instead (iterate until coefficients stabilize within the given tolerance, up to `gmm_iterations` as a safety cap); `IvResults.converged`/`n_iterations` report the outcome. `method="2sls"` ignores all of these fields.

## Diagnostics

`IvResults` exposes three diagnostics in addition to the coefficient table:

- `weak_instrument_f_statistics`: partial F-statistic per endogenous variable, testing the excluded instruments' joint significance after partialling out `x_exog` (always under the classical formula, regardless of `cov_type`).
- `overid_statistic`/`overid_p_value`: the overidentification test — Sargan (`method="2sls"`) or Hansen J (`method="gmm"`). `None` when just-identified (`len(instruments) == len(x_endog)`).
- `wu_hausman_statistic`/`wu_hausman_p_value`: regression-based endogeneity test (adds first-stage residuals to the structural equation). Only available for `method="2sls"`; always `None` for `method="gmm"`.

## First-stage results

`IvResults.first_stage()` returns a `dict[str, OlsResults]` keyed by endogenous variable name, one plain-OLS regression of `x_endog[i]` on `x_exog + instruments` per endogenous variable. Its `f_statistic` includes `x_exog`'s contribution and is not the same as `weak_instrument_f_statistics`.

::: econometricsmodels.IV
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.IvOptions

::: econometricsmodels.IvResults
