# IV: options and results

**IV**: [Usage](../usage/iv.md) · **Options and results** · [Definitions](../methods/iv.md)

`IV` estimates a linear model with endogenous regressors by two-stage least squares (2SLS) or generalized method of moments (GMM), with the same constructor and `.fit()` as [OLS](ols.md). The independent variables are split into `x_exog` (exogenous) and `x_endog` (endogenous), plus `instruments` (the excluded instruments, at least one per endogenous variable). `IVOptions.estimator` selects the estimator, and a single `IV`/`IVResults` pair serves both. The estimators, covariance types and diagnostics are defined in [Definitions: IV](../methods/iv.md).

## Options at a glance

Pass an `IVOptions` as `options=`. Options of `None` take the effective default in parentheses where the option is used.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `estimator` | `"2sls"` or `"gmm"` | `"2sls"` | |
| `cov_type` | Standard error type: `"classical"`, `"hc0"` to `"hc3"`, `"hac"`, `"cluster"` | `"classical"` | For GMM it is independent of `gmm_weight_type` |
| `cluster` | Column that defines the clusters | `None` | With `cov_type="cluster"` or `gmm_weight_type="cluster"` |
| `hac_time` | Column that gives the time order | `None` | Required with `cov_type="hac"` or `gmm_weight_type="hac"` |
| `hac_lags` | Number of Newey-West lags | `None` (automatic) | With `"hac"` in either |
| `gmm_type` | `"one_step"`, `"two_step"` or `"iterated"` | `None` (`"two_step"`) | GMM only |
| `gmm_weight_type` | Weight matrix for the point estimate: `"classical"`, `"robust"`, `"cluster"`, `"hac"` | `None` (`"classical"`) | GMM only; not with `gmm_type="one_step"` |
| `gmm_max_iter` | Maximum number of estimations, 3 to 10,000 | `None` (100) | `gmm_type="iterated"` only |
| `gmm_tol` | Convergence tolerance | `None` (`1e-6`) | `gmm_type="iterated"` only |
| `raise_on_non_convergence` | Raise an error if iteration does not converge | `None` (`True`) | `gmm_type="iterated"` only |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `include_intercept` | Add a constant to `x_exog` | `True` | |

Which options go with which `cov_type` (an option set where it is not used raises a `ValidationError`):

| `cov_type` | Required | Optional |
|---|---|---|
| `"classical"`, `"hc0"` to `"hc3"` | | |
| `"hac"` | `hac_time` | `hac_lags` |
| `"cluster"` | `cluster` | |

For GMM, `gmm_weight_type` is chosen separately from `cov_type`: `"cluster"` needs `cluster`, and `"hac"` needs `hac_time` (and takes `hac_lags`). The GMM-only options (`gmm_type`, `gmm_weight_type`, `gmm_max_iter`, `gmm_tol`, `raise_on_non_convergence`) raise a `ValidationError` with `estimator="2sls"`. `gmm_max_iter`, `gmm_tol` and `raise_on_non_convergence` need `gmm_type="iterated"`, and `gmm_weight_type` is not allowed with `gmm_type="one_step"`. The estimator, GMM type and weight type actually used are echoed back on the results.

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.IV
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.IVOptions

::: econometricsmodels.IVResults
