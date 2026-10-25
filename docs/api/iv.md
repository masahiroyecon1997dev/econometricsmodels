# IV

`IV` estimates a linear model with endogenous regressors by two-stage least squares (2SLS) or generalized method of moments (GMM). Unlike [OLS](ols.md), the independent variables are split into `x_exog` (exogenous) and `x_endog` (endogenous), plus `instruments` (the excluded instruments, at least one per endogenous variable). `IVOptions.estimator` (`"2sls"`, the default, or `"gmm"`) selects the estimator; a single `IV`/`IVResults` pair serves both.

Several `IVOptions` fields apply to GMM only (`gmm_type`, `gmm_weight_type`, `gmm_max_iter`, `gmm_tol`, `raise_on_non_convergence`). Setting any of them with `estimator="2sls"` raises a `ValidationError`, as do `gmm_max_iter` and `gmm_tol` unless `gmm_type="iterated"`, and `gmm_weight_type` with `gmm_type="one_step"`. Their defaults are `None`, which means the effective defaults described in the docstring. The estimator, GMM type and weight type actually used are echoed back on the results.

The estimators, `cov_type` for both of them, the test distributions, and the weak-instrument, overidentification and endogeneity diagnostics are defined in [Methods: IV](../methods/iv.md). Getting Started has [examples](../getting-started.md#iv-instrumental-variables-2slsgmm).

::: econometricsmodels.IV
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.IVOptions

::: econometricsmodels.IVResults
