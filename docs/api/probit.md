# Probit

`Probit` estimates a binary probit regression model by maximum likelihood. It shares the general API shape with [Logit](logit.md) (`data`/`y`/`x`/`options` constructor, `.fit()`, `predict()`/`pred_table()`/`marginal_effects()`) and differs only in the link function: Probit uses the standard normal CDF (`Φ`) in place of the logistic CDF (`Λ`).

## Standard error types

`ProbitOptions.cov_type` supports `"classical"` (observed-information / Hessian-based), `"opg"` (outer product of gradients), `"hc0"`, `"hc1"`, and `"cluster"` (requires `cluster_col`). As with [Logit](logit.md#standard-error-types), HC2/HC3 and HAC are not available.

## Solver options

`ProbitOptions.method` selects the optimization algorithm used to maximize the log-likelihood: `"newton"` (default, Newton-Raphson), `"bfgs"`, or `"lbfgs"`. All three converge to the same maximum-likelihood estimate; they differ in iteration cost and behavior on ill-conditioned problems. `max_iter` and `tol` control the iteration limit and the gradient-norm convergence threshold. When the solver does not converge within `max_iter` iterations, a `ComputationError` is raised unless `raise_on_non_convergence=False`, in which case `ProbitResults.converged` is `False` instead.

## Marginal effects

`ProbitResults.marginal_effects()` computes `dy/dx` with delta-method standard errors, evaluated at `"overall"` (average marginal effects, the default), `"mean"`, or `"median"`. The constant term is excluded from the output. The API is identical to [Logit](logit.md#marginal-effects); see the [Getting Started](../getting-started.md#marginal-effects) example there (substitute `Probit`/`ProbitOptions`).

::: econometricsmodels.Probit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.ProbitOptions

::: econometricsmodels.ProbitResults
