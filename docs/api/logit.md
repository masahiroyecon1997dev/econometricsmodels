# Logit

`Logit` estimates a binary logistic regression model by maximum likelihood. It shares the general API shape with [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), but differs in several ways specific to a maximum-likelihood, discrete-choice model: standard errors use a z-test rather than a t-test, `cov_type` supports `opg` (outer product of gradients) in place of HC2/HC3/HAC, and `LogitResults` adds `marginal_effects()`, `predict()`, and `pred_table()`.

## Standard error types

`LogitOptions.cov_type` supports `"classical"` (observed-information / Hessian-based), `"opg"` (outer product of gradients), `"hc0"`, `"hc1"`, and `"cluster"` (requires `cluster_col`). Unlike [OLS](../getting-started.md#switching-the-type-of-standard-error), HC2/HC3 and HAC are not available for Logit.

## Solver options

`LogitOptions.method` selects the optimization algorithm used to maximize the log-likelihood: `"newton"` (default, Newton-Raphson), `"bfgs"`, or `"lbfgs"`. All three converge to the same maximum-likelihood estimate; they differ in iteration cost and behavior on ill-conditioned problems. `max_iter` and `tol` control the iteration limit and the gradient-norm convergence threshold. When the solver does not converge within `max_iter` iterations, a `ComputationError` is raised unless `raise_on_non_convergence=False`, in which case `LogitResults.converged` is `False` instead.

## Marginal effects

`LogitResults.marginal_effects()` computes `dy/dx` with delta-method standard errors, evaluated at `"overall"` (average marginal effects, the default), `"mean"`, or `"median"`. The constant term is excluded from the output. See [Getting Started](../getting-started.md#marginal-effects) for an example.

::: econometricsmodels.Logit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.LogitOptions

::: econometricsmodels.LogitResults
