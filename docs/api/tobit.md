# Tobit

`Tobit` estimates a censored normal (Tobit) regression model by maximum likelihood. The observed dependent variable `y` is a continuous latent regression `y* = x'β + ε` (`ε ~ N(0, σ²)`) censored at a lower bound, an upper bound, or both. It shares the general maximum-likelihood API shape with [Logit](logit.md) (`data`/`y`/`x`/`options` constructor, `.fit()`, z-tests, `cov_type`, `marginal_effects()`, `predict()`), but `y` is continuous rather than 0/1, `σ` is reported as an estimated parameter, and the results add `censoring_fit_check()` in place of `pred_table()`.

## Censoring bounds

`TobitOptions.lower` (default `0.0`) and `TobitOptions.upper` (default `None`) set the censoring bounds; `None` means that side is not censored. The default is the standard left-censored-at-zero Tobit. For a right-censored-only model, pass `lower=None` explicitly. `fit()` raises `ValidationError` if both bounds are `None`, if `lower >= upper`, or if any `y` value falls outside the given bounds, and also if no observation lies strictly inside the bounds (the model is then unidentified).

`TobitResults.lower` / `.upper` echo back the bounds actually used.

## Standard error types

`TobitOptions.cov_type` supports `"classical"` (observed-information / Hessian-based), `"opg"` (outer product of gradients), `"hc0"`, `"hc1"`, and `"cluster"` (requires `cluster_col`). As with [Logit](logit.md#standard-error-types), HC2/HC3 and HAC are not available. Internally the optimizer works in `(β, log σ)` space; the reported covariance is transformed to `(β, σ)` space so that `TobitResults.std_errors` includes a standard error for `sigma` alongside the coefficients. `param_names` is `["const", <x…>, "sigma"]`.

## Overall significance test

`TobitResults.wald_statistic` / `wald_p_value` report a Wald test of the joint hypothesis that all non-intercept coefficients are zero (chi-square distributed, matching `AER::tobit`). Tobit does not provide a likelihood-ratio statistic or a pseudo-R², because the intercept-only Tobit model has no closed-form log-likelihood.

## Solver options

`TobitOptions.method` selects the optimization algorithm: `"newton"` (default, Newton-Raphson with a Levenberg-Marquardt damped step for the indefinite-Hessian regions specific to the `(β, log σ)` likelihood), `"bfgs"`, or `"lbfgs"`. All three converge to the same maximum-likelihood estimate. `max_iter` and `tol` control the iteration limit and the gradient-norm convergence threshold. When the solver does not converge within `max_iter` iterations, a `ComputationError` is raised unless `raise_on_non_convergence=False`, in which case `TobitResults.converged` is `False` instead.

## Marginal effects and predictions

`TobitResults.marginal_effects()` and `predict()` both take a `target`:

| `target` | Quantity |
|---|---|
| `"expected_latent"` | `E[y*\|x] = x'β` (the linear predictor) |
| `"expected_observed"` (default) | `E[y\|x]`, the censoring-adjusted conditional mean (McDonald–Moffitt) |
| `"prob_uncensored"` | `P(uncensored\|x)` |

`marginal_effects()` computes `dy/dx` with delta-method standard errors, evaluated at `"overall"` (average marginal effects, the default), `"mean"`, or `"median"`; the constant term is excluded. `predict()` returns row-oriented fitted values for the training data (in-sample only). See [Getting Started](../getting-started.md#tobit-censored-regression) for an example.

## Censoring fit check

`TobitResults.censoring_fit_check()` replaces `pred_table()`. For each censored direction it compares the observed censoring rate (fraction of `y` exactly at the boundary) against the model-implied rate (mean fitted boundary probability), returned as a per-direction breakdown (`"lower"` / `"uncensored"` / `"upper"`).

::: econometricsmodels.Tobit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.TobitOptions

::: econometricsmodels.TobitResults
