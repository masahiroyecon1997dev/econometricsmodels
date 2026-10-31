# Logit: options and results

**Logit**: [Usage](../usage/logit.md) · **Options and results** · [Definitions](../methods/logit.md)

`Logit` estimates a binary logistic regression model by maximum likelihood. It shares the general API shape with [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), and its results add `marginal_effects()`, `predict()` and `pred_table()`.

The estimator, the `cov_type` options (`classical`, `opg`, `hc0`, `hc1`, `cluster`), the solvers and convergence rules, separation, the z tests, the likelihood-ratio test and the marginal effects are defined in [Definitions: Logit](../methods/logit.md).

## Options at a glance

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"opg"`, `"hc0"`, `"hc1"`, `"cluster"` | `"classical"` | `"cluster"` needs `cluster` |
| `cluster` | Column that defines the clusters | `None` | Only with `cov_type="cluster"` |
| `solver` | `"newton"`, `"bfgs"` or `"lbfgs"` | `"newton"` | All converge to the same estimate |
| `max_iter` | Iteration limit, 1 to 10,000 | `35` | |
| `tol` | Convergence tolerance on the gradient | `1e-6` for `"newton"`, `1e-8` for the others | The meaning depends on the solver; see the docstring |
| `raise_on_non_convergence` | Raise an error if the solver does not converge | `True` | With `False` the last iterate is returned with `converged=False` |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `include_intercept` | Add a constant column (`const`) | `True` | |

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.Logit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.LogitOptions

::: econometricsmodels.LogitResults
