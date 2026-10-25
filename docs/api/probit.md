# Probit: options and results

**Probit**: [Usage](../usage/probit.md) · **Options and results** · [Definitions](../methods/probit.md)

`Probit` estimates a binary probit regression model by maximum likelihood. It shares the API of [Logit](logit.md) (`data`/`y`/`x`/`options` constructor, `.fit()`, `predict()`, `pred_table()`, `marginal_effects()`) and differs only in the link function: the standard normal CDF in place of the logistic CDF.

The definitions are in [Methods: Probit](../methods/probit.md).

## Options at a glance

`ProbitOptions` has the same fields as `LogitOptions`.

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

::: econometricsmodels.Probit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.ProbitOptions

::: econometricsmodels.ProbitResults
