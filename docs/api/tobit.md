# Tobit: options and results

**Tobit**: [Usage](../usage/tobit.md) · **Options and results** · [Definitions](../methods/tobit.md)

`Tobit` estimates a censored normal regression model by maximum likelihood. It shares the maximum-likelihood API shape of [Logit](logit.md) (`data`/`y`/`x`/`options` constructor, `.fit()`, z tests, `cov_type`, `marginal_effects()`, `predict()`), but `y` is continuous rather than 0/1, `sigma` is reported as an estimated parameter (`param_names` is `["const", <x…>, "sigma"]`), and the results add `censoring_fit_check()` in place of `pred_table()`.

`TobitOptions.lower` (default `0.0`) and `upper` (default `None`) set the censoring bounds; `None` means that side is not censored. For a right-censored-only model pass `lower=None` explicitly. The likelihood, the bounds the data must satisfy, the Wald overall test, the four prediction `target` values and the censoring fit check are defined in [Methods: Tobit](../methods/tobit.md).

## Options at a glance

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `lower` | Lower censoring bound | `0.0` | `None` means not censored from below |
| `upper` | Upper censoring bound | `None` | `None` means not censored from above; at least one bound must be set |
| `cov_type` | Standard error type: `"classical"`, `"opg"`, `"hc0"`, `"hc1"`, `"cluster"` | `"classical"` | `"cluster"` needs `cluster` |
| `cluster` | Column that defines the clusters | `None` | Only with `cov_type="cluster"` |
| `solver` | `"newton"`, `"bfgs"` or `"lbfgs"` | `"newton"` | All converge to the same estimate |
| `max_iter` | Iteration limit, 1 to 10,000 | `35` | |
| `tol` | Convergence tolerance on the gradient | `1e-6` for `"newton"`, `1e-8` for the others | The meaning depends on the solver; see the docstring |
| `raise_on_non_convergence` | Raise an error if the solver does not converge | `True` | With `False` the last iterate is returned with `converged=False` |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `include_intercept` | Add a constant column (`const`) | `True` | |

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.Tobit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.TobitOptions

::: econometricsmodels.TobitResults
