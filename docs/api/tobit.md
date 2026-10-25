# Tobit

`Tobit` estimates a censored normal regression model by maximum likelihood. It shares the maximum-likelihood API shape of [Logit](logit.md) (`data`/`y`/`x`/`options` constructor, `.fit()`, z tests, `cov_type`, `marginal_effects()`, `predict()`), but `y` is continuous rather than 0/1, `sigma` is reported as an estimated parameter (`param_names` is `["const", <x…>, "sigma"]`), and the results add `censoring_fit_check()` in place of `pred_table()`.

`TobitOptions.lower` (default `0.0`) and `upper` (default `None`) set the censoring bounds; `None` means that side is not censored. For a right-censored-only model pass `lower=None` explicitly. The likelihood, the bounds the data must satisfy, the Wald overall test, the four prediction `target` values and the censoring fit check are defined in [Methods: Tobit](../methods/tobit.md). Getting Started has [examples](../getting-started.md#tobit-censored-regression).

::: econometricsmodels.Tobit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.TobitOptions

::: econometricsmodels.TobitResults
