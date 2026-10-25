# Probit

`Probit` estimates a binary probit regression model by maximum likelihood. It shares the API of [Logit](logit.md) (`data`/`y`/`x`/`options` constructor, `.fit()`, `predict()`, `pred_table()`, `marginal_effects()`) and differs only in the link function: the standard normal CDF in place of the logistic CDF.

The definitions are in [Methods: Probit](../methods/probit.md). Getting Started has [examples](../getting-started.md#probit-probit-regression).

::: econometricsmodels.Probit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.ProbitOptions

::: econometricsmodels.ProbitResults
