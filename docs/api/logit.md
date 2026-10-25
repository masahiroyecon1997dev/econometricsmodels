# Logit

`Logit` estimates a binary logistic regression model by maximum likelihood. It shares the general API shape with [OLS](ols.md) (`data`/`y`/`x`/`options` constructor, `.fit()`), and its results add `marginal_effects()`, `predict()` and `pred_table()`.

The estimator, the `cov_type` options (`classical`, `opg`, `hc0`, `hc1`, `cluster`), the solvers and convergence rules, separation, the z tests, the likelihood-ratio test and the marginal effects are defined in [Methods: Logit](../methods/logit.md). Getting Started has [examples](../getting-started.md#logit-binary-logistic-regression).

::: econometricsmodels.Logit
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.LogitOptions

::: econometricsmodels.LogitResults
