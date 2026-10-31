# Probit: usage

**Probit**: **Usage** · [Options and results](../api/probit.md) · [Definitions](../methods/probit.md)

`Probit` estimates a binary probit regression model by maximum likelihood. The dependent variable `y` must be coded 0/1. Its API is identical to [Logit](logit.md) — the only difference is the link function (the standard normal CDF `Φ` in place of the logistic CDF `Λ`).

```python
import polars as pl
from econometricsmodels import Probit

df = pl.DataFrame(
    {
        "y": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "group": [1, 1, 2, 2, 3, 3, 4, 4],
    }
)

result = Probit(df, y="y", x=["x1"]).fit()

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)  # {"const": ..., "x1": ...}
print(result.pseudo_r_squared)
```

## Standard errors and solvers

`ProbitOptions` supports the same `cov_type` (`"classical"`, `"opg"`, `"hc0"`, `"hc1"`, `"cluster"`) and `solver` choices as `LogitOptions`, and the guidance on [choosing between them](logit.md#standard-errors-and-solvers) applies unchanged. See [Options and results](../api/probit.md) for the full list of options.

```python
from econometricsmodels import ProbitOptions

robust = Probit(df, y="y", x=["x1"], options=ProbitOptions(cov_type="hc0")).fit()
clustered = Probit(
    df, y="y", x=["x1"], options=ProbitOptions(cov_type="cluster", cluster="group")
).fit()

print(robust.std_errors["x1"], clustered.std_errors["x1"])
```

## Predictions and marginal effects

`ProbitResults.predict()`, `augment()`, `pred_table()`, and `marginal_effects()` work exactly like their [Logit](logit.md#predicted-values-and-classification-table) counterparts (substitute `Probit`/`ProbitOptions` for `Logit`/`LogitOptions` in the examples above).

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name: the covariance of `x1` and `x2` is `result.cov_params["x1"]["x2"]`, and the standard errors are the square roots of its diagonal. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[Logit](logit.md), [Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
