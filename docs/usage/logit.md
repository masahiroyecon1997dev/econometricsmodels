# Logit: usage

**Logit**: **Usage** · [Options and results](../api/logit.md) · [Definitions](../methods/logit.md)

`Logit` estimates a binary logistic regression model by maximum likelihood. The dependent variable `y` must be coded 0/1.

```python
import polars as pl
from econometricsmodels import Logit

df = pl.DataFrame(
    {
        "y": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "group": [1, 1, 2, 2, 3, 3, 4, 4],
    }
)

result = Logit(df, y="y", x=["x1"]).fit()

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)  # {"const": ..., "x1": ...}
print(result.pseudo_r_squared)
```

## Standard errors and solvers

`LogitOptions` supports `cov_type` (`"classical"`, `"opg"`, `"hc0"`, `"hc1"`, or `"cluster"`) and `solver` (`"newton"`, `"bfgs"`, or `"lbfgs"`); see [Options and results](../api/logit.md) for the full list of options. The coefficients do not depend on `cov_type`. Tests and confidence intervals always use the normal distribution (`stat_dist` is `"normal"`).

| `cov_type` | Use it when |
|---|---|
| `"classical"` (default) | The model is correctly specified: the inverse of the observed information matrix |
| `"opg"` | As `"classical"`, from the outer product of the scores |
| `"hc0"`, `"hc1"` | You want standard errors that stay valid when the model is not exactly right (a sandwich estimator); `"hc1"` rescales `"hc0"` by `n / (n - k)` |
| `"cluster"` | Observations are correlated within groups; name the column in `cluster=` |

HC2, HC3 and HAC are not available for the maximum-likelihood models.

```python
from econometricsmodels import LogitOptions

robust = Logit(df, y="y", x=["x1"], options=LogitOptions(cov_type="hc0")).fit()
clustered = Logit(
    df, y="y", x=["x1"], options=LogitOptions(cov_type="cluster", cluster="group")
).fit()

print(robust.std_errors["x1"], clustered.std_errors["x1"])
print(result.converged, result.n_iter)  # convergence of the default solver
```

A fit that does not converge raises a `ComputationError`, and so does data that are perfectly separated by a regressor: no finite estimate exists then (see [Definitions: Logit](../methods/logit.md#separation)).

## Predicted values and classification table

`LogitResults.predict()` returns predicted probabilities (not a 0/1 class prediction, unlike `OLSResults.predict()`, which returns a point prediction of `y` — this is the standard statsmodels convention). With no arguments, it returns fitted probabilities for the training data; passing `new_data` returns out-of-sample predictions for new data instead (same `new_data` semantics as `OLSResults.predict()`). `pred_table()` returns a 2x2 classification (confusion) table for a given probability threshold (default 0.5), for the training data only.

```python
predicted = result.predict()
print(predicted)  # [0.11, 0.18, 0.29, ...] (a plain list of probabilities)

new_data = pl.DataFrame({"x1": [1.0, 2.0]})
predicted = result.predict(new_data=new_data)
print(predicted)  # [0.11, 0.18] (probabilities for the two new rows)

table = result.pred_table()
for row in table:
    print(row["actual"], row["predicted_0"], row["predicted_1"])
```

`LogitResults.augment()` takes the same `new_data` argument as `predict()`, but returns a polars DataFrame (the source data plus a new `"predicted_probability"` column) instead of a plain list, mirroring `OLSResults.augment()`.

```python
augmented = result.augment(new_data=new_data)
print(
    augmented
)  # original `new_data` columns, plus a "predicted_probability" column
```

## Marginal effects

`LogitResults.marginal_effects()` returns `dy/dx` for each explanatory variable (the constant term is excluded), with delta-method standard errors. Use `at` to choose the representative point: `"overall"` (default, average marginal effects), `"mean"`, or `"median"`.

```python
for row in result.marginal_effects():
    print(row["param_name"], row["effect"], row["std_error"], row["p_value"])

# Marginal effects evaluated at the mean of the explanatory variables
mean_effects = result.marginal_effects(at="mean")
```

## Probit

[Probit](probit.md) has the same interface with a normal link in place of the logistic one; everything on this page applies to it.

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name: the covariance of `x1` and `x2` is `result.cov_params["x1"]["x2"]`, and the standard errors are the square roots of its diagonal. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
