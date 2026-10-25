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
    }
)

result = Logit(df, y="y", x=["x1"]).fit()

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)  # {"const": ..., "x1": ...}
print(result.pseudo_r_squared)
```

`LogitOptions` supports `cov_type` (`"classical"`, `"opg"`, `"hc0"`, `"hc1"`, or `"cluster"`) and `solver` (`"newton"`, `"bfgs"`, or `"lbfgs"`); see [Options and results](../api/logit.md) for the full list of options.

## Predicted values and classification table

`LogitResults.predict()` returns predicted probabilities (not a 0/1 class prediction, unlike `OLSResults.predict()`, which returns a point prediction of `y` — this is the standard statsmodels convention). With no arguments, it returns fitted probabilities for the training data; passing `new_data` returns out-of-sample predictions for new data instead (same `new_data` semantics as `OLSResults.predict()`). `pred_table()` returns a 2x2 classification (confusion) table for a given probability threshold (default 0.5), for the training data only.

```python
predicted = result.predict()
print(predicted)  # [0.83, 0.12, ...] (a plain list of probabilities)

new_data = pl.DataFrame({"x1": [1.0, 2.0]})
predicted = result.predict(new_data)
print(predicted)  # [..., ...]

table = result.pred_table()
for row in table:
    print(row["actual"], row["predicted_0"], row["predicted_1"])
```

`LogitResults.augment()` takes the same `new_data` argument as `predict()`, but returns a polars DataFrame (the source data plus a new `"predicted_probability"` column) instead of a plain list, mirroring `OLSResults.augment()`.

```python
augmented = result.augment(new_data)
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
