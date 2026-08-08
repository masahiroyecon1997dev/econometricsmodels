# Getting Started

## OLS (Ordinary Least Squares)

Pass the column names of the dependent variable (`y`) and independent variables (`x`), along with the polars DataFrame to estimate on, to `OLS`, then call `.fit()`.

```python
import polars as pl
from econometricsmodels import OLS

df = pl.DataFrame(
    {
        "y": [2.1, 3.9, 6.2, 8.1, 9.8],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)

result = OLS(df, y="y", x=["x1"]).fit()

print(result.params)       # {"const": ..., "x1": ...}
print(result.std_errors)   # {"const": ..., "x1": ...}
print(result.r_squared)
```

With `include_intercept` (default `True`), a constant term (`const`) is automatically added to the design matrix, separate from the columns specified in `x`.

## Switching the type of standard error

Setting `cov_type` on `OLSOptions` lets you switch to heteroskedasticity-robust standard errors (HC0-HC3), cluster-robust standard errors, or HAC (Newey-West) standard errors.

```python
from econometricsmodels import OLS, OLSOptions

# Heteroskedasticity-robust standard errors (HC1)
options = OLSOptions(cov_type="hc1")
result = OLS(df, y="y", x=["x1"], options=options).fit()

# Cluster-robust standard errors (specify a column name from data)
options = OLSOptions(cov_type="cluster", cluster_col="group_id")
result = OLS(df, y="y", x=["x1"], options=options).fit()
```

See the [API Reference](api/ols.md) for the full list of available options.

## Retrieving results

`OlsResults` exposes coefficients, standard errors, etc. as dictionaries keyed by coefficient name (`str`). If you need a row-oriented listing — e.g. for a REST API response — use `coef_table()`.

```python
for row in result.coef_table():
    print(row["param"], row["coef"], row["std_err"], row["p_value"])
```

## Predicted values

`OlsResults.predict()` returns predicted values. With no arguments, it returns the fitted values for the training data used in `fit()`; passing `new_data` returns out-of-sample predictions for new data instead.

```python
# Fitted values for the training data
fitted = result.predict()

# Predictions for new data (columns must match the `x` columns used at fit
# time by name; column order does not matter, and the constant column must
# not be included)
new_data = pl.DataFrame({"x1": [6.0, 7.0]})
predicted = result.predict(new_data)

print(predicted)  # [{"fitted": ...}, {"fitted": ...}]
```

## WLS (Weighted Least Squares)

`WLS` is `OLS` with an added `weight` argument (the column name of the weight column). Weights are treated as analytic weights proportional to the inverse of the variance, and do not need to be normalized. Values less than or equal to 0 raise a `ValidationError`.

```python
from econometricsmodels import WLS

df = df.with_columns(pl.Series("w", [1.0, 1.0, 1.0, 1.0, 1.0]))

result = WLS(df, y="y", x=["x1"], weight="w").fit()

print(result.params)
print(result.std_errors)
```

Estimation options (`cov_type`, etc.) use the same `OLSOptions` as OLS. See "Switching the type of standard error" above for how to switch standard error types, and the [API Reference](api/wls.md) for details on the `weight` argument.

## Logit (binary logistic regression)

`Logit` estimates a binary logistic regression model by maximum likelihood. The dependent variable `y` must be coded 0/1.

```python
import polars as pl
from econometricsmodels import Logit

df = pl.DataFrame(
    {
        "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)

result = Logit(df, y="y", x=["x1"]).fit()

print(result.params)         # {"const": ..., "x1": ...}
print(result.std_errors)     # {"const": ..., "x1": ...}
print(result.pseudo_r_squared)
```

`LogitOptions` supports `cov_type` (`"classical"`, `"opg"`, `"hc0"`, `"hc1"`, or `"cluster"`) and `method` (`"newton"`, `"bfgs"`, or `"lbfgs"`); see the [API Reference](api/logit.md) for the full list of options.

### Predicted values and classification table

`LogitResults.predict()` returns fitted probabilities for the training data. `pred_table()` returns a 2x2 classification (confusion) table for a given probability threshold (default 0.5).

```python
predicted = result.predict()
print(predicted)  # [{"probability": ...}, ...]

table = result.pred_table()
for row in table:
    print(row["actual"], row["predicted_0"], row["predicted_1"])
```

### Marginal effects

`LogitResults.marginal_effects()` returns `dy/dx` for each explanatory variable (the constant term is excluded), with delta-method standard errors. Use `at` to choose the representative point: `"overall"` (default, average marginal effects), `"mean"`, or `"median"`.

```python
for row in result.marginal_effects():
    print(row["param"], row["dydx"], row["std_err"], row["p_value"])

# Marginal effects evaluated at the mean of the explanatory variables
mean_effects = result.marginal_effects(at="mean")
```

## Probit (probit regression)

`Probit` estimates a binary probit regression model by maximum likelihood. The dependent variable `y` must be coded 0/1. Its API is identical to [Logit](#logit-binary-logistic-regression) — the only difference is the link function (the standard normal CDF `Φ` in place of the logistic CDF `Λ`).

```python
import polars as pl
from econometricsmodels import Probit

df = pl.DataFrame(
    {
        "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)

result = Probit(df, y="y", x=["x1"]).fit()

print(result.params)         # {"const": ..., "x1": ...}
print(result.std_errors)     # {"const": ..., "x1": ...}
print(result.pseudo_r_squared)
```

`ProbitOptions` supports the same `cov_type` and `method` choices as `LogitOptions`; see the [API Reference](api/probit.md) for the full list of options. `ProbitResults.predict()`, `pred_table()`, and `marginal_effects()` work exactly like their [Logit](#predicted-values-and-classification-table) counterparts (substitute `Probit`/`ProbitOptions` for `Logit`/`LogitOptions` in the examples above).

## Error handling

Invalid input or options (a missing column, missing values, etc.) raise `ValidationError` (a subclass of `ValueError`). Problems detected during computation (e.g. a singular design matrix) raise `ComputationError` (a subclass of `RuntimeError`).

```python
from econometricsmodels import ComputationError, ValidationError

try:
    result = OLS(df, y="y", x=["x1", "x2"]).fit()
except ValidationError as e:
    print("Input error:", e)
except ComputationError as e:
    print("Computation error:", e)
```
