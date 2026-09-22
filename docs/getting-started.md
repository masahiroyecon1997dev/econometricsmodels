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

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)  # {"const": ..., "x1": ...}
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

`OLSResults` exposes coefficients, standard errors, etc. as dictionaries keyed by coefficient name (`str`). If you need a row-oriented listing — e.g. for a REST API response — use `coef_table()`.

```python
for row in result.coef_table():
    print(row["param"], row["coef"], row["std_err"], row["p_value"])
```

## Predicted values

`OLSResults.predict()` returns predicted values. With no arguments, it returns the fitted values for the training data used in `fit()`; passing `new_data` returns out-of-sample predictions for new data instead.

```python
# Fitted values for the training data
fitted = result.predict()

# Predictions for new data (columns must match the `x` columns used at fit
# time by name; column order does not matter, and the constant column must
# not be included)
new_data = pl.DataFrame({"x1": [6.0, 7.0]})
predicted = result.predict(new_data)

print(predicted)  # [{"predicted": ...}, {"predicted": ...}]
```

`OLSResults.augment()` takes the same `new_data` argument, but instead returns the source data (the training data, or `new_data` when given) with the predicted values appended as a `"predicted"` column — a polars DataFrame rather than a row-oriented list. This is the one exception to the library's general policy of not returning DataFrames.

```python
augmented = result.augment(new_data)
print(augmented)  # original `new_data` columns, plus a "predicted" column
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

Estimation options are configured via `WLSOptions`, which has the same fields as `OLSOptions` (`cov_type`, etc.). See "Switching the type of standard error" above for how to switch standard error types, and the [API Reference](api/wls.md) for details on the `weight` argument.

`WLSResults.predict()` and `WLSResults.augment()` work exactly like their `OLSResults` counterparts (see "Predicted values" above); weights play no role in either the training-data or out-of-sample case.

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

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)  # {"const": ..., "x1": ...}
print(result.pseudo_r_squared)
```

`LogitOptions` supports `cov_type` (`"classical"`, `"opg"`, `"hc0"`, `"hc1"`, or `"cluster"`) and `method` (`"newton"`, `"bfgs"`, or `"lbfgs"`); see the [API Reference](api/logit.md) for the full list of options.

### Predicted values and classification table

`LogitResults.predict()` returns predicted probabilities (not a 0/1 class prediction, unlike `OLSResults.predict()`'s `"predicted"` — this is the standard statsmodels convention). With no arguments, it returns fitted probabilities for the training data; passing `new_data` returns out-of-sample predictions for new data instead (same `new_data` semantics as `OLSResults.predict()`). `pred_table()` returns a 2x2 classification (confusion) table for a given probability threshold (default 0.5), for the training data only.

```python
predicted = result.predict()
print(predicted)  # [{"probability": ...}, ...]

new_data = pl.DataFrame({"x1": [1.0, 2.0]})
predicted = result.predict(new_data)
print(predicted)  # [{"probability": ...}, {"probability": ...}]

table = result.pred_table()
for row in table:
    print(row["actual"], row["predicted_0"], row["predicted_1"])
```

`LogitResults.augment()` takes the same `new_data` argument as `predict()`, but returns a polars DataFrame (the source data plus a new `"probability"` column) instead of a row-oriented list, mirroring `OLSResults.augment()`.

```python
augmented = result.augment(new_data)
print(augmented)  # original `new_data` columns, plus a "probability" column
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

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)  # {"const": ..., "x1": ...}
print(result.pseudo_r_squared)
```

`ProbitOptions` supports the same `cov_type` and `method` choices as `LogitOptions`; see the [API Reference](api/probit.md) for the full list of options. `ProbitResults.predict()`, `augment()`, `pred_table()`, and `marginal_effects()` work exactly like their [Logit](#predicted-values-and-classification-table) counterparts (substitute `Probit`/`ProbitOptions` for `Logit`/`LogitOptions` in the examples above).

## Tobit (censored regression)

`Tobit` estimates a censored normal regression model by maximum likelihood. The observed `y` is a continuous latent regression censored at a lower bound, an upper bound, or both. `TobitOptions.lower` (default `0.0`) and `upper` (default `None`) set the bounds; `None` means that side is not censored.

```python
import polars as pl
from econometricsmodels import Tobit, TobitOptions

df = pl.DataFrame(
    {
        "y": [0.0, 0.0, 1.2, 2.5, 3.1, 4.8, 6.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)

result = Tobit(df, y="y", x=["x1"]).fit()  # left-censored at 0 by default

print(result.params)  # {"const": ..., "x1": ..., "sigma": ...}
print(result.sigma)
print(result.wald_statistic, result.wald_p_value)
```

`param_names` ends with `"sigma"` — the error standard deviation is reported as an estimated parameter with its own standard error. Instead of a likelihood-ratio test, `TobitResults` exposes `wald_statistic` / `wald_p_value` for the joint hypothesis that all slopes are zero.

### Predictions and marginal effects

`predict()` and `marginal_effects()` take a `target`: `"expected_latent"` (`E[y*|x] = x'β`), `"expected_observed"` (the default, the censoring-adjusted mean `E[y|x]`), or `"prob_uncensored"`. `marginal_effects()` also takes `at` (`"overall"`, `"mean"`, `"median"`) and excludes the constant term.

```python
for row in result.marginal_effects(target="expected_observed"):
    print(row["param"], row["dydx"], row["std_err"])

# predict() returns a list of {"predicted": ...} dicts
fitted = result.predict(target="expected_observed")

# new_data (out-of-sample) works the same way as OLS/Logit/Probit
new_data = pl.DataFrame({"x1": [1.0, 2.0]})
predicted = result.predict(target="expected_observed", new_data=new_data)
```

`augment()` takes the same `target`/`new_data` arguments as `predict()`, but returns a polars DataFrame instead of a row-oriented list. Unlike Logit/Probit's fixed `"probability"` column, the appended column is named `"predicted_{target}"` (e.g. `"predicted_expected_observed"`), since `predict()`'s meaning depends on `target` — this also lets you call `augment()` once per `target` on the same DataFrame without a column name collision.

```python
augmented = result.augment(target="expected_observed")
print(augmented)  # original columns, plus "predicted_expected_observed"

# Stack a second target onto the same DataFrame without a name collision
augmented = result.augment(target="prob_uncensored", new_data=augmented)
```

### Censoring fit check

`censoring_fit_check()` compares the observed censoring rate against the model-implied rate for each censored direction (`"lower"` / `"uncensored"` / `"upper"`):

```python
for row in result.censoring_fit_check():
    print(row["category"], row["observed_rate"], row["model_implied_rate"])
```

See the [API Reference](api/tobit.md) for the full list of options.

## IV (instrumental variables: 2SLS/GMM)

`IV` estimates a linear model with endogenous regressors. Independent variables are split into `x_exog` (exogenous) and `x_endog` (endogenous), plus `instruments` (excluded instruments — at least one per endogenous variable for identification).

```python
import polars as pl
from econometricsmodels import IV

df = pl.DataFrame(
    {
        "y": [1.0, 2.4, 2.9, 4.3, 5.1, 5.8, 7.2, 7.9],
        "endog1": [1.1, 1.9, 2.8, 3.6, 4.4, 5.3, 6.0, 6.9],
        "z1": [0.9, 2.1, 2.7, 3.9, 4.2, 5.6, 5.8, 7.1],
    }
)

result = IV(df, y="y", x_exog=[], x_endog=["endog1"], instruments=["z1"]).fit()

print(result.params)  # {"const": ..., "endog1": ...}
print(result.std_errors)  # {"const": ..., "endog1": ...}
print(result.r_squared)
```

`IVOptions.method` selects `"2sls"` (default) or `"gmm"`. `cov_type` supports the same range as [OLS](#switching-the-type-of-standard-error); for `method="gmm"`, a separate `weight_type` selects the weight matrix used for point estimation. See the [API Reference](api/iv.md) for the full list of options.

### Diagnostics and first-stage results

```python
print(result.weak_instrument_f_statistics)  # {"endog1": ...}
print(
    result.overid_statistic, result.overid_p_value
)  # None, None (just-identified)
print(
    result.wu_hausman_statistic, result.wu_hausman_p_value
)  # method="2sls" only

first_stage = result.first_stage()
print(
    first_stage["endog1"].params
)  # OLSResults for endog1 ~ x_exog + instruments
```

See the [API Reference](api/iv.md#diagnostics) for what each diagnostic tests and when it is `None`.

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
