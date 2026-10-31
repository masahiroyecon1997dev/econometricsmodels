# Getting Started

econometricsmodels takes a polars DataFrame and column names, and returns a result object. Every estimator is used the same way:

1. Create the model with the data, the column names and, optionally, an options object.
2. Call `.fit()`. All validation happens here, and invalid input raises an error instead of being silently repaired.
3. Read the result: attributes such as `params` and `std_errors`, or methods such as `predict()`.

```bash
pip install econometricsmodels
```

## A first regression: OLS

Pass the data, the dependent variable `y` (one column name) and the independent variables `x` (a list of column names).

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

A constant (`const`) is added automatically; there is no formula string. Estimation settings go in an options object:

```python
from econometricsmodels import OLSOptions

options = OLSOptions(cov_type="hc1")  # heteroskedasticity-robust standard errors
result = OLS(df, y="y", x=["x1"], options=options).fit()
```

## Reading the results

Coefficient-level results are dictionaries keyed by coefficient name. For a row-oriented listing, for example for a REST API response, use `params_table()`, which is a plain list of dictionaries that can be passed to polars or `json.dumps`.

```python
for row in result.params_table():
    print(row["param_name"], row["param"], row["std_error"], row["p_value"])
```

Every method's results share these fields, so code that reads one result type works with the others:

| Field | Content |
|---|---|
| `params`, `std_errors`, `test_stats`, `p_values` | `dict` keyed by coefficient name |
| `conf_int` | `dict` of `(lower, upper)` keyed by coefficient name |
| `cov_params` | Covariance matrix of the coefficients under `cov_type`: a `dict` of `dict`s keyed by coefficient name on both axes (Tobit also has `"sigma"` as the last row and column) |
| `param_names` | The coefficient names, in order (`"const"` first when there is an intercept) |
| `stat_dist`, `stat_df` | The distribution behind `test_stats` and `p_values` (`"t"` or `"normal"`) and its degrees of freedom (`None` for the normal distribution) |
| `cov_type`, `n_obs`, `df_resid` | The standard error type used, the number of observations and the residual degrees of freedom |
| `log_likelihood`, `aic`, `bic` | Information criteria (every method except IV) |

The remaining fields depend on the method (for example `r_squared` for OLS, `lr_statistic` for Logit, `hausman_statistic` for RE). They are grouped by purpose at the top of each results class on the method's *Options and results* page.

## The same interface for other models: Logit

Moving to another model does not mean learning a new interface. A binary logistic regression uses the same arguments and the same result accessors; only the outcome (coded 0/1) and some extra methods differ.

```python
from econometricsmodels import Logit

df = pl.DataFrame(
    {
        "y": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
)

result = Logit(df, y="y", x=["x1"]).fit()

print(result.params)
print(result.pseudo_r_squared)
print(result.predict())  # fitted probabilities
print(result.marginal_effects())  # dy/dx with delta-method standard errors
```

Methods with extra inputs add arguments: `IV` takes `x_exog`, `x_endog` and `instruments`, and the panel models `FE` and `RE` take `entity`.

## Handling errors

Invalid input or options (a missing column, missing values, ...) raise `ValidationError`, a subclass of `ValueError`. A problem found during computation (for example a singular design matrix) raises `ComputationError`, a subclass of `RuntimeError`. A wrong argument *type* raises the built-in `TypeError`.

Both package exceptions carry a stable `code` and a `params` dictionary with the values behind the message, so a program can branch on them instead of parsing text:

```python
from econometricsmodels import ComputationError, ErrorCode, ValidationError

try:
    result = OLS(df, y="y", x=["x1", "x2"]).fit()
except ValidationError as e:
    print(e.code, e.params)  # e.g. COLUMN_NOT_FOUND {'column': 'x2', ...}
    print("Input error:", e)  # message and hint
except ComputationError as e:
    if e.code == ErrorCode.DESIGN_MATRIX_SINGULAR:
        print("Collinear columns:", e.params["columns"])
```

Missing values are never dropped for you and collinear columns are never removed: see [Validation and errors](guide/validation.md#reading-an-error) for what an exception contains and why input is rejected rather than repaired, and [Error codes](guide/error-codes.md) for every `code`.

## Every method

Each method has three pages: a short **usage** guide with examples, the **options and results** reference (what each option does and which fields a result has), and the **definitions** (the formulas, covariance types and test distributions).

| Method | For | Usage | Options and results | Definitions |
|---|---|---|---|---|
| OLS | Linear regression | [Usage](usage/ols.md) | [Reference](api/ols.md) | [Definitions](methods/ols.md) |
| WLS | Linear regression with weights | [Usage](usage/wls.md) | [Reference](api/wls.md) | [Definitions](methods/wls.md) |
| Logit | Binary outcome (logistic) | [Usage](usage/logit.md) | [Reference](api/logit.md) | [Definitions](methods/logit.md) |
| Probit | Binary outcome (normal) | [Usage](usage/probit.md) | [Reference](api/probit.md) | [Definitions](methods/probit.md) |
| Tobit | Censored outcome | [Usage](usage/tobit.md) | [Reference](api/tobit.md) | [Definitions](methods/tobit.md) |
| IV | Endogenous regressors (2SLS, GMM) | [Usage](usage/iv.md) | [Reference](api/iv.md) | [Definitions](methods/iv.md) |
| FE | Panel data, fixed effects | [Usage](usage/fe.md) | [Reference](api/fe.md) | [Definitions](methods/fe.md) |
| RE | Panel data, random effects | [Usage](usage/re.md) | [Reference](api/re.md) | [Definitions](methods/re.md) |

The [Guides](guide/accepted-data.md) cover the data the estimators accept, validation and errors, how the test distributions compare with other packages, and how to [pickle and copy](guide/pickling.md) options and results.
