# Getting Started

econometricsmodels takes a polars DataFrame and column names, and returns a result object. Every estimator is used the same way:

1. Create the model with the data, the column names and, optionally, an options object.
2. Call `.fit()`. All validation happens here, and invalid input raises an error instead of being silently repaired.
3. Read the result: attributes such as `params` and `std_errors`, or methods such as `predict()`.

```bash
pip install econometricsmodels
```

It needs Python 3.12 or later; polars (1.38.1 or later) is installed with it.

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

Every method's results share these fields (except where the table says otherwise), so code that reads one result type works with the others:

| Field | Content |
|---|---|
| `params`, `std_errors`, `test_stats`, `p_values` | `dict` keyed by coefficient name |
| `conf_int` | `dict` of `(lower, upper)` keyed by coefficient name |
| `cov_params` | Covariance matrix of the coefficients under `cov_type`: a `dict` of `dict`s keyed by coefficient name on both axes (Tobit also has `"sigma"` as the last row and column) |
| `param_names` | The coefficient names, in order (`"const"` first when there is an intercept) |
| `stat_dist`, `stat_df` | The distribution behind `test_stats` and `p_values` (`"t"` or `"normal"`) and its degrees of freedom (`None` for the normal distribution) |
| `cov_type`, `n_obs`, `df_resid` | The standard error type used, the number of observations and the residual degrees of freedom |
| `log_likelihood`, `aic`, `bic` | Log-likelihood and information criteria (every method except IV) |

The remaining fields depend on the method (for example `r_squared` for OLS, `lr_statistic` for Logit, `hausman_statistic` for RE). They are grouped by purpose at the top of each results class on the method's *Options and results* page.

## The same interface for other models: Logit

Moving to another model does not mean learning a new interface. A binary logistic regression uses the same arguments and the same result accessors; only the outcome (coded 0/1) and some extra methods differ.

```python
import polars as pl
from econometricsmodels import Logit

binary = pl.DataFrame(
    {
        "y": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
)

logit_result = Logit(binary, y="y", x=["x1"]).fit()

print(logit_result.params)
print(logit_result.pseudo_r_squared)
print(logit_result.predict())  # fitted probabilities
print(logit_result.marginal_effects())  # dy/dx with delta-method standard errors
```

Methods with extra inputs add arguments: `IV` takes `x_exog`, `x_endog` and `instruments`, `WLS` takes `weight`, and the panel models `FE` and `RE` take `effects`, a list of one column (the entity) or two (the entity, then the time period). Columns that define the model are arguments; everything optional, including optional columns such as `cluster`, goes in the options object. [Arguments and options](guide/api-conventions.md) gives the rule and lists the arguments of every model.

## Handling errors

Invalid input or options (a missing column, missing values, ...) raise `ValidationError`, a subclass of `ValueError`. A problem found during computation (for example a singular design matrix) raises `ComputationError`, a subclass of `RuntimeError`. A wrong argument *type* raises the built-in `TypeError`.

Both package exceptions carry a stable `code` and a `params` dictionary with the values behind the message, so a program can branch on them instead of parsing text:

```python
import polars as pl
from econometricsmodels import OLS, ComputationError, ErrorCode, ValidationError

data = pl.DataFrame(
    {
        "y": [2.1, 3.9, 6.2, 8.1, 9.8],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x1_twice": [2.0, 4.0, 6.0, 8.0, 10.0],
    }
)

for columns in (["x1", "x2"], ["x1", "x1_twice"]):
    try:
        OLS(data, y="y", x=columns).fit()
    except ValidationError as e:
        # "x2" does not exist: COLUMN_NOT_FOUND {'column': 'x2', ...}
        print(e.code, e.params)
        print("Input error:", e)  # message and hint
    except ComputationError as e:
        # "x1_twice" is a multiple of "x1": the design matrix is singular
        if e.code == ErrorCode.DESIGN_MATRIX_SINGULAR:
            print("Collinear columns:", e.params["columns"])
```

Missing values are never dropped for you and collinear columns are never removed: see [Validation and errors](guide/validation.md#reading-an-error) for what an exception contains and why input is rejected rather than repaired, and [Error codes](guide/error-codes.md) for every `code`.

## Every method

Each method has three pages: a short **usage** guide with examples, the **options and results** reference (what each option does and which fields a result has), and the **definitions** (the formulas, covariance types and test distributions). The [table of methods on the home page](index.md#supported-methods) says when to use each one and links to its pages.

The Guides cover [how arguments and options are named and placed](guide/api-conventions.md), the data the estimators accept ([Accepted data](guide/accepted-data.md)), validation and errors, how the test distributions compare with other packages, and how to [pickle and copy](guide/pickling.md) options and results.
