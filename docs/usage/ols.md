# OLS: usage

**OLS**: **Usage** · [Options and results](../api/ols.md) · [Definitions](../methods/ols.md)

Pass the column names of the dependent variable (`y`) and independent variables (`x`), along with the polars DataFrame to estimate on, to `OLS`, then call `.fit()`. The data set below also has grouping and time columns, which the later examples on this page use.

```python
import polars as pl
from econometricsmodels import OLS

df = pl.DataFrame(
    {
        "y": [4.7, 6.3, 8.9, 10.2, 10.4, 14.7, 13.1, 17.8, 19.2, 21.9, 22.5, 26.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "x2": [3.1, 4.5, 3.9, 1.1, 1.5, 4.4, 0.0, 4.1, 4.0, 2.3, 1.5, 1.4],
        "firm_size": [11.0, 31.0, 29.0, 47.0, 35.0, 31.0, 30.0, 20.0, 10.0, 18.0, 38.0, 18.0],
        "group_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "period": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "year": [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
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
options = OLSOptions(cov_type="cluster", cluster="group_id")
result = OLS(df, y="y", x=["x1"], options=options).fit()

# HAC (Newey-West) standard errors: name the column that gives the time order.
# The row order is never assumed, so add an index column if the rows are already sorted.
options = OLSOptions(cov_type="hac", hac_time="period")
result = OLS(df, y="y", x=["x1"], options=options).fit()
```

When `hac_lags` is omitted, the number of lags is chosen automatically as `floor(4 * (n / 100) ** (2 / 9))`. The lag count actually used (the explicit `hac_lags` or the automatic value) is available as `result.hac_lags_used`, which is `None` for any other `cov_type`.

See [Options and results](../api/ols.md) for the full list of available options.

## Retrieving results

`OLSResults` exposes coefficients, standard errors, etc. as dictionaries keyed by coefficient name (`str`). If you need a row-oriented listing — e.g. for a REST API response — use `params_table()`.

Every method's results share these fields, so code that reads one result type works with the others:

| Field | Content |
|---|---|
| `params`, `std_errors`, `test_stats`, `p_values` | `dict` keyed by coefficient name |
| `conf_int` | `dict` of `(lower, upper)` keyed by coefficient name |
| `param_names` | The coefficient names, in order (`"const"` first when there is an intercept) |
| `stat_dist`, `stat_df` | The distribution behind `test_stats` and `p_values` (`"t"` or `"normal"`) and its degrees of freedom (`None` for the normal distribution) |
| `cov_type`, `n_obs`, `df_resid` | The standard error type used, the number of observations and the residual degrees of freedom |
| `log_likelihood`, `aic`, `bic` | Information criteria (OLS, WLS, Logit, Probit, Tobit, FE, RE) |

The remaining fields depend on the method (for example `r_squared` for OLS, `lr_statistic` for Logit, `hausman_statistic` for RE) and are listed on each method's API page.

```python
for row in result.params_table():
    print(row["param_name"], row["param"], row["std_error"], row["p_value"])
```

## Predicted values

`OLSResults.predict()` returns predicted values. With no arguments, it returns the fitted values for the training data used in `fit()`; passing `new_data` returns out-of-sample predictions for new data instead.

```python
# Fitted values for the training data
fitted = result.predict()

# Predictions for new data (columns must match the `x` columns used at fit
# time by name; column order does not matter, and the constant column must
# not be included)
new_data = pl.DataFrame({"x1": [13.0, 14.0]})
predicted = result.predict(new_data=new_data)

print(predicted)  # [..., ...] (a plain list of floats)
```

`OLSResults.augment()` takes the same `new_data` argument, but instead returns the source data (the training data, or `new_data` when given) with the predicted values appended as a `"predicted"` column — a polars DataFrame rather than a plain list. This is the one exception to the library's general policy of not returning DataFrames.

```python
augmented = result.augment(new_data=new_data)
print(augmented)  # original `new_data` columns, plus a "predicted" column
```

## Diagnostics: White test for heteroskedasticity

`OLSResults.white_test()` is a post-estimation diagnostic: it is never computed by `fit()`, and you call it when you want it. It regresses the squared residuals on the independent variables, their squares and their pairwise products. `statistic="lm"` (default) returns the LM version `n * R²` (chi-squared); `statistic="f"` returns the F version of the same auxiliary regression.

```python
result = OLS(df, y="y", x=["x1", "x2"]).fit()  # two regressors
white = result.white_test()
print(white.statistic, white.p_value, white.df)  # attribute access
print(white.aux_terms)  # ["const", "x1", "x2", "x1^2", "x2^2", "x1:x2"]
print(white.dropped_terms)  # terms left out, e.g. ["d^2"] for a 0/1 dummy d

white.to_dict()  # a JSON-ready dict
white_f = result.white_test(statistic="f")
```

The auxiliary regression always includes a constant, even when the model was fitted with `include_intercept=False`, so the first element of `aux_terms` is always `"const"`. Terms that are constant or numerically identical to an earlier term (the square of a 0/1 dummy equals the dummy itself) are dropped, and the degrees of freedom count the terms that remain. The term labels only describe what was used; they are not a formula and are never parsed. The test does not depend on `cov_type`.

## Diagnostics: Breusch-Pagan test for heteroskedasticity

`OLSResults.breusch_pagan_test(*, variables=None, statistic="lm")` is another post-estimation test for heteroskedasticity, never computed by `fit()`. It regresses the squared residuals on a constant and the columns you name in `variables`; with no `variables` it uses the model's regressors. Unlike the White test you choose which variables may drive the error variance, and they need not be in the model.

```python
bp = result.breusch_pagan_test()  # the model's regressors
print(bp.statistic, bp.p_value, bp.df)
print(bp.aux_terms)  # ["const", "x1", "x2"]

bp_size = result.breusch_pagan_test(variables=["firm_size"])  # any numeric column
bp_f = result.breusch_pagan_test(variables=["x1"], statistic="f")  # F version
```

This is the studentized (Koenker) version, which does not assume normal errors. The auxiliary regression always includes a constant, constant or duplicate columns are dropped (`dropped_terms` shows them), and the test does not depend on `cov_type`.

## Diagnostics: Breusch-Godfrey test for serial correlation

`OLSResults.breusch_godfrey_test(time, nlags, *, statistic="lm")` tests for serial correlation of the errors by regressing the residuals on the regressors and their own lags. It is a post-estimation diagnostic, never computed by `fit()`. `time` is the column that gives the time order and `nlags` the number of lags; both are required, because the row order is never assumed to be the time order and there is no natural default lag.

```python
bg = result.breusch_godfrey_test("year", 2)  # LM version (chi-squared)
print(bg.statistic, bg.p_value, bg.df, bg.nlags)
bg_f = result.breusch_godfrey_test("year", 2, statistic="f")  # F version
bg.to_dict()  # a JSON-ready dict
```

The rows may be in any order: they are sorted by `time` before the lags are taken. Only the order of the `time` values is used (gaps between periods are not checked), and the values must be distinct. Lags before the first observation are filled with zero. The test is meaningless for cross-sectional data.
