# Accepted data

This page describes what kind of data and arguments the estimators accept: the input type, the column types (dtypes) that can be used in each role, how values are converted, and what happens with anything else. For what is checked once the data is accepted, such as missing values and sample size, see [Validation and errors](validation.md).

## Input type

The data must be an eager `polars.DataFrame`. Nothing else is accepted: not pandas, not a `polars.LazyFrame`, not a `polars.Series`, not a dictionary or a NumPy array.

- A `LazyFrame` raises a `ValidationError` that tells you to call `.collect()` first.
- Only the columns you name are read. Every other column is ignored, whatever its dtype, including nested columns (`List`, `Array`, `Struct`) and columns of types the estimators cannot use.

## Column types by role

A column is read in one of four roles, and each role accepts different dtypes. A dtype that is not accepted in a role raises a `ValidationError` that names the column and its dtype. Nothing is converted behind your back: in particular, a string column is never parsed as numbers, and a date column is never turned into a count of days.

| Role | Arguments | Accepted dtypes |
|---|---|---|
| Numeric value | `y`, `x`, `weight`, `x_exog`, `x_endog`, `instruments` | Integers (`Int8` to `Int128`, `UInt8` to `UInt64`), floats (`Float16`, `Float32`, `Float64`), `Boolean`, `Decimal` |
| Group identity | `entity`, `cluster` | Integers, floats, `String`, `Categorical`, `Enum`, `Boolean`, `Date`, `Datetime` (no time zone) |
| Time period (panel) | `time`, `dk_time` | Integers, floats, `String`, `Categorical`, `Enum`, `Date`, `Datetime` (no time zone) |
| Time order (HAC) | `hac_time` | The numeric dtypes above, `Date`, `Datetime` (with or without a time zone) |

Not accepted in a numeric role: `String`, `Categorical`, `Enum`, `Date`, `Datetime`, `Time`, `Duration`, `Binary`, `List`, `Array`, `Struct`. Also not accepted as a group identity or a time period: `Time`, `Duration`, `Decimal`, `Binary`, `List`, `Array`, `Struct`. A `Datetime` with a time zone is rejected as a group identity or a time period, with a message that says how to remove the zone: convert it to the zone you want to keep, then call `.dt.replace_time_zone(None)`. (`hac_time` is read as a number and accepts a time zone.) A column of the `Null` dtype (every value is missing) is reported as containing missing values, not as a dtype problem.

If the numbers you want to use are stored as strings or as dates, convert them yourself so that the intent is explicit:

```python
import polars as pl

df = df.with_columns(
    pl.col("price").cast(pl.Float64),  # "12.5" -> 12.5
    pl.col("date").dt.year().alias("year"),  # a number you chose
)
```

Booleans are accepted as numbers (`True` is 1 and `False` is 0). This is convenient for dummy variables and binary outcomes built from comparisons, for example `(pl.col("wage") > 20).alias("high_wage")`. The columns returned by `DataFrame.to_dummies()` (`UInt8`) work as regressors directly.

## How values are converted

Numeric values are converted to 64-bit floating point before the estimation. Float16, Float32, integers up to 2^53 in absolute value, and booleans convert exactly. Other values can lose precision:

- **Integers beyond 2^53** (`Int64`, `UInt64`, `Int128`) are rounded to the nearest representable float. Large identifiers stored as integers are safe as `entity` or `cluster`, which compare their text form, but not as numeric values.
- **Decimals** are converted to the nearest float, so a decimal with more than about 15 significant digits is rounded.

The conversion is silent. If the exact value matters, check the range of your data before estimating.

Group identity and time period columns are compared by their text form, so only equality matters (the order of time periods is covered next). Floats are accepted but must be finite: a NaN or an infinite value in a float column of any of these roles is an error, as it is for numeric values. Because the comparison is on the text form, `0.0` and `-0.0` are different groups. If such values can occur in a float key, convert the column to an integer or a string first.

## Order of time periods

For `cov_type="dk"` (Driscoll–Kraay) in FE and RE, the order of the periods is taken from the sort order of the time labels **as text**. `Date` and `Datetime` columns, ISO 8601 strings and zero-padded labels sort as intended. Integer labels with different numbers of digits do not: `1, 2, ..., 12` sort as `1, 10, 11, 12, 2, ...`, which changes the standard errors without any error. The same applies to the other types that are sorted by their text: the labels of an `Enum` or a `Categorical` are sorted alphabetically, not in the order the type defines. Until this is changed, give the time column a form whose text order is the time order, for example a `Date`, a `Datetime` without a time zone, or integers zero-padded in a string column. This does not affect the point estimates, other `cov_type` values, or `hac_time` (which is read as a number).

## Argument types

Arguments that name columns must have the expected Python type. A mistake in the type, as opposed to the value, raises the built-in `TypeError`, and the message names the argument.

| Argument | Type |
|---|---|
| `y`, `entity`, `weight` | `str` |
| `x`, `x_exog`, `x_endog`, `instruments` | `list` of `str` (not a `str`, a tuple, or a `polars.Series`) |
| `options` | The options class of the method (for example `OLSOptions`), or omitted |

```python
# TypeError: 'x' must be a list of column names (e.g. x=["x1"]), got str
OLS(df, y="y", x="x1")

OLS(df, y="y", x=["x1"])  # correct
```

Numeric options are checked the same way, both when the options object is created and when an attribute is assigned. An integer option (`hac_lags`, `max_iter`, `gmm_max_iter`, `dk_bandwidth`) must be an `int`, and a real-valued option (`confidence_level`, `tol`, `gmm_tol`, `lower`, `upper`) must be an `int` or a `float`. A `bool`, a string, or `None` where a number is required raises `TypeError`: `OLSOptions(hac_lags=True)` is an error, not one lag. Whether the value is acceptable (a range, NaN, a very large integer) is checked at `fit()` and raises `ValidationError`.

## Where to look next

- [Validation and errors](validation.md): the errors raised for missing values, sample size and options, and what is not checked.
- [Inference conventions](inference-conventions.md): the distributions and degrees of freedom behind the reported statistics.
