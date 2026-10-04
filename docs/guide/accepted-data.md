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
| Time order (HAC) | `hac_time` | Integers, floats, `Decimal`, `Date`, `Datetime` (with or without a time zone). Not `Boolean`, which has at most two distinct values |

Not accepted in a numeric role: `String`, `Categorical`, `Enum`, `Date`, `Datetime`, `Time`, `Duration`, `Binary`, `List`, `Array`, `Struct`. Also not accepted as a group identity or a time period: `Time`, `Duration`, `Decimal`, `Binary`, `List`, `Array`, `Struct`. A `Datetime` with a time zone is rejected as a group identity or a time period, with a message that says how to remove the zone: convert it to the zone you want to keep, then call `.dt.replace_time_zone(None)`. (`hac_time` accepts a time zone, and is compared in its own dtype rather than converted to a float; see [Order of observations for HAC](#order-of-observations-for-hac).) A column of the `Null` dtype (every value is missing) is reported as containing missing values, not as a dtype problem.

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

Group identity and time period columns are compared by their text form, so only equality matters for a group identity. A time period also has an order, which comes from the values of the column rather than from the text (see the next section). Floats are accepted but must be finite: a NaN or an infinite value in a float column of any of these roles is an error, as it is for numeric values. Because the comparison is on the text form, `0.0` and `-0.0` are different groups. If such values can occur in a float key, convert the column to an integer or a string first.

## Order of time periods

For `cov_type="dk"` (Driscoll–Kraay) in FE and RE, the periods must be put in time order. The order comes from the values of the time column, not from the text of the labels, so integers with different numbers of digits (`1, 2, ..., 12`), negative numbers and decimals are ordered correctly:

| Dtype of the time column | Order of the periods |
|---|---|
| Integers, floats | Numeric, ascending |
| `Date`, `Datetime` (no time zone) | Chronological |
| `Enum` | The order of the categories in the type, not alphabetical |
| `String`, `Categorical` | Alphabetical (by the bytes of the text) |

A string or a `Categorical` has no time order of its own, so its labels are sorted as text: `"9"` comes after `"10"`, and `"Q10"` comes right after `"Q1"`. If your periods are strings, use labels whose text order is the time order (ISO 8601 dates, zero-padded numbers), or use an `Enum` with the categories listed in time order. The row order of the data does not matter.

The same order applies to the two-way fixed effects returned by `fixed_effects()`: the `"time"` dictionary lists the periods in time order, and the first period is the reference with an effect of 0.

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

Numeric options are checked the same way, both when the options object is created and when an attribute is assigned. An integer option (`hac_lags`, `max_iter`, `gmm_max_iter`, `dk_bandwidth`) must be an `int`, and a real-valued option (`confidence_level`, `tol`, `gmm_tol`, `lower`, `upper`) must be an `int` or a `float`. A `bool`, a string, or `None` where a number is required raises `TypeError`: `OLSOptions(hac_lags=True)` is an error, not one lag. Whether the value is acceptable (a range, NaN, a very large integer) is checked at `fit()` and raises `ValidationError`. The iteration limits `max_iter` (Logit, Probit, Tobit) and `gmm_max_iter` (IV) must be between 1 and 10,000, and between 3 and 10,000: a larger value is rejected, because a problem that does not converge would run for an unbounded time and cannot be interrupted.

## Where to look next

- [Validation and errors](validation.md): the errors raised for missing values, sample size and options, and what is not checked.
- [Inference conventions](inference-conventions.md): the distributions and degrees of freedom behind the reported statistics.

## Order of observations for HAC

For `cov_type="hac"` in OLS, WLS and IV, and for `gmm_weight_type="hac"` in IV, the autocovariances are computed in time order. Without `hac_time` the row order of the data is the time order. With `hac_time`, the observations are sorted by that column, and only the order of its values is used, not the gaps between them.

Every value of `hac_time` must be different. If two rows share a value, their order is undefined, so the call raises a `ValidationError` that names the first two such rows. It does so for a constant column and for a column where only one pair repeats, instead of resolving the tie by row order and returning an estimate that looks time-ordered. If your data has several observations per time point, as in a panel, use FE or RE with `cov_type="dk"` (where observations at the same time are expected); otherwise make the values distinct or leave `hac_time` out to use the row order.

The column is compared in its own dtype, so the conversion to a 64-bit float described above does not apply to it: large integers and nanosecond `Datetime` values keep their exact order.
