# FE: usage

**FE**: **Usage** · [Options and results](../api/fe.md) · [Definitions](../methods/fe.md)

`FE` estimates a fixed effects (within) panel regression with one to five columns in `effects`. Pass the columns that define the effects as a list in `effects`: one column (the entity, or panel unit, identifier) gives one-way effects, two columns (the entity, then the time period) give two-way effects, and three to five columns add further sets of effects such as an industry or a region ([three or more effects](#three-or-more-effects)). `FE` takes at most five columns in `effects` ([RE](re.md) takes at most two).

```python
import polars as pl
from econometricsmodels import FE

df = pl.DataFrame(
    {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
        "id": ["a", "a", "b", "b", "c", "c"],
    }
)

result = FE(df, y="y", x=["x1"], effects=["id"]).fit()

print(result.params)  # {"x1": ...} — no intercept (within-transformed away)
print(result.std_errors)
print(result.r_squared_within)
```

`fixed_effects()` recovers the fixed effects post-hoc from the fitted coefficients. It returns a single dictionary with one entry per column of `effects`, keyed by the column name. Each entry maps the groups of that column to their effects. The group keys are always strings, even for an integer column such as `year`:

```python
print(result.fixed_effects())  # {"id": {"a": ..., "b": ..., "c": ...}} for one-way
```

FE has no `predict()`, `augment()`, or `summary()`. See [Options and results](../api/fe.md) for the full list of options.

## Two-way effects

Pass two columns in `effects`, the entity first and the time period second, to add time effects as well as entity effects. The panel may be unbalanced (an entity need not be observed in every period); the same (entity, time) combination may also appear in more than one row. The first column is also the default clustering column and the first key of `fixed_effects()`. The example panel below has six firms observed in six years and is used in the rest of this page.

```python
from econometricsmodels import FEOptions

panel = pl.DataFrame(
    {
        "firm": [f"firm{i}" for i in range(1, 7) for _ in range(6)],
        "year": list(range(2015, 2021)) * 6,
        "y": [1.37, 0.04, 0.1, 1.32, 0.96, 1.03, 0.46, 2.14, -0.27, 1.22, 0.59, 3.04,
              -0.33, 1.83, -3.18, -1.24, 0.82, 0.33, -3.02, 0.01, -2.52, -2.68, -0.38, -0.54,
              -0.1, 1.72, -1.51, -1.07, -0.05, 0.28, -2.55, -1.66, -0.84, -3.91, -0.29, -1.17],
        "x1": [0.11, -0.93, -0.03, 0.7, -1.34, -0.46, -1.6, -0.99, -1.54, 0.06, -0.97, 0.57,
               -0.12, -0.46, -2.79, -0.81, -0.32, -0.16, -2.42, -1.37, -1.87, -1.7, 0.17, -1.7,
               -0.49, 0.43, -1.04, -0.57, -0.34, -0.39, -2.22, -0.92, 0.37, -2.54, -0.13, -0.87],
        "x2": [-0.64, 2.0, 0.76, -1.2, 0.07, 0.58, -0.19, 0.68, -0.07, 0.67, 1.44, -0.68,
               0.2, -0.46, 0.13, -1.19, -0.58, -0.2, 0.9, 1.15, -1.32, -0.79, 0.65, -1.99,
               -0.46, -0.1, 1.26, 0.69, -0.33, -0.37, -0.25, 1.52, -0.43, -0.3, 0.35, -0.12],
    }
)

two_way = FE(panel, y="y", x=["x1", "x2"], effects=["firm", "year"]).fit()

print(two_way.params)
print(two_way.fixed_effects())  # {"firm": {...}, "year": {...}}
```

The number of groups in each column is in `two_way.n_groups`, a dictionary keyed by column name in the order of `effects`: here `{"firm": 6, "year": 6}`. The first period (in time order) is the reference with a time effect of 0, and the entity effects absorb the overall level.

In a two-way model the order of the two columns affects only:

- the default clustering column (the first column);
- the order of the keys of `fixed_effects()`;
- which column has the reference period (the second column);
- `r_squared_between`, which uses the means of the first column.

The coefficients do not depend on the order. Neither do the standard errors, unless the default `cov_type="cluster"` is used without `cluster=`, because the default clustering column is then the first column of `effects`.

A regressor that the effects absorb is rejected: one that is constant within every entity, or one that is constant across entities in every period (it varies over time only).

## Three or more effects

Pass three to five columns in `effects` to absorb further sets of effects, for example a firm, a year and an industry. The first column is the entity and the second the time period, as before; every later column only has to identify a group, and it may be a string, an integer, a category or a date (its values are used only as labels: they are not treated as ordered, and are sorted as text just to choose the reference group of `fixed_effects()`). The example adds an industry label to the `panel` of the previous section; it is made up for the example and varies by row:

```python
three = panel.with_columns(
    industry=(
        (pl.col("year") * 2 + pl.col("firm").str.slice(4).cast(pl.Int64)) % 3
    ).replace_strict({0: "tech", 1: "retail", 2: "food"})
)

multi = FE(
    three,
    y="y",
    x=["x1", "x2"],
    effects=["firm", "year", "industry"],
).fit()

print(multi.params)
print(multi.n_groups)  # {"firm": 6, "year": 6, "industry": 3}
print(multi.fixed_effects()["industry"])  # {"food": 0.0, ...}
```

`"food"` is the reference group of `industry` (the first in sorted order), so its effect is 0.

Any combination of the columns may appear in more than one row (for example several plants of the same industry in a firm and year), and the panel need not be complete: the effects are removed by alternating projections and the result is the same as including a dummy variable for every group of every column. `n_groups` and `fixed_effects()` have one entry per column, in the order of `effects`. The other arguments and options work as with fewer columns: `cov_type` defaults to `"cluster"` on the first column, and every `cov_type` is available. The first two columns keep their roles (the entity and the time period); the later columns can be given in any order, apart from the order of the keys of `fixed_effects()`, which follows `effects`. The coefficients never depend on the order of the columns.

The effects of the different columns are not separately identified, only their sum for a row is. `fixed_effects()` therefore fixes a reference: the first column absorbs the overall level, and in each of the other columns one reference group is set to 0 (the first period in time order for the second column, the first label in sorted order for the others). If the groups of a column lie inside the groups of another (every firm in a single industry, say), the two sets of effects cannot be separated and only their sum is meaningful.

Things specific to three or more effects:

- A group with a single observation in any column is rejected, as in a two-way model; drop such rows first. A regressor that the effects absorb completely is rejected too. See [Validation and errors](../guide/validation.md).
- `cov_type="hc2"` and `"hc3"` cannot be used when the groups of all columns except the one with the most groups total more than 3,000; use another `cov_type`, or fewer groups.
- With `cov_type="cluster"`, an effect nested in the clusters (for example the firm effects when clustering by firm) is treated as not estimated in the small-sample correction; see [Definitions: FE](../methods/fe.md#cluster-robust-small-sample-factor).
- On a thinly connected panel the projections may reach `max_iter` and raise an error; raise `tol` or `max_iter` (see [Options and results](../api/fe.md)).

## Choosing the type of standard error

The examples below use the `panel` data frame defined above.

`FEOptions.cov_type` defaults to `"cluster"` on the first column of `effects`, unlike OLS, because the errors of one entity are usually correlated over time. `"hc0"` is not supported for FE.

| `cov_type` | Use it when |
|---|---|
| `"cluster"` (default) | Errors may be correlated within an entity over time (and heteroskedastic). Needs enough entities: the result is unreliable with few. `FEOptions(cluster=...)` names another column to cluster on |
| `"classical"`, `"hc1"` to `"hc3"` | Errors are independent across observations once the effects are removed (homoskedastic for `"classical"`) |
| `"dk"` | Errors are also correlated across entities in the same period (cross-sectional dependence), for example common shocks. Needs `dk_time`, and works best with many periods |

The Driscoll-Kraay type (`"dk"`) needs `dk_time`, the column that gives the time periods. It is independent of `effects`: it may be the second column of `effects` or another column, but not `y` or the first column of `effects`. `dk_bandwidth` sets the number of lags of the kernel (by default chosen from the number of periods); it must be less than the number of periods. The number of periods must also exceed the number of slope coefficients.

```python
by_entity = FE(panel, y="y", x=["x1", "x2"], effects=["firm"]).fit()  # cluster by firm
dk = FE(
    panel,
    y="y",
    x=["x1", "x2"],
    effects=["firm"],
    options=FEOptions(cov_type="dk", dk_time="year"),
).fit()

print(by_entity.std_errors, dk.std_errors)
```

The other types are described in [OLS: choosing the type of standard error](ols.md#choosing-the-type-of-standard-error), and the formulas and small-sample corrections in [Definitions: FE](../methods/fe.md#covariance-of-the-coefficients-cov_type).

## Fixed or random effects?

If a regressor is endogenous, use [FEIV](fe_iv.md), which adds excluded instruments to this model.

Use FE when the entity effects (or the time effects in a two-way model) may be correlated with the regressors, which is the usual concern in observational data: it estimates the effect of changes within an entity and discards differences between entities. [RE](re.md) is more efficient when the effects are uncorrelated with the regressors, and it can estimate the effect of a regressor that does not change over time, which FE cannot. The Hausman test that `RE` computes compares the two: see [RE: reading the Hausman test](re.md#reading-the-hausman-test).

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name: the covariance of `x1` and `x2` is `result.cov_params["x1"]["x2"]`, and the standard errors are the square roots of its diagonal. It holds the slope coefficients only: there is no `"const"` and no fixed effect. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[RE](re.md), [FEIV](fe_iv.md), [Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
