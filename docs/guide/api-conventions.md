# Arguments and options

Every model is called the same way: the data, the columns that define the model, and an optional options object.

```python
from econometricsmodels import FE, OLS, FEOptions, OLSOptions

# data: a polars DataFrame with the columns y, x1, x2 and id
OLS(data, y="y", x=["x1", "x2"], options=OLSOptions(cov_type="hc1")).fit()
FE(data, y="y", x=["x1", "x2"], effects=["id"], options=FEOptions(cov_type="hc1")).fit()
```

This page gives the rule for what is an argument and what is an option, lists the arguments of every model, and explains why `IV` takes `x_exog` and `x_endog` instead of `x`, and why `WLS` takes `weight`, not `weights`.

## The rule

**A required column is an argument. An optional or conditionally required setting is a field of the options object.**

- Required columns are the inputs that define the model. Without them the model does not exist, so they come right after `data` and have no default.
- Everything else belongs to the options object (`OLSOptions`, `FEOptions`, ...). That includes columns that only some analyses need, such as `cluster`, `hac_time` and `dk_time`. They default to `None`. `hac_time` and `dk_time` are required only when another option asks for them (for example, `cov_type="hac"` needs `hac_time`), and so is `cluster` (for example, `cov_type="cluster"` in `OLS` needs `cluster`). In `FE` and `RE`, `cluster` is optional: when it is not given, the first column of `effects` is used. These are called column options below.
- `options` is always the last parameter, and it can only be passed by keyword (`options=...`). The same holds for every other optional parameter of the models and their results, so a new optional parameter never changes the meaning of an existing positional call.

## Naming

- A column argument that takes one column is a `str` and has a singular name: `y`, `weight`, and column options such as `cluster` and `hac_time`.
- A column argument that takes several columns is a `list[str]`. It is called `x`, starts with `x_`, or is a plural: `x`, `x_exog`, `x_endog`, `instruments`, `effects`.

The `TypeError` raised for a value of the wrong type is described in [Argument types](accepted-data.md#argument-types).

## Arguments and column options of every model

| Model | Required arguments after `data` | Options that name a column |
|---|---|---|
| [`OLS`](../usage/ols.md) | `y`, `x` | `cluster`, `hac_time` |
| [`WLS`](../usage/wls.md) | `y`, `x`, `weight` | `cluster`, `hac_time` |
| [`Logit`](../usage/logit.md) | `y`, `x` | `cluster` |
| [`Probit`](../usage/probit.md) | `y`, `x` | `cluster` |
| [`Tobit`](../usage/tobit.md) | `y`, `x` | `cluster` |
| [`IV`](../usage/iv.md) | `y`, `x_exog`, `x_endog`, `instruments` | `cluster`, `hac_time` |
| [`FE`](../usage/fe.md) | `y`, `x`, `effects` | `cluster`, `dk_time` |
| [`RE`](../usage/re.md) | `y`, `x`, `effects` | `cluster`, `dk_time` |

`effects` in `FE` and `RE` is always a list, even for one-way effects (`effects=["id"]`, not `effects="id"`). One column gives one-way effects, and two columns give two-way effects. In the two-way case the order matters: the first column is the entity and the second the time period. The first column is also the default clustering column (see [FE](../usage/fe.md#choosing-the-type-of-standard-error)).

The other options (`cov_type`, `include_intercept`, `confidence_level`, and so on) are described on each model's "Options and results" page, for example [OLS](../api/ols.md).

## Why `IV` has `x_exog` and `x_endog` instead of `x`

An instrumental-variables model has two kinds of regressors, and telling them apart is the point of the method: exogenous regressors are their own instruments, endogenous regressors need excluded instruments. A single `x` would hide that split and would have to be told which columns are endogenous somewhere else. The split into `x_exog`, `x_endog` and `instruments` is the one used by linearmodels (`exog`, `endog`, `instruments`), and Stata's `ivregress` writes the same three groups, as `[exogenous] (endogenous = instruments)`. The `x_` prefix keeps the names consistent with `x`. `instruments` lists only the excluded instruments; the exogenous regressors are added to the first stage automatically.

## Why `WLS` has `weight`

The weights are a single column, so the argument is a `str` and singular, like `y` and `cluster`. The name is not `weights` because the package uses one rule for all models: a `str` argument is one column, and a plural name would suggest a list. If you type `weights`, Python reports the unexpected name. Recent versions of Python also suggest the right one (the output below is from Python 3.14):

```text
TypeError: WLS.__init__() got an unexpected keyword argument 'weights'. Did you mean 'weight'?
```

The package accepts only one spelling, so that each concept has one name.

## Why `FE` and `RE` have `effects`

The effects define the structure of the model: which groups are removed or modelled, and whether the time period is part of it. Without them the model does not exist, so they are a required argument, not an option. Both the entity and the time period are dimensions of the effects, so they are passed together in one list: one column gives one-way effects and two columns give two-way effects, with one signature for both. The columns play different roles, so the position decides which is which: the first column is the entity (it also decides the default clustering column and, in `RE`, the variance component `sigma2_u`), and the second is the time period. The package does not check the order. The Driscoll-Kraay time column `dk_time` is a different thing: it is an option, required only for `cov_type="dk"`, and it is independent of `effects`.

## Where to look next

- [Accepted data](accepted-data.md): the input type, the column types each role accepts, and the `TypeError` raised for a wrong argument type.
- The "Options and results" page of each model, for example [OLS](../api/ols.md): what each option does.
- [FE](../usage/fe.md) and [RE](../usage/re.md): how `effects` sets one-way or two-way effects.
