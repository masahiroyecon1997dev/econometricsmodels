# Arguments and options

Every model is called the same way: the data, the columns that define the model, and an optional options object.

```python
OLS(data, y="y", x=["x1", "x2"], options=OLSOptions(cov_type="hc1")).fit()
```

This page gives the rule for what is an argument and what is an option, lists the arguments of every model, and explains the two names that surprise people most (`x_exog` / `x_endog` in `IV`, and `weight` in `WLS`).

## The rule

**A required column is an argument. An optional or conditionally required setting is a field of the options object.**

- Required columns are the inputs that define the model. Without them the model does not exist, so they come right after `data` and have no default.
- Everything else belongs to the options object (`OLSOptions`, `FEOptions`, ...). That includes columns that only some analyses need, such as `cluster`, `hac_time`, `dk_time` and `time`. They default to `None` and are required only when another option asks for them. For example, `cov_type="cluster"` needs `cluster`.
- `options` is always the last parameter, and it can only be passed by keyword. Every other optional parameter in the package is keyword-only too, so adding an option never changes the meaning of a positional call.

## Naming

- A column argument that takes one column is a `str` and has a singular name: `y`, `weight`, `entity`, and the column options `cluster`, `hac_time`.
- A column argument that takes several columns is a `list[str]`. It is called `x`, starts with `x_`, or is a plural: `x`, `x_exog`, `x_endog`, `instruments`.

## Arguments and column options of every model

| Model | Required arguments after `data` | Column options |
|---|---|---|
| [`OLS`](../usage/ols.md) | `y`, `x` | `cluster`, `hac_time` |
| [`WLS`](../usage/wls.md) | `y`, `x`, `weight` | `cluster`, `hac_time` |
| [`Logit`](../usage/logit.md) | `y`, `x` | `cluster` |
| [`Probit`](../usage/probit.md) | `y`, `x` | `cluster` |
| [`Tobit`](../usage/tobit.md) | `y`, `x` | `cluster` |
| [`IV`](../usage/iv.md) | `y`, `x_exog`, `x_endog`, `instruments` | `cluster`, `hac_time` |
| [`FE`](../usage/fe.md) | `y`, `x`, `entity` | `time`, `cluster`, `dk_time` |
| [`RE`](../usage/re.md) | `y`, `x`, `entity` | `time`, `cluster`, `dk_time` |

The column options are the ones that name a column. The other options (`cov_type`, `include_intercept`, `confidence_level`, and so on) are described on each model's "Options and results" page.

## Why `IV` has `x_exog` and `x_endog` instead of `x`

An instrumental-variables model has two kinds of regressors, and telling them apart is the point of the method: exogenous regressors are their own instruments, endogenous regressors need excluded instruments. A single `x` would hide that split and would have to be told which columns are endogenous somewhere else. The three arguments `x_exog`, `x_endog` and `instruments` follow the usual notation of linearmodels and Stata's `ivregress`. `instruments` lists only the excluded instruments; the exogenous regressors are added to the first stage automatically.

## Why `WLS` has `weight`

The weights are a single column, so the argument is a `str` and singular, like `y` and `cluster`. The name is not `weights` because the package uses one rule for all models: a `str` argument is one column, and a plural name would suggest a list. Python's error message points at the right name if you type `weights`:

```text
TypeError: WLS.__init__() got an unexpected keyword argument 'weights'. Did you mean 'weight'?
```

The package does not accept both spellings, so that each concept has one name.
