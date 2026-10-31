# Tobit: usage

**Tobit**: **Usage** · [Options and results](../api/tobit.md) · [Definitions](../methods/tobit.md)

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

## Setting the censoring bounds

`TobitOptions(lower=..., upper=...)` gives the bounds. An observation is censored when `y` equals a bound exactly, so the bounds must have the same floating-point values as the data. A side set to `None` is not censored, and at least one bound is required.

```python
# Right-censored at 5.0 (for example a capped score): no lower bound
capped = pl.DataFrame(
    {
        "y": [0.5, 1.2, 2.0, 3.1, 3.8, 5.0, 5.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)
right = Tobit(
    capped, y="y", x=["x1"], options=TobitOptions(lower=None, upper=5.0)
).fit()

# Censored on both sides: at 0.0 below and at 5.0 above
both_sides = pl.DataFrame(
    {
        "y": [0.0, 0.0, 1.2, 2.5, 3.1, 5.0, 5.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)
two_sided = Tobit(
    both_sides, y="y", x=["x1"], options=TobitOptions(lower=0.0, upper=5.0)
).fit()
print(two_sided.params)
```

`fit()` raises a `ValidationError` if both bounds are `None`, if `lower >= upper`, if a bound is NaN or infinite, if a value of `y` lies outside the bounds, or if no observation lies strictly between the bounds.

## Predictions and marginal effects

`predict()`, `augment()` and `marginal_effects()` take a `target`, which says which quantity they report:

| `target` | Quantity | Use it to answer |
|---|---|---|
| `"expected_latent"` | `E[y*]` given `x`, which equals `x'β`; the mean of the uncensored latent outcome | What would the outcome be without censoring? |
| `"expected_observed"` (default) | `E[y]` given `x`: the censoring-adjusted mean of the observed outcome | What do we expect to observe, censoring included? |
| `"prob_uncensored"` | `P(lower < y < upper)` given `x` | How likely is the outcome to be strictly inside the bounds? |
| `"expected_uncensored"` | `E[y]` given `lower < y < upper` and `x`: the mean among uncensored observations (`E[y given y > 0]` for a lower bound of 0) | Among those not at a bound, what do we expect? |

`marginal_effects()` returns the derivative of the chosen quantity with respect to each explanatory variable, so its effect depends on the `target`. It also takes `at` (`"overall"`, `"mean"`, `"median"`) and excludes the constant term. The definitions are in [Definitions: Tobit](../methods/tobit.md#predictions-and-marginal-effects-target).

```python
for row in result.marginal_effects(target="expected_observed"):
    print(row["param_name"], row["effect"], row["std_error"])

# predict() returns a plain list of floats
fitted = result.predict(target="expected_observed")

# new_data (out-of-sample) works the same way as OLS/Logit/Probit
new_data = pl.DataFrame({"x1": [1.0, 2.0]})
predicted = result.predict(new_data=new_data, target="expected_observed")
```

`augment()` takes the same `target`/`new_data` arguments as `predict()`, but returns a polars DataFrame instead of a plain list. Like Logit/Probit's `"predicted_probability"` column, the appended column has a `predicted_` prefix; here it is named `"predicted_{target}"` (e.g. `"predicted_expected_observed"`), since `predict()`'s meaning depends on `target` — this also lets you call `augment()` once per `target` on the same DataFrame without a column name collision.

```python
augmented = result.augment(target="expected_observed")
print(augmented)  # original columns, plus "predicted_expected_observed"

# Stack a second target onto the same DataFrame without a name collision
augmented = result.augment(new_data=augmented, target="prob_uncensored")
```

## Censoring fit check

`censoring_fit_check()` compares the observed share of observations in each category (`"lower"` / `"uncensored"` / `"upper"`; a side without a bound has no category) with the share the model implies. A large gap suggests the censored-normal model does not describe the data well.

```python
for row in result.censoring_fit_check():
    print(row["category"], row["observed_rate"], row["model_implied_rate"])
```

See [Options and results](../api/tobit.md) for the full list of options. Standard errors (`cov_type`: `"classical"`, `"opg"`, `"hc0"`, `"hc1"`, `"cluster"`) are chosen as for [Logit](logit.md#standard-errors-and-solvers); the tests use the normal distribution.

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name: the covariance of `x1` and `x2` is `result.cov_params["x1"]["x2"]`, and the standard errors are the square roots of its diagonal. The matrix also contains `"sigma"`, as the last row and column. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[Logit](logit.md), [Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
