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

## Predictions and marginal effects

`predict()` and `marginal_effects()` take a `target`: `"expected_latent"` (`E[y*|x] = x'β`), `"expected_observed"` (the default, the censoring-adjusted mean `E[y|x]`), `"prob_uncensored"` (`P(uncensored|x)`), or `"expected_uncensored"` (`E[y|lower<y<upper,x]`, the mean among uncensored observations; `E[y|y>0,x]` for a lower bound of 0). `marginal_effects()` also takes `at` (`"overall"`, `"mean"`, `"median"`) and excludes the constant term.

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

`censoring_fit_check()` compares the observed censoring rate against the model-implied rate for each censored direction (`"lower"` / `"uncensored"` / `"upper"`):

```python
for row in result.censoring_fit_check():
    print(row["category"], row["observed_rate"], row["model_implied_rate"])
```

See [Options and results](../api/tobit.md) for the full list of options.

## Covariance matrix of the coefficients

The covariance matrix of the coefficients is available as `result.cov_params`, in the same form for every method. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients). The matrix also contains `"sigma"`, as the last row and column.
