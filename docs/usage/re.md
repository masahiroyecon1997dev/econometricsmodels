# RE: usage

**RE**: **Usage** · [Options and results](../api/re.md) · [Definitions](../methods/re.md)

`RE` estimates a random effects (GLS) panel regression: entity-direction effects, or entity and time effects on a balanced panel when `REOptions.time` is set. The variance components are estimated by Swamy-Arora by default; `REOptions(variance_method="wallace_hussain")` or `"amemiya"` selects the other estimators. Unlike FE, RE has an intercept, and a Hausman test comparing against the equivalent FE specification is computed automatically as part of `fit()`.

```python
import polars as pl
from econometricsmodels import RE

df = pl.DataFrame(
    {
        "y": [1.0, 2.5, 3.0, 4.2, 5.1, 6.0],
        "x1": [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
        "id": ["a", "a", "b", "b", "c", "c"],
    }
)

result = RE(df, y="y", x=["x1"], entity="id").fit()

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)
print(result.hausman_statistic, result.hausman_p_value)
```

`REOptions.cov_type` has the same defaults and support as [FE](fe.md#choosing-the-type-of-standard-error) (`"cluster"` on `entity` by default; `"hc0"` unsupported), and the Driscoll-Kraay type (`cov_type="dk"`) needs `dk_time`. See [Options and results](../api/re.md) for the full list of options. Like FE, RE has no `predict()`, `augment()`, or `summary()`.

## Two-way effects, variance components and standard errors

The example panel has six firms observed in six years.

```python
from econometricsmodels import REOptions

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

one_way = RE(panel, y="y", x=["x1", "x2"], entity="firm").fit()

# Entity and time effects (balanced panel): set `time`
two_way = RE(
    panel, y="y", x=["x1", "x2"], entity="firm", options=REOptions(time="year")
).fit()
print(two_way.sigma2_u, two_way.sigma2_lambda, two_way.sigma2_eps)

# Another estimator of the variance components
wallace = RE(
    panel,
    y="y",
    x=["x1", "x2"],
    entity="firm",
    options=REOptions(variance_method="wallace_hussain"),
).fit()

# Driscoll-Kraay standard errors: name the column of the time periods
dk = RE(
    panel,
    y="y",
    x=["x1", "x2"],
    entity="firm",
    options=REOptions(cov_type="dk", dk_time="year"),
).fit()
print(one_way.sigma2_u, wallace.sigma2_u, dk.std_errors)
```

`variance_method` chooses how the variance of the entity effect (`sigma2_u`), of the time effect (`sigma2_lambda`, two-way only) and of the idiosyncratic error (`sigma2_eps`) are estimated: `"swamy_arora"` (the default), `"wallace_hussain"` or `"amemiya"`. The formulas, and the unbalanced or time-invariant cases that `"amemiya"` rejects, are in [Definitions: RE](../methods/re.md#variance-components-variance_method). Time-invariant regressors are accepted (FE rejects them). The weights of the GLS transformation are available as `theta` (one-way, by entity) or `theta_entity`, `theta_time` and `theta_total` (two-way).

## Reading the Hausman test

`fit()` always computes a Hausman test, and its result is in `hausman_statistic`, `hausman_p_value` and `hausman_df`. The null hypothesis is that the entity effects are uncorrelated with the regressors, which is what makes RE consistent and efficient. A small p-value rejects it: the RE coefficients differ systematically from those of [FE](fe.md), and FE is the safer choice. A large p-value means the data do not show a difference, and RE, which is more efficient, can be used. Like any test, a large p-value does not prove that the null is true, especially in a small panel.

```python
print(one_way.hausman_statistic, one_way.hausman_p_value, one_way.hausman_df)
```

The test follows `cov_type` (with the default `"cluster"` it is the cluster-robust version). It tests the regressors that vary within entities: the three values are `None` when no coefficient can be tested. See [Definitions: RE](../methods/re.md#hausman-test).

## Fixed or random effects?

Choose [FE](fe.md) when the entity effects may be correlated with the regressors, and RE when you believe they are not, for example when units are sampled at random from a large population, or when you need the coefficient of a regressor that does not change over time. The Hausman test above compares the two on your data.

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name: the covariance of `x1` and `x2` is `result.cov_params["x1"]["x2"]`, and the standard errors are the square roots of its diagonal. The constant `"const"` comes first. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[FE](fe.md), [Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
