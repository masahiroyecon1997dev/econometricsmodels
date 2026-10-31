# FE: usage

**FE**: **Usage** · [Options and results](../api/fe.md) · [Definitions](../methods/fe.md)

`FE` estimates a one-way or two-way fixed effects (within) panel regression. Pass the entity (panel unit) identifier column via `entity`; enable two-way effects by also setting `FEOptions.time`.

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

result = FE(df, y="y", x=["x1"], entity="id").fit()

print(result.params)  # {"x1": ...} — no intercept (within-transformed away)
print(result.std_errors)
print(result.r_squared_within)
```

`FEOptions.cov_type` defaults to `"cluster"` on `entity`; `"hc0"` is not supported for FE. See [Switching the type of standard error](ols.md#switching-the-type-of-standard-error) for the other choices, and the [Options and results](../api/fe.md) for the full list of options, including two-way effects (`time`).

`fixed_effects()` recovers the entity (and, for two-way, time) fixed effects post-hoc from the fitted coefficients:

```python
print(result.fixed_effects())  # {"a": ..., "b": ..., "c": ...} for one-way
```

FE has no `predict()`, `augment()`, or `summary()`.

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name: the covariance of `x1` and `x2` is `result.cov_params["x1"]["x2"]`, and the standard errors are the square roots of its diagonal. It holds the slope coefficients only: there is no `"const"` and no fixed effect. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.
