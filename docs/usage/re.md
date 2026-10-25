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

`REOptions.cov_type` has the same defaults and support as FE (`"cluster"` on `entity` by default; `"hc0"` unsupported). See [Options and results](../api/re.md) for the full list of options, including Driscoll-Kraay standard errors (`cov_type="dk"`, requires `dk_time`). Like FE, RE has no `predict()`, `augment()`, or `summary()`.
