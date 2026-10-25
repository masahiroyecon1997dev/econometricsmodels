# Probit: usage

**Probit**: **Usage** · [Options and results](../api/probit.md) · [Definitions](../methods/probit.md)

`Probit` estimates a binary probit regression model by maximum likelihood. The dependent variable `y` must be coded 0/1. Its API is identical to [Logit](logit.md) — the only difference is the link function (the standard normal CDF `Φ` in place of the logistic CDF `Λ`).

```python
import polars as pl
from econometricsmodels import Probit

df = pl.DataFrame(
    {
        "y": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
)

result = Probit(df, y="y", x=["x1"]).fit()

print(result.params)  # {"const": ..., "x1": ...}
print(result.std_errors)  # {"const": ..., "x1": ...}
print(result.pseudo_r_squared)
```

`ProbitOptions` supports the same `cov_type` and `solver` choices as `LogitOptions`; see [Options and results](../api/probit.md) for the full list of options. `ProbitResults.predict()`, `augment()`, `pred_table()`, and `marginal_effects()` work exactly like their [Logit](logit.md#predicted-values-and-classification-table) counterparts (substitute `Probit`/`ProbitOptions` for `Logit`/`LogitOptions` in the examples above).
