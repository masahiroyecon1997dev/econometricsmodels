# FEIV: usage

**FEIV**: **Usage** · [Options and results](../api/fe_iv.md) · [Definitions](../methods/fe_iv.md)

`FEIV` estimates a panel regression with fixed effects in which some regressors are endogenous (fixed effects two-stage least squares, FE-2SLS). It removes the fixed effects as [FE](fe.md) does and then applies two-stage least squares to the within-transformed data, so it fits a model such as the effect of `endog` on `y` within a firm, where `endog` is instrumented. The arguments combine those of [IV](iv.md) and [FE](fe.md):

- `x_exog`: exogenous regressors. May be empty.
- `x_endog`: endogenous regressors. At least one.
- `instruments`: the excluded instruments, with at least as many as `x_endog`. An instrument must vary within the groups of `effects`: one that is constant within every group of a column in `effects` has no variation left after the within transformation, and `fit()` raises a `ValidationError`.
- `effects`: a list of one to five columns that define the fixed effects, exactly as in [FE](fe.md#two-way-effects): the entity first, then the time period for two-way effects, and further grouping columns after that. The first column is also the default clustering column.

Why `x_exog` and `x_endog` rather than a single `x`: see [Why `IV` has `x_exog` and `x_endog` instead of `x`](../guide/api-conventions.md#why-iv-has-x_exog-and-x_endog-instead-of-x).

```python
import polars as pl
from econometricsmodels import FEIV, FEIVOptions

panel = pl.DataFrame(
    {
        "firm": [f"firm{i}" for i in range(1, 9) for _ in range(6)],
        "year": list(range(2015, 2021)) * 8,
        "y": [
              -5.23, 0.31, -0.71, 1.88, -1.74, -2.64, -2.12, 4.91, 0.57, 1.90, 1.15, -1.27,
              -1.32, 0.24, 2.05, 2.61, -0.39, -0.04, -1.58, -0.11, 1.12, 2.46, -0.86, -4.36,
              -4.01, -1.31, 0.81, 2.93, 2.66, 0.93, -2.14, 0.55, -2.73, 2.61, -6.43, -3.84,
              -4.41, 2.34, 0.71, 0.14, -3.24, -4.15, -4.37, -2.50, -0.30, -5.27, 0.01, -1.48,
        ],
        "x1": [
              0.47, -0.27, -0.72, -0.17, 0.79, -0.69, -1.65, 1.80, 0.44, -0.96, -0.09, -1.56,
              -0.07, 0.86, -1.93, 0.87, 0.81, -0.27, 0.10, -0.33, -0.45, -1.86, 1.63, -0.67,
              0.13, -0.23, 0.46, 0.57, -1.02, 0.36, -0.27, -0.88, 1.30, 1.18, -0.90, -0.30,
              -1.01, -1.14, -0.98, 0.55, -1.42, -1.46, 0.29, -0.72, 1.23, -0.96, -0.49, -0.78,
        ],
        "endog": [
              -2.29, -0.01, -0.66, 1.31, -1.27, -2.06, -1.84, 2.43, -0.03, 1.27, 1.26, -1.75,
              -1.51, -1.01, 1.76, 0.77, -0.18, 0.31, -0.86, -1.11, 0.54, 2.22, -1.41, -2.96,
              -2.55, -2.17, 0.62, 1.76, 2.08, -1.10, -0.13, 0.46, -1.20, 1.42, -3.94, -2.51,
              -1.39, 1.81, 1.60, -0.19, -1.16, -2.34, -2.00, -1.22, -1.36, -2.09, 1.41, -0.86,
        ],
        "z1": [
              -1.20, -0.84, 0.60, 0.02, -0.46, -0.24, -1.43, 1.23, -1.22, 0.04, 2.14, -2.55,
              -1.41, -0.72, 0.12, -1.71, -0.24, -0.03, 0.17, -2.39, 0.65, 1.60, 0.44, -0.72,
              -0.61, -2.65, 0.48, 1.50, 0.65, -2.98, -0.43, -0.10, -0.48, 1.03, -1.74, -0.72,
              0.60, 0.99, 0.17, 1.06, 0.52, -1.06, 1.29, -0.51, -1.63, 0.44, 1.81, 0.96,
        ],
        "z2": [
              1.00, 0.19, -2.47, 0.68, 0.16, -2.64, -0.65, -0.88, 1.06, 0.12, 0.30, 0.24,
              -1.33, -1.96, 0.06, -0.29, -0.01, -0.17, -0.50, 0.56, 0.19, -0.62, -1.90, -2.59,
              -1.44, -0.15, 0.91, -0.42, 0.68, 1.16, 2.38, 0.29, 1.49, 0.23, -1.24, -2.27,
              0.30, 0.89, 2.47, -0.67, 0.49, -0.12, -2.37, 0.37, -0.32, -1.28, 1.69, -0.62,
        ],
    }
)

result = FEIV(
    panel,
    y="y",
    x_exog=["x1"],
    x_endog=["endog"],
    instruments=["z1", "z2"],
    effects=["firm"],
).fit()

print(result.params)  # {"x1": ..., "endog": ...}: no intercept
print(result.std_errors)
print(result.n_groups)  # {"firm": 8}
```

There is no intercept, since the fixed effects absorb it. Two more things differ from the other estimators:

- **Standard errors.** `FEIVOptions.cov_type` defaults to `"cluster"` on the first column of `effects`, which is the usual choice when each entity is observed over several periods. Use `"dk"` for dependence across entities in the same period. The choices are explained in [FE: choosing the type of standard error](fe.md#choosing-the-type-of-standard-error); `"hc0"` is not available.
- **Scope.** `FEIV` is two-stage least squares only. Unlike `IV`, it has no GMM estimator, no `"hac"` covariance and no `c_stat()`, and like `FE` it has no `predict()`, `augment()` or `summary()`. Use `FE` when every regressor is exogenous once the effects are removed, and `FEIV` when a regressor is still correlated with the error within the entities.

See [Options and results](../api/fe_iv.md) for the full list of options.

## Two-way effects and the fixed effects themselves

Pass the time period as the second column of `effects` to add time effects. `fixed_effects()` recovers the effect of every group of every column in `effects`, with the same reference groups as [`FE.fixed_effects()`](fe.md#two-way-effects). It uses the actual endogenous regressors with the estimated coefficients, and it re-reads the data passed to `fit()` on every call:

```python
two_way = FEIV(
    panel,
    y="y",
    x_exog=["x1"],
    x_endog=["endog"],
    instruments=["z1", "z2"],
    effects=["firm", "year"],
).fit()

print(two_way.params)
print(two_way.n_groups)  # {"firm": 8, "year": 6}
print(list(two_way.fixed_effects()))  # ["firm", "year"]
```

## Diagnostics and first-stage results

The diagnostics are the same as in [IV](iv.md#diagnostics-and-first-stage-results): the weak-instrument partial F statistic, the Sargan overidentification test and the Wu-Hausman endogeneity test are computed by `fit()` and read as fields, and the Cragg-Donald, Kleibergen-Paap and effective F statistics are methods that you call afterwards.

```python
print(result.weak_instrument_f_statistics)  # {"endog": ...}
print(result.overid_statistic, result.overid_p_value)  # Sargan
print(result.wu_hausman_statistic, result.wu_hausman_p_value)

first_stage = result.first_stage()
print(first_stage["endog"].params)  # FEResults for endog ~ x_exog + instruments
print(first_stage["endog"].fixed_effects())
```

How to read them is the same as for `IV` (see [Diagnostics and first-stage results](iv.md#diagnostics-and-first-stage-results): a weak-instrument F below about 10 signals weak instruments, a small Sargan p-value signals an invalid instrument or a misspecified model, and a small Wu-Hausman p-value supports treating the regressor as endogenous). Two differences concern the diagnostics and one the fit statistics, and all three come from the fixed effects:

- Every statistic is computed from the within-transformed data, with the absorbed fixed effects counted in its degrees of freedom. The Sargan statistic therefore uses the variance `e'e / (n - n_absorbed)`, where `n` is the number of observations and `n_absorbed` is the number of absorbed fixed-effect parameters (`sum(n_groups) - (number of columns in effects - 1)`), and the partial F has `n - l - n_absorbed` residual degrees of freedom, where `l` is the number of exogenous variables plus instruments. [Definitions: FEIV](../methods/fe_iv.md#degrees-of-freedom-of-the-diagnostics) gives the count for each diagnostic.
- `first_stage()` returns one [`FEResults`](../api/fe.md) per endogenous variable, in the order of `x_endog`, with the same fixed effects and `cov_type` as the structural equation. Its overall F statistic is `None`: it is not computed for a first stage.
- The structural R² statistics (`r_squared_within`, `r_squared_between`, `r_squared_overall`) are computed from the structural residuals, so they can be low or negative. `FEIV` has no log-likelihood, AIC or BIC.

## Weak-instrument diagnostics, post-estimation

`cragg_donald()`, `kleibergen_paap()` and `effective_f()` are methods, not fields. `fit()` never computes them; each re-reads the data passed to `fit()` and estimates the model again, so it costs about as much as `fit()`.

```python
cd = result.cragg_donald()
print(cd.statistic)
print(cd.tsls_size_critical_values[0.10])  # keys: 0.10, 0.15, 0.20, 0.25

kp = result.kleibergen_paap()
print(kp.rk_lm_statistic, kp.rk_lm_p_value, kp.rk_wald_f)

eff = result.effective_f(alpha=0.05)
print(eff.statistic)
print(eff.simplified_critical_values[0.10])  # keys: 0.05, 0.10, 0.20, 0.30
```

Which one to use, and how to read the critical values, is the same as in [IV](iv.md#weak-instrument-diagnostics-post-estimation): `cragg_donald()` assumes homoskedastic errors, whereas `kleibergen_paap()` and `effective_f()` follow the `cov_type` of the fit (and its `cluster`, `dk_time` and `dk_bandwidth`). `effective_f()` is for one endogenous variable. They return the same result types as in `IV` (see [Options and results](../api/fe_iv.md)). Two cases are specific to the panel:

- With `cov_type="dk"`, `kleibergen_paap()` and `effective_f()` need more time periods `T` than `r = (number of excluded instruments) - (number of endogenous variables) + 1`. With `T <= r` neither Kleibergen-Paap statistic is defined (the rk LM statistic would be a constant that does not depend on the data), and the method raises a `ComputationError`.
- With `cov_type="hc2"` or `"hc3"`, the leverages include the absorbed fixed effects, as in the covariance of the coefficients.

If a statistic cannot be computed for your data, the method raises a `ComputationError`; see [Validation and errors](../guide/validation.md).

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name, in the order of `param_names` (`x_exog` first, then `x_endog`). It holds the slope coefficients only: there is no `"const"` and no fixed effect. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[IV](iv.md), [FE](fe.md), [Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
