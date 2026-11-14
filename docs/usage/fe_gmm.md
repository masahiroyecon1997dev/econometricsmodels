# FEGMM: usage

**FEGMM**: **Usage** · [Options and results](../api/fe_gmm.md) · [Definitions](../methods/fe_gmm.md)

`FEGMM` estimates a panel regression with fixed effects in which some regressors are endogenous, by the generalized method of moments (fixed effects GMM). It removes the fixed effects as [FE](fe.md) does and then applies GMM to the within-transformed data, so it fits the same model as [FEIV](fe_iv.md) but weights the moment conditions by an estimate of their covariance. With more instruments than endogenous regressors that is more efficient than two-stage least squares when the errors are correlated within the clusters, and it gives the Hansen J test of the overidentifying restrictions that is robust to that correlation. The arguments are those of [FEIV](fe_iv.md):

- `x_exog`: exogenous regressors. May be empty.
- `x_endog`: endogenous regressors. At least one.
- `instruments`: the excluded instruments, with at least as many as `x_endog`. An instrument must vary within the groups of `effects`: one that is constant within every group of a column in `effects` has no variation left after the within transformation, and `fit()` raises a `ValidationError`.
- `effects`: a list of one to five columns that define the fixed effects, exactly as in [FE](fe.md#two-way-effects). The first column is also the default clustering column.

```python
import polars as pl
from econometricsmodels import FEGMM, FEGMMOptions

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

result = FEGMM(
    panel,
    y="y",
    x_exog=["x1"],
    x_endog=["endog"],
    instruments=["z1", "z2"],
    effects=["firm"],
).fit()

print(result.params)  # {"x1": ..., "endog": ...}: no intercept
print(result.std_errors)
print(result.gmm_type, result.gmm_weight_type)  # two_step cluster
print(result.overid_statistic, result.overid_p_value)  # Hansen J
```

There is no intercept, since the fixed effects absorb it. The defaults differ from [IV](iv.md) in two ways:

- **Estimation.** `FEGMMOptions.gmm_type` defaults to `"two_step"` and `gmm_weight_type` to `"cluster"`, with the clusters of `cluster` (by default the first column of `effects`). The cluster weight allows the errors of an entity to be correlated over time, which is the usual situation in a panel: the within transformation even makes the errors of an entity sum to zero. `"robust"` assumes they are uncorrelated and `"classical"` that they are also homoskedastic; there is no `"hac"` weight. `gmm_type="one_step"` uses no weight matrix and gives the two-stage least squares point estimate; `"iterated"` repeats the second step until the coefficients converge.
- **Standard errors.** `cov_type` defaults to `"cluster"` as in [FEIV](fe_iv.md) and is chosen independently of the weight. `"hc0"` is not available. The tests use the normal distribution and the overall Wald test the chi-squared distribution, not the t and F distributions of [FEIV](fe_iv.md).

```python
options = FEGMMOptions(gmm_weight_type="robust", cov_type="hc1")
robust = FEGMM(
    panel,
    y="y",
    x_exog=["x1"],
    x_endog=["endog"],
    instruments=["z1", "z2"],
    effects=["firm"],
    options=options,
).fit()
print(robust.gmm_weight_type, robust.cov_type)  # robust hc1
```

See [Options and results](../api/fe_gmm.md) for the full list of options.

## Overidentification: the Hansen J test

When there are more instruments than endogenous regressors, `fit()` computes Hansen's J statistic of the overidentifying restrictions. A small p-value signals an invalid instrument or a misspecified model. It is `None` when the model is just identified.

```python
print(result.overid_statistic, result.overid_p_value, result.overid_df)
```

The covariance of the moment conditions in J is the one of the weight used for the point estimate, so J follows `gmm_weight_type`, and with the cluster weight it is robust to correlation within the clusters. With `gmm_weight_type="classical"` it equals the Sargan statistic of [FEIV](fe_iv.md#diagnostics-and-first-stage-results). With the `"robust"` weight on a short panel J rejects too often, because the within-transformed errors are correlated over time (see [Definitions: FEGMM](../methods/fe_gmm.md#overidentification-overid_statistic)).

## Exogeneity and instrument validity, post-estimation

`c_stat()` is a method, not a field: `fit()` never computes it. It tests whether endogenous regressors in `variables` are in fact exogenous and whether excluded instruments in `instruments` are valid, as the difference of two Hansen J statistics. It re-reads the data passed to `fit()` and estimates the model twice, so it costs about as much as `fit()`. Keep that data unchanged.

```python
exogeneity = result.c_stat(variables=["endog"])
print(exogeneity.statistic, exogeneity.p_value, exogeneity.df)

validity = result.c_stat(instruments=["z2"])
print(validity.statistic, validity.p_value)
print(validity.j_with_tested, validity.j_without_tested)
```

It returns a `CStatisticResult`, the same type as [`IVResults.c_stat()`](iv.md#exogeneity-and-instrument-validity-post-estimation), and is read in the same way. Two things are specific to `FEGMM`. First, the covariance of the moment conditions follows `gmm_weight_type` (the classical one for `gmm_type="one_step"`), not `cov_type`. Second, with the cluster weight the statistic needs at least as many clusters as the columns of the full instrument set (the exogenous regressors, the tested endogenous variables and all excluded instruments). `fit()` checks the clusters needed for the weight of the fit, and a shortfall raises a `ValidationError`. `c_stat()` can only find out while computing, and raises a `ComputationError`.

## What `FEGMM` does not have

`FEGMM` has no first-stage regressions, no weak-instrument diagnostics (`weak_instrument_f_statistics`, `cragg_donald()`, `kleibergen_paap()`, `effective_f()`), no Wu-Hausman test and no `fixed_effects()`; use `c_stat(variables=...)` for the exogeneity of a regressor. Like `FE` it has no `predict()`, `augment()` or `summary()`. The structural R² statistics (`r_squared_within`, `r_squared_between`, `r_squared_overall`) are computed from the structural residuals, so they can be low or negative.

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name, in the order of `param_names` (`x_exog` first, then `x_endog`). It holds the slope coefficients only: there is no `"const"` and no fixed effect. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[FEIV](fe_iv.md), [IV](iv.md), [FE](fe.md), [Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
