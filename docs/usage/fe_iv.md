# FEIV: usage

**FEIV**: **Usage** · [Options and results](../api/fe_iv.md) · [Definitions](../methods/fe_iv.md)

`FEIV` estimates a panel regression with fixed effects in which some regressors are endogenous. It removes the fixed effects as [FE](fe.md) does and then applies two-stage least squares (FE-2SLS, the default) or the generalized method of moments (FE-GMM, `estimator="gmm"`) to the within-transformed data, so it fits a model such as the effect of `endog` on `y` within a firm, where `endog` is instrumented. As in [IV](iv.md), `estimator` in `FEIVOptions` selects the estimator and the arguments are the same for both. The arguments combine those of [IV](iv.md) and [FE](fe.md):

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
- **Scope.** Unlike `IV`, `FEIV` has no `"hac"` covariance (use `"dk"`), and like `FE` it has no `predict()`, `augment()` or `summary()`. Use `FE` when every regressor is exogenous once the effects are removed, and `FEIV` when a regressor is still correlated with the error within the entities. [GMM](#gmm-estimator) is chosen with `estimator="gmm"`.

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

The diagnostics are the same as in [IV](iv.md#diagnostics-and-first-stage-results) (except that FEIV also reports the Wu-Hausman test with GMM): the weak-instrument partial F statistic, the overidentification test and the Wu-Hausman endogeneity test are computed by `fit()` and read as fields, and the Cragg-Donald, Kleibergen-Paap and effective F statistics are methods that you call afterwards. The first stage, the weak-instrument diagnostics and the Wu-Hausman test depend only on the first stage (and, for Wu-Hausman, on a least-squares auxiliary regression), so they are the same with `estimator="2sls"` and with `estimator="gmm"`: for the same data and `cov_type` the two estimators return the same values. Only the overidentification test differs (Sargan with two-stage least squares, Hansen J with GMM; see [GMM estimator](#gmm-estimator)).

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

## GMM estimator

`FEIVOptions(estimator="gmm")` applies the generalized method of moments to the within-transformed data, weighting the moment conditions by an estimate of their covariance. Use it when you have more instruments than endogenous regressors and want a Hansen J test and a C statistic that allow for correlation within the clusters. With more instruments than endogenous regressors the GMM estimate is asymptotically more efficient than two-stage least squares, under heteroskedasticity or correlation within the clusters, but with few clusters the estimated weight matrix can be unstable. When the model is just identified, the two estimators give the same coefficients.

```python
gmm = FEIV(
    panel,
    y="y",
    x_exog=["x1"],
    x_endog=["endog"],
    instruments=["z1", "z2"],
    effects=["firm"],
    options=FEIVOptions(estimator="gmm"),
).fit()

print(gmm.params)  # {"x1": ..., "endog": ...}: no intercept
print(gmm.std_errors)
print(gmm.estimator, gmm.gmm_type, gmm.gmm_weight_type)  # gmm two_step cluster
print(gmm.overid_statistic, gmm.overid_p_value)  # Hansen J
```

The options are those of `IVOptions` (`gmm_type`, `gmm_weight_type`, `gmm_max_iter`, `gmm_tol`, `raise_on_non_convergence`), and they are accepted only with `estimator="gmm"`. The defaults of the weight and of the standard errors differ from `IV`:

- **Weight matrix.** `gmm_weight_type` defaults to `"cluster"`, with the clusters of `cluster` (by default the first column of `effects`). The cluster weight allows the errors of an entity to be correlated over time, which is the usual situation in a panel: the within transformation even makes the errors of an entity sum to zero. Keep it unless you have a reason not to, and prefer it when the panel is short. `"robust"` assumes the errors are uncorrelated and `"classical"` that they are also homoskedastic; there is no `"hac"` weight.
- **`gmm_type`.** `"one_step"` uses the fixed weight \((\tilde Z'\tilde Z)^{-1}\) and has no weight type; it gives the two-stage least squares point estimate. `"two_step"` (the default) re-estimates once with the estimated weight, and `"iterated"` repeats that until the coefficients converge.
- **Standard errors.** `cov_type` defaults to `"cluster"` as for two-stage least squares and is chosen independently of the weight. `"hc0"` is not available. The tests use the normal distribution and the overall Wald test the chi-squared distribution, not the t and F distributions of two-stage least squares: `stat_dist` is `"normal"` and `wald_dist` is `"chi2"`.

To iterate the weight until the coefficients converge, and use heteroskedasticity-robust standard errors:

```python
options = FEIVOptions(estimator="gmm", gmm_type="iterated", cov_type="hc1")
iterated = FEIV(
    panel,
    y="y",
    x_exog=["x1"],
    x_endog=["endog"],
    instruments=["z1", "z2"],
    effects=["firm"],
    options=options,
).fit()
print(iterated.gmm_type, iterated.gmm_weight_type, iterated.cov_type)  # iterated cluster hc1
print(iterated.n_iter, iterated.converged)
```

### Overidentification: the Hansen J test

When there are more instruments than endogenous regressors, `fit()` computes Hansen's J statistic of the overidentifying restrictions, in `overid_statistic`, `overid_p_value` and `overid_df` (the Sargan statistic with two-stage least squares). A small p-value signals an invalid instrument or a misspecified model. It is `None` when the model is just identified.

The covariance of the moment conditions in J is the one of the weight used for the point estimate, so J follows `gmm_weight_type`, and with the cluster weight it is robust to correlation within the clusters. With `gmm_weight_type="classical"` it equals the Sargan statistic of two-stage least squares. With the `"robust"` weight on a short panel J rejects too often, because the within-transformed errors are correlated over time (see [Definitions: FEIV](../methods/fe_iv.md#hansen-j-statistic)).

### Exogeneity and instrument validity, post-estimation

`c_stat()` is a method, not a field: `fit()` never computes it. It tests whether endogenous regressors in `variables` are in fact exogenous and whether excluded instruments in `instruments` are valid, as the difference of two overidentification statistics (Hansen J statistics with GMM; for two-stage least squares the same construction, with the weight that follows `cov_type`). It re-reads the data passed to `fit()` and estimates the model twice, so it costs about as much as `fit()`. Keep that data unchanged.

```python
exogeneity = gmm.c_stat(variables=["endog"])
print(exogeneity.statistic, exogeneity.p_value, exogeneity.df)

validity = gmm.c_stat(instruments=["z2"])
print(validity.statistic, validity.p_value)
print(validity.j_with_tested, validity.j_without_tested)
```

A small p-value rejects the null that the tested moment conditions are valid: for `variables` that the tested regressors are exogenous, for `instruments` that the tested instruments are uncorrelated with the error (given that the remaining instruments are valid). `df` is the number of tested variables plus the number of tested instruments.

It returns a `CStatisticResult`, the same type as [`IVResults.c_stat()`](iv.md#exogeneity-and-instrument-validity-post-estimation), and is read in the same way. The covariance of the moment conditions (the weight) follows `gmm_weight_type` (the classical one for `gmm_type="one_step"`), not `cov_type`. With the cluster weight the statistic needs at least as many clusters as the columns of the full instrument set (the exogenous regressors, the tested endogenous variables and all excluded instruments). `fit()` checks the clusters needed for the weight of the fit, and a shortfall raises a `ValidationError`. `c_stat()` can detect this only while computing, and raises a `ComputationError`.

`c_stat()` is also available with `estimator="2sls"`. Two-stage least squares has no weight to choose, so the weight follows `cov_type`, as in [IV](iv.md#exogeneity-and-instrument-validity-post-estimation): the classical weight for `"classical"`, the robust weight for `"hc1"`, `"hc2"` and `"hc3"` (the small-sample corrections of those types do not apply to the weight), and the cluster weight for `"cluster"`, grouped by `cluster` (by default the first column of `effects`). `cov_type="dk"` has no corresponding weight matrix, and `c_stat()` raises a `ValidationError` (`C_STATISTIC_COV_TYPE_NOT_SUPPORTED`); use another `cov_type` or `estimator="gmm"`. With the robust weight on a short panel the statistic rejects too often, as Hansen J does. The default cluster weight does not have this problem when every column of `effects` is nested in the clusters; with other clusters the same caveats as for Hansen J apply (see [Definitions: FEIV](../methods/fe_iv.md#exogeneity-and-instrument-validity-post-estimation-c_stat)).

### What each estimator provides

Every field and method is available with both estimators. They differ in the test distributions, in the overidentification test and in how `c_stat()` chooses its weight:

| | `estimator="2sls"` | `estimator="gmm"` |
|---|---|---|
| `first_stage()`, `fixed_effects()` | yes (the coefficients of the estimator used for `fixed_effects()`) | yes |
| `weak_instrument_f_statistics`, `cragg_donald()`, `kleibergen_paap()`, `effective_f()` | yes | yes, the same values |
| `wu_hausman_statistic` and its p-value | yes | yes, the same values |
| `overid_statistic` | Sargan | Hansen J |
| `c_stat()` weight | follows `cov_type` (not `"dk"`) | follows `gmm_weight_type` |
| `stat_dist`, `wald_dist` | `"t"`, `"f"` | `"normal"`, `"chi2"` |

The first-stage coefficients, the weak-instrument diagnostics and the Wu-Hausman test do not depend on the second-stage estimator: both estimators use the same within-transformed variables. They follow `cov_type` (or are homoskedastic, as described above), never the GMM weight. The structural R² statistics are computed from the structural residuals with both estimators.

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name, in the order of `param_names` (`x_exog` first, then `x_endog`). It holds the slope coefficients only: there is no `"const"` and no fixed effect. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[IV](iv.md), [FE](fe.md), [Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
