# IV: usage

**IV**: **Usage** · [Options and results](../api/iv.md) · [Definitions](../methods/iv.md)

`IV` estimates a linear model in which some regressors are endogenous, that is, correlated with the error term (because of omitted variables, measurement error or simultaneity). The independent variables are split into three groups:

- `x_exog`: exogenous regressors, which are assumed uncorrelated with the error. They act as their own instruments.
- `x_endog`: endogenous regressors, the ones you are worried about.
- `instruments`: the excluded instruments, variables that are not in the model for `y`, are correlated with the endogenous regressors and are uncorrelated with the error. There must be at least one per endogenous variable; with more, the model is over-identified and can be tested (below).

Why these names rather than a single `x`: see [Why `IV` has `x_exog` and `x_endog` instead of `x`](../guide/api-conventions.md#why-iv-has-x_exog-and-x_endog-instead-of-x).

```python
import polars as pl
from econometricsmodels import IV

df = pl.DataFrame(
    {
        "y": [1.0, 2.4, 2.9, 4.3, 5.1, 5.8, 7.2, 7.9],
        "endog1": [1.1, 1.9, 2.8, 3.6, 4.4, 5.3, 6.0, 6.9],
        "z1": [0.9, 2.1, 2.7, 3.9, 4.2, 5.6, 5.8, 7.1],
    }
)

result = IV(df, y="y", x_exog=[], x_endog=["endog1"], instruments=["z1"]).fit()

print(result.params)  # {"const": ..., "endog1": ...}
print(result.std_errors)  # {"const": ..., "endog1": ...}
print(result.r_squared)
```

`IVOptions.estimator` selects `"2sls"` (default) or `"gmm"`. `cov_type` supports the same range as [OLS](ols.md#switching-the-type-of-standard-error), and the guidance on [choosing between them](ols.md#choosing-the-type-of-standard-error) applies; for `estimator="gmm"`, a separate `gmm_weight_type` selects the weight matrix used for point estimation. See [Options and results](../api/iv.md) for the full list of options. `IV` has no `predict()` or `augment()`: the fitted values of an IV regression are not predictions of `y` from the observed regressors.

## 2SLS and GMM

The example below has two instruments for one endogenous variable, so the model is over-identified. With the default classical weight, GMM gives the same coefficients as 2SLS; its advantage appears with a weight matrix that allows heteroskedasticity or clustering.

```python
from econometricsmodels import IVOptions

iv_df = pl.DataFrame(
    {
        "y": [0.53, 2.41, 2.53, 1.31, 3.27, -3.04, 0.37, 0.95, 2.06, 2.16, 2.44, 3.88,
              1.52, -2.06, 0.04, 4.1, 1.63, -1.63, 3.24, 1.23, 0.73, 1.0, 1.36, -3.18],
        "x1": [-0.01, 1.05, 0.74, 0.72, 1.62, -1.21, -0.63, -1.32, -0.11, 1.0, -0.02, 0.5,
               -1.91, 0.15, -0.91, 1.78, 0.89, 0.95, -0.06, 0.61, 0.66, -0.34, -0.5, -0.11],
        "endog1": [-0.42, 0.21, -0.07, -0.29, 0.85, -2.35, 0.86, 0.38, 0.44, -0.06, 0.85, 1.85,
                   0.66, -1.38, 0.23, 1.76, -0.41, -1.59, 1.1, 0.37, -0.37, -0.1, -0.15, -2.69],
        "z1": [-0.61, -0.59, -0.28, -0.73, 0.77, -1.6, 0.82, -0.63, -0.55, -1.35, -0.14, -0.25,
               0.19, -0.53, 0.09, 1.82, 0.41, -0.57, 0.95, -0.13, 0.59, 0.61, -0.39, -1.93],
        "z2": [-0.35, 0.55, -0.38, 0.44, 0.98, -0.54, 1.23, 1.62, 1.08, 1.17, 1.1, 2.25,
               0.19, 0.0, 0.6, -0.91, -1.55, -0.88, 0.37, 0.47, -1.54, -1.88, -0.32, -0.19],
    }
)

two_sls = IV(
    iv_df, y="y", x_exog=["x1"], x_endog=["endog1"], instruments=["z1", "z2"]
).fit()

gmm = IV(
    iv_df,
    y="y",
    x_exog=["x1"],
    x_endog=["endog1"],
    instruments=["z1", "z2"],
    options=IVOptions(estimator="gmm", gmm_weight_type="robust"),
).fit()

print(two_sls.params["endog1"], two_sls.stat_dist)  # t and F tests
print(gmm.params["endog1"], gmm.stat_dist)  # normal and chi-squared tests
```

2SLS reports t and F statistics; GMM, which is an asymptotic estimator, reports normal and chi-squared statistics. The definitions are in [Definitions: IV](../methods/iv.md#estimators-estimator).

## Diagnostics and first-stage results

```python
print(two_sls.weak_instrument_f_statistics)  # {"endog1": ...}
print(two_sls.overid_statistic, two_sls.overid_p_value)  # Sargan (2SLS)
print(two_sls.wu_hausman_statistic, two_sls.wu_hausman_p_value)  # 2SLS only
print(gmm.overid_statistic, gmm.overid_p_value)  # Hansen J (GMM)

first_stage = two_sls.first_stage()
print(first_stage["endog1"].params)  # OLSResults for endog1 ~ x_exog + instruments
```

How to read them:

- **Weak instruments** (`weak_instrument_f_statistics`, one per endogenous variable). The F statistic tests that the excluded instruments do not explain the endogenous variable in the first stage. A small value means the instruments are weak, and the IV estimate and its standard errors are then unreliable. A common rule of thumb is a value above about 10; no verdict is reported, and the statistic assumes homoskedastic errors (see [Inference conventions](../guide/inference-conventions.md#weak-instruments)).
- **Weak instruments, post-estimation** (`cragg_donald()`, `kleibergen_paap()`, `effective_f()`). These are methods, not fields, and have their own section [below](#weak-instrument-diagnostics-post-estimation).
- **Overidentification** (`overid_statistic`, `overid_p_value`; the Sargan test for 2SLS and Hansen's J for GMM). The null hypothesis is that all instruments are valid, that is uncorrelated with the error. A small p-value says that at least one instrument is invalid or the model is misspecified; a large one is consistent with validity but does not prove it. Both are `None` when the model is just-identified (as many instruments as endogenous variables), because there is nothing to test. Only Hansen's J is robust to heteroskedasticity and clustering.
- **Endogeneity** (`wu_hausman_statistic`, `wu_hausman_p_value`, 2SLS only). The null hypothesis is that the suspect regressors are in fact exogenous, so that OLS would be consistent. A small p-value supports treating them as endogenous and using IV; a large one means the data cannot distinguish IV from OLS, and OLS is then more precise. For GMM, or for a test that is robust to heteroskedasticity or clustering, use [`c_stat()`](#exogeneity-and-instrument-validity-post-estimation).

See [Definitions: IV](../methods/iv.md#diagnostics) for what each diagnostic tests, and the [Options and results](../api/iv.md) for when it is `None`.

## Weak-instrument diagnostics, post-estimation

`cragg_donald()`, `kleibergen_paap()` and `effective_f()` are methods rather than fields. They are optional, and a statistic can be undefined for data on which the fit itself succeeds, so `fit()` never computes them. Each re-reads the data passed to `fit()`, and each works the same for `estimator="2sls"` and `estimator="gmm"`.

```python
cd = two_sls.cragg_donald()
print(cd.statistic)
print(cd.tsls_size_critical_values[0.10])  # keys: 0.10, 0.15, 0.20, 0.25

kp = two_sls.kleibergen_paap()
print(kp.rk_lm_statistic, kp.rk_lm_p_value, kp.rk_wald_f)

eff = two_sls.effective_f(alpha=0.05)
print(eff.statistic)
print(eff.simplified_critical_values[0.10])  # keys: 0.05, 0.10, 0.20, 0.30
```

Which one to use:

- `cragg_donald()` is the joint statistic for several endogenous variables and comes with Stock-Yogo critical values. It assumes homoskedastic errors and ignores `cov_type`.
- `kleibergen_paap()` follows the `cov_type` of the fit (for GMM, `cov_type` and not `gmm_weight_type`), so it stays valid under heteroskedasticity, clustering and autocorrelation, and it handles several endogenous variables. It returns no critical values. The rk LM test has the null of underidentification, so rejecting it does not show that the instruments are strong. The rk Wald F is the robust counterpart of the Cragg-Donald statistic; with `cov_type="classical"` the two are equal.
- `effective_f()` follows the `cov_type` of the fit and is for one endogenous variable, with critical values for the worst-case 2SLS bias. It is the statistic designed for one endogenous variable under heteroskedastic, clustered or autocorrelated errors. With one or two excluded instruments, prefer its `simplified_critical_values`.

The critical values are `dict`s keyed by the threshold, and a table that does not cover your model gives `None`. A statistic above the critical value for a threshold rejects the null that the instruments are weak by that criterion: a 2SLS relative bias above the threshold or a size distortion above the threshold for `cragg_donald()`, and a worst-case bias above the threshold for `effective_f()`. It does not prove that the instruments are strong.

If no statistic can be computed for your data, the method raises a `ComputationError`. Two kinds of field can still be `None`: one of the two Kleibergen-Paap statistics (when only the other is defined), and critical values that the tables do not cover (`cragg_donald()`) or the bound that is not defined (`effective_f()`). `effective_f()` raises a `ValidationError` for several endogenous variables or an `alpha` outside (0, 1). The definitions are in [Definitions: IV](../methods/iv.md#weak-instruments-post-estimation).

## Exogeneity and instrument validity, post-estimation

`c_stat()` is the C statistic (difference-in-Hansen test). Like the weak-instrument diagnostics it is a method that `fit()` never calls, and it works for both estimators. It estimates the model twice, with and without the tested moment conditions, and compares the two Hansen J statistics.

```python
# Are the endogenous variables in fact exogenous?
c = gmm.c_stat(variables=["endog1"])
print(c.statistic, c.p_value, c.df)

# Is an excluded instrument uncorrelated with the error?
c = two_sls.c_stat(instruments=["z1"])
print(c.statistic, c.p_value, c.j_with_tested, c.j_without_tested)
```

- `variables` lists the endogenous variables (from `x_endog`) to treat as exogenous. A small p-value rejects their exogeneity, so IV is needed; a large one means the data cannot tell, and OLS would be more precise if they were exogenous.
- `instruments` lists the excluded instruments to test. A small p-value says that at least one tested instrument is not valid, or the model is misspecified; a large one is consistent with their validity but does not prove it. At least as many instruments as endogenous variables must remain, otherwise the model without the tested instruments is not identified.
- Both lists may be given. The null hypothesis is that all tested moment conditions are valid, with one chi-squared degree of freedom for each tested variable and instrument.
- The covariance of the moment conditions follows `gmm_weight_type` for `estimator="gmm"`, and the family of `cov_type` otherwise, so the test is valid under the same assumptions as the fit. It does not depend on `gmm_type`: a model fitted with `gmm_type="one_step"` gets the two-step statistic.
- If no statistic can be computed for your data, the method raises a `ComputationError`; a cluster weight needs at least as many clusters as the model with the tested moment conditions has instruments, which can be more than the fit needed. Names that are not in the model, empty or repeated lists, and a test that leaves the model unidentified raise a `ValidationError`.

The definition is in [Definitions: IV](../methods/iv.md#exogeneity-and-instrument-validity-post-estimation-c_stat).

## Covariance matrix of the coefficients

`result.cov_params` is the covariance matrix of the coefficients under your `cov_type`, as a dictionary of dictionaries keyed by coefficient name: the covariance of `x1` and `x2` is `result.cov_params["x1"]["x2"]`, and the standard errors are the square roots of its diagonal. See [OLS: covariance matrix of the coefficients](ols.md#covariance-matrix-of-the-coefficients) for an example that builds a standard error from it.

## See also

[OLS](ols.md), [Accepted data](../guide/accepted-data.md), [Validation and errors](../guide/validation.md), [Inference conventions](../guide/inference-conventions.md) and [Verification](../guide/verification.md).
