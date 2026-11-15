# FEIV: options and results

**FEIV**: [Usage](../usage/fe_iv.md) · **Options and results** · [Definitions](../methods/fe_iv.md)

`FEIV` estimates a fixed effects panel regression with endogenous regressors by two-stage least squares (FE-2SLS) or by the generalized method of moments (FE-GMM), with the same pattern as [OLS](ols.md): construct the estimator, then call `.fit()`. As in [IV](iv.md), `FEIVOptions.estimator` selects the estimator, and a single `FEIV`/`FEIVResults` pair serves both. The independent variables are split into `x_exog` (exogenous, may be empty) and `x_endog` (endogenous, at least one), plus `instruments` (the excluded instruments, at least one per endogenous variable). `effects` is a required list of one to five columns, as in [FE](fe.md): one column gives one-way effects, two columns (the entity, then the time period) give two-way effects, and three to five columns add further sets of effects. There is no intercept, since the effects absorb it. The estimators, covariance types and diagnostics are defined in [Definitions: FEIV](../methods/fe_iv.md).

## Options at a glance

Pass an `FEIVOptions` as `options=`. `effects` is an argument of `FEIV`, not an option, and there is no `include_intercept`. The options are those of [`FEOptions`](fe.md) with `estimator` and the five GMM options of [`IVOptions`](iv.md) added, and `"hc0"` and `"hac"` are not available. For an option whose default is `None`, the value that is used when it is left unset is given in parentheses.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `estimator` | `"2sls"` or `"gmm"` | `"2sls"` | |
| `cov_type` | Standard error type: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, `"dk"` | `"cluster"` | `"hc0"` is not available. With `estimator="2sls"` also used by the first-stage regressions, by `kleibergen_paap()` and `effective_f()`, and by the Wu-Hausman test; with `estimator="gmm"` independent of `gmm_weight_type` |
| `cluster` | Column that defines the clusters | `None` (the first column of `effects`) | With `cov_type="cluster"` or, with GMM, `gmm_weight_type="cluster"` |
| `dk_time` | Column that defines the time periods of the Driscoll-Kraay kernel | `None` | Required with `cov_type="dk"`; never taken from `effects` |
| `dk_bandwidth` | Kernel bandwidth | `None` (automatic) | Only with `cov_type="dk"` |
| `gmm_type` | `"one_step"` (the fixed weight \((\tilde Z'\tilde Z)^{-1}\), the two-stage least squares estimate), `"two_step"` or `"iterated"` (repeat the second step until the coefficients converge) | `None` (`"two_step"`) | `estimator="gmm"` only |
| `gmm_weight_type` | Weight matrix of the point estimate: `"classical"`, `"robust"` or `"cluster"` | `None` (`"cluster"`) | `estimator="gmm"` only; not with `gmm_type="one_step"`, which has no weight type; there is no `"hac"`. Also the weight used by `c_stat()` |
| `gmm_max_iter` | Maximum number of estimations, 3 to 10000 | `None` (`100`) | `gmm_type="iterated"` only |
| `gmm_tol` | Convergence tolerance | `None` (`1e-6`) | `gmm_type="iterated"` only. The comparison with the previous coefficients also has an absolute floor of `1e-8` |
| `raise_on_non_convergence` | Raise an error if iteration does not converge | `None` (`True`) | `gmm_type="iterated"` only |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `tol` | Convergence tolerance of the alternating projections that remove the effects of a two-way model on an unbalanced panel, or of a model with three or more effects | `None` (1e-12) | Ignored on a balanced two-way panel and in a one-way model; a loose value also coarsens the detection of regressors collinear with the effects. Not related to `gmm_tol` |
| `max_iter` | Maximum number of passes of those projections, from 1 to 10000 | `10000` | A thinly connected panel needs many passes; reaching the limit raises `ComputationError`. Not related to `gmm_max_iter` |

Which options go with which setting (an option set where it is not used raises a `ValidationError`):

| Setting | Required | Optional |
|---|---|---|
| `cov_type` `"classical"`, `"hc1"` to `"hc3"` | | |
| `cov_type="cluster"` (default) | | `cluster` (default: the first column of `effects`) |
| `cov_type="dk"` | `dk_time` | `dk_bandwidth` |
| `estimator="gmm"`, `gmm_weight_type="cluster"` (default) | | `cluster` (default: the first column of `effects`) |

`cluster` is shared by `cov_type="cluster"` and, with `estimator="gmm"`, `gmm_weight_type="cluster"`. It is used as soon as either needs it, so with the GMM defaults it is always used, and it is rejected only when neither the covariance nor the weight uses it. `gmm_type`, `gmm_weight_type`, `gmm_max_iter`, `gmm_tol` and `raise_on_non_convergence` need `estimator="gmm"`; `gmm_max_iter`, `gmm_tol` and `raise_on_non_convergence` also need `gmm_type="iterated"`, and `gmm_weight_type` is not allowed with `gmm_type="one_step"`. The estimator, GMM type and weight type actually used are echoed back on the results (`estimator`, `gmm_type`, `gmm_weight_type`; both GMM fields are `None` for `"2sls"`, and `gmm_weight_type` is `None` for `"one_step"`).

`dk_time` is independent of `effects`: `effects` sets the effects of the model, whereas `dk_time` sets the periods of the Driscoll-Kraay kernel, and it may be the same column as the second column of `effects` or a different one. It cannot be the `y` column or the first column of `effects`.

## What each estimator provides

A method that the estimator does not support raises a `ValidationError` (`METHOD_NOT_AVAILABLE_FOR_ESTIMATOR`), and a field that it does not support is `None`.

| Result | `estimator="2sls"` | `estimator="gmm"` |
|---|---|---|
| `first_stage()`, `fixed_effects()` | yes | method raises |
| `cragg_donald()`, `kleibergen_paap()`, `effective_f()` | yes | method raises |
| `weak_instrument_f_statistics` and its degrees of freedom, `wu_hausman_*` | yes | `None` |
| `overid_*` | Sargan | Hansen J |
| `c_stat()` | method raises | yes |
| `stat_dist`, `wald_dist` | `"t"`, `"f"` | `"normal"`, `"chi2"` |

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.FEIV
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.FEIVOptions

::: econometricsmodels.FEIVResults

`FEIVResults.first_stage()` returns an [`FEResults`](fe.md) for each endogenous variable. `FEIVResults.cragg_donald()`, `kleibergen_paap()` and `effective_f()` return the same result types as `IVResults`, which are described on the [IV page](iv.md#econometricsmodels.CraggDonaldResult) (see also [Usage](../usage/fe_iv.md#weak-instrument-diagnostics-post-estimation)): `CraggDonaldResult`, `KleibergenPaapResult` and `EffectiveFResult`. `FEIVResults.c_stat()` returns a `CStatisticResult`, the same result type as `IVResults.c_stat()`, which is described on the [IV page](iv.md#econometricsmodels.CStatisticResult) (see also [Usage](../usage/fe_iv.md#exogeneity-and-instrument-validity-post-estimation)).
