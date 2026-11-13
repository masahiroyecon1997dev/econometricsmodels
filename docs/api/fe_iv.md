# FEIV: options and results

**FEIV**: [Usage](../usage/fe_iv.md) · **Options and results** · [Definitions](../methods/fe_iv.md)

`FEIV` estimates a fixed effects panel regression with endogenous regressors by two-stage least squares (FE-2SLS), with the same pattern as [OLS](ols.md): construct the estimator, then call `.fit()`. The independent variables are split into `x_exog` (exogenous, may be empty) and `x_endog` (endogenous, at least one), plus `instruments` (the excluded instruments, at least one per endogenous variable). `effects` is a required list of one to five columns, as in [FE](fe.md): one column gives one-way effects, two columns (the entity, then the time period) give two-way effects, and three to five columns add further sets of effects. There is no intercept, since the effects absorb it. The estimator, covariance types and diagnostics are defined in [Definitions: FEIV](../methods/fe_iv.md).

## Options at a glance

Pass an `FEIVOptions` as `options=`. `effects` is an argument of `FEIV`, not an option, and there is no `include_intercept`. The options are those of [`FEOptions`](fe.md). `FEIV` is two-stage least squares only, so there is no `estimator` option (as there is in [`IVOptions`](iv.md)).

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `cov_type` | Standard error type: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, `"dk"` | `"cluster"` | `"hc0"` is not available. Also used by the first-stage regressions, by `kleibergen_paap()` and `effective_f()`, and by the Wu-Hausman test |
| `cluster` | Column that defines the clusters | `None` (the first column of `effects`) | Only with `cov_type="cluster"` |
| `dk_time` | Column that defines the time periods of the Driscoll-Kraay kernel | `None` | Required with `cov_type="dk"`; never taken from `effects` |
| `dk_bandwidth` | Kernel bandwidth | `None` (automatic) | Only with `cov_type="dk"` |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `tol` | Convergence tolerance of the alternating projections that remove the effects of a two-way model on an unbalanced panel, or of a model with three or more effects | `None` (1e-12) | Ignored on a balanced two-way panel and in a one-way model; a loose value also coarsens the detection of regressors collinear with the effects |
| `max_iter` | Maximum number of passes of those projections, from 1 to 10000 | `10000` | A thinly connected panel needs many passes; reaching the limit raises `ComputationError` |

Which options go with which `cov_type` (an option set where it is not used raises a `ValidationError`):

| `cov_type` | Required | Optional |
|---|---|---|
| `"classical"`, `"hc1"` to `"hc3"` | | |
| `"cluster"` (default) | | `cluster` (default: the first column of `effects`) |
| `"dk"` | `dk_time` | `dk_bandwidth` |

`dk_time` is independent of `effects`: `effects` sets the effects of the model, whereas `dk_time` sets the periods of the Driscoll-Kraay kernel, and it may be the same column as the second column of `effects` or a different one. It cannot be the `y` column or the first column of `effects`.

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.FEIV
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.FEIVOptions

::: econometricsmodels.FEIVResults

`FEIVResults.first_stage()` returns an [`FEResults`](fe.md) for each endogenous variable. `FEIVResults.cragg_donald()`, `kleibergen_paap()` and `effective_f()` return the same result types as `IVResults`, which are described on the [IV page](iv.md#econometricsmodels.CraggDonaldResult) (see also [Usage](../usage/fe_iv.md#weak-instrument-diagnostics-post-estimation)): `CraggDonaldResult`, `KleibergenPaapResult` and `EffectiveFResult`.
