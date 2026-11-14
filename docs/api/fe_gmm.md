# FEGMM: options and results

**FEGMM**: [Usage](../usage/fe_gmm.md) · **Options and results** · [Definitions](../methods/fe_gmm.md)

`FEGMM` estimates a fixed effects panel regression with endogenous regressors by the generalized method of moments (GMM), with the same pattern as [OLS](ols.md): construct the estimator, then call `.fit()`. The independent variables are split into `x_exog` (exogenous, may be empty) and `x_endog` (endogenous, at least one), plus `instruments` (the excluded instruments, at least one per endogenous variable), as in [FEIV](fe_iv.md). `effects` is a required list of one to five columns, as in [FE](fe.md). There is no intercept, since the effects absorb it. The estimator, covariance types and diagnostics are defined in [Definitions: FEGMM](../methods/fe_gmm.md).

## Options at a glance

Pass an `FEGMMOptions` as `options=`. `effects` is an argument of `FEGMM`, not an option, and there is no `include_intercept`. Options of `None` take the effective default in parentheses where the option is used.

| Option | Purpose | Default | Notes |
|---|---|---|---|
| `gmm_type` | `"one_step"`, `"two_step"` or `"iterated"` | `None` (`"two_step"`) | |
| `gmm_weight_type` | Weight matrix of the point estimate: `"classical"`, `"robust"` or `"cluster"` | `None` (`"cluster"`) | Not with `gmm_type="one_step"`; there is no `"hac"`. Also the weight used by `c_stat()` |
| `gmm_max_iter` | Maximum number of estimations, 3 to 10,000 | `None` (100) | `gmm_type="iterated"` only |
| `gmm_tol` | Convergence tolerance | `None` (`1e-6`) | `gmm_type="iterated"` only |
| `raise_on_non_convergence` | Raise an error if iteration does not converge | `None` (`True`) | `gmm_type="iterated"` only |
| `cov_type` | Standard error type: `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, `"dk"` | `"cluster"` | `"hc0"` is not available. Independent of `gmm_weight_type` |
| `cluster` | Column that defines the clusters | `None` (the first column of `effects`) | With `cov_type="cluster"` or `gmm_weight_type="cluster"` |
| `dk_time` | Column that defines the time periods of the Driscoll-Kraay kernel | `None` | Required with `cov_type="dk"`; never taken from `effects` |
| `dk_bandwidth` | Kernel bandwidth | `None` (automatic) | Only with `cov_type="dk"` |
| `confidence_level` | Level of `conf_int`, in (0, 1) | `0.95` | |
| `tol` | Convergence tolerance of the alternating projections that remove the effects of a two-way model on an unbalanced panel, or of a model with three or more effects | `None` (1e-12) | Ignored on a balanced two-way panel and in a one-way model. Not related to `gmm_tol` |
| `max_iter` | Maximum number of passes of those projections, from 1 to 10000 | `10000` | Reaching the limit raises `ComputationError`. Not related to `gmm_max_iter` |

Which options go with which setting (an option set where it is not used raises a `ValidationError`):

| Setting | Required | Optional |
|---|---|---|
| `cov_type` `"classical"`, `"hc1"` to `"hc3"` | | |
| `cov_type="cluster"` (default) | | `cluster` (default: the first column of `effects`) |
| `cov_type="dk"` | `dk_time` | `dk_bandwidth` |
| `gmm_weight_type="cluster"` (default) | | `cluster` (default: the first column of `effects`) |

`cluster` is shared by `cov_type="cluster"` and `gmm_weight_type="cluster"`. It is used as soon as either needs it, so with the defaults it is always used, and it is rejected only when neither the covariance nor the weight uses it. `gmm_max_iter`, `gmm_tol` and `raise_on_non_convergence` need `gmm_type="iterated"`, and `gmm_weight_type` is not allowed with `gmm_type="one_step"`. The GMM type and weight type actually used are echoed back on the results (`gmm_type`, `gmm_weight_type`; `gmm_weight_type` is `None` for `"one_step"`).

`dk_time` is independent of `effects`: `effects` sets the effects of the model, whereas `dk_time` sets the periods of the Driscoll-Kraay kernel, and it may be the same column as the second column of `effects` or a different one. It cannot be the `y` column or the first column of `effects`.

The fields of the results are listed by purpose at the top of the results class below.

::: econometricsmodels.FEGMM
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.FEGMMOptions

::: econometricsmodels.FEGMMResults

`FEGMMResults.c_stat()` returns the same result type as `IVResults.c_stat()`, which is described on the [IV page](iv.md#econometricsmodels.CStatisticResult) (see also [Usage](../usage/fe_gmm.md#exogeneity-and-instrument-validity-post-estimation)): `CStatisticResult`.
