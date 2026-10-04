# FE

`FE` estimates a one-way (entity) or two-way (entity + time) fixed effects ("within") panel
regression. It shares the general API shape with [OLS](ols.md) (`data`/`y`/`x`/`options`
constructor, `.fit()`), but `entity` (and optionally `time`) is a required argument rather than
an option, since a fixed effects model has no meaning without a panel structure. There is no
intercept: the within transformation removes it structurally, so `FEOptions` has no
`include_intercept` field and `FEResults.param_names` never includes `"const"`.

## One-way vs. two-way

Whether the model is one-way (entity only) or two-way (entity + time) is controlled entirely by
`FEOptions.time`: `None` (default) gives one-way, any column name gives two-way. Two-way fixed
effects require a balanced panel; an unbalanced panel with `time` set raises a
`ValidationError`. One-way fixed effects support unbalanced panels without restriction.

`FEResults.n_periods` is the number of unique time periods for two-way effects and `None` for one-way.

## Standard error types

`FEOptions.cov_type` defaults to `"cluster"` (clustered on `entity`) rather than `"classical"` —
a deliberate departure from [OLS's default](../getting-started.md#switching-the-type-of-standard-error),
following the same convention as `fixest`: panel data almost always has within-entity serial
correlation, and defaulting to a robust-but-not-clustered type would understate it. Supported
values are `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, and `"dk"` — `"hc0"` is **not**
supported (neither `linearmodels` nor `fixest` offer it for panel/FE models).

`"dk"` is not the same Newey-West estimator as OLS's: it is a **Driscoll-Kraay** panel HAC
estimator (`fixest`'s `vcov="DK"`, Stata's `xtscc`), which is robust to both cross-entity and
within-entity correlation. `FEOptions.dk_time` is required with `"dk"`: it names the column that
defines the time periods. It is never taken from `FEOptions.time`, which only sets the two-way
fixed effects, so the fixed effects and the HAC kernel may use different time granularities (for
example quarterly fixed effects with yearly periods for the kernel). `FEOptions.dk_bandwidth` sets
the kernel bandwidth explicitly; when omitted it is chosen automatically from the number of
unique time periods.

The small-sample corrections and the degrees of freedom of the t and F tests follow `fixest`'s
`ssc()` defaults: `"cluster"` scales by `G/(G-1) · (n-1)/(n-K)` and uses `G - 1` degrees of
freedom (`G` = number of clusters), `"dk"` uses `T/(T-1)` with `T` = number of time periods in
place of `G` and `T - 1` degrees of freedom, and the other types use `df_resid`. These differ
from `linearmodels`, which does not apply the `G/(G-1)` factor. See
[Inference conventions](../guide/inference-conventions.md) for the full table.

## Panel R²

`FEResults` reports three separate R² values instead of OLS's single `r_squared`/`adj_r_squared`
pair, since "the" R² is not well-defined once fixed effects are involved:
`r_squared_within` (based on the within-transformed variables), `r_squared_between` (based on
entity-mean variables), and `r_squared_overall` (based on the untransformed variables).

## Recovering the fixed effects

The fixed effects themselves (`α_i` for entity, `γ_t` for time) are not part of the `fit()`
result; call `FEResults.fixed_effects()` separately. One-way returns `dict[str, float]` keyed by
entity id. Two-way returns `dict[str, dict[str, float]]` with top-level keys `"entity"` and
`"time"` — the two-way case has a normalization choice (which effect absorbs the overall level),
so its values do not always match `fixest::fixef()` numerically; see the method's docstring for
the exact convention.

::: econometricsmodels.FE
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.FEOptions

::: econometricsmodels.FEResults
