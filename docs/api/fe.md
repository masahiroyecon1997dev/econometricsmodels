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

## Standard error types

`FEOptions.cov_type` defaults to `"cluster"` (clustered on `entity`) rather than `"classical"` —
a deliberate departure from [OLS's default](../getting-started.md#switching-the-type-of-standard-error),
following the same convention as `fixest`: panel data almost always has within-entity serial
correlation, and defaulting to a robust-but-not-clustered type would understate it. Supported
values are `"classical"`, `"hc1"`, `"hc2"`, `"hc3"`, `"cluster"`, and `"hac"` — `"hc0"` is **not**
supported (neither `linearmodels` nor `fixest` offer it for panel/FE models).

`"hac"` is not the same Newey-West estimator as OLS's: it is a **Driscoll-Kraay** panel HAC
estimator (`fixest`'s `vcov="DK"`, Stata's `xtscc`), which is robust to both cross-entity and
within-entity correlation. Its time ordering comes from `FEOptions.time` by default, or from
`FEOptions.time_col` when set (`time_col` always takes priority, letting the fixed effects
structure and the HAC kernel use different time granularities). `FEOptions.dk_bandwidth` sets
the kernel bandwidth explicitly; when omitted it is chosen automatically from the number of
unique time periods.

## Panel R²

`FEResults` reports three separate R² values instead of OLS's single `r_squared`/`r_squared_adj`
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
