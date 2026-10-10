# Validation and errors

This page explains how econometricsmodels treats invalid input, which exceptions it raises and when, and, just as important, what it does **not** check. A call that does not raise an error does not mean that the model is appropriate for your data.

## Design philosophy

### Reject instead of repair

Where the data or the options cannot be used as given, the package stops with an error rather than quietly fixing the problem. It never changes the estimation sample or the model on its own:

- **Missing values are never dropped.** A null, a NaN or an infinite value in any column used by the model raises an error. There is no listwise deletion.
- **Collinear columns are never removed.** A perfectly collinear design raises an error instead of estimating on a reduced set of regressors.
- **Small groups are never discarded.** Singleton entities in a panel are reported, not dropped.
- **Options that would have no effect are rejected.** For example, passing `cluster` while `cov_type` is not `"cluster"` raises an error instead of being ignored.

### Why missing values are not dropped automatically

Dropping the rows with missing values is a modelling decision, not a technicality. If the values are missing for a reason related to the outcome or to the regressors, the remaining rows are a selected sample, and the estimates describe that selected sample rather than the population you care about (sample selection bias). When rows disappear silently, the number of observations changes without anyone choosing it, and the person running the analysis may not notice, especially in scripts and applications where nobody looks at the intermediate data.

Making the error explicit puts the decision where it belongs. You decide whether to drop the rows, impute the values, or model the missingness, and your code records the choice. With polars this is a few lines:

```python
import polars as pl
from econometricsmodels import OLS

cols = ["y", "x1", "x2"]
clean = df.drop_nulls(cols).filter(pl.all_horizontal(pl.col(cols).is_finite()))
result = OLS(clean, y="y", x=["x1", "x2"]).fit()
print(result.n_obs)  # the sample you chose
```

Null (no value) and NaN (a value that is not a number) are different things in polars, and both are rejected.

### Fail early, at `fit()`

Constructing a model object checks nothing. All validation happens when you call `fit()`, and again in `predict()` and the other methods that take new data. An unknown column name, for example, raises at `fit()`, not when the object is created. The options objects check the *type* of their numeric fields when they are created or assigned (see [Accepted data](accepted-data.md#argument-types)), while the *values* (ranges, NaN) are checked at `fit()`.

### Errors rather than warnings

The package does not emit Python warnings. Conditions that make a result unusable raise an exception. Non-fatal diagnostics are reported as fields of the result object, such as `converged` and `n_iter` for iterative estimators, and statistics that are `None` when they are not available (for example the over-identification test for a just-identified IV model).

### Not a model checker

The checks cover whether the input and the numerical computation are well defined. They do not judge whether the model is appropriate. See [What is not checked](#what-is-not-checked).

## The two exception classes

| Exception | Base class | Meaning |
|---|---|---|
| `ValidationError` | `ValueError` | The input or an option is invalid. It can be decided from the inputs alone, before any estimation work. |
| `ComputationError` | `RuntimeError` | The inputs are acceptable on their face, but the numerical computation fails or cannot be trusted. |

```python
from econometricsmodels import ComputationError, ValidationError

try:
    result = OLS(df, y="y", x=["x1", "x2"]).fit()
except ValidationError as e:
    print("Fix the input or the options:", e)
except ComputationError as e:
    print("The data do not support this model:", e)
```

The exception class is the stable interface. The message text can change between versions, so do not parse it. `ComputationError` has no subclasses: the cause is in the message only.

The boundary is simple in principle: if the problem is visible without doing matrix algebra, it is a `ValidationError`, and if it only shows up while computing, it is a `ComputationError`. Some conditions have been moved from the second class to the first once they turned out to be decidable up front, for example too few clusters for a joint test. Where a case sits close to the boundary, it is listed below under the class that is raised today.

A wrong argument *type*, as opposed to a wrong value, raises the built-in `TypeError`, not `ValidationError`, following the Python convention. The message names the argument and the type it got: passing `x="x1"` instead of `x=["x1"]`, a `bool` or a non-integer for an integer option such as `hac_lags`, or an options object of the wrong class are all `TypeError`. An unknown option name is a `TypeError` as well. The types each argument accepts are listed in [Accepted data](accepted-data.md#argument-types).

## When `ValidationError` is raised

### In every method

| Category | Situation |
|---|---|
| Data type | `data` (or `new_data`) is not a `polars.DataFrame`, for example a pandas DataFrame, a `polars.Series` or a `polars.LazyFrame` (call `.collect()` first). |
| Columns | A named column does not exist. The same column is used in two roles (for example `y` also in `x`, or the weight column as `y`). For the Driscoll–Kraay `dk_time` column only `y` and `entity` are checked: it may also be one of the regressors, or the same column as FE's `time`. `x` is empty, or names a column twice. With an intercept, `x` contains a column called `"const"`. A column has a dtype that its role does not accept, such as a string, date or categorical column used as a number (see [Accepted data](accepted-data.md#column-types-by-role)). The `hac_time` column has the same value in two or more rows, including a constant column, so the observations have no time order (see [Order of observations](accepted-data.md#order-of-observations-for-hac)). |
| Missing values | A null in any used column, or a NaN or infinite value in a numeric column or in a float entity, cluster or time column. A column of the `Null` dtype counts as all missing. |
| Sample size | There are not enough observations: `n <= k`. There are no regressors. |
| Options | `confidence_level` is outside (0, 1) or NaN. An unknown `cov_type` or other option value. `hac_lags` is outside `[0, n)`, including a very large integer. `tol` or `gmm_tol` is not a positive finite number. An option is set that the chosen `cov_type` or estimator does not use. `cov_type="hac"` (or IV's `gmm_weight_type="hac"`) without `hac_time`. |
| Clustering | `cov_type="cluster"` without a `cluster` column. Fewer than two clusters, or no more clusters than slope coefficients, so that the joint test is impossible. |
| Prediction | Columns of `new_data` are missing, null or non-finite. |

### Method-specific

| Method | Situation |
|---|---|
| WLS | A weight is zero, negative, null or NaN. The weight column is missing. |
| OLS (`white_test()`) | `statistic` is not `"lm"` or `"f"`. Too few observations for the auxiliary regression, which has far more columns than the model (all squares and products of the regressors). The result has no retained training data (the `OLSResults` returned by `IVResults.first_stage()`). |
| Logit, Probit | `y` is not coded exactly 0 or 1. `max_iter` is not an integer from 1 to 10,000, or `tol` is not a positive finite number. |
| Tobit | `max_iter` is not an integer from 1 to 10,000, or `tol` is not a positive finite number. Both censoring bounds are `None`, or the lower bound is not below the upper bound. `y` lies outside the bounds. There is no uncensored observation. `x` contains a column named `"sigma"`. |
| IV | Fewer instruments than endogenous regressors (the order condition). `x_endog` or `instruments` is empty. GMM options are set for an estimator that does not use them, or are out of range (`gmm_max_iter` outside 3 to 10,000, or `gmm_tol` not a positive finite number). With `gmm_weight_type="cluster"`, there are too few clusters for the weight matrix. |
| FE | A singleton entity, or in a two-way model a singleton time period. A two-way model on an unbalanced panel, or without `time`. A regressor with no within-variation (constant over time within every entity). Too few degrees of freedom. |
| RE | With `cov_type="cluster"`, too few clusters for the Hausman auxiliary regression. |
| FE, RE | `cov_type="dk"` without `dk_time` (FE does not borrow `time` for it), with an invalid `dk_bandwidth`, with too few time periods for the number of coefficients tested, or with a degenerate two-period structure. |

## When `ComputationError` is raised

| Cause | Where | What it means |
|---|---|---|
| Perfect multicollinearity | All methods | The design matrix is singular, for example one regressor is a multiple of another, or a constant column is combined with an intercept. In IV the first stage fails, and in FE and RE the within transformation leaves a singular design. |
| Near-singular matrices | All methods | A matrix needed for the covariance or a joint test is numerically singular, typically because regressors are on extremely different scales (for example `1e6` and `1e-3`) or because of an almost perfect linear dependence. Rescale the variables. |
| Non-convergence | Logit, Probit, Tobit | The optimizer did not converge within `max_iter`. |
| Singular Hessian or OPG matrix | Logit, Probit, Tobit | The information matrix cannot be inverted. |
| Suspected separation | Logit, Probit | The gradient is near zero but the coefficients are implausibly large, which is the signature of a (quasi-)complete separation of the outcome. |
| Non-convergence | IV (GMM) | The iterated GMM did not converge. |
| Degenerate variance components | RE | For example, no remaining idiosyncratic variance in noise-free data. |
| Singular auxiliary regression | OLS (`white_test()`) | Every regressor is constant, or the auxiliary design stays linearly dependent after constant and duplicate terms are dropped (for example a full set of dummies in a model fitted with `include_intercept=False`), or the auxiliary R² is undefined. The message says `White test`, so it is not confused with the model's own collinearity. |

How the cause is reported can depend on the estimator and on the solver. For instance, completely separated data in a binary-choice model may surface as suspected separation, as non-convergence or as a singular Hessian.

For Logit, Probit, Tobit and IV (GMM), `raise_on_non_convergence=False` returns the result of the last iteration with `converged=False` instead of raising when the optimizer does not converge. Other failures, such as a singular Hessian, are still raised, so inspect `converged` before using a result obtained this way.

## What is not checked

The absence of an error says nothing about the following. They are your responsibility.

- **Model specification.** Functional form, omitted variables, and the choice of regressors are not assessed.
- **Heteroskedasticity, autocorrelation and clustering structure.** The package computes the standard errors you ask for through `cov_type`. It does not test which one is appropriate.
- **Endogeneity.** For OLS, FE and RE nothing tests whether regressors are correlated with the error term. The Hausman test for RE and the Wu–Hausman test for IV are reported as statistics, not used to stop the estimation.
- **Instrument validity and strength.** In IV only the order condition (at least as many instruments as endogenous regressors) is checked up front. The rank condition is not, so irrelevant instruments may surface later as a singular first stage. The weak-instrument F statistic is reported without a verdict, and the over-identification test is `None` when the model is just identified.
- **Stationarity, outliers and influential observations.**
- **Near-separation.** Data that are close to, but not at, separation estimate without an error. Only the detected pathology above is an error, using a heuristic threshold, and very small samples can still be misclassified.
- **Panel structure beyond what the estimator needs.** A one-way FE model does not complain about unbalanced panels or about repeated (entity, time) pairs. Only the two-way model requires a balanced panel.
- **Numeric precision.** Numeric columns are converted to 64-bit floating point without a warning, so integers beyond 2^53 and decimals with many digits can lose precision. See [Accepted data](accepted-data.md#how-values-are-converted).

## Where to look next

- [Accepted data](accepted-data.md): the input type, the column types accepted in each role, and the argument types.
- [Inference conventions](inference-conventions.md): the distributions and degrees of freedom behind the reported statistics.
- [Verification](verification.md): which reference implementations the results are compared with, including the error paths.
- The [API reference](../api/ols.md) for the options of each method.
