# Validation and errors

This page explains how to read and handle the exceptions econometricsmodels raises, why it rejects invalid input instead of repairing it, when each error occurs, and, just as important, what it does **not** check. A call that does not raise an error does not mean that the model is appropriate for your data.

## Reading an error

A call that cannot run raises one of three exceptions. The first thing to decide is which one, because it tells you who has to act:

| Exception | Base class | Meaning | Usually fixed by |
|---|---|---|---|
| `ValidationError` | `ValueError` | The data or an option is invalid. It can be decided from the inputs alone, before any estimation work. | Changing the data or the options. |
| `ComputationError` | `RuntimeError` | The inputs are acceptable on their face, but a numerical routine fails or cannot be trusted (a singular matrix, an optimizer that does not converge). | Changing the model, or rescaling the data. |
| `TypeError` | `Exception` | An argument has the wrong *type*, as opposed to a wrong value. Built in, so it has no `code`. | Changing the code that calls the estimator. |

### What an exception contains

`ValidationError` and `ComputationError` carry the same four attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `code` | `str` | A stable identifier of the situation, such as `COLUMN_NOT_FOUND`. |
| `params` | `dict` | The values that appear in the text, keyed by name. |
| `message` | `str` | What is wrong. |
| `hint` | `str` or `None` | What to do next. |

`str(e)` is `message` and `hint` joined by a space. For an unknown column:

```python
import polars as pl
from econometricsmodels import OLS, ValidationError

df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x1": [1.0, 2.0, 3.0, 5.0]})

try:
    OLS(df, y="y", x=["zz"]).fit()
except ValidationError as e:
    print(e.code)  # COLUMN_NOT_FOUND
    print(e.params)  # {'column': 'zz', 'argument': 'x', 'data_argument': 'data'}
    print(e.message)  # Column 'zz' (given as x) does not exist in data.
    print(e.hint)  # Check the spelling against data.columns.
    print(e)  # message and hint, joined by a space
```

`params` holds exactly the values that were substituted into `message` and `hint`, so it is also where the data for your own handling comes from. The keys differ by `code`: a missing-values error has `column` and `count`, a collinearity error has `columns` (a list). The keys and their types for every code are in [Error codes](error-codes.md).

### Handling an error in code

Branch on the exception class for the coarse distinction, and on `code` for anything finer. Never parse `message` or `hint`: their wording can be improved between versions. `code`, the class of a `code`, and the keys and types of `params` are the stable interface.

```python
from econometricsmodels import (
    OLS,
    ComputationError,
    ErrorCode,
    ValidationError,
)

try:
    result = OLS(df, y="y", x=["x1", "x2"]).fit()
except ValidationError as e:
    if e.code == ErrorCode.COLUMN_CONTAINS_NULL:
        # e.params["column"], e.params["count"]: your decision how to clean
        ...
    else:
        raise
except ComputationError as e:
    if e.code == ErrorCode.DESIGN_MATRIX_SINGULAR:
        print("Collinear columns:", e.params["columns"])
    else:
        raise
```

`ErrorCode` is a `StrEnum`, so `e.code == ErrorCode.COLUMN_NOT_FOUND` and `e.code == "COLUMN_NOT_FOUND"` are both true. New codes can be added in a later version, so keep a fallback for a `code` you do not know, such as showing `str(e)`. [Error codes](error-codes.md) lists every code and shows how to write your own messages, for example in another language. The reference for the exception classes is on the [Errors](../api/errors.md) page.

In an application, a useful split is to treat `ValidationError` as a response to the caller ("your input is wrong", for example HTTP 400 or 422), with `code` and `params` passed on so the client can render a message, and `ComputationError` as "this data does not support this model". An internal error (a bug in the package) is also a `ComputationError`, with the code `INTERNAL_ERROR`; it is never silently swallowed, and a panic in the Rust core arrives as an ordinary exception that `except Exception` catches.

### Which class is raised

The boundary follows the nature of the check: could you see the problem by looking at the inputs, without running the estimator? If so it is a `ValidationError`; if it needs a numerical routine such as a matrix factorization, or only shows up in an estimated quantity, it is a `ComputationError`. The class does not depend on where in `fit()` the check happens to run. It is a coarse distinction between "fix the input or the options" and "the data do not support this model", although some `ComputationError` codes also suggest a change to the input, such as removing a collinear column or rescaling a variable. For anything finer, branch on the `code`: a code always has the same class. Cases close to the boundary are listed below under their class. For example, a regressor with no variation left after the within transformation is a `ValidationError` (a simple statistic of a column), while a singular design matrix found by the factorization is a `ComputationError`.

### Wrong argument types

A wrong argument *type* raises the built-in `TypeError`, following the Python convention. The message names the argument, the type it expected and the type it got. Passing a pandas DataFrame or a `polars.LazyFrame` as `data`, passing `x="x1"` instead of `x=["x1"]`, a `bool` or a non-integer for an integer option such as `hac_lags`, an options object of the wrong class and an unknown option name are all `TypeError`. The types each argument accepts are listed in [Accepted data](accepted-data.md#argument-types).

## Design philosophy

### Reject instead of repair

Where the data or the options cannot be used as given, the package stops with an error rather than quietly fixing the problem. It never changes the estimation sample or the model on its own:

- **Missing values are never dropped.** A null, a NaN or an infinite value in any column used by the model raises an error. There is no listwise deletion.
- **Collinear columns are never removed.** A perfectly collinear design raises an error instead of estimating on a reduced set of regressors. The one exception is RE's internal regressions that only estimate variance components or run the Hausman test (the between regression and the Hausman auxiliary regression), which drop aliased columns like R `lm` so that a correlated-random-effects specification (a regressor together with its entity mean) can be estimated; the final regression keeps every column.
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

## When `ValidationError` is raised

### Order in which errors are reported

A call is checked in a fixed order, the same for every method, and the **first** problem found is the one reported. If an input has several problems, fixing the error you see can reveal the next one. The order is:

1. **Argument types.** A wrong type raises `TypeError` (see [Wrong argument types](#wrong-argument-types)), before anything else.
2. **Option values.** An unknown string for `estimator`, `cov_type`, `gmm_type`, `gmm_weight_type`, `solver` or `variance_method`, checked in this order (a method without the option skips it). An unknown value is reported even if the option would not be used, for example `gmm_type="bogus"` with `estimator="2sls"`.
3. **Option combinations.** An option that the chosen `cov_type` or estimator does not use (for example `cluster` with `cov_type="classical"`), and then a required option that is missing (for example `hac_time` with `cov_type="hac"`, or `dk_time` with `cov_type="dk"`).
4. **Roles.** The same column used in two roles, a column named twice, an empty `x`, `"const"` in `x` when an intercept is added.
5. **The `y`, `x` and other model columns.** A column that does not exist, has an unsupported dtype or contains missing values.
6. **The columns named by options.** `cluster`, `hac_time` and `dk_time`, with the same checks as in step 5.
7. **The estimation itself.** Checks that need the sizes of the problem, such as `confidence_level`, `max_iter`, the number of observations or clusters, and singular matrices.

Steps 2 to 4 look only at the arguments and options, never at the data. So a typo in `cov_type` is reported even if a column name is also wrong, and an option that the chosen `cov_type` ignores is reported before a column it names is looked up. `ComputationError` can only come from step 7. One detail of step 7: a negative `gmm_max_iter` cannot be converted to a count, so it is rejected together with the columns of step 6, before the other checks of step 7; a positive value outside 3 to 10,000 is checked in step 7 itself.

### In every method

Each bullet is a situation that raises a `ValidationError`.

**Columns**

- A named column does not exist.
- The same column is used in two roles (for example `y` also in `x`, or the weight column as `y`).
- For the Driscoll-Kraay `dk_time` column only `y` and `entity` are checked: it may also be one of the regressors, or the same column as FE's `time`.
- `x` is empty, or names a column twice.
- With an intercept, `x` contains a column called `"const"`.
- A column has a dtype that its role does not accept, such as a string, date or categorical column used as a number (see [Accepted data](accepted-data.md#column-types-by-role)).
- The `hac_time` column has the same value in two or more rows, including a constant column, so the observations have no time order (see [Order of observations](accepted-data.md#order-of-observations-for-hac)).

**Outcome**

- In a model with an intercept, `y` has the same value in every row, up to rounding (there is nothing to explain; the test is the same relative one as for a regressor with no within-variation in FE).
- A model without an intercept may have a constant `y`.

**Missing values**

- A null in any used column, or a NaN or infinite value in a numeric column or in a float entity, cluster or time column.
- A column of the `Null` dtype counts as all missing.

**Sample size**

- There are not enough observations: `n <= k`.
- There are no regressors.
- In IV the structural equation is checked first (`n` must exceed its number of coefficients, including the intercept), then the first stage (`n` must exceed the number of instruments including the intercept and the exogenous regressors).
- At `n` equal to that number the first stage fits perfectly and the 2SLS estimate equals OLS; below it the instruments are singular.

**Options**

- `confidence_level` is outside (0, 1) or NaN.
- An unknown `cov_type` or other option value.
- `hac_lags` is outside `[0, n)`, including a very large integer.
- `tol` or `gmm_tol` is not a positive finite number.
- An option is set that the chosen `cov_type` or estimator does not use.
- `cov_type="hac"` (or IV's `gmm_weight_type="hac"`) without `hac_time`.

**Clustering**

- `cov_type="cluster"` without a `cluster` column.
- Fewer than two clusters, or no more clusters than slope coefficients, so that the joint test is impossible.
- In IV the slope coefficients of the structural equation are counted.
- A first-stage regression with more slope coefficients than clusters does not fail: its overall F statistic is `None` (see [Inference conventions](inference-conventions.md#weak-instruments)).

**Prediction**

- Columns of `new_data` are missing, null or non-finite.

### Method-specific

These are in addition to the situations above.

**WLS**

- A weight is zero, negative, null or NaN.
- The weight column is missing.

**OLS (`breusch_godfrey_test()`)**

- `statistic` is not `"lm"` or `"f"`.
- `nlags` is below 1, or too large for the number of observations (`n <= k + nlags`).
- The `time` column does not exist, has an unsupported dtype, or contains missing values, NaN, infinity or duplicates (the row order is never used as the time order).
- The result has no retained training data (the `OLSResults` returned by `IVResults.first_stage()`).
- `time` or `statistic` that is not a `str`, or `nlags` that is not an `int` (a `bool` or `float` too), raises `TypeError`.

**OLS (`breusch_pagan_test()`)**

- `statistic` is not `"lm"` or `"f"`.
- `variables` is empty, names a column twice, or names a column that does not exist, has an unsupported dtype or contains missing values, NaN or infinity (a column that is not in the model is checked too).
- Every variable is constant.
- Too few observations for the auxiliary regression.
- The result has no retained training data (the `OLSResults` returned by `IVResults.first_stage()`).
- `variables` that is not a `list` of `str`, or `statistic` that is not a `str`, raises `TypeError`.

**OLS (`white_test()`)**

- `statistic` is not `"lm"` or `"f"`.
- Every regressor is constant.
- Too few observations for the auxiliary regression, which has far more columns than the model (all squares and products of the regressors).
- The result has no retained training data (the `OLSResults` returned by `IVResults.first_stage()`).

**Logit, Probit**

- `y` is not coded exactly 0 or 1.
- `max_iter` is not an integer from 1 to 10,000, or `tol` is not a positive finite number.

**Tobit**

- `max_iter` is not an integer from 1 to 10,000, or `tol` is not a positive finite number.
- Both censoring bounds are `None`, or the lower bound is not below the upper bound.
- `y` lies outside the bounds.
- There is no uncensored observation.
- `x` contains a column named `"sigma"`.

**IV**

- Fewer instruments than endogenous regressors (the order condition).
- `x_endog` or `instruments` is empty.
- GMM options are set for an estimator that does not use them, or are out of range (`gmm_max_iter` outside 3 to 10,000, or `gmm_tol` not a positive finite number).
- With `gmm_weight_type="cluster"`, there are too few clusters for the weight matrix.

**FE**

- A singleton entity, or in a two-way model a singleton time period.
- A two-way model on an unbalanced panel, or without `time`.
- A regressor with no within-variation (constant over time within every entity).
- The outcome has no variation left after the fixed effects are removed (constant within every entity, or in a two-way model fully explained by the entity and time effects).
- Too few degrees of freedom.

**RE**

- The outcome is constant within every entity.
- A singleton entity, or in a two-way model (`time` set) a singleton time period.
- A two-way model on an unbalanced panel.
- With `cov_type="cluster"`, too few clusters for the Hausman auxiliary regression (counting only the identified columns).
- A time-invariant regressor is allowed, unlike in FE, and so is a regressor together with its entity or time mean (correlated random effects, Mundlak): the aliased columns of the between and Hausman auxiliary regressions are dropped and `hausman_df` counts the identified ones.
- An unknown `variance_method`, or `variance_method="amemiya"` with a regressor that has no within variation (time-invariant; in a two-way model, varying in at most one dimension): use `"swamy_arora"` or `"wallace_hussain"`.

**FE, RE**

- `cov_type="dk"` without `dk_time` (neither borrows `time` for it), with an invalid `dk_bandwidth`, with too few time periods for the number of coefficients tested, or with a degenerate two-period structure.

## When `ComputationError` is raised

| Cause | Where | What it means |
|---|---|---|
| Perfect multicollinearity | All methods | The design matrix is singular and the error names the columns that take part in one dependency (`columns`; only the first dependency is reported, see [Error codes](error-codes.md#collinear-columns)), for example one regressor is a multiple of another, or a constant column is combined with an intercept. In IV the first stage fails (collinear instruments, an instrument collinear with an exogenous regressor, or a constant instrument), or, when the first stages succeed but the endogenous variables are collinear with each other or with an exogenous regressor, the second stage fails (2SLS) or `X'ZWZ'X` cannot be inverted (GMM); in FE and RE the within transformation leaves a singular design. In RE, a time-invariant regressor that is constant across entities (collinear with the intercept) or collinear with another time-invariant regressor also gives a singular design. |
| Near-singular matrices | All methods | A matrix needed for the covariance or a joint test is numerically singular, typically because regressors are on extremely different scales (for example `1e6` and `1e-3`) or because of an almost perfect linear dependence. Rescale the variables. |
| Non-convergence | Logit, Probit, Tobit | The optimizer did not converge within `max_iter`. |
| Singular Hessian or OPG matrix | Logit, Probit, Tobit | The information matrix cannot be inverted. |
| Uncensored interval too narrow | Tobit (`predict()`, `augment()`, `marginal_effects()` with `target="expected_uncensored"`) | With two-sided censoring, the interval between the bounds is narrower than 1% of the estimated `sigma`, so the model puts almost no probability on uncensored values and the expected value among them cannot be computed reliably. Use another `target`. |
| Separation | Logit, Probit | Some combination of the regressors perfectly predicts the binary outcome, either for all observations (complete separation) or for a subset (quasi-complete separation, for example a dummy variable that equals 1 only when the outcome is 1). No finite maximum likelihood estimate exists, so the coefficients would diverge. It is detected exactly after the optimizer converges, whatever the sample size. Remove or combine the offending regressors, or add observations that break the separation. |
| Non-convergence | IV (GMM) | The iterated GMM did not converge. |
| Perfect fit | OLS, WLS, FE, RE, IV | The residuals are zero up to rounding, so the standard errors and the test statistics are not defined: the residual sum of squares is at most `k · ε` times the total sum of squares (`k` coefficients, `ε` the machine epsilon; equivalent to `1 - R² ≤ k · ε`, that is a residual standard deviation below about `1e-8` of the standard deviation of `y`). It happens when `y` is an exact function of the regressors, typically in synthetic data built without noise, or with an accounting identity. The result does not depend on `cov_type`. In FE it is the within regression, in RE the within regression that estimates the idiosyncratic variance, in IV the structural residuals (GMM: the initial estimate) and each first stage (an instrument that predicts an endogenous regressor exactly; IV's `wu_hausman_statistic` is `None` instead when only its augmented regression is exact). Just above the threshold RE can still raise `MATRIX_NEAR_SINGULAR` (stage `QUASI_DEMEANED_REGRESSION`), because the idiosyncratic variance is then so small that the quasi-demeaned constant column is almost zero. Check that `y` is not an exact function of the regressors; add noise to simulated data. |
| Singular pooled regression, non-positive idiosyncratic variance | RE (`variance_method="wallace_hussain"` or `"amemiya"`) | The first-stage pooled regression of Wallace-Hussain is singular (for example a time-invariant regressor and a multiple of it), the estimated `σ_ε²` is not positive so the quasi-demeaning weights are undefined (plm sets it to 0 and returns `NaN` weights), or the moment equations of an unbalanced one-way panel are singular. |
| Auxiliary matrix too large | OLS (`breusch_godfrey_test()`) | The auxiliary regression matrix (`n` rows by the number of regressors plus `nlags` columns) cannot be allocated, which happens only when `nlags` is close to `n` and `n` is large. Use a smaller `nlags`. |
| Singular auxiliary regression | OLS (`white_test()`, `breusch_pagan_test()`, `breusch_godfrey_test()`) | The auxiliary design stays linearly dependent after constant and duplicate terms are dropped (for example a full set of dummies in a model fitted with `include_intercept=False`), or the auxiliary R² is undefined. For `breusch_godfrey_test()` it also covers residuals that are all zero or fitted exactly by the auxiliary regression. Only columns identical to a constant or to an earlier column are dropped; a column that is a multiple of another (for example `2 * x1`) is not, and makes the auxiliary regression singular. The message names the test (`White test`, `Breusch-Pagan test` or `Breusch-Godfrey test`), so it is not confused with the model's own collinearity. |

How the cause is reported can depend on the estimator and on the solver. For instance, separated data in a binary-choice model may surface as detected separation, as non-convergence or as a singular Hessian, because the optimizer can fail before the separation check runs.

For Logit, Probit, Tobit and IV (GMM), `raise_on_non_convergence=False` returns the result of the last iteration with `converged=False` instead of raising when the optimizer does not converge. Other failures, such as a singular Hessian, are still raised, so inspect `converged` before using a result obtained this way. Detected separation is treated like non-convergence: with `raise_on_non_convergence=False` the result is returned with `converged=False`.

## What is not checked

The absence of an error says nothing about the following. They are your responsibility.

- **Model specification.** Functional form, omitted variables, and the choice of regressors are not assessed.
- **Heteroskedasticity, autocorrelation and clustering structure.** The package computes the standard errors you ask for through `cov_type`. It does not test which one is appropriate.
- **Endogeneity.** For OLS, FE and RE nothing tests whether regressors are correlated with the error term. The Hausman test for RE and the Wu-Hausman test for IV are reported as statistics, not used to stop the estimation.
- **Instrument validity and strength.** In IV only the order condition (at least as many instruments as endogenous regressors) is checked up front. The rank condition is not, so irrelevant instruments may surface later as a singular first stage. The weak-instrument F statistic is reported without a verdict, and the over-identification test is `None` when the model is just identified.
- **Stationarity, outliers and influential observations.**
- **Near-separation.** Data that are close to, but not at, separation have a finite estimate, so they estimate without an error, although the standard errors can be very large. Only exact separation is an error.
- **Panel structure beyond what the estimator needs.** A one-way FE model does not complain about unbalanced panels or about repeated (entity, time) pairs. Only the two-way models (FE, and RE with `time`) require a balanced panel.
- **Numeric precision.** Numeric columns are converted to 64-bit floating point without a warning, so integers beyond 2^53 and decimals with many digits can lose precision. See [Accepted data](accepted-data.md#how-values-are-converted).

## Where to look next

- [Errors](../api/errors.md): the reference for `ValidationError`, `ComputationError` and `ErrorCode`.
- [Error codes](error-codes.md): every `code`, its parameters and the English message and hint templates.
- [Accepted data](accepted-data.md): the input type, the column types accepted in each role, and the argument types.
- [Inference conventions](inference-conventions.md): the distributions and degrees of freedom behind the reported statistics.
- [Verification](verification.md): which reference implementations the results are compared with, including the error paths.
- The [API reference](../api/ols.md) for the options of each method.
