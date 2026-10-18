# Error codes

Every `ValidationError` and `ComputationError` raised by econometricsmodels carries a stable `code` and the values that appear in its text, so that applications can show their own message (for example in another language) instead of the English one.

```python
from econometricsmodels import ErrorCode, ValidationError

try:
    ...
except ValidationError as e:
    if e.code == ErrorCode.COLUMN_NOT_FOUND:
        print(e.params["column"])
```

An exception has these attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `code` | `str` | Stable identifier of the situation (UPPER_SNAKE_CASE). |
| `message` | `str` | What is wrong. |
| `hint` | `str` or `None` | What to do next. |
| `params` | `dict` | The values that appear in `message` and `hint`, keyed by name. |

`str(e)` is `message` and `hint` joined by a space.

The catalog below lists, for every `code`, the exception class, the keys and types of `params`, and the English templates of `message` and `hint` (`{name}` is replaced by the value of the parameter `name`).

## Rendering the templates yourself

To build a message in your own language, keep one template per `code` and replace each `{name}` with the value of `params["name"]` written as follows:

| Type of the value | Written as |
|---|---|
| `int`, `str` | As is. |
| `float` | Like Rust's debug format: `1.0`, `0.25`, `1e-7`, `NaN`, `inf`, `-inf`. |
| `float or None` | The same as `float`, or `None`. |
| `bool` | `True` or `False`. |
| list of `str` (option values) | Each item in double quotes, joined by `, `: `"a", "b"`. |
| list of `str` (column names) | Each item in single quotes, joined by `, `: `'a', 'b'`. |

Quotation marks around single values (such as `'{column}'`) are part of the template. The placeholder `{test_label}` is not a key of `params`: it is the name for `params["test"]` in the Test names table. Parameters marked "not shown in the text" are supplementary values for your own use and do not appear in any template.

## Catalog

<!-- BEGIN GENERATED CATALOG (tools/render_error_catalog.py) -->

### ValidationError

| Code | Parameters | Message | Hint |
|---|---|---|---|
| `COLUMN_NOT_FOUND` | `column` (str)<br>`argument` (str)<br>`data_argument` (str) | Column '{column}' (given as {argument}) does not exist in {data_argument}. | Check the spelling against {data_argument}.columns. |
| `COLUMN_CONTAINS_NULL` | `column` (str)<br>`argument` (str)<br>`data_argument` (str)<br>`count` (int) | Column '{column}' (given as {argument}) in {data_argument} contains {count} missing value(s) (null), and missing values are never dropped automatically. | Drop or impute the missing values first, for example with drop_nulls(). |
| `COLUMN_NON_FINITE` | `column` (str)<br>`argument` (str)<br>`data_argument` (str)<br>`row` (int)<br>`value` (float) | Column '{column}' (given as {argument}) in {data_argument} contains a non-finite value ({value}) at row {row} (0-based), and non-finite values are never dropped automatically. | Remove or replace the NaN and infinite values first. |
| `COLUMN_UNSUPPORTED_DTYPE` | `column` (str)<br>`argument` (str)<br>`data_argument` (str)<br>`dtype` (str)<br>`allowed` (list of str) | Column '{column}' (given as {argument}) in {data_argument} has dtype {dtype}, which cannot be used for {argument}. | Use a column with one of these dtypes: {allowed}, or cast the column first. |
| `COLUMN_TIMEZONE_NOT_SUPPORTED` | `column` (str)<br>`argument` (str)<br>`data_argument` (str)<br>`time_zone` (str) | Column '{column}' (given as {argument}) in {data_argument} is a Datetime with time zone '{time_zone}', which cannot be used for {argument}. | Remove the time zone first, for example with .dt.replace_time_zone(None) after converting to the zone you want to keep. |
| `COLUMN_VALUES_TIED` | `column` (str)<br>`argument` (str)<br>`data_argument` (str)<br>`first_row` (int)<br>`second_row` (int) | Column '{column}' (given as {argument}) in {data_argument} has the same value at rows {first_row} and {second_row} (0-based), so the observations cannot be put in time order. | Make the values distinct, or add an index column such as df.with_row_index("t") if the rows are already in time order. |
| `ROLE_EMPTY` | `argument` (str) | {argument} must contain at least one column name. | Pass a non-empty list, for example {argument}=["x1"]. |
| `ROLE_DUPLICATE_COLUMN` | `column` (str)<br>`argument` (str) | Column '{column}' is listed more than once in {argument}. | List each column once. |
| `ROLE_COLUMN_OVERLAP` | `column` (str)<br>`argument` (str)<br>`other_argument` (str) | Column '{column}' is used both as {argument} and as {other_argument}. | Use a different column for each role, or remove the column from one of them. |
| `INTERCEPT_NAME_COLLISION` | `argument` (str) | {argument} contains a column named "const", which collides with the intercept that is added automatically. | Rename the column, or set include_intercept=False if it already is your constant. |
| `TOBIT_SIGMA_NAME_COLLISION` | - | x contains a column named "sigma", which collides with the error standard deviation that the result reports as "sigma". | Rename the column. |
| `AUGMENT_COLUMN_EXISTS` | `column` (str) | The data already has a column named '{column}', and augment() would overwrite it. | Rename or drop that column before calling augment(). |
| `OPTION_UNKNOWN_VALUE` | `option` (str)<br>`value` (str)<br>`allowed` (list of str) | Unknown value "{value}" for {option}. | Use one of: {allowed}. |
| `OPTION_UNUSED` | `option` (str)<br>`condition` (str) | {option} is only used when {condition}, so it would be silently ignored. | Set {condition}, or remove {option}. |
| `HAC_TIME_REQUIRED` | `setting` (str) | {setting}="hac" requires the hac_time option, the column that gives the time order of the observations; the row order of the data is never assumed to be the time order. | Pass hac_time, using an index column such as df.with_row_index("t") if the rows are already in time order. |
| `DK_TIME_REQUIRED` | - | cov_type="dk" requires the dk_time option, the column that defines the time periods of the Driscoll-Kraay estimator; it is not taken from the time option, which only sets the two-way effects. | Pass dk_time explicitly. |
| `CLUSTER_REQUIRED` | `setting` (str) | {setting}="cluster" requires the cluster option, the column that identifies the clusters. | Pass the column name, for example cluster="firm_id". |
| `CONFIDENCE_LEVEL_OUT_OF_RANGE` | `value` (float) | confidence_level must be strictly between 0 and 1, got {value}. | Use a value such as 0.95. |
| `TOLERANCE_NOT_POSITIVE_FINITE` | `option` (str)<br>`value` (float) | {option} must be a positive finite number, got {value}. | Omit the option to use the default, or pass a small positive number. |
| `MAX_ITER_OUT_OF_RANGE` | `value` (int)<br>`min` (int)<br>`max` (int) | max_iter must be an integer from {min} to {max}, got {value}. | Pass a value from {min} to {max}. |
| `GMM_MAX_ITER_OUT_OF_RANGE` | `value` (int)<br>`min` (int)<br>`max` (int) | gmm_max_iter must be an integer from {min} to {max} (the initial estimate counts as one iteration), got {value}. | Pass a value from {min} to {max}, or use gmm_type="two_step" for a two-step estimate without iteration. |
| `HAC_LAGS_OUT_OF_RANGE` | `value` (int)<br>`n_obs` (int)<br>`max` (int) | hac_lags must be from 0 to {max}, one less than the number of observations ({n_obs}), got {value}. | Pass a smaller hac_lags, or omit it to use the default. |
| `DK_BANDWIDTH_OUT_OF_RANGE` | `value` (int)<br>`n_periods` (int)<br>`max` (int) | dk_bandwidth must be from 0 to {max}, one less than the number of unique time periods in dk_time ({n_periods}), got {value}. | Pass a smaller dk_bandwidth, or omit it to use the default. |
| `NLAGS_NOT_POSITIVE` | `value` (int) | nlags must be a positive integer, got {value}. | Pass nlags of 1 or more. |
| `COV_TYPE_NOT_AVAILABLE_FOR_MODEL` | `value` (str)<br>`model` (str)<br>`allowed` (list of str) | cov_type="{value}" is not available for {model}. | Use one of: {allowed}. |
| `INSUFFICIENT_OBSERVATIONS` | `n_obs` (int)<br>`n_params` (int)<br>`min_obs` (int)<br>`stage` (optional, see Stages) | Not enough observations: {n_obs} observations for {n_params} coefficients (including the intercept); at least {min_obs} are needed. | Add observations or remove regressors. |
| `CLUSTER_COUNT_TOO_SMALL` | `setting` (str)<br>`n_clusters` (int)<br>`min_clusters` (int)<br>`stage` (optional, see Stages) | {setting}="cluster" needs at least {min_clusters} clusters, but the cluster column has {n_clusters}. | Use a cluster column with more distinct values, or set a different {setting}. |
| `CLUSTER_COUNT_NOT_ABOVE_SLOPES` | `setting` (str)<br>`n_clusters` (int)<br>`n_slopes` (int)<br>`min_clusters` (int)<br>`stage` (optional, see Stages) | {setting}="cluster" needs more clusters than slope coefficients for the overall F/Wald test: the cluster column has {n_clusters} clusters for {n_slopes} slope coefficient(s); at least {min_clusters} are needed. | Use a cluster column with more distinct values, remove regressors, or set a different {setting}. |
| `WEIGHT_NOT_POSITIVE` | `row` (int)<br>`value` (float) | The weight at row {row} (0-based) must be positive, got {value}. | Drop or correct the rows with zero or negative weights. |
| `INSUFFICIENT_OBSERVATIONS_FOR_DIAGNOSTIC` | `test` (str)<br>`n_obs` (int)<br>`n_params` (int)<br>`min_obs` (int) | The {test_label} needs at least {min_obs} observations for its auxiliary regression ({n_params} coefficients), but there are {n_obs}. | Add observations, or reduce the number of variables or lags. |
| `TRAINING_DATA_NOT_RETAINED` | `method` (str) | {method}() needs the original training data, which this result does not keep (results returned by IVResults.first_stage() keep no training data). | Call it on a result you fitted directly with OLS(...).fit(). |
| `AUGMENT_REQUIRES_NEW_DATA` | - | augment() without new_data needs the original training data, which this result does not keep. | Pass new_data. |
| `OUTCOME_NOT_BINARY` | `row` (int)<br>`value` (float) | The outcome y at row {row} (0-based) must be 0 or 1, got {value}. | Code the outcome as 0 and 1, for example by casting a boolean column to an integer. |
| `CENSORING_BOUNDS_MISSING` | - | Both censoring bounds are None. | Set lower, upper, or both. |
| `CENSORING_BOUNDS_NOT_ORDERED` | `lower` (float)<br>`upper` (float) | lower ({lower}) must be less than upper ({upper}). | Swap or correct the bounds. |
| `CENSORING_BOUND_NOT_FINITE` | `lower` (float or None)<br>`upper` (float or None) | The censoring bounds must be finite numbers or None, got lower={lower}, upper={upper}. | Pass finite numbers, or None for a side that is not censored. |
| `OUTCOME_OUTSIDE_CENSORING_BOUNDS` | `row` (int)<br>`value` (float)<br>`lower` (float or None)<br>`upper` (float or None) | The outcome y at row {row} (0-based) is {value}, outside the censoring bounds (lower={lower}, upper={upper}). | Correct the outcome, or set lower and upper to the actual censoring points. |
| `NO_UNCENSORED_OBSERVATIONS` | `lower` (float or None)<br>`upper` (float or None) | No observation lies strictly between the censoring bounds (lower={lower}, upper={upper}), so the model cannot be identified. | Check the bounds: at least one y must be strictly between them. |
| `INSUFFICIENT_INSTRUMENTS` | `n_instruments` (int)<br>`n_endog` (int) | The model has {n_endog} endogenous regressor(s) but only {n_instruments} instrument(s); it needs at least as many instruments as endogenous regressors (order condition). | Add instruments, or reduce the number of endogenous regressors. |
| `INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE` | `n_obs` (int)<br>`n_params` (int)<br>`min_obs` (int) | The first stage needs more observations than its {n_params} coefficients (instruments, exogenous regressors and the intercept): {n_obs} observations, at least {min_obs} needed. | Add observations, or remove instruments or exogenous regressors. |
| `GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL` | `n_clusters` (int)<br>`n_instruments` (int)<br>`min_clusters` (int) | gmm_weight_type="cluster" needs at least {min_clusters} clusters for {n_instruments} instruments (including exogenous regressors), but the cluster column has {n_clusters}. | Use a cluster column with more distinct values, or set a different gmm_weight_type. |
| `INSUFFICIENT_DEGREES_OF_FREEDOM` | `n_obs` (int)<br>`n_params` (int)<br>`n_effects` (int)<br>`min_obs` (int)<br>`stage` (optional, see Stages) | Not enough observations for the panel model: {n_obs} observations, but {n_params} coefficients and {n_effects} fixed-effect parameters (entity effects, plus time effects in a two-way model) must be estimated; at least {min_obs} are needed. | Add observations, or remove regressors. |
| `SINGLETON_GROUP` | `argument` (str)<br>`group` (str) | The {argument} group '{group}' has only one observation, and singleton groups are not dropped automatically. | Remove its rows from the input. |
| `PANEL_NOT_BALANCED_FOR_TWO_WAY` | `n_obs` (int)<br>`n_entities` (int)<br>`n_periods` (int)<br>`expected_obs` (int) | Two-way estimation requires a balanced panel: {n_obs} observations for {n_entities} entities and {n_periods} periods, but {expected_obs} are expected. | Make every entity observed in every period, or estimate one-way effects by omitting the time option. |
| `REGRESSOR_NO_WITHIN_VARIATION` | `column` (str) | Regressor '{column}' has no variation left after the within transformation: it is time-invariant, or collinear with the fixed effects. | Remove the regressor from x. |
| `DK_PERIODS_TOO_FEW` | `n_periods` (int)<br>`min_periods` (int) | cov_type="dk" needs at least {min_periods} unique time periods in dk_time, but there are {n_periods}. | Use a dk_time column with more distinct periods, or set a different cov_type. |
| `DK_PERIODS_NOT_ABOVE_COEFFICIENTS` | `n_periods` (int)<br>`n_coefficients` (int)<br>`min_periods` (int)<br>`stage` (optional, see Stages) | cov_type="dk" needs more time periods than jointly tested coefficients: dk_time has {n_periods} periods for {n_coefficients} coefficient(s); at least {min_periods} are needed. | Use a dk_time column with more distinct periods, remove regressors, or set a different cov_type. |
| `DK_TWO_PERIODS_DEGENERATE` | - | With 2 time periods, every entity (and with two-way effects every period) is observed exactly once in each, so the within transformation cancels the two per-period scores and the Driscoll-Kraay covariance is zero. | Use more time periods or another cov_type. |
| `CLUSTER_TWO_GROUPS_DEGENERATE` | - | With 2 clusters, every entity (and with two-way effects every period) is observed exactly once in each, so the within transformation cancels the two cluster scores and the cluster-robust covariance is zero. | Use more clusters or another cov_type. |
| `AMEMIYA_REQUIRES_WITHIN_VARIATION` | `columns` (list of str (column names)) | variance_method="amemiya" needs every regressor to vary within the effects, but these do not: {columns}. | Use variance_method="swamy_arora" or "wallace_hussain", which accept such regressors. |

### ComputationError

| Code | Parameters | Message | Hint |
|---|---|---|---|
| `DESIGN_MATRIX_SINGULAR` | `stage` (optional, see Stages) | The design matrix is singular: some columns are perfectly collinear. | Remove or combine the collinear columns, for example duplicated columns or a constant column together with the intercept. |
| `MATRIX_NEAR_SINGULAR` | `matrix` (str, not shown in the text)<br>`stage` (optional, see Stages) | A matrix needed to compute the estimates or their covariance is numerically singular. | Rescale the variables so that they are on similar scales, or remove nearly collinear regressors. |
| `NOT_CONVERGED` | `n_iter` (int)<br>`iter_option` (str) | The optimizer did not converge after {n_iter} iterations. | Increase {iter_option}, or pass raise_on_non_convergence=False to receive the last iteration (check converged before using it). |
| `LINE_SEARCH_FAILED` | `n_evaluations` (int, not shown in the text) | The optimizer's line search did not converge. | Try a different solver, or rescale the regressors. |
| `SEPARATION_DETECTED` | - | The outcome is perfectly predicted by some combination of the regressors (complete or quasi-complete separation), so no finite maximum likelihood estimate exists. | Remove or combine the offending regressors, or add observations that break the separation. |
| `SEPARATION_CHECK_FAILED` | - | The check for separation in the data could not be completed. | Rescale the regressors so that they are on similar scales, or remove nearly collinear regressors. |
| `UNCENSORED_INTERVAL_TOO_NARROW` | `lower` (float)<br>`upper` (float)<br>`sigma` (float) | The uncensored interval [{lower}, {upper}] is narrower than 1% of the estimated sigma ({sigma}), so the expected value among uncensored observations cannot be computed reliably. | Use another target. |
| `INTERNAL_ERROR` | `detail` (str) | An internal error occurred: {detail}. This should not happen. | Report it as a bug, with the data or a minimal example that reproduces it. |
| `DIAGNOSTIC_VARIABLES_CONSTANT` | `test` (str) | The {test_label} cannot be computed: all of its variables are constant. | Use variables that vary across observations. |
| `DIAGNOSTIC_VARIABLE_DEGENERATE` | `test` (str) | The {test_label} cannot be computed: a variable has zero or non-finite variance after scaling. | Rescale the variables so that they are on similar scales. |
| `DIAGNOSTIC_STATISTIC_UNDEFINED` | `test` (str) | The {test_label} statistic is undefined: the auxiliary regression is degenerate (zero residuals, a perfect fit or non-finite values). | Check the model for a perfect fit or non-finite values. |
| `BREUSCH_GODFREY_MATRIX_TOO_LARGE` | `n_obs` (int)<br>`n_columns` (int) | The auxiliary regression matrix of the Breusch-Godfrey test ({n_obs} rows by {n_columns} columns) is too large to allocate. | Use a smaller nlags. |
| `NON_POSITIVE_IDIOSYNCRATIC_VARIANCE` | `sigma2_eps` (float)<br>`variance_method` (str) | The estimated idiosyncratic variance is not positive ({sigma2_eps}) with variance_method="{variance_method}", so the random-effects weights are undefined. | Try a different variance_method. |
| `VARIANCE_COMPONENTS_MOMENTS_SINGULAR` | `variance_method` (str) | The moment equations that estimate the variance components are singular with variance_method="{variance_method}". | Try a different variance_method. |

### Stages

When an error happens inside a nested regression, `params` also holds `stage` (one of the identifiers below, plus the extra parameters listed) and the message starts with the phrase shown.

| Stage | Extra parameters | Phrase |
|---|---|---|
| `FIRST_STAGE` | `endog_name` | In the first stage regression for '{endog_name}' |
| `SECOND_STAGE` | - | In the second stage regression |
| `WU_HAUSMAN_REGRESSION` | - | In the Wu-Hausman test regression |
| `WITHIN_REGRESSION` | - | In the within-transformed regression |
| `OVERALL_F_TEST` | - | In the overall F test of the slope coefficients |
| `BETWEEN_REGRESSION_ENTITY` | - | In the between regression on entity means used to estimate a variance component |
| `BETWEEN_REGRESSION_TIME` | - | In the between regression on time means used to estimate a variance component |
| `POOLED_REGRESSION` | - | In the pooled regression used to estimate a variance component |
| `QUASI_DEMEANED_REGRESSION` | - | In the quasi-demeaned regression |
| `HAUSMAN_REGRESSION` | - | In the Hausman test auxiliary regression |
| `DIAGNOSTIC_AUXILIARY_REGRESSION` | `test` | In the auxiliary regression of the {test_label} |

### Test names

| `test` | Name used in the message |
|---|---|
| `WHITE` | White test |
| `BREUSCH_PAGAN` | Breusch-Pagan test |
| `BREUSCH_GODFREY` | Breusch-Godfrey test |

<!-- END GENERATED CATALOG -->
