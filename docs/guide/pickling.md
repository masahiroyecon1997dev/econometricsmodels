# Pickling and copying

Every options object and every result can be pickled and copied, so they travel through the usual pipelines: `multiprocessing`, `concurrent.futures`, `joblib`, on-disk caches and task queues such as Celery. This page explains what survives a round trip, what a pickle costs, and what it does not promise.

## What can be pickled

- All `*Options` classes (`OLSOptions`, `LogitOptions`, `FEOptions`, ...).
- All results (`OLSResults`, `IVResults`, `FEResults`, ...), including the objects they return: the first-stage results of `IVResults.first_stage()`, the tables of `marginal_effects()` and `censoring_fit_check()`, and the diagnostic test results.
- `ValidationError` and `ComputationError`, with their `code`, `params`, `message` and `hint`.

`pickle.dumps`, `copy.copy` and `copy.deepcopy` work with every protocol.

```python
import copy
import pickle

import polars as pl
from econometricsmodels import OLS, OLSOptions

df = pl.DataFrame(
    {
        "y": [1.0, 2.2, 2.9, 4.1, 5.3, 5.9, 7.2, 7.8],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "x2": [0.5, 0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4],
    }
)
result = OLS(df, y="y", x=["x1", "x2"], options=OLSOptions(cov_type="hc1")).fit()

restored = pickle.loads(pickle.dumps(result))
print(restored.params == result.params)  # True
print(copy.deepcopy(result).std_errors == result.std_errors)  # True
```

A restored result behaves like the original. Every property returns the same value, and the methods that read the training data (`predict()`, `augment()`, `marginal_effects()`, `pred_table()` and the post-estimation diagnostic tests) give the same answer, because the pickle contains the data that was passed to `fit()` (see [What a pickle contains](#what-a-pickle-contains)).

```python
print(restored.predict() == result.predict())  # True
print(restored.augment().equals(result.augment()))  # True
print(restored.white_test() == result.white_test())  # True
```

## Sending results between processes

Options go to the worker, the result comes back. Results are ordinary pickleable objects, so no conversion is needed. The worker function must be importable (defined at the top level of a module), as for any pickled function; this is the usual `multiprocessing` rule, not one of this package.

```py
from concurrent.futures import ProcessPoolExecutor

import polars as pl
from econometricsmodels import OLS, OLSOptions


def fit_one(args):
    data, options = args
    return OLS(data, y="y", x=["x1", "x2"], options=options).fit()


if __name__ == "__main__":
    # `frames` is a list of polars DataFrames, one per group or sample.
    options = OLSOptions(cov_type="hc1")
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(fit_one, [(frame, options) for frame in frames]))
```

`joblib` works the same way:

```py
from joblib import Parallel, delayed

results = Parallel(n_jobs=4)(delayed(fit_one)((frame, options)) for frame in frames)
```

The estimators release the Python GIL while they compute, so threads (`ThreadPoolExecutor`, `joblib`'s threading backend) also run fits in parallel and need no pickling at all. Use processes when the surrounding Python work is the bottleneck, or when the results have to leave the process (a cache, a queue).

## What a pickle contains

A pickle contains everything the result needs, **including the `polars.DataFrame` that was passed to `fit()`** (all of its columns, also the ones the model does not use). That is what lets a restored result call `augment()` and the diagnostic tests; a result without the data would raise `TRAINING_DATA_NOT_RETAINED` there.

The consequence is size: **a pickle is at least as large as the DataFrame you fitted on.** For a DataFrame of three float columns with 100,000 rows (4.0 MB), the pickle of a `LogitResult` is about 4.0 MB (the DataFrame plus the coefficients), and that of an `OLSResults` about 5.6 MB, because it also holds the residuals and fitted values (two values per row). Logit, Probit and Tobit do not store their design matrix: it is rebuilt from the DataFrame when the pickle is restored, with the same column extraction as `fit()`, so restoring costs about as much as reading the columns again. The time to pickle grows in proportion to the size.

When only the numbers are needed, take them out of the result before sending it, instead of sending the result:

```python
summary = {
    "params": result.params,
    "std_errors": result.std_errors,
    "r_squared": result.r_squared,
    "table": result.params_table(),
}
print(len(pickle.dumps(summary)) < len(pickle.dumps(result)))  # True
```

`params_table()` is a list of plain dictionaries, and `params`, `std_errors`, `conf_int` and the other coefficient fields are plain dictionaries keyed by the column name, so they pickle (and convert to JSON) without this package.

Some results cost a little more than others:

- `FEResults` computes the fixed effects when it is pickled or copied, if `fixed_effects()` has not been called yet (the work grows with the number of entities). `fit()` itself is not slowed down.
- `IVResults` keeps the first-stage regression of every endogenous variable, so its pickle holds one result per endogenous variable.
- The results returned by `IVResults.first_stage()` keep no training data, so they cannot call `augment()` or the diagnostic tests, before or after a round trip (`TRAINING_DATA_NOT_RETAINED`).

## `copy.copy` and `copy.deepcopy`

`copy.deepcopy` gives an independent object, including its own copy of the training DataFrame. `copy.copy` also makes a new result object, but the copy and the original share one training DataFrame, so it costs no extra memory for the data. A result never modifies its data, so sharing is safe. Use `deepcopy` when the copy has to be independent of the original.

## Versions and trust

A pickle is for moving results between processes and for short-lived caches, not for archiving them.

- **The version of econometricsmodels must be the same** when the pickle is written and read. Each pickle carries a format version, and restoring one that does not match (an older or newer package, or a changed set of fields) raises `ValidationError` with the code `PICKLE_FORMAT_MISMATCH`. The package does not convert old pickles. When a cache outlives an upgrade, treat that error as a cache miss and fit again.
- **The DataFrame inside is pickled by polars**, so the polars versions of the two sides must be compatible in the way polars itself requires. An incompatibility there is reported by polars, not by `PICKLE_FORMAT_MISMATCH`.
- **Only unpickle data you trust.** This is true of every `pickle`. The restore checks the structure and the types of the stored values, but not their numerical meaning: an edited pickle can hold coefficients that the estimator never produced, and the restored result reports them as it is told to.
- To keep a result for the long term, store the values you need (for example `result.params_table()` as JSON), together with the data and the options, and fit again when you need the object.

```python
import json

print(json.dumps(result.params_table())[:60])
```
