# Pickling and copying

Every options object and every result can be pickled and copied, so they travel through the usual pipelines: `multiprocessing`, `concurrent.futures`, `joblib`, on-disk caches and task queues such as Celery (with its pickle serializer enabled, see [Versions and trust](#versions-and-trust)). This page explains what survives a round trip, what a pickle costs, and what it does not promise.

## What can be pickled

- All `*Options` classes (`OLSOptions`, `LogitOptions`, `FEOptions`, ...).
- All results (`OLSResults`, `IVResults`, `FEResults`, ...), including the objects they return: the first-stage results of `IVResults.first_stage()` and `FEIVResults.first_stage()`, the tables of `marginal_effects()` and `censoring_fit_check()`, and the diagnostic test results.
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
# Continues the first example (`result`, `restored`).
print(restored.predict() == result.predict())  # True
print(restored.augment().equals(result.augment()))  # True
print(restored.white_test() == result.white_test())  # True
```

## Sending results between processes

**Prefer threads when the fits only have to run in parallel within one process.** The estimators release the Python GIL while they compute, so threads (`ThreadPoolExecutor`, `joblib`'s threading backend) run fits in parallel and do not need pickling (see [Using fits from threads](performance.md#using-fits-from-threads)). Use processes when the surrounding Python work is the bottleneck, or when the results have to leave the process (a cache, a queue).

The data and the options go to the worker, and the result comes back. Results are ordinary pickleable objects, so no conversion is needed. Each task pickles its DataFrame on the way in and, for the results that keep their training data, again on the way out; see [What a pickle contains](#what-a-pickle-contains) for the cost, and for sending only the numbers back. In the examples below, `frames` stands in for your real groups or samples.

With `ProcessPoolExecutor` (and `multiprocessing`), the worker function must be importable, that is, defined at the top level of a module (a script run with `python` counts). This is the usual rule for any pickled function, not one of this package. Keep the code that starts the pool under an `if __name__ == "__main__":` guard. With the `spawn` start method (the default on Windows and macOS) and the `forkserver` start method (the default on Linux from Python 3.14), the main script is imported again in a new process, and without the guard that import would try to start another pool and fail. With `fork` the guard is not required, but it does no harm, so use it in every script. Save the examples as scripts and run them with `python`.

<!-- doc-test: subprocess -->
```python
from concurrent.futures import ProcessPoolExecutor

import polars as pl
from econometricsmodels import OLS, OLSOptions


def fit_one(args):
    # `pool.map` passes one argument per task, so the data and the options come as a tuple.
    data, options = args
    return OLS(data, y="y", x=["x1", "x2"], options=options).fit()


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "y": [1.0, 2.2, 2.9, 4.1, 5.3, 5.9, 7.2, 7.8],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": [0.5, 0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4],
        }
    )
    # Stand-in for real groups or samples: three copies of `df` with `y` shifted by 0, 1 and 2.
    frames = [df.with_columns(pl.col("y") + shift) for shift in range(3)]
    options = OLSOptions(cov_type="hc1")
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(fit_one, [(frame, options) for frame in frames]))
    for result in results:
        print(round(result.params["x1"], 3))
```

`joblib` (install it separately, for example with `pip install joblib`) works the same way. Its default backend sends functions defined in the main script to the workers with `cloudpickle`, so they need not be importable:

<!-- doc-test: subprocess -->
```python
import polars as pl
from econometricsmodels import OLS, OLSOptions
from joblib import Parallel, delayed


def fit_one(data, options):
    return OLS(data, y="y", x=["x1", "x2"], options=options).fit()


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "y": [1.0, 2.2, 2.9, 4.1, 5.3, 5.9, 7.2, 7.8],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": [0.5, 0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4],
        }
    )
    # Stand-in for real groups or samples: three copies of `df` with `y` shifted by 0, 1 and 2.
    frames = [df.with_columns(pl.col("y") + shift) for shift in range(3)]
    options = OLSOptions(cov_type="hc1")
    results = Parallel(n_jobs=2)(
        delayed(fit_one)(frame, options) for frame in frames
    )
    for result in results:
        print(round(result.params["x1"], 3))
```

## What a pickle contains

A pickle contains everything the result needs. For `OLSResults`, `WLSResults`, `IVResults`, `LogitResults`, `ProbitResults` and `TobitResults`, which read the training data in methods such as `augment()` and the diagnostic tests, this includes **the `polars.DataFrame` that was passed to `fit()`** (all of its columns, also the ones the model does not use). That is what lets a restored result call `augment()` and the diagnostic tests; a result without the data raises `AUGMENT_REQUIRES_NEW_DATA` in `augment()` and `TRAINING_DATA_NOT_RETAINED` in the diagnostic tests (see [Validation and errors](validation.md) and [Error codes](error-codes.md)). `FEResults` and `REResults` do not keep the DataFrame: their pickles hold the estimates and the residuals. `FEIVResults` has no `augment()` or diagnostic tests, but it also keeps the DataFrame, because `fixed_effects()`, `cragg_donald()`, `kleibergen_paap()` and `effective_f()` estimate the model again from it. `FEGMMResults` keeps the DataFrame and the options passed to `fit()` as well, because `c_stat()` estimates the model again from them. `IVResults` also holds the first-stage results and the options passed to `fit()`, and reads the DataFrame in `cragg_donald()`, `kleibergen_paap()`, `effective_f()` and `c_stat()`.

The consequence is size: **for the results that keep the data, a pickle is at least as large as the DataFrame you fitted on.** For a DataFrame of three float columns with 100,000 rows (about 2.4 MB), the pickle of a `LogitResults` is about 2.4 MB (the DataFrame plus the coefficients), and that of an `OLSResults` about 4.0 MB, because it also holds the residuals and fitted values (two values per row). Logit, Probit and Tobit do not store their design matrix: it is rebuilt from the DataFrame when the pickle is restored, with the same column extraction as `fit()`, so restoring costs about as much as reading the columns again. The pickles of `FEResults` and `REResults` are usually smaller than the DataFrame, because they hold no copy of it. The time to pickle grows in proportion to the size.

When only the numbers are needed, take them out of the result before sending it, instead of sending the result:

```python
# Continues the first example (`result`, `pickle`).
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
- `FEIVResults` keeps the first-stage regression of every endogenous variable as well, and the order of `x_endog`, so the dictionaries it returns keep that order after a round trip.
- The results returned by `IVResults.first_stage()` keep no training data, so they cannot call `augment()` without `new_data` (`AUGMENT_REQUIRES_NEW_DATA`) or the diagnostic tests (`TRAINING_DATA_NOT_RETAINED`), before or after a round trip.

## `copy.copy` and `copy.deepcopy`

`copy.deepcopy` gives an independent object, including its own copy of the training DataFrame (for the results that keep it). `copy.copy` also makes a new result object, but the copy and the original share one training DataFrame, so it costs no extra memory for the data. A result never modifies its data, so sharing is safe. Use `deepcopy` when the copy has to be independent of the original.

## Versions and trust

A pickle is for moving results between processes and for short-lived caches, not for archiving them.

- **Use the same version of econometricsmodels** to write and to read a pickle. Each pickle carries a format version, which changes when the stored fields change. Restoring a pickle whose format version differs from the one of the installed package raises `ValidationError` with the code `PICKLE_FORMAT_MISMATCH`. Two package versions can share a format version, but the package does not promise it, and it does not convert old pickles. When a cache outlives an upgrade, treat that error as a cache miss and fit again.
- **The DataFrame inside is pickled by polars**, so the polars versions of the two sides must be compatible in the way polars itself requires. An incompatibility there is reported by polars, not by `PICKLE_FORMAT_MISMATCH`.
- **Only unpickle data you trust.** This is true of every `pickle`. The restore checks the structure and the types of the stored values, but not their numerical meaning: an edited pickle can hold coefficients that the estimator never produced, and the restored result reports them as it is told to.
- **Task queues such as Celery** accept only JSON messages by default (its task and result serializers have been JSON since Celery 4.0). To send a pickle you have to enable the pickle serializer explicitly, and that is safe only when nobody untrusted can reach the broker. Sending `result.params_table()` as JSON avoids the question.
- To keep a result for the long term, store the values you need (for example `result.params_table()` as JSON), together with the data and the options, and fit again when you need the object.

```python
# Continues the first example (`result`).
import json

print(json.dumps(result.params_table()[0]))
```

## Where to look next

- [Performance](performance.md#using-fits-from-threads): using fits from threads and running several fits at once.
- [Validation and errors](validation.md): the errors raised when a pickle cannot be restored.
- [Accepted data](accepted-data.md): the data a result keeps for `augment()` and the diagnostic tests.
