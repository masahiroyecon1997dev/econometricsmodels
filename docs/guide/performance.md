# Performance

The computational core is written in Rust. This page summarizes how its execution time and memory use compare with the reference implementations the results are verified against ([statsmodels](https://www.statsmodels.org/) and [linearmodels](https://bashtage.github.io/linearmodels/)), what the numbers do and do not tell you, and the known performance limitations. The full per-method tables are on [Performance results](performance-results.md).

## Summary

The first table shows each method with its default standard error (classical) at the largest sample size at which every library was measured (`k = 5` regressors). "Speed-up" is the reference time divided by the econometricsmodels time; a value below 1 means the reference was faster. Rows can differ in `n` because the reference implementation is not timed at the largest sizes for every method (for example Probit stops at 100,000).

The second table covers every standard-error type of every method at one sample size, so that no covariance path is left out of the comparison.

<!-- BEGIN GENERATED SUMMARY (performance.render_docs_results) -->

| Method | Standard error | n | econometricsmodels | Reference | Speed-up | Peak RSS (econometricsmodels / reference) |
|---|---|---|---|---|---|---|
| OLS | classical | 1,000,000 | 0.0583s vs 0.1468s | statsmodels | 2.5x | 324.1MB / 572.7MB |
| WLS | classical | 1,000,000 | 0.1070s vs 0.2252s | statsmodels | 2.1x | 340.8MB / 614.3MB |
| Logit | classical | 1,000,000 | 0.4901s vs 0.5017s | statsmodels | 1.0x | 411.9MB / 512.2MB |
| Probit | classical | 100,000 | 0.1478s vs 0.1970s | statsmodels | 1.3x | 177.3MB / 265.9MB |
| Tobit | classical | 1,000,000 | 2.1555s | - | - | 395.8MB |
| IV | classical | 1,000,000 | 0.4836s vs 12.3919s | linearmodels | 25.6x | 1117.5MB / 3988.7MB |
| FE | classical | 1,000,000 | 0.2969s vs 3.8888s | linearmodels | 13.1x | 547.4MB / 957.6MB |
| RE | classical | 1,000,000 | 0.3074s vs 2.9774s | linearmodels | 9.7x | 945.6MB / 1277.5MB |

**Speed-up by standard-error type**

Every standard-error type is measured at a single sample size (`n = 100,000`, `k = 5`). Each cell is the reference time divided by the econometricsmodels time (below 1: the reference was faster). `cluster` is the lowest value over the cluster cases (50 clusters or `n / 10` clusters, with the rows of a cluster scattered or contiguous); FE and RE are clustered by entity only. `engine only` marks standard errors that the reference implementation does not offer, and `-` those the method does not support. Times and peak memory for every case are on [Performance results](performance-results.md).

| Method | classical | hc0 | hc1 | hc2 | hc3 | opg | hac | dk | cluster |
|---|---|---|---|---|---|---|---|---|---|
| OLS | 2.6x | 2.2x | 2.0x | 1.8x | 2.0x | - | 2.7x | - | 1.8x |
| WLS | 2.2x | 1.9x | 1.8x | 2.0x | 1.9x | - | 2.5x | - | 1.6x |
| Logit | 0.85x | 0.86x | engine only | - | - | engine only | - | - | 0.89x |
| Probit | 1.4x | 1.3x | engine only | - | - | engine only | - | - | 1.3x |
| Tobit | engine only | engine only | engine only | - | - | engine only | - | - | engine only |
| IV | 24.2x | 22.3x | 20.8x | engine only | engine only | - | 13.7x | - | 18.1x |
| FE | 14.0x | - | 13.3x | engine only | engine only | - | - | 12.3x | 17.4x |
| RE | 10.4x | - | 9.9x | engine only | engine only | - | - | 9.8x | 11.0x |

<!-- END GENERATED SUMMARY -->

## How to read these numbers

- **Relative, not absolute.** The numbers come from the *Benchmark (performance)* workflow on shared GitHub Actions runners, so absolute times vary from run to run, often by tens of percent (the statsmodels Logit time at 1,000,000 observations differed by more than a factor of two between two CI runs). Compare the two columns within a row, not the absolute values across pages or releases. They are refreshed at each release.
- **Single-threaded.** The linear-algebra backends of both sides are pinned to one thread. This isolates the efficiency of the computation itself from thread-pool behavior. It does not favor this package: with a small number of regressors and many observations, which is the typical econometric shape, multi-threaded BLAS made the reference implementation *slower*, and with a few dozen regressors it was only slightly faster, so the one-thread setting shows the reference at its best. Speed on many cores for very wide models is a different question that these tables do not answer.
- **Like for like.** The package computes the full set of reported statistics (fit statistics, tests, information criteria) in every call, while the reference packages compute many of them lazily. The benchmark touches those lazy results so both sides do the same work. Converting a polars frame to pandas for the reference packages is excluded from the timing, since it is not part of their estimation.
- **End to end.** Times include the Python call, the Arrow hand-over and the conversion of results, as a user would experience them.
- **Peak RSS** is the peak resident memory of the whole process, including the Python interpreter and imported libraries. At small `n` it mostly reflects that baseline rather than the estimator.
- **Release build.** The extension is built in release mode. A debug build can be more than ten times slower.
- **Scope of the sweeps.** Each method is measured over sample size (`k = 5`) and number of regressors (`n = 10,000`) for the default standard error, and over every standard-error type it supports at one sample size (`n = 100,000`, `k = 5`). Clustered errors are measured for 50 clusters and for `n / 10` clusters, with the rows of each cluster either scattered through the data or contiguous, because the cost of the cluster computation depends on both. Solver, estimator and effects variants are measured at one representative point; for IV, two GMM variants are part of the standard-error sweep (`gmm_cluster_weight`: two-step GMM with a cluster weight matrix and cluster-robust errors; `gmm_iterated`: iterated GMM with a heteroskedasticity-robust weight matrix). This is meant to show relative trends and to make a slowdown in any covariance path visible, not to cover every option combination.

## Known limitations

Performance is a goal rather than a guarantee. The points below are the ones worth knowing about.

- **Driscoll–Kraay for FE and RE is measured with `k = 5` only.** The Driscoll–Kraay covariance has rank at most `T - 1` (the number of time periods minus one), so a joint test of more coefficients than that is not possible. With six periods and `k = 20`, `fit()` rejects `cov_type="dk"` with a `ValidationError`: for FE because of the slope F-test, and for RE because of the Hausman test, which is always included. The standard-error sweep therefore uses `k = 5`, and the regressor sweep uses the classical covariance.
- **RE: this package does more work than the reference.** `RE.fit()` always computes the Hausman test (which includes an internal fixed-effects fit), while linearmodels' `RandomEffects` has no equivalent, so the RE speed-up understates the difference in efficiency.
- **Some standard errors are timed for this package alone.** Where a reference implementation has no equivalent, the row shows only the econometricsmodels time and the speed-up matrix says "engine only": HC2 and HC3 for IV, FE and RE (linearmodels has no equivalent), and the outer-product-of-gradients (`opg`) and `hc1` covariances for Logit and Probit (statsmodels does not offer `opg` for these models, and its `hc1` falls back to `hc0`). Cluster-robust errors for FE and RE are clustered by entity only. For FE and RE, the `cluster` and `dk` covariances use the small-sample corrections and degrees of freedom of fixest/Stata, which differ from linearmodels by a scalar factor (a few parts per thousand for `cluster`, about 10% for `dk` with six periods); the amount of computation is the same, so the rows are compared, but the standard errors are not identical. `classical` and `hc1` agree to machine precision.
- **Logit, Probit and Tobit are on par with statsmodels rather than far ahead.** They are iterative maximum-likelihood fits, so at the largest sizes measured the ratio to statsmodels ranges from about 1.0x (Logit at 1,000,000 observations, a tie) to 1.3x (Probit at 100,000). Logit at 100,000 observations was slightly behind statsmodels for every standard-error type in the latest run (0.85x to 0.91x), while it was ahead at 10,000 (about 2x) and tied at 1,000,000. Part of the difference is a check this package always runs after convergence: it decides exactly whether the data are separated (complete or quasi-complete separation), which adds about 11% to a Newton fit and which statsmodels does not do. The advantage also shrinks as the number of regressors grows: for Logit with the classical covariance it falls from about 2.1x at `k = 5` to 1.1x at `k = 20`. The default Newton solver is the fastest or tied for fastest; `bfgs` and `lbfgs` are about equal or slower.
- **Tobit has no reference timing.** There is no suitable Python Tobit implementation to time in the same process (statsmodels has none), so the Tobit rows show this package alone (absolute time and scaling with `n` and `k`). Its correctness is checked against R, see [Verification](verification.md).
- **Probit and Tobit at the largest sizes are measured for this package only.** The reference implementation is not timed at `n = 200,000` and `n = 1,000,000` for Probit, so the largest comparison for Probit is at 100,000 observations.
- **Instrumental variables: part of the gap is symmetric work.** linearmodels computes its first-stage diagnostics by re-fitting each first stage, and the benchmark includes that, because this package always reports the weak-instrument statistics from the same pass. In an earlier local measurement the 2SLS gap was still about threefold with that re-fit excluded. GMM with a kernel (HAC) weight matrix is not benchmarked, because linearmodels takes tens of seconds even at 100,000 observations.

## Using fits from threads

`fit()` releases the Python global interpreter lock (GIL) while it estimates, for every method. Other Python threads keep running during a long fit, so a web server or a GUI is not blocked, and fits running in separate threads run in parallel.

- The linear algebra of each fit runs on a single thread. Speed-ups on many cores come from running several fits at once, for example a thread pool over bootstrap samples, specifications or groups.
- Each running fit holds its own copy of the numeric columns it uses, so peak memory grows with the number of fits in flight.
- A running fit cannot be interrupted with `KeyboardInterrupt`; the interrupt is raised when the fit returns.
- Reading the data from the `polars.DataFrame` happens before the lock is released. The diagnostic tests on OLS results (`white_test()`, `breusch_pagan_test()`, `breusch_godfrey_test()`) and `marginal_effects()` on Logit, Probit and Tobit results also release the lock while they compute. The other methods on a results object (`predict()`, `augment()` and so on) are cheap and still hold it.

## Reproducing

The measurement scripts live in the [`performance/`](https://github.com/masahiroyecon1997dev/econometricsmodels/tree/main/performance) directory of the repository; `performance/compare_<method>.py` runs one method and the shared harness takes care of process isolation, thread pinning and the symmetric measurement. Build the extension with `maturin develop --release` first.
