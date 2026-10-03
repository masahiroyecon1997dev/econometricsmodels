# Performance

The computational core is written in Rust. This page summarizes how its execution time and memory use compare with the reference implementations the results are verified against ([statsmodels](https://www.statsmodels.org/) and [linearmodels](https://bashtage.github.io/linearmodels/)), what the numbers do and do not tell you, and the known performance limitations. The full per-method tables are on [Performance results](performance-results.md).

## Summary

Each row is the largest sample size at which every library was measured (`k = 5` regressors). "Speed-up" is the reference time divided by the econometricsmodels time; a value below 1 means the reference was faster. Rows can differ in `n` because the largest sizes are measured for the default standard error only (for example Tobit: classical at 1,000,000, cluster at 100,000).

<!-- BEGIN GENERATED SUMMARY (performance.render_docs_results) -->

| Method | cov_type | n | econometricsmodels | Reference | Speed-up | Peak RSS (econometricsmodels / reference) |
|---|---|---|---|---|---|---|
| OLS | classical | 1,000,000 | 0.0586s vs 0.1669s | statsmodels | 2.8x | 299.7MB / 544.5MB |
| OLS | hac | 1,000,000 | 0.1645s vs 0.7258s | statsmodels | 4.4x | 345.5MB / 630.2MB |
| WLS | classical | 1,000,000 | 0.0639s vs 0.1466s | statsmodels | 2.3x | 315.1MB / 590.1MB |
| WLS | hac | 1,000,000 | 0.1567s vs 0.6351s | statsmodels | 4.1x | 360.7MB / 645.5MB |
| Logit | classical | 1,000,000 | 0.7391s vs 1.3950s | statsmodels | 1.9x | 413.7MB / 511.5MB |
| Logit | cluster | 1,000,000 | 1.0160s vs 1.4890s | statsmodels | 1.5x | 454.5MB / 511.0MB |
| Probit | classical | 100,000 | 0.1146s vs 0.1379s | statsmodels | 1.2x | 172.8MB / 271.2MB |
| Probit | cluster | 100,000 | 0.1499s vs 0.1451s | statsmodels | 0.97x | 186.2MB / 271.3MB |
| Tobit | classical | 1,000,000 | 2.1595s | - | - | 390.9MB |
| Tobit | cluster | 100,000 | 0.2529s | - | - | 188.7MB |
| IV | classical | 1,000,000 | 0.5216s vs 14.6751s | linearmodels | 28.1x | 1093.9MB / 3952.5MB |
| IV | hac | 1,000,000 | 1.1675s vs 15.3787s | linearmodels | 13.2x | 1154.8MB / 3952.9MB |
| FE | classical | 1,000,000 | 1.6362s vs 3.5529s | linearmodels | 2.2x | 572.7MB / 945.9MB |
| FE | dk | 1,000,000 | 1.6675s vs 3.5605s | linearmodels | 2.1x | 634.8MB / 948.0MB |
| RE | classical | 1,000,000 | 6.3024s vs 5.2547s | linearmodels | 0.83x | 1012.6MB / 1258.4MB |
| RE | dk | 1,000,000 | 6.2983s vs 5.2321s | linearmodels | 0.83x | 1060.9MB / 1257.4MB |

<!-- END GENERATED SUMMARY -->

## How to read these numbers

- **Relative, not absolute.** The numbers come from the *Benchmark (performance)* workflow on shared GitHub Actions runners, so absolute times vary from run to run, often by tens of percent. Compare the two columns within a row, not the absolute values across pages or releases. They are refreshed at each release.
- **Single-threaded.** The linear-algebra backends of both sides are pinned to one thread. This isolates the efficiency of the computation itself from thread-pool behavior. It does not favor this package: with a small number of regressors and many observations, which is the typical econometric shape, multi-threaded BLAS made the reference implementation *slower*, and with a few dozen regressors it was only slightly faster, so the one-thread setting shows the reference at its best. Speed on many cores for very wide models is a different question that these tables do not answer.
- **Like for like.** The package computes the full set of reported statistics (fit statistics, tests, information criteria) in every call, while the reference packages compute many of them lazily. The benchmark touches those lazy results so both sides do the same work. Converting a polars frame to pandas for the reference packages is excluded from the timing, since it is not part of their estimation.
- **End to end.** Times include the Python call, the Arrow hand-over and the conversion of results, as a user would experience them.
- **Peak RSS** is the peak resident memory of the whole process, including the Python interpreter and imported libraries. At small `n` it mostly reflects that baseline rather than the estimator.
- **Release build.** The extension is built in release mode. A debug build can be more than ten times slower.
- **Scope of the sweeps.** Each method is measured over sample size (`k = 5`), number of regressors (`n = 10,000`) and one or two representative standard-error types (the cheapest, and the most expensive one). Solver, estimator and effects variants are measured at one representative point. This is meant to show relative trends, not every option combination.

## Known limitations

Performance is a goal rather than a guarantee. The points below are the ones worth knowing about.

- **Random effects (RE) is slower than linearmodels at very large `n`.** At `n = 1,000,000` it takes about 6.3 s against 5.3 s for linearmodels (speed-up 0.8x), while it is faster at `n` up to 100,000. RE does more work than FE per call: it estimates the variance components and runs an internal fixed-effects fit for the Hausman test, which is always included. Making the large-`n` case faster is open work.
- **RE with many regressors and `cov_type="dk"` can fail.** With few time periods and many regressors (`k = 20` with six periods in the benchmark), the Hausman auxiliary regression's Driscoll–Kraay covariance becomes singular and `fit()` raises a `ComputationError`. The benchmark therefore has no timing for that point. FE has the same structural limit, which is why its regressor sweep covers the classical covariance only.
- **Logit, Probit and Tobit are on par with statsmodels rather than far ahead.** They are iterative maximum-likelihood fits, so the advantage over statsmodels is modest (about 1.0x to 1.9x for Logit and Probit at the sizes measured), and it shrinks as the number of regressors grows: Probit with clustered errors and 20 regressors is not faster than statsmodels. The default Newton solver is the fastest or tied for fastest; `bfgs` and `lbfgs` are about equal or slower.
- **Tobit has no reference timing.** There is no suitable Python Tobit implementation to time in the same process (statsmodels has none), so the Tobit rows show this package alone (absolute time and scaling with `n` and `k`). Its correctness is checked against R, see [Verification](verification.md).
- **Probit and Tobit at the largest sizes are measured for this package only.** The reference implementation is not timed at `n = 200,000` and `n = 1,000,000` for Probit, so the largest comparison for Probit is at 100,000 observations.
- **Instrumental variables: part of the gap is symmetric work.** linearmodels computes its first-stage diagnostics by re-fitting each first stage, and the benchmark includes that, because this package always reports the weak-instrument statistics from the same pass. In an earlier local measurement the 2SLS gap was still about threefold with that re-fit excluded. GMM with a kernel (HAC) weight matrix is not benchmarked, because linearmodels takes tens of seconds even at 100,000 observations.

## Reproducing

The measurement scripts live in the [`performance/`](https://github.com/masahiroyecon1997dev/econometricsmodels/tree/main/performance) directory of the repository; `performance/compare_<method>.py` runs one method and the shared harness takes care of process isolation, thread pinning and the symmetric measurement. Build the extension with `maturin develop --release` first.
