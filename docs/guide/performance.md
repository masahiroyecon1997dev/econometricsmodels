# Performance

The computational core is written in Rust. This page summarizes how its execution time and memory use compare with the reference implementations the results are verified against ([statsmodels](https://www.statsmodels.org/) and [linearmodels](https://bashtage.github.io/linearmodels/)), what the numbers do and do not tell you, and the known performance limitations. The full per-method tables are on [Performance results](performance-results.md).

## Summary

Each row is the largest sample size at which every library was measured (`k = 5` regressors). "Speed-up" is the reference time divided by the econometricsmodels time; a value below 1 means the reference was faster. Rows can differ in `n` because the largest sizes are measured for the default standard error only (for example Tobit: classical at 1,000,000, cluster at 100,000).

<!-- BEGIN GENERATED SUMMARY (performance.render_docs_results) -->

| Method | cov_type | n | econometricsmodels | Reference | Speed-up | Peak RSS (econometricsmodels / reference) |
|---|---|---|---|---|---|---|
| OLS | classical | 1,000,000 | 0.0725s vs 0.2176s | statsmodels | 3.0x | 320.4MB / 566.4MB |
| OLS | hac | 1,000,000 | 0.2352s vs 0.9347s | statsmodels | 4.0x | 382.4MB / 651.5MB |
| WLS | classical | 1,000,000 | 0.1037s vs 0.2424s | statsmodels | 2.3x | 335.6MB / 612.1MB |
| WLS | hac | 1,000,000 | 0.2673s vs 0.9800s | statsmodels | 3.7x | 390.0MB / 667.6MB |
| Logit | classical | 1,000,000 | 0.7033s vs 0.9501s | statsmodels | 1.4x | 408.6MB / 524.1MB |
| Logit | cluster | 1,000,000 | 0.9410s vs 1.0482s | statsmodels | 1.1x | 461.2MB / 525.5MB |
| Probit | classical | 100,000 | 0.0828s vs 0.0961s | statsmodels | 1.2x | 175.3MB / 273.1MB |
| Probit | cluster | 100,000 | 0.1063s vs 0.1060s | statsmodels | 1.00x | 185.2MB / 273.7MB |
| Tobit | classical | 1,000,000 | 1.3214s | - | - | 401.3MB |
| Tobit | cluster | 100,000 | 0.1552s | - | - | 188.5MB |
| IV | classical | 1,000,000 | 0.4884s vs 14.5023s | linearmodels | 29.7x | 1122.0MB / 3999.6MB |
| IV | hac | 1,000,000 | 1.1754s vs 15.4506s | linearmodels | 13.1x | 1214.9MB / 3978.6MB |
| FE | classical | 1,000,000 | 0.1953s vs 2.1864s | linearmodels | 11.2x | 616.1MB / 963.1MB |
| FE | dk | 1,000,000 | 0.2675s vs 2.2150s | linearmodels | 8.3x | 683.7MB / 963.5MB |
| RE | classical | 1,000,000 | 0.6162s vs 5.9855s | linearmodels | 9.7x | 1040.1MB / 1273.6MB |
| RE | dk | 1,000,000 | 0.7399s vs 5.9611s | linearmodels | 8.1x | 1107.4MB / 1275.2MB |

<!-- END GENERATED SUMMARY -->

## How to read these numbers

- **Relative, not absolute.** The numbers come from the *Benchmark (performance)* workflow on shared GitHub Actions runners, so absolute times vary from run to run, often by tens of percent (the statsmodels Logit time at 1,000,000 observations differed by more than a factor of two between two CI runs). Compare the two columns within a row, not the absolute values across pages or releases. They are refreshed at each release.
- **Single-threaded.** The linear-algebra backends of both sides are pinned to one thread. This isolates the efficiency of the computation itself from thread-pool behavior. It does not favor this package: with a small number of regressors and many observations, which is the typical econometric shape, multi-threaded BLAS made the reference implementation *slower*, and with a few dozen regressors it was only slightly faster, so the one-thread setting shows the reference at its best. Speed on many cores for very wide models is a different question that these tables do not answer.
- **Like for like.** The package computes the full set of reported statistics (fit statistics, tests, information criteria) in every call, while the reference packages compute many of them lazily. The benchmark touches those lazy results so both sides do the same work. Converting a polars frame to pandas for the reference packages is excluded from the timing, since it is not part of their estimation.
- **End to end.** Times include the Python call, the Arrow hand-over and the conversion of results, as a user would experience them.
- **Peak RSS** is the peak resident memory of the whole process, including the Python interpreter and imported libraries. At small `n` it mostly reflects that baseline rather than the estimator.
- **Release build.** The extension is built in release mode. A debug build can be more than ten times slower.
- **Scope of the sweeps.** Each method is measured over sample size (`k = 5`), number of regressors (`n = 10,000`) and one or two representative standard-error types (the cheapest, and the most expensive one). Solver, estimator and effects variants are measured at one representative point. This is meant to show relative trends, not every option combination.

## Known limitations

Performance is a goal rather than a guarantee. The points below are the ones worth knowing about.

- **The regressor sweep for FE and RE covers the classical covariance only.** The Driscoll–Kraay covariance has rank at most `T - 1` (the number of time periods minus one), so a joint test of more coefficients than that is not possible. With six periods and `k = 20` in the benchmark, `fit()` rejects `cov_type="dk"` with a `ValidationError`: for FE because of the slope F-test, and for RE because of the Hausman test, which is always included.
- **Logit, Probit and Tobit are on par with statsmodels rather than far ahead.** They are iterative maximum-likelihood fits, so at the largest sizes measured the ratio to statsmodels ranges from about 1.0x to 1.4x (Probit with clustered errors at 100,000 observations was tied with statsmodels in the latest run; in earlier runs Logit with clustered errors at 1,000,000 observations was slower). The advantage also shrinks as the number of regressors grows: for Logit with the classical covariance it falls from about 2.2x at `k = 5` to 1.4x at `k = 20`. The default Newton solver is the fastest or tied for fastest; `bfgs` and `lbfgs` are about equal or slower.
- **Tobit has no reference timing.** There is no suitable Python Tobit implementation to time in the same process (statsmodels has none), so the Tobit rows show this package alone (absolute time and scaling with `n` and `k`). Its correctness is checked against R, see [Verification](verification.md).
- **Probit and Tobit at the largest sizes are measured for this package only.** The reference implementation is not timed at `n = 200,000` and `n = 1,000,000` for Probit, so the largest comparison for Probit is at 100,000 observations.
- **Instrumental variables: part of the gap is symmetric work.** linearmodels computes its first-stage diagnostics by re-fitting each first stage, and the benchmark includes that, because this package always reports the weak-instrument statistics from the same pass. In an earlier local measurement the 2SLS gap was still about threefold with that re-fit excluded. GMM with a kernel (HAC) weight matrix is not benchmarked, because linearmodels takes tens of seconds even at 100,000 observations.

## Reproducing

The measurement scripts live in the [`performance/`](https://github.com/masahiroyecon1997dev/econometricsmodels/tree/main/performance) directory of the repository; `performance/compare_<method>.py` runs one method and the shared harness takes care of process isolation, thread pinning and the symmetric measurement. Build the extension with `maturin develop --release` first.
