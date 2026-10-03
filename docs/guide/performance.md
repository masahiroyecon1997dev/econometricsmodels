# Performance

The computational core is written in Rust. This page summarizes how its execution time and memory use compare with the reference implementations the results are verified against ([statsmodels](https://www.statsmodels.org/) and [linearmodels](https://bashtage.github.io/linearmodels/)), what the numbers do and do not tell you, and the known performance limitations. The full per-method tables are on [Performance results](performance-results.md).

## Summary

Each row is the largest sample size at which every library was measured (`k = 5` regressors). "Speed-up" is the reference time divided by the econometricsmodels time; a value below 1 means the reference was faster. Rows can differ in `n` because the largest sizes are measured for the default standard error only (for example Tobit: classical at 1,000,000, cluster at 100,000).

<!-- BEGIN GENERATED SUMMARY (performance.render_docs_results) -->

| Method | cov_type | n | econometricsmodels | Reference | Speed-up | Peak RSS (econometricsmodels / reference) |
|---|---|---|---|---|---|---|
| OLS | classical | 1,000,000 | 0.0817s vs 0.2246s | statsmodels | 2.7x | 299.2MB / 542.4MB |
| OLS | hac | 1,000,000 | 0.2269s vs 0.9443s | statsmodels | 4.2x | 345.3MB / 627.7MB |
| WLS | classical | 1,000,000 | 0.1029s vs 0.2338s | statsmodels | 2.3x | 314.6MB / 587.9MB |
| WLS | hac | 1,000,000 | 0.2505s vs 0.9591s | statsmodels | 3.8x | 360.4MB / 643.1MB |
| Logit | classical | 1,000,000 | 0.5874s vs 0.5986s | statsmodels | 1.0x | 370.1MB / 527.4MB |
| Logit | cluster | 1,000,000 | 0.8068s vs 0.6613s | statsmodels | 0.82x | 462.8MB / 521.7MB |
| Probit | classical | 100,000 | 0.1401s vs 0.1901s | statsmodels | 1.4x | 174.5MB / 266.2MB |
| Probit | cluster | 100,000 | 0.1820s vs 0.2028s | statsmodels | 1.1x | 187.8MB / 268.0MB |
| Tobit | classical | 1,000,000 | 2.1524s | - | - | 400.4MB |
| Tobit | cluster | 100,000 | 0.2547s | - | - | 184.5MB |
| IV | classical | 1,000,000 | 0.4983s vs 14.7670s | linearmodels | 29.6x | 1101.6MB / 3952.8MB |
| IV | hac | 1,000,000 | 1.1949s vs 15.4124s | linearmodels | 12.9x | 1162.2MB / 3952.4MB |
| FE | classical | 1,000,000 | 0.3965s vs 3.7882s | linearmodels | 9.6x | 599.0MB / 943.2MB |
| FE | dk | 1,000,000 | 0.4874s vs 3.7707s | linearmodels | 7.7x | 660.7MB / 943.6MB |
| RE | classical | 1,000,000 | 0.5447s vs 5.1234s | linearmodels | 9.4x | 1023.1MB / 1258.1MB |
| RE | dk | 1,000,000 | 0.6762s vs 5.1571s | linearmodels | 7.6x | 1087.1MB / 1257.2MB |

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
- **Logit, Probit and Tobit are on par with statsmodels rather than far ahead.** They are iterative maximum-likelihood fits, so at the largest sizes measured the ratio to statsmodels ranges from about 0.8x to 1.4x (Logit with clustered errors at 1,000,000 observations was slower than statsmodels in the latest run). The advantage also shrinks as the number of regressors grows: for Logit with the classical covariance it falls from about 2.2x at `k = 5` to 1.4x at `k = 20`. The default Newton solver is the fastest or tied for fastest; `bfgs` and `lbfgs` are about equal or slower.
- **Tobit has no reference timing.** There is no suitable Python Tobit implementation to time in the same process (statsmodels has none), so the Tobit rows show this package alone (absolute time and scaling with `n` and `k`). Its correctness is checked against R, see [Verification](verification.md).
- **Probit and Tobit at the largest sizes are measured for this package only.** The reference implementation is not timed at `n = 200,000` and `n = 1,000,000` for Probit, so the largest comparison for Probit is at 100,000 observations.
- **Instrumental variables: part of the gap is symmetric work.** linearmodels computes its first-stage diagnostics by re-fitting each first stage, and the benchmark includes that, because this package always reports the weak-instrument statistics from the same pass. In an earlier local measurement the 2SLS gap was still about threefold with that re-fit excluded. GMM with a kernel (HAC) weight matrix is not benchmarked, because linearmodels takes tens of seconds even at 100,000 observations.

## Reproducing

The measurement scripts live in the [`performance/`](https://github.com/masahiroyecon1997dev/econometricsmodels/tree/main/performance) directory of the repository; `performance/compare_<method>.py` runs one method and the shared harness takes care of process isolation, thread pinning and the symmetric measurement. Build the extension with `maturin develop --release` first.
