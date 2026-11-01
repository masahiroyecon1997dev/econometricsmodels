//! `TobitOptions`（Tobitの推定オプション）。

use crate::nonlinear::common::resolve_mle_tol_default;
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_int, extract_strict_opt_column, extract_strict_opt_float,
    extract_strict_text,
};
use pyo3::prelude::*;

/// Estimation options for Tobit.
///
/// All arguments are keyword-only.
///
/// Options by purpose:
///
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`.
/// - Censoring: `lower`, `upper` (at least one must be set).
/// - Optimizer: `solver`, `max_iter`, `tol`, `raise_on_non_convergence`.
/// - Confidence intervals: `confidence_level`.
/// - Model: `include_intercept`.
///
// 各フィールドの意味・既定値の根拠は`docs/spec/tobit-spec.md`。
// `LogitOptions`と同じ理由（pyo3 0.28以降のFromPyObject自動導出の仕様変更、
// mkdocstringsのalias解決対応）で`from_py_object` + `module`を明示する。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct TobitOptions {
    /// Standard error type: one of "classical", "opg", "hc0",
    /// "hc1", "cluster". Case-insensitive.
    #[pyo3(get)]
    pub cov_type: String,

    /// Whether to automatically add an intercept column.
    /// When true, a column of all 1.0 is prepended to the design matrix.
    /// If the user's `x` already contains a constant column while this is true,
    /// the resulting perfect collinearity raises `ComputationError` (singular
    /// design matrix).
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`.
    /// Refers to a column in `data` rather than being passed as a separate array.
    /// Specifying it with any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Optimization solver: one of "newton" (default), "bfgs", "lbfgs".
    /// Case-insensitive.
    #[pyo3(get)]
    pub solver: String,

    /// Maximum number of solver iterations, an integer from 1 to 10,000 (anything else
    /// raises `ValidationError`).
    #[pyo3(get)]
    pub max_iter: i64,

    /// Convergence tolerance. For `solver="newton"`, an absolute threshold on
    /// the total gradient norm (default `1e-6`). For `solver="bfgs"`/`"lbfgs"`,
    /// a per-observation-average gradient threshold (the total gradient norm
    /// divided by the number of observations must fall below `tol`, default
    /// `1e-8`), matching how statsmodels/scipy normalize convergence checks
    /// by `nobs`. The two defaults are not interchangeable: Newton's absolute
    /// semantics with a `1e-8` threshold (or bfgs/lbfgs's normalized
    /// semantics with a `1e-6` threshold) measurably degrades either speed or
    /// precision. Passing `tol` explicitly always uses the
    /// semantics of the chosen `solver`. For `bfgs`/`lbfgs`, the solver also
    /// reports convergence when the gradient norm is within `100 * tol` of the
    /// target and the line search can no longer make progress because the
    /// log-likelihood has reached floating-point precision (common with very
    /// large samples); this window scales with `tol`, so it widens if you
    /// loosen `tol`. Note: the solver-dependent default is
    /// resolved once, at construction time. Changing `solver` afterwards via
    /// the setter does not re-resolve `tol` — set both together (or set `tol`
    /// explicitly) if you change `solver` after construction.
    #[pyo3(get)]
    pub tol: f64,

    /// If true (default), raise `ComputationError` when the solver fails to
    /// converge within `max_iter` iterations. If false, return the result at the
    /// final iterate instead (with `converged=False`).
    #[pyo3(get, set)]
    pub raise_on_non_convergence: bool,

    /// Lower censoring bound. `None` means "no censoring from below". Defaults to
    /// `0.0` (the standard left-censored-at-zero Tobit model). At least one of
    /// `lower`/`upper` must be set (both `None` raises `ValidationError`).
    #[pyo3(get)]
    pub lower: Option<f64>,

    /// Upper censoring bound. `None` (default) means "no censoring from above".
    #[pyo3(get)]
    pub upper: Option<f64>,
}

crate::shared::pickle::impl_pickle_state!(TobitOptions {
    cov_type,
    include_intercept,
    confidence_level,
    cluster,
    solver,
    max_iter,
    tol,
    raise_on_non_convergence,
    lower,
    upper,
});

impl TobitOptions {
    /// このオプションが指名するデータの列（`cluster`）。使われるかどうか（`cov_type`等）には
    /// よらず、指名されたものを全て返す。`fit()`はこれらと`y`・`x`等だけをデータから取り込む。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().cloned().collect()
    }
}

#[pymethods]
impl TobitOptions {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    #[new]
    #[pyo3(signature = (
        *,
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster = None,
        solver = "newton".to_string(),
        max_iter = 35,
        tol = None,
        raise_on_non_convergence = true,
        lower = Some(0.0),
        upper = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        include_intercept: bool,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::solver_arg)] solver: String,
        #[pyo3(from_py_with = crate::shared::option_values::max_iter_arg)] max_iter: i64,
        #[pyo3(from_py_with = crate::shared::option_values::tol_arg)] tol: Option<f64>,
        raise_on_non_convergence: bool,
        #[pyo3(from_py_with = crate::shared::option_values::lower_arg)] lower: Option<f64>,
        #[pyo3(from_py_with = crate::shared::option_values::upper_arg)] upper: Option<f64>,
    ) -> Self {
        // `tol`の既定値はsolver依存（理由は`resolve_mle_tol_default`参照）。
        let tol = resolve_mle_tol_default(&solver, tol);
        Self {
            cov_type,
            include_intercept,
            confidence_level,
            cluster,
            solver,
            max_iter,
            tol,
            raise_on_non_convergence,
            lower,
            upper,
        }
    }

    #[setter]
    fn set_confidence_level(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.confidence_level = extract_strict_float(value, "confidence_level")?;
        Ok(())
    }

    #[setter]
    fn set_max_iter(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.max_iter = extract_strict_int(value, "max_iter")?;
        Ok(())
    }

    #[setter]
    fn set_tol(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.tol = extract_strict_float(value, "tol")?;
        Ok(())
    }

    #[setter]
    fn set_lower(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.lower = extract_strict_opt_float(value, "lower")?;
        Ok(())
    }

    #[setter]
    fn set_upper(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.upper = extract_strict_opt_float(value, "upper")?;
        Ok(())
    }

    #[setter]
    fn set_cov_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.cov_type = extract_strict_text(value, "cov_type")?;
        Ok(())
    }

    #[setter]
    fn set_cluster(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.cluster = extract_strict_opt_column(value, "cluster")?;
        Ok(())
    }

    #[setter]
    fn set_solver(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.solver = extract_strict_text(value, "solver")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "TobitOptions(cov_type={}, include_intercept={}, confidence_level={}, \
             cluster={}, solver={}, max_iter={}, tol={}, raise_on_non_convergence={}, \
             lower={}, upper={})",
            self.cov_type.py_repr(),
            self.include_intercept.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.solver.py_repr(),
            self.max_iter.py_repr(),
            self.tol.py_repr(),
            self.raise_on_non_convergence.py_repr(),
            self.lower.py_repr(),
            self.upper.py_repr()
        )
    }
}

/// `build_tobit_input`などのテストで使う既定の`TobitOptions`（`cov_type="classical"`・
/// `include_intercept=true`・`solver="newton"`・`lower=Some(0.0)`・`upper=None`）。
/// フィールドごとに上書きして使う。
#[cfg(test)]
pub(super) fn default_options() -> TobitOptions {
    TobitOptions::new(
        "classical".to_string(),
        true,
        0.95,
        None,
        "newton".to_string(),
        35,
        Some(1e-6),
        true,
        Some(0.0),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tol=None`のとき、`solver`に応じた既定値（`newton`は絶対閾値`1e-6`、
    /// `bfgs`/`lbfgs`は観測数正規化基準`1e-8`）が解決されるはず
    /// （`LogitOptions`と同じロジック）。
    #[test]
    fn new_resolves_tol_default_based_on_method_when_tol_is_none() {
        let newton = TobitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "newton".to_string(),
            35,
            None,
            true,
            Some(0.0),
            None,
        );
        assert_eq!(newton.tol, 1e-6);

        let bfgs = TobitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "bfgs".to_string(),
            35,
            None,
            true,
            Some(0.0),
            None,
        );
        assert_eq!(bfgs.tol, 1e-8);

        let lbfgs = TobitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "lbfgs".to_string(),
            35,
            None,
            true,
            Some(0.0),
            None,
        );
        assert_eq!(lbfgs.tol, 1e-8);
    }
}
