//! `FEIVOptions`（FEIVの推定オプション）のpyclass定義。値の解釈・組合せの検査は`option_values.rs`。
//!
//! 各フィールドの意味・既定値の根拠は`FEOptions`（`panel/fe.rs`）と`IVOptions`（`iv/common/options.rs`）と同じ。
//! `estimator`で2SLSとGMMを切り替える（`IVOptions`と同じ。別クラスにしない）。

use engine::panel::fe::FeAbsorption;
use pyo3::prelude::*;

use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_int, extract_strict_opt_column, extract_strict_opt_float,
    extract_strict_opt_int, extract_strict_opt_text, extract_strict_text,
};

/// Estimation options for FEIV (fixed effects instrumental variables, with two-stage least
/// squares or GMM).
///
/// All arguments are keyword-only.
///
/// Options by purpose:
///
/// - Estimator: `estimator`.
/// - GMM (with `estimator="gmm"` only): `gmm_type`; with `"two_step"` or `"iterated"` also
///   `gmm_weight_type`; with `"iterated"` also `gmm_max_iter`, `gmm_tol` and
///   `raise_on_non_convergence`.
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`; with
///   `"dk"` also `dk_time` (required) and `dk_bandwidth`. With `estimator="gmm"`, `cluster` is
///   also used by `gmm_weight_type="cluster"`.
/// - Confidence intervals: `confidence_level`.
/// - Removing the fixed effects of an unbalanced two-way model or of a model with
///   three or more effects: `tol`, `max_iter`.
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct FEIVOptions {
    /// Estimator: "2sls" (two-stage least squares) or "gmm" (generalized method of moments),
    /// case-insensitive. Defaults to "2sls". Options of the GMM estimator (`gmm_type`,
    /// `gmm_weight_type`, `gmm_max_iter`, `gmm_tol`, `raise_on_non_convergence`) raise
    /// `ValidationError` with `estimator="2sls"`.
    #[pyo3(get)]
    pub estimator: String,

    /// Standard error type: one of "classical", "hc1", "hc2", "hc3", "cluster", "dk".
    /// Case-insensitive. Unlike OLS/WLS/IV, "hc0" is **not** supported. With `estimator="2sls"`
    /// the first-stage regressions and the diagnostics that depend on a covariance
    /// (`kleibergen_paap()` and `effective_f()`) follow it. With `estimator="gmm"` it is
    /// independent of `gmm_weight_type` (the weight matrix of the point estimate): the sandwich
    /// covariance is built from the weight matrix actually used and the moment variance of this
    /// type, and "hc2" and "hc3" have not been compared with another implementation.
    #[pyo3(get)]
    pub cov_type: String,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key, for `cov_type="cluster"` and, with
    /// `estimator="gmm"`, for `gmm_weight_type="cluster"` (`gmm_type="two_step"` or
    /// `"iterated"`). When `None`, the first column of `effects` is used automatically.
    /// Specifying it when neither uses it raises `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Column name that defines the time periods of the Driscoll-Kraay HAC, independent
    /// of `effects`. Required when `cov_type="dk"`: leaving it unset raises
    /// `ValidationError`, and the columns of `effects` are not used instead. It may not be
    /// the same column as `y` or the first column of `effects` (`ValidationError`).
    /// Specifying it with any other `cov_type` raises `ValidationError`. The periods are
    /// ordered by the values of the column: numerically for integers and floats,
    /// chronologically for `Date` and `Datetime`, in the order of the categories for
    /// `Enum`, and alphabetically for strings and `Categorical`.
    #[pyo3(get)]
    pub dk_time: Option<String>,

    /// Bandwidth for Driscoll-Kraay HAC when `cov_type="dk"`. When `None`, computed
    /// automatically via `floor(4*(t/100)^(2/9))` (`t` = number of unique time periods).
    /// Specifying it with any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub dk_bandwidth: Option<i64>,

    /// GMM variant: one of "one_step", "two_step", "iterated" (case-insensitive).
    /// `None` (default) means "two_step" for `estimator="gmm"`. "one_step" uses the initial
    /// weight `(Z'Z)^-1` only and gives the two-stage least squares point estimate; "two_step"
    /// re-estimates once with the weight built from the first-step residuals; "iterated"
    /// repeats that until the coefficients converge. Specifying it with `estimator="2sls"`
    /// raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_type: Option<String>,

    /// Weight matrix of the moment conditions: one of "classical", "robust", "cluster"
    /// (case-insensitive). `None` (default) means "cluster", with the groups of `cluster`
    /// (the first column of `effects` if `cluster` is `None`). "classical" gives the same
    /// point estimate as two-stage least squares; "robust" assumes the within-transformed
    /// errors are uncorrelated over time, which does not hold for short panels, so it is
    /// not recommended there. There is no "hac" weight. Specifying it with `estimator="2sls"`
    /// or with `gmm_type="one_step"` (which uses the fixed weight `(Z'Z)^-1` and has no weight
    /// type) raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,

    /// Maximum number of GMM estimations for `gmm_type="iterated"`, counting the initial
    /// estimate; an integer from 3 to 10,000 (use `gmm_type="two_step"` for two steps).
    /// `None` (default) means 100. Specifying it with any other `gmm_type` or with
    /// `estimator="2sls"` raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_max_iter: Option<i64>,

    /// Convergence tolerance for `gmm_type="iterated"`: iteration stops once every
    /// coefficient changes by at most `gmm_tol` relative to its previous value (with an
    /// absolute floor of 1e-8, so that coefficients near zero converge). `None` (default) means
    /// 1e-6. Specifying it with any other `gmm_type` or with `estimator="2sls"` raises
    /// `ValidationError`.
    #[pyo3(get)]
    pub gmm_tol: Option<f64>,

    /// Whether to raise an error if `gmm_type="iterated"` does not converge within
    /// `gmm_max_iter`. If `False`, returns the result with `converged=False` instead of
    /// raising. `None` (default) means `True`. Specifying it with any other `gmm_type`
    /// (which never checks convergence) or with `estimator="2sls"` raises `ValidationError`.
    #[pyo3(get, set)]
    pub raise_on_non_convergence: Option<bool>,

    /// Convergence tolerance of the alternating projections that remove the fixed
    /// effects of a two-way model on an **unbalanced** panel, or of a model with three or
    /// more effects (a balanced two-way panel and a one-way model are computed in closed
    /// form, so they ignore it). A pass stops when the largest group mean it subtracts is at
    /// most `tol` times the size of the column. Must be a positive finite number. Defaults
    /// to 1e-12. Not related to `gmm_tol`.
    #[pyo3(get)]
    pub tol: f64,

    /// Maximum number of passes of the alternating projections (see `tol`), from 1 to
    /// 10000. Defaults to 10000. If the limit is reached, `ComputationError` is raised.
    /// Not related to `gmm_max_iter`.
    #[pyo3(get)]
    pub max_iter: i64,
}

crate::shared::pickle::impl_pickle_state!(FEIVOptions {
    estimator,
    cov_type,
    confidence_level,
    cluster,
    dk_time,
    dk_bandwidth,
    gmm_type,
    gmm_weight_type,
    gmm_max_iter,
    gmm_tol,
    raise_on_non_convergence,
    tol,
    max_iter,
});

impl FEIVOptions {
    /// このオプションが指名するデータの列（`cluster`・`dk_time`）。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().chain(&self.dk_time).cloned().collect()
    }
}

#[pymethods]
impl FEIVOptions {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    #[new]
    #[pyo3(signature = (
        *,
        estimator = "2sls".to_string(),
        cov_type = "cluster".to_string(),
        confidence_level = 0.95,
        cluster = None,
        dk_time = None,
        dk_bandwidth = None,
        gmm_type = None,
        gmm_weight_type = None,
        gmm_max_iter = None,
        gmm_tol = None,
        raise_on_non_convergence = None,
        tol = None,
        max_iter = 10000,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::estimator_arg)] estimator: String,
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_time_arg)] dk_time: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_bandwidth_arg)] dk_bandwidth: Option<
            i64,
        >,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_type_arg)] gmm_type: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_weight_type_arg)]
        gmm_weight_type: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_max_iter_arg)] gmm_max_iter: Option<
            i64,
        >,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_tol_arg)] gmm_tol: Option<f64>,
        raise_on_non_convergence: Option<bool>,
        #[pyo3(from_py_with = crate::shared::option_values::tol_arg)] tol: Option<f64>,
        #[pyo3(from_py_with = crate::shared::option_values::max_iter_arg)] max_iter: i64,
    ) -> Self {
        let defaults = FeAbsorption::default();
        Self {
            estimator,
            cov_type,
            confidence_level,
            cluster,
            dk_time,
            dk_bandwidth,
            gmm_type,
            gmm_weight_type,
            gmm_max_iter,
            gmm_tol,
            raise_on_non_convergence,
            tol: tol.unwrap_or(defaults.tol),
            max_iter,
        }
    }

    #[setter]
    fn set_estimator(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.estimator = extract_strict_text(value, "estimator")?;
        Ok(())
    }

    #[setter]
    fn set_confidence_level(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.confidence_level = extract_strict_float(value, "confidence_level")?;
        Ok(())
    }

    #[setter]
    fn set_dk_bandwidth(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.dk_bandwidth = extract_strict_opt_int(value, "dk_bandwidth")?;
        Ok(())
    }

    #[setter]
    fn set_tol(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.tol = extract_strict_float(value, "tol")?;
        Ok(())
    }

    #[setter]
    fn set_max_iter(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.max_iter = extract_strict_int(value, "max_iter")?;
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
    fn set_dk_time(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.dk_time = extract_strict_opt_column(value, "dk_time")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_type = extract_strict_opt_text(value, "gmm_type")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_weight_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_weight_type = extract_strict_opt_text(value, "gmm_weight_type")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_max_iter(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_max_iter = extract_strict_opt_int(value, "gmm_max_iter")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_tol(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_tol = extract_strict_opt_float(value, "gmm_tol")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "FEIVOptions(estimator={}, cov_type={}, confidence_level={}, cluster={}, \
             dk_time={}, dk_bandwidth={}, gmm_type={}, gmm_weight_type={}, \
             gmm_max_iter={}, gmm_tol={}, raise_on_non_convergence={}, tol={}, max_iter={})",
            self.estimator.py_repr(),
            self.cov_type.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.dk_time.py_repr(),
            self.dk_bandwidth.py_repr(),
            self.gmm_type.py_repr(),
            self.gmm_weight_type.py_repr(),
            self.gmm_max_iter.py_repr(),
            self.gmm_tol.py_repr(),
            self.raise_on_non_convergence.py_repr(),
            self.tol.py_repr(),
            self.max_iter.py_repr()
        )
    }
}

// 保持するオプションは、それ自身の`__reduce__`でpickleされるオブジェクトとして状態に入れる（`IVOptions`と同じ）。
impl crate::shared::pickle::PickleField for FEIVOptions {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Py::new(py, self.clone())?.into_bound(py).into_any())
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let options = obj.cast::<FEIVOptions>().ok()?.try_borrow().ok()?;
        Some(options.clone())
    }
}
