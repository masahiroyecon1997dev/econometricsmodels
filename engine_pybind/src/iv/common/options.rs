//! `IVOptions`（IVの推定オプション）。

use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_opt_column, extract_strict_opt_float,
    extract_strict_opt_int, extract_strict_opt_text, extract_strict_text,
};
use pyo3::prelude::*;

/// Estimation options for IV (2SLS/GMM).
///
/// All arguments are keyword-only.
///
/// A single options class serves both estimation methods; fields that apply to
/// only one estimator are documented as such.
///
/// Options by purpose:
///
/// - Estimator: `estimator`.
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`; with
///   `"hac"` also `hac_time` (required) and `hac_lags`.
/// - GMM only: `gmm_type`, `gmm_weight_type` (the weight matrix of the point
///   estimate, with the same `cluster`, `hac_time` and `hac_lags` fields).
/// - GMM, `gmm_type="iterated"` only: `gmm_max_iter`, `gmm_tol`,
///   `raise_on_non_convergence`.
/// - Confidence intervals: `confidence_level`.
/// - Model: `include_intercept`.
// 各フィールドの意味・既定値の根拠は`docs/spec/iv-spec.md`。
// module/from_py_objectの理由は`OLSOptions`/`LogitOptions`と同じ
// （`engine_pybind/src/linear/ols/options.rs`のコメント参照）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct IVOptions {
    /// Estimator: "2sls" (default) or "gmm". Case-insensitive.
    #[pyo3(get)]
    pub estimator: String,

    /// Standard error type: one of "classical", "hc0" through "hc3", "hac", "cluster".
    /// Case-insensitive. For `estimator="gmm"`, this is independent of `gmm_weight_type`
    /// (the weight matrix used for point estimation, see `gmm_weight_type` below).
    #[pyo3(get)]
    pub cov_type: String,

    /// Whether to automatically add an intercept column to `x_exog`.
    /// `x_endog`/`instruments` never get an automatic intercept column.
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key. Used by `cov_type="cluster"` and,
    /// with `estimator="gmm"`, by `gmm_weight_type="cluster"` (also when
    /// `cov_type` is not "cluster"). Specifying it when neither uses it raises
    /// `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Number of lags (bandwidth) for HAC (Newey-West), used by `cov_type="hac"` and, with
    /// `estimator="gmm"`, by `gmm_weight_type="hac"`. When `None`, computed automatically.
    /// Specifying it when neither uses it raises `ValidationError`.
    #[pyo3(get)]
    pub hac_lags: Option<i64>,

    /// Column name giving the time order for HAC, used by `cov_type="hac"` and, with
    /// `estimator="gmm"`, by `gmm_weight_type="hac"`. Required when either is "hac"
    /// (omitting it raises `ValidationError`): the row order of the data is never assumed
    /// to be the time order. Specifying it when neither uses it raises `ValidationError`.
    #[pyo3(get)]
    pub hac_time: Option<String>,

    /// GMM estimation type: "one_step" (weight matrix `(Z'Z)^-1` only), "two_step"
    /// (efficient two-step GMM), or "iterated" (repeat until convergence).
    /// Case-insensitive. `None` (default) means "two_step" for `estimator="gmm"`.
    /// Specifying it with `estimator="2sls"` raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_type: Option<String>,

    /// Weight matrix used for GMM point estimation: one of "classical" (homoskedastic),
    /// "robust" (heteroskedasticity-robust), "cluster", "hac" (Newey-West). Same vocabulary
    /// as `cov_type`. Case-insensitive. `None` (default) means "classical" for
    /// `gmm_type="two_step"`/`"iterated"`. Specifying it with `estimator="2sls"` or
    /// `gmm_type="one_step"` (which does not use a weight matrix) raises `ValidationError`.
    /// "cluster"/"hac" draw from the same `cluster`/`hac_lags`/`hac_time` fields as
    /// `cov_type` (there are no separate fields for GMM).
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,

    /// Maximum number of GMM estimations for `gmm_type="iterated"`, counting the initial
    /// estimate; an integer from 3 to 10,000 (use `gmm_type="two_step"` for two steps).
    /// `None` (default) means 100. Specifying it with any other `gmm_type` or with
    /// `estimator="2sls"` raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_max_iter: Option<i64>,

    /// Convergence tolerance for `gmm_type="iterated"`: iteration stops once every
    /// coefficient changes by less than this relative to its previous value (with an absolute
    /// floor of 1e-8, so that coefficients near zero converge). `None` (default)
    /// means 1e-6. Specifying it with any other `gmm_type` or with `estimator="2sls"`
    /// raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_tol: Option<f64>,

    /// Whether to raise an error if `gmm_type="iterated"` does not converge within
    /// `gmm_max_iter`. If `False`, returns the result with `converged=False` instead of
    /// raising. `None` (default) means `True`. Specifying it with any other `gmm_type`
    /// (which never checks convergence) or with `estimator="2sls"` raises
    /// `ValidationError`.
    #[pyo3(get, set)]
    pub raise_on_non_convergence: Option<bool>,
}

crate::shared::pickle::impl_pickle_state!(IVOptions {
    estimator,
    cov_type,
    include_intercept,
    confidence_level,
    cluster,
    hac_lags,
    hac_time,
    gmm_type,
    gmm_weight_type,
    gmm_max_iter,
    gmm_tol,
    raise_on_non_convergence,
});

impl IVOptions {
    /// このオプションが指名するデータの列（`cluster`・`hac_time`）。使われるかどうか（`cov_type`等）には
    /// よらず、指名されたものを全て返す。`fit()`はこれらと`y`・`x`等だけをデータから取り込む。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().chain(&self.hac_time).cloned().collect()
    }
}

#[pymethods]
impl IVOptions {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    #[new]
    #[pyo3(signature = (
        *,
        estimator = "2sls".to_string(),
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster = None,
        hac_lags = None,
        hac_time = None,
        gmm_type = None,
        gmm_weight_type = None,
        gmm_max_iter = None,
        gmm_tol = None,
        raise_on_non_convergence = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::estimator_arg)] estimator: String,
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        include_intercept: bool,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::hac_lags_arg)] hac_lags: Option<i64>,
        #[pyo3(from_py_with = crate::shared::option_values::hac_time_arg)] hac_time: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_type_arg)] gmm_type: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_weight_type_arg)]
        gmm_weight_type: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_max_iter_arg)] gmm_max_iter: Option<
            i64,
        >,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_tol_arg)] gmm_tol: Option<f64>,
        raise_on_non_convergence: Option<bool>,
    ) -> Self {
        Self {
            estimator,
            cov_type,
            include_intercept,
            confidence_level,
            cluster,
            hac_lags,
            hac_time,
            gmm_type,
            gmm_weight_type,
            gmm_max_iter,
            gmm_tol,
            raise_on_non_convergence,
        }
    }

    #[setter]
    fn set_confidence_level(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.confidence_level = extract_strict_float(value, "confidence_level")?;
        Ok(())
    }

    #[setter]
    fn set_hac_lags(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.hac_lags = extract_strict_opt_int(value, "hac_lags")?;
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

    #[setter]
    fn set_estimator(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.estimator = extract_strict_text(value, "estimator")?;
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
    fn set_hac_time(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.hac_time = extract_strict_opt_column(value, "hac_time")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_weight_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_weight_type = extract_strict_opt_text(value, "gmm_weight_type")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_type = extract_strict_opt_text(value, "gmm_type")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "IVOptions(estimator={}, cov_type={}, include_intercept={}, \
             confidence_level={}, cluster={}, hac_lags={}, hac_time={}, \
             gmm_type={}, gmm_weight_type={}, gmm_max_iter={}, gmm_tol={}, \
             raise_on_non_convergence={})",
            self.estimator.py_repr(),
            self.cov_type.py_repr(),
            self.include_intercept.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.hac_lags.py_repr(),
            self.hac_time.py_repr(),
            self.gmm_type.py_repr(),
            self.gmm_weight_type.py_repr(),
            self.gmm_max_iter.py_repr(),
            self.gmm_tol.py_repr(),
            self.raise_on_non_convergence.py_repr(),
        )
    }
}

/// `build_iv_input`などのテストで使う既定の`IVOptions`（`estimator="2sls"`・
/// `cov_type="classical"`・`include_intercept=true`）。フィールドごとに上書きして使う。
#[cfg(test)]
pub(super) fn default_options() -> IVOptions {
    IVOptions::new(
        "2sls".to_string(),
        "classical".to_string(),
        true,
        0.95,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}
