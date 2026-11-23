//! `OLSOptions`（OLSの推定オプション）。

use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_opt_column, extract_strict_opt_int, extract_strict_text,
};
use pyo3::prelude::*;

/// Estimation options for OLS.
///
/// All arguments are keyword-only.
///
/// Options by purpose:
///
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`; with
///   `"hac"` also `hac_time` (required) and `hac_lags`.
/// - Confidence intervals: `confidence_level`.
/// - Model: `include_intercept`.
// 各フィールドの意味・既定値の根拠は`docs/spec/ols-spec.md`「API引数」。
// `fit_ols`がPython側から`OLSOptions`インスタンスを引数として受け取るため、
// `FromPyObject`実装を明示的に維持する（pyo3 0.28以降、Cloneを実装する#[pyclass]の
// FromPyObject自動導出はopt-inに変更されたため）。
// module: PyO3の#[pyclass]はデフォルトで__module__="builtins"になり、
// mkdocstrings（griffe）がPythonでの再エクスポートのalias解決に失敗する原因になる。
// 実際のインポート元(`econometricsmodels._lib`)を明示する。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct OLSOptions {
    /// Standard error type: one of "classical" (default), "hc0", "hc1", "hc2", "hc3",
    /// "hac", "cluster". Case-insensitive.
    #[pyo3(get)]
    pub cov_type: String,

    /// Whether to automatically add an intercept column.
    /// When true, a column of all 1.0 is prepended to the design matrix.
    /// If the user's `x` already contains a constant column while this is true,
    /// the resulting perfect collinearity raises `ComputationError` (singular matrix).
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval). This is the coverage, not the
    /// significance level `alpha`.
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key. Required when `cov_type="cluster"`
    /// (omitting it raises `ValidationError`); specifying it with any other `cov_type`
    /// also raises `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Number of lags (bandwidth) for HAC (Newey-West) when `cov_type="hac"`.
    /// When `None`, computed automatically via `L = floor(4*(n/100)^(2/9))`.
    /// Specifying it with any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub hac_lags: Option<i64>,

    /// Column name giving the time order for HAC. Required when
    /// `cov_type="hac"` (omitting it raises `ValidationError`): the row order
    /// of `data` is never assumed to be the time order. Specifying it with
    /// any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub hac_time: Option<String>,
}

crate::shared::pickle::impl_pickle_state!(OLSOptions {
    cov_type,
    include_intercept,
    confidence_level,
    cluster,
    hac_lags,
    hac_time,
});

impl OLSOptions {
    /// このオプションが指名するデータの列（`cluster`・`hac_time`）。使われるかどうか（`cov_type`）には
    /// よらず、指名されたものを全て返す。`fit()`はこれらと`y`・`x`だけをデータから取り込む。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().chain(&self.hac_time).cloned().collect()
    }
}

#[pymethods]
impl OLSOptions {
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
        hac_lags = None,
        hac_time = None,
    ))]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        include_intercept: bool,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::hac_lags_arg)] hac_lags: Option<i64>,
        #[pyo3(from_py_with = crate::shared::option_values::hac_time_arg)] hac_time: Option<String>,
    ) -> Self {
        Self {
            cov_type,
            include_intercept,
            confidence_level,
            cluster,
            hac_lags,
            hac_time,
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

    fn __repr__(&self) -> String {
        format!(
            "OLSOptions(cov_type={}, include_intercept={}, confidence_level={}, \
             cluster={}, hac_lags={}, hac_time={})",
            self.cov_type.py_repr(),
            self.include_intercept.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.hac_lags.py_repr(),
            self.hac_time.py_repr()
        )
    }
}
