//! FEIV（固定効果つきの2SLS）の推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::panel::fe_iv::FeIvEstimator`を呼び出すところまでの一連の処理（データ抽出・pyclass定義、
//! `FeIvEstimator::fit_with_absorption`への配線、`first_stage()`・`fixed_effects()`・弱操作変数診断のメソッド）。
//!
//! 【責務分離・言語方針】`panel/fe.rs`と同じ（`.claude/rules/rust-style.md`）。公開API（`FEIVOptions`/
//! `FEIVResult`）のdocコメントと`ValidationError`のメッセージは英語、それ以外は日本語。
//!
//! ## 列の役割
//!
//! `y`・`x_exog`・`x_endog`・`instruments`・`effects`。`x_exog`は空でよい（固定効果が切片を吸収するので、
//! 外生変数が無いモデルも成り立つ）。`x_endog`・`instruments`は空を許さない。`effects`・`cluster`・`dk_time`は
//! FEと同じ規約（`panel/fe.rs`のモジュールdoc）。
//!
//! ## 診断・`fixed_effects()`は元のデータから再計算する
//!
//! 結果は`fit()`に渡した`DataFrame`（`source_data`）と`FEIVOptions`・列名を保持し、`cragg_donald()`・
//! `kleibergen_paap()`・`effective_f()`・`fixed_effects()`は呼ばれたときに`fit()`と同じ手順で`FeIvEstimator`を
//! 作り直して呼ぶ（IVの診断と同じ。推定量（設計行列を持つ）を結果に持たせない）。第一段階の結果は
//! `first_stage()`が`FEResult`で返す。固定効果の辞書は第一段階ごとに`fit()`で計算して持つ（個体数に比例する小さな
//! もの）ので、`FeEstimator`を保持しない。

use crate::iv::common::{CraggDonaldOutput, EffectiveFOutput, KleibergenPaapOutput};
use crate::shared::errors::{guard, internal_error};
use engine::panel::fe::{FeAbsorption, FeCovType, FeEffects};
use engine::panel::fe_iv::{FeIvEstimator, FeIvInput, RoleColumns};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use super::common::{
    PanelModel, check_panel_cov_options, extract_effect_groups, groups_dict, panel_error_to_pyerr,
    parse_panel_cov_kind, validate_fe_iv_roles,
};
use super::fe::{
    CovColumns, FEResult, LazyFixedEffects, fe_result_fields, fixed_effects_dict,
    resolve_fe_cov_type,
};
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_f64_column, extract_f64_columns, select_columns,
};
use crate::shared::mat_conversion::{mat_to_rows, mat_to_vec};
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_int, extract_strict_opt_column, extract_strict_opt_int,
    extract_strict_text,
};

/// Estimation options for FEIV (fixed effects instrumental variables, two-stage least squares).
///
/// All arguments are keyword-only.
///
/// Options by purpose:
///
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`; with
///   `"dk"` also `dk_time` (required) and `dk_bandwidth`.
/// - Confidence intervals: `confidence_level`.
/// - Removing the fixed effects of an unbalanced two-way model or of a model with
///   three or more effects: `tol`, `max_iter`.
// 各フィールドの意味・既定値の根拠は`FEOptions`と同じ（`panel/fe.rs`）。FEIVはGMMを持たないので、
// `estimator`・`gmm_*`は今は持たない（GMMを足すときにキーワード引数として足す）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct FEIVOptions {
    /// Standard error type: one of "classical", "hc1", "hc2", "hc3", "cluster", "dk".
    /// Case-insensitive. Unlike OLS/WLS/IV, "hc0" is **not** supported. The first-stage
    /// regressions and the diagnostics that depend on a covariance (`kleibergen_paap()` and
    /// `effective_f()`) follow it.
    #[pyo3(get)]
    pub cov_type: String,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`. When
    /// `None`, the first column of `effects` is used automatically. Specifying it with
    /// any other `cov_type` raises `ValidationError`.
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

    /// Convergence tolerance of the alternating projections that remove the fixed
    /// effects of a two-way model on an **unbalanced** panel, or of a model with three or
    /// more effects (a balanced two-way panel and a one-way model are computed in closed
    /// form, so they ignore it). A pass stops when the largest group mean it subtracts is at
    /// most `tol` times the size of the column. Must be a positive finite number. Defaults
    /// to 1e-12.
    #[pyo3(get)]
    pub tol: f64,

    /// Maximum number of passes of the alternating projections (see `tol`), from 1 to
    /// 10000. Defaults to 10000. If the limit is reached, `ComputationError` is raised.
    #[pyo3(get)]
    pub max_iter: i64,
}

crate::shared::pickle::impl_pickle_state!(FEIVOptions {
    cov_type,
    confidence_level,
    cluster,
    dk_time,
    dk_bandwidth,
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
        cov_type = "cluster".to_string(),
        confidence_level = 0.95,
        cluster = None,
        dk_time = None,
        dk_bandwidth = None,
        tol = None,
        max_iter = 10000,
    ))]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_time_arg)] dk_time: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_bandwidth_arg)] dk_bandwidth: Option<
            i64,
        >,
        #[pyo3(from_py_with = crate::shared::option_values::tol_arg)] tol: Option<f64>,
        #[pyo3(from_py_with = crate::shared::option_values::max_iter_arg)] max_iter: i64,
    ) -> Self {
        let defaults = FeAbsorption::default();
        Self {
            cov_type,
            confidence_level,
            cluster,
            dk_time,
            dk_bandwidth,
            tol: tol.unwrap_or(defaults.tol),
            max_iter,
        }
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

    fn __repr__(&self) -> String {
        format!(
            "FEIVOptions(cov_type={}, confidence_level={}, cluster={}, \
             dk_time={}, dk_bandwidth={}, tol={}, max_iter={})",
            self.cov_type.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.dk_time.py_repr(),
            self.dk_bandwidth.py_repr(),
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

// 第一段階の結果（`FEResult`）は、それぞれの`__reduce__`でpickleされるオブジェクトとして状態に入れる。
impl crate::shared::pickle::PickleField for Vec<(String, FEResult)> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObjectExt;
        let items = self
            .iter()
            .map(|(name, result)| Ok((name.clone(), Py::new(py, result.clone())?)))
            .collect::<PyResult<Vec<_>>>()?;
        items.into_bound_py_any(py)
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let items: Vec<(String, Bound<'_, PyAny>)> = obj.extract().ok()?;
        items
            .into_iter()
            .map(|(name, item)| {
                let result = item.cast::<FEResult>().ok()?.try_borrow().ok()?;
                Some((name, result.clone()))
            })
            .collect()
    }
}

/// Estimation results for FEIV (fixed effects instrumental variables, two-stage least squares).
///
/// Structured data only (no `summary()`). All array-valued fields (`params`,
/// `std_errors`, etc.) share the same order as `param_names` (`x_exog`, then `x_endog`).
///
/// `test_stats` holds the t-statistics; `stat_df` is their degrees of freedom. `r_squared_*`
/// are computed from the structural residuals (`y - X b` with the actual endogenous
/// regressors `X`) and can be negative.
///
/// The first-stage regressions, the fixed effects themselves and the Cragg-Donald,
/// Kleibergen-Paap and effective F diagnostics are not fields: they are the methods
/// `first_stage()`, `fixed_effects()`, `cragg_donald()`, `kleibergen_paap()` and
/// `effective_f()`.
// `Clone`を派生しない: 第一段階の`FEResult`は複製できるが、`source_data`（`Py<PyAny>`）が
// GILなしでは複製できない（`IVResult`と同じ理由）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct FEIVResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    /// Covariance matrix of the coefficients under `cov_type`, one row per entry of
    /// `param_names` (same order on both axes).
    #[pyo3(get)]
    pub cov_params: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub test_stats: Vec<f64>,
    /// Distribution of `test_stats`; always `"t"` for FEIV with two-stage least squares.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the t distribution. May differ from `df_resid`: cluster-robust
    /// inference uses `G - 1`, where `G` is the number of clusters, and Driscoll-Kraay uses
    /// the number of `dk_time` periods minus 1.
    #[pyo3(get)]
    pub stat_df: Option<i64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    #[pyo3(get)]
    pub conf_upper: Vec<f64>,
    #[pyo3(get)]
    pub param_names: Vec<String>,
    /// Structural residuals `y - X b` of the within-transformed data, with the actual
    /// endogenous regressors `X`.
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub dep_var_name: String,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub df_resid: usize,
    #[pyo3(get)]
    pub df_model: usize,
    /// The columns of `effects`, in the order they were given.
    effects: Vec<String>,
    /// Number of groups (unique values) of each column of `effects`, in the same order.
    group_counts: Vec<usize>,
    /// Standard error type actually used (echoes `FEIVOptions.cov_type`, normalized to lowercase).
    #[pyo3(get)]
    pub cov_type: String,
    /// Driscoll-Kraay bandwidth actually used. `None` unless `cov_type="dk"`.
    #[pyo3(get)]
    pub dk_bandwidth_used: Option<i64>,
    /// Statistic of the joint test that all slope coefficients are zero.
    #[pyo3(get)]
    pub wald_statistic: f64,
    #[pyo3(get)]
    pub wald_p_value: f64,
    /// Distribution of `wald_statistic`; always `"f"` for two-stage least squares.
    #[pyo3(get)]
    pub wald_dist: String,
    #[pyo3(get)]
    pub wald_df_num: Option<usize>,
    /// Follows `cov_type` like the t-tests, so it equals `stat_df`.
    #[pyo3(get)]
    pub wald_df_denom: Option<usize>,
    #[pyo3(get)]
    pub r_squared_within: f64,
    #[pyo3(get)]
    pub r_squared_between: f64,
    #[pyo3(get)]
    pub r_squared_overall: f64,
    /// Weak-instrument partial F statistic of each endogenous variable, keyed by its name.
    /// Homoskedastic whatever `cov_type` is; its denominator degrees of freedom are
    /// `n - l - n_absorbed`, where `l` is the number of exogenous variables plus instruments and
    /// `n_absorbed` the number of absorbed fixed-effect parameters.
    #[pyo3(get)]
    pub weak_instrument_f_statistics: HashMap<String, f64>,
    /// Numerator degrees of freedom of the weak-instrument F statistics (the number of excluded
    /// instruments).
    #[pyo3(get)]
    pub weak_instrument_f_df_num: Option<usize>,
    /// Denominator degrees of freedom of the weak-instrument F statistics.
    #[pyo3(get)]
    pub weak_instrument_f_df_denom: Option<usize>,
    /// Sargan overidentification statistic, computed from the within-transformed data with
    /// `e'e / (n - n_absorbed)` as the variance. `None` when the model is exactly identified.
    #[pyo3(get)]
    pub overid_statistic: Option<f64>,
    #[pyo3(get)]
    pub overid_p_value: Option<f64>,
    #[pyo3(get)]
    pub overid_df: Option<usize>,
    /// Wu-Hausman endogeneity test (the regression form, an auxiliary fixed-effects regression
    /// with the first-stage residuals added; follows `cov_type`). `None` when the auxiliary
    /// regression cannot be estimated.
    #[pyo3(get)]
    pub wu_hausman_statistic: Option<f64>,
    #[pyo3(get)]
    pub wu_hausman_p_value: Option<f64>,
    #[pyo3(get)]
    pub wu_hausman_df_num: Option<usize>,
    #[pyo3(get)]
    pub wu_hausman_df_denom: Option<usize>,
    // `first_stage()`が読む。内生変数ごとの第一段階（固定効果は計算済み）。Python側には公開しない。
    first_stage: Vec<(String, FEResult)>,
    // 診断・`fixed_effects()`が`FeIvEstimator`を作り直すために保持する（`IVResult`と同じ）。`source_data`は
    // `fit()`に渡された`polars.DataFrame`（参照を増やすだけ）、`options`は渡した`FEIVOptions`の複製。
    // `source_data`と`options`は`fit`が結果を組み立てた後でGILを持って付ける。
    source_data: Option<Py<PyAny>>,
    options: Option<FEIVOptions>,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
}

crate::shared::pickle::impl_pickle_state!(FEIVResult {
    params,
    std_errors,
    cov_params,
    test_stats,
    stat_dist,
    stat_df,
    p_values,
    conf_lower,
    conf_upper,
    param_names,
    residuals,
    dep_var_name,
    n_obs,
    df_resid,
    df_model,
    effects,
    group_counts,
    cov_type,
    dk_bandwidth_used,
    wald_statistic,
    wald_p_value,
    wald_dist,
    wald_df_num,
    wald_df_denom,
    r_squared_within,
    r_squared_between,
    r_squared_overall,
    weak_instrument_f_statistics,
    weak_instrument_f_df_num,
    weak_instrument_f_df_denom,
    overid_statistic,
    overid_p_value,
    overid_df,
    wu_hausman_statistic,
    wu_hausman_p_value,
    wu_hausman_df_num,
    wu_hausman_df_denom,
    first_stage,
    source_data,
    options,
    x_exog,
    x_endog,
    instruments,
});

impl FEIVResult {
    /// 元のデータ（`fit()`に渡された`polars.DataFrame`）。`fit`が必ず付けるので`None`にはならない
    /// （pickleにも含まれる）。`None`なら到達不能な状態として`INTERNAL_ERROR`。
    fn retained_source<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.source_data
            .as_ref()
            .map(|source| source.bind(py).clone())
            .ok_or_else(|| internal_error("an FEIVResult without its training data"))
    }

    /// `fit()`に渡した`FEIVOptions`の複製。`retained_source`と同じく`None`にはならない。
    fn retained_options(&self) -> PyResult<FEIVOptions> {
        self.options
            .clone()
            .ok_or_else(|| internal_error("an FEIVResult without its options"))
    }

    /// `fit()`と同じ手順で`FeIvEstimator`を作り直し、`compute`に渡す。元のデータの取り込み（GILが要る）を
    /// 済ませた後は、GILを手放して走る。
    fn with_estimator<T: Send>(
        &self,
        py: Python<'_>,
        compute: impl FnOnce(&FeIvEstimator) -> PyResult<T> + Send,
    ) -> PyResult<T> {
        let source = self.retained_source(py)?;
        let options = self.retained_options()?;
        let columns = FeIvColumns {
            y: self.dep_var_name.clone(),
            x_exog: self.x_exog.clone(),
            x_endog: self.x_endog.clone(),
            instruments: self.instruments.clone(),
            effects: self.effects.clone(),
        };
        let frame = select_columns(&source, "data", &columns.data_columns(&options))?.frame;
        py.detach(|| {
            let estimator = estimate(&frame, columns, &options)?;
            compute(&estimator)
        })
    }
}

#[pymethods]
impl FEIVResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Number of groups (unique values) of each column of `effects`, as a `dict[str, int]`
    /// keyed by column name, in the order the columns were given.
    #[getter]
    fn n_groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        groups_dict(py, &self.effects, &self.group_counts)
    }

    /// Per-endogenous-variable first-stage regression results
    /// (`x_endog[i] ~ x_exog + instruments`, with the same fixed effects and `cov_type`),
    /// keyed by the endogenous variable name.
    ///
    /// Each value is a full `FEResults` (the same type FE's `fit_fe` returns). Its
    /// `f_statistic` is `None` (the overall test is not computed for a first stage) and its
    /// `fixed_effects()` is available. The weak-instrument partial F statistic is
    /// `weak_instrument_f_statistics`, computed separately.
    fn first_stage(&self) -> PyResult<HashMap<String, FEResult>> {
        guard(|| {
            Ok(self
                .first_stage
                .iter()
                .map(|(name, result)| (name.clone(), result.clone()))
                .collect())
        })
    }

    /// The fixed effects themselves, recovered post-hoc from the fitted coefficients
    /// (`α_i = ȳ_i - x̄_i'β̂` with the actual endogenous regressors).
    ///
    /// Always a `dict[str, dict[str, float]]` keyed by the column names in `effects`, then by
    /// group. The normalization is the same as `FEResults.fixed_effects()`. It re-reads the
    /// data passed to `fit()` and estimates the model again on every call, so it costs about as
    /// much as `fit()`.
    fn fixed_effects(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        guard(|| {
            let fixed = self.with_estimator(py, |estimator| Ok(estimator.fixed_effects()))?;
            fixed_effects_dict(py, &self.effects, &fixed)
        })
    }

    /// Cragg-Donald statistic for the joint strength of the instruments, with the Stock-Yogo
    /// critical values where they are tabulated.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`. It re-reads the variables from
    /// the data passed to `fit()` and estimates the model again on every call. Homoskedastic, so
    /// it does not depend on `cov_type`; the
    /// absorbed fixed-effect parameters are subtracted from its degrees of freedom. With one
    /// endogenous variable it equals the partial F-statistic in `weak_instrument_f_statistics`.
    ///
    /// Raises
    /// ------
    /// ComputationError
    ///     The statistic is undefined for these data (collinear instruments, a perfectly
    ///     fitted first stage, ...).
    fn cragg_donald(&self, py: Python<'_>) -> PyResult<CraggDonaldOutput> {
        guard(|| {
            self.with_estimator(py, |estimator| {
                let cd = estimator.cragg_donald().map_err(panel_error_to_pyerr)?;
                Ok(CraggDonaldOutput::from_engine(cd))
            })
        })
    }

    /// Kleibergen-Paap rk LM and rk Wald F statistics, robust to the `cov_type` of the fit.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`; it estimates the model again
    /// on every call. The covariance of the first-stage scores follows `cov_type`, `cluster`,
    /// `dk_time` and `dk_bandwidth` of the options passed to `fit()`. Either statistic may be
    /// `None` when only the other is defined.
    ///
    /// Raises
    /// ------
    /// ComputationError
    ///     Neither statistic is defined for these data.
    fn kleibergen_paap(&self, py: Python<'_>) -> PyResult<KleibergenPaapOutput> {
        guard(|| {
            self.with_estimator(py, |estimator| {
                let kp = estimator.kleibergen_paap().map_err(panel_error_to_pyerr)?;
                Ok(KleibergenPaapOutput::from_engine(kp))
            })
        })
    }

    /// Montiel Olea-Pflueger effective F statistic and its critical values.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`; it estimates the model again
    /// on every call. Defined for one endogenous variable; the covariance follows the options
    /// passed to `fit()`.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     `alpha` is not a real number.
    /// ValidationError
    ///     `alpha` is not strictly between 0 and 1, or there is not exactly one endogenous
    ///     variable.
    /// ComputationError
    ///     The statistic is undefined for these data.
    fn effective_f(&self, py: Python<'_>, alpha: &Bound<'_, PyAny>) -> PyResult<EffectiveFOutput> {
        guard(|| {
            let alpha = extract_strict_float(alpha, "alpha")?;
            self.with_estimator(py, |estimator| {
                let eff = estimator.effective_f(alpha).map_err(panel_error_to_pyerr)?;
                Ok(EffectiveFOutput::from_engine(eff))
            })
        })
    }
}

/// `fit()`に渡された列の役割（`y`・`x_exog`・`x_endog`・`instruments`・`effects`の列名）。
#[derive(Debug, Clone)]
pub(crate) struct FeIvColumns {
    pub(crate) y: String,
    pub(crate) x_exog: Vec<String>,
    pub(crate) x_endog: Vec<String>,
    pub(crate) instruments: Vec<String>,
    pub(crate) effects: Vec<String>,
}

impl FeIvColumns {
    /// データから取り込む役割の列（`y`・`effects`・`x_exog`・`x_endog`・`instruments`）。FEIVと`FEGMM`が共有する。
    pub(crate) fn role_columns(&self) -> Vec<String> {
        let mut columns = vec![self.y.clone()];
        columns.extend(self.effects.iter().cloned());
        columns.extend(self.x_exog.iter().cloned());
        columns.extend(self.x_endog.iter().cloned());
        columns.extend(self.instruments.iter().cloned());
        columns
    }

    /// データから取り込む列（役割の列と、`options`が指名する列）。
    pub(crate) fn data_columns(&self, options: &FEIVOptions) -> Vec<String> {
        let mut columns = self.role_columns();
        columns.extend(options.data_columns());
        columns
    }
}

/// [`build_fe_iv_input`]の戻り値。
#[derive(Debug)]
pub(crate) struct BuiltFeIvInput {
    pub(crate) input: FeIvInput,
    pub(crate) effects: FeEffects,
    /// エンジンに渡す`cov_type`（`cluster`のグループや`dk`の時点を持つ）。
    pub(crate) cov_type: FeCovType,
    /// `FEIVResult.cov_type`に格納する正規形（小文字）の名前。
    pub(crate) cov_type_name: String,
}

/// Pythonから渡された`data`・列名・`options`を検証し、`engine::panel::fe_iv::FeIvInput`を組み立てる。
///
/// 報告順は`build_fe_input`と同じ: オプションの値 → オプションの組合せ → ロール構造 → 列（`y`・`x_exog`・
/// `x_endog`・`instruments`・`effects`）→ `cov_type`の列 → エンジン。
///
/// # Errors
/// - オプションの値・組合せ、ロール構造（`x_endog`・`instruments`が空、役割間・役割内の重複、`effects`の
///   列数、`dk_time`の役割）、列の抽出時に発覚する問題は`ValidationError`
/// - それ以外（行数の不一致等）は`PanelError`から`panel_error_to_pyerr`で変換
pub(crate) fn build_fe_iv_input(
    df: &DataFrame,
    columns: FeIvColumns,
    options: &FEIVOptions,
) -> PyResult<BuiltFeIvInput> {
    let FeIvColumns {
        y,
        x_exog,
        x_endog,
        instruments,
        effects,
    } = columns;
    let cov_kind = parse_panel_cov_kind(&options.cov_type, PanelModel::FeIv)?;
    check_panel_cov_options(
        cov_kind,
        options.cluster.as_deref(),
        options.dk_time.as_deref(),
        options.dk_bandwidth,
    )?;
    validate_fe_iv_roles(
        PanelModel::FeIv,
        &y,
        &x_exog,
        &x_endog,
        &instruments,
        &effects,
        options.dk_time.as_deref(),
    )?;

    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;
    let exog_slices = extract_f64_columns(df, &x_exog, ColumnSource::new("x_exog", "data"))?;
    let endog_slices = extract_f64_columns(df, &x_endog, ColumnSource::new("x_endog", "data"))?;
    let instrument_slices =
        extract_f64_columns(df, &instruments, ColumnSource::new("instruments", "data"))?;

    let groups = extract_effect_groups(df, &effects)?;
    let fe_effects = match groups.len() {
        1 => FeEffects::OneWay,
        2 => FeEffects::TwoWay,
        _ => FeEffects::MultiWay,
    };
    let cov_type = resolve_fe_cov_type(
        df,
        cov_kind,
        &CovColumns {
            cluster: options.cluster.as_deref(),
            dk_time: options.dk_time.as_deref(),
            dk_bandwidth: options.dk_bandwidth,
        },
    )?;

    let input = FeIvInput::from_group_codes(
        &y_slice,
        y,
        RoleColumns {
            names: x_exog,
            columns: &exog_slices,
        },
        RoleColumns {
            names: x_endog,
            columns: &endog_slices,
        },
        RoleColumns {
            names: instruments,
            columns: &instrument_slices,
        },
        groups,
    )
    .map_err(panel_error_to_pyerr)?
    .with_effect_names(effects);

    Ok(BuiltFeIvInput {
        input,
        effects: fe_effects,
        cov_type,
        cov_type_name: cov_kind.name().to_string(),
    })
}

/// 入力を組み立てて`FeIvEstimator`を推定する（`fit`と診断・`fixed_effects()`の再計算が共有する）。
fn estimate(
    df: &DataFrame,
    columns: FeIvColumns,
    options: &FEIVOptions,
) -> PyResult<FeIvEstimator> {
    estimate_built(build_fe_iv_input(df, columns, options)?, options)
}

fn estimate_built(built: BuiltFeIvInput, options: &FEIVOptions) -> PyResult<FeIvEstimator> {
    let BuiltFeIvInput {
        input,
        effects,
        cov_type,
        ..
    } = built;
    let absorption = FeAbsorption {
        tol: options.tol,
        max_iter: options.max_iter,
    };
    FeIvEstimator::fit_with_absorption(
        input,
        effects,
        cov_type,
        options.confidence_level,
        &absorption,
    )
    .map_err(panel_error_to_pyerr)
}

/// Pythonから渡された`data`・列名・`options`を検証し、`FeIvEstimator`を推定して`FEIVResult`として返す。
///
/// # Errors
/// - `build_fe_iv_input`が返すエラー（`ValidationError`）
/// - `FeIvEstimator::fit_with_absorption`が返す`PanelError`（識別の順序条件・singleton・自由度不足・
///   分散ゼロ・第一段階の失敗・構造式の最小二乗・完全適合・共分散の検証等）は`panel_error_to_pyerr`で変換
pub(crate) fn fit(
    py: Python<'_>,
    data: SelectedData,
    columns: FeIvColumns,
    options: &FEIVOptions,
) -> PyResult<FEIVResult> {
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる。`source`は結果に持たせる。
    let SelectedData { source, frame } = data;
    let mut result = py.detach(|| fit_frame(&frame, columns, options))?;
    // 診断・`fixed_effects()`が`FeIvEstimator`を作り直すための元のデータとオプション（GILを持って付ける）。
    result.source_data = Some(source);
    result.options = Some(options.clone());
    Ok(result)
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。
fn fit_frame(df: &DataFrame, columns: FeIvColumns, options: &FEIVOptions) -> PyResult<FEIVResult> {
    let built = build_fe_iv_input(df, columns.clone(), options)?;
    let cov_type_name = built.cov_type_name.clone();
    let estimator = estimate_built(built, options)?;
    result_from(&estimator, columns, cov_type_name)
}

fn result_from(
    estimator: &FeIvEstimator,
    columns: FeIvColumns,
    cov_type_name: String,
) -> PyResult<FEIVResult> {
    let FeIvColumns {
        x_exog,
        x_endog,
        instruments,
        effects,
        ..
    } = columns;
    let input = estimator.input();
    // 第一段階は固定効果をその場で計算して持つ（`FeEstimator`を結果に持たせない）。
    let first_stage: Vec<(String, FEResult)> = input
        .x_endog_names()
        .iter()
        .zip(estimator.first_stages())
        .map(|(name, stage)| {
            let fixed = LazyFixedEffects::Computed(stage.fixed_effects());
            (
                name.clone(),
                fe_result_fields(stage, effects.clone(), cov_type_name.clone(), fixed),
            )
        })
        .collect();
    let (f_statistic, f_p_value) = estimator
        .f_statistic()
        .zip(estimator.f_p_value())
        .ok_or_else(|| internal_error("the joint test of a FEIV fit is missing"))?;
    let (wald_df_num, wald_df_denom) = estimator.f_df().unzip();
    let weak_df = estimator.weak_instrument_f_df();
    let wu_hausman_df = estimator.wu_hausman_df();

    Ok(FEIVResult {
        params: mat_to_vec(estimator.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        cov_params: mat_to_rows(estimator.cov_params()),
        test_stats: mat_to_vec(estimator.test_stats()),
        stat_dist: "t".to_string(),
        stat_df: Some(estimator.df_inference() as i64),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.param_names().to_vec(),
        residuals: mat_to_vec(estimator.residuals()),
        dep_var_name: input.dep_var_name().to_string(),
        n_obs: input.nobs(),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        effects,
        group_counts: estimator.n_groups().to_vec(),
        cov_type: cov_type_name,
        dk_bandwidth_used: estimator.dk_bandwidth_used().map(|bw| bw as i64),
        wald_statistic: f_statistic,
        wald_p_value: f_p_value,
        wald_dist: "f".to_string(),
        wald_df_num,
        wald_df_denom,
        r_squared_within: estimator.r_squared_within(),
        r_squared_between: estimator.r_squared_between(),
        r_squared_overall: estimator.r_squared_overall(),
        weak_instrument_f_statistics: estimator
            .weak_instrument_f_statistics()
            .iter()
            .cloned()
            .collect(),
        weak_instrument_f_df_num: weak_df.map(|(num, _)| num),
        weak_instrument_f_df_denom: weak_df.map(|(_, denom)| denom),
        overid_statistic: estimator.sargan_statistic(),
        overid_p_value: estimator.sargan_p_value(),
        overid_df: estimator.sargan_df(),
        wu_hausman_statistic: estimator.wu_hausman_statistic(),
        wu_hausman_p_value: estimator.wu_hausman_p_value(),
        wu_hausman_df_num: wu_hausman_df.map(|(num, _)| num),
        wu_hausman_df_denom: wu_hausman_df.map(|(_, denom)| denom),
        first_stage,
        source_data: None,
        options: None,
        x_exog,
        x_endog,
        instruments,
    })
}
