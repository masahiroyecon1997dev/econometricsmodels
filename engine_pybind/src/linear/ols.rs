//! OLSの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::linear::ols`（正規方程式ソルバー・標準誤差・適合度統計量）を呼び出し、
//! 結果をPython側に返すところまでの一連の処理。
//!
//! 【責務分離】`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照。
//! polars DataFrameから列ごとの`Vec<f64>`/`Vec<String>`への抽出はここ（`column_extraction`
//! 経由）の責務。`faer::Mat`の組み立て（切片列の自動追加を含む）は`engine::linear::ols::OlsInput`
//! に委ねる（本ファイルはもう`faer`を直接扱わない）。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`OLSOptions`/`OLSResult`）のdocコメントと、`ValidationError`のメッセージ文字列は英語。
//! それ以外（このファイルの説明・非公開関数のdocコメント等）は日本語のまま。

use engine::linear::diagnostics::{
    AuxRegressionTest, breusch_godfrey_test, breusch_pagan_test, white_test,
};
use engine::linear::ols::{OlsEstimator, OlsInput};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::common::{
    check_cov_options, least_squares_error_to_pyerr, parse_cov_kind, resolve_cov_type,
};
use crate::shared::augment::append_predicted_column;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_column_list, extract_f64_column, extract_f64_columns,
    extract_model_x_columns, extract_time_order_ranks, select_columns, select_model_x_columns,
    x_column_names,
};
use crate::shared::errors::{guard, unknown_option_value, user_error};
use crate::shared::mat_conversion::mat_to_vec;
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_int, extract_strict_opt_column, extract_strict_opt_int,
    extract_strict_text,
};
use crate::shared::validation::{
    validate_common_roles, validate_no_duplicate_within_role, validate_x_non_empty,
};
use engine::shared::error_catalog::{ErrorCode, ParamValue};

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
    /// Standard error type: one of "classical", "hc0", "hc1", "hc2", "hc3", "hac", "cluster".
    /// Case-insensitive.
    #[pyo3(get)]
    pub cov_type: String,

    /// Whether the engine should automatically add an intercept column.
    /// When true, a column of all 1.0 is prepended to the design matrix.
    /// If the user's `x` already contains a constant column while this is true,
    /// the resulting perfect collinearity raises `ComputationError` (singular matrix).
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval). Named `confidence_level` rather
    /// than `alpha` to avoid confusion with the significance level (the 0.05 side).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`.
    /// Refers to a column in `data` rather than being passed as a separate array.
    /// Specifying it with any other `cov_type` raises `ValidationError`.
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

impl OLSOptions {
    /// このオプションが指名するデータの列（`cluster`・`hac_time`）。使われるかどうか（`cov_type`）には
    /// よらず、指名されたものを全て返す。`fit()`はこれらと`y`・`x`だけをデータから取り込む。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().chain(&self.hac_time).cloned().collect()
    }
}

#[pymethods]
impl OLSOptions {
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
    #[allow(clippy::too_many_arguments)]
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

/// Raw result of `OLSResult.white_test()`; the Python package wraps it in its
/// `WhiteTestResult` dataclass.
///
/// `statistic`/`p_value`/`df`/`df_denom`/`distribution` describe the selected version
/// (`"lm"` or `"f"`). `aux_terms` lists the auxiliary-regression terms actually used (the
/// first entry is always the auxiliary regression's constant, `"const"`), `dropped_terms`
/// the terms removed because they were constant or numerically identical to an earlier term.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct WhiteTestOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    /// Degrees of freedom of the LM test (chi-squared), or the numerator degrees of
    /// freedom of the F test.
    #[pyo3(get)]
    pub df: usize,
    /// Denominator degrees of freedom of the F test (`None` for the LM version).
    #[pyo3(get)]
    pub df_denom: Option<usize>,
    /// `"chi2"` for the LM version, `"f"` for the F version.
    #[pyo3(get)]
    pub distribution: String,
    #[pyo3(get)]
    pub aux_terms: Vec<String>,
    #[pyo3(get)]
    pub dropped_terms: Vec<String>,
}

/// Raw result of `OLSResult.breusch_pagan_test()`; the Python package wraps it in its
/// `BreuschPaganTestResult` dataclass.
///
/// `statistic`/`p_value`/`df`/`df_denom`/`distribution` describe the selected version
/// (`"lm"` or `"f"`). `aux_terms` lists the auxiliary-regression terms actually used (the
/// first entry is always the auxiliary regression's constant, `"const"`), `dropped_terms`
/// the variables removed because they were constant or numerically identical to an earlier
/// one.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct BreuschPaganTestOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    /// Degrees of freedom of the LM test (chi-squared), or the numerator degrees of
    /// freedom of the F test.
    #[pyo3(get)]
    pub df: usize,
    /// Denominator degrees of freedom of the F test (`None` for the LM version).
    #[pyo3(get)]
    pub df_denom: Option<usize>,
    /// `"chi2"` for the LM version, `"f"` for the F version.
    #[pyo3(get)]
    pub distribution: String,
    #[pyo3(get)]
    pub aux_terms: Vec<String>,
    #[pyo3(get)]
    pub dropped_terms: Vec<String>,
}

/// 診断検定（`white_test`・`breusch_pagan_test`・`breusch_godfrey_test`）の`statistic`引数で選ぶ検定統計量の版。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatisticVersion {
    Lm,
    F,
}

/// 診断検定の`statistic`引数（`"lm"`/`"f"`、大文字小文字を区別しない）をパースする。
///
/// # Errors
/// `statistic`が`"lm"`でも`"f"`でもない: `ValidationError`
fn parse_statistic_version(statistic: &str) -> PyResult<StatisticVersion> {
    match statistic.to_lowercase().as_str() {
        "lm" => Ok(StatisticVersion::Lm),
        "f" => Ok(StatisticVersion::F),
        _ => Err(unknown_option_value("statistic", statistic, &["lm", "f"])),
    }
}

/// 選んだ版（LM/F）の統計量・p値・分母自由度・分布名を取り出す（White検定・Breusch-Pagan検定
/// の結果の詰め替えで共通）。
fn select_statistic(
    result: &AuxRegressionTest,
    version: StatisticVersion,
) -> (f64, f64, Option<usize>, String) {
    match version {
        StatisticVersion::F => (
            result.f_statistic,
            result.f_p_value,
            Some(result.f_df_denom),
            "f".to_string(),
        ),
        StatisticVersion::Lm => (
            result.lm_statistic,
            result.lm_p_value,
            None,
            "chi2".to_string(),
        ),
    }
}

/// Raw result of `OLSResult.breusch_godfrey_test()`; the Python package wraps it in its
/// `BreuschGodfreyTestResult` dataclass.
///
/// `statistic`/`p_value`/`df`/`df_denom`/`distribution` describe the selected version
/// (`"lm"` or `"f"`); `nlags` is the number of residual lags used.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct BreuschGodfreyTestOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    /// Degrees of freedom of the LM test (chi-squared), or the numerator degrees of
    /// freedom of the F test. Equal to `nlags`.
    #[pyo3(get)]
    pub df: usize,
    /// Denominator degrees of freedom of the F test (`None` for the LM version).
    #[pyo3(get)]
    pub df_denom: Option<usize>,
    /// `"chi2"` for the LM version, `"f"` for the F version.
    #[pyo3(get)]
    pub distribution: String,
    #[pyo3(get)]
    pub nlags: usize,
}

/// Estimation results for OLS.
///
/// Structured data only (no `summary()`); see `docs/spec/ols-spec.md`
/// ("結果構造体"). Row-oriented table construction (e.g. a `params_table`) is left to
/// `python_package`. All array-valued fields (`params`, `std_errors`, etc.) share the
/// same order as `param_names`.
// `OLSResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`OLSOptions`の`from_py_object`とは
// 対照的。pyo3 0.28以降、Cloneを実装する#[pyclass]のFromPyObject自動導出はopt-inになった）。
//
// `get_all`（クラス単位で全フィールドに#[pyo3(get)]を付与する）ではなく、フィールドごとに
// 個別`#[pyo3(get)]`を付ける方式にしている。`fitted_values`（`predict(new_data=None)`が
// 返す値のキャッシュ、`docs/spec/ols-spec.md`「predict()」）はPython側に独立したプロパティとして
// 公開しない設計上の決定のため、この1フィールドだけ`#[pyo3(get)]`を付けずに残す必要がある。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct OLSResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub test_stats: Vec<f64>,
    /// Distribution of `test_stats`; always `"t"` for OLS.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the t distribution (always `Some` for OLS). May differ from
    /// `df_resid` (e.g. cluster-robust inference uses `G - 1`).
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
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub dep_var_name: String,
    #[pyo3(get)]
    pub n_obs: usize,
    /// Standard error type actually used (echoes `OLSOptions.cov_type`, normalized to
    /// lowercase; e.g. `"classical"`, `"hc1"`, `"hac"`, `"cluster"`).
    #[pyo3(get)]
    pub cov_type: String,
    /// Number of HAC (Newey-West) lags actually used: the explicit `hac_lags` if given,
    /// otherwise the value chosen automatically, `floor(4 * (n / 100) ^ (2 / 9))`.
    /// `None` unless `cov_type="hac"`.
    #[pyo3(get)]
    pub hac_lags_used: Option<i64>,
    #[pyo3(get)]
    pub r_squared: f64,
    #[pyo3(get)]
    pub adj_r_squared: f64,
    /// Overall F-test statistic of the slope coefficients. `None` when the test is not
    /// defined: there are no slope coefficients (intercept only), or, for a first-stage regression of
    /// `IVResults.first_stage()` with `cov_type="cluster"`, the number of clusters is not larger
    /// than the number of slope coefficients.
    #[pyo3(get)]
    pub f_statistic: Option<f64>,
    /// P-value of `f_statistic` (`None` exactly when `f_statistic` is `None`).
    #[pyo3(get)]
    pub f_p_value: Option<f64>,
    /// Numerator degrees of freedom of `f_statistic` (`None` when `f_statistic` is `None`).
    #[pyo3(get)]
    pub f_df_num: Option<usize>,
    /// Denominator degrees of freedom of `f_statistic` (`None` when `f_statistic` is `None`). May differ
    /// from `df_resid` (e.g. OLS/WLS with cluster-robust inference uses `G - 1`).
    #[pyo3(get)]
    pub f_df_denom: Option<usize>,
    /// Residual degrees of freedom (`n - k`).
    #[pyo3(get)]
    pub df_resid: usize,
    /// Model degrees of freedom (number of slope coefficients, excluding the intercept).
    #[pyo3(get)]
    pub df_model: usize,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    /// Fitted values for the training data (`ŷ = Xβ̂`), cached at fit time.
    /// Not exposed to Python directly; only `predict(new_data=None)` reads it
    /// (`docs/spec/ols-spec.md` "predict()" — the Python-facing surface is unified into a
    /// single `predict()` method rather than a separate `fitted_values` property).
    fitted_values: Vec<f64>,
    /// Whether `fit()` was called with `include_intercept=True`. Not exposed to
    /// Python; only `predict()` reads it to decide whether to auto-prepend a
    /// constant column for out-of-sample data.
    ///
    /// This must NOT be inferred from `param_names[0] == "const"`: when
    /// `include_intercept=False`, a user-supplied `x` column may legitimately be
    /// named `"const"` (the collision check in `fit()` only rejects that name when
    /// `include_intercept=True`), which would make such an inference silently
    /// wrong instead of erroring.
    has_intercept: bool,
    /// The `polars.DataFrame` passed to `fit()`, as given. `augment(new_data=None)` appends
    /// the predicted column to it (so every column is kept), and the diagnostic tests read the
    /// columns they need from it on demand.
    ///
    /// `None` for `OLSResult`s built by `ols_estimator_to_result` without going
    /// through this file's `fit()` (currently only `IVResult.first_stage()`,
    /// `engine_pybind/src/iv/common.rs`): those per-equation regressions have no
    /// single source DataFrame, so `augment(new_data=None)` and the diagnostic tests
    /// on such a result raise `ValidationError` instead.
    source_data: Option<Py<PyAny>>,
}

// `#[pymethods]`ブロックの外に置く非公開実装（pyo3は`#[pymethods]`内の全メソッドを
// Python公開シグネチャとして扱おうとするため、`&DataFrame`のような`FromPyObject`
// 未実装の型を引数に取るヘルパーはこちらに置く必要がある）。
impl OLSResult {
    /// 元のデータ（`fit()`に渡された`polars.DataFrame`）。保持していない結果（`IVResult.first_stage()`）は
    /// `TRAINING_DATA_NOT_RETAINED`。
    fn retained_source<'py>(&self, py: Python<'py>, method: &str) -> PyResult<Bound<'py, PyAny>> {
        self.source_data
            .as_ref()
            .map(|source| source.bind(py).clone())
            .ok_or_else(|| {
                user_error(
                    ErrorCode::TrainingDataNotRetained,
                    vec![("method", ParamValue::from(method))],
                )
            })
    }

    /// `predict()`/`augment()`のSome分岐で共有する、`df`に対するout-of-sample予測
    /// （`x`列の抽出→`predict_new_data`呼び出し）。
    fn predict_for(&self, df: &DataFrame) -> PyResult<Vec<f64>> {
        let has_intercept = self.has_intercept;
        let x_columns = extract_model_x_columns(
            df,
            &self.param_names,
            has_intercept,
            0,
            ColumnSource::new("x", "new_data"),
        )?;

        Ok(engine::linear::ols::predict_new_data(
            &self.params,
            has_intercept,
            &x_columns,
        ))
    }
}

#[pymethods]
impl OLSResult {
    /// Predicted values.
    ///
    /// With `new_data=None` (default), returns the fitted values for the training
    /// data used in `fit()`. With `new_data` given, computes out-of-sample
    /// predictions for a new polars DataFrame: it must contain columns with the same
    /// names as the `x` columns passed at fit time (matched by name; column order
    /// does not matter). If `include_intercept=True` was used at fit time, the
    /// constant column is added automatically and must not be included in `new_data`.
    ///
    /// # Errors
    /// - A required `x` column is missing from `new_data`, cannot be cast to a
    ///   numeric type, or contains missing/NaN/infinite values: `ValidationError`
    ///   (same validation as `fit()`'s column extraction, via `extract_f64_column`).
    #[pyo3(signature = (new_data=None))]
    fn predict(&self, new_data: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        guard(|| {
            let Some(new_data) = new_data else {
                return Ok(self.fitted_values.clone());
            };

            let selected = select_model_x_columns(
                new_data,
                "new_data",
                &self.param_names,
                self.has_intercept,
                0,
            )?;
            self.predict_for(&selected.frame)
        })
    }

    /// The source data (training data, or `new_data` when given) with the
    /// predicted values appended as a new `"predicted"` column.
    ///
    /// Same `new_data`/`include_intercept` semantics as `predict()`, but returns
    /// a polars DataFrame (original columns plus `"predicted"`, row order
    /// preserved) instead of a bare list of floats.
    ///
    /// # Errors
    /// - Same as `predict()`: a required `x` column missing from `new_data`,
    ///   non-numeric, or containing missing/NaN/infinite values: `ValidationError`.
    /// - The source data already has a column named `"predicted"`:
    ///   `ValidationError` (would otherwise silently overwrite it).
    /// - `new_data=None` and this result has no retained source data (currently
    ///   only possible for `IVResult.first_stage()` results): `ValidationError`.
    #[pyo3(signature = (new_data=None))]
    fn augment(&self, py: Python<'_>, new_data: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        guard(|| {
            let (source, predicted) = match new_data {
                Some(new_data) => {
                    let selected = select_model_x_columns(
                        new_data,
                        "new_data",
                        &self.param_names,
                        self.has_intercept,
                        0,
                    )?;
                    let predicted = self.predict_for(&selected.frame)?;
                    (new_data.clone(), predicted)
                }
                None => {
                    let source = self
                        .source_data
                        .as_ref()
                        .ok_or_else(|| user_error(ErrorCode::AugmentRequiresNewData, Vec::new()))?;
                    (source.bind(py).clone(), self.fitted_values.clone())
                }
            };

            append_predicted_column(&source, "predicted", predicted)
        })
    }

    /// White test for heteroskedasticity.
    ///
    /// Regresses the squared residuals on the independent variables, their squares and
    /// their pairwise products (always with a constant, even when the model was fitted
    /// with `include_intercept=False`) and returns either the LM version
    /// (`statistic="lm"`, `n * R^2`, chi-squared) or the F version (`statistic="f"`).
    /// Terms that are constant or numerically identical to an earlier term (for example
    /// the square of a 0/1 dummy) are dropped, and the degrees of freedom count the terms
    /// that remain. The test does not depend on `cov_type`.
    ///
    /// # Errors
    /// - `statistic` is not `"lm"` or `"f"`: `ValidationError`.
    /// - This result has no retained source data (currently only `IVResult.first_stage()`
    ///   results): `ValidationError`.
    /// - Every regressor is constant: `ValidationError`.
    /// - Too few observations for the auxiliary regression: `ValidationError`.
    /// - The auxiliary design matrix is still singular after dropping, or its R-squared
    ///   is undefined: `ComputationError`.
    #[pyo3(signature = (statistic="lm"))]
    fn white_test(&self, py: Python<'_>, statistic: &str) -> PyResult<WhiteTestOutput> {
        guard(|| {
            let version = parse_statistic_version(statistic)?;
            let source = self.retained_source(py, "white_test")?;

            let x_names = x_column_names(&self.param_names, self.has_intercept, 0);
            let df = select_columns(&source, "data", x_names)?.frame;
            let x_columns = extract_f64_columns(&df, x_names, ColumnSource::new("x", "data"))?;
            let result = white_test(&x_columns, x_names, &self.residuals)
                .map_err(least_squares_error_to_pyerr)?;

            let (statistic, p_value, df_denom, distribution) = select_statistic(&result, version);
            Ok(WhiteTestOutput {
                statistic,
                p_value,
                df: result.df,
                df_denom,
                distribution,
                aux_terms: result.aux_terms,
                dropped_terms: result.dropped_terms,
            })
        })
    }

    /// Breusch-Pagan test for heteroskedasticity (Koenker's studentized version).
    ///
    /// Regresses the squared residuals on a constant and the columns named in `variables`
    /// (always with a constant, even when the model was fitted with `include_intercept=False`)
    /// and returns either the LM version (`statistic="lm"`, `n * R^2`, chi-squared) or the
    /// F version (`statistic="f"`). It does not assume normal errors.
    ///
    /// `variables` is a list of column names of the data passed to `fit()`. It need not be
    /// the model's independent variables: any numeric column may be used, including one
    /// that is not in the model. When it is `None`, the independent variables of the model
    /// are used. Variables that are constant or numerically identical to an earlier one
    /// (for example a copy of a dummy) are dropped, and the degrees of freedom count the
    /// variables that remain. The test does not depend on `cov_type`.
    ///
    /// # Errors
    /// - `variables` is not a `list` of `str`: `TypeError`. `statistic` is not a `str`:
    ///   `TypeError`.
    /// - `statistic` is not `"lm"` or `"f"`, `variables` is empty or names a column twice, a
    ///   column does not exist, has an unsupported dtype or contains missing values, NaN or
    ///   infinity, every variable is constant, or too few observations for the auxiliary
    ///   regression: `ValidationError`.
    /// - This result has no retained source data (currently only `IVResult.first_stage()`
    ///   results): `ValidationError`.
    /// - The auxiliary design matrix is still singular after dropping, or its R-squared
    ///   is undefined: `ComputationError`.
    #[pyo3(signature = (variables=None, statistic="lm"))]
    fn breusch_pagan_test(
        &self,
        py: Python<'_>,
        variables: Option<&Bound<'_, PyAny>>,
        statistic: &str,
    ) -> PyResult<BreuschPaganTestOutput> {
        guard(|| {
            let version = parse_statistic_version(statistic)?;
            let variables = variables
                .map(|ob| extract_column_list(ob, "variables"))
                .transpose()?;
            let source = self.retained_source(py, "breusch_pagan_test")?;

            let z_names = match variables.as_deref() {
                Some(names) => {
                    validate_x_non_empty("variables", names)?;
                    validate_no_duplicate_within_role("variables", names)?;
                    names
                }
                None => x_column_names(&self.param_names, self.has_intercept, 0),
            };
            let z_argument = if variables.is_some() {
                "variables"
            } else {
                "x"
            };
            let df = select_columns(&source, "data", z_names)?.frame;
            let z_columns =
                extract_f64_columns(&df, z_names, ColumnSource::new(z_argument, "data"))?;
            let result = breusch_pagan_test(&z_columns, z_names, &self.residuals)
                .map_err(least_squares_error_to_pyerr)?;

            let (statistic, p_value, df_denom, distribution) = select_statistic(&result, version);
            Ok(BreuschPaganTestOutput {
                statistic,
                p_value,
                df: result.df,
                df_denom,
                distribution,
                aux_terms: result.aux_terms,
                dropped_terms: result.dropped_terms,
            })
        })
    }

    /// Breusch-Godfrey test for serial correlation of the errors.
    ///
    /// Regresses the residuals on the model's independent variables and their own lags
    /// 1 to `nlags` (in the order given by the `time` column; lags before the first
    /// observation are 0), and tests that the lag coefficients are zero. The auxiliary
    /// regression uses the model's regressors as they are: no constant is added to a
    /// model fitted with `include_intercept=False`. Returns the LM version
    /// (`statistic="lm"`, `n * R^2`, chi-squared) or the F version (`statistic="f"`).
    /// The test does not depend on `cov_type`.
    ///
    /// `time` names a column of the data passed to `fit()`; the row order is never
    /// assumed to be the time order. The values only give the order (no gaps are
    /// checked): lags are taken in that order.
    ///
    /// # Errors
    /// - `time` or `statistic` is not a `str`, or `nlags` is not an `int` (`bool` and `float`
    ///   included): `TypeError`.
    /// - `statistic` is not `"lm"` or `"f"`, `nlags < 1`, or too few observations for the
    ///   auxiliary regression (`n <= k + nlags`): `ValidationError`.
    /// - `time` does not exist, has an unsupported dtype, or contains missing values, NaN,
    ///   infinity or duplicates: `ValidationError`.
    /// - This result has no retained source data (currently only `IVResult.first_stage()`
    ///   results): `ValidationError`.
    /// - The auxiliary design matrix is singular, or the residuals are all zero / fitted
    ///   exactly by the auxiliary regression: `ComputationError`.
    #[pyo3(signature = (time, nlags, statistic=None))]
    fn breusch_godfrey_test(
        &self,
        py: Python<'_>,
        time: &Bound<'_, PyAny>,
        nlags: &Bound<'_, PyAny>,
        statistic: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<BreuschGodfreyTestOutput> {
        guard(|| {
            let time = extract_strict_text(time, "time")?;
            let nlags = extract_strict_int(nlags, "nlags")?;
            let statistic = match statistic {
                Some(value) => extract_strict_text(value, "statistic")?,
                None => "lm".to_string(),
            };
            let version = parse_statistic_version(&statistic)?;
            let source = self.retained_source(py, "breusch_godfrey_test")?;

            let mut needed: Vec<String> =
                x_column_names(&self.param_names, self.has_intercept, 0).to_vec();
            needed.push(time.clone());
            let df = select_columns(&source, "data", &needed)?.frame;
            let x_columns = extract_model_x_columns(
                &df,
                &self.param_names,
                self.has_intercept,
                0,
                ColumnSource::new("x", "data"),
            )?;
            let time_order =
                extract_time_order_ranks(&df, &time, ColumnSource::new("time", "data"))?;
            let result = breusch_godfrey_test(
                &x_columns,
                self.has_intercept,
                &self.residuals,
                &time_order,
                nlags,
            )
            .map_err(least_squares_error_to_pyerr)?;

            Ok(match version {
                StatisticVersion::F => BreuschGodfreyTestOutput {
                    statistic: result.f_statistic,
                    p_value: result.f_p_value,
                    df: result.df,
                    df_denom: Some(result.f_df_denom),
                    distribution: "f".to_string(),
                    nlags: result.df,
                },
                StatisticVersion::Lm => BreuschGodfreyTestOutput {
                    statistic: result.lm_statistic,
                    p_value: result.lm_p_value,
                    df: result.df,
                    df_denom: None,
                    distribution: "chi2".to_string(),
                    nlags: result.df,
                },
            })
        })
    }
}

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `engine::linear::ols::OlsInput::from_columns` + `OlsEstimator::fit`を呼び出して
/// OLSを推定し、`OLSResult`として返す。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`・`x`の重複、`include_intercept=true`のときの`"const"`列との衝突は
///   ここ（受け口）の責務で`ValidationError`（`engine`の一般的な`SingularMatrix`より
///   先に、分かりやすいメッセージで弾く）
/// - `cov_type`の文字列が不正な場合は`ValidationError`
/// - それ以外（観測数不足・信頼水準の範囲外・特異行列・クラスター数不足・
///   `hac_lags`の範囲外・クラスターキー未指定等）は`engine::linear::common::LeastSquaresError`から
///   `least_squares_error_to_pyerr`で変換
pub fn fit(
    py: Python<'_>,
    data: SelectedData,
    y: String,
    x: Vec<String>,
    options: &OLSOptions,
) -> PyResult<OLSResult> {
    let SelectedData { source, frame } = data;
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる
    // （他のPythonスレッドを止めず、複数のfitをスレッドで並行できる）。GILが要る`source`は
    // クロージャの外に残し、結果への取り付けは戻ってから行う。
    let mut result = py.detach(|| fit_frame(&frame, y, x, options))?;
    result.source_data = Some(source);
    Ok(result)
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。`PyErr`は遅延生成でGILを要求しない
/// ので、ここで`?`により返してよい。
fn fit_frame(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    options: &OLSOptions,
) -> PyResult<OLSResult> {
    // DataFrameを見ない検査を先に行う。報告順は`docs/guide/validation.md`
    // 「Order in which errors are reported」: オプションの値 → オプションの組合せ →
    // ロール構造 → `y`/`x`の列 → オプション列 → エンジン。
    let cov_kind = parse_cov_kind(&options.cov_type)?;
    check_cov_options(
        cov_kind,
        options.cluster.as_deref(),
        options.hac_lags,
        options.hac_time.as_deref(),
    )?;

    // 完全な多重共線性を早期に、分かりやすいエラーで防ぐ（`validation.rs`に集約、
    // WLS/Logitと共通、`.claude/rules/rust-style.md`参照）。
    validate_common_roles(&y, &x, options.include_intercept)?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;

    // ── x列の抽出 ──────────────────────────────────────────────────────
    let x_slices = extract_f64_columns(df, &x, ColumnSource::new("x", "data"))?;

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let cov_type = resolve_cov_type(
        df,
        cov_kind,
        options.cluster.as_deref(),
        options.hac_lags,
        options.hac_time.as_deref(),
    )?;

    let input = OlsInput::from_columns(&y_slice, &x_slices, x, options.include_intercept, y)
        .map_err(least_squares_error_to_pyerr)?;
    let estimator = OlsEstimator::fit(input, cov_type, options.confidence_level)
        .map_err(least_squares_error_to_pyerr)?;

    Ok(ols_estimator_to_result(
        &estimator,
        cov_kind.name().to_string(),
    ))
}

/// フィット済み`OlsEstimator`を`OLSResult`（pyclass、Pythonに返す形）に変換する。
///
/// `fit`（本ファイル、OLS本体）と`iv::common`の`first_stage()`（IVの第一段階回帰
/// `x_endog[i] ~ x_exog + instruments`の結果を`dict[str, OLSResults]`として返す）
/// の両方で使う共通の変換ロジック。第一段階回帰はそれ自体が正しい
/// （ナイーブな）通常のOLS回帰であり（`engine::iv::two_sls`のモジュールdocコメント
/// 「第一段階の各`OlsEstimator`はそれ自体が正しい」参照）、`OLSResult`への変換方法に
/// OLS本体との違いは無いため、このように同じ関数をそのまま再利用できる（`OLSResult`の
/// 非公開フィールド`fitted_values`/`has_intercept`にアクセスする都合上、`OLSResult`と
/// 同じ`linear::ols`モジュール内に置く）。
///
/// `cov_type_lower`を引数で受け取るのは、`fit`では`CovKind::name()`で作った正規形の名前、
/// `first_stage()`では`IVResult.cov_type`
/// （呼び出し元が指定した`cov_type`、第一段階にもそのまま使われる、`iv::two_sls`の
/// モジュールdocコメント参照）と、呼び出し元ごとに文字列の出どころが異なるため。
/// **呼び出し元が正規形の値を渡す責任を持つ**（この関数自体は
/// 正規化・妥当性検証を行わない。`OlsEstimator`が実際に使った`cov_type`と一致する文字列を
/// 呼び出し元の責任で渡す契約）。
pub(crate) fn ols_estimator_to_result(
    estimator: &OlsEstimator,
    cov_type_lower: String,
) -> OLSResult {
    OLSResult {
        params: mat_to_vec(estimator.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        test_stats: mat_to_vec(estimator.test_stats()),
        stat_dist: estimator.stat_dist().name().to_string(),
        stat_df: estimator.stat_dist().df().map(|df| df as i64),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.input().param_names().to_vec(),
        residuals: mat_to_vec(estimator.residuals()),
        dep_var_name: estimator.input().dep_var_name().to_string(),
        n_obs: estimator.input().nobs(),
        cov_type: cov_type_lower,
        hac_lags_used: estimator.hac_lags_used().map(|lags| lags as i64),
        r_squared: estimator.r_squared(),
        adj_r_squared: estimator.adj_r_squared(),
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
        f_df_num: estimator.f_df().map(|(num, _)| num),
        f_df_denom: estimator.f_df().map(|(_, denom)| denom),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        log_likelihood: estimator.log_likelihood(),
        aic: estimator.aic(),
        bic: estimator.bic(),
        fitted_values: mat_to_vec(&estimator.fitted_values()),
        has_intercept: estimator.input().has_intercept(),
        // `fit()`（本ファイル）が呼び出し後に`Some(df)`で上書きする。この関数の
        // もう一つの呼び出し元`iv::common::first_stage()`は単一のソースDataFrameを
        // 持たないため`None`のまま（`OLSResult`のdocコメント参照）。
        source_data: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_statistic_version_accepts_lm_and_f_case_insensitively() {
        assert_eq!(parse_statistic_version("lm").unwrap(), StatisticVersion::Lm);
        assert_eq!(parse_statistic_version("LM").unwrap(), StatisticVersion::Lm);
        assert_eq!(parse_statistic_version("f").unwrap(), StatisticVersion::F);
        assert_eq!(parse_statistic_version("F").unwrap(), StatisticVersion::F);
    }

    #[test]
    fn parse_statistic_version_rejects_unknown_value() {
        assert!(parse_statistic_version("chi2").is_err());
        assert!(parse_statistic_version("").is_err());
    }

    #[test]
    fn select_statistic_picks_the_requested_version() {
        let result = AuxRegressionTest {
            lm_statistic: 1.0,
            lm_p_value: 0.1,
            f_statistic: 2.0,
            f_p_value: 0.2,
            df: 3,
            f_df_denom: 11,
            aux_terms: vec!["const".to_string()],
            dropped_terms: Vec::new(),
        };
        assert_eq!(
            select_statistic(&result, StatisticVersion::Lm),
            (1.0, 0.1, None, "chi2".to_string())
        );
        assert_eq!(
            select_statistic(&result, StatisticVersion::F),
            (2.0, 0.2, Some(11), "f".to_string())
        );
    }
}
