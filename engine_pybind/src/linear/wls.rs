//! WLSの推定オプション・結果、およびPython（polars DataFrame + `y`/`x`/`weight`列名 +
//! オプション）から`engine::linear::wls::WlsEstimator`を呼び出し、結果をPython側に返す
//! ところまでの一連の処理。
//!
//! `weight`は`y`と同じく`data`内の列名を指すトップレベル引数として扱う
//! （`docs/spec/wls-spec.md`「API引数」参照）。`WLSOptions`は`OLSOptions`と
//! フィールド構成が完全に同一の独立したpyclassである（下記
//! `WLSOptions`のdocコメント参照）。エラー変換（`least_squares_error_to_pyerr`）・
//! `cov_type`のパース（`parse_cov_kind`・`check_cov_options`・`resolve_cov_type`）は`super::common`のものをそのまま再利用する
//! （`LeastSquaresError`がOLS・WLS共通のエラー型のため。`.claude/rules/rust-style.md`
//! 「系統内で共有するロジックはcommon.rsに置く」）。`Mat<f64>`→`Vec<f64>`変換（`mat_to_vec`）は
//! 系統によらず使うため`crate::shared::mat_conversion`のものを再利用する。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`WLSOptions`/`WLSResult`）のdocコメントは英語。それ以外（このファイルの
//! 説明・非公開関数のdocコメント等）は日本語のまま。

use crate::shared::errors::guard;
use engine::linear::wls::WlsEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::common::{
    check_cov_options, least_squares_error_to_pyerr, parse_cov_kind, resolve_cov_type,
};
use crate::shared::augment::append_predicted_column;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_f64_column, extract_f64_columns, extract_model_x_columns,
    select_model_x_columns,
};
use crate::shared::mat_conversion::mat_to_vec;
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_opt_column, extract_strict_opt_int, extract_strict_text,
};
use crate::shared::validation::{RoleValue, validate_common_roles, validate_no_duplicate_roles};

/// Estimation options for WLS.
///
/// `weight` is not an option: it is a top-level argument of `WLS`, next to `y` and `x`.
// 各フィールドの意味・既定値の根拠は`docs/spec/wls-spec.md`「API引数」。
// `OLSOptions`と同じフィールド構成だが、将来WLS固有のオプションを足してもOLS利用者に
// 影響しないよう、結果型`WLSResult`と同様に独立した型にしている。このフィールドの重複は
// 意図的に共通baseへ切り出さない（`#[pyclass]`のコンストラクタはフラットなキーワード引数前提で、
// 共通型を挟むとPython側のAPIの形に影響するため。`IVOptions`も同じ方針）。
// `fit`がPython側から`WLSOptions`インスタンスを引数として受け取るため、
// `FromPyObject`実装を明示的に維持する（`OLSOptions`と同じ理由、pyo3 0.28以降、Cloneを
// 実装する#[pyclass]のFromPyObject自動導出はopt-inに変更されたため）。
// module: PyO3の#[pyclass]はデフォルトで__module__="builtins"になり、
// mkdocstrings（griffe）がPythonでの再エクスポートのalias解決に失敗する原因になる。
// 実際のインポート元(`econometricsmodels._lib`)を明示する。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct WLSOptions {
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

impl WLSOptions {
    /// このオプションが指名するデータの列（`cluster` ・`hac_time`）。使われるかどうか（`cov_type`等）には
    /// よらず、指名されたものを全て返す。`fit()`はこれらと`y`・`x`等だけをデータから取り込む。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().chain(&self.hac_time).cloned().collect()
    }
}

#[pymethods]
impl WLSOptions {
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
            "WLSOptions(cov_type={}, include_intercept={}, confidence_level={}, \
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

/// Estimation results for WLS.
///
// `OLSResult`と同じフィールド構成だが、将来WLS固有の項目（重み付き残差等）を足しても
// `OLSResult`に影響しないよう独立した型にしている（`docs/spec/wls-spec.md`「結果構造体」）。
// `get_all`ではなく個別`#[pyo3(get)]`にしている理由は`OLSResult`と同じ
// （`fitted_values`/`has_intercept`をPython側に公開しないため）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct WLSResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub test_stats: Vec<f64>,
    /// Distribution of `test_stats`; always `"t"` for WLS.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the t distribution (always `Some` for WLS). May differ from
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
    /// Original-scale (unweighted) residuals `y_i - x_i'β̂`. Not the weighted residuals
    /// used internally for the standard error calculations
    /// (`docs/spec/wls-spec.md`, "結果構造体").
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
    /// defined: there are no slope coefficients.
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
    /// Original-scale (unweighted) fitted values for the training data (`ŷ = Xβ̂`),
    /// cached at fit time. Not exposed to Python directly; only `predict(new_data=None)`
    /// reads it (same design as `OLSResult`).
    fitted_values: Vec<f64>,
    /// Whether `fit()` was called with `include_intercept=True`. Not exposed to
    /// Python; only `predict()` reads it to decide whether to auto-prepend a
    /// constant column for out-of-sample data (same reasoning as `OLSResult`).
    has_intercept: bool,
    /// The `polars.DataFrame` passed to `fit()`, as given. `augment(new_data=None)` appends
    /// the predicted column to it, so every column is kept, including the ones the model
    /// does not read.
    source_data: Py<PyAny>,
}

// `#[pymethods]`ブロックの外に置く非公開実装（`OLSResult`と同じ理由）。
impl WLSResult {
    /// `predict()`/`augment()`のSome分岐で共有する、`df`に対するout-of-sample予測
    /// （`x`列の抽出→`predict_new_data`呼び出し、`OLSResult`と同じ設計）。
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
impl WLSResult {
    /// Predicted values.
    ///
    /// With `new_data=None` (default), returns the fitted values for the training
    /// data used in `fit()`, on the original (unweighted) scale — `ŷ = Xβ̂`, the same
    /// scale as `residuals` (`docs/spec/wls-spec.md`, "結果構造体"). With `new_data`
    /// given, computes out-of-sample predictions for a new polars DataFrame: it must
    /// contain columns with the same names as the `x` columns passed at fit time
    /// (matched by name; column order does not matter). Weights play no role in
    /// either case — the predicted value is `x_i'β̂` regardless of `new_data`, and
    /// out-of-sample observations have no weight to apply. If `include_intercept=True`
    /// was used at fit time, the constant column is added automatically and must not
    /// be included in `new_data`.
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
    /// Same `new_data`/`include_intercept` semantics as `predict()` (same design
    /// as `OLSResult::augment()`), including weights playing no role.
    ///
    /// # Errors
    /// - Same as `predict()`: a required `x` column missing from `new_data`,
    ///   non-numeric, or containing missing/NaN/infinite values: `ValidationError`.
    /// - The source data already has a column named `"predicted"`:
    ///   `ValidationError` (would otherwise silently overwrite it).
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
                None => (
                    self.source_data.bind(py).clone(),
                    self.fitted_values.clone(),
                ),
            };

            append_predicted_column(&source, "predicted", predicted)
        })
    }
}

/// Pythonから渡された `data` / `y` / `x` / `weight` / `options` を検証し、
/// `engine::linear::wls::WlsEstimator::fit`を呼び出してWLSを推定し、`WLSResult`として返す。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`と`x`の重複、`weight`と`y`の重複（`weight`と`x`の重複は許容、
///   下記コメント参照）、`include_intercept=true`のときの`"const"`列との衝突は
///   ここ（受け口）の責務で`ValidationError`（OLSの`fit`と同じパターン）
/// - `cov_type`の文字列が不正な場合は`ValidationError`
/// - それ以外（観測数不足・信頼水準の範囲外・特異行列・クラスター数不足・
///   `hac_lags`の範囲外・クラスターキー未指定・重みの次元不一致・非正の重み等）は
///   `engine::linear::common::LeastSquaresError`から`least_squares_error_to_pyerr`で変換
pub fn fit(
    data: SelectedData,
    y: String,
    x: Vec<String>,
    weight: String,
    options: &WLSOptions,
) -> PyResult<WLSResult> {
    let SelectedData { source, frame: df } = data;

    // DataFrameを見ない検査を先に行う（報告順は`ols.rs`の`fit`と同じ）。
    let cov_kind = parse_cov_kind(&options.cov_type)?;
    check_cov_options(
        cov_kind,
        options.cluster.as_deref(),
        options.hac_lags,
        options.hac_time.as_deref(),
    )?;

    // 誤って同じ列を複数の役割に指定するミスを、分かりやすいエラーで早期に防ぐ
    // （`docs/spec/wls-spec.md`「API引数」参照）。`weight`と`x`の重複は禁止しない
    // （重みに使った列を説明変数としても含める実務上の利用例があるため。
    // `weight == y`は`y`を独立変数としても使うのと同型の致命的な問題のため引き続き禁止）。
    validate_common_roles(&y, &x, options.include_intercept)?;
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(&y)),
        ("weight", RoleValue::Single(&weight)),
    ])?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(&df, &y, ColumnSource::new("y", "data"))?;

    // ── x列の抽出 ──────────────────────────────────────────────────────
    let x_slices = extract_f64_columns(&df, &x, ColumnSource::new("x", "data"))?;

    // ── weight列の抽出 ─────────────────────────────────────────────────
    // NaN/無限大・欠損値の検証はextract_f64_columnがy/xと同じ経路で行う。0以下の値
    // （analytic weightとして不正）の検証はengine側（LeastSquaresError::NonPositiveWeight）に
    // 委ねる（`docs/spec/wls-spec.md`「エラー型」参照）。
    let weight_slice = extract_f64_column(&df, &weight, ColumnSource::new("weight", "data"))?;

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ、OLSと同じ）─────
    let cov_type = resolve_cov_type(
        &df,
        cov_kind,
        options.cluster.as_deref(),
        options.hac_lags,
        options.hac_time.as_deref(),
    )?;

    let wls_estimator = WlsEstimator::fit(
        &y_slice,
        &x_slices,
        x,
        options.include_intercept,
        y,
        &weight_slice,
        cov_type,
        options.confidence_level,
    )
    .map_err(least_squares_error_to_pyerr)?;

    let estimator = wls_estimator.estimator();

    Ok(WLSResult {
        params: mat_to_vec(estimator.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        test_stats: mat_to_vec(estimator.test_stats()),
        stat_dist: estimator.stat_dist().name().to_string(),
        stat_df: estimator.stat_dist().df().map(|df| df as i64),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.input().param_names().to_vec(),
        residuals: wls_estimator.residuals().to_vec(),
        dep_var_name: estimator.input().dep_var_name().to_string(),
        n_obs: estimator.input().nobs(),
        cov_type: cov_kind.name().to_string(),
        hac_lags_used: estimator.hac_lags_used().map(|lags| lags as i64),
        // r_squared/adj_r_squared/log_likelihood/aic/bicは`estimator`（変換後データに対する
        // OLS）ではなく`wls_estimator`側の値を使う。元の（変換前の）y・weightsを使って
        // 計算し直したもので、`estimator`側の値は変換のヤコビアン補正等が欠けており
        // statsmodelsと一致しない（`engine::linear::wls`モジュール冒頭のdocコメント参照）。
        r_squared: wls_estimator.r_squared(),
        adj_r_squared: wls_estimator.adj_r_squared(),
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
        f_df_num: estimator.f_df().map(|(num, _)| num),
        f_df_denom: estimator.f_df().map(|(_, denom)| denom),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        log_likelihood: wls_estimator.log_likelihood(),
        aic: wls_estimator.aic(),
        bic: wls_estimator.bic(),
        fitted_values: wls_estimator.fitted_values().to_vec(),
        has_intercept: estimator.input().has_intercept(),
        source_data: source,
    })
}
