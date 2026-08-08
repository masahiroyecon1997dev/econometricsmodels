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

use engine::linear::ols::{CovType as EngineCovType, OlsEstimator, OlsInput};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use super::common::{least_squares_error_to_pyerr, mat_to_vec};
use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::ValidationError;
use crate::validation::{
    validate_no_const_collision, validate_no_duplicate_roles, validate_no_duplicate_x,
    validate_x_non_empty,
};

/// Estimation options for OLS.
///
/// See `docs/spec/ols-spec.md` ("API引数") for the rationale behind each
/// field's meaning and default value.
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
    #[pyo3(get, set)]
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
    #[pyo3(get, set)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`.
    /// Refers to a column in `data` rather than being passed as a separate array.
    /// Ignored when `cov_type` is not "cluster".
    #[pyo3(get, set)]
    pub cluster_col: Option<String>,

    /// Number of lags (bandwidth) for HAC (Newey-West) when `cov_type="hac"`.
    /// When `None`, computed automatically via `L = floor(4*(n/100)^(2/9))`.
    /// Ignored when `cov_type` is not "hac".
    #[pyo3(get, set)]
    pub hac_lags: Option<i64>,

    /// Column name giving the time order for HAC when `cov_type="hac"`.
    /// When `None`, the row order of `data` is treated as the time order.
    /// Ignored when `cov_type` is not "hac".
    #[pyo3(get, set)]
    pub time_col: Option<String>,
}

#[pymethods]
impl OLSOptions {
    #[new]
    #[pyo3(signature = (
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster_col = None,
        hac_lags = None,
        time_col = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cov_type: String,
        include_intercept: bool,
        confidence_level: f64,
        cluster_col: Option<String>,
        hac_lags: Option<i64>,
        time_col: Option<String>,
    ) -> Self {
        Self {
            cov_type,
            include_intercept,
            confidence_level,
            cluster_col,
            hac_lags,
            time_col,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OLSOptions(cov_type={:?}, include_intercept={}, confidence_level={}, \
             cluster_col={:?}, hac_lags={:?}, time_col={:?})",
            self.cov_type,
            self.include_intercept,
            self.confidence_level,
            self.cluster_col,
            self.hac_lags,
            self.time_col
        )
    }
}

/// Estimation results for OLS.
///
/// Structured data only (no `summary()`); see `docs/spec/ols-spec.md`
/// ("結果構造体"). Row-oriented table construction (e.g. a `coef_table`) is left to
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
#[derive(Debug, Clone)]
pub struct OLSResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub t_stats: Vec<f64>,
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
    #[pyo3(get)]
    pub r_squared: f64,
    #[pyo3(get)]
    pub r_squared_adj: f64,
    #[pyo3(get)]
    pub f_statistic: f64,
    #[pyo3(get)]
    pub f_p_value: f64,
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
    fn predict(&self, new_data: Option<PyDataFrame>) -> PyResult<Vec<f64>> {
        let Some(new_data) = new_data else {
            return Ok(self.fitted_values.clone());
        };

        let df: DataFrame = new_data.into();
        let has_intercept = self.has_intercept;
        let x_names: &[String] = if has_intercept {
            &self.param_names[1..]
        } else {
            &self.param_names[..]
        };

        let mut x_columns: Vec<Vec<f64>> = Vec::with_capacity(x_names.len());
        for name in x_names {
            x_columns.push(extract_f64_column(&df, name)?);
        }

        Ok(engine::linear::ols::predict_new_data(
            &self.params,
            has_intercept,
            &x_columns,
        ))
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
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: &OLSOptions,
) -> PyResult<OLSResult> {
    let df: DataFrame = data.into();
    let cov_type_lower = options.cov_type.to_lowercase();

    // 完全な多重共線性を早期に、分かりやすいエラーで防ぐ（`validation.rs`に集約、
    // WLS/Logitと共通、`.claude/rules/rust-style.md`参照）。
    validate_x_non_empty(&x)?;
    validate_no_duplicate_roles(&[("y", &y)], &x)?;
    validate_no_duplicate_x(&x)?;
    validate_no_const_collision(&x, options.include_intercept)?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(&df, &y)?;

    // ── x列の抽出 ──────────────────────────────────────────────────────
    let mut x_slices: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for col_name in &x {
        x_slices.push(extract_f64_column(&df, col_name)?);
    }

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    // `cluster_col`/`time_col`が指定されていても、cov_typeがcluster/hacでなければ
    // 無視する（`docs/spec/ols-spec.md`「標準誤差」のHAC参照）。
    let cluster_groups = if cov_type_lower == "cluster" {
        options
            .cluster_col
            .as_ref()
            .map(|col_name| extract_group_key_column(&df, col_name))
            .transpose()?
    } else {
        None
    };

    let time_order = if cov_type_lower == "hac" {
        options
            .time_col
            .as_ref()
            .map(|col_name| extract_f64_column(&df, col_name))
            .transpose()?
    } else {
        None
    };

    let cov_type = match cov_type_lower.as_str() {
        "classical" | "nonrobust" => EngineCovType::Classical,
        "hc0" => EngineCovType::Hc0,
        "hc1" => EngineCovType::Hc1,
        "hc2" => EngineCovType::Hc2,
        "hc3" => EngineCovType::Hc3,
        "hac" => EngineCovType::Hac {
            lags: options.hac_lags,
            time_order,
        },
        "cluster" => EngineCovType::Cluster {
            groups: cluster_groups,
        },
        other => {
            return Err(ValidationError::new_err(format!(
                "unknown cov_type: '{other}'. Expected one of 'classical', 'hc0' through \
                 'hc3', 'hac', or 'cluster'"
            )));
        }
    };

    let input = OlsInput::from_columns(&y_slice, &x_slices, x, options.include_intercept, y)
        .map_err(least_squares_error_to_pyerr)?;
    let estimator = OlsEstimator::fit(input, cov_type, options.confidence_level)
        .map_err(least_squares_error_to_pyerr)?;

    Ok(OLSResult {
        params: mat_to_vec(estimator.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        t_stats: mat_to_vec(estimator.t_stats()),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.input().param_names().to_vec(),
        residuals: mat_to_vec(estimator.residuals()),
        dep_var_name: estimator.input().dep_var_name().to_string(),
        n_obs: estimator.input().nobs(),
        cov_type: cov_type_lower,
        r_squared: estimator.r_squared(),
        r_squared_adj: estimator.r_squared_adj(),
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
        log_likelihood: estimator.log_likelihood(),
        aic: estimator.aic(),
        bic: estimator.bic(),
        fitted_values: mat_to_vec(&estimator.fitted_values()),
        has_intercept: estimator.input().has_intercept(),
    })
}
