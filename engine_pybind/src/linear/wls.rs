//! WLSの推定オプション・結果、およびPython（polars DataFrame + `y`/`x`/`weight`列名 +
//! オプション）から`engine::linear::wls::WlsEstimator`を呼び出し、結果をPython側に返す
//! ところまでの一連の処理。
//!
//! `weight`は`y`と同じく`data`内の列名を指すトップレベル引数として扱う
//! （`docs/spec/wls-spec.md`「API引数」参照）。`WLSOptions`は`OLSOptions`と
//! フィールド構成が完全に同一の独立したpyclassである（Issue #308、下記
//! `WLSOptions`のdocコメント参照）。エラー変換（`least_squares_error_to_pyerr`）・
//! `Mat<f64>`→`Vec<f64>`変換（`mat_to_vec`）・`cov_type`のパース（`parse_cov_type`）は
//! `super::common`のものをそのまま再利用する（`LeastSquaresError`がOLS・WLS共通の
//! エラー型のため。`.claude/rules/rust-style.md`「系統内で共有するロジックは
//! common.rsに置く」）。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`WLSOptions`/`WLSResult`）のdocコメントは英語。それ以外（このファイルの
//! 説明・非公開関数のdocコメント等）は日本語のまま。

use engine::linear::wls::WlsEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use super::common::{least_squares_error_to_pyerr, mat_to_vec, parse_cov_type};
use crate::column_extraction::extract_f64_column;
use crate::validation::{
    RoleValue, validate_no_const_collision, validate_no_duplicate_roles,
    validate_no_duplicate_within_role, validate_x_non_empty,
};

/// Estimation options for WLS.
///
/// See `docs/spec/wls-spec.md` ("API引数") for the rationale behind each
/// field's meaning and default value.
///
/// Field-for-field identical to `OLSOptions` today (`cov_type`/`include_intercept`/
/// `confidence_level`/`cluster_col`/`hac_lags`/`time_col`, same defaults and semantics
/// — `docs/spec/wls-spec.md` "API引数" confirms `hac_lags`/`time_col` mean exactly the
/// same thing for WLS as for OLS). Kept as an independent pyclass rather than reusing
/// `OLSOptions` (the pre-Issue #308 design) so that a future WLS-specific option can be
/// added without affecting `OLSOptions`/OLS users — the same reasoning `WLSResult`
/// already uses relative to `OLSResult`.
///
/// This field-for-field duplication with `OLSOptions` (and, in the `nonlinear` system,
/// `LogitOptions`/`ProbitOptions`/`TobitOptions`'s duplicated `method`/`max_iter`/`tol`/
/// `raise_on_non_convergence`) is intentional and will not be collapsed into a shared
/// base struct/trait here: PyO3's `#[pyclass]`/`#[pymethods]` constructor is inherently a
/// flat keyword-argument surface, so a shared base type would either leak into the
/// Python-facing API shape (composition: `WLSOptions(cov_type=..., mle=MleOptions(...))`)
/// or add indirection without reducing the Python surface. `IvOptions`
/// (`engine_pybind/src/iv/common.rs`) already re-declares this same field group
/// independently from `OLSOptions`, so this duplication is consistent with the existing
/// precedent in this codebase (Issue #308 decision, 2026-09-12). Mechanical
/// deduplication of the Rust-side boilerplate itself (field declarations/constructor/
/// `__repr__`) via `macro_rules!` is tracked separately in Issue #315.
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
impl WLSOptions {
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
            "WLSOptions(cov_type={:?}, include_intercept={}, confidence_level={}, \
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

/// Estimation results for WLS.
///
/// Field-for-field identical to `OLSResult` today, but kept as a separate type: WLS-specific
/// fields (e.g. weighted residuals) may be added later without affecting `OLSResult`
/// (`docs/spec/wls-spec.md`, "結果構造体").
#[pyclass(get_all, skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct WLSResult {
    pub params: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_lower: Vec<f64>,
    pub conf_upper: Vec<f64>,
    pub param_names: Vec<String>,
    /// Original-scale (unweighted) residuals `y_i - x_i'β̂`. Not the weighted residuals
    /// used internally for the standard error calculations
    /// (`docs/spec/wls-spec.md`, "結果構造体").
    pub residuals: Vec<f64>,
    pub dep_var_name: String,
    pub n_obs: usize,
    /// Standard error type actually used (echoes `OLSOptions.cov_type`, normalized to
    /// lowercase; e.g. `"classical"`, `"hc1"`, `"hac"`, `"cluster"`).
    pub cov_type: String,
    pub r_squared: f64,
    pub r_squared_adj: f64,
    pub f_statistic: f64,
    pub f_p_value: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
}

/// Pythonから渡された `data` / `y` / `x` / `weight` / `options` を検証し、
/// `engine::linear::wls::WlsEstimator::fit`を呼び出してWLSを推定し、`WLSResult`として返す。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`と`x`の重複、`weight`と`y`の重複（`weight`と`x`の重複はIssue #277により許容、
///   下記コメント参照）、`include_intercept=true`のときの`"const"`列との衝突は
///   ここ（受け口）の責務で`ValidationError`（OLSの`fit`と同じパターン）
/// - `cov_type`の文字列が不正な場合は`ValidationError`
/// - それ以外（観測数不足・信頼水準の範囲外・特異行列・クラスター数不足・
///   `hac_lags`の範囲外・クラスターキー未指定・重みの次元不一致・非正の重み等）は
///   `engine::linear::common::LeastSquaresError`から`least_squares_error_to_pyerr`で変換
pub fn fit(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    weight: String,
    options: &WLSOptions,
) -> PyResult<WLSResult> {
    let df: DataFrame = data.into();

    // 誤って同じ列を複数の役割に指定するミスを、分かりやすいエラーで早期に防ぐ
    // （`docs/spec/wls-spec.md`「API引数」参照）。`weight`と`x`の重複は禁止しない
    // （Issue #277: 重みに使った列を説明変数としても含める実務上の利用例があるため。
    // `weight == y`は`y`を独立変数としても使うのと同型の致命的な問題のため引き続き禁止）。
    validate_x_non_empty(&x)?;
    validate_no_duplicate_roles(&[("y", RoleValue::Single(&y)), ("x", RoleValue::Multi(&x))])?;
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(&y)),
        ("weight", RoleValue::Single(&weight)),
    ])?;
    validate_no_duplicate_within_role("x", &x)?;
    validate_no_const_collision(&x, options.include_intercept)?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(&df, &y)?;

    // ── x列の抽出 ──────────────────────────────────────────────────────
    let mut x_slices: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for col_name in &x {
        x_slices.push(extract_f64_column(&df, col_name)?);
    }

    // ── weight列の抽出 ─────────────────────────────────────────────────
    // NaN/無限大・欠損値の検証はextract_f64_columnがy/xと同じ経路で行う。0以下の値
    // （analytic weightとして不正）の検証はengine側（LeastSquaresError::NonPositiveWeight）に
    // 委ねる（`docs/spec/wls-spec.md`「エラー型」参照）。
    let weight_slice = extract_f64_column(&df, &weight)?;

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ、OLSと同じ）─────
    let (cov_type, cov_type_lower) = parse_cov_type(
        &df,
        &options.cov_type,
        options.cluster_col.as_deref(),
        options.hac_lags,
        options.time_col.as_deref(),
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
        t_stats: mat_to_vec(estimator.t_stats()),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.input().param_names().to_vec(),
        residuals: wls_estimator.residuals().to_vec(),
        dep_var_name: estimator.input().dep_var_name().to_string(),
        n_obs: estimator.input().nobs(),
        cov_type: cov_type_lower,
        // r_squared/r_squared_adj/log_likelihood/aic/bicは`estimator`（変換後データに対する
        // OLS）ではなく`wls_estimator`側の値を使う。元の（変換前の）y・weightsを使って
        // 計算し直したもので、`estimator`側の値は変換のヤコビアン補正等が欠けており
        // statsmodelsと一致しない（`engine::linear::wls`モジュール冒頭のdocコメント参照）。
        r_squared: wls_estimator.r_squared(),
        r_squared_adj: wls_estimator.r_squared_adj(),
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
        log_likelihood: wls_estimator.log_likelihood(),
        aic: wls_estimator.aic(),
        bic: wls_estimator.bic(),
    })
}
