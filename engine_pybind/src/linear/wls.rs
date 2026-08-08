//! WLSの推定結果、およびPython（polars DataFrame + `y`/`x`/`weight`列名 + オプション）から
//! `engine::linear::wls::WlsEstimator`を呼び出し、結果をPython側に返すところまでの一連の処理。
//!
//! `weight`は`y`と同じく`data`内の列名を指すトップレベル引数として扱う（`WLSOptions`という
//! 専用のOptions型は新設せず`OLSOptions`をそのまま再利用する。
//! `docs/spec/wls-spec.md`「API引数」参照）。エラー変換（`least_squares_error_to_pyerr`）・
//! `Mat<f64>`→`Vec<f64>`変換（`mat_to_vec`）は`super::common`のものをそのまま再利用する
//! （`LeastSquaresError`がOLS・WLS共通のエラー型のため。`.claude/rules/rust-style.md`
//! 「系統内で共有するロジックはcommon.rsに置く」）。

use engine::linear::ols::CovType as EngineCovType;
use engine::linear::wls::WlsEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use super::common::{least_squares_error_to_pyerr, mat_to_vec};
use super::ols::OLSOptions;
use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::ValidationError;
use crate::validation::{
    validate_no_const_collision, validate_no_duplicate_roles, validate_no_duplicate_x,
    validate_x_non_empty,
};

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
/// - `y`・`x`・`weight`の重複、`include_intercept=true`のときの`"const"`列との衝突は
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
    options: &OLSOptions,
) -> PyResult<WLSResult> {
    let df: DataFrame = data.into();
    let cov_type_lower = options.cov_type.to_lowercase();

    // 誤って同じ列を複数の役割に指定するミスを、分かりやすいエラーで早期に防ぐ
    // （`docs/spec/wls-spec.md`「API引数」参照）。
    validate_x_non_empty(&x)?;
    validate_no_duplicate_roles(&[("y", &y), ("weight", &weight)], &x)?;
    validate_no_duplicate_x(&x)?;
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
