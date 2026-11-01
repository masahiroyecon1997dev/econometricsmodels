//! Python境界の`fit`: 列の抽出・検証から`engine::linear::ols`の呼び出し、`OLSResult`の組み立てまで。

use crate::linear::common::{
    check_cov_options, least_squares_error_to_pyerr, parse_cov_kind, resolve_cov_type,
};
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{SelectedData, extract_f64_column, extract_f64_columns};
use crate::shared::mat_conversion::{mat_to_rows, mat_to_vec};
use crate::shared::validation::validate_common_roles;
use engine::linear::ols::{OlsEstimator, OlsInput};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::options::OLSOptions;
use super::result::OLSResult;

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
        cov_params: mat_to_rows(estimator.cov_params()),
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
