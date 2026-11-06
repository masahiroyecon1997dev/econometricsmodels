//! `WlsEstimator::fit`の本体。`sqrt(weight)`で変換した入力を`OlsEstimator`で推定し、
//! 元スケールの予測値・残差と重み付きの適合度を組み立てる。契約は`wls.rs`のモジュールdocと
//! `WlsEstimator::fit`のdocコメント参照。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use super::WlsEstimator;
use super::statistics::{original_scale_fitted_and_residuals, weighted_fit_statistics};
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};

/// [`WlsEstimator::fit`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_wls(
    y: &[f64],
    x_columns: &[Vec<f64>],
    x_names: Vec<String>,
    include_intercept: bool,
    dep_var_name: String,
    weights: &[f64],
    cov_type: CovType,
    confidence_level: f64,
) -> Result<WlsEstimator, LeastSquaresError> {
    // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
    crate::shared::parallelism::ensure_serial();

    let input = OlsInput::from_columns_weighted(
        y,
        x_columns,
        x_names,
        include_intercept,
        dep_var_name,
        weights,
    )?;
    let estimator = OlsEstimator::fit(input, cov_type, confidence_level)?;
    let (fitted_values, residuals) =
        original_scale_fitted_and_residuals(y, x_columns, include_intercept, &estimator);
    let (r_squared, adj_r_squared, log_likelihood, aic, bic) =
        weighted_fit_statistics(y, weights, &residuals, &estimator);

    Ok(WlsEstimator {
        estimator,
        fitted_values,
        residuals,
        r_squared,
        adj_r_squared,
        log_likelihood,
        aic,
        bic,
    })
}

#[cfg(test)]
mod tests;
