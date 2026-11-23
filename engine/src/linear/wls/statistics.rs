//! 元スケールの予測値・残差と、重み付きのR²・調整済みR²・対数尤度・AIC・BICの計算。
//! 統計量ごとの理由は`wls.rs`のモジュールdoc参照。

use crate::linear::ols::OlsEstimator;

/// R²・調整済みR²・対数尤度・AIC・BICを、元の（変換前の）`y`・`weights`から計算し直す。
///
/// 統計量ごとの理由はモジュール冒頭のdocコメント参照。SSRは元スケールの残差と`weights`から
/// `Σ w_i (y_i - ŷ_i)²`として計算し直す。`estimator`が変換後データに対して計算したSSRと
/// 数学的に同一の値になる（`sqrt(w_i)(y_i-ŷ_i)`の二乗が`w_i(y_i-ŷ_i)²`のため）が、
/// `OlsEstimator`はSSRを公開していないため、O(n)の再計算で得ている。
pub(super) fn weighted_fit_statistics(
    y: &[f64],
    weights: &[f64],
    residuals: &[f64],
    estimator: &OlsEstimator,
) -> (f64, f64, f64, f64, f64) {
    let n = estimator.input().nobs();
    let k = estimator.input().k();
    let k_constant = usize::from(estimator.input().has_intercept());
    let df_resid = n - k;

    let ssr: f64 = residuals.iter().zip(weights).map(|(r, w)| w * r * r).sum();

    let sst: f64 = if estimator.input().has_intercept() {
        let weight_sum: f64 = weights.iter().sum();
        let weighted_mean: f64 =
            y.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / weight_sum;
        y.iter()
            .zip(weights)
            .map(|(v, w)| w * (v - weighted_mean).powi(2))
            .sum()
    } else {
        y.iter().zip(weights).map(|(v, w)| w * v * v).sum()
    };

    let r_squared = 1.0 - ssr / sst;
    let adj_r_squared = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

    // statsmodels WLS.loglikeと同じ導出（変換後データに対するOLSの対数尤度に、
    // sqrt(weight)変換のヤコビアン補正項 +0.5*Σlog(w_i) を加える）。
    let sum_log_weights: f64 = weights.iter().map(|w| w.ln()).sum();
    let log_likelihood = -(n as f64 / 2.0)
        * ((2.0 * std::f64::consts::PI).ln() + (ssr / n as f64).ln() + 1.0)
        + 0.5 * sum_log_weights;
    let aic = -2.0 * log_likelihood + 2.0 * (k as f64);
    let bic = -2.0 * log_likelihood + (n as f64).ln() * (k as f64);

    (r_squared, adj_r_squared, log_likelihood, aic, bic)
}

/// 元の（重み変換前の）`y`・`x_columns`と推定済みの係数から、元スケールの予測値
/// `ŷ_i = x_i'β̂`と残差`y_i - ŷ_i`を計算する。`estimator.residuals()`（重み付き残差）を
/// そのまま使わない理由は`WlsEstimator`のdocコメントを参照。
pub(super) fn original_scale_fitted_and_residuals(
    y: &[f64],
    x_columns: &[Vec<f64>],
    include_intercept: bool,
    estimator: &OlsEstimator,
) -> (Vec<f64>, Vec<f64>) {
    let params = estimator.params();
    (0..y.len())
        .map(|i| {
            let mut fitted = 0.0;
            let mut col = 0;
            if include_intercept {
                fitted += *params.get(0, 0);
                col = 1;
            }
            for (j, x_col) in x_columns.iter().enumerate() {
                fitted += x_col[i] * *params.get(col + j, 0);
            }
            (fitted, y[i] - fitted)
        })
        .unzip()
}
