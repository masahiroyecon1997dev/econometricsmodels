//! 正規誤差を仮定した最小二乗の適合度統計量（R²・調整済みR²・対数尤度・AIC・BIC）。
//!
//! 重み付き最小二乗（WLS）は重み付き平均のTSSと変換のヤコビアン補正が要るため、この式を
//! そのまま使ってはいけない（`WlsEstimator`が別に計算している）。

use faer::Mat;

/// [`gaussian_goodness_of_fit`]の結果。
#[derive(Debug, Clone, Copy)]
pub(crate) struct GaussianGoodnessOfFit {
    pub(crate) r_squared: f64,
    pub(crate) adj_r_squared: f64,
    pub(crate) log_likelihood: f64,
    pub(crate) aic: f64,
    pub(crate) bic: f64,
}

/// `y`（n×1）に`k`個の係数（切片を含む）をあてはめた残差平方和`ssr`から適合度統計量を求める。
///
/// - `R² = 1 - SSR/SST`。`has_intercept`なら`SST = Σ(y-ȳ)²`、なければ`Σy²`（非中心化）。
/// - 調整済みR²は切片の有無で自由度の分子を変える（`n - k_constant`）。
/// - 対数尤度は最尤推定量`σ̂² = SSR/n`ベース（classical標準誤差の`SSR/(n-k)`とは異なる）。
/// - AIC/BICの罰則項の乗数は`k`。FE等で乗数を差し替えたいときは`log_likelihood`だけ使う。
pub(crate) fn gaussian_goodness_of_fit(
    y: &Mat<f64>,
    ssr: f64,
    k: usize,
    has_intercept: bool,
) -> GaussianGoodnessOfFit {
    let n = y.nrows();
    let df_resid = n - k;
    let k_constant = usize::from(has_intercept);

    let sst: f64 = if has_intercept {
        let y_mean: f64 = (0..n).map(|i| *y.get(i, 0)).sum::<f64>() / (n as f64);
        (0..n).map(|i| (*y.get(i, 0) - y_mean).powi(2)).sum()
    } else {
        (0..n).map(|i| (*y.get(i, 0)).powi(2)).sum()
    };
    let r_squared = 1.0 - ssr / sst;
    let adj_r_squared = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

    let log_likelihood =
        -(n as f64 / 2.0) * ((2.0 * std::f64::consts::PI).ln() + (ssr / n as f64).ln() + 1.0);
    let aic = -2.0 * log_likelihood + 2.0 * (k as f64);
    let bic = -2.0 * log_likelihood + (n as f64).ln() * (k as f64);

    GaussianGoodnessOfFit {
        r_squared,
        adj_r_squared,
        log_likelihood,
        aic,
        bic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(values: &[f64]) -> Mat<f64> {
        Mat::from_fn(values.len(), 1, |i, _| values[i])
    }

    #[test]
    fn with_intercept_uses_the_centered_total_sum_of_squares() {
        // y = [1, 2, 3, 4]: ȳ=2.5, SST = 2.25+0.25+0.25+2.25 = 5。SSR=1のとき R²=0.8。
        let fit = gaussian_goodness_of_fit(&column(&[1.0, 2.0, 3.0, 4.0]), 1.0, 2, true);

        assert!((fit.r_squared - 0.8).abs() < 1e-12);
        // 1 - ((n-1)/(n-k))(1-R²) = 1 - (3/2)(0.2) = 0.7
        assert!((fit.adj_r_squared - 0.7).abs() < 1e-12);
    }

    #[test]
    fn without_intercept_uses_the_uncentered_total_sum_of_squares() {
        // SST = 1+4+9+16 = 30。SSR=3のとき R²=0.9。n=4・k=3なので`df_resid=1`で、
        // 調整済みは 1 - (4/1)(0.1)。
        let fit = gaussian_goodness_of_fit(&column(&[1.0, 2.0, 3.0, 4.0]), 3.0, 3, false);

        assert!((fit.r_squared - 0.9).abs() < 1e-12);
        assert!((fit.adj_r_squared - (1.0 - (4.0 / 1.0) * 0.1)).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_and_information_criteria_follow_the_gaussian_mle_formulas() {
        let n = 4.0_f64;
        let ssr = 2.0_f64;
        let fit = gaussian_goodness_of_fit(&column(&[1.0, 2.0, 3.0, 5.0]), ssr, 2, true);

        let expected_ll = -(n / 2.0) * ((2.0 * std::f64::consts::PI).ln() + (ssr / n).ln() + 1.0);
        assert!((fit.log_likelihood - expected_ll).abs() < 1e-12);
        assert!((fit.aic - (-2.0 * expected_ll + 4.0)).abs() < 1e-12);
        assert!((fit.bic - (-2.0 * expected_ll + n.ln() * 2.0)).abs() < 1e-12);
    }
}
