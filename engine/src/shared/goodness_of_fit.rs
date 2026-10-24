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

/// 完全フィット（残差が丸め誤差の範囲で0）か。`SSR <= k * epsilon * SST`。
///
/// このとき`sigma^2`が0になり、標準誤差とそれを使う検定統計量は定義できない（0割りの無意味な値
/// か、`F`が`1e35`のような桁になる）。残差の大きさが厳密に0かどうかは丸め誤差で決まるため、
/// 厳密な0ではなく、総平方和に対する相対閾値で判定する（Breusch-Godfrey検定の補助回帰と同じ
/// 発想。`1 - R^2 <= k * epsilon`と同じ）。`sst`は呼び出し側のモデルの総平方和（切片があれば
/// 中心化、なければ非中心化、重み付きなら重み付き）で、`k`は推定した係数の数（1未満は1）。
/// `sst`が0なら`ssr`も0のときだけ真になる（`ssr`は非負）。
pub(crate) fn is_perfect_fit(ssr: f64, sst: f64, k: usize) -> bool {
    ssr <= (k.max(1) as f64) * f64::EPSILON * sst
}

/// 被説明変数の総平方和。`has_intercept`なら`sum (y - mean)^2`、なければ`sum y^2`
/// （重みがあれば`w`を掛け、平均は重み付き平均）。R-squaredと完全フィットの判定が共有する。
pub(crate) fn total_sum_of_squares(y: &[f64], weights: Option<&[f64]>, has_intercept: bool) -> f64 {
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    if has_intercept {
        let weight_sum: f64 = (0..y.len()).map(w).sum();
        let mean = (0..y.len()).map(|i| w(i) * y[i]).sum::<f64>() / weight_sum;
        (0..y.len()).map(|i| w(i) * (y[i] - mean).powi(2)).sum()
    } else {
        (0..y.len()).map(|i| w(i) * y[i] * y[i]).sum()
    }
}

/// `y`（n×1）に`k`個の係数（切片を含む）をあてはめた残差平方和`ssr`から適合度統計量を求める。
///
/// - `R² = 1 - SSR/SST`。`has_intercept`なら`SST = Σ(y-ȳ)²`、なければ`Σy²`（非中心化）。
/// - 調整済みR²は切片の有無で自由度の分子を変える（`n - k_constant`）。
/// - 対数尤度は最尤推定量`σ̂² = SSR/n`ベース（classical標準誤差の`SSR/(n-k)`とは異なる）。
/// - AIC/BICの罰則項の乗数は`k`。FE等で乗数を差し替えたいときは`log_likelihood`だけ使う。
///
/// 前提: `n > k`（`n - k`は`usize`の減算）。呼び出し側が観測数の検証を済ませていること。
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
    fn perfect_fit_is_residuals_within_k_epsilon_of_the_total_sum_of_squares() {
        let sst = 100.0;
        let k = 2;
        let limit = (k as f64) * f64::EPSILON * sst;

        assert!(is_perfect_fit(0.0, sst, k));
        assert!(is_perfect_fit(limit, sst, k));
        assert!(!is_perfect_fit(limit * 1.01, sst, k));
        // 係数の数が多いほど閾値が緩む。k=0（係数なし）はk=1として扱う。
        assert!(is_perfect_fit(limit * 1.5, sst, 3));
        assert_eq!(
            is_perfect_fit(f64::EPSILON * sst, sst, 0),
            is_perfect_fit(f64::EPSILON * sst, sst, 1)
        );
        // 総平方和が0なら、残差も0のときだけ完全フィット。
        assert!(is_perfect_fit(0.0, 0.0, 2));
        assert!(!is_perfect_fit(1e-30, 0.0, 2));
        // 実用的な大きさの残差は完全フィットではない。
        assert!(!is_perfect_fit(1.0, 100.0, 2));
    }

    #[test]
    fn total_sum_of_squares_centers_only_with_an_intercept() {
        let y = [1.0, 2.0, 3.0, 4.0];

        assert!((total_sum_of_squares(&y, None, true) - 5.0).abs() < 1e-12);
        assert!((total_sum_of_squares(&y, None, false) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn total_sum_of_squares_uses_the_weighted_mean_with_weights() {
        // 重み付き平均 = (1*1 + 3*2 + 1*6)/5 = 2.6。SST = 1*2.56 + 3*0.36 + 1*11.56 = 15.2。
        let y = [1.0, 2.0, 6.0];
        let w = [1.0, 3.0, 1.0];

        assert!((total_sum_of_squares(&y, Some(&w), true) - 15.2).abs() < 1e-12);
        // 切片なしは非中心化: 1*1 + 3*4 + 1*36 = 49。
        assert!((total_sum_of_squares(&y, Some(&w), false) - 49.0).abs() < 1e-12);
    }

    #[test]
    fn total_sum_of_squares_matches_the_matrix_form_used_for_r_squared() {
        // 重みなしの値は、R-squaredに使う`gaussian_goodness_of_fit`の総平方和と同じ。
        let y = [0.1, 0.1, 0.1, 0.7, 1.3];
        let fit = gaussian_goodness_of_fit(&column(&y), 0.5, 2, true);
        let sst = total_sum_of_squares(&y, None, true);

        assert!((fit.r_squared - (1.0 - 0.5 / sst)).abs() < 1e-15);
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
