//! 最小二乗の係数・残差・`(X'X)⁻¹`を求める部品。
//!
//! OLS/WLSの`OlsEstimator::fit`と、within変換後のデータに最小二乗をあてはめるFEが共有する。
//! 推論（標準誤差・検定）は含まない。呼び出し側が[`super::covariance`]等で組み立てる。

use faer::Mat;
use faer::prelude::SolveLstsq;

use super::covariance::xtx_inverse;
use super::linear_algebra::{RankDeficient, checked_col_piv_qr};

/// [`least_squares`]の結果。
#[derive(Debug)]
pub(crate) struct LeastSquaresFit {
    /// 係数 (k, 1)
    pub(crate) params: Mat<f64>,
    /// 残差 (n, 1) = y - Xβ̂
    pub(crate) residuals: Mat<f64>,
    /// `(X'X)⁻¹` (k, k)
    pub(crate) xtx_inv: Mat<f64>,
}

/// `y`（n×1）を`x`（n×k）に列ピボットQRであてはめる。
///
/// `X'Xβ=X'y`をCholeskyで解く方式は使わない（`X'X`の条件数が2乗になり不利。QRなら特異性の
/// 検出と計算を同じ分解で行える）。`(X'X)⁻¹`は標準誤差の計算に使うため`X'X`自体の
/// Cholesky分解で別途求める（`R`因子から導く案は実測で速くならなかった）。
///
/// `k = 0`（列が無い`x`）も受理し、係数が空・残差が`y`そのものの結果を返す
/// （固定効果のみのモデルのwithin変換後の設計行列が該当する）。
///
/// # Errors
/// `x`がフルカラムランクでない（完全な多重共線性等）、または`X'X`のCholesky分解が
/// 丸めで失敗した場合に[`RankDeficient`]を返す。後者は前者の判定を通った時点で理論上
/// 起こらないが、境界的なケースに備えて`Result`にしている。
pub(crate) fn least_squares(x: &Mat<f64>, y: &Mat<f64>) -> Result<LeastSquaresFit, RankDeficient> {
    let qr = checked_col_piv_qr(x)?;
    let params = qr.solve_lstsq(y);
    let residuals = y - x * &params;
    let xtx_inv = xtx_inverse(x)?;
    Ok(LeastSquaresFit {
        params,
        residuals,
        xtx_inv,
    })
}

/// 残差平方和 `Σ e_i²`。
pub(crate) fn residual_sum_of_squares(residuals: &Mat<f64>) -> f64 {
    (0..residuals.nrows())
        .map(|i| (*residuals.get(i, 0)).powi(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(values: &[f64]) -> Mat<f64> {
        Mat::from_fn(values.len(), 1, |i, _| values[i])
    }

    #[test]
    fn least_squares_recovers_exact_coefficients() {
        // y = 1 + 2x を厳密に満たすデータ。
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let x = Mat::from_fn(5, 2, |i, j| if j == 0 { 1.0 } else { xs[i] });
        let y = column(&xs.map(|v| 1.0 + 2.0 * v));

        let fit = least_squares(&x, &y).unwrap();

        assert!((*fit.params.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((*fit.params.get(1, 0) - 2.0).abs() < 1e-12);
        for i in 0..5 {
            assert!((*fit.residuals.get(i, 0)).abs() < 1e-12);
        }
    }

    #[test]
    fn least_squares_returns_the_inverse_gram_matrix() {
        let x = Mat::from_fn(4, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let y = column(&[1.0, 3.0, 2.0, 5.0]);

        let fit = least_squares(&x, &y).unwrap();

        let identity = &(x.transpose() * &x) * &fit.xtx_inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((*identity.get(i, j) - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn least_squares_residuals_are_orthogonal_to_the_columns() {
        let x = Mat::from_fn(6, 2, |i, j| if j == 0 { 1.0 } else { (i * i) as f64 });
        let y = column(&[2.0, 1.0, 4.0, 3.0, 8.0, 5.0]);

        let fit = least_squares(&x, &y).unwrap();

        let xte = x.transpose() * &fit.residuals;
        for j in 0..2 {
            assert!((*xte.get(j, 0)).abs() < 1e-9);
        }
    }

    #[test]
    fn least_squares_rejects_perfectly_collinear_columns() {
        let x = Mat::from_fn(4, 2, |i, j| (i as f64 + 1.0) * (j as f64 + 1.0));
        let y = column(&[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(least_squares(&x, &y).unwrap_err(), RankDeficient);
    }

    #[test]
    fn least_squares_rejects_all_zero_design_matrix() {
        // 全ゼロ行列はランク落ちとして弾かれる。NaN明示チェック単体の検証は
        // `checked_col_piv_qr_detects_nan_diagonal_from_all_zero_matrix`が担う（ここでは
        // 後段の`xtx_inverse`の失敗でも`RankDeficient`になるため区別できない）。
        let x = Mat::<f64>::zeros(4, 2);
        let y = column(&[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(least_squares(&x, &y).unwrap_err(), RankDeficient);
    }

    #[test]
    fn least_squares_succeeds_without_columns_and_returns_y_as_residuals() {
        // 固定効果のみのモデル（`x=[]`）のwithin変換後の設計行列は0列。
        let x = Mat::<f64>::zeros(4, 0);
        let y = column(&[1.0, -2.0, 3.0, 0.5]);

        let fit = least_squares(&x, &y).unwrap();

        assert_eq!(fit.params.nrows(), 0);
        assert_eq!(fit.xtx_inv.nrows(), 0);
        for i in 0..4 {
            assert_eq!(*fit.residuals.get(i, 0), *y.get(i, 0));
        }
    }

    #[test]
    fn residual_sum_of_squares_adds_up_squared_residuals() {
        let residuals = column(&[1.0, -2.0, 3.0]);
        assert_eq!(residual_sum_of_squares(&residuals), 14.0);
    }
}
