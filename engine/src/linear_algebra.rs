//! 系統（`linear`/`nonlinear`等）をまたいで共有する、統計手法に依存しない純粋な
//! 線形代数ユーティリティ（`.claude/rules/rust-style.md`「全手法で共有するロジック」参照）。

use crate::error::CommonError;
use faer::{Mat, Side};

/// 対称正定値のはずの行列`v`（k×k）が数値的にほぼ特異でないことを、固有値分解
/// （`SelfAdjointEigen`）による相対閾値判定で確認する。
///
/// 非ピボットCholesky分解（`Llt`）のL因子対角成分は、行列の成分間のスケール差に
/// 起因する数値的なほぼ特異性を検出できない（OLSの`wald_f_test`で実測確認済み。
/// `col_piv_qr`のR対角成分を使う`ensure_full_rank`とは異なり、
/// Choleskyはピボットしないため）。分散共分散行列（傾き係数の同時共分散部分行列、
/// 観測情報行列の逆行列、OPG行列の逆行列等）にCholesky分解を適用する前は、
/// この関数で固有値ベースの判定を先に行う必要がある。
///
/// `context`はエラーメッセージに埋め込む説明文字列（例:
/// `"coefficient covariance submatrix for the F-test"`）。呼び出し元ごとに
/// 意味のあるメッセージを出せるよう引数化している。
///
/// # Errors
/// 固有値の絶対値が最大固有値に対して相対的に小さすぎる場合（`k * f64::EPSILON *
/// max_abs_eigenvalue`以下、`ensure_full_rank`と同じ相対閾値の考え方）、
/// `CommonError::ComputationFailed`を返す。
pub fn ensure_well_conditioned_symmetric_matrix(
    v: &Mat<f64>,
    k: usize,
    context: &str,
) -> Result<(), CommonError> {
    debug_assert_eq!(v.nrows(), k, "v must be a k x k matrix (caller contract)");
    debug_assert_eq!(v.ncols(), k, "v must be a k x k matrix (caller contract)");

    // `v`は対称正定値のはずのため、理論上`SelfAdjointEigen::new`は失敗しない
    // （呼び出し元の`Llt`失敗と同様、浮動小数点演算の丸めによる境界的な失敗に
    // 備えた防御的な`Result`化）。
    let eigen =
        faer::linalg::solvers::SelfAdjointEigen::new(v.as_ref(), Side::Lower).map_err(|_| {
            CommonError::ComputationFailed(format!(
                "failed to compute eigendecomposition of {context}"
            ))
        })?;
    let eigenvalues = eigen.S().column_vector();
    let max_abs_eigenvalue = (0..k)
        .map(|i| (*eigenvalues.get(i)).abs())
        .fold(0.0_f64, f64::max);
    let threshold = (k as f64) * f64::EPSILON * max_abs_eigenvalue;

    for i in 0..k {
        if (*eigenvalues.get(i)).abs() <= threshold {
            return Err(CommonError::ComputationFailed(format!(
                "{context} is near-singular (condition number exceeds double-precision limits, \
                 e.g. due to extreme scale differences)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_well_conditioned_symmetric_matrix_accepts_well_conditioned_matrix() {
        let v = Mat::from_fn(2, 2, |i, j| if i == j { [2.0, 5.0][i] } else { 0.0 });
        assert!(ensure_well_conditioned_symmetric_matrix(&v, 2, "test matrix").is_ok());
    }

    #[test]
    fn ensure_well_conditioned_symmetric_matrix_rejects_exactly_singular_matrix() {
        let v = Mat::<f64>::zeros(2, 2);
        let result = ensure_well_conditioned_symmetric_matrix(&v, 2, "test matrix");
        assert!(matches!(result, Err(CommonError::ComputationFailed(_))));
    }

    #[test]
    fn ensure_well_conditioned_symmetric_matrix_rejects_extreme_scale_difference() {
        // スケール比1e6/1e-3相当の対角行列。非ピボットCholeskyのL因子対角成分では
        // 検出できないケース（OLSのwald_f_testで実測確認済み）だが、
        // 固有値ベースの判定なら検出できるはず。
        let v = Mat::from_fn(2, 2, |i, j| if i == j { [1e12, 1e-6][i] } else { 0.0 });
        let result = ensure_well_conditioned_symmetric_matrix(&v, 2, "test matrix");
        assert!(matches!(result, Err(CommonError::ComputationFailed(_))));
    }
}
