//! 系統（`linear`/`nonlinear`等）をまたいで共有する、統計手法に依存しない純粋な
//! 線形代数ユーティリティ（`.claude/rules/rust-style.md`「全手法で共有するロジック」参照）。

use super::error::CommonError;
use faer::linalg::solvers::ColPivQr;
use faer::{Mat, Side};

/// [`checked_col_piv_qr`]がランク落ちを検出したことを表す、系統に依存しないエラー。
/// 呼び出し側が`map_err`で自系統のエラー（`LeastSquaresError::SingularMatrix`・
/// `MleError::SingularHessian`・`MleError::SingularDesignMatrix`等）に変換する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankDeficient;

/// `a`（m×n）を列ピボットQR分解し、フルカラムランクならそのQR分解を返す。ランク落ちなら
/// [`RankDeficient`]を返す。OLS/WLSの設計行列、Newton法のHessian、Logit/Probit/Tobitの
/// 設計行列の特異性判定が共有する。
///
/// 判定は`R`の対角成分に対する相対閾値（`n * f64::EPSILON * max|R_ii|`、
/// `.claude/rules/rust-style.md`「線形代数」。列ごとの一様スケーリングに対して不変）。
/// `col_piv_qr`は列ピボットにより対角成分が絶対値の降順になるため、最大値を基準にする。
///
/// **NaNを明示的に弾く**: 全ゼロの行列では`col_piv_qr`が列選択時の0除算で`R`の対角成分に
/// NaNを生成しうる（faer 0.24.4で実機確認済み）。NaNとの比較は常に`false`になるため
/// `diag <= threshold`だけだとすり抜ける。
///
/// `m < n`（行数が列数に満たない）は必ずランク落ちとして扱う。`n == 0`（列が無い行列）は
/// 検査する対角成分が無いため成功を返す。
pub fn checked_col_piv_qr(a: &Mat<f64>) -> Result<ColPivQr<f64>, RankDeficient> {
    let n = a.ncols();
    if a.nrows() < n {
        return Err(RankDeficient);
    }
    let qr = a.col_piv_qr();
    let r = qr.thin_R();
    let max_abs_diag = (0..n).map(|i| (*r.get(i, i)).abs()).fold(0.0_f64, f64::max);
    let threshold = (n as f64) * f64::EPSILON * max_abs_diag;
    for i in 0..n {
        let diag = (*r.get(i, i)).abs();
        if diag.is_nan() || diag <= threshold {
            return Err(RankDeficient);
        }
    }
    Ok(qr)
}

/// 対称正定値のはずの行列`v`（k×k）が数値的にほぼ特異でないことを、固有値分解
/// （`SelfAdjointEigen`）による相対閾値判定で確認する。
///
/// 非ピボットCholesky分解（`Llt`）のL因子対角成分は、行列の成分間のスケール差に
/// 起因する数値的なほぼ特異性を検出できない（OLSの`wald_f_test`で実測確認済み。
/// `col_piv_qr`のR対角成分を使う`checked_col_piv_qr`とは異なり、
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
/// max_abs_eigenvalue`以下、`checked_col_piv_qr`と同じ相対閾値の考え方）、
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

/// [`independent_columns_in_order`]の既定の許容誤差（Rの`lm`の`tol = 1e-7`と同じ）。
pub const COLLINEARITY_TOLERANCE: f64 = 1e-7;

/// 列を与えられた順に見て、それより前の（独立と判定した）列の線形結合で表せる列を見つける。
/// 返す配列は列ごとに「前の列から独立か」（`true`なら残す、`false`なら従属で落とす）。
///
/// Rの`lm`（LINPACK `dqrdc2`）と同じ規則: 前の独立な列に射影した残差のノルムが、元の列のノルムの
/// `tolerance`倍以下なら従属とみなす（列ごとの一様スケーリングに対して不変）。ピボットの
/// 並べ替えをしないため、対称な別名（`x`と`x̄`が同じ）では後ろの列が落ちる。`checked_col_piv_qr`
/// （全列がフルランクかの判定）と違い、どの列を落とすかを呼び出し側の列順で決めたいときに使う。
/// 全ゼロの列と、非有限値を含む列は従属として扱う。直交化は2回行い（再直交化）、近共線な列でも
/// 射影後の残差の丸め誤差を抑える。
///
/// 全ての列は同じ長さでなければならない（呼び出し側の契約）。
pub fn independent_columns_in_order(columns: &[&[f64]], tolerance: f64) -> Vec<bool> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    columns
        .iter()
        .map(|col| {
            let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() || norm == 0.0 {
                return false;
            }
            let mut residual = col.to_vec();
            for _ in 0..2 {
                for q in &basis {
                    let dot: f64 = residual.iter().zip(q).map(|(a, b)| a * b).sum();
                    for (r, b) in residual.iter_mut().zip(q) {
                        *r -= dot * b;
                    }
                }
            }
            let residual_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
            if residual_norm <= tolerance * norm {
                return false;
            }
            basis.push(residual.iter().map(|v| v / residual_norm).collect());
            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_columns_in_order_drops_the_later_of_two_identical_columns() {
        let a = [1.0, 2.0, 3.0, 5.0];
        let b = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(
            independent_columns_in_order(&[&b, &a, &a], COLLINEARITY_TOLERANCE),
            vec![true, true, false]
        );
    }

    #[test]
    fn independent_columns_in_order_drops_a_linear_combination_of_earlier_columns() {
        let ones = [1.0; 5];
        let x = [1.0, 4.0, 2.0, 8.0, 5.0];
        let z = [3.0, -1.0, 0.5, 2.0, 7.0];
        let combo: Vec<f64> = (0..5).map(|i| 2.0 * x[i] - 3.0 * z[i] + 1.0e6).collect();
        // 落ちるのは列順で後ろの列: 結合の列が後ろなら結合の列、前なら`z`。
        assert_eq!(
            independent_columns_in_order(&[&ones, &x, &z, &combo], COLLINEARITY_TOLERANCE),
            vec![true, true, true, false]
        );
        assert_eq!(
            independent_columns_in_order(&[&ones, &x, &combo, &z], COLLINEARITY_TOLERANCE),
            vec![true, true, true, false]
        );
        // `z`と結合の列を入れ替えても、前の2列の独立性は変わらない。
        assert_eq!(
            independent_columns_in_order(&[&z, &combo, &ones, &x], COLLINEARITY_TOLERANCE),
            vec![true, true, true, false]
        );
    }

    #[test]
    fn independent_columns_in_order_is_invariant_to_column_scaling() {
        let ones = [1.0; 4];
        let x = [1.0, 2.0, 4.0, 3.0];
        let scaled: Vec<f64> = x.iter().map(|v| v * 1.0e9).collect();
        let tiny: Vec<f64> = x.iter().map(|v| v * 1.0e-9).collect();
        assert_eq!(
            independent_columns_in_order(&[&ones, &x, &scaled, &tiny], COLLINEARITY_TOLERANCE),
            vec![true, true, false, false]
        );
    }

    #[test]
    fn independent_columns_in_order_keeps_nearly_but_not_collinear_columns() {
        // 相関が1に近くても、相対残差が許容誤差（1e-7）より大きければ残す。
        let ones = [1.0; 6];
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let near: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v + 1.0e-3 * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let nearer: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v + 1.0e-9 * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert_eq!(
            independent_columns_in_order(&[&ones, &x, &near], COLLINEARITY_TOLERANCE),
            vec![true, true, true]
        );
        assert_eq!(
            independent_columns_in_order(&[&ones, &x, &nearer], COLLINEARITY_TOLERANCE),
            vec![true, true, false]
        );
    }

    #[test]
    fn independent_columns_in_order_treats_zero_and_non_finite_columns_as_dependent() {
        let ones = [1.0; 3];
        let zeros = [0.0; 3];
        let nan = [1.0, f64::NAN, 2.0];
        assert_eq!(
            independent_columns_in_order(&[&ones, &zeros, &nan], COLLINEARITY_TOLERANCE),
            vec![true, false, false]
        );
        assert!(independent_columns_in_order(&[], COLLINEARITY_TOLERANCE).is_empty());
    }

    #[test]
    fn checked_col_piv_qr_returns_qr_for_full_rank_matrix() {
        let a = Mat::from_fn(4, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        assert!(checked_col_piv_qr(&a).is_ok());
    }

    #[test]
    fn checked_col_piv_qr_rejects_perfectly_collinear_columns() {
        let a = Mat::from_fn(4, 2, |i, j| (i as f64 + 1.0) * (j as f64 + 1.0));
        assert_eq!(checked_col_piv_qr(&a).unwrap_err(), RankDeficient);
    }

    #[test]
    fn checked_col_piv_qr_detects_nan_diagonal_from_all_zero_matrix() {
        // 全ゼロ行列のcol_piv_qrは列選択時の0除算によりRの対角成分がNaNになりうる
        // （faer 0.24.4で実機確認済み）。`diag <= threshold`のみだとNaNとの比較は
        // 常にfalseになりすり抜けてしまうため、`diag.is_nan()`の明示チェックが必要。
        let zeros = Mat::<f64>::zeros(4, 2);
        let diag = (*zeros.col_piv_qr().thin_R().get(0, 0)).abs();
        assert!(diag.is_nan(), "precondition: expected R diagonal to be NaN");

        assert_eq!(checked_col_piv_qr(&zeros).unwrap_err(), RankDeficient);
    }

    #[test]
    fn checked_col_piv_qr_accepts_matrix_without_columns() {
        // 固定効果のみのモデル（`x=[]`）のwithin変換後の設計行列は0列になる。
        let a = Mat::<f64>::zeros(4, 0);
        assert!(checked_col_piv_qr(&a).is_ok());
    }

    #[test]
    fn checked_col_piv_qr_rejects_matrix_with_fewer_rows_than_columns() {
        let a = Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64 + 1.0);
        assert_eq!(checked_col_piv_qr(&a).unwrap_err(), RankDeficient);
    }

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
