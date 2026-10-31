//! `engine`が返す`faer::Mat<f64>`をPython側に渡す`Vec<f64>`へ変換する共通処理。
//!
//! OLS/WLS（`linear`）・IV・固定効果/変量効果（`panel`）の結果構築が、係数・標準誤差・
//! 残差等の列ベクトル（n×1またはk×1）を同じ形で`Vec<f64>`にするため、系統の`common.rs`
//! ではなくここに置く（`.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」）。

/// `faer::Mat<f64>`（n×1またはk×1の列ベクトル）を`Vec<f64>`に変換する。
pub(crate) fn mat_to_vec(mat: &faer::Mat<f64>) -> Vec<f64> {
    (0..mat.nrows()).map(|i| *mat.get(i, 0)).collect()
}

/// `faer::Mat<f64>`（k×k等の行列）を行ごとの`Vec<Vec<f64>>`に変換する。
/// 係数の分散共分散行列（`cov_params`）をPython側へ渡すときに使う。
pub(crate) fn mat_to_rows(mat: &faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..mat.nrows())
        .map(|i| (0..mat.ncols()).map(|j| *mat.get(i, j)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_to_rows_preserves_row_and_column_order() {
        let mat = faer::Mat::from_fn(2, 3, |i, j| (10 * i + j) as f64);
        assert_eq!(
            mat_to_rows(&mat),
            vec![vec![0.0, 1.0, 2.0], vec![10.0, 11.0, 12.0]]
        );
    }

    #[test]
    fn mat_to_rows_handles_empty() {
        let mat = faer::Mat::<f64>::zeros(0, 0);
        assert!(mat_to_rows(&mat).is_empty());
    }

    #[test]
    fn mat_to_vec_preserves_row_order() {
        let mat = faer::Mat::from_fn(3, 1, |i, _| (i as f64) + 1.5);
        assert_eq!(mat_to_vec(&mat), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn mat_to_vec_handles_single_element() {
        let mat = faer::Mat::from_fn(1, 1, |_, _| -2.0);
        assert_eq!(mat_to_vec(&mat), vec![-2.0]);
    }

    #[test]
    fn mat_to_vec_handles_empty() {
        let mat = faer::Mat::<f64>::zeros(0, 1);
        assert!(mat_to_vec(&mat).is_empty());
    }

    #[test]
    fn mat_to_vec_reads_only_first_column() {
        // 列ベクトル専用：複数列の行列を渡しても0列目だけを読む（n×1以外は想定外の入力）。
        let mat = faer::Mat::from_fn(2, 2, |i, j| (10 * j + i) as f64);
        assert_eq!(mat_to_vec(&mat), vec![0.0, 1.0]);
    }
}
