//! `nonlinear`系統（Logit/Probit/Tobit等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//!
//! `MleError`の`Common`バリアント（`engine::error::CommonError`、`linear`系統の
//! `LeastSquaresError`と共有する6種のバリデーションエラー）は`crate::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::nonlinear::common::MleError;
use pyo3::PyErr;

use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};

/// `engine::nonlinear::common::MleError`をPython例外に変換する。
///
/// `MleError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの
/// 外で定義された型のため、orphan rule（`impl`の対象は自クレート内で定義された
/// トレイトか型のどちらかを含む必要がある）により`impl From<MleError> for PyErr`は
/// 書けない。関数として実装し、呼び出し側で`.map_err(mle_error_to_pyerr)?`する
/// （`least_squares_error_to_pyerr`と同じ理由、`engine_pybind/src/linear/common.rs`参照）。
///
/// 対応表は`docs/planning/specs/nonlinear-implementation-notes.md`「エラー型: nonlinear系統で
/// 共有（MleError）」参照。
pub(crate) fn mle_error_to_pyerr(err: MleError) -> PyErr {
    match err {
        MleError::Common(common) => common_error_to_pyerr(common),
        MleError::InvalidMaxIter { .. }
        | MleError::InvalidTol { .. }
        | MleError::InvalidCensoringBounds { .. }
        | MleError::InvalidBinaryY { .. } => ValidationError::new_err(err.to_string()),
        MleError::NonConvergence { .. }
        | MleError::SingularHessian
        | MleError::SingularOpgMatrix => ComputationError::new_err(err.to_string()),
    }
}

/// `faer::Mat<f64>`を行指向の`Vec<Vec<f64>>`に変換する（`pred_table`のような小さい
/// 2次元行列をPython側にそのまま返すため）。`engine_pybind/src/linear/common.rs`の
/// `mat_to_vec`は列ベクトル（n×1）専用のため、任意の形状に対応するこちらを別途用意した。
pub(crate) fn mat_to_nested_vec(mat: &faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..mat.nrows())
        .map(|i| (0..mat.ncols()).map(|j| *mat.get(i, j)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_to_nested_vec_preserves_row_major_order() {
        let mat = faer::Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
        assert_eq!(
            mat_to_nested_vec(&mat),
            vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]
        );
    }

    #[test]
    fn mat_to_nested_vec_handles_single_row() {
        let mat = faer::Mat::from_fn(1, 2, |_, j| j as f64);
        assert_eq!(mat_to_nested_vec(&mat), vec![vec![0.0, 1.0]]);
    }
}
