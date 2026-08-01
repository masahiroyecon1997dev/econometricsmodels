//! `linear`系統（OLS/WLS等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く。以前はOLSしかなく未作成だったが、WLSが`LeastSquaresError`の
//! エラー変換・`Mat<f64>`→`Vec<f64>`変換の両方をOLSと共有する形で実装されたため作成した
//! （`docs/planning/specs/wls-api-design.md`4.2節）。
//!
//! `LeastSquaresError`は元々`OlsError`という名前だったが、OLS単体のエラー型ではなくWLSも
//! 含む`linear`系統共通のエラー型であることを名前に反映するため、`engine`側で
//! `engine::linear::common::LeastSquaresError`に改名・移動した。
//!
//! `LeastSquaresError`の`Common`バリアント（`engine::error::CommonError`、nonlinear系統の
//! `MleError`と共有する6種のバリデーションエラー）は`crate::errors::common_error_to_pyerr`
//! に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::linear::common::LeastSquaresError;
use pyo3::PyErr;

use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};

/// `engine::linear::common::LeastSquaresError`をPython例外に変換する。
///
/// `LeastSquaresError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの
/// 外で定義された型のため、orphan rule（`impl`の対象は自クレート内で定義された
/// トレイトか型のどちらかを含む必要がある）により`impl From<LeastSquaresError> for PyErr`は
/// 書けない。関数として実装し、呼び出し側で`.map_err(least_squares_error_to_pyerr)?`する。
///
/// 対応表は`docs/spec/ols-spec.md`「engine/engine_pybind間のデータ受け渡し・エラー変換」参照。
pub(crate) fn least_squares_error_to_pyerr(err: LeastSquaresError) -> PyErr {
    match err {
        LeastSquaresError::Common(common) => common_error_to_pyerr(common),
        LeastSquaresError::WeightDimensionMismatch { .. }
        | LeastSquaresError::NonPositiveWeight { .. }
        | LeastSquaresError::InvalidHacLags { .. } => ValidationError::new_err(err.to_string()),
        LeastSquaresError::SingularMatrix => ComputationError::new_err(err.to_string()),
    }
}

/// `faer::Mat<f64>`（n×1またはk×1の列ベクトル）を`Vec<f64>`に変換する。
pub(crate) fn mat_to_vec(mat: &faer::Mat<f64>) -> Vec<f64> {
    (0..mat.nrows()).map(|i| *mat.get(i, 0)).collect()
}
