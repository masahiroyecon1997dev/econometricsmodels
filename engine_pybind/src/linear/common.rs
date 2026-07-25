//! `linear`系統（OLS/WLS等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く。以前はOLSしかなく未作成だったが、WLSが`OlsError`のエラー変換・
//! `Mat<f64>`→`Vec<f64>`変換の両方をOLSと共有する形で実装されたため作成した
//! （`docs/planning/specs/wls-api-design.md`4.2節）。

use engine::linear::ols::OlsError;
use pyo3::PyErr;

use crate::errors::{ComputationError, ValidationError};

/// `engine::linear::ols::OlsError`をPython例外に変換する。
///
/// `OlsError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの
/// 外で定義された型のため、orphan rule（`impl`の対象は自クレート内で定義された
/// トレイトか型のどちらかを含む必要がある）により`impl From<OlsError> for PyErr`は
/// 書けない。関数として実装し、呼び出し側で`.map_err(ols_error_to_pyerr)?`する。
///
/// `OlsError`という名前だが、WLSも含む`linear`系統共通のエラー型として扱う
/// （`docs/planning/specs/wls-api-design.md`4.2節）。対応表は
/// `docs/planning/specs/ols-implementation-notes.md`「1. エラーハンドリング」参照。
pub(crate) fn ols_error_to_pyerr(err: OlsError) -> PyErr {
    match err {
        OlsError::DimensionMismatch { .. }
        | OlsError::WeightDimensionMismatch { .. }
        | OlsError::NonPositiveWeight { .. }
        | OlsError::InsufficientObservations { .. }
        | OlsError::MissingClusterColumn
        | OlsError::InvalidConfidenceLevel { .. }
        | OlsError::InsufficientClusters { .. }
        | OlsError::InvalidHacLags { .. } => ValidationError::new_err(err.to_string()),
        OlsError::SingularMatrix | OlsError::ComputationFailed(_) => {
            ComputationError::new_err(err.to_string())
        }
    }
}

/// `faer::Mat<f64>`（n×1またはk×1の列ベクトル）を`Vec<f64>`に変換する。
pub(crate) fn mat_to_vec(mat: &faer::Mat<f64>) -> Vec<f64> {
    (0..mat.nrows()).map(|i| *mat.get(i, 0)).collect()
}
