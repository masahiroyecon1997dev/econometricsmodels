//! Python側に公開する例外クラス。
//!
//! `.claude/rules/rust-style.md`「エラーハンドリング」の方針に基づき、
//! 全エラーを`PyValueError`にまとめず、カテゴリ別に分ける。
//!
//! - `ValidationError`: 入力・パラメータが不正（列が存在しない、型が数値でない、
//!   欠損値がある、観測数不足、confidence_levelの範囲外等）。
//! - `ComputationError`: 計算過程で発覚した問題（特異行列等）。
//!
//! Python側では以下のように使う想定:
//! ```python
//! from econometricsmodels import ValidationError, ComputationError
//! try:
//!     result = ols(data=df, y="wage", x=["educ"], options=options)
//! except ValidationError as e:
//!     ...  # 入力を直すべきケース
//! except ComputationError as e:
//!     ...  # データの統計的な性質に起因するケース（特異行列等）
//! ```

use engine::error::CommonError;
use pyo3::PyErr;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

// 第1引数はPythonモジュール名。python_package側の実際のモジュール名に合わせて要調整。
create_exception!(econometricsmodels, ValidationError, PyValueError);
create_exception!(econometricsmodels, ComputationError, PyRuntimeError);

/// `engine::error::CommonError`をPython例外に変換する。
///
/// `engine::linear::common::LeastSquaresError`・`engine::nonlinear::common::MleError`
/// 等、`CommonError`を`#[error(transparent)] Common(#[from] CommonError)`で包む
/// 各系統のエラー型は、`Common`アームでこの関数に委譲する（Issue #113。系統ごとに
/// 同じ判定ロジックを重複させない）。orphan ruleにより`impl From`ではなく関数として
/// 実装する（`least_squares_error_to_pyerr`と同じ理由、`engine_pybind/src/linear/
/// CLAUDE.md`参照）。
pub(crate) fn common_error_to_pyerr(err: CommonError) -> PyErr {
    match err {
        CommonError::DimensionMismatch { .. }
        | CommonError::InsufficientObservations { .. }
        | CommonError::InvalidConfidenceLevel { .. }
        | CommonError::MissingClusterColumn
        | CommonError::InsufficientClusters { .. } => ValidationError::new_err(err.to_string()),
        CommonError::ComputationFailed(_) => ComputationError::new_err(err.to_string()),
    }
}
