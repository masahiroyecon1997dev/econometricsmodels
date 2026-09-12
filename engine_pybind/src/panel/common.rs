//! `panel`系統（FE/RE）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/iv/common.rs`と同じ位置づけ）。
//!
//! `PanelError`の`Common`バリアント（`engine::error::CommonError`）は`crate::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::panel::common::PanelError;
use pyo3::PyErr;

use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};
use crate::linear::common::least_squares_error_is_computation_error;

/// `engine::panel::common::PanelError`をPython例外に変換する。
///
/// `PanelError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの外で
/// 定義された型のため、orphan ruleにより`impl From<PanelError> for PyErr`は書けない。関数
/// として実装し、呼び出し側で`.map_err(panel_error_to_pyerr)?`する（`iv_error_to_pyerr`と
/// 同じ理由、`engine_pybind/src/iv/common.rs`参照）。
///
/// バリアントの分類方針:
/// - `Common`: `common_error_to_pyerr`へ委譲。
/// - FE/RE固有のバリデーションエラー（`IdentifierDimensionMismatch`・
///   `InsufficientDegreesOfFreedom`・`SingletonGroup`・`UnbalancedPanelForTwoWay`・
///   `ZeroVarianceAfterDemeaning`・`TwoWayRequiresTime`）はいずれも入力・オプションの
///   不正なので`ValidationError`。
/// - `WithinRegressionFailed`: 委譲先の`LeastSquaresError`の分類基準
///   （`least_squares_error_is_computation_error`）にそのまま従う。`IvError::
///   SecondStageFailed`と同じ扱い。Pythonに渡すメッセージは`source.to_string()`ではなく
///   `PanelError`自身の`to_string()`（「within変換後の推定で失敗した」という文脈を含む）を
///   使うため、`least_squares_error_to_pyerr`自体は呼ばない。
///
/// Issue #172時点ではFE/REの`fit()`本体が未実装のため、この関数はまだどこからも呼び出され
/// ない（実接続は後続issue）。`pub(crate)`関数は呼び出し元が無いと`dead_code`警告が出るため、
/// `#[expect]`（`#[allow]`と異なり、指定したlintが実際には発火しなくなった時点で
/// `unfulfilled_lint_expectations`として`-D warnings`下で逆に検知される）で回避する。接続
/// issueで実際に呼び出されるようになったら、この属性ごと削除すること（削除し忘れても
/// コンパイラが警告してくれる）。
#[expect(dead_code, reason = "接続issue（FE/REのfit()実装）まで未使用")]
pub(crate) fn panel_error_to_pyerr(err: PanelError) -> PyErr {
    let message = err.to_string();
    match err {
        PanelError::Common(common) => common_error_to_pyerr(common),
        PanelError::IdentifierDimensionMismatch { .. }
        | PanelError::InsufficientDegreesOfFreedom { .. }
        | PanelError::SingletonGroup { .. }
        | PanelError::UnbalancedPanelForTwoWay { .. }
        | PanelError::ZeroVarianceAfterDemeaning { .. }
        | PanelError::TwoWayRequiresTime => ValidationError::new_err(message),
        PanelError::WithinRegressionFailed { source } => {
            if least_squares_error_is_computation_error(&source) {
                ComputationError::new_err(message)
            } else {
                ValidationError::new_err(message)
            }
        }
    }
}
