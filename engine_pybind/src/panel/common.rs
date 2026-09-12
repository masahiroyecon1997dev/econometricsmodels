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
///   `ZeroVarianceAfterDemeaning`・`TwoWayRequiresTime`・`HacRequiresTime`・
///   `InvalidHacBandwidth`）はいずれも入力・オプションの不正なので`ValidationError`。
/// - `WithinRegressionFailed`: 委譲先の`LeastSquaresError`の分類基準
///   （`least_squares_error_is_computation_error`）にそのまま従う。`IvError::
///   SecondStageFailed`と同じ扱い。Pythonに渡すメッセージは`source.to_string()`ではなく
///   `PanelError`自身の`to_string()`（「within変換後の推定で失敗した」という文脈を含む）を
///   使うため、`least_squares_error_to_pyerr`自体は呼ばない。
///
/// Issue #172時点ではFE/REの`fit()`本体が未実装で、この関数は`#[cfg(test)] mod tests`
/// からしか呼び出されなかった。当初は`#[expect(dead_code, ...)]`（`cargo build`では未到達で
/// 発火するが`cargo test`/`clippy --all-targets`ではテストから到達可能になり発火しない、
/// という非対称性を前提にした属性）を使っていたが、Issue #186で`panel::fe::build_fe_input`
/// （同じく`#[cfg(test)] mod tests`からのみ呼ばれる）がこの関数を呼ぶようになったことで
/// `unfulfilled_lint_expectations`（`-D warnings`下でエラー）が発火し、`#[allow(dead_code)]`
/// （無条件抑制）に変更した経緯がある（`engine_pybind/src/iv/CLAUDE.md`「踏んだ罠」に
/// 記録済みの罠そのもの）。Issue #187で`panel::fe::fit`（`fit_fe`が`#[pymodule]`経由で
/// 呼ぶ本番経路）がこの関数を呼ぶようになったため、`#[allow(dead_code)]`は不要になった。
pub(crate) fn panel_error_to_pyerr(err: PanelError) -> PyErr {
    let message = err.to_string();
    match err {
        PanelError::Common(common) => common_error_to_pyerr(common),
        PanelError::IdentifierDimensionMismatch { .. }
        | PanelError::InsufficientDegreesOfFreedom { .. }
        | PanelError::SingletonGroup { .. }
        | PanelError::UnbalancedPanelForTwoWay { .. }
        | PanelError::ZeroVarianceAfterDemeaning { .. }
        | PanelError::TwoWayRequiresTime
        | PanelError::HacRequiresTime
        | PanelError::InvalidHacBandwidth { .. } => ValidationError::new_err(message),
        PanelError::WithinRegressionFailed { source } | PanelError::FTestFailed { source } => {
            if least_squares_error_is_computation_error(&source) {
                ComputationError::new_err(message)
            } else {
                ValidationError::new_err(message)
            }
        }
    }
}
