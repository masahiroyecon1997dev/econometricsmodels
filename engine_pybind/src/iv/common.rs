//! `iv`系統（2SLS/GMM）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//!
//! `IvError`の`Common`バリアント（`engine::error::CommonError`）は`crate::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::iv::common::IvError;
use pyo3::PyErr;

use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};
use crate::linear::common::least_squares_error_is_computation_error;

/// `engine::iv::common::IvError`をPython例外に変換する。
///
/// `IvError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの外で
/// 定義された型のため、orphan rule（`impl`の対象は自クレート内で定義されたトレイトか型の
/// どちらかを含む必要がある）により`impl From<IvError> for PyErr`は書けない。関数として
/// 実装し、呼び出し側で`.map_err(iv_error_to_pyerr)?`する（`least_squares_error_to_pyerr`と
/// 同じ理由、`engine_pybind/src/linear/common.rs`参照）。
///
/// `FirstStageFailed`/`SecondStageFailed`は2SLS（`engine::iv::two_sls`）が内部で委譲する
/// `OlsEstimator::fit`の失敗を包んだもの（Issue #157）。`ValidationError`/`ComputationError`の
/// 判定は`least_squares_error_is_computation_error`（`engine_pybind/src/linear/common.rs`）に
/// 委譲し、`least_squares_error_to_pyerr`と同じ基準を保つ（分類ロジックを重複させない）。
/// Pythonに渡すメッセージは`source.to_string()`ではなく`IvError`自身の`to_string()`
/// （「第一段階/第二段階のどの内生変数で失敗したか」という文脈を含む）を使うため、
/// `least_squares_error_to_pyerr`自体はそのまま呼ばない。
///
/// この関数は`engine_pybind`側の2SLS/GMM `fit()`接続（後続issue）が実装されるまで
/// どこからも呼び出されない。`pub(crate)`関数は呼び出し元が無いと`dead_code`警告が出るため、
/// `#[expect]`（`#[allow]`と異なり、指定したlintが実際には発火しなくなった時点で
/// `unfulfilled_lint_expectations`として`-D warnings`下で逆に検知される）で回避する。
/// 接続issueで実際に呼び出されるようになったら、この属性ごと削除すること
/// （削除し忘れてもコンパイラが警告してくれる）。
#[expect(dead_code, reason = "接続issue（2SLS/GMMのfit()実装）まで未使用")]
pub(crate) fn iv_error_to_pyerr(err: IvError) -> PyErr {
    let message = err.to_string();
    match err {
        IvError::Common(common) => common_error_to_pyerr(common),
        IvError::InsufficientInstruments { .. } => ValidationError::new_err(message),
        IvError::FirstStageFailed { source, .. } | IvError::SecondStageFailed { source } => {
            if least_squares_error_is_computation_error(&source) {
                ComputationError::new_err(message)
            } else {
                ValidationError::new_err(message)
            }
        }
    }
}
