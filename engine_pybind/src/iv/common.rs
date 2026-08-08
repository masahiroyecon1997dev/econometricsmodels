//! `iv`系統（2SLS/GMM）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//!
//! `IvError`の`Common`バリアント（`engine::error::CommonError`）は`crate::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::iv::common::IvError;
use pyo3::PyErr;

use crate::errors::{ValidationError, common_error_to_pyerr};

/// `engine::iv::common::IvError`をPython例外に変換する。
///
/// `IvError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの外で
/// 定義された型のため、orphan rule（`impl`の対象は自クレート内で定義されたトレイトか型の
/// どちらかを含む必要がある）により`impl From<IvError> for PyErr`は書けない。関数として
/// 実装し、呼び出し側で`.map_err(iv_error_to_pyerr)?`する（`least_squares_error_to_pyerr`と
/// 同じ理由、`engine_pybind/src/linear/common.rs`参照）。
///
/// Issue #155時点では2SLS/GMMの`fit()`本体が未実装のため、この関数はまだどこからも
/// 呼び出されない（実接続は後続issue）。`pub(crate)`関数は呼び出し元が無いと`dead_code`
/// 警告が出るため、`#[expect]`（`#[allow]`と異なり、指定したlintが実際には発火しなく
/// なった時点で`unfulfilled_lint_expectations`として`-D warnings`下で逆に検知される）で
/// 回避する。接続issueで実際に呼び出されるようになったら、この属性ごと削除すること
/// （削除し忘れてもコンパイラが警告してくれる）。
#[expect(dead_code, reason = "接続issue（2SLS/GMMのfit()実装）まで未使用")]
pub(crate) fn iv_error_to_pyerr(err: IvError) -> PyErr {
    match err {
        IvError::Common(common) => common_error_to_pyerr(common),
        IvError::InsufficientInstruments { .. } => ValidationError::new_err(err.to_string()),
    }
}
