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
///
/// `#[allow(dead_code)]`について: Issue #65時点では呼び出し元が`build_logit_input`
/// （同じく本Issueで新設、テストからのみ呼ばれる）のみで、`LogitEstimator::fit`を
/// 実際に呼ぶ`fit_logit`（Issue #66で実装予定）がまだ無いため、`cargo build`
/// （`#[cfg(test)]`を含まないlibターゲットのビルド）からは到達不能に見え`dead_code`警告に
/// なる（`engine_pybind`は`engine`と異なりPython拡張モジュール専用の薄いバインディング層
/// であり、クレート外に`pub`なRust APIを公開する設計ではないため、`pub`化での回避は
/// 見送った。rust-reviewerの指摘）。Issue #66で`fit_logit`がこの関数を実際に呼ぶように
/// なった時点でこの属性は不要になる（削除すること）。
#[allow(dead_code)]
pub(crate) fn mle_error_to_pyerr(err: MleError) -> PyErr {
    match err {
        MleError::Common(common) => common_error_to_pyerr(common),
        MleError::InvalidMaxIter { .. } | MleError::InvalidCensoringBounds { .. } => {
            ValidationError::new_err(err.to_string())
        }
        MleError::NonConvergence { .. }
        | MleError::SingularHessian
        | MleError::SingularOpgMatrix => ComputationError::new_err(err.to_string()),
    }
}
