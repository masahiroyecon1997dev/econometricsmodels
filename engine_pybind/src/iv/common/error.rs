//! `IvError`から`PyErr`への変換。

use crate::shared::errors::to_pyerr;
use engine::iv::common::IvError;
use pyo3::prelude::*;

/// `engine::iv::common::IvError`をPython例外に変換する。
///
/// `IvError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの外で
/// 定義された型のため、orphan rule（`impl`の対象は自クレート内で定義されたトレイトか型の
/// どちらかを含む必要がある）により`impl From<IvError> for PyErr`は書けない。関数として
/// 実装し、呼び出し側で`.map_err(iv_error_to_pyerr)?`する（`least_squares_error_to_pyerr`と
/// 同じ理由、`engine_pybind/src/linear/common.rs`参照）。
///
/// `FirstStageFailed`/`SecondStageFailed`/`HausmanRegressionFailed`は、内部で委譲する
/// `OlsEstimator::fit`の失敗を包んだもの。`report()`が原因の`code`に`stage`を付けて返すので、
/// ここでは分岐しない（`ValidationError`/`ComputationError`もカタログの`code`で決まる）。
pub(crate) fn iv_error_to_pyerr(err: IvError) -> PyErr {
    to_pyerr(&err)
}
