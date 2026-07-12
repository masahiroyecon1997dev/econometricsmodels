//! `engine_pybind`: `engine` に対するPyO3の薄いバインディング層。
//!
//! `#[pymodule]` の定義のみを行い、計算ロジックは持たない（`engine` を呼び出すだけに留める）。

use pyo3::prelude::*;

/// Python 側からインポートされるネイティブ拡張モジュール（`econometricsmodels._lib`）。
///
/// 各推定手法の関数・クラスは実装が進み次第、ここに `add_function` / `add_class` で登録する。
#[pymodule]
fn _lib(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}