//! `iv`系統（2SLS/GMM）で共有するエラー型・入力データ構造・第一段階回帰・入力の検証。
//!
//! `nonlinear/common.rs`（`MleError`に加え`CovType`/`run_solver`等の共有ロジックも
//! 保持する）と同じ位置づけで、`iv`系統でも「エラー型のみのファイル」にせず、
//! 2SLS/GMMの両方が使う共有ロジックをここに集約する。
//!
//! このファイルは公開パス（`crate::iv::common::*`）を保つ再エクスポートだけを持ち、中身は
//! `common/`の子モジュールに役割別に置く: `error.rs`（`IvError`）・`input.rs`（`IvInput`と
//! 列変換ヘルパー）・`validation.rs`（`fit()`冒頭の入力検証）・`first_stage.rs`（第一段階回帰と
//! 弱操作変数診断）・`estimate.rs`（`IvEstimate`）・`proptest_support.rs`（`#[cfg(test)]`の
//! property-basedテスト用ケース生成器）。

mod error;
mod estimate;
mod first_stage;
mod input;
#[cfg(test)]
pub(crate) mod proptest_support;
mod validation;

pub use error::IvError;
pub use estimate::IvEstimate;
pub use first_stage::{FirstStageResult, compute_first_stage};
pub use input::IvInput;
pub(crate) use input::{hcat, mat_column_to_vec, mat_to_columns, without_baked_in_intercept};
pub use validation::validate_iv_fit_inputs;
