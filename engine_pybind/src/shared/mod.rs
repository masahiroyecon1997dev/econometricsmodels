//! 複数の系統（`linear`/`nonlinear`/`iv`/`panel`）が共有するコード。
//!
//! 「何をするコードか」（DataFrameからの列抽出・入力検証・オプション値の取り出し・例外クラス）で
//! 分け、どの系統が使うかでは分けない（`engine/src/shared/`と同じ方針）。

pub(crate) mod column_extraction;
pub(crate) mod errors;
pub(crate) mod option_values;
pub(crate) mod validation;
