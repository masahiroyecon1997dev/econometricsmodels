//! `sandwich.rs`（GMMの係数共分散・検定統計量）のテスト。観点別に子モジュールへ分ける。
//!
//! - `validation.rs`: 入力検証・エラー返却
//! - `reference.rs`: 手計算の公式との照合
//! - `basic.rs`: getter・退化ケース・整合性

mod basic;
mod reference;
mod validation;
