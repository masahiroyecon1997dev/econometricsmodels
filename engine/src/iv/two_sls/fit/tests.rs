//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `basic.rs`: 基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）
//! - `reference.rs`: 参照実装・閉形式・独立再計算との照合
//! - `validation.rs`: 入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序
//! - `proptests.rs`: property-basedテスト

mod basic;
mod proptests;
mod reference;
mod validation;
