//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `basic.rs`: 基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）
//! - `cov_type.rs`: `cov_type`別の共分散・検定統計量
//! - `validation.rs`: 入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序
//! - `convergence.rs`: ソルバー・収束・非収束・分離・スケール
//! - `proptests.rs`: property-basedテスト

mod basic;
mod convergence;
mod cov_type;
mod proptests;
mod validation;
