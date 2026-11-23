//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `reference.rs`: 参照実装・閉形式・独立再計算との照合
//! - `cov_type.rs`: `cov_type`別の共分散・検定統計量
//! - `validation.rs`: 入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序
//! - `convergence.rs`: ソルバー・収束・非収束・分離・スケール
//! - `proptests.rs`: property-basedテスト

mod convergence;
mod cov_type;
mod proptests;
mod reference;
mod validation;
