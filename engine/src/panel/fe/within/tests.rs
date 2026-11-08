//! `within.rs`（within変換と事前検証）のテスト。対象の関数別に子モジュールへ分ける。
//!
//! - `within_transform.rs`: within変換
//! - `singleton.rs`: シングルトン群の検出
//! - `zero_variance.rs`: 変動のない説明変数の検出
//! - `balanced_panel.rs`: バランスパネルの検証
//! - `combinations.rs`: 全ての(entity, time)の組合せがそろうかの判定
//! - `proptests.rs`: property-basedテスト

mod balanced_panel;
mod combinations;
mod proptests;
mod singleton;
mod within_transform;
mod zero_variance;
