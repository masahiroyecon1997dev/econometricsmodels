//! `Fitted`（限界効果・予測・的中表・`from_parts`）のテスト。機能別に子モジュールへ分ける。
//!
//! - `marginal_effects.rs`: 限界効果（overall・at mean・at median）
//! - `predict.rs`: `predict`・`predict_new_data`・`from_parts`による復元
//! - `pred_table.rs`: `pred_table`（的中表）

mod marginal_effects;
mod pred_table;
mod predict;
