//! `variance.rs`（REの分散成分とbetween回帰）のテスト。観点別に子モジュールへ分ける。
//!
//! - `validation.rs`: エラーの伝播・変換
//! - `clipping.rs`: 負の分散の0への切り詰め
//! - `variance_components.rs`: linearmodels・plmとの照合・between回帰の列の扱い
//! - `trace_form.rs`: 1-wayのトレース形（観測ごとの行列で書いた参照・バランスの閉形式との一致）

mod clipping;
mod trace_form;
mod validation;
mod variance_components;
