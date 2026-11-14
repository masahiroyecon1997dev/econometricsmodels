//! `cov_params.rs`（OLSの`cov_type`別の共分散）のテスト。観点別に子モジュールへ分ける。
//!
//! - `validation.rs`: 入力検証・エラー返却
//! - `cluster.rs`: クラスター共分散・`fit_with_optional_f_test`（クラスター数が傾き以下のときの全体F検定）
//! - `hc_hac.rs`: HC・HAC共分散

mod cluster;
mod hc_hac;
mod validation;
