//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `basic.rs`: 基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）
//! - `validation.rs`: 入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序
//! - `release_design.rs`: `fit_releasing_design`（設計行列を手放す経路）が`fit`と同じ結果・同じエラーであること
//! - `proptests.rs`: property-basedテスト

use crate::linear::ols::OlsInput;

mod basic;
mod proptests;
mod release_design;
mod validation;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

/// y = 1 + 2*x、ノイズなしの厳密解を持つデータ。
pub(super) fn exact_fit_input() -> OlsInput {
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let x_columns = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
    OlsInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}
