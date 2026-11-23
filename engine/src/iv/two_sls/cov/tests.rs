//! `cov.rs`（2SLSの`cov_type`別の共分散）のテスト。観点別に子モジュールへ分ける。
//!
//! - `validation.rs`: 入力検証・エラー返却・クラスター数の境界
//! - `hac.rs`: HAC共分散（ラグ・時間順序）
//! - `reference.rs`: 手計算の公式との照合・整合性

mod hac;
mod reference;
mod validation;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

pub(super) fn assert_slices_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-8, "index {i}: got {a}, expected {e}");
    }
}
