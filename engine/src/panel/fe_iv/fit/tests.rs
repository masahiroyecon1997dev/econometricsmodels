//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `reference.rs`: fixestの`feols(y ~ x1 | 固定効果 | x2 ~ z1 + z2)`との照合（`data.rs`のデータ）
//! - `cov_type.rs`: Driscoll-Kraay（時点でクラスタリングした共分散との恒等式、自由度の境界）
//! - `lsdv.rs`: ダミー変数を明示した2SLS（LSDV）との照合（FWLにより、within変換した2SLSと係数・
//!   共分散が一致する）。不均衡な2-wayと3次元もここで確かめる
//! - `sargan.rs`: Sargan過剰識別検定（ダミー変数を明示した2SLSでの再計算、`σ̂²`の分母、帰無仮説下のサイズ）
//! - `structure.rs`: 構造残差・R²・固定効果・第一段階・不変性
//! - `weak_instruments.rs`: 弱操作変数診断（ダミーを回帰子に入れたIV本体の診断との照合、`cov_type`別の吸収したパラメータ数、DK）
//! - `wu_hausman.rs`: Wu-Hausman（fixestの補助回帰との照合、ダミー込みの補助回帰での再計算、`None`になる条件）
//! - `validation.rs`: 入力検証・エラー返却・検証順序

use crate::panel::common::PanelError;
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::{FeIvEstimator, FeIvInput};

mod cov_type;
mod lsdv;
mod reference;
mod sargan;
mod structure;
mod validation;
mod weak_instruments;
mod wu_hausman;

pub(super) use crate::panel::fe_iv::test_support::{
    N_ENTITIES, all_rows, entity_labels, input_spec, input_with, one_way_input, third_labels,
    time_labels, two_way_input,
};

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

pub(super) fn fit(
    input: FeIvInput,
    effects: FeEffects,
    cov_type: FeCovType,
) -> Result<FeIvEstimator, PanelError> {
    FeIvEstimator::fit(input, effects, cov_type, 0.95)
}

pub(super) fn entity_cluster() -> FeCovType {
    FeCovType::Cluster { groups: None }
}
