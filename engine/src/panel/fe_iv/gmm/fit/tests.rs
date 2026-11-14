//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `basic.rs`: 基本挙動（getter・推定回数と収束・係数名・自由度）
//! - `cov_type.rs`: `cov_type`別の標準誤差（FE-2SLSへの還元、ダミー変数を明示したGMM、手計算との照合）・推論・Wald・R²・検証
//! - `hansen_j.rs`: Hansen J過剰識別検定（FE-2SLSのSarganへの還元、ダミー変数を明示したGMM、手計算、`None`の条件）
//! - `reference.rs`: FE-2SLSへの還元、ダミー変数を明示したGMM（FWL）、クラスター重みの手計算との照合
//! - `validation.rs`: 入力検証・エラー返却・検証順序・クラスター数の境界

use crate::panel::common::PanelError;
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::{FeGmmEstimator, FeGmmType, FeIvInput, FeWeightType};
use crate::shared::group_codes::GroupCodes;

mod basic;
mod cov_type;
mod hansen_j;
mod reference;
mod validation;

pub(super) use crate::panel::fe_iv::test_support::{
    N_ENTITIES, N_PERIODS, all_rows, dummies, entity_labels, input_spec, input_with, one_way_input,
    third_labels, time_labels, two_way_input,
};

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

pub(super) fn fit(
    input: FeIvInput,
    effects: FeEffects,
    gmm_type: FeGmmType,
) -> Result<FeGmmEstimator, PanelError> {
    FeGmmEstimator::fit(input, effects, gmm_type, true, FeCovType::Classical, 0.95)
}

/// `cov_type`を指定して推定する（信頼水準95%）。
pub(super) fn fit_with(
    input: FeIvInput,
    effects: FeEffects,
    gmm_type: FeGmmType,
    cov_type: FeCovType,
) -> Result<FeGmmEstimator, PanelError> {
    FeGmmEstimator::fit(input, effects, gmm_type, true, cov_type, 0.95)
}

pub(super) fn two_step(weight: FeWeightType) -> FeGmmType {
    FeGmmType::TwoStep { weight }
}

/// 個体（`entity_labels`）でクラスタリングする重み。
pub(super) fn entity_cluster_weight() -> FeWeightType {
    FeWeightType::Cluster {
        groups: GroupCodes::from_labels_without_keys(&entity_labels()),
    }
}

/// 時点（`time_labels`）でクラスタリングする重み。
pub(super) fn time_cluster_weight() -> FeWeightType {
    FeWeightType::Cluster {
        groups: GroupCodes::from_labels_without_keys(&time_labels()),
    }
}
