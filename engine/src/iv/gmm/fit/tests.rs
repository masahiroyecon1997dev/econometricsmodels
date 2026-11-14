//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `basic.rs`: 基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）
//! - `validation.rs`: 入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序
//! - `iteration.rs`: 反復GMM（`gmm_type`・`gmm_max_iter`・`gmm_tol`・収束・非収束）
//! - `weight_validation.rs`: 重み行列の入力検証・エラー返却・クラスター数の境界（受理する側を含む）
//! - `weight_robust_hac.rs`: ロバスト重み・HAC重み（ラグの解決・時間順序・手計算との照合）
//! - `weight_cluster.rs`: クラスター重み（クラスター数と操作変数の数・モーメント共分散）
//! - `proptests.rs`: property-basedテスト

use crate::iv::common::IvError;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::WeightType;
use crate::iv::gmm::test_support::*;
use crate::linear::ols::CovType;
use crate::shared::group_codes::GroupCodes;

mod basic;
mod iteration;
mod proptests;
mod validation;
mod weight_cluster;
mod weight_robust_hac;
mod weight_validation;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

/// `weight_type=Cluster`の重み行列`S`（l×l）のクラスター数`g`と全操作変数数`l`の
/// テスト用ヘルパー。`n_instruments`本の操作変数（`z1`のみ／`z1,z2`）で、
/// `g`個のクラスターに`i % g`で割り当てて`fit`し結果を返す。
pub(super) fn fit_cluster_weighted(
    n_instruments: usize,
    g: usize,
) -> Result<GmmEstimator, IvError> {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % g)).collect();
    let (instruments, names) = if n_instruments == 1 {
        (vec![z1], vec!["z1".to_string()])
    } else {
        (vec![z1, z2], vec!["z1".to_string(), "z2".to_string()])
    };
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &instruments,
        names,
        true,
        "y".to_string(),
    )
    .unwrap();
    GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        },
        true,
        CovType::Classical,
        0.95,
    )
}
