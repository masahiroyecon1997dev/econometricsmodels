//! 入力検証・エラー返却・クラスター数の境界（受理する側を含む）のテスト。

use super::fit_cluster_weighted;
use crate::iv::common::IvError;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::test_support::*;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::gmm_weight::WeightType;
use crate::shared::group_codes::GroupCodes;

/// 2段階GMMで`weight_type`が不正（`Cluster`で`groups`未指定）なら
/// `MissingClusterColumn`になる（1段階GMMは`weight_type`を持たないため検証もしない、
/// `gmm.rs`のモジュールdoc参照）。
#[test]
fn fit_returns_missing_cluster_column_error_when_two_step_gmm_has_invalid_cluster_weight_type() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &[z1, z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Cluster { groups: None },
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::MissingClusterColumn {
            setting: ClusterSetting::GmmWeightType,
        })
    );
}

/// 同様に2段階GMMで`weight_type=Hac`の`lags`が範囲外なら
/// `InvalidHacLags`になる。
#[test]
fn fit_returns_invalid_hac_lags_error_when_two_step_gmm_has_invalid_hac_weight_type() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &[z1, z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Hac {
                lags: Some(-1),
                time_order: row_time_order(n),
            },
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InvalidHacLags { hac_lags: -1, n })
    );
}

/// 過剰識別（l=3: const, z1, z2、k=2）では`S`は`rank(S) <= G`のため`G < l`で構造的に
/// 特異になり、`fit()`冒頭で`InsufficientClustersForWeightMatrix`
/// （`ValidationError`）として弾く（行列計算に入ると`NearSingularMatrix`の
/// `ComputationError`になる。
/// `fit_computes_cluster_weighted_estimate_matching_manual_formula`のdocコメント参照）。
#[test]
fn fit_returns_insufficient_clusters_for_weight_matrix_when_overidentified_and_g_less_than_l() {
    let result = fit_cluster_weighted(2, 2);
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InsufficientClustersForWeightMatrix {
            g: 2,
            l: 3,
            min_clusters: 3
        })
    );
}

/// 丁度識別（l=2: const, z1、k=2）では`Z'ê = 0`により`rank(S) <= G-1`のため、
/// `G == l`でも特異になり`InsufficientClustersForWeightMatrix`で弾く。
#[test]
fn fit_returns_insufficient_clusters_for_weight_matrix_when_just_identified_and_g_equals_l() {
    let result = fit_cluster_weighted(1, 2);
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InsufficientClustersForWeightMatrix {
            g: 2,
            l: 2,
            min_clusters: 3
        })
    );
}

/// クラスター頑健版でグループ列未指定なら`MissingClusterColumn`。
#[test]
fn fit_returns_missing_cluster_column_error_when_cluster_weight_type_has_no_groups() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &[z1, z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Cluster { groups: None },
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::MissingClusterColumn {
            setting: ClusterSetting::GmmWeightType,
        })
    );
}

/// クラスターが1種類しかない場合`InsufficientClusters`。
#[test]
fn fit_returns_insufficient_clusters_error_when_only_one_group() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &[z1, z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let groups = vec!["only_group".to_string(); n];
    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InsufficientClusters {
            setting: ClusterSetting::GmmWeightType,
            g: 1
        })
    );
}

/// `weight_type=Hac`の`lags`が範囲外（負・n以上）なら`InvalidHacLags`。
#[test]
fn fit_returns_invalid_hac_lags_error_when_out_of_range() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let build_input = || {
        IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };

    let result = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Hac {
                lags: Some(-1),
                time_order: row_time_order(n),
            },
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InvalidHacLags { hac_lags: -1, n })
    );

    let result = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Hac {
                lags: Some(n as i64),
                time_order: row_time_order(n),
            },
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InvalidHacLags {
            hac_lags: n as i64,
            n
        })
    );
}

/// 過剰識別では`G == l`はエラーにならない（`Z'ê = 0`が成り立たず`rank(S) <= G`が
/// 上限のため、`G == l`は非特異になりうる）。
#[test]
fn fit_succeeds_when_overidentified_and_g_equals_l() {
    assert!(fit_cluster_weighted(2, 3).is_ok());
}

/// 丁度識別でも`G > l`なら成功する（`G=l+1`が成功パスの境界）。
#[test]
fn fit_succeeds_when_just_identified_and_g_exceeds_l() {
    assert!(fit_cluster_weighted(1, 3).is_ok());
}
