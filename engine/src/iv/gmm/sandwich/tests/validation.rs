//! 入力検証・エラー返却のテスト。

use crate::iv::common::IvError;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::WeightType;
use crate::iv::gmm::test_support::*;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

/// `cov_type=Cluster`で`groups`未指定なら`MissingClusterColumn`
/// （`weight_type`側の同種の検証、`fit_returns_missing_cluster_column_error_when_
/// cluster_weight_type_has_no_groups`と対になる、こちらは`cov_type`側）。
#[test]
fn fit_returns_missing_cluster_column_error_when_cov_type_cluster_has_no_groups() {
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
            weight: WeightType::Robust,
        },
        true,
        CovType::Cluster { groups: None },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::MissingClusterColumn {
            setting: ClusterSetting::CovType,
        })
    );
}

/// `cov_type=Cluster`でクラスター数が1（`G>=2`未満）なら`InsufficientClusters`
/// （`weight_type`側の同種の検証と対になる、こちらは`cov_type`側）。
#[test]
fn fit_returns_insufficient_clusters_error_when_cov_type_cluster_has_only_one_group() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let groups: Vec<String> = vec!["g0".to_string(); n];
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
            weight: WeightType::Robust,
        },
        true,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    );
    assert!(matches!(
        result.unwrap_err(),
        IvError::Common(CommonError::InsufficientClusters { .. })
    ));
}

/// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下だと、
/// クラスター寄与スコアの総和がゼロ（一次条件）で`rank(Ŝ) ≤ g - 1`のため、ロバスト
/// Wald（χ²）検定の`q×q`部分行列が構造的に特異になる。`fit()`冒頭
/// （反復推定・`validate_weight_type`より前）で構造方程式の`q`を使って
/// `CommonError::InsufficientClustersForInference`を返す（2SLSと同型。
/// `gmm_tol`非収束等が先に起きないよう最適化前に弾く）。`x_exog=[z1]`・
/// `x_endog=[endog1]`・切片ありで`q = k - k_constant = 3 - 1 = 2`、`g=2`（`g == q`）。
/// `weight_type=Cluster`の重み行列`S`（l×l）が`G<l`で特異になる別軸の問題とは区別する。
#[test]
fn fit_returns_validation_error_when_cov_type_cluster_count_at_most_slopes() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let groups: Vec<String> = (0..y.len())
        .map(|i| if i < 8 { "a" } else { "b" }.to_string())
        .collect();
    let input = IvInput::from_columns(
        &y,
        std::slice::from_ref(&z1),
        vec!["z1".to_string()],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        std::slice::from_ref(&z2),
        vec!["z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Robust,
        },
        true,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
    );
}

/// `cov_type=Hac`の`lags`が範囲外（負・n以上）なら`InvalidHacLags`
/// （`weight_type`側の同種の検証と対になる、こちらは`cov_type`側）。
#[test]
fn fit_returns_invalid_hac_lags_error_when_cov_type_hac_lags_out_of_range() {
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
            weight: WeightType::Robust,
        },
        true,
        CovType::Hac {
            lags: Some(-1),
            time_order: row_time_order(n),
        },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InvalidHacLags { hac_lags: -1, n })
    );
}
