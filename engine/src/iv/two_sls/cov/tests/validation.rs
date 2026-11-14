//! 入力検証・エラー返却・クラスター数の境界のテスト。

use super::assert_slices_close;
use crate::iv::common::IvError;
use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::iv::two_sls::test_support::*;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

#[test]
fn fit_returns_invalid_hac_lags_error_when_out_of_range() {
    let result = TwoSlsEstimator::fit(
        x_endog_empty_input(),
        CovType::Hac {
            lags: Some(-1),
            time_order: row_time_order(5),
        },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InvalidHacLags { hac_lags: -1, n: 5 })
    );
}

#[test]
fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
    let result = TwoSlsEstimator::fit(
        x_endog_empty_input(),
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

#[test]
fn fit_returns_insufficient_clusters_error_when_only_one_group() {
    let groups = vec!["a".to_string(); 5];
    let result = TwoSlsEstimator::fit(
        x_endog_empty_input(),
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InsufficientClusters {
            setting: ClusterSetting::CovType,
            g: 1
        })
    );
}

/// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`以下だと、クラスター寄与
/// スコアの総和がゼロ（正規方程式`X'e = 0`）で`rank(Ŝ) ≤ g - 1`のため全体Wald/F
/// 検定の`q×q`部分行列が構造的に特異になる。`g`・`q`は入力だけから判定できるため
/// `fit()`冒頭のバリデーションが`CommonError::InsufficientClustersForInference`で
/// 弾く（OLSと同型）。`cluster_test_input_and_groups`の設計行列は
/// `[const, x1, endog1]`（`k=3`・`k_constant=1`・`q=2`）なので、`g=2`（`g == q`）と
/// 組み合わせる。
#[test]
fn fit_returns_validation_error_when_cluster_count_at_most_slopes() {
    let (input, _) = cluster_test_input_and_groups();
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
    ];
    let result = TwoSlsEstimator::fit(
        input,
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

/// 構造方程式は`G > q_struct`だが第一段階では`G == q_first`（`nontrivial_x_exog_input`は
/// `[const, x1, endog1]`で`q_struct=2`、第一段階は`x1`・`z1`・`z2`で`q_first=3`、`g=3`）の
/// とき、第一段階の全体F検定が定義できなくても`fit()`全体は成功する。第一段階の
/// `f_statistic()`は`None`、構造方程式の結果（Wald検定を含む）は計算され、弱操作変数診断は
/// `cov_type`に依存しないため古典的な`cov_type`と同じ値になる。
#[test]
fn fit_succeeds_when_cluster_count_is_between_structural_and_first_stage_slopes() {
    let n = nontrivial_x_exog_input().nobs();
    let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 3)).collect();
    let estimator = TwoSlsEstimator::fit(
        nontrivial_x_exog_input(),
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    )
    .unwrap();

    let (_, first_stage) = &estimator.first_stage_estimators()[0];
    assert_eq!(first_stage.f_statistic(), None);
    assert_eq!(first_stage.f_df(), None);
    assert!(estimator.wald_statistic().is_finite());

    let classical =
        TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
    assert_eq!(
        estimator.weak_instrument_f_statistics(),
        classical.weak_instrument_f_statistics()
    );
}

/// `G=2`クラスター・`x_exog=[]`・丁度識別（`instruments`1本）という、
/// ベンチマーク作成中に発見した`ComputationError`（`engine/src/iv/CLAUDE.md`
/// 「修正済み」参照）の再現条件そのもの。第一段階回帰の`OlsInput::
/// from_columns`に`has_intercept=false`を渡していたため、`IvInput::x_exog()`が
/// 焼き込み済みの定数項列まで「傾き係数」としてWald F検定の対象に含めてしまい
/// （真のq=1のところq=2）、`rank(Ŝ)≤G-1=1`の構造的特異性で必ず失敗していた
/// （`without_baked_in_intercept`の導入で修正、`common/input.rs`参照）。
#[test]
fn fit_succeeds_with_cluster_g2_boundary_when_x_exog_is_empty() {
    let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
    let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
    let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
    ];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        std::slice::from_ref(&z1),
        vec!["z1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let cov_type = CovType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&groups)),
    };
    let estimator = TwoSlsEstimator::fit(input, cov_type, 0.95).unwrap();

    // 第一段階（`endog1 ~ const + z1`）が、素の`OlsEstimator::fit`
    // （同じG=2クラスター、`has_intercept=true`）と一致することを確認する
    // （`compute_first_stage`の`has_intercept`修正が正しく効いていることの直接確認、
    // `test_matches_closed_form_ols_when_instrument_perfectly_predicts_endog`と
    // 同じ発想）。
    let (_, first_stage_estimator) = &estimator.first_stage_estimators()[0];
    let ols_input = OlsInput::from_columns(
        &x_endog,
        &[z1],
        vec!["z1".to_string()],
        true,
        "endog1".to_string(),
    )
    .unwrap();
    let expected = OlsEstimator::fit(
        ols_input,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    )
    .unwrap();

    assert_slices_close(
        &(0..2)
            .map(|j| *first_stage_estimator.params().get(j, 0))
            .collect::<Vec<_>>(),
        &(0..2)
            .map(|j| *expected.params().get(j, 0))
            .collect::<Vec<_>>(),
    );
    assert_slices_close(
        &(0..2)
            .map(|j| *first_stage_estimator.std_errors().get(j, 0))
            .collect::<Vec<_>>(),
        &(0..2)
            .map(|j| *expected.std_errors().get(j, 0))
            .collect::<Vec<_>>(),
    );

    // `has_intercept=false`を渡していた旧版では、`first_stage()`が返す`r_squared`/
    // `wald_statistic`も定数項を傾き係数扱いした非中心化TSS・過大なqで静かに間違って
    // いた（`engine/src/iv/CLAUDE.md`「修正済み」参照）。修正後は素の
    // `OlsEstimator::fit`と一致する。
    assert!((first_stage_estimator.r_squared() - expected.r_squared()).abs() < 1e-10);
    assert!(
        (first_stage_estimator.f_statistic().unwrap() - expected.f_statistic().unwrap()).abs()
            < 1e-8
    );
}
