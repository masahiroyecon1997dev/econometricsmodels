//! `fit_from_within`（変換済みの列から推定する後半）のテスト。

use super::{three_period_input, three_period_time};
use crate::panel::common::PanelError;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::input::{FeAbsorption, FeCovType, FeEffects, FeInput};
use crate::panel::fe::within::{WithinOutput, within_transform_columns};
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

/// `input`の`y`・`x`を、`levels`の次元で変換した結果。
fn transform(input: &FeInput, levels: &[&GroupCodes]) -> WithinOutput {
    let mut columns: Vec<&[f64]> = vec![input.y()];
    columns.extend(input.x().iter().map(Vec::as_slice));
    let out = within_transform_columns(levels, &columns, &FeAbsorption::default(), false).unwrap();
    let mut cols = out.columns.into_iter();
    WithinOutput {
        y: cols.next().unwrap(),
        x: cols.collect(),
        absorbed: None,
        noise: out.noise,
    }
}

#[test]
fn fitting_from_the_transformed_columns_equals_fit() {
    let time = three_period_time();
    for (effects, with_time) in [(FeEffects::OneWay, false), (FeEffects::TwoWay, true)] {
        let make = || three_period_input(2, with_time.then_some(time.as_slice()));
        let reference = FeEstimator::fit(make(), effects, FeCovType::Hc1, 0.95).unwrap();

        let input = make();
        let mut levels = vec![input.entity_codes()];
        levels.extend(input.time_codes());
        let within = transform(&input, &levels);
        let estimator =
            FeEstimator::fit_from_within(make(), effects, FeCovType::Hc1, 0.95, within, true)
                .unwrap();

        assert_eq!(estimator.params(), reference.params());
        assert_eq!(estimator.std_errors(), reference.std_errors());
        assert_eq!(estimator.df_resid(), reference.df_resid());
        assert_eq!(estimator.f_statistic(), reference.f_statistic());
        assert_eq!(estimator.r_squared_within(), reference.r_squared_within());
    }
}

/// 3時点（12行）を、クラスター変数にする`cov_type`。
fn time_clusters() -> FeCovType {
    FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&three_period_time())),
    }
}

#[test]
fn without_the_f_test_the_cluster_count_may_be_at_most_the_number_of_slopes() {
    // 3時点でクラスタリングし（G=3）、傾きは3個。F検定をするなら`G <= q`で拒否される。
    let input = three_period_input(3, None);
    let err = FeEstimator::fit(
        three_period_input(3, None),
        FeEffects::OneWay,
        time_clusters(),
        0.95,
    )
    .map(|_| ())
    .unwrap_err();
    assert!(matches!(
        err,
        PanelError::Common(CommonError::InsufficientClustersForInference { .. })
    ));

    let within = transform(&input, &[input.entity_codes()]);
    let estimator = FeEstimator::fit_from_within(
        three_period_input(3, None),
        FeEffects::OneWay,
        time_clusters(),
        0.95,
        within,
        false,
    )
    .unwrap();
    assert_eq!(estimator.f_statistic(), None);
    assert_eq!(estimator.df_inference(), 2);
    for i in 0..3 {
        let se = *estimator.std_errors().get(i, 0);
        assert!(se.is_finite() && se > 0.0, "se {se}");
    }
}
