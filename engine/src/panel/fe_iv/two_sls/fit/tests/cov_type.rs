//! Driscoll-Kraay（DK）。

use super::{entity_labels, fit, input_spec, one_way_input, time_labels};
use crate::panel::common::TimeKeys;
use crate::panel::fe::{FeCovType, FeEffects};
use crate::shared::error::ErrorInfo;
use crate::shared::error_catalog::ErrorCode;
use crate::shared::group_codes::GroupCodes;

fn dk(bandwidth: Option<i64>) -> FeCovType {
    FeCovType::Dk {
        bandwidth,
        time: TimeKeys::lexicographic(time_labels()),
    }
}

/// ラグ項の無いDKは、時点でクラスタリングした共分散と同じ（1-wayのFEでは、時点は個体のFEに入れ子に
/// ならず、どちらも`K = df_model`、スケール`T/(T − 1)`、自由度`T − 1`）。
#[test]
fn dk_without_lags_equals_clustering_by_time() {
    let by_time = FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&time_labels())),
    };
    let reference = fit(one_way_input(), FeEffects::OneWay, by_time).unwrap();
    let estimator = fit(one_way_input(), FeEffects::OneWay, dk(Some(0))).unwrap();
    for r in 0..2 {
        for c in 0..2 {
            let (a, b) = (
                *estimator.cov_params().get(r, c),
                *reference.cov_params().get(r, c),
            );
            assert!(
                (a - b).abs() <= 1e-10 * b.abs().max(1e-12),
                "{r}{c}: {a} vs {b}"
            );
        }
    }
    assert_eq!(estimator.df_inference(), 3);
    assert_eq!(estimator.dk_bandwidth_used(), Some(0));
}

#[test]
fn dk_uses_the_automatic_bandwidth_when_none_is_given() {
    let estimator = fit(one_way_input(), FeEffects::OneWay, dk(None)).unwrap();
    assert!(estimator.dk_bandwidth_used().is_some());
}

/// 構造式の傾き`k`が時点数以下なら、DKは検定の共分散が特異になるので拒否する。
#[test]
fn dk_with_as_many_slopes_as_periods_is_rejected() {
    // 外生3個 + 内生1個 = 傾き4個、時点4つ。
    let rows: Vec<usize> = (0..40).collect();
    let err = fit(
        input_spec(&rows, &[entity_labels()], 2, 2, 3),
        FeEffects::OneWay,
        dk(Some(0)),
    )
    .map(|_| ())
    .unwrap_err();
    assert_eq!(
        err.report().code(),
        ErrorCode::DkPeriodsNotAboveCoefficients
    );
}

/// 第一段階の傾き`l`が時点数以上でも、構造式の`k`が時点数より小さければ、標準誤差は計算し第一段階のFは`None`。
#[test]
fn dk_with_as_many_first_stage_slopes_as_periods_still_gives_the_standard_errors() {
    // 外生1個 + 操作変数3個で`l = 4 = T`、構造式は`k = 2`。
    let rows: Vec<usize> = (0..40).collect();
    let estimator = fit(
        input_spec(&rows, &[entity_labels()], 1, 1, 3),
        FeEffects::OneWay,
        dk(Some(0)),
    )
    .unwrap();
    assert!(estimator.f_statistic().is_some());
    assert_eq!(estimator.first_stages()[0].f_statistic(), None);
    assert!(estimator.first_stages()[0].std_errors().nrows() == 4);
}
