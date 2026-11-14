//! 入力検証・エラー返却・検証順序・クラスター数の境界。

use super::{
    all_rows, entity_cluster_weight, entity_labels, fit, input_spec, input_with, one_way_input,
    time_labels, two_step,
};
use crate::panel::common::{ColumnRole, PanelError};
use crate::panel::fe::{FeAbsorption, FeEffects};
use crate::panel::fe_iv::{FeGmmEstimator, FeIvInput, RoleColumns};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::ErrorCode;
use crate::shared::gmm_iteration::GmmType;
use crate::shared::gmm_weight::WeightType;
use crate::shared::group_codes::GroupCodes;

fn error_of(result: Result<FeGmmEstimator, PanelError>) -> PanelError {
    result.map(|_| ()).unwrap_err()
}

fn groups_of(count: usize) -> WeightType {
    let labels: Vec<String> = (0..40).map(|r| format!("g{}", r % count)).collect();
    WeightType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&labels)),
    }
}

// ── `gmm_type`の範囲 ───────────────────────────────────────────────

#[test]
fn invalid_max_iter_is_rejected_before_anything_else() {
    for invalid in [0_usize, 1, 2, 10_001] {
        let err = error_of(fit(
            one_way_input(),
            FeEffects::OneWay,
            GmmType::Iterated {
                weight: WeightType::Classical,
                max_iter: invalid,
                tol: 1e-6,
            },
        ));
        assert_eq!(
            err,
            PanelError::Common(CommonError::InvalidGmmMaxIter {
                max_iter: invalid as i64
            })
        );
    }
}

#[test]
fn invalid_tol_is_rejected() {
    for invalid in [0.0, -1e-6, f64::NAN, f64::INFINITY] {
        let err = error_of(fit(
            one_way_input(),
            FeEffects::OneWay,
            GmmType::Iterated {
                weight: WeightType::Classical,
                max_iter: 10,
                tol: invalid,
            },
        ));
        assert!(
            matches!(err, PanelError::Common(CommonError::InvalidGmmTol { .. })),
            "{err:?}"
        );
    }
}

/// `max_iter`の範囲の端（3と10,000）は受理する。
#[test]
fn max_iter_boundaries_are_accepted() {
    for max_iter in [3_usize, 10_000] {
        let result = fit(
            one_way_input(),
            FeEffects::OneWay,
            GmmType::Iterated {
                weight: WeightType::Classical,
                max_iter,
                tol: 1e-8,
            },
        );
        assert!(result.is_ok(), "max_iter={max_iter}");
    }
}

#[test]
fn non_convergence_raises_by_default() {
    let err = error_of(fit(
        input_spec(&all_rows(), &[entity_labels()], 1, 1, 2),
        FeEffects::OneWay,
        GmmType::Iterated {
            weight: WeightType::Robust,
            max_iter: 3,
            tol: 1e-300,
        },
    ));
    assert_eq!(
        err,
        PanelError::Common(CommonError::GmmNonConvergence { n_iter: 3 })
    );
    assert_eq!(err.report().code(), ErrorCode::NotConverged);
}

// ── 重み ──────────────────────────────────────────────────────────

/// 固定効果つきGMMは`Hac`の重みを持たない（バインディングは作らない）。到達不能な状態として内部エラー。
#[test]
fn hac_weight_is_an_internal_error() {
    let err = error_of(fit(
        one_way_input(),
        FeEffects::OneWay,
        two_step(WeightType::Hac {
            lags: Some(1),
            time_order: (0..40).map(f64::from).collect(),
        }),
    ));
    assert_eq!(err.report().code(), ErrorCode::InternalError);
}

#[test]
fn cluster_weight_without_groups_is_rejected() {
    let err = error_of(fit(
        one_way_input(),
        FeEffects::OneWay,
        two_step(WeightType::Cluster { groups: None }),
    ));
    assert!(
        matches!(
            err,
            PanelError::Common(CommonError::MissingClusterColumn { .. })
        ),
        "{err:?}"
    );
}

/// `OneStep`は重みを持たないので、クラスター数の検証も受けない。
#[test]
fn one_step_does_not_validate_a_weight() {
    assert!(fit(one_way_input(), FeEffects::OneWay, GmmType::OneStep).is_ok());
}

/// 過剰識別（`l = 3 > k = 2`）は`G < l`で拒否、`G == l`は受理する。
#[test]
fn over_identified_cluster_count_boundary() {
    let make = || input_spec(&all_rows(), &[entity_labels()], 1, 1, 2);
    let err = error_of(fit(make(), FeEffects::OneWay, two_step(groups_of(2))));
    assert_eq!(
        err,
        PanelError::Common(CommonError::InsufficientClustersForWeightMatrix {
            g: 2,
            l: 3,
            min_clusters: 3
        })
    );
    assert_eq!(
        err.report().code(),
        ErrorCode::GmmWeightClusterCountTooSmall
    );
    assert!(fit(make(), FeEffects::OneWay, two_step(groups_of(3))).is_ok());
}

/// 丁度識別（`l = k = 2`）は`G <= l`で拒否（`Z'ê = 0`で階数が`G − 1`以下）。`G = l + 1`は受理する。
#[test]
fn just_identified_cluster_count_boundary() {
    let make = || input_spec(&all_rows(), &[entity_labels()], 1, 1, 1);
    let err = error_of(fit(make(), FeEffects::OneWay, two_step(groups_of(2))));
    assert_eq!(
        err,
        PanelError::Common(CommonError::InsufficientClustersForWeightMatrix {
            g: 2,
            l: 2,
            min_clusters: 3
        })
    );
    assert!(fit(make(), FeEffects::OneWay, two_step(groups_of(3))).is_ok());
}

/// クラスターの列の長さが観測数と違う。
#[test]
fn cluster_weight_with_wrong_length_is_rejected() {
    let labels: Vec<String> = (0..39).map(|r| format!("g{}", r % 5)).collect();
    let err = error_of(fit(
        one_way_input(),
        FeEffects::OneWay,
        two_step(WeightType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labels)),
        }),
    ));
    assert!(matches!(err, PanelError::Common(_)), "{err:?}");
}

// ── 識別・索引・変換（FE-2SLSと同じ前段） ─────────────────────────────────

#[test]
fn fewer_instruments_than_endogenous_variables_is_rejected() {
    let err = error_of(fit(
        input_spec(&all_rows(), &[entity_labels()], 1, 2, 1),
        FeEffects::OneWay,
        two_step(WeightType::Robust),
    ));
    assert_eq!(
        err,
        PanelError::InsufficientInstruments {
            n_instruments: 1,
            n_endog: 2
        }
    );
}

#[test]
fn two_way_without_a_time_dimension_is_rejected() {
    let err = error_of(fit(
        one_way_input(),
        FeEffects::TwoWay,
        two_step(WeightType::Robust),
    ));
    assert_eq!(err, PanelError::TwoWayRequiresTime);
}

#[test]
fn singleton_groups_are_rejected() {
    // 個体0の行を1つだけ残す。
    let rows: Vec<usize> = std::iter::once(0).chain(4..40).collect();
    let err = error_of(fit(
        input_with(&rows, &[entity_labels()]),
        FeEffects::OneWay,
        two_step(WeightType::Robust),
    ));
    assert!(matches!(err, PanelError::SingletonGroup { .. }), "{err:?}");
}

#[test]
fn time_invariant_regressor_is_reported_with_its_role() {
    let base = one_way_input();
    // 個体ごとに一定の外生変数（固定効果を除くと分散が残らない）。
    let constant_within: Vec<f64> = (0..40).map(|r| (r / 4) as f64 + 0.5).collect();
    let input = FeIvInput::from_group_codes(
        base.y(),
        "y".to_string(),
        RoleColumns {
            names: vec!["w".to_string()],
            columns: &[constant_within],
        },
        RoleColumns {
            names: vec!["x2".to_string()],
            columns: base.x_endog(),
        },
        RoleColumns {
            names: vec!["z1".to_string(), "z2".to_string()],
            columns: base.instruments(),
        },
        vec![GroupCodes::from_labels(&entity_labels())],
    )
    .unwrap();
    let err = error_of(fit(input, FeEffects::OneWay, two_step(WeightType::Robust)));
    assert!(
        matches!(
            err,
            PanelError::ZeroVarianceAfterDemeaning {
                role: ColumnRole::XExog,
                ..
            }
        ),
        "{err:?}"
    );
}

/// 自由度の不足は段階のないエラー（FE-2SLSと違い第一段階の`stage`を付けない）。
#[test]
fn insufficient_degrees_of_freedom_has_no_stage() {
    // 4個体×2時点、2-wayで`neffects = 4 + 2 − 1 = 5`、`l = 1 + 2 = 3`、`n = 8 <= l + neffects`。
    let rows: Vec<usize> = (0..4).flat_map(|e| [e * 4, e * 4 + 1]).collect();
    let err = error_of(fit(
        input_with(&rows, &[entity_labels(), time_labels()]),
        FeEffects::TwoWay,
        two_step(WeightType::Classical),
    ));
    assert!(
        matches!(err, PanelError::InsufficientDegreesOfFreedom { .. }),
        "{err:?}"
    );
    assert_eq!(err.report().stage(), None);
}

#[test]
fn absorption_options_are_validated() {
    let result = FeGmmEstimator::fit_with_absorption(
        one_way_input(),
        FeEffects::OneWay,
        two_step(WeightType::Robust),
        true,
        &FeAbsorption {
            tol: -1.0,
            ..FeAbsorption::default()
        },
    );
    assert!(
        matches!(result, Err(PanelError::InvalidTol { .. })),
        "{result:?}"
    );
}

/// 検証の順序: `gmm_type`の範囲は、識別の検査より先。
#[test]
fn gmm_type_is_validated_before_the_identification_check() {
    let err = error_of(fit(
        input_spec(&all_rows(), &[entity_labels()], 1, 2, 1),
        FeEffects::OneWay,
        GmmType::Iterated {
            weight: WeightType::Classical,
            max_iter: 1,
            tol: 1e-6,
        },
    ));
    assert!(matches!(
        err,
        PanelError::Common(CommonError::InvalidGmmMaxIter { .. })
    ));
}

/// `entity_cluster_weight`は10個体・過剰識別`l = 3`で受理される（`G = 10 >= 3`）。
#[test]
fn entity_cluster_weight_is_accepted() {
    assert!(
        fit(
            input_spec(&all_rows(), &[entity_labels()], 1, 1, 2),
            FeEffects::OneWay,
            two_step(entity_cluster_weight()),
        )
        .is_ok()
    );
}
