//! 入力検証・エラー返却・検証順序。

use super::{
    entity_cluster, entity_labels, fit, input_with, one_way_input, third_labels, time_labels,
    two_way_input,
};
use crate::panel::common::{ColumnRole, PanelError};
use crate::panel::fe::{FeAbsorption, FeCovType, FeEffects};
use crate::panel::fe_iv::{FeIvEstimator, FeIvInput, RoleColumns};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::{ErrorCode, Stage};
use crate::shared::group_codes::GroupCodes;

/// 10個体×4時点（行は個体ごと）の列から作る1-way入力。
fn custom(
    y: &[f64],
    x_exog: impl AsRef<[Vec<f64>]>,
    x_endog: impl AsRef<[Vec<f64>]>,
    instruments: impl AsRef<[Vec<f64>]>,
) -> FeIvInput {
    let (x_exog, x_endog, instruments) = (x_exog.as_ref(), x_endog.as_ref(), instruments.as_ref());
    let names = |prefix: &str, n: usize| (0..n).map(|j| format!("{prefix}{j}")).collect();
    FeIvInput::from_group_codes(
        y,
        "y".to_string(),
        RoleColumns {
            names: names("x", x_exog.len()),
            columns: x_exog,
        },
        RoleColumns {
            names: names("e", x_endog.len()),
            columns: x_endog,
        },
        RoleColumns {
            names: names("z", instruments.len()),
            columns: instruments,
        },
        vec![GroupCodes::from_labels(&entity_labels())],
    )
    .unwrap()
}

/// `y`・`x1`・`x2`・`z1`・`z2`。
type Columns = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn columns() -> Columns {
    let input = one_way_input();
    (
        input.y().to_vec(),
        input.x_exog()[0].clone(),
        input.x_endog()[0].clone(),
        input.instruments()[0].clone(),
        input.instruments()[1].clone(),
    )
}

fn entity_constant(per_entity: impl Fn(usize) -> f64) -> Vec<f64> {
    (0..40).map(|row| per_entity(row / 4)).collect()
}

fn fit_one_way(input: FeIvInput) -> Result<FeIvEstimator, PanelError> {
    fit(input, FeEffects::OneWay, FeCovType::Classical)
}

fn error_of(result: Result<FeIvEstimator, PanelError>) -> PanelError {
    result.map(|_| ()).unwrap_err()
}

// ── 識別・索引 ────────────────────────────────────────────────────

#[test]
fn fewer_instruments_than_endogenous_variables_is_rejected() {
    let (y, x1, x2, z1, _) = columns();
    let err = error_of(fit_one_way(custom(
        &y,
        vec![x1.clone()],
        vec![x2.clone()],
        &[],
    )));
    assert_eq!(
        err,
        PanelError::InsufficientInstruments {
            n_instruments: 0,
            n_endog: 1,
        }
    );
    let err = error_of(fit_one_way(custom(
        &y,
        &[x1],
        &[x2.clone(), z1.clone()],
        &[z1],
    )));
    assert_eq!(
        err,
        PanelError::InsufficientInstruments {
            n_instruments: 1,
            n_endog: 2,
        }
    );
}

#[test]
fn two_way_needs_a_second_dimension() {
    let err = error_of(fit(
        one_way_input(),
        FeEffects::TwoWay,
        FeCovType::Classical,
    ));
    assert_eq!(err, PanelError::TwoWayRequiresTime);
}

#[test]
fn an_invalid_absorption_setting_is_rejected() {
    let absorption = FeAbsorption {
        tol: 0.0,
        ..FeAbsorption::default()
    };
    let err = error_of(FeIvEstimator::fit_with_absorption(
        two_way_input(),
        FeEffects::TwoWay,
        FeCovType::Classical,
        0.95,
        &absorption,
    ));
    assert!(matches!(err, PanelError::InvalidTol { .. }));
}

// ── singleton・定数列（役割つき） ───────────────────────────────────

#[test]
fn a_singleton_group_is_rejected_with_the_effect_column() {
    // 個体0の行を1つだけ残す。
    let rows: Vec<usize> = std::iter::once(0).chain(4..40).collect();
    let err = error_of(fit_one_way(input_with(&rows, &[entity_labels()])));
    assert!(matches!(err, PanelError::SingletonGroup { .. }), "{err:?}");
}

#[test]
fn a_column_that_is_constant_in_the_raw_data_is_reported_with_its_role() {
    let (y, x1, x2, z1, z2) = columns();
    let constant = vec![2.5; 40];
    let cases = [
        (
            custom(
                &y,
                vec![constant.clone()],
                vec![x2.clone()],
                &[z1.clone(), z2.clone()],
            ),
            ColumnRole::XExog,
            "x0",
        ),
        (
            custom(
                &y,
                vec![x1.clone()],
                vec![constant.clone()],
                &[z1.clone(), z2.clone()],
            ),
            ColumnRole::XEndog,
            "e0",
        ),
        (
            custom(&y, &[x1], &[x2], &[constant, z2]),
            ColumnRole::Instruments,
            "z0",
        ),
    ];
    for (input, role, column) in cases {
        assert_eq!(
            error_of(fit_one_way(input)),
            PanelError::RegressorConstant {
                role,
                column: column.to_string(),
            }
        );
    }
}

#[test]
fn a_constant_outcome_is_reported_before_the_transformation() {
    let (_, x1, x2, z1, z2) = columns();
    let err = error_of(fit_one_way(custom(&vec![1.0; 40], &[x1], &[x2], &[z1, z2])));
    assert_eq!(err, PanelError::Common(CommonError::OutcomeConstant));
}

#[test]
fn a_column_without_within_variation_is_reported_with_its_role() {
    // 個体内で一定（時間不変）の列は、固定効果を除くと分散が残らない。
    let (y, x1, x2, z1, z2) = columns();
    let invariant = entity_constant(|i| i as f64);
    let cases = [
        (
            custom(
                &y,
                vec![invariant.clone()],
                vec![x2.clone()],
                &[z1.clone(), z2.clone()],
            ),
            ColumnRole::XExog,
            "x0",
        ),
        (
            custom(
                &y,
                vec![x1.clone()],
                vec![invariant.clone()],
                &[z1.clone(), z2.clone()],
            ),
            ColumnRole::XEndog,
            "e0",
        ),
        (
            custom(&y, &[x1], &[x2], &[z1, invariant]),
            ColumnRole::Instruments,
            "z1",
        ),
    ];
    for (input, role, column) in cases {
        assert_eq!(
            error_of(fit_one_way(input)),
            PanelError::ZeroVarianceAfterDemeaning {
                role,
                column: column.to_string(),
            }
        );
    }
}

#[test]
fn an_outcome_that_is_constant_within_entities_is_rejected() {
    let (_, x1, x2, z1, z2) = columns();
    let y = entity_constant(|i| i as f64 * 0.7);
    let err = error_of(fit_one_way(custom(&y, &[x1], &[x2], &[z1, z2])));
    assert_eq!(err, PanelError::OutcomeConstantWithinEntity);
}

// ── 自由度 ────────────────────────────────────────────────────────

#[test]
fn too_few_observations_for_the_structural_equation_are_reported_first() {
    // 2個体×2時点: 構造式は`k + neffects = 2 + 2 = 4 = n`。
    let rows = [0, 1, 4, 5];
    let err = error_of(fit_one_way(input_with(&rows, &[entity_labels()])));
    assert!(matches!(
        err,
        PanelError::InsufficientDegreesOfFreedom { n_obs: 4, k: 2, .. }
    ));
}

#[test]
fn too_few_observations_for_the_first_stage_carry_the_first_stage_stage() {
    // 2個体（2行・3行）: 構造式は`5 > 2 + 2`で足りるが、第一段階は`l + neffects = 3 + 2 = 5 = n`。
    let rows = [0, 1, 4, 5, 6];
    let err = error_of(fit_one_way(input_with(&rows, &[entity_labels()])));
    let report = err.report();
    assert_eq!(report.code(), ErrorCode::InsufficientDegreesOfFreedom);
    assert_eq!(
        report.stage(),
        Some(&Stage::FirstStage {
            endog_name: "x2".to_string()
        })
    );
}

// ── 共分散の検証 ──────────────────────────────────────────────────

#[test]
fn clusters_not_above_the_number_of_structural_slopes_are_rejected() {
    // 2クラスター（傾きは2個）。
    let groups: Vec<String> = (0..40).map(|row| (row % 2).to_string()).collect();
    let cov = FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&groups)),
    };
    let err = error_of(fit(one_way_input(), FeEffects::OneWay, cov));
    assert!(matches!(
        err,
        PanelError::Common(CommonError::InsufficientClustersForInference { .. })
    ));
}

#[test]
fn a_perfect_structural_fit_is_a_second_stage_error() {
    // `y`が`0.5 x1 + 0.8 x2`と個体ごとの定数の和（構造残差が0）。
    let (_, x1, x2, z1, z2) = columns();
    let y: Vec<f64> = (0..40)
        .map(|r| 0.5 * x1[r] + 0.8 * x2[r] + (r / 4) as f64)
        .collect();
    let err = error_of(fit_one_way(custom(&y, &[x1], &[x2], &[z1, z2])));
    let report = err.report();
    assert_eq!(report.code(), ErrorCode::PerfectFit);
    assert_eq!(report.stage(), Some(&Stage::SecondStage));
}

#[test]
fn a_confidence_level_outside_the_unit_interval_is_rejected() {
    let err = error_of(FeIvEstimator::fit(
        one_way_input(),
        FeEffects::OneWay,
        entity_cluster(),
        1.5,
    ));
    assert!(matches!(err, PanelError::WithinRegressionFailed { .. }));
}

#[test]
fn the_third_dimension_is_a_multi_way_fit() {
    // `MultiWay`は3次元以上を要する。
    let err = error_of(fit(
        two_way_input(),
        FeEffects::MultiWay,
        FeCovType::Classical,
    ));
    assert!(matches!(
        err,
        PanelError::Common(CommonError::Internal { .. })
    ));
    // 3次元は受け付ける。
    let rows: Vec<usize> = (0..40).collect();
    fit(
        input_with(&rows, &[entity_labels(), time_labels(), third_labels()]),
        FeEffects::MultiWay,
        FeCovType::Classical,
    )
    .unwrap();
}

// ── 交互射影（不均衡な2-wayと3次元）が残す誤差 ───────────────────────

/// 行を落とした不均衡な2-way（個体・時点）の入力の列から作る。
fn unbalanced_custom(
    y: &[f64],
    x_exog: impl AsRef<[Vec<f64>]>,
    x_endog: impl AsRef<[Vec<f64>]>,
    instruments: impl AsRef<[Vec<f64>]>,
) -> FeIvInput {
    let skip = [3usize, 17, 22, 30];
    let keep: Vec<usize> = (0..40).filter(|r| !skip.contains(r)).collect();
    let pick = |column: &[f64]| -> Vec<f64> { keep.iter().map(|&r| column[r]).collect() };
    let picks =
        |columns: &[Vec<f64>]| -> Vec<Vec<f64>> { columns.iter().map(|c| pick(c)).collect() };
    let labels = |all: Vec<String>| -> GroupCodes {
        GroupCodes::from_labels(&keep.iter().map(|&r| all[r].clone()).collect::<Vec<_>>())
    };
    let (x_exog, x_endog, instruments) = (
        picks(x_exog.as_ref()),
        picks(x_endog.as_ref()),
        picks(instruments.as_ref()),
    );
    let names = |prefix: &str, n: usize| (0..n).map(|j| format!("{prefix}{j}")).collect();
    FeIvInput::from_group_codes(
        &pick(y),
        "y".to_string(),
        RoleColumns {
            names: names("x", x_exog.len()),
            columns: &x_exog,
        },
        RoleColumns {
            names: names("e", x_endog.len()),
            columns: &x_endog,
        },
        RoleColumns {
            names: names("z", instruments.len()),
            columns: &instruments,
        },
        vec![labels(entity_labels()), labels(time_labels())],
    )
    .unwrap()
}

fn fit_two_way(input: FeIvInput) -> Result<FeIvEstimator, PanelError> {
    fit(input, FeEffects::TwoWay, FeCovType::Classical)
}

#[test]
fn a_column_explained_by_the_two_effects_is_reported_after_the_alternating_projections() {
    // 個体の効果と時点の効果の和（時間・個体のどちらでも説明される）の操作変数は、不均衡でも変換後に
    // 分散が残らない（誤差は`noise`の範囲）。
    let (y, x1, x2, z1, _) = columns();
    let additive: Vec<f64> = (0..40)
        .map(|r| (r / 4) as f64 * 0.37 + (r % 4) as f64 * 1.9)
        .collect();
    let err = error_of(fit_two_way(unbalanced_custom(
        &y,
        vec![x1],
        vec![x2],
        vec![z1, additive],
    )));
    assert_eq!(
        err,
        PanelError::ZeroVarianceAfterDemeaning {
            role: ColumnRole::Instruments,
            column: "z1".to_string(),
        }
    );
}

#[test]
fn a_perfect_structural_fit_with_large_coefficients_is_rejected_after_the_alternating_projections()
{
    // 係数が大きい（5と80）ので、残差に乗る射影の誤差は`y`の大きさだけでは測れない。
    let (_, x1, x2, z1, z2) = columns();
    let y: Vec<f64> = (0..40)
        .map(|r| 5.0 * x1[r] + 80.0 * x2[r] + (r / 4) as f64 + 0.5 * (r % 4) as f64)
        .collect();
    // 打ち切りの許容度を緩めて、射影が残す誤差を大きくする。
    let absorption = FeAbsorption {
        tol: 1e-5,
        ..FeAbsorption::default()
    };
    let err = error_of(FeIvEstimator::fit_with_absorption(
        unbalanced_custom(&y, vec![x1], vec![x2], vec![z1, z2]),
        FeEffects::TwoWay,
        FeCovType::Classical,
        0.95,
        &absorption,
    ));
    assert_eq!(err.report().code(), ErrorCode::PerfectFit);
    assert_eq!(err.report().stage(), Some(&Stage::SecondStage));
}

#[test]
fn alternating_projections_that_do_not_converge_are_reported() {
    let absorption = FeAbsorption {
        max_iter: 1,
        ..FeAbsorption::default()
    };
    let rows: Vec<usize> = (0..40).filter(|r| ![3, 17, 22, 30].contains(r)).collect();
    let err = error_of(FeIvEstimator::fit_with_absorption(
        input_with(&rows, &[entity_labels(), time_labels()]),
        FeEffects::TwoWay,
        FeCovType::Classical,
        0.95,
        &absorption,
    ));
    assert!(
        matches!(err, PanelError::AbsorptionNotConverged { .. }),
        "{err:?}"
    );
}
