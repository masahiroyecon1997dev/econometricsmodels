//! 構造残差・R²・固定効果・第一段階・自由度・不変性。

use super::{
    all_rows, entity_labels, fit, input_with, one_way_input, third_labels, time_labels,
    two_way_input,
};
use crate::panel::common::TimeKeys;
use crate::panel::fe::{FeCovType, FeEffects, FixedEffects};
use crate::shared::group_codes::GroupCodes;

const TOL: f64 = 1e-10;

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= TOL * (1.0 + expected.abs()),
        "{label}: {actual} vs {expected}"
    );
}

/// 1-wayの個体平均を引いた列。
fn demean_by_entity(column: &[f64]) -> Vec<f64> {
    column
        .chunks(4)
        .flat_map(|rows| {
            let mean = rows.iter().sum::<f64>() / rows.len() as f64;
            rows.iter().map(move |v| v - mean)
        })
        .collect()
}

fn one_way_fit() -> crate::panel::fe_iv::FeIvEstimator {
    fit(one_way_input(), FeEffects::OneWay, FeCovType::Classical).unwrap()
}

// ── 構造残差・R²・固定効果 ─────────────────────────────────────────

#[test]
fn residuals_use_the_actual_endogenous_variable() {
    let estimator = one_way_fit();
    let input = one_way_input();
    let (b1, b2) = (*estimator.params().get(0, 0), *estimator.params().get(1, 0));
    let (y, x1, x2) = (
        demean_by_entity(input.y()),
        demean_by_entity(&input.x_exog()[0]),
        demean_by_entity(&input.x_endog()[0]),
    );
    for i in 0..40 {
        close(
            *estimator.residuals().get(i, 0),
            y[i] - b1 * x1[i] - b2 * x2[i],
            "structural residual",
        );
    }
}

#[test]
fn the_r_squared_values_follow_the_structural_residuals() {
    let estimator = one_way_fit();
    let input = one_way_input();
    let ssr: f64 = (0..40)
        .map(|i| estimator.residuals().get(i, 0).powi(2))
        .sum();
    let y_tilde = demean_by_entity(input.y());
    let sst: f64 = y_tilde.iter().map(|v| v * v).sum();
    close(estimator.r_squared_within(), 1.0 - ssr / sst, "within");

    // overall: 元の`y`・`X`に`β̂`だけを当てはめた残差。
    let (b1, b2) = (*estimator.params().get(0, 0), *estimator.params().get(1, 0));
    let ssr_overall: f64 = (0..40)
        .map(|i| (input.y()[i] - b1 * input.x_exog()[0][i] - b2 * input.x_endog()[0][i]).powi(2))
        .sum();
    let tss_overall: f64 = input.y().iter().map(|v| v * v).sum();
    close(
        estimator.r_squared_overall(),
        1.0 - ssr_overall / tss_overall,
        "overall",
    );

    // between: 個体平均に`β̂`を当てはめた残差（`FeEstimator`と同じ式）。
    let means = |column: &[f64]| -> Vec<f64> {
        column
            .chunks(4)
            .map(|rows| rows.iter().sum::<f64>() / 4.0)
            .collect()
    };
    let (my, mx1, mx2) = (
        means(input.y()),
        means(&input.x_exog()[0]),
        means(&input.x_endog()[0]),
    );
    let ssr_between: f64 = (0..10)
        .map(|g| (my[g] - b1 * mx1[g] - b2 * mx2[g]).powi(2))
        .sum();
    let tss_between: f64 = my.iter().map(|v| v * v).sum();
    close(
        estimator.r_squared_between(),
        1.0 - ssr_between / tss_between,
        "between",
    );
}

#[test]
fn one_way_fixed_effects_are_the_entity_means_of_the_structural_residual_without_the_slopes() {
    let estimator = one_way_fit();
    let input = one_way_input();
    let (b1, b2) = (*estimator.params().get(0, 0), *estimator.params().get(1, 0));
    let FixedEffects::OneWay(effects) = estimator.fixed_effects() else {
        panic!("a 1-way fit gives one-way effects");
    };
    assert_eq!(effects.len(), 10);
    for (g, rows) in (0..10).map(|g| (g, g * 4..g * 4 + 4)) {
        let mean = rows
            .map(|i| input.y()[i] - b1 * input.x_exog()[0][i] - b2 * input.x_endog()[0][i])
            .sum::<f64>()
            / 4.0;
        close(effects[&g.to_string()], mean, "alpha_i");
    }
}

// ── 第一段階 ──────────────────────────────────────────────────────

#[test]
fn each_first_stage_is_orthogonal_to_the_exogenous_variables_and_instruments() {
    let estimator = one_way_fit();
    let input = one_way_input();
    assert_eq!(estimator.first_stages().len(), 1);
    let stage = &estimator.first_stages()[0];
    assert_eq!(stage.within_input().param_names(), ["x1", "z1", "z2"]);
    for column in [
        &input.x_exog()[0],
        &input.instruments()[0],
        &input.instruments()[1],
    ] {
        let tilde = demean_by_entity(column);
        let dot: f64 = (0..40)
            .map(|i| tilde[i] * stage.residuals().get(i, 0))
            .sum();
        close(dot, 0.0, "orthogonality");
    }
}

#[test]
fn a_first_stage_does_not_test_the_slopes_but_keeps_the_standard_errors() {
    let estimator = one_way_fit();
    assert_eq!(estimator.first_stages()[0].f_statistic(), None);
    assert!(estimator.f_statistic().is_some());
    assert_eq!(estimator.first_stages()[0].std_errors().nrows(), 3);
}

#[test]
fn clusters_not_above_the_first_stage_slopes_still_give_the_standard_errors() {
    // 3クラスター: 構造式の傾き2個より多いが、第一段階の傾き3個以下。
    let cov = FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&third_labels())),
    };
    let estimator = fit(one_way_input(), FeEffects::OneWay, cov).unwrap();
    assert_eq!(estimator.df_inference(), 2);
    assert_eq!(estimator.first_stages()[0].f_statistic(), None);
    for j in 0..3 {
        let se = *estimator.first_stages()[0].std_errors().get(j, 0);
        assert!(se.is_finite() && se > 0.0, "first-stage se {se}");
    }
}

// ── 自由度 ────────────────────────────────────────────────────────

#[test]
fn the_degrees_of_freedom_follow_the_number_of_absorbed_parameters() {
    let one_way = one_way_fit();
    assert_eq!(one_way.n_groups(), [10]);
    assert_eq!((one_way.df_model(), one_way.df_resid()), (12, 28));
    assert_eq!(one_way.df_inference(), 28);

    let two_way = fit(two_way_input(), FeEffects::TwoWay, FeCovType::Hc1).unwrap();
    assert_eq!(two_way.n_groups(), [10, 4]);
    // `neffects = 10 + 4 - 1`。
    assert_eq!((two_way.df_model(), two_way.df_resid()), (15, 25));

    let three_way = fit(
        input_with(
            &all_rows(),
            &[entity_labels(), time_labels(), third_labels()],
        ),
        FeEffects::MultiWay,
        FeCovType::Classical,
    )
    .unwrap();
    // `neffects = 10 + 4 + 3 - 2`。
    assert_eq!((three_way.df_model(), three_way.df_resid()), (17, 23));
}

#[test]
fn the_inference_degrees_of_freedom_depend_on_the_cov_type() {
    let cluster = fit(
        one_way_input(),
        FeEffects::OneWay,
        FeCovType::Cluster { groups: None },
    )
    .unwrap();
    assert_eq!(cluster.df_inference(), 9);
    assert_eq!(cluster.f_df(), Some((2, 9)));

    let ids: Vec<String> = time_labels();
    let time = TimeKeys::lexicographic(ids);
    let dk = fit(
        two_way_input(),
        FeEffects::TwoWay,
        FeCovType::Dk {
            bandwidth: Some(1),
            time,
        },
    )
    .unwrap();
    assert_eq!(dk.df_inference(), 3);
    assert_eq!(dk.dk_bandwidth_used(), Some(1));
}

// ── 不変性 ────────────────────────────────────────────────────────

#[test]
fn adding_an_entity_constant_to_the_outcome_changes_nothing_but_the_fixed_effects() {
    let reference = one_way_fit();
    let input = one_way_input();
    let shifted_y: Vec<f64> = (0..40)
        .map(|r| input.y()[r] + 3.0 * (r / 4) as f64 - 1.0)
        .collect();
    let shifted = crate::panel::fe_iv::FeIvInput::from_group_codes(
        &shifted_y,
        "y".to_string(),
        crate::panel::fe_iv::RoleColumns {
            names: vec!["x1".to_string()],
            columns: input.x_exog(),
        },
        crate::panel::fe_iv::RoleColumns {
            names: vec!["x2".to_string()],
            columns: input.x_endog(),
        },
        crate::panel::fe_iv::RoleColumns {
            names: vec!["z1".to_string(), "z2".to_string()],
            columns: input.instruments(),
        },
        vec![GroupCodes::from_labels(&entity_labels())],
    )
    .unwrap();
    let estimator = fit(shifted, FeEffects::OneWay, FeCovType::Classical).unwrap();
    for j in 0..2 {
        close(
            *estimator.params().get(j, 0),
            *reference.params().get(j, 0),
            "params",
        );
        close(
            *estimator.std_errors().get(j, 0),
            *reference.std_errors().get(j, 0),
            "std_errors",
        );
    }
}

#[test]
fn reordering_the_rows_does_not_change_the_estimates() {
    let reference = fit(
        two_way_input(),
        FeEffects::TwoWay,
        FeCovType::Cluster { groups: None },
    )
    .unwrap();
    let rows: Vec<usize> = (0..40).rev().collect();
    let reordered = fit(
        input_with(&rows, &[entity_labels(), time_labels()]),
        FeEffects::TwoWay,
        FeCovType::Cluster { groups: None },
    )
    .unwrap();
    for j in 0..2 {
        close(
            *reordered.params().get(j, 0),
            *reference.params().get(j, 0),
            "params",
        );
        close(
            *reordered.std_errors().get(j, 0),
            *reference.std_errors().get(j, 0),
            "std_errors",
        );
    }
}

// ── 並列度 ────────────────────────────────────────────────────────

#[test]
fn fit_pins_faer_global_parallelism_to_seq() {
    // `fit()`冒頭の`ensure_serial()`が、faerのグローバル並列度を`Par::Seq`へ引き戻す回帰ガード。
    faer::set_global_parallelism(faer::Par::rayon(0));
    let _ = one_way_fit();
    assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
}
