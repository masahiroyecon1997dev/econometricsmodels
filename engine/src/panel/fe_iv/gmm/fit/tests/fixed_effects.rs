//! 固定効果自体の復元（`fixed_effects()`）。構造式の`y`・実際の`X`・GMMの係数`β̂`から、FE-2SLSと同じ
//! 復元（`α_i = ȳ_i − x̄_i'β̂`、基準化を含む）をする。係数だけがFE-2SLSと違う。
//!
//! - 復元した効果は、構造式の残差`y − Xβ̂`を効果のダミーに射影した当てはめ値なので、残差から効果の和を引いた
//!   ものは、効果の各次元の各水準で和が0になる（正規方程式）。1-way・2-way・不均衡な2-way・3次元で、
//!   重みごと（係数が違う）に確かめる。
//! - `OneStep`は2SLSと同じ係数なので、FE-2SLSの`fixed_effects()`と一致する。

use std::collections::BTreeMap;

use super::{
    STRUCTURES, entity_labels, fit_with, model, one_way_input, third_labels, time_labels, two_step,
    unbalanced_rows,
};
use crate::panel::fe::{FeCovType, FeEffects, FixedEffects};
use crate::panel::fe_iv::gmm::{FeGmmEstimator, FeGmmType, FeWeightType};
use crate::panel::fe_iv::two_sls::FeTwoSlsEstimator;
use crate::shared::group_codes::GroupCodes;

/// 次元ごとの`水準 → 効果`。
fn effect_maps(fixed: FixedEffects) -> Vec<BTreeMap<String, f64>> {
    match fixed {
        FixedEffects::OneWay(entity) => vec![entity],
        FixedEffects::TwoWay { entity, time } => vec![entity, time.into_iter().collect()],
        FixedEffects::MultiWay(dimensions) => dimensions
            .into_iter()
            .map(|dimension| dimension.into_iter().collect())
            .collect(),
    }
}

/// 構造が使う行の、次元ごとのラベル。
fn dimension_labels(structure: &str) -> Vec<Vec<String>> {
    let pick = |labels: Vec<String>, rows: &[usize]| -> Vec<String> {
        rows.iter().map(|&r| labels[r].clone()).collect()
    };
    let rows: Vec<usize> = if structure == "two_way_unbalanced" {
        unbalanced_rows()
    } else {
        (0..entity_labels().len()).collect()
    };
    match structure {
        "one_way" | "just_identified" | "two_endogenous" => {
            vec![pick(entity_labels(), &rows)]
        }
        "two_way" | "two_way_unbalanced" => {
            vec![pick(entity_labels(), &rows), pick(time_labels(), &rows)]
        }
        "three_way" => vec![
            pick(entity_labels(), &rows),
            pick(time_labels(), &rows),
            pick(third_labels(), &rows),
        ],
        other => panic!("unknown structure {other}"),
    }
}

fn gmm_types(structure: &str) -> Vec<(&'static str, FeGmmType)> {
    let rows = if structure == "two_way_unbalanced" {
        unbalanced_rows()
    } else {
        (0..entity_labels().len()).collect()
    };
    let labels: Vec<String> = rows.iter().map(|&r| entity_labels()[r].clone()).collect();
    vec![
        ("one_step", FeGmmType::OneStep),
        ("two_step_robust", two_step(FeWeightType::Robust)),
        (
            "two_step_cluster",
            two_step(FeWeightType::Cluster {
                groups: GroupCodes::from_labels_without_keys(&labels),
            }),
        ),
    ]
}

/// 構造式の残差`y − Xβ̂`（変換前の`y`・実際の`X`と係数）。
fn structural_residual(estimator: &FeGmmEstimator) -> Vec<f64> {
    let input = estimator.input();
    let columns: Vec<&Vec<f64>> = input.x_exog().iter().chain(input.x_endog()).collect();
    (0..input.nobs())
        .map(|i| {
            input.y()[i]
                - columns
                    .iter()
                    .enumerate()
                    .map(|(j, column)| column[i] * estimator.params().get(j, 0))
                    .sum::<f64>()
        })
        .collect()
}

/// 復元した効果が正規方程式を満たす: 残差から効果の和を引いたものが、各次元の各水準で和が0。
#[test]
fn the_effects_project_the_structural_residual_onto_the_effect_dummies() {
    for structure in STRUCTURES {
        for (label, gmm_type) in gmm_types(structure) {
            let (input, effects) = model(structure);
            let estimator = fit_with(input, effects, gmm_type, FeCovType::Classical).unwrap();
            let maps = effect_maps(estimator.fixed_effects().unwrap());
            let dimensions = dimension_labels(structure);
            assert_eq!(maps.len(), dimensions.len(), "{structure}/{label}");

            let residual = structural_residual(&estimator);
            let rest: Vec<f64> = (0..residual.len())
                .map(|i| {
                    residual[i]
                        - maps
                            .iter()
                            .zip(&dimensions)
                            .map(|(map, labels)| map[&labels[i]])
                            .sum::<f64>()
                })
                .collect();
            let scale = residual.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for (d, labels) in dimensions.iter().enumerate() {
                let mut sums: BTreeMap<&String, f64> = BTreeMap::new();
                for (i, group) in labels.iter().enumerate() {
                    *sums.entry(group).or_default() += rest[i];
                }
                for (group, sum) in sums {
                    assert!(
                        sum.abs() <= 1e-9 * scale.max(1.0),
                        "{structure}/{label}: dimension {d}, group {group}: sum {sum}"
                    );
                }
            }
        }
    }
}

/// `OneStep`の係数は2SLSと同じなので、効果もFE-2SLSの`fixed_effects()`と一致する（基準化を含む）。
#[test]
fn one_step_effects_equal_those_of_fe_2sls() {
    for structure in STRUCTURES {
        let (input, effects) = model(structure);
        let gmm = fit_with(input, effects, FeGmmType::OneStep, FeCovType::Classical).unwrap();
        let (input, effects) = model(structure);
        let two_sls = FeTwoSlsEstimator::fit(input, effects, FeCovType::Classical, 0.95).unwrap();
        let (a, b) = (
            effect_maps(gmm.fixed_effects().unwrap()),
            effect_maps(two_sls.fixed_effects()),
        );
        assert_eq!(a.len(), b.len(), "{structure}");
        for (d, (a, b)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{structure}: dimension {d}"
            );
            for (group, value) in a {
                assert!(
                    (value - b[group]).abs() <= 1e-9 * b[group].abs().max(1.0),
                    "{structure}: dimension {d}, group {group}: {value} vs {}",
                    b[group]
                );
            }
        }
    }
}

/// 過剰識別で重みがロバストなら係数が2SLSと違うので、効果も違う（GMMの係数で復元している）。
#[test]
fn effects_follow_the_gmm_coefficients_not_the_2sls_ones() {
    let gmm = fit_with(
        one_way_input(),
        FeEffects::OneWay,
        two_step(FeWeightType::Robust),
        FeCovType::Classical,
    )
    .unwrap();
    let two_sls = FeTwoSlsEstimator::fit(
        one_way_input(),
        FeEffects::OneWay,
        FeCovType::Classical,
        0.95,
    )
    .unwrap();
    let (a, b) = (
        effect_maps(gmm.fixed_effects().unwrap()),
        effect_maps(two_sls.fixed_effects()),
    );
    let largest_difference = a[0]
        .iter()
        .map(|(group, value)| (value - b[0][group]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        largest_difference > 1e-6,
        "the robust-weight GMM coefficients must differ from 2SLS: {largest_difference}"
    );
}
