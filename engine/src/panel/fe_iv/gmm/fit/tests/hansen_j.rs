//! Hansen J過剰識別検定。
//!
//! 独立な確かめ方は3つ:
//! - **FE-2SLSのSarganへの還元**: `Classical`重みのJは、`S = σ̂²Z̃'Z̃`（`σ̂² = e'e/(n − neffects)`）で、
//!   `e'Z̃(Z̃'Z̃)⁻¹Z̃'e/σ̂²`のSarganと同じ。`OneStep`・`Classical`重みの2段階・反復で一致する。
//! - **ダミー変数を明示したGMM**: ダミーを外生変数に加えた`iv::gmm::GmmEstimator`の`J`（`σ̂² = e'e/n`）に
//!   `(n − neffects)/n`を掛けた値と一致する（`OneStep`と`Classical`重みの2段階。ダミーのモーメントは
//!   `D'ê = 0`で`J`に寄与しない）。
//! - **手計算**: `Robust`・`Cluster`重みは補正せず、個体内の平均を引いた列から`(Z̃'ẽ)'S⁻¹(Z̃'ẽ)`を組む。

use faer::Mat;
use faer::prelude::Solve;

use super::reference::{
    Structure, dummy_iv_input, manual_dummy_projection_full, manual_one_way_full, structures,
};
use super::{entity_cluster_weight, entity_labels, fit, fit_with, input_spec, two_step};
use crate::iv::gmm::GmmEstimator;
use crate::linear::ols::CovType;
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::{
    FeGmmEstimator, FeGmmType, FeIvEstimator, FeIvInput, FeWeightType, RoleColumns,
};
use crate::shared::group_codes::GroupCodes;

const RTOL: f64 = 1e-8;

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= RTOL * expected.abs().max(1e-12),
        "{label}: {actual} vs {expected}"
    );
}

/// 過剰識別の入力（外生変数・内生変数・操作変数の数）。
const OVER_IDENTIFIED: [(usize, usize, usize); 4] = [(1, 1, 2), (1, 1, 3), (0, 2, 3), (2, 1, 3)];

fn classical_types() -> [FeGmmType; 3] {
    [
        FeGmmType::OneStep,
        two_step(FeWeightType::Classical),
        FeGmmType::Iterated {
            weight: FeWeightType::Classical,
            max_iter: 10,
            tol: 1e-8,
        },
    ]
}

// ── FE-2SLSのSarganへの還元 ───────────────────────────────────────

#[test]
fn classical_weight_equals_fe_2sls_sargan() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures()
    {
        for (k_exog, k_endog, k_inst) in OVER_IDENTIFIED {
            let make = || input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            let sargan = FeIvEstimator::fit(make(), effects, FeCovType::Classical, 0.95).unwrap();
            for gmm_type in classical_types() {
                let label = format!("{name} {k_exog}/{k_endog}/{k_inst} {gmm_type:?}");
                let estimator = fit(make(), effects, gmm_type).unwrap();
                assert_close(
                    estimator.hansen_j_statistic().unwrap(),
                    sargan.sargan_statistic().unwrap(),
                    &label,
                );
                assert_close(
                    estimator.hansen_j_p_value().unwrap(),
                    sargan.sargan_p_value().unwrap(),
                    &label,
                );
                assert_eq!(estimator.hansen_j_df(), sargan.sargan_df(), "{label}");
            }
        }
    }
}

// ── ダミー変数を明示したGMM ───────────────────────────────────────

#[test]
fn matches_dummy_variable_hansen_j_after_rescaling_the_variance() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures().into_iter().take(3)
    {
        for (k_exog, k_endog, k_inst) in [(1, 1, 2), (0, 2, 3)] {
            let make = || input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            for gmm_type in [FeGmmType::OneStep, two_step(FeWeightType::Classical)] {
                let label = format!("{name} {k_exog}/{k_endog}/{k_inst} {gmm_type:?}");
                let estimator = fit(make(), effects, gmm_type.clone()).unwrap();
                let n = estimator.nobs();
                // 吸収した効果の数は、ダミーの列数（`iv::gmm`の入力の外生変数の列数から実際の外生変数を引く）で
                // 独立に数える。
                let dummy_input = dummy_iv_input(&make());
                let neffects = dummy_input.k_exog() - k_exog;
                assert_eq!(
                    neffects,
                    estimator.df_model() - (k_exog + k_endog),
                    "{label}"
                );
                let dummy = GmmEstimator::fit(
                    dummy_input,
                    gmm_type.to_gmm_type(),
                    true,
                    CovType::Classical,
                    0.95,
                )
                .unwrap();
                // ダミー込みは`σ̂² = e'e/n`なので、`(n − neffects)/n`倍するとwithinの`σ̂²`の尺度になる。
                let expected =
                    dummy.hansen_j_statistic().unwrap() * (n - neffects) as f64 / n as f64;
                assert_close(estimator.hansen_j_statistic().unwrap(), expected, &label);
                assert_eq!(estimator.hansen_j_df(), dummy.hansen_j_df(), "{label}");
            }
        }
    }
}

// ── 手計算 ────────────────────────────────────────────────────────

/// `J = (Z̃'ẽ)'S⁻¹(Z̃'ẽ)`。LU分解で解く（本体のCholeskyとは別の経路）。
fn manual_j(z: &Mat<f64>, e: &Mat<f64>, s: &Mat<f64>) -> f64 {
    let zte = z.transpose() * e;
    let solved = s.partial_piv_lu().solve(&zte);
    (0..zte.nrows())
        .map(|i| zte.get(i, 0) * solved.get(i, 0))
        .sum()
}

/// `Robust`・`Cluster`重みのJは、補正なしの`S`で組んだ手計算と一致する（2段階と3回の反復）。
#[test]
fn robust_and_cluster_weights_are_not_rescaled() {
    let make = || input_spec(&super::all_rows(), &[entity_labels()], 1, 1, 3);
    let entity_group: Vec<usize> = (0..40).map(|r| r / 4).collect();
    let row_group: Vec<usize> = (0..40).collect();
    for (weight, group, label) in [
        (FeWeightType::Robust, &row_group, "robust"),
        (entity_cluster_weight(), &entity_group, "cluster"),
    ] {
        for estimates in [2_usize, 3] {
            let manual = manual_one_way_full(&make(), group, estimates);
            let expected = manual_j(&manual.z, &manual.residuals, &manual.s);
            let gmm_type = if estimates == 2 {
                two_step(weight.clone())
            } else {
                FeGmmType::Iterated {
                    weight: weight.clone(),
                    max_iter: estimates,
                    tol: 1e-300,
                }
            };
            let estimator = FeGmmEstimator::fit(
                make(),
                FeEffects::OneWay,
                gmm_type,
                false,
                FeCovType::Classical,
                0.95,
            )
            .unwrap();
            let label = format!("{label} estimates={estimates}");
            assert_close(estimator.hansen_j_statistic().unwrap(), expected, &label);
            assert_eq!(estimator.hansen_j_df(), Some(2), "{label}");
        }
    }
}

// ── `None`の条件・検出力 ─────────────────────────────────────────────

/// 丁度識別（操作変数の数が内生変数の数に等しい）は検定が無い。
#[test]
fn just_identified_has_no_test() {
    for gmm_type in [
        FeGmmType::OneStep,
        two_step(FeWeightType::Robust),
        two_step(entity_cluster_weight()),
    ] {
        let estimator = fit(
            input_spec(&super::all_rows(), &[entity_labels()], 1, 1, 1),
            FeEffects::OneWay,
            gmm_type,
        )
        .unwrap();
        assert_eq!(estimator.hansen_j_statistic(), None);
        assert_eq!(estimator.hansen_j_p_value(), None);
        assert_eq!(estimator.hansen_j_df(), None);
    }
}

/// 自由度は`len(instruments) − len(x_endog)`、`J`は非負。`J`は`cov_type`に依らない（標準誤差の報告方法と
/// 検定は別。`S`は点推定に使った重みだけで決まる）。
#[test]
fn degrees_of_freedom_follow_the_instrument_count_and_j_ignores_cov_type() {
    let rows = super::all_rows();
    for (k_exog, k_endog, k_inst, df) in [(1, 1, 2, 1), (1, 1, 3, 2), (0, 2, 3, 1)] {
        let make = || input_spec(&rows, &[entity_labels()], k_exog, k_endog, k_inst);
        let base = fit_with(
            make(),
            FeEffects::OneWay,
            two_step(entity_cluster_weight()),
            FeCovType::Classical,
        )
        .unwrap();
        assert_eq!(base.hansen_j_df(), Some(df));
        assert!(base.hansen_j_statistic().unwrap() >= 0.0);
        for cov_type in [
            FeCovType::Hc1,
            FeCovType::Hc2,
            FeCovType::Hc3,
            FeCovType::Cluster { groups: None },
        ] {
            let other = fit_with(
                make(),
                FeEffects::OneWay,
                two_step(entity_cluster_weight()),
                cov_type.clone(),
            )
            .unwrap();
            assert_eq!(
                other.hansen_j_statistic(),
                base.hansen_j_statistic(),
                "{cov_type:?}"
            );
            assert_eq!(
                other.hansen_j_p_value(),
                base.hansen_j_p_value(),
                "{cov_type:?}"
            );
        }
    }
}

/// 内生変数が無い入力（操作変数だけがある）は、検定が無い。
#[test]
fn no_endogenous_variable_has_no_test() {
    let base = input_spec(&super::all_rows(), &[entity_labels()], 1, 1, 2);
    let input = FeIvInput::from_group_codes(
        base.y(),
        "y".to_string(),
        RoleColumns {
            names: base.x_exog_names().to_vec(),
            columns: base.x_exog(),
        },
        RoleColumns {
            names: Vec::new(),
            columns: &[],
        },
        RoleColumns {
            names: base.instrument_names().to_vec(),
            columns: base.instruments(),
        },
        vec![GroupCodes::from_labels(&entity_labels())],
    )
    .unwrap();
    let estimator = fit(input, FeEffects::OneWay, two_step(FeWeightType::Robust)).unwrap();
    assert_eq!(estimator.hansen_j_statistic(), None);
    assert_eq!(estimator.hansen_j_df(), None);
}

/// 2-way・不均衡・3次元でも、`Cluster`重みの`J`は、ダミーへの回帰の残差から組んだ手計算（補正なし）と
/// 一致する（`n`・`neffects`の取り違えは`Classical`・`OneStep`の再スケールにだけ効くので、ここでは
/// 再スケールが掛からないことと、`S`・残差の配線を見る）。
#[test]
fn cluster_weight_matches_manual_computation_for_multi_way_effects() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures().into_iter().skip(1)
    {
        let make = || input_spec(&rows, &dimensions, 1, 1, 3);
        let labels: Vec<String> = rows.iter().map(|&r| dimensions[0][r].clone()).collect();
        let group: Vec<usize> = GroupCodes::from_labels(&labels).codes().to_vec();
        let weight = FeWeightType::Cluster {
            groups: GroupCodes::from_labels_without_keys(&labels),
        };
        let manual = manual_dummy_projection_full(&make(), &group, 2);
        let expected = manual_j(&manual.z, &manual.residuals, &manual.s);
        let estimator = fit(make(), effects, two_step(weight)).unwrap();
        assert_close(estimator.hansen_j_statistic().unwrap(), expected, name);
    }
}

/// 決定的な疑似乱数（線形合同法＋Box-Muller）による標準正規。
fn normals(seed: u64, count: usize) -> Vec<f64> {
    // 連番のシードで系列が似ないよう、シードを散らす。
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678);
    let mut uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let (u1, u2) = (uniform(), uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

/// 外生性を満たさない操作変数（構造式の誤差と相関する）を足すと、Jが大きくなりp値が小さくなる。
/// 30個体×4時点。`x = 0.8z1 + 0.8z2 + 0.6u + v`（内生）、`y = α_i + x + w + u`。有効な操作変数は
/// `z1`・`z2`、無効な操作変数は`z3 = 0.9u + 0.5ν`。
#[test]
fn an_invalid_instrument_is_rejected() {
    let (n_entities, periods) = (30_usize, 4_usize);
    let n = n_entities * periods;
    let (z1, z2, z_noise) = (normals(1, n), normals(2, n), normals(3, n));
    let (u, v, w) = (normals(4, n), normals(5, n), normals(6, n));
    let alpha = normals(7, n_entities);
    let x: Vec<f64> = (0..n)
        .map(|i| 0.8 * z1[i] + 0.8 * z2[i] + 0.6 * u[i] + v[i])
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| alpha[i / periods] + x[i] + w[i] + u[i])
        .collect();
    let z3: Vec<f64> = (0..n).map(|i| 0.9 * u[i] + 0.5 * z_noise[i]).collect();
    let labels: Vec<String> = (0..n).map(|i| (i / periods).to_string()).collect();
    let build = |instruments: Vec<Vec<f64>>| -> FeIvInput {
        let names: Vec<String> = (0..instruments.len()).map(|j| format!("z{j}")).collect();
        FeIvInput::from_group_codes(
            &y,
            "y".to_string(),
            RoleColumns {
                names: vec!["w".to_string()],
                columns: std::slice::from_ref(&w),
            },
            RoleColumns {
                names: vec!["x".to_string()],
                columns: std::slice::from_ref(&x),
            },
            RoleColumns {
                names,
                columns: &instruments,
            },
            vec![GroupCodes::from_labels(&labels)],
        )
        .unwrap()
    };
    let cluster = FeWeightType::Cluster {
        groups: GroupCodes::from_labels_without_keys(&labels),
    };
    for weight in [FeWeightType::Classical, cluster] {
        let valid = fit(
            build(vec![z1.clone(), z2.clone()]),
            FeEffects::OneWay,
            two_step(weight.clone()),
        )
        .unwrap();
        let invalid = fit(
            build(vec![z1.clone(), z2.clone(), z3.clone()]),
            FeEffects::OneWay,
            two_step(weight.clone()),
        )
        .unwrap();
        let (p_valid, p_invalid) = (
            valid.hansen_j_p_value().unwrap(),
            invalid.hansen_j_p_value().unwrap(),
        );
        assert!(p_valid > 0.01, "{weight:?}: valid p = {p_valid}");
        assert!(p_invalid < 0.001, "{weight:?}: invalid p = {p_invalid}");
        assert_eq!(valid.hansen_j_df(), Some(1));
        assert_eq!(invalid.hansen_j_df(), Some(2));
    }
}
