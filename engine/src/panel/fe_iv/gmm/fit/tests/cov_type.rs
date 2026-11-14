//! `cov_type`別の標準誤差・推論・Wald χ²・R²・検証。
//!
//! 独立な確かめ方は3つ:
//! - **FE-2SLSへの還元**: `OneStep`と`Classical`重みの2段階は係数が2SLSと同じで、サンドイッチ
//!   `B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`は2SLSの`X̂`のサンドイッチになる。`Classical`・`Hc1`・`Cluster`・`Dk`の共分散は
//!   `FeIvEstimator`と一致する（`Hc2`・`Hc3`は`X̂`と`Z̃`でレバレッジが違うので一致しない）。
//! - **ダミー変数を明示したGMM**: 同じ重みでダミーを外生変数に加えた`iv::gmm::GmmEstimator`は、`OneStep`と
//!   `Classical`重みの2段階では残差がwithin残差と同じなので（FWL）、`Classical`・`Hc1`・`Hc2`・`Hc3`の
//!   共分散がwithin変換した固定効果つきGMMと一致する（レバレッジはダミーのレバレッジを足したもの）。
//!   ロバスト・クラスター重みの2段階は残差が違うので一致しない（`reference.rs`冒頭）。
//! - **手計算**: 個体内の平均を引いた列から、素朴な二重ループ・LU分解で`Ω̂`とサンドイッチを組む
//!   （`Robust`・`Cluster`重みの2段階。1-way）。

use faer::Mat;
use faer::prelude::Solve;
use statrs::distribution::{ContinuousCDF, Normal};

use super::reference::{
    Structure, assert_params_close, cluster_s, dummy_variable_gmm_params, manual_one_way_full,
    structures,
};
use super::{
    entity_cluster_weight, entity_labels, fit_with, input_spec, one_way_input, time_labels,
    two_step,
};
use crate::iv::common::{IvEstimate, IvInput};
use crate::iv::gmm::GmmEstimator;
use crate::linear::ols::CovType;
use crate::panel::common::{PanelError, TimeKeys};
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::{FeGmmEstimator, FeGmmType, FeIvEstimator, FeIvInput, FeWeightType};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::ErrorCode;
use crate::shared::group_codes::GroupCodes;

const RTOL: f64 = 1e-8;

fn assert_mat_close(actual: &Mat<f64>, expected: &Mat<f64>, label: &str) {
    assert_eq!(
        (actual.nrows(), actual.ncols()),
        (expected.nrows(), expected.ncols()),
        "{label}"
    );
    let scale = (0..expected.nrows())
        .flat_map(|i| (0..expected.ncols()).map(move |j| (i, j)))
        .map(|(i, j)| expected.get(i, j).abs())
        .fold(1e-12, f64::max);
    for i in 0..expected.nrows() {
        for j in 0..expected.ncols() {
            let (got, want) = (*actual.get(i, j), *expected.get(i, j));
            assert!(
                (got - want).abs() <= RTOL * scale,
                "{label}[{i}][{j}]: {got} vs {want}"
            );
        }
    }
}

/// `rows`の行に対応する時点（全行分のラベルから拾う）。
fn dk(bandwidth: Option<i64>, rows: &[usize]) -> FeCovType {
    let all = time_labels();
    let picked: Vec<String> = rows.iter().map(|&r| all[r].clone()).collect();
    FeCovType::Dk {
        bandwidth,
        time: TimeKeys::lexicographic(picked),
    }
}

fn entity_cluster() -> FeCovType {
    FeCovType::Cluster { groups: None }
}

fn time_cluster(rows: &[usize]) -> FeCovType {
    let all = time_labels();
    let picked: Vec<String> = rows.iter().map(|&r| all[r].clone()).collect();
    FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&picked)),
    }
}

// ── FE-2SLSへの還元 ───────────────────────────────────────────────

#[test]
fn covariance_reduces_to_fe_2sls_for_classical_weight() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures()
    {
        for (k_exog, k_endog, k_inst) in [(1, 1, 1), (1, 1, 2), (1, 1, 3), (0, 2, 3)] {
            let make = || input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            let cov_types = [
                FeCovType::Classical,
                FeCovType::Hc1,
                entity_cluster(),
                time_cluster(&rows),
                dk(Some(0), &rows),
                dk(Some(2), &rows),
                dk(None, &rows),
            ];
            for cov_type in cov_types {
                let expected = FeIvEstimator::fit(make(), effects, cov_type.clone(), 0.95).unwrap();
                for gmm_type in [
                    FeGmmType::OneStep,
                    two_step(FeWeightType::Classical),
                    FeGmmType::Iterated {
                        weight: FeWeightType::Classical,
                        max_iter: 10,
                        tol: 1e-8,
                    },
                ] {
                    let label =
                        format!("{name} {k_exog}/{k_endog}/{k_inst} {cov_type:?} {gmm_type:?}");
                    let estimator = fit_with(make(), effects, gmm_type, cov_type.clone()).unwrap();
                    assert_mat_close(estimator.cov_params(), expected.cov_params(), &label);
                    assert_mat_close(estimator.std_errors(), expected.std_errors(), &label);
                    assert_eq!(
                        estimator.dk_bandwidth_used(),
                        expected.dk_bandwidth_used(),
                        "{label}"
                    );
                }
            }
        }
    }
}

/// 丁度識別は、どの重みでも係数が2SLSと同じなので、ロバスト・クラスター重みでも共分散が2SLSと一致する
/// （`Ω̂`は重みに依らず、`B`が正方で`W`が相殺する）。
#[test]
fn just_identified_covariance_equals_fe_2sls_for_any_weight() {
    let rows = super::all_rows();
    let make = || input_spec(&rows, &[entity_labels(), time_labels()], 1, 1, 1);
    for cov_type in [FeCovType::Hc1, entity_cluster(), dk(Some(1), &rows)] {
        let expected =
            FeIvEstimator::fit(make(), FeEffects::TwoWay, cov_type.clone(), 0.95).unwrap();
        for weight in [FeWeightType::Robust, super::entity_cluster_weight()] {
            let estimator = fit_with(
                make(),
                FeEffects::TwoWay,
                two_step(weight.clone()),
                cov_type.clone(),
            )
            .unwrap();
            assert_mat_close(
                estimator.cov_params(),
                expected.cov_params(),
                &format!("{cov_type:?} {weight:?}"),
            );
        }
    }
}

// ── ダミー変数を明示したGMM（FWL） ──────────────────────────────────────

/// ダミーを外生変数に加えた`iv::gmm::GmmEstimator`の係数の共分散（実際の外生変数・内生変数の分だけ）。
fn dummy_variable_gmm_cov(input: &FeIvInput, gmm_type: &FeGmmType, cov_type: CovType) -> Mat<f64> {
    let d = super::dummies(input.index().effect_codes());
    let (ke, kn) = (input.k_exog(), input.k_endog());
    let mut x_exog: Vec<Vec<f64>> = input.x_exog().to_vec();
    x_exog.extend(d.iter().cloned());
    let mut exog_names: Vec<String> = input.x_exog_names().to_vec();
    exog_names.extend((0..d.len()).map(|j| format!("d{j}")));
    let iv_input = IvInput::from_columns(
        input.y(),
        &x_exog,
        exog_names,
        input.x_endog(),
        input.x_endog_names().to_vec(),
        input.instruments(),
        input.instrument_names().to_vec(),
        false,
        "y".to_string(),
    )
    .unwrap();
    let estimator =
        GmmEstimator::fit(iv_input, gmm_type.to_gmm_type(), true, cov_type, 0.95).unwrap();
    let cov = estimator.cov_params();
    let endog_start = ke + d.len();
    let picked: Vec<usize> = (0..ke).chain(endog_start..endog_start + kn).collect();
    Mat::from_fn(picked.len(), picked.len(), |a, b| {
        *cov.get(picked[a], picked[b])
    })
}

/// `Classical`・`Hc1`・`Hc2`・`Hc3`は、ダミー込みのGMM（`OneStep`・`Classical`重みの2段階）と一致する。
/// ダミー込みの`HC1`の補正`n/(n−k−ダミー数)`はwithin変換した`n/df_resid`、HC2・HC3のレバレッジは
/// ダミー込みの`Z`のレバレッジ（`Z̃`の素のレバレッジ＋吸収した効果のレバレッジ）。
#[test]
fn covariance_matches_dummy_variable_gmm() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures().into_iter()
    {
        for (k_exog, k_endog, k_inst) in [(1, 1, 2), (0, 2, 3)] {
            let make = || input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            for gmm_type in [FeGmmType::OneStep, two_step(FeWeightType::Classical)] {
                for (fe_cov, iv_cov) in [
                    (FeCovType::Classical, CovType::Classical),
                    (FeCovType::Hc1, CovType::Hc1),
                    (FeCovType::Hc2, CovType::Hc2),
                    (FeCovType::Hc3, CovType::Hc3),
                ] {
                    let label =
                        format!("{name} {k_exog}/{k_endog}/{k_inst} {gmm_type:?} {fe_cov:?}");
                    let expected = dummy_variable_gmm_cov(&make(), &gmm_type, iv_cov);
                    let estimator = fit_with(make(), effects, gmm_type.clone(), fe_cov).unwrap();
                    assert_mat_close(estimator.cov_params(), &expected, &label);
                }
            }
        }
    }
}

/// 同じ重みでも、ダミー込みGMMの係数が`FeGmmEstimator`と一致する（共分散の比較の前提）。
#[test]
fn dummy_variable_gmm_has_the_same_coefficients() {
    let make = || input_spec(&super::all_rows(), &[entity_labels()], 1, 1, 2);
    let estimator = fit_with(
        make(),
        FeEffects::OneWay,
        FeGmmType::OneStep,
        FeCovType::Hc2,
    )
    .unwrap();
    let expected = dummy_variable_gmm_params(&make(), &FeGmmType::OneStep);
    assert_params_close(estimator.params(), &expected, "one step");
}

// ── 手計算 ────────────────────────────────────────────────────────

/// `B⁻¹ (X'Z W Ω̂ W Z'X) B⁻¹`（`W = S⁻¹`、`B = X'ZWZ'X`）。LU分解で解く（本体のCholeskyとは別の経路）。
fn sandwich(z: &Mat<f64>, x: &Mat<f64>, s: &Mat<f64>, omega: &Mat<f64>) -> Mat<f64> {
    let zx = z.transpose() * x;
    let s_inv_zx = s.partial_piv_lu().solve(&zx);
    let bread = zx.transpose() * &s_inv_zx;
    let meat = s_inv_zx.transpose() * omega * &s_inv_zx;
    let b_inv = bread
        .partial_piv_lu()
        .solve(Mat::<f64>::identity(bread.nrows(), bread.nrows()));
    &b_inv * meat * &b_inv
}

fn outer_sum(z: &Mat<f64>, scale: impl Fn(usize) -> f64) -> Mat<f64> {
    let (n, l) = (z.nrows(), z.ncols());
    let mut omega = Mat::<f64>::zeros(l, l);
    for i in 0..n {
        let w = scale(i);
        for a in 0..l {
            for b in 0..l {
                *omega.get_mut(a, b) += w * z.get(i, a) * z.get(i, b);
            }
        }
    }
    omega
}

/// 10個体×4時点・1-wayのFE。`entity_group`は個体、時点は`r % 4`。
fn manual_omega(cov_type: &FeCovType, z: &Mat<f64>, e: &Mat<f64>, k: usize) -> Mat<f64> {
    let (n, l) = (z.nrows(), z.ncols());
    let (n_entities, periods) = (10_usize, 4_usize);
    let df_model = k + n_entities;
    let df_resid = n - df_model;
    let e2 = |i: usize| e.get(i, 0).powi(2);
    match cov_type {
        FeCovType::Classical => {
            let sigma2 = (0..n).map(e2).sum::<f64>() / df_resid as f64;
            outer_sum(z, |_| sigma2)
        }
        FeCovType::Hc1 => outer_sum(z, |i| n as f64 / df_resid as f64 * e2(i)),
        FeCovType::Hc2 | FeCovType::Hc3 => {
            let ztz = z.transpose() * z;
            let ztz_inv = ztz.partial_piv_lu().solve(Mat::<f64>::identity(l, l));
            let h = |i: usize| {
                let mut quad = 0.0;
                for a in 0..l {
                    for b in 0..l {
                        quad += z.get(i, a) * ztz_inv.get(a, b) * z.get(i, b);
                    }
                }
                // 個体ダミーのレバレッジ`1/T_i`。
                quad + 1.0 / periods as f64
            };
            if matches!(cov_type, FeCovType::Hc2) {
                outer_sum(z, |i| e2(i) / (1.0 - h(i)))
            } else {
                outer_sum(z, |i| e2(i) / (1.0 - h(i)).powi(2))
            }
        }
        FeCovType::Cluster { groups } => {
            // 個体（行は個体ごと）か時点（`r % 4`）。
            let by_time = groups.is_some();
            let group: Vec<usize> = (0..n)
                .map(|r| if by_time { r % periods } else { r / periods })
                .collect();
            let g = group.iter().max().unwrap() + 1;
            // fixestの`K.fixef="nonnested"`: 個体でのクラスターは個体効果に入れ子（個体効果を`1`と数える）、
            // 時点でのクラスターは入れ子でない（個体効果を全て数える）。
            let big_k = if by_time { df_model } else { k + 1 };
            let correction =
                (g as f64 / (g as f64 - 1.0)) * ((n as f64 - 1.0) / (n - big_k) as f64);
            let mut omega = cluster_s(z, e, &group);
            for a in 0..l {
                for b in 0..l {
                    *omega.get_mut(a, b) *= correction;
                }
            }
            omega
        }
        FeCovType::Dk { bandwidth, .. } => {
            let bw = bandwidth.unwrap_or(0) as usize;
            // ξ_t = Σ_{i: 時点t} e_i z_i。
            let xi = |t: usize, a: usize| -> f64 {
                (0..n)
                    .filter(|r| r % periods == t)
                    .map(|r| e.get(r, 0) * z.get(r, a))
                    .sum()
            };
            let mut s_hat = Mat::<f64>::zeros(l, l);
            for lag in 0..=bw {
                let weight = 1.0 - lag as f64 / (bw as f64 + 1.0);
                for t in lag..periods {
                    for a in 0..l {
                        for b in 0..l {
                            let term = xi(t, a) * xi(t - lag, b);
                            if lag == 0 {
                                *s_hat.get_mut(a, b) += term;
                            } else {
                                *s_hat.get_mut(a, b) += weight * (term + xi(t - lag, a) * xi(t, b));
                            }
                        }
                    }
                }
            }
            let scale = (periods as f64 / (periods as f64 - 1.0))
                * ((n as f64 - 1.0) / (n - df_model) as f64);
            Mat::from_fn(l, l, |a, b| scale * s_hat.get(a, b))
        }
    }
}

/// `Robust`・`Cluster`重みの2段階GMMの共分散は、手計算の`Ω̂`・サンドイッチと一致する。
#[test]
fn covariance_matches_manual_computation() {
    let rows = super::all_rows();
    let entity_group: Vec<usize> = (0..40).map(|r| r / 4).collect();
    let time_cluster_cov = time_cluster(&rows);
    let cov_types = [
        FeCovType::Classical,
        FeCovType::Hc1,
        FeCovType::Hc2,
        FeCovType::Hc3,
        entity_cluster(),
        time_cluster_cov,
        dk(Some(0), &rows),
        dk(Some(2), &rows),
    ];
    for (k_exog, k_endog, k_inst) in [(1, 1, 2), (0, 2, 3)] {
        let make = || input_spec(&rows, &[entity_labels()], k_exog, k_endog, k_inst);
        for (weight, label) in [
            (FeWeightType::Robust, "robust"),
            (entity_cluster_weight(), "cluster"),
        ] {
            // 手計算の重み（クラスター重みは個体、ロバスト重みは各行が別クラスター）。
            let group: Vec<usize> = if label == "robust" {
                (0..40).collect()
            } else {
                entity_group.clone()
            };
            let manual = manual_one_way_full(&make(), &group, 2);
            for cov_type in &cov_types {
                let expected_omega =
                    manual_omega(cov_type, &manual.z, &manual.residuals, k_exog + k_endog);
                let expected = sandwich(&manual.z, &manual.x, &manual.s, &expected_omega);
                let estimator = fit_with(
                    make(),
                    FeEffects::OneWay,
                    two_step(weight.clone()),
                    cov_type.clone(),
                )
                .unwrap();
                assert_mat_close(
                    estimator.cov_params(),
                    &expected,
                    &format!("{k_exog}/{k_endog}/{k_inst} {label} {cov_type:?}"),
                );
            }
        }
    }
}

// ── 推論・Wald・R² ────────────────────────────────────────────────

#[test]
fn inference_table_uses_the_normal_distribution() {
    let rows = super::all_rows();
    let estimator = FeGmmEstimator::fit(
        input_spec(&rows, &[entity_labels()], 1, 1, 2),
        FeEffects::OneWay,
        two_step(entity_cluster_weight()),
        true,
        entity_cluster(),
        0.9,
    )
    .unwrap();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_crit = normal.inverse_cdf(0.95);
    for j in 0..2 {
        let (beta, se) = (
            *estimator.params().get(j, 0),
            *estimator.std_errors().get(j, 0),
        );
        assert!((se - estimator.cov_params().get(j, j).sqrt()).abs() <= 1e-12 * se);
        let z = beta / se;
        assert!((estimator.test_stats().get(j, 0) - z).abs() <= 1e-12 * z.abs().max(1.0));
        let p = 2.0 * (1.0 - normal.cdf(z.abs()));
        assert!((estimator.p_values().get(j, 0) - p).abs() <= 1e-10);
        assert!((estimator.conf_lower().get(j, 0) - (beta - z_crit * se)).abs() <= 1e-10);
        assert!((estimator.conf_upper().get(j, 0) - (beta + z_crit * se)).abs() <= 1e-10);
    }
    // 共分散は対称。
    let cov = estimator.cov_params();
    assert_eq!(*cov.get(0, 1), *cov.get(1, 0));
}

/// 傾き係数の同時Wald χ²検定は`β'V⁻¹β`（自由度`k`）。係数が1つなら`z²`に等しい。
#[test]
fn wald_statistic_is_the_quadratic_form_and_equals_z_squared_for_one_coefficient() {
    let rows = super::all_rows();
    let two = fit_with(
        input_spec(&rows, &[entity_labels()], 1, 1, 2),
        FeEffects::OneWay,
        two_step(FeWeightType::Robust),
        FeCovType::Hc1,
    )
    .unwrap();
    let beta = two.params();
    let solved = two.cov_params().partial_piv_lu().solve(beta);
    let quad = (beta.transpose() * &solved).get(0, 0).to_owned();
    let got = two.wald_statistic().unwrap();
    assert!((got - quad).abs() <= 1e-8 * quad.abs(), "{got} vs {quad}");
    let chi2 = statrs::distribution::ChiSquared::new(2.0).unwrap();
    let p = 1.0 - statrs::distribution::ContinuousCDF::cdf(&chi2, got);
    assert!((two.wald_p_value().unwrap() - p).abs() <= 1e-9);

    let one = fit_with(
        input_spec(&rows, &[entity_labels()], 0, 1, 2),
        FeEffects::OneWay,
        two_step(FeWeightType::Robust),
        FeCovType::Hc1,
    )
    .unwrap();
    let z = *one.test_stats().get(0, 0);
    assert!((one.wald_statistic().unwrap() - z * z).abs() <= 1e-9 * (z * z));
    assert!((one.wald_p_value().unwrap() - *one.p_values().get(0, 0)).abs() <= 1e-9);
}

/// R²は構造残差ベースで、`OneStep`（2SLSと同じ係数・残差）ではFE-2SLSと同じ。
#[test]
fn r_squared_matches_fe_2sls_for_one_step() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures()
    {
        let make = || input_spec(&rows, &dimensions, 1, 1, 2);
        let expected = FeIvEstimator::fit(make(), effects, FeCovType::Hc1, 0.95).unwrap();
        let estimator = fit_with(make(), effects, FeGmmType::OneStep, FeCovType::Hc1).unwrap();
        for (got, want, what) in [
            (
                estimator.r_squared_within(),
                expected.r_squared_within(),
                "within",
            ),
            (
                estimator.r_squared_between(),
                expected.r_squared_between(),
                "between",
            ),
            (
                estimator.r_squared_overall(),
                expected.r_squared_overall(),
                "overall",
            ),
        ] {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "{name} {what}: {got} vs {want}"
            );
        }
    }
}

// ── 検証 ──────────────────────────────────────────────────────────

fn error_of(result: Result<FeGmmEstimator, PanelError>) -> PanelError {
    result.map(|_| ()).unwrap_err()
}

#[test]
fn cluster_count_must_exceed_the_number_of_coefficients() {
    let make = || input_spec(&super::all_rows(), &[entity_labels()], 1, 1, 2);
    let groups = |g: usize| {
        let labels: Vec<String> = (0..40).map(|r| format!("g{}", r % g)).collect();
        FeCovType::Cluster {
            groups: Some(GroupCodes::from_labels(&labels)),
        }
    };
    // `k = 2`。`G = 2`は`G <= k`で拒否、`G = 3`は受理。
    let err = error_of(fit_with(
        make(),
        FeEffects::OneWay,
        FeGmmType::OneStep,
        groups(2),
    ));
    assert_eq!(
        err,
        PanelError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
    );
    assert!(fit_with(make(), FeEffects::OneWay, FeGmmType::OneStep, groups(3)).is_ok());
}

#[test]
fn dk_periods_must_exceed_the_number_of_coefficients_and_bandwidth_must_fit() {
    let rows = super::all_rows();
    // `T = 4`、`k = 4`（外生2・内生2）は`T <= k`で拒否。
    let wide = || input_spec(&rows, &[entity_labels()], 2, 2, 3);
    let err = error_of(fit_with(
        wide(),
        FeEffects::OneWay,
        FeGmmType::OneStep,
        dk(Some(1), &rows),
    ));
    assert!(matches!(err, PanelError::InStage { .. }), "{err:?}");
    // バンド幅は`[0, T)`。
    let err = error_of(fit_with(
        input_spec(&rows, &[entity_labels()], 1, 1, 2),
        FeEffects::OneWay,
        FeGmmType::OneStep,
        dk(Some(4), &rows),
    ));
    assert_eq!(err, PanelError::InvalidDkBandwidth { bandwidth: 4, t: 4 });
}

#[test]
fn invalid_confidence_levels_are_rejected() {
    for level in [0.0, 1.0, -0.5, f64::NAN] {
        let err = error_of(FeGmmEstimator::fit(
            one_way_input(),
            FeEffects::OneWay,
            FeGmmType::OneStep,
            true,
            FeCovType::Classical,
            level,
        ));
        assert!(
            matches!(err, PanelError::WithinRegressionFailed { .. }),
            "level={level}: {err:?}"
        );
        // `confidence_level`の範囲違反は段階を付けずに返す。
        assert_eq!(err.report().code(), ErrorCode::ConfidenceLevelOutOfRange);
        assert_eq!(err.report().stage(), None);
    }
}

/// `Cluster { groups: None }`は個体（`effects`の1つ目）でクラスタリングする。
#[test]
fn default_cluster_is_the_entity() {
    let make = || input_spec(&super::all_rows(), &[entity_labels()], 1, 1, 2);
    let explicit = FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&entity_labels())),
    };
    let a = fit_with(
        make(),
        FeEffects::OneWay,
        two_step(FeWeightType::Robust),
        entity_cluster(),
    )
    .unwrap();
    let b = fit_with(
        make(),
        FeEffects::OneWay,
        two_step(FeWeightType::Robust),
        explicit,
    )
    .unwrap();
    assert_eq!(a.cov_params(), b.cov_params());
}
