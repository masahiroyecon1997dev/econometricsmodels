//! C統計量（`fe_gmm_c_statistic`）。
//!
//! 独立な確かめ方は3つ:
//! - **ダミー変数を明示した`iv::gmm::compute_c_statistic`**: 効果のダミーを外生変数に加えると、`Classical`重みの
//!   `J_e`・`J_c`・`C`は、`(n − neffects)/n`を掛けた値と一致する（FWL。ダミーのモーメントは`J`に寄与しない。
//!   `Robust`・`Cluster`は2段階目の残差がwithin残差と違うので一致しない）。
//! - **手計算**: within変換した列（1-wayは素朴な平均除去、2-way・不均衡・3次元はダミーへの回帰の残差）から、
//!   `Z_full`・`Z_c`・`S_e`・`S_c`をLU分解で組んで`J_e − J_c`を求める。
//! - **Hansen Jとの整合**: 除外操作変数だけを検定するとき、`J_e`は同じ重みの2段階GMMの`fit`のHansen Jと
//!   一致する（`S`の尺度の規則が同じであることの確認）。

use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::hansen_j::manual_j;
use super::reference::{
    Structure, cluster_s, demean_by_blocks, dummy_iv_input, dummy_projection_demean, gmm_estimate,
    mat, structures,
};
use super::{
    all_rows, entity_cluster_weight, entity_labels, fit, input_spec, one_way_input, third_labels,
    two_step,
};
use crate::iv::gmm::compute_c_statistic;
use crate::panel::common::PanelError;
use crate::panel::fe::{FeAbsorption, FeEffects};
use crate::panel::fe_iv::{FeGmmType, FeIvInput, FeWeightType, RoleColumns, fe_gmm_c_statistic};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::ErrorCode;
use crate::shared::gmm_c_statistic::{CStatistic, CStatisticTarget};
use crate::shared::gmm_weight::WeightType;
use crate::shared::group_codes::GroupCodes;

const RTOL: f64 = 1e-8;

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= RTOL * expected.abs().max(1e-8),
        "{label}: {actual} vs {expected}"
    );
}

fn target(endog: &[usize], instruments: &[usize]) -> CStatisticTarget {
    CStatisticTarget {
        endog: endog.to_vec(),
        instruments: instruments.to_vec(),
    }
}

fn c_statistic(
    input: &FeIvInput,
    effects: FeEffects,
    target: &CStatisticTarget,
    weight: &FeWeightType,
) -> Result<CStatistic, PanelError> {
    fe_gmm_c_statistic(input, effects, &FeAbsorption::default(), target, weight)
}

/// 外生変数・内生変数・操作変数の数と、検定する対象の組（どれも`Z_c`が識別される）。
fn cases() -> Vec<((usize, usize, usize), CStatisticTarget)> {
    vec![
        ((1, 1, 2), target(&[0], &[])),
        ((1, 1, 2), target(&[], &[1])),
        ((1, 1, 2), target(&[0], &[1])),
        ((1, 1, 3), target(&[], &[2])),
        ((1, 1, 3), target(&[], &[1, 2])),
        ((1, 1, 3), target(&[0], &[0])),
        ((0, 2, 3), target(&[1], &[])),
        ((0, 2, 3), target(&[0, 1], &[])),
        ((0, 2, 3), target(&[], &[2])),
        ((2, 1, 3), target(&[0], &[2])),
    ]
}

// ── ダミー変数を明示した`iv::gmm` ──────────────────────────────────────

#[test]
fn classical_weight_matches_the_dummy_variable_c_statistic_after_rescaling() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures().into_iter().take(3)
    {
        for ((k_exog, k_endog, k_inst), target) in cases() {
            let label = format!("{name} {k_exog}/{k_endog}/{k_inst} {target:?}");
            let input = input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            let n = input.nobs();
            // 吸収した効果の数は、ダミーの列数（外生変数の列数の増分）で独立に数える。
            let dummy_input = dummy_iv_input(&input);
            let neffects = dummy_input.k_exog() - k_exog;
            let dummy = compute_c_statistic(&dummy_input, &target, &WeightType::Classical).unwrap();
            // ダミー込みは`σ̂² = e'e/n`なので、`(n − neffects)/n`倍するとwithinの`σ̂²`の尺度になる。
            let scale = (n - neffects) as f64 / n as f64;

            let fe = c_statistic(&input, effects, &target, &FeWeightType::Classical).unwrap();
            assert_close(fe.statistic, dummy.statistic * scale, &label);
            assert_close(fe.j_with_tested, dummy.j_with_tested * scale, &label);
            assert_close(fe.j_without_tested, dummy.j_without_tested * scale, &label);
            assert_eq!(fe.df, dummy.df, "{label}");
            let expected_p = ChiSquared::new(fe.df as f64)
                .unwrap()
                .sf(dummy.statistic * scale);
            assert_close(fe.p_value, expected_p, &label);
        }
    }
}

// ── 手計算 ────────────────────────────────────────────────────────

/// 手計算の重み`S_e`の種類。
enum ManualWeight<'a> {
    /// `σ̂² = e'e/(n − neffects)`の`σ̂²Z'Z`。
    Classical { neffects: usize },
    /// `Σ ẽ²zz'`（各行が1つのクラスター）。
    Robust,
    /// クラスター`group`（行ごとの添字）。
    Cluster(&'a [usize]),
}

/// `(J_e, J_c)`を、within変換した列から素朴に組む。`J_c`は`S_e`の部分行列`S_c`を重みに`Z_c`で推定する。
fn manual_j_pair(
    input: &FeIvInput,
    demean: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    weight: &ManualWeight<'_>,
    target: &CStatisticTarget,
) -> (f64, f64) {
    let y = mat(&[demean(&input.y().to_vec())]);
    let exog: Vec<Vec<f64>> = input.x_exog().iter().map(demean).collect();
    let endog: Vec<Vec<f64>> = input.x_endog().iter().map(demean).collect();
    let instruments: Vec<Vec<f64>> = input.instruments().iter().map(demean).collect();
    let n = y.nrows();

    let x = mat(&[exog.clone(), endog.clone()].concat());
    let tested_endog: Vec<Vec<f64>> = target.endog.iter().map(|&j| endog[j].clone()).collect();
    let z_full = mat(&[exog.clone(), tested_endog, instruments.clone()].concat());
    // `Z_c`は、`Z_full`の外生変数と、検定しない除外操作変数。
    let kept_instruments: Vec<usize> = (0..instruments.len())
        .filter(|j| !target.instruments.contains(j))
        .collect();
    let z_c = mat(&[
        exog.clone(),
        kept_instruments
            .iter()
            .map(|&j| instruments[j].clone())
            .collect(),
    ]
    .concat());
    let kept_columns: Vec<usize> = (0..exog.len())
        .chain(
            kept_instruments
                .iter()
                .map(|&j| exog.len() + target.endog.len() + j),
        )
        .collect();

    // Z_fullで1段階目（2SLS）→S_e→2段階目。
    let ztz = z_full.transpose() * &z_full;
    let beta0 = gmm_estimate(&z_full, &x, &y, &ztz);
    let e0 = &y - &x * &beta0;
    let s_e = match weight {
        ManualWeight::Classical { neffects } => {
            let ssr: f64 = (0..n).map(|i| e0.get(i, 0).powi(2)).sum();
            let sigma2 = ssr / (n - neffects) as f64;
            Mat::from_fn(ztz.nrows(), ztz.ncols(), |a, b| sigma2 * ztz.get(a, b))
        }
        ManualWeight::Robust => cluster_s(&z_full, &e0, &(0..n).collect::<Vec<_>>()),
        ManualWeight::Cluster(group) => cluster_s(&z_full, &e0, group),
    };
    let beta_e = gmm_estimate(&z_full, &x, &y, &s_e);
    let j_e = manual_j(&z_full, &(&y - &x * &beta_e), &s_e);

    // Z_cが丁度識別ならJ_c = 0。そうでなければ、S_eの部分行列を重みに推定する。
    let j_c = if z_c.ncols() == x.ncols() {
        0.0
    } else {
        let s_c = Mat::from_fn(kept_columns.len(), kept_columns.len(), |a, b| {
            *s_e.get(kept_columns[a], kept_columns[b])
        });
        let beta_c = gmm_estimate(&z_c, &x, &y, &s_c);
        manual_j(&z_c, &(&y - &x * &beta_c), &s_c)
    };
    (j_e, j_c)
}

/// 個体でクラスタリングする重みと、そのクラスターの添字（行ごと）。
fn entity_cluster(rows: &[usize], dimensions: &[Vec<String>]) -> (FeWeightType, Vec<usize>) {
    let labels: Vec<String> = rows.iter().map(|&r| dimensions[0][r].clone()).collect();
    (
        FeWeightType::Cluster {
            groups: GroupCodes::from_labels_without_keys(&labels),
        },
        GroupCodes::from_labels(&labels).codes().to_vec(),
    )
}

/// `Classical`・`Robust`・`Cluster`の重みのC統計量は、手計算（`S_e`・`S_c`をLU分解で組む）と一致する。
/// `Classical`は`σ̂²`の分母が`n − neffects`、`Robust`・`Cluster`は補正なし。
#[test]
fn matches_the_manual_computation_for_every_weight_and_effect_structure() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures()
    {
        let (cluster_weight, group) = entity_cluster(&rows, &dimensions);
        for ((k_exog, k_endog, k_inst), target) in cases() {
            let input = input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            let demean = dummy_projection_demean(&input);
            let neffects = super::dummies(input.index().effect_codes()).len();
            let weights = [
                (
                    FeWeightType::Classical,
                    ManualWeight::Classical { neffects },
                    "classical",
                ),
                (FeWeightType::Robust, ManualWeight::Robust, "robust"),
                (
                    cluster_weight.clone(),
                    ManualWeight::Cluster(&group),
                    "cluster",
                ),
            ];
            for (weight, manual_weight, weight_name) in weights {
                let label = format!("{name} {weight_name} {k_exog}/{k_endog}/{k_inst} {target:?}");
                let (j_e, j_c) = manual_j_pair(&input, &demean, &manual_weight, &target);
                let fe = c_statistic(&input, effects, &target, &weight).unwrap();
                assert_close(fe.j_with_tested, j_e, &label);
                assert_close(fe.j_without_tested, j_c, &label);
                assert_close(fe.statistic, (j_e - j_c).max(0.0), &label);
            }
        }
    }
}

/// 1-wayは、ダミーへの回帰ではなく素朴な個体内の平均除去からも同じ値になる（変換の経路の違い）。
#[test]
fn one_way_matches_the_manual_computation_with_naive_demeaning() {
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 1, 3);
    let demean = |c: &Vec<f64>| demean_by_blocks(c, 4);
    let entity_group: Vec<usize> = (0..40).map(|r| r / 4).collect();
    let target = target(&[0], &[2]);
    for (weight, manual_weight) in [
        (
            entity_cluster_weight(),
            ManualWeight::Cluster(&entity_group),
        ),
        (
            FeWeightType::Classical,
            ManualWeight::Classical { neffects: 10 },
        ),
    ] {
        let (j_e, j_c) = manual_j_pair(&input, &demean, &manual_weight, &target);
        let fe = c_statistic(&input, FeEffects::OneWay, &target, &weight).unwrap();
        assert_close(fe.j_with_tested, j_e, "J_e");
        assert_close(fe.j_without_tested, j_c, "J_c");
    }
}

// ── Hansen Jとの整合・不変条件 ───────────────────────────────────────

/// 除外操作変数だけを検定するとき、`J_e`は同じ重みの2段階GMMの`fit`のHansen Jと一致する
/// （`Classical`の`σ̂²`の分母の補正も、`Robust`・`Cluster`の無補正も同じ規則）。
#[test]
fn j_with_tested_equals_the_hansen_j_of_the_two_step_fit() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures()
    {
        let (cluster_weight, _) = entity_cluster(&rows, &dimensions);
        let make = || input_spec(&rows, &dimensions, 1, 1, 3);
        for weight in [
            FeWeightType::Classical,
            FeWeightType::Robust,
            cluster_weight,
        ] {
            let label = format!("{name} {weight:?}");
            let estimator = fit(make(), effects, two_step(weight.clone())).unwrap();
            let c = c_statistic(&make(), effects, &target(&[], &[2]), &weight).unwrap();
            assert_close(
                c.j_with_tested,
                estimator.hansen_j_statistic().unwrap(),
                &label,
            );
        }
        // `OneStep`は`Classical`の重みと同じ点推定・同じ`S`の尺度。
        let one_step = fit(make(), effects, FeGmmType::OneStep).unwrap();
        let c = c_statistic(
            &make(),
            effects,
            &target(&[], &[2]),
            &FeWeightType::Classical,
        )
        .unwrap();
        assert_close(
            c.j_with_tested,
            one_step.hansen_j_statistic().unwrap(),
            name,
        );
    }
}

/// `Z_c`が丁度識別なら`J_c = 0`で、`C = J_e`。
#[test]
fn restricted_model_just_identified_gives_c_equal_to_j_with_tested() {
    for weight in [
        FeWeightType::Classical,
        FeWeightType::Robust,
        entity_cluster_weight(),
    ] {
        let c = c_statistic(
            &input_spec(&all_rows(), &[entity_labels()], 1, 1, 2),
            FeEffects::OneWay,
            &target(&[], &[1]),
            &weight,
        )
        .unwrap();
        assert_eq!(c.j_without_tested, 0.0, "{weight:?}");
        assert_eq!(c.statistic, c.j_with_tested, "{weight:?}");
        assert_eq!(c.df, 1);
    }
}

/// `C ≥ 0`、`p`は`χ²(df)`の上側確率、`df`は検定するモーメント条件の数（全ての重み・効果の構造）。
#[test]
fn statistic_is_non_negative_and_p_value_follows_chi_squared() {
    for Structure {
        rows,
        dimensions,
        effects,
        ..
    } in structures()
    {
        let (cluster_weight, _) = entity_cluster(&rows, &dimensions);
        for ((k_exog, k_endog, k_inst), target) in cases() {
            let input = input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            for weight in [
                FeWeightType::Classical,
                FeWeightType::Robust,
                cluster_weight.clone(),
            ] {
                let c = c_statistic(&input, effects, &target, &weight).unwrap();
                assert!(c.statistic >= 0.0, "{target:?} {weight:?}");
                assert_eq!(c.df, target.endog.len() + target.instruments.len());
                let expected_p = ChiSquared::new(c.df as f64).unwrap().sf(c.statistic);
                assert_close(c.p_value, expected_p, &format!("{target:?}"));
            }
        }
    }
}

/// 検定する内生変数の列の指定（`x_endog`の何番目か）が、結果に反映される（別の列を指定すると値が変わる）。
#[test]
fn testing_a_different_endogenous_column_gives_a_different_statistic() {
    let input = input_spec(&all_rows(), &[entity_labels()], 0, 2, 3);
    let first = c_statistic(
        &input,
        FeEffects::OneWay,
        &target(&[0], &[]),
        &FeWeightType::Classical,
    )
    .unwrap();
    let second = c_statistic(
        &input,
        FeEffects::OneWay,
        &target(&[1], &[]),
        &FeWeightType::Classical,
    )
    .unwrap();
    assert!((first.statistic - second.statistic).abs() > 1e-6);
}

// ── 検証 ──────────────────────────────────────────────────────────

fn error_of(input: &FeIvInput, target: &CStatisticTarget, weight: &FeWeightType) -> PanelError {
    c_statistic(input, FeEffects::OneWay, target, weight).unwrap_err()
}

#[test]
fn an_empty_target_is_rejected() {
    let err = error_of(
        &one_way_input(),
        &target(&[], &[]),
        &FeWeightType::Classical,
    );
    assert_eq!(
        err,
        PanelError::Common(CommonError::CStatisticNothingToTest)
    );
    assert_eq!(err.report().code(), ErrorCode::CStatisticNothingToTest);
}

#[test]
fn testing_too_many_instruments_is_rejected() {
    // 内生変数2個に対し、除外操作変数は3個。2個外すと1個しか残らない。
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 2, 3);
    let err = error_of(&input, &target(&[], &[0, 1]), &FeWeightType::Classical);
    assert_eq!(
        err,
        PanelError::Common(CommonError::CStatisticTooManyTestedInstruments {
            n_tested: 2,
            n_instruments: 3,
            n_endog: 2,
        })
    );
}

#[test]
fn invalid_positions_are_internal_errors() {
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 1, 3);
    for bad in [
        target(&[1], &[]),
        target(&[], &[3]),
        target(&[], &[1, 1]),
        target(&[], &[2, 1]),
    ] {
        let err = error_of(&input, &bad, &FeWeightType::Classical);
        assert!(
            matches!(err, PanelError::Common(CommonError::Internal { .. })),
            "{bad:?}: {err:?}"
        );
    }
}

/// クラスター数が`Z_full`の列数に足りないと`S_e`が構造的に特異で、未定義。
#[test]
fn too_few_clusters_for_the_full_instrument_set_are_undefined() {
    // `Z_full = [x_exog(1), B₁(1), instruments(3)]`で5列。クラスターは3つ。
    let weight = FeWeightType::Cluster {
        groups: GroupCodes::from_labels_without_keys(&third_labels()),
    };
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 1, 3);
    let err = error_of(&input, &target(&[0], &[]), &weight);
    assert_eq!(err, PanelError::Common(CommonError::CStatisticUndefined));
    assert_eq!(err.report().code(), ErrorCode::CStatisticUndefined);
    // 除外操作変数だけの検定なら`Z_full`は4列で、個体のクラスター（10個）で定義できる。
    assert!(
        c_statistic(
            &input,
            FeEffects::OneWay,
            &target(&[], &[2]),
            &entity_cluster_weight()
        )
        .is_ok()
    );
}

#[test]
fn a_cluster_column_with_the_wrong_length_is_rejected() {
    let weight = FeWeightType::Cluster {
        groups: GroupCodes::from_labels_without_keys(&entity_labels()[..39]),
    };
    let err = error_of(&one_way_input(), &target(&[], &[1]), &weight);
    assert!(
        matches!(
            err,
            PanelError::Common(CommonError::ClusterDimensionMismatch { .. })
        ),
        "{err:?}"
    );
}

/// 構造式を完全に説明する`y`は、`S_e`が丸め誤差になる前に未定義とする。
#[test]
fn a_perfectly_fitted_outcome_is_undefined() {
    let base = one_way_input();
    let y: Vec<f64> = (0..40)
        .map(|r| 1.5 * base.x_exog()[0][r] - 0.7 * base.x_endog()[0][r])
        .collect();
    let input = FeIvInput::from_group_codes(
        &y,
        "y".to_string(),
        RoleColumns {
            names: base.x_exog_names().to_vec(),
            columns: base.x_exog(),
        },
        RoleColumns {
            names: base.x_endog_names().to_vec(),
            columns: base.x_endog(),
        },
        RoleColumns {
            names: base.instrument_names().to_vec(),
            columns: base.instruments(),
        },
        vec![GroupCodes::from_labels(&entity_labels())],
    )
    .unwrap();
    for weight in [
        FeWeightType::Classical,
        FeWeightType::Robust,
        entity_cluster_weight(),
    ] {
        let err = error_of(&input, &target(&[], &[1]), &weight);
        assert_eq!(
            err,
            PanelError::Common(CommonError::CStatisticUndefined),
            "{weight:?}"
        );
    }
}

/// 操作変数が共線（同じ列の複製）だと`Z_full'Z_full`が特異で、未定義。
#[test]
fn collinear_instruments_are_undefined() {
    let base = input_spec(&all_rows(), &[entity_labels()], 1, 1, 2);
    let mut instruments = base.instruments().to_vec();
    instruments.push(instruments[0].clone());
    let input = FeIvInput::from_group_codes(
        base.y(),
        "y".to_string(),
        RoleColumns {
            names: base.x_exog_names().to_vec(),
            columns: base.x_exog(),
        },
        RoleColumns {
            names: base.x_endog_names().to_vec(),
            columns: base.x_endog(),
        },
        RoleColumns {
            names: vec!["z1".to_string(), "z2".to_string(), "z1_copy".to_string()],
            columns: &instruments,
        },
        vec![GroupCodes::from_labels(&entity_labels())],
    )
    .unwrap();
    for weight in [FeWeightType::Classical, FeWeightType::Robust] {
        let err = error_of(&input, &target(&[0], &[]), &weight);
        assert_eq!(
            err,
            PanelError::Common(CommonError::CStatisticUndefined),
            "{weight:?}"
        );
    }
}

/// 交互射影の設定（不均衡な2-wayと3次元）が結果に反映される: 厳しい`tol`は既定と同じ値（打ち切り誤差の
/// 範囲）、掃引が足りなければ`AbsorptionNotConverged`、範囲外の値は検証で弾かれる。
#[test]
fn the_absorption_settings_are_applied() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures()
        .into_iter()
        .filter(|s| s.name != "one-way" && s.name != "two-way")
    {
        let input = input_spec(&rows, &dimensions, 1, 1, 3);
        let target = target(&[0], &[2]);
        let weight = FeWeightType::Classical;
        let default = c_statistic(&input, effects, &target, &weight).unwrap();
        let tight = fe_gmm_c_statistic(
            &input,
            effects,
            &FeAbsorption {
                tol: 1e-14,
                ..FeAbsorption::default()
            },
            &target,
            &weight,
        )
        .unwrap();
        assert!(
            (tight.statistic - default.statistic).abs() <= 1e-6 * default.statistic.max(1e-6),
            "{name}: {} vs {}",
            tight.statistic,
            default.statistic
        );
        let few_sweeps = fe_gmm_c_statistic(
            &input,
            effects,
            &FeAbsorption {
                tol: 1e-14,
                max_iter: 1,
            },
            &target,
            &weight,
        )
        .unwrap_err();
        assert!(
            matches!(few_sweeps, PanelError::AbsorptionNotConverged { .. }),
            "{name}: {few_sweeps:?}"
        );
    }
    let invalid = fe_gmm_c_statistic(
        &one_way_input(),
        FeEffects::OneWay,
        &FeAbsorption {
            tol: -1.0,
            ..FeAbsorption::default()
        },
        &target(&[0], &[]),
        &FeWeightType::Classical,
    )
    .unwrap_err();
    assert!(
        matches!(invalid, PanelError::InvalidTol { .. }),
        "{invalid:?}"
    );
}

/// 固定効果つきGMMの前段の検査（識別・自由度など）はそのまま働く。
#[test]
fn the_within_preparation_checks_apply() {
    // 操作変数1個で内生変数2個。
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 2, 1);
    let err = error_of(&input, &target(&[0], &[]), &FeWeightType::Classical);
    assert_eq!(
        err,
        PanelError::InsufficientInstruments {
            n_instruments: 1,
            n_endog: 2,
        }
    );
    // 2-wayで効果が1つ。
    let err = c_statistic(
        &one_way_input(),
        FeEffects::TwoWay,
        &target(&[0], &[]),
        &FeWeightType::Classical,
    )
    .unwrap_err();
    assert_eq!(err, PanelError::TwoWayRequiresTime);
}
