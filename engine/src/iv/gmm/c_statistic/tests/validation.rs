//! 入力検証・エラー返却・未定義。

use super::*;
use crate::iv::common::IvError;
use crate::iv::gmm::{CStatisticTarget, compute_c_statistic};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::ErrorCode;

fn target(endog: &[usize], instruments: &[usize]) -> CStatisticTarget {
    CStatisticTarget {
        endog: endog.to_vec(),
        instruments: instruments.to_vec(),
    }
}

fn error_of(input: &IvInput, target: &CStatisticTarget, weight: &WeightType) -> IvError {
    compute_c_statistic(input, target, weight).unwrap_err()
}

// ── 検定するモーメント条件 ──────────────────────────────────────────

#[test]
fn returns_nothing_to_test_when_the_target_is_empty() {
    let err = error_of(&input(), &target(&[], &[]), &WeightType::Robust);
    assert_eq!(err, IvError::CStatisticNothingToTest);
    assert_eq!(err.report().code(), ErrorCode::CStatisticNothingToTest);
}

#[test]
fn returns_an_internal_error_for_positions_that_are_out_of_range_repeated_or_unsorted() {
    for bad in [
        target(&[2], &[]),
        target(&[], &[3]),
        target(&[0, 0], &[]),
        target(&[1, 0], &[]),
        target(&[], &[2, 1]),
    ] {
        let err = error_of(&input(), &bad, &WeightType::Robust);
        assert!(
            matches!(err, IvError::Common(CommonError::Internal { .. })),
            "{bad:?}: {err:?}"
        );
    }
}

#[test]
fn returns_too_many_tested_instruments_when_the_remaining_instruments_cannot_identify_the_model() {
    // 内生変数2個に対し、除外操作変数は3個。2個外すと1個しか残らない。
    let err = error_of(&input(), &target(&[], &[0, 1]), &WeightType::Robust);
    assert_eq!(
        err,
        IvError::CStatisticTooManyTestedInstruments {
            n_tested: 2,
            n_instruments: 3,
            n_endog: 2,
        }
    );
    assert_eq!(
        err.report().code(),
        ErrorCode::CStatisticTooManyTestedInstruments
    );
}

#[test]
fn accepts_removing_instruments_down_to_the_number_of_endogenous_variables() {
    let result = compute_c_statistic(&input(), &target(&[], &[2]), &WeightType::Robust).unwrap();
    assert_eq!(result.df, 1);
}

#[test]
fn counts_the_degrees_of_freedom_as_the_number_of_tested_endogenous_variables_and_instruments() {
    let result =
        compute_c_statistic(&input(), &target(&[0, 1], &[2]), &WeightType::Classical).unwrap();
    assert_eq!(result.df, 3);
}

// ── 重み（クラスター・HAC）の設定 ───────────────────────────────────

#[test]
fn returns_a_missing_cluster_column_error_when_the_cluster_weight_has_no_groups() {
    let err = error_of(
        &input(),
        &target(&[0], &[]),
        &WeightType::Cluster { groups: None },
    );
    assert!(
        matches!(
            err,
            IvError::Common(CommonError::MissingClusterColumn { .. })
        ),
        "{err:?}"
    );
}

#[test]
fn returns_an_error_when_the_cluster_codes_do_not_match_the_number_of_rows() {
    let labels: Vec<String> = (0..10).map(|i| format!("g{}", i % 3)).collect();
    let weight = WeightType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&labels)),
    };
    let err = error_of(&input(), &target(&[0], &[]), &weight);
    assert!(matches!(err, IvError::Common(_)), "{err:?}");
}

#[test]
fn returns_an_error_when_the_hac_lags_are_out_of_range() {
    let weight = WeightType::Hac {
        lags: Some(N as i64),
        time_order: (0..N).map(|i| i as f64).collect(),
    };
    let err = error_of(&input(), &target(&[0], &[]), &weight);
    assert!(
        matches!(err, IvError::Common(CommonError::InvalidHacLags { .. })),
        "{err:?}"
    );
}

// ── 未定義 ──────────────────────────────────────────────────────────

#[test]
fn is_undefined_when_the_clusters_are_fewer_than_the_moment_conditions_of_the_model_with_tested_moments()
 {
    // 検定する内生変数1個: Z_fullは定数・x1・d1・z1・z2・z3の6列。6クラスターなら計算でき、5では
    // 構造的に特異。
    let at_boundary = compute_c_statistic(&input(), &target(&[0], &[]), &cluster_weight(6));
    assert!(at_boundary.is_ok());
    let err = error_of(&input(), &target(&[0], &[]), &cluster_weight(5));
    assert_eq!(err, IvError::CStatisticUndefined);
    assert_eq!(err.report().code(), ErrorCode::CStatisticUndefined);
}

#[test]
fn counts_the_tested_variables_in_the_moment_conditions_that_the_clusters_must_cover() {
    // 内生変数2個を検定すると7列になり、6クラスターでは足りない。
    let err = error_of(&input(), &target(&[0, 1], &[]), &cluster_weight(6));
    assert_eq!(err, IvError::CStatisticUndefined);
}

#[test]
fn is_undefined_when_the_instruments_are_collinear() {
    let z1 = data::Z1.to_vec();
    let input = IvInput::from_columns(
        &data::Y,
        &[data::X1.to_vec()],
        vec!["x1".to_string()],
        &[data::D1.to_vec(), data::D2.to_vec()],
        vec!["d1".to_string(), "d2".to_string()],
        &[z1.clone(), data::Z2.to_vec(), z1],
        vec!["z1".to_string(), "z2".to_string(), "z1_copy".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let err = error_of(&input, &target(&[0], &[]), &WeightType::Robust);
    assert_eq!(err, IvError::CStatisticUndefined);
}

#[test]
fn is_undefined_when_the_outcome_is_fitted_perfectly() {
    // yが外生変数と内生変数の厳密な一次結合なら、残差が0でモーメント共分散が0行列になる。
    let y: Vec<f64> = (0..N)
        .map(|i| 1.0 + data::X1[i] + data::D1[i] + data::D2[i])
        .collect();
    let input = IvInput::from_columns(
        &y,
        &[data::X1.to_vec()],
        vec!["x1".to_string()],
        &[data::D1.to_vec(), data::D2.to_vec()],
        vec!["d1".to_string(), "d2".to_string()],
        &[data::Z1.to_vec(), data::Z2.to_vec(), data::Z3.to_vec()],
        vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    for weight in [
        WeightType::Classical,
        WeightType::Robust,
        hac_weight(),
        cluster_weight(8),
    ] {
        let err = error_of(&input, &target(&[0], &[]), &weight);
        assert_eq!(err, IvError::CStatisticUndefined, "{weight:?}");
    }
}
