//! 恒等式・不変性。

use super::*;
use crate::iv::gmm::{
    CStatisticTarget, GmmEstimator, GmmType, compute_c_statistic, weight_type_for_cov_type,
};
use crate::linear::ols::CovType;

fn target(endog: &[usize], instruments: &[usize]) -> CStatisticTarget {
    CStatisticTarget {
        endog: endog.to_vec(),
        instruments: instruments.to_vec(),
    }
}

fn weights() -> Vec<(&'static str, WeightType)> {
    vec![
        ("classical", WeightType::Classical),
        ("robust", WeightType::Robust),
        ("hac", hac_weight()),
        ("cluster", cluster_weight(8)),
    ]
}

fn assert_close(a: f64, b: f64, label: &str) {
    assert!(
        (a - b).abs() <= 1e-9 * b.abs().max(1.0),
        "{label}: {a:?} vs {b:?}"
    );
}

/// 列を差し替えた入力（`y`・`x1`・`d1`・`d2`・`z1`・`z2`・`z3`の順）。
fn input_from(columns: [Vec<f64>; 7]) -> IvInput {
    let [y, x1, d1, d2, z1, z2, z3] = columns;
    IvInput::from_columns(
        &y,
        &[x1],
        vec!["x1".to_string()],
        &[d1, d2],
        vec!["d1".to_string(), "d2".to_string()],
        &[z1, z2, z3],
        vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

fn columns() -> [Vec<f64>; 7] {
    [
        data::Y.to_vec(),
        data::X1.to_vec(),
        data::D1.to_vec(),
        data::D2.to_vec(),
        data::Z1.to_vec(),
        data::Z2.to_vec(),
        data::Z3.to_vec(),
    ]
}

/// d1を外生変数として扱うモデル（`x_exog=[x1, d1]`、`x_endog=[d2]`）。
fn treated() -> IvInput {
    IvInput::from_columns(
        &data::Y,
        &[data::X1.to_vec(), data::D1.to_vec()],
        vec!["x1".to_string(), "d1".to_string()],
        &[data::D2.to_vec()],
        vec!["d2".to_string()],
        &[data::Z1.to_vec(), data::Z2.to_vec(), data::Z3.to_vec()],
        vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

// ── 恒等式 ──────────────────────────────────────────────────────────

#[test]
fn is_non_negative_and_the_difference_of_the_two_hansen_statistics() {
    for (name, weight) in weights() {
        for (endog, instruments) in [(vec![0], vec![]), (vec![], vec![2]), (vec![1], vec![0])] {
            let result =
                compute_c_statistic(&input(), &target(&endog, &instruments), &weight).unwrap();
            let label = format!("{name} {endog:?} {instruments:?}");
            assert!(result.statistic >= 0.0, "{label}");
            assert!((0.0..=1.0).contains(&result.p_value), "{label}");
            assert_close(
                result.statistic,
                result.j_with_tested - result.j_without_tested,
                &label,
            );
        }
    }
}

#[test]
fn equals_the_hansen_statistic_of_the_model_with_tested_moments_when_the_other_model_is_just_identified()
 {
    // 内生変数2個・除外操作変数2個: d1を操作変数に加えない側（Z_c）は丁度識別でJ_c=0。
    let input = input_with_instruments(&[0, 1]);
    for (name, weight) in weights() {
        let result = compute_c_statistic(&input, &target(&[0], &[]), &weight).unwrap();
        assert!(result.j_without_tested.abs() < 1e-9, "{name}");
        assert_close(result.statistic, result.j_with_tested, name);
    }
}

#[test]
fn j_with_tested_equals_the_hansen_j_of_gmm_fit_when_only_instruments_are_tested() {
    // 除外操作変数だけを検定するとき、Z_fullは元のモデルの操作変数。2段階GMMのHansen Jと一致する。
    let cases: [(&str, WeightType); 3] = [
        ("classical", WeightType::Classical),
        ("robust", WeightType::Robust),
        ("hac", hac_weight()),
    ];
    for (name, weight) in cases {
        let fitted = GmmEstimator::fit(
            input(),
            GmmType::TwoStep {
                weight: weight.clone(),
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let result = compute_c_statistic(&input(), &target(&[], &[2]), &weight).unwrap();
        assert_close(
            result.j_with_tested,
            fitted.hansen_j_statistic().unwrap(),
            name,
        );
    }
}

#[test]
fn j_with_tested_equals_the_hansen_j_of_the_model_that_treats_the_tested_variable_as_exogenous() {
    // d1を外生変数として扱うモデルを、別の入力（x_exog=[x1, d1]、x_endog=[d2]）として組み立てて
    // GMMを当てはめたHansen Jと一致する。
    for (name, weight) in [("robust", WeightType::Robust), ("hac", hac_weight())] {
        let fitted = GmmEstimator::fit(
            treated(),
            GmmType::TwoStep {
                weight: weight.clone(),
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let result = compute_c_statistic(&input(), &target(&[0], &[]), &weight).unwrap();
        assert_close(
            result.j_with_tested,
            fitted.hansen_j_statistic().unwrap(),
            name,
        );
    }
}

// ── 不変性 ──────────────────────────────────────────────────────────

#[test]
fn is_invariant_to_the_scale_of_an_instrument_and_of_the_outcome() {
    let base = columns();
    let mut scaled_instrument = columns();
    scaled_instrument[5] = base[5].iter().map(|v| v * 7.0).collect();
    let mut scaled_outcome = columns();
    scaled_outcome[0] = base[0].iter().map(|v| v * 3.0).collect();
    for (name, weight) in weights() {
        for tested in [target(&[0], &[]), target(&[], &[2]), target(&[1], &[0])] {
            let reference = compute_c_statistic(&input_from(columns()), &tested, &weight)
                .unwrap()
                .statistic;
            for (what, cols) in [
                ("instrument", scaled_instrument.clone()),
                ("outcome", scaled_outcome.clone()),
            ] {
                let scaled = compute_c_statistic(&input_from(cols), &tested, &weight)
                    .unwrap()
                    .statistic;
                assert_close(scaled, reference, &format!("{name} {what} {tested:?}"));
            }
        }
    }
}

#[test]
fn is_invariant_to_the_order_of_the_rows_for_a_weight_without_time_or_group_structure() {
    let reversed: [Vec<f64>; 7] = columns().map(|c| c.into_iter().rev().collect());
    for (name, weight) in [
        ("classical", WeightType::Classical),
        ("robust", WeightType::Robust),
    ] {
        for tested in [target(&[0], &[]), target(&[], &[2]), target(&[1], &[0])] {
            let reference = compute_c_statistic(&input_from(columns()), &tested, &weight)
                .unwrap()
                .statistic;
            let permuted = compute_c_statistic(&input_from(reversed.clone()), &tested, &weight)
                .unwrap()
                .statistic;
            assert_close(permuted, reference, &format!("{name} {tested:?}"));
        }
    }
}

#[test]
fn does_not_depend_on_the_order_in_which_the_tested_endogenous_variables_enter_the_instruments() {
    // d1とd2を入れ替えた入力でd2を検定した値は、元の入力でd1を検定した値と一致する。
    let mut swapped = columns();
    swapped.swap(2, 3);
    for (name, weight) in weights() {
        let original = compute_c_statistic(&input_from(columns()), &target(&[0], &[]), &weight)
            .unwrap()
            .statistic;
        let swapped_result =
            compute_c_statistic(&input_from(swapped.clone()), &target(&[1], &[]), &weight)
                .unwrap()
                .statistic;
        assert_close(swapped_result, original, name);
    }
}

#[test]
fn does_not_depend_on_the_order_of_the_instrument_columns() {
    // 検定する操作変数と残す操作変数の位置を入れ替えても、同じ操作変数を検定した値は変わらない。
    // 外す列の選び方（位置のずれ）が結果に効くのは、外した後も過剰識別のときだけ。
    let all = [data::Z1, data::Z2, data::Z3, data::Z4];
    let build = |order: &[usize]| {
        let columns: Vec<Vec<f64>> = order.iter().map(|&p| all[p].to_vec()).collect();
        IvInput::from_columns(
            &data::Y,
            &[data::X1.to_vec()],
            vec!["x1".to_string()],
            &[data::D1.to_vec()],
            vec!["d1".to_string()],
            &columns,
            order.iter().map(|p| format!("z{}", p + 1)).collect(),
            true,
            "y".to_string(),
        )
        .unwrap()
    };
    for (name, weight) in weights() {
        // z3とz4を検定する。元の順序[z1, z2, z3, z4]では位置[2, 3]、入れ替えた順序[z4, z1, z3, z2]では[0, 2]。
        let original =
            compute_c_statistic(&build(&[0, 1, 2, 3]), &target(&[], &[2, 3]), &weight).unwrap();
        let reordered =
            compute_c_statistic(&build(&[3, 0, 2, 1]), &target(&[], &[0, 2]), &weight).unwrap();
        assert!(original.j_without_tested.abs() > 1e-6, "{name}");
        assert_close(reordered.statistic, original.statistic, name);
        assert_close(reordered.j_without_tested, original.j_without_tested, name);
    }
}

// ── 2SLSの重み ──────────────────────────────────────────────────────

#[test]
fn maps_each_cov_type_to_the_weight_of_the_same_family() {
    assert_eq!(
        weight_type_for_cov_type(&CovType::Classical),
        WeightType::Classical
    );
    for cov_type in [CovType::Hc0, CovType::Hc1, CovType::Hc2, CovType::Hc3] {
        assert_eq!(weight_type_for_cov_type(&cov_type), WeightType::Robust);
    }
    let hac = CovType::Hac {
        lags: Some(2),
        time_order: vec![1.0, 2.0, 3.0],
    };
    assert_eq!(
        weight_type_for_cov_type(&hac),
        WeightType::Hac {
            lags: Some(2),
            time_order: vec![1.0, 2.0, 3.0],
        }
    );
    let labels: Vec<String> = ["a", "b", "a"].iter().map(|s| s.to_string()).collect();
    let groups = GroupCodes::from_labels_without_keys(&labels);
    assert_eq!(
        weight_type_for_cov_type(&CovType::Cluster {
            groups: Some(groups.clone())
        }),
        WeightType::Cluster {
            groups: Some(groups)
        }
    );
}
