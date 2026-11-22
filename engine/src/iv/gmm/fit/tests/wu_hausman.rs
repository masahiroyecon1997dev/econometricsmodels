//! Wu-Hausman内生性検定（2SLSとの一致・第一段階の残差の受け渡し・補助回帰だけの失敗）のテスト。

use crate::iv::common::IvError;
use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::common::compute_first_stage;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::WeightType;
use crate::iv::gmm::test_support::*;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::linear::ols::CovType;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

/// 過剰識別（`z1`・`z2`）・内生変数1個・定数項あり（16観測）。
fn over_identified_input() -> IvInput {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &[z1, z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// 内生変数ごとの第一段階の残差（`fit`と同じ`cov_type`・`confidence_level`で回したもの）。
fn first_stage_residual_columns(input: &IvInput, cov_type: &CovType) -> Vec<Vec<f64>> {
    compute_first_stage(input, cov_type, 0.95)
        .unwrap()
        .0
        .into_iter()
        .map(|(_, estimator)| {
            (0..estimator.residuals().nrows())
                .map(|i| *estimator.residuals().get(i, 0))
                .collect()
        })
        .collect()
}

fn as_slices(columns: &[Vec<f64>]) -> Vec<&[f64]> {
    columns.iter().map(Vec::as_slice).collect()
}

fn gmm_types() -> Vec<GmmType> {
    vec![
        GmmType::OneStep,
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        GmmType::TwoStep {
            weight: WeightType::Robust,
        },
        GmmType::Iterated {
            weight: WeightType::Robust,
            max_iter: 10,
            tol: 1e-8,
        },
    ]
}

/// 同じデータ・`cov_type`なら、GMMのWu-Hausmanは2SLSと同じ値になる（補助回帰は第一段階の残差を足した
/// 最小二乗で、GMMの点推定・重みを使わない）。GMMの方式（1段階・2段階・反復と重みの種類）によらない。
#[test]
fn gmm_matches_two_sls_for_every_gmm_type_and_cov_type() {
    for cov_type in [
        CovType::Classical,
        CovType::Hc0,
        CovType::Hc1,
        CovType::Hc2,
        CovType::Hc3,
    ] {
        let two_sls =
            TwoSlsEstimator::fit(over_identified_input(), cov_type.clone(), 0.95).unwrap();
        let expected = (
            two_sls.wu_hausman_statistic().unwrap(),
            two_sls.wu_hausman_p_value().unwrap(),
            two_sls.wu_hausman_df().unwrap(),
        );
        for gmm_type in gmm_types() {
            let label = format!("{cov_type:?}/{gmm_type:?}");
            let gmm = GmmEstimator::fit(
                over_identified_input(),
                gmm_type,
                true,
                cov_type.clone(),
                0.95,
            )
            .unwrap();
            let stat = gmm.wu_hausman_statistic().unwrap();
            let p_value = gmm.wu_hausman_p_value().unwrap();
            assert!(
                (stat - expected.0).abs() <= 1e-10 * expected.0.abs(),
                "{label}: stat={stat}, expected={}",
                expected.0
            );
            assert!(
                (p_value - expected.1).abs() <= 1e-10,
                "{label}: p_value={p_value}, expected={}",
                expected.1
            );
            assert_eq!(gmm.wu_hausman_df().unwrap(), expected.2, "{label}");
        }
    }
}

/// 呼び出し側が渡した第一段階の残差でも、`fit()`が自分で回した場合と同じ結果になり、点推定・標準誤差・
/// Hansen Jには影響しない。
#[test]
fn passing_first_stage_residuals_gives_the_same_result_as_fit() {
    let cov_type = CovType::Hc1;
    let gmm_type = GmmType::TwoStep {
        weight: WeightType::Robust,
    };
    let input = over_identified_input();
    let residuals = first_stage_residual_columns(&input, &cov_type);
    let given = GmmEstimator::fit_with_first_stage_residuals(
        input,
        gmm_type.clone(),
        true,
        cov_type.clone(),
        0.95,
        &as_slices(&residuals),
    )
    .unwrap();
    let computed =
        GmmEstimator::fit(over_identified_input(), gmm_type, true, cov_type, 0.95).unwrap();

    assert_eq!(
        given.wu_hausman_statistic(),
        computed.wu_hausman_statistic()
    );
    assert_eq!(given.wu_hausman_p_value(), computed.wu_hausman_p_value());
    assert_eq!(given.wu_hausman_df(), computed.wu_hausman_df());
    assert_eq!(given.params(), computed.params());
    assert_eq!(given.std_errors(), computed.std_errors());
    assert_eq!(given.hansen_j_statistic(), computed.hansen_j_statistic());
}

/// 内生変数の数・観測数に合わない残差は、呼び出し側の誤り（`INTERNAL_ERROR`）で、推定の前に拒否する。
#[test]
fn mismatched_first_stage_residuals_are_rejected_as_an_internal_error() {
    let n = over_identified_input().nobs();
    for residuals in [
        vec![],
        vec![vec![0.0; n], vec![0.0; n]],
        vec![vec![0.0; n - 1]],
    ] {
        let err = GmmEstimator::fit_with_first_stage_residuals(
            over_identified_input(),
            GmmType::OneStep,
            true,
            CovType::Classical,
            0.95,
            &as_slices(&residuals),
        )
        .unwrap_err();
        assert!(
            matches!(err, IvError::Common(CommonError::Internal { .. })),
            "{err:?}"
        );
    }
}

/// `cov_type=Cluster`で構造式は`G > q`でも、補助回帰は残差の列の分だけ傾きが増える（`q_aug = q + k_endog`）ため
/// `G <= q_aug`になる。この診断だけ`None`にして、`fit()`は成功する。
#[test]
fn wu_hausman_is_none_when_cluster_count_is_at_most_the_augmented_slopes() {
    // 構造式`[const, endog1]`は`q=1`、補助回帰は`q_aug=2`。`G=2`は`q < G <= q_aug`。
    let groups: Vec<String> = (0..16).map(|i| format!("g{}", i % 2)).collect();
    let estimator = GmmEstimator::fit(
        over_identified_input(),
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    )
    .unwrap();

    assert_eq!(estimator.wu_hausman_statistic(), None);
    assert_eq!(estimator.wu_hausman_p_value(), None);
    assert_eq!(estimator.wu_hausman_df(), None);
}

/// `y = 1 + 2*x_endog + 3*v`（`v`は第一段階の残差）は、構造残差も第一段階も完全フィットではないが、
/// 補助回帰`y ~ const + x_endog + v`は厳密に成り立つ。`fit()`は成功し、この診断だけ`None`になる。
#[test]
fn wu_hausman_is_none_when_the_augmented_regression_is_exact() {
    let base = just_identified_input();
    let v = first_stage_residual_columns(&base, &CovType::Classical).remove(0);
    let x_endog = base.x_endog().col_as_slice(0).to_vec();
    let z = base.instruments().col_as_slice(0).to_vec();
    let y: Vec<f64> = (0..x_endog.len())
        .map(|i| 1.0 + 2.0 * x_endog[i] + 3.0 * v[i])
        .collect();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &[x_endog],
        vec!["x_endog".to_string()],
        &[z],
        vec!["z".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator =
        GmmEstimator::fit(input, GmmType::OneStep, true, CovType::Classical, 0.95).unwrap();

    assert!((*estimator.params().get(1, 0) - 2.0).abs() < 1e-8);
    assert_eq!(estimator.wu_hausman_statistic(), None);
    assert_eq!(estimator.wu_hausman_p_value(), None);
}

/// 内生変数2個・外生変数1個（定数項あり）・操作変数3個（30観測）。`Classical`・`Hc1`・`Hac`・`Cluster`の
/// 各`cov_type`で、GMMのWu-Hausmanが2SLSと一致し、自由度が`(2, n - k - 2)`（クラスターは`G - 1`）になる。
/// 第一段階の残差の列の数と並びが合っていることを固定する。
#[test]
fn gmm_matches_two_sls_with_two_endogenous_variables() {
    let n = 30;
    let x1: Vec<f64> = (0..n)
        .map(|i| ((i as f64) * 0.7).sin() * 3.0 + (i as f64) * 0.1)
        .collect();
    let z1: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.9).sin() * 5.0).collect();
    let z2: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.3).cos() * 4.0).collect();
    let z3: Vec<f64> = (0..n)
        .map(|i| ((i as f64) * 0.6).sin() * 3.0 + (i as f64) * 0.2)
        .collect();
    let endog1: Vec<f64> = (0..n)
        .map(|i| z1[i] + 0.5 * z2[i] + 0.2 * x1[i] + ((i as f64) * 2.1).sin())
        .collect();
    let endog2: Vec<f64> = (0..n)
        .map(|i| 0.7 * z2[i] + z3[i] - 0.3 * x1[i] + ((i as f64) * 1.7).cos())
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * endog1[i] - 1.5 * endog2[i] + 0.5 * x1[i] + ((i as f64) * 0.83).sin() * 2.0
        })
        .collect();
    let build = || {
        IvInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            &[endog1.clone(), endog2.clone()],
            vec!["endog1".to_string(), "endog2".to_string()],
            &[z1.clone(), z2.clone(), z3.clone()],
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };
    let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 6)).collect();
    let cov_types = [
        CovType::Classical,
        CovType::Hc1,
        CovType::Hac {
            lags: Some(2),
            time_order: (0..n).map(|i| i as f64).collect(),
        },
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
    ];
    for cov_type in cov_types {
        let label = format!("{cov_type:?}");
        let two_sls = TwoSlsEstimator::fit(build(), cov_type.clone(), 0.95).unwrap();
        let gmm = GmmEstimator::fit(
            build(),
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            cov_type,
            0.95,
        )
        .unwrap();

        let expected = two_sls.wu_hausman_statistic().unwrap();
        let stat = gmm.wu_hausman_statistic().unwrap();
        assert!(
            (stat - expected).abs() <= 1e-10 * expected.abs(),
            "{label}: stat={stat}, expected={expected}"
        );
        assert_eq!(gmm.wu_hausman_df(), two_sls.wu_hausman_df(), "{label}");
        assert_eq!(gmm.wu_hausman_df().unwrap().0, 2, "{label}");
        assert_eq!(
            gmm.wu_hausman_p_value(),
            two_sls.wu_hausman_p_value(),
            "{label}"
        );
    }
}
