//! `fit`の基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）のテスト。

use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::WeightType;
use crate::iv::gmm::test_support::*;
use crate::linear::ols::CovType;
use crate::linear::ols::CovType as OlsCovType;
use faer::Mat;

/// `dep_var_name()`/`weight_type()`/`gmm_tol()`/`nobs()`/`k()`が、
/// `fit()`に渡した値・入力データと整合することを確認する（`two_sls/fit/tests/basic.rs`の
/// `fit_succeeds_when_over_identified`と同じ「基本メタデータgetter群」の検証方針）。
#[test]
fn fit_exposes_basic_metadata_getters() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let input = IvInput::from_columns(
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
    .unwrap();

    let estimator = GmmEstimator::fit(
        input,
        GmmType::Iterated {
            weight: WeightType::Robust,
            max_iter: 5,
            tol: 1e-6,
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    assert_eq!(estimator.dep_var_name(), "y");
    assert_eq!(estimator.weight_type(), Some(&WeightType::Robust));
    assert_eq!(
        estimator.gmm_type(),
        &GmmType::Iterated {
            weight: WeightType::Robust,
            max_iter: 5,
            tol: 1e-6,
        }
    );
    assert_eq!(estimator.nobs(), n);
    assert_eq!(estimator.k(), 2);
}

/// `residuals()`が最終推定値に基づく`y - Xβ̂`であることを確認する。
#[test]
fn residuals_are_consistent_with_final_params() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let input = IvInput::from_columns(
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
    .unwrap();
    let estimator = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Robust,
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
    for (i, &y_i) in y.iter().enumerate() {
        let expected = y_i
            - (*estimator.params().get(0, 0)) * (*x.get(i, 0))
            - (*estimator.params().get(1, 0)) * (*x.get(i, 1));
        assert!(
            (*estimator.residuals().get(i, 0) - expected).abs() < 1e-8,
            "row {i}"
        );
    }
}

/// 2SLSと同じデータ（`x_exog`に実変数を含む、過剰識別）で、`WeightType::Classical`の
/// GMM点推定が`TwoSlsEstimator`の点推定と厳密に一致することを確認する（`gmm.rs`の
/// モジュールdoc「2SLSとの共通化の判断」の根拠）。
#[test]
fn fit_matches_two_sls_point_estimate_when_weight_type_is_classical() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
    let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
    let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];

    let build_input = || {
        IvInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };

    let gmm = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();
    let two_sls =
        crate::iv::two_sls::TwoSlsEstimator::fit(build_input(), OlsCovType::Classical, 0.95)
            .unwrap();

    assert_eq!(gmm.param_names(), two_sls.param_names());
    for j in 0..gmm.params().nrows() {
        assert!(
            (*gmm.params().get(j, 0) - *two_sls.params().get(j, 0)).abs() < 1e-8,
            "param {j}: gmm={}, two_sls={}",
            *gmm.params().get(j, 0),
            *two_sls.params().get(j, 0)
        );
    }
}

#[test]
fn fit_matches_two_sls_when_just_identified() {
    let gmm = GmmEstimator::fit(
        just_identified_input(),
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();
    let two_sls =
        crate::iv::two_sls::TwoSlsEstimator::fit(just_identified_input(), CovType::Classical, 0.95)
            .unwrap();

    assert_eq!(gmm.param_names(), ["const", "x_endog"]);
    for j in 0..2 {
        assert!((*gmm.params().get(j, 0) - *two_sls.params().get(j, 0)).abs() < 1e-8);
    }
}
