//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use super::exact_fit_input;
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::shared::error::CommonError;

#[test]
fn fit_returns_perfect_fit_for_exact_fit_data() {
    // 残差が全て0に近いと`sigma^2`が0で標準誤差・F検定が定義できないため、共分散を計算する前に止める。
    let result = OlsEstimator::fit(exact_fit_input(), CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::PerfectFit)
    );
}

#[test]
fn fit_returns_outcome_constant_when_the_outcome_is_the_same_in_every_row() {
    let input = OlsInput::from_columns(
        &[3.0; 5],
        &[vec![0.0, 1.0, 2.0, 3.0, 5.0]],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::OutcomeConstant)
    );
}

#[test]
fn fit_treats_an_outcome_that_differs_by_one_ulp_as_constant() {
    // 計算で作った定数（`0.1 + 0.2`と`0.3`）は1 ulp違うが、定数として扱う。
    let y = [0.1 + 0.2, 0.3, 0.3, 0.1 + 0.2, 0.3];
    let input = OlsInput::from_columns(
        &y,
        &[vec![0.0, 1.0, 2.0, 3.0, 5.0]],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::OutcomeConstant)
    );
}

#[test]
fn fit_returns_no_regressors_error_when_k_is_zero() {
    // include_intercept=falseかつx_columns=[]の病的な入力。
    // Logit/Probitと同じ`CommonError::NoRegressors`で早期リジェクトされる。
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = OlsInput::from_columns(&y, &[], vec![], false, "y".to_string()).unwrap();

    let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::NoRegressors { n: 5 })
    );
}

#[test]
fn fit_returns_insufficient_observations_when_n_le_k() {
    let y = vec![1.0, 2.0];
    let x_columns = vec![vec![1.0, 2.0]];
    // include_intercept=trueでk=2、n=2 (n<=k)
    let input = OlsInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
    );
}

#[test]
fn fit_returns_singular_matrix_for_perfectly_collinear_columns() {
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0]; // x2 = 2 * x1 (完全な多重共線性)
    let input = OlsInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::SingularMatrix {
            columns: vec!["x1".to_string(), "x2".to_string()]
        }
    );
}

/// 列の単位を変えても（`x1`は1e6倍、`x2`は1e-3倍）、F統計量・p値は単位を揃えたときと
/// 一致する（傾き係数間の分散の比は1e18を超えるが、悪条件の判定は相関スケール化した行列で
/// 行う。`x1`・`x2`・`x3`は線形従属ではない）。
#[test]
fn fit_f_test_is_invariant_to_a_change_of_units() {
    let n = 10;
    let unit_x1: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let unit_x2: Vec<f64> = (1..=n).map(|i| (i as f64).powi(2)).collect();
    let x3: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let noise = if i % 2 == 0 { 1e3 } else { -1e3 };
            1.0 + 2.0 * 1e6 * unit_x1[i] + 3.0 * 1e-3 * unit_x2[i] + 0.5 * x3[i] + noise
        })
        .collect();
    let names = || vec!["x1".to_string(), "x2".to_string(), "x3".to_string()];
    let fit = |x1_scale: f64, x2_scale: f64| {
        let x1: Vec<f64> = unit_x1.iter().map(|v| v * x1_scale).collect();
        let x2: Vec<f64> = unit_x2.iter().map(|v| v * x2_scale).collect();
        let input =
            OlsInput::from_columns(&y, &[x1, x2, x3.clone()], names(), true, "y".to_string())
                .unwrap();
        OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap()
    };

    let scaled = fit(1e6, 1e-3);
    let unscaled = fit(1.0, 1.0);

    let (f_scaled, f_unscaled) = (
        scaled.f_statistic().unwrap(),
        unscaled.f_statistic().unwrap(),
    );
    assert!(f_scaled.is_finite() && f_scaled > 0.0);
    assert!((f_scaled / f_unscaled - 1.0).abs() < 1e-9);
    let (p_scaled, p_unscaled) = (scaled.f_p_value().unwrap(), unscaled.f_p_value().unwrap());
    assert!((p_scaled - p_unscaled).abs() <= 1e-9 * p_unscaled);
}

#[test]
fn fit_returns_invalid_confidence_level_when_out_of_range() {
    let y = vec![1.0, 2.0, 3.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0]];
    let input = OlsInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = OlsEstimator::fit(input, CovType::Classical, 1.5);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5
        })
    );
}

/// `confidence_level`の境界値（0.0・1.0ちょうど）が範囲外として拒否されることを確認する
/// （`!(level > 0.0 && level < 1.0)`という判定式の境界そのものの検証）。
#[test]
fn fit_returns_invalid_confidence_level_at_exact_boundaries() {
    for level in [0.0, 1.0, -0.1] {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, level);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: level
            }),
            "level={level}"
        );
    }
}
