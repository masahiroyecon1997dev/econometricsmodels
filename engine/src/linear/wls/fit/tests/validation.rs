//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use crate::linear::common::LeastSquaresError;
use crate::linear::ols::CovType;
use crate::linear::wls::WlsEstimator;
use crate::shared::error::CommonError;

#[test]
fn fit_returns_perfect_fit_for_exact_data_under_any_weights() {
    // y = 1 + 2*x のノイズなしの厳密解は、重みを変えても残差が0で標準誤差が定義できない。
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let x_columns = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
    let weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];

    let result = WlsEstimator::fit(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::PerfectFit)
    );
}

#[test]
fn fit_returns_outcome_constant_for_a_constant_outcome_under_any_weights() {
    // 重み付き平均で中心化した総平方和も0になる。
    let result = WlsEstimator::fit(
        &[4.0; 5],
        &[vec![0.0, 1.0, 2.0, 3.0, 5.0]],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        &[1.0, 2.0, 0.5, 3.0, 1.5],
        CovType::Classical,
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::OutcomeConstant)
    );
}

#[test]
fn fit_propagates_non_positive_weight_error() {
    let y = vec![1.0, 2.0, 3.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0]];
    let weights = vec![1.0, 0.0, 1.0];

    let result = WlsEstimator::fit(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::NonPositiveWeight {
            row: 1,
            weight: 0.0
        }
    );
}

#[test]
fn fit_propagates_no_regressors_error_when_k_is_zero() {
    // `WlsEstimator::fit`は常に（ゲート付きの）`OlsEstimator::fit`に委譲するため、
    // k=0拒否もそのまま伝播する（WLSはk=0になりえない呼び出し方をする対象ではなく、
    // OLSと同じくPython向け公開APIとしてk=0を拒否すべき対象）。
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0; 5];

    let result = WlsEstimator::fit(
        &y,
        &[],
        vec![],
        false,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(crate::shared::error::CommonError::NoRegressors { n: 5 })
    );
}
