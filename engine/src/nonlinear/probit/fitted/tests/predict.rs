//! `predict`・`predict_new_data`・`from_parts`による復元のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MarginalEffectsAt;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::probit::ProbitEstimator;
use crate::nonlinear::probit::ProbitFitted;
use crate::nonlinear::probit::ProbitInput;
use crate::nonlinear::probit::test_support::*;
use faer::Mat;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// 切片のみモデルは全観測で`p_i=ȳ`（closed form、他のテストと同じ性質）なので、
/// `predict()`が返す予測確率が`ȳ`と一致することを検証できる
/// （`LogitEstimator`の対応するテストと同じ技法）。
#[test]
fn predict_matches_closed_form_for_intercept_only_model() {
    let estimator = ProbitEstimator::fit(
        intercept_only_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let y_bar: f64 = 4.0 / 7.0;
    let predicted = estimator.predict();
    assert_eq!(predicted.len(), 7);
    for p in predicted {
        assert!((p - y_bar).abs() < 1e-6);
    }
}

/// 多変量モデルでは`predict()`に閉じた形の解析解が無いため、`Normal::standard().cdf`
/// から直接`p_i=Φ(x_i'θ)`を計算する式で独立に再計算し、突き合わせる
/// （`LogitEstimator`の対応するテストの`logistic`をΦに置き換えたProbit版）。
#[test]
fn predict_matches_independently_recomputed_normal_cdf_of_linear_predictor() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let params = estimator.params();
    let x = estimator.input().x();
    let n = 4;
    let k = 3;
    let normal = Normal::standard();
    let predicted = estimator.predict();
    for (i, &p_i) in predicted.iter().enumerate().take(n) {
        let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        assert!((p_i - normal.cdf(z)).abs() < 1e-12);
    }
}

/// `predict_new_data`（out-of-sample）が独立に再計算した
/// `p_i=Φ(x_i'θ)`と一致すること（`LogitEstimator`の対応するテストの`logistic`を
/// Φに置き換えたProbit版）。
#[test]
fn predict_new_data_matches_independently_recomputed_normal_cdf_of_linear_predictor() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let params = estimator.params().to_vec();
    let normal = Normal::standard();
    // 学習データには無い新規のx値
    let new_x_columns = vec![vec![15.0, 25.0], vec![0.0, -3.0]];
    let predicted = estimator.predict_new_data(&new_x_columns);

    assert_eq!(predicted.len(), 2);
    for (i, &p_i) in predicted.iter().enumerate() {
        let z = params[0] + new_x_columns[0][i] * params[1] + new_x_columns[1][i] * params[2];
        assert!((p_i - normal.cdf(z)).abs() < 1e-12);
    }
}

/// `predict_new_data`が`has_intercept=false`でも正しく動作すること
/// （`LogitEstimator`の対応するテストのΦ版）。
#[test]
fn predict_new_data_without_intercept_matches_independently_recomputed_normal_cdf_of_linear_predictor()
 {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        false,
        "y".to_string(),
    )
    .unwrap();

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let params = estimator.params().to_vec();
    let normal = Normal::standard();
    // 学習データには無い新規のx値
    let new_x_columns = vec![vec![15.0, 25.0], vec![0.0, -3.0]];
    let predicted = estimator.predict_new_data(&new_x_columns);

    assert_eq!(predicted.len(), 2);
    for (i, &p_i) in predicted.iter().enumerate() {
        let z = new_x_columns[0][i] * params[0] + new_x_columns[1][i] * params[1];
        assert!((p_i - normal.cdf(z)).abs() < 1e-12);
    }
}

fn fitted_test_estimator() -> ProbitEstimator {
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let x_columns = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 1.0, 2.5],
    ];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap()
}

/// `original`と同じ設計行列を、本番の復元と同じ`from_columns`で作り直す。
fn rebuilt_input(original: &ProbitFitted) -> ProbitInput {
    let input = original.input();
    let start = usize::from(input.has_intercept());
    let n = input.nobs();
    let x_columns: Vec<Vec<f64>> = (start..input.k())
        .map(|j| (0..n).map(|i| *input.x().get(i, j)).collect())
        .collect();
    let y: Vec<f64> = (0..n).map(|i| *input.y().get(i, 0)).collect();
    ProbitInput::from_columns(
        &y,
        &x_columns,
        input.param_names()[start..].to_vec(),
        input.has_intercept(),
        input.dep_var_name().to_string(),
    )
    .unwrap()
}

fn rebuilt_from_parts(original: &ProbitFitted) -> ProbitFitted {
    ProbitFitted::from_parts(
        rebuilt_input(original),
        original.params().to_vec(),
        original.cov_params().clone(),
    )
    .unwrap()
}

#[test]
fn fitted_rebuilt_from_parts_reproduces_predict_pred_table_and_marginal_effects() {
    let estimator = fitted_test_estimator();
    let expected_effects = estimator
        .marginal_effects(MarginalEffectsAt::Overall, 0.9)
        .unwrap();
    let expected_table = estimator.pred_table(0.4);
    let expected_predict = estimator.predict();
    let original = estimator.into_fitted();
    let rebuilt = rebuilt_from_parts(&original);

    assert_eq!(rebuilt.predict(), expected_predict);
    assert_eq!(rebuilt.params(), original.params());
    let table = rebuilt.pred_table(0.4);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(table.get(i, j), expected_table.get(i, j));
        }
    }
    let effects = rebuilt
        .marginal_effects(MarginalEffectsAt::Overall, 0.9)
        .unwrap();
    assert_eq!(effects.effect(), expected_effects.effect());
    assert_eq!(effects.std_errors(), expected_effects.std_errors());
}

#[test]
fn from_parts_rejects_params_or_covariance_of_the_wrong_size() {
    let original = fitted_test_estimator().into_fitted();
    let make_input = || rebuilt_input(&original);
    let k = original.params().len();
    assert!(
        ProbitFitted::from_parts(
            make_input(),
            vec![0.0; k + 1],
            original.cov_params().clone()
        )
        .is_err()
    );
    assert!(
        ProbitFitted::from_parts(
            make_input(),
            original.params().to_vec(),
            Mat::zeros(k, k + 1)
        )
        .is_err()
    );
    assert!(
        ProbitFitted::from_parts(
            make_input(),
            original.params().to_vec(),
            Mat::zeros(k + 1, k)
        )
        .is_err()
    );
}
