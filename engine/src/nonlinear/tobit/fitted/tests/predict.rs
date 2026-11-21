//! `predict`・`predict_new_data`・`from_parts`による復元のテスト。

use super::expected_observed_closed_form;
use super::noisy_censored_input;
use super::prob_uncensored_closed_form;
use super::right_censored_regression_input;
use super::two_sided_censored_regression_input;
use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MarginalEffectsAt;
use crate::nonlinear::common::MleError;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::tobit::TobitEstimator;
use crate::nonlinear::tobit::TobitFitted;
use crate::nonlinear::tobit::TobitInput;
use crate::nonlinear::tobit::target::MIN_UNCENSORED_WIDTH_RATIO;
use crate::nonlinear::tobit::target::MarginalEffectsTarget;
use crate::nonlinear::tobit::test_support::*;
use faer::Mat;

/// ほぼ全観測が両端に打ち切られた当てはめ（`σ̂`が区間幅の100倍超）。
fn narrow_interval_estimator(n_interior: usize) -> TobitEstimator {
    let n = 2000usize;
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 7919) % 1009) as f64 / 1009.0)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            if i < n_interior {
                0.5
            } else if i % 2 == 0 {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    let input = TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        Some(1.0),
    )
    .unwrap();
    TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 200,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap()
}

/// `predict`/`predict_new_data`が`ExpectedUncensored`で独立の閉形式と一致し、
/// 学習データへの`predict`と同じ`x`での`predict_new_data`が一致する。
#[test]
fn predict_and_predict_new_data_expected_uncensored_match_closed_form() {
    let inputs = [
        noisy_censored_input(Some(0.0), None),
        noisy_censored_input(None, Some(6.0)),
        noisy_censored_input(Some(0.0), Some(6.0)),
    ];
    for input in inputs {
        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let predicted = estimator
            .predict(MarginalEffectsTarget::ExpectedUncensored)
            .unwrap();
        for (i, &p) in predicted.iter().enumerate() {
            let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
            let expected = expected_uncensored_closed_form(mu, sigma, lower, upper);
            assert!(
                (p - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                "lower={lower:?}, upper={upper:?}, i={i}: actual={p}, expected={expected}"
            );
        }

        let x1: Vec<f64> = (0..x.nrows()).map(|i| *x.get(i, 1)).collect();
        let predicted_new = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedUncensored, &[x1])
            .unwrap();
        assert_eq!(predicted_new.len(), predicted.len());
        for (a, b) in predicted_new.iter().zip(&predicted) {
            assert!((a - b).abs() < 1e-12, "predict_new_data={a}, predict={b}");
        }
    }
}

/// 狭い区間の当てはめ（`σ̂≈160`、`δ≈6e-3`）では`ExpectedUncensored`の`predict`・
/// `predict_new_data`・`marginal_effects`が`UncensoredIntervalTooNarrow`になり、他の3targetは
/// 通常どおり計算できる（エラーは`expected_uncensored`固有）。区間がちょうど十分な当てはめ
/// （`σ̂≈40`、`δ≈0.025`）では`ExpectedUncensored`も計算できる。
#[test]
fn expected_uncensored_errors_only_when_the_fitted_interval_is_too_narrow() {
    let narrow = narrow_interval_estimator(5);
    assert!(
        1.0 / narrow.sigma() < MIN_UNCENSORED_WIDTH_RATIO,
        "fixture is not narrow"
    );
    let new_x = vec![vec![0.2, 0.8]];
    assert!(matches!(
        narrow.predict(MarginalEffectsTarget::ExpectedUncensored),
        Err(MleError::UncensoredIntervalTooNarrow { .. })
    ));
    assert!(matches!(
        narrow.predict_new_data(MarginalEffectsTarget::ExpectedUncensored, &new_x),
        Err(MleError::UncensoredIntervalTooNarrow { .. })
    ));
    assert!(matches!(
        narrow.marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedUncensored,
            0.95
        ),
        Err(MleError::UncensoredIntervalTooNarrow { .. })
    ));
    for target in [
        MarginalEffectsTarget::ExpectedLatent,
        MarginalEffectsTarget::ExpectedObserved,
        MarginalEffectsTarget::ProbUncensored,
    ] {
        let predicted = narrow.predict(target).unwrap();
        assert!(predicted.iter().all(|v| v.is_finite()), "{target:?}");
        assert!(
            narrow.predict_new_data(target, &new_x).is_ok(),
            "{target:?}"
        );
        assert!(
            narrow
                .marginal_effects(MarginalEffectsAt::Overall, target, 0.95)
                .is_ok(),
            "{target:?}"
        );
    }

    let wide_enough = narrow_interval_estimator(20);
    assert!(1.0 / wide_enough.sigma() >= MIN_UNCENSORED_WIDTH_RATIO);
    let predicted = wide_enough
        .predict(MarginalEffectsTarget::ExpectedUncensored)
        .unwrap();
    assert!(
        predicted
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    );
}

/// `predict(ExpectedLatent)`は`E[y*|x]=x'β`という自明な式（打ち切りに依存しない）
/// なので、`x.get(i,·)`から直接計算した線形予測子と厳密に一致するはず。
#[test]
fn fit_predict_expected_latent_equals_linear_predictor() {
    let estimator = TobitEstimator::fit(
        censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    let x = estimator.input().x();
    let n = x.nrows();
    let k = 2;
    let params = estimator.params();

    let predicted = estimator
        .predict(MarginalEffectsTarget::ExpectedLatent)
        .unwrap();
    assert_eq!(predicted.len(), n);
    for (i, &predicted_i) in predicted.iter().enumerate() {
        let expected: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        assert!((predicted_i - expected).abs() < 1e-12);
    }
}

/// `predict(ExpectedObserved)`・`predict(ProbUncensored)`を、`marginal_effects`の
/// テストで使った閉形式（`expected_observed_closed_form`/`prob_uncensored_closed_form`、
/// `target_w_and_s`/`predicted_value`の実装コードとは独立の`(lower,upper)`ごとの
/// `match`分岐）と観測ごとに突き合わせる（左打ち切りのみ）。
#[test]
fn fit_predict_matches_independent_recomputation_for_left_only_censoring() {
    let estimator = TobitEstimator::fit(
        censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    let x = estimator.input().x();
    let n = x.nrows();
    let k = 2;
    let params = estimator.params();
    let sigma = estimator.sigma();
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    assert_eq!(lower, Some(0.0));
    assert_eq!(upper, None);

    let predicted_observed = estimator
        .predict(MarginalEffectsTarget::ExpectedObserved)
        .unwrap();
    let predicted_prob = estimator
        .predict(MarginalEffectsTarget::ProbUncensored)
        .unwrap();
    for i in 0..n {
        let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
        let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
        assert!(
            (predicted_observed[i] - expected_observed).abs() < 1e-9,
            "row={i}, actual={}, expected={expected_observed}",
            predicted_observed[i]
        );
        assert!(
            (predicted_prob[i] - expected_prob).abs() < 1e-9,
            "row={i}, actual={}, expected={expected_prob}",
            predicted_prob[i]
        );
    }
}

/// 右打ち切りのみのデータでも`predict(ExpectedObserved)`・`predict(ProbUncensored)`が
/// 独立再計算と一致することを確認する。
#[test]
fn fit_predict_matches_independent_recomputation_for_right_only_censoring() {
    let estimator = TobitEstimator::fit(
        right_censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    let x = estimator.input().x();
    let n = x.nrows();
    let k = 2;
    let params = estimator.params();
    let sigma = estimator.sigma();
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    assert_eq!(lower, None);
    assert_eq!(upper, Some(6.0));

    let predicted_observed = estimator
        .predict(MarginalEffectsTarget::ExpectedObserved)
        .unwrap();
    let predicted_prob = estimator
        .predict(MarginalEffectsTarget::ProbUncensored)
        .unwrap();
    for i in 0..n {
        let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
        let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
        assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
        assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
    }
}

/// 両側打ち切りのデータでも`predict(ExpectedObserved)`・`predict(ProbUncensored)`が
/// 独立再計算と一致することを確認する。
#[test]
fn fit_predict_matches_independent_recomputation_for_two_sided_censoring() {
    let estimator = TobitEstimator::fit(
        two_sided_censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    let x = estimator.input().x();
    let n = x.nrows();
    let k = 2;
    let params = estimator.params();
    let sigma = estimator.sigma();
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    assert_eq!(lower, Some(0.0));
    assert_eq!(upper, Some(6.0));

    let predicted_observed = estimator
        .predict(MarginalEffectsTarget::ExpectedObserved)
        .unwrap();
    let predicted_prob = estimator
        .predict(MarginalEffectsTarget::ProbUncensored)
        .unwrap();
    for i in 0..n {
        let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
        let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
        assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
        assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
    }
}

/// 上記の`predict`テストは全て単一の傾き係数（`k=2`、切片+`x1`）のフィクスチャのみを
/// 使っており、`(0..k).map(|j| *x.get(i,j)*self.params[j])`という列インデックス計算の
/// 配線ミス（列の取り違え等）を構造的に検出できない（`multivariate_censored_input`の
/// docコメント「配線ミスは単一変数データでは検出できない」と同じ理由）。
/// 傾き係数2個（`x1`・`x2`）の多変量データセットで
/// `expected_observed_closed_form`/`prob_uncensored_closed_form`との独立再計算を検証する。
#[test]
fn fit_predict_matches_independent_recomputation_for_multivariate_design() {
    let estimator = TobitEstimator::fit(
        multivariate_censored_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    let x = estimator.input().x();
    let n = x.nrows();
    let k = 3;
    let params = estimator.params();
    let sigma = estimator.sigma();
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();

    let predicted_latent = estimator
        .predict(MarginalEffectsTarget::ExpectedLatent)
        .unwrap();
    let predicted_observed = estimator
        .predict(MarginalEffectsTarget::ExpectedObserved)
        .unwrap();
    let predicted_prob = estimator
        .predict(MarginalEffectsTarget::ProbUncensored)
        .unwrap();
    for i in 0..n {
        let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        assert!((predicted_latent[i] - mu).abs() < 1e-12);
        let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
        let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
        assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
        assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
    }
}

/// `predict_new_data`（out-of-sample）が独立に再計算した
/// 値と一致すること。`fit_predict_matches_independent_recomputation_for_
/// multivariate_design`と同じモデルを使い、学習データとは異なる新規のx値で
/// 3つの`target`すべてを検証する。
#[test]
fn predict_new_data_matches_independent_recomputation_for_multivariate_design() {
    let estimator = TobitEstimator::fit(
        multivariate_censored_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    let params = estimator.params().to_vec();
    let sigma = estimator.sigma();
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();

    // 学習データには無い新規のx値
    let new_x_columns = vec![vec![15.0, 25.0], vec![3.0, -1.0]];
    let predicted_latent = estimator
        .predict_new_data(MarginalEffectsTarget::ExpectedLatent, &new_x_columns)
        .unwrap();
    let predicted_observed = estimator
        .predict_new_data(MarginalEffectsTarget::ExpectedObserved, &new_x_columns)
        .unwrap();
    let predicted_prob = estimator
        .predict_new_data(MarginalEffectsTarget::ProbUncensored, &new_x_columns)
        .unwrap();

    assert_eq!(predicted_latent.len(), 2);
    for i in 0..2 {
        let mu = params[0] + new_x_columns[0][i] * params[1] + new_x_columns[1][i] * params[2];
        assert!((predicted_latent[i] - mu).abs() < 1e-12);
        let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
        let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
        assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
        assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
    }
}

/// `predict_new_data`が`has_intercept=false`でも3つの`target`すべてで
/// 正しく動作すること（Logit/Probitの対応するテストと同じ理由の対称性確保）。
#[test]
fn predict_new_data_without_intercept_matches_independent_recomputation() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0];
    let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
    let input = TobitInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        false,
        "y".to_string(),
        Some(0.0),
        None,
    )
    .unwrap();

    let estimator = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    let params = estimator.params().to_vec();
    let sigma = estimator.sigma();
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();

    let new_x_columns = vec![vec![15.0, 25.0], vec![3.0, -1.0]];
    let predicted_latent = estimator
        .predict_new_data(MarginalEffectsTarget::ExpectedLatent, &new_x_columns)
        .unwrap();
    let predicted_observed = estimator
        .predict_new_data(MarginalEffectsTarget::ExpectedObserved, &new_x_columns)
        .unwrap();
    let predicted_prob = estimator
        .predict_new_data(MarginalEffectsTarget::ProbUncensored, &new_x_columns)
        .unwrap();

    assert_eq!(predicted_observed.len(), 2);
    for i in 0..2 {
        let mu = new_x_columns[0][i] * params[0] + new_x_columns[1][i] * params[1];
        assert!((predicted_latent[i] - mu).abs() < 1e-12);
        let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
        let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
        assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
        assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
    }
}

fn fitted_test_estimator() -> TobitEstimator {
    TobitEstimator::fit(
        censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap()
}

/// `original`と同じ設計行列を、本番の復元と同じ`from_columns`で作り直す。
fn rebuilt_input(original: &TobitFitted) -> TobitInput {
    let input = original.input();
    let start = usize::from(input.has_intercept());
    let n = input.nobs();
    let x_columns: Vec<Vec<f64>> = (start..input.k())
        .map(|j| (0..n).map(|i| *input.x().get(i, j)).collect())
        .collect();
    let y: Vec<f64> = (0..n).map(|i| *input.y().get(i, 0)).collect();
    TobitInput::from_columns(
        &y,
        &x_columns,
        input.param_names()[start..].to_vec(),
        input.has_intercept(),
        input.dep_var_name().to_string(),
        input.lower(),
        input.upper(),
    )
    .unwrap()
}

#[test]
fn fitted_rebuilt_from_parts_reproduces_predict_marginal_effects_and_censoring_check() {
    let estimator = fitted_test_estimator();
    let targets = [
        MarginalEffectsTarget::ExpectedLatent,
        MarginalEffectsTarget::ExpectedObserved,
    ];
    let expected_predict: Vec<_> = targets
        .iter()
        .map(|&t| estimator.predict(t).unwrap())
        .collect();
    let expected_effects = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            0.9,
        )
        .unwrap();
    let expected_check = estimator.censoring_fit_check();
    let original = estimator.into_fitted();
    let rebuilt = TobitFitted::from_parts(
        rebuilt_input(&original),
        original.params().to_vec(),
        original.sigma(),
        original.cov_params().clone(),
    )
    .unwrap();

    for (target, expected) in targets.iter().zip(&expected_predict) {
        assert_eq!(&rebuilt.predict(*target).unwrap(), expected);
    }
    let effects = rebuilt
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            0.9,
        )
        .unwrap();
    assert_eq!(effects.effect(), expected_effects.effect());
    assert_eq!(effects.std_errors(), expected_effects.std_errors());
    let check = rebuilt.censoring_fit_check();
    assert_eq!(
        check.uncensored().observed_rate(),
        expected_check.uncensored().observed_rate()
    );
    assert_eq!(
        check.uncensored().model_implied_rate(),
        expected_check.uncensored().model_implied_rate()
    );
}

#[test]
fn from_parts_rejects_parts_of_the_wrong_size_or_a_bad_sigma() {
    let original = fitted_test_estimator().into_fitted();
    let k = original.params().len();
    let cov = || original.cov_params().clone();
    let make = |params: Vec<f64>, sigma: f64, cov: Mat<f64>| {
        TobitFitted::from_parts(rebuilt_input(&original), params, sigma, cov)
    };
    let params = || original.params().to_vec();
    let sigma = original.sigma();
    assert!(make(params(), sigma, cov()).is_ok());
    assert!(make(vec![0.0; k + 1], sigma, cov()).is_err());
    // `cov_params`は`(k+1)×(k+1)`（`σ`の行・列を含む）
    assert!(make(params(), sigma, Mat::zeros(k, k)).is_err());
    assert!(make(params(), sigma, Mat::zeros(k + 1, k + 2)).is_err());
    assert!(make(params(), 0.0, cov()).is_err());
    assert!(make(params(), -1.0, cov()).is_err());
    assert!(make(params(), f64::NAN, cov()).is_err());
}
