//! 打ち切りの適合度確認（`censoring_fit_check`）のテスト。

use super::prob_uncensored_closed_form;
use super::right_censored_regression_input;
use super::two_sided_censored_regression_input;
use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::tobit::TobitEstimator;
use crate::nonlinear::tobit::test_support::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// `censoring_fit_check`（左打ち切りのみ）: `lower`カテゴリが`Some`・`upper`カテゴリが
/// `None`であること、実測比率（`y==0.0`の観測数/n）・モデル含意の平均確率
/// （`Φ((lower-μᵢ)/σ)`の平均、`predicted_value`とは独立に`Normal::cdf`を直接呼んで
/// 計算）の両方を独立再計算と突き合わせる。3カテゴリの`model_implied_rate`が合計1に
/// なる恒等式も確認する。
#[test]
fn fit_censoring_fit_check_matches_independent_recomputation_for_left_only_censoring() {
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
    let normal = Normal::new(0.0, 1.0).unwrap();

    let check = estimator.censoring_fit_check();
    assert!(check.upper().is_none());
    let lower_check = check.lower().unwrap();

    // censored_regression_input: y=[0,0,1.15,2.9,5.2,6.85,9.1,10.95]、lower=0.0で
    // 打ち切られているのはx=1,2の2件（実測打ち切り率2/8=0.25）
    assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
    assert!((check.uncensored().observed_rate() - 0.75).abs() < 1e-12);

    let expected_lower_rate: f64 = (0..n)
        .map(|i| {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            normal.cdf((0.0 - mu) / sigma)
        })
        .sum::<f64>()
        / (n as f64);
    assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);

    let expected_uncensored_rate: f64 = (0..n)
        .map(|i| {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            prob_uncensored_closed_form(mu, sigma, Some(0.0), None)
        })
        .sum::<f64>()
        / (n as f64);
    assert!((check.uncensored().model_implied_rate() - expected_uncensored_rate).abs() < 1e-9);

    // model_implied_rateは合計1になるはず（lower + uncensored、upperはこのケースでは無い）
    assert!(
        (lower_check.model_implied_rate() + check.uncensored().model_implied_rate() - 1.0).abs()
            < 1e-9
    );
}

/// `censoring_fit_check`（右打ち切りのみ）: `lower`が`None`・`upper`が`Some`になること、
/// 実測比率・モデル含意の平均確率を独立再計算と突き合わせる。
#[test]
fn fit_censoring_fit_check_matches_independent_recomputation_for_right_only_censoring() {
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
    let normal = Normal::new(0.0, 1.0).unwrap();

    let check = estimator.censoring_fit_check();
    assert!(check.lower().is_none());
    let upper_check = check.upper().unwrap();

    // right_censored_regression_input: y=[-2.85,-0.95,1.15,2.9,5.2,6.0,6.0,6.0]、
    // upper=6.0で打ち切られているのはx=6,7,8の3件（実測打ち切り率3/8=0.375）
    assert!((upper_check.observed_rate() - 0.375).abs() < 1e-12);
    assert!((check.uncensored().observed_rate() - 0.625).abs() < 1e-12);

    let expected_upper_rate: f64 = (0..n)
        .map(|i| {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            1.0 - normal.cdf((6.0 - mu) / sigma)
        })
        .sum::<f64>()
        / (n as f64);
    assert!((upper_check.model_implied_rate() - expected_upper_rate).abs() < 1e-9);
}

/// `censoring_fit_check`（両側打ち切り）: `lower`・`upper`双方が`Some`になり、
/// 3カテゴリの実測比率・モデル含意の平均確率いずれも独立再計算と一致し、かつ
/// それぞれ合計1になることを確認する。
#[test]
fn fit_censoring_fit_check_matches_independent_recomputation_for_two_sided_censoring() {
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
    let normal = Normal::new(0.0, 1.0).unwrap();

    let check = estimator.censoring_fit_check();
    let lower_check = check.lower().unwrap();
    let upper_check = check.upper().unwrap();

    // two_sided_censored_regression_input: y=[0,0,1.15,2.9,5.2,6.0,6.0,6.0]、
    // lower=0.0でx=1,2（2件）・upper=6.0でx=6,7,8（3件）が打ち切り
    assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
    assert!((upper_check.observed_rate() - 0.375).abs() < 1e-12);
    assert!((check.uncensored().observed_rate() - 0.375).abs() < 1e-12);
    assert!(
        (lower_check.observed_rate()
            + check.uncensored().observed_rate()
            + upper_check.observed_rate()
            - 1.0)
            .abs()
            < 1e-12
    );

    let expected_lower_rate: f64 = (0..n)
        .map(|i| {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            normal.cdf((0.0 - mu) / sigma)
        })
        .sum::<f64>()
        / (n as f64);
    let expected_upper_rate: f64 = (0..n)
        .map(|i| {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            1.0 - normal.cdf((6.0 - mu) / sigma)
        })
        .sum::<f64>()
        / (n as f64);
    let expected_uncensored_rate: f64 = (0..n)
        .map(|i| {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            prob_uncensored_closed_form(mu, sigma, Some(0.0), Some(6.0))
        })
        .sum::<f64>()
        / (n as f64);
    assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);
    assert!((upper_check.model_implied_rate() - expected_upper_rate).abs() < 1e-9);
    assert!((check.uncensored().model_implied_rate() - expected_uncensored_rate).abs() < 1e-9);
    assert!(
        (lower_check.model_implied_rate()
            + check.uncensored().model_implied_rate()
            + upper_check.model_implied_rate()
            - 1.0)
            .abs()
            < 1e-9
    );
}

/// `censoring_fit_check`版の多変量配線ミス検出テスト（`predict.rs`の`fit_predict_matches_
/// independent_recomputation_for_multivariate_design`と同じ理由）。
#[test]
fn fit_censoring_fit_check_matches_independent_recomputation_for_multivariate_design() {
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
    let normal = Normal::new(0.0, 1.0).unwrap();

    let check = estimator.censoring_fit_check();
    assert!(check.upper().is_none());
    let lower_check = check.lower().unwrap();

    // multivariate_censored_input: y=[0,0,1.15,2.9,5.2,6.85,9.1,10.95]、lower=0.0で
    // 打ち切られているのはx1=1,2の2件（実測打ち切り率2/8=0.25、censored_regression_
    // inputと同じyだがx2列が追加された多変量設計行列）
    assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
    assert!((check.uncensored().observed_rate() - 0.75).abs() < 1e-12);

    let expected_lower_rate: f64 = (0..n)
        .map(|i| {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            normal.cdf((0.0 - mu) / sigma)
        })
        .sum::<f64>()
        / (n as f64);
    assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);
}
