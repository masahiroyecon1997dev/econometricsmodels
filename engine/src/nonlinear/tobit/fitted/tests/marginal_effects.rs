//! `marginal_effects`のテスト。

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
use crate::nonlinear::tobit::target::MarginalEffectsTarget;
use crate::nonlinear::tobit::test_support::*;
use crate::shared::error::CommonError;
use statrs::distribution::Continuous;
use statrs::distribution::Normal;

/// `∂P(uncensored|x)/∂μ`の閉形式（`prob_uncensored_closed_form`を`mu`について解析的に
/// 微分した式。`target_w_and_s`の`ProbUncensored`分岐とは独立に書き下したコード）。
fn dprob_uncensored_dmu_closed_form(
    mu: f64,
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    match (lower, upper) {
        (Some(c), None) => normal.pdf((c - mu) / sigma) / sigma,
        (None, Some(c)) => -normal.pdf((c - mu) / sigma) / sigma,
        (Some(c1), Some(c2)) => {
            (normal.pdf((c1 - mu) / sigma) - normal.pdf((c2 - mu) / sigma)) / sigma
        }
        (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
    }
}

/// `at="overall"`の限界効果（effect・デルタ法のSE）を、独立の閉形式`m`から作った
/// AME関数の中心差分ヤコビアンで再計算した値と突き合わせる（左のみ・右のみ・両側）。
#[test]
fn fit_marginal_effects_expected_uncensored_overall_matches_independent_recomputation() {
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
        let n = x.nrows();
        let k = 2;
        let label = format!("lower={lower:?}, upper={upper:?}");

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedUncensored,
                0.95,
            )
            .unwrap();

        // AME_j = (1/n)Σ w_i·β_j, w_i=∂m/∂μ を m の中心差分で作る（k=2なのでj=1のみ）
        let ame = |params: &[f64], sigma: f64| -> f64 {
            let h = 1e-5;
            (0..n)
                .map(|i| {
                    let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                    let m = |mu: f64| expected_uncensored_closed_form(mu, sigma, lower, upper);
                    (m(mu + h) - m(mu - h)) / (2.0 * h) * params[1]
                })
                .sum::<f64>()
                / (n as f64)
        };
        let params = estimator.params();
        let sigma = estimator.sigma();
        assert!(
            (effects.effect()[0] - ame(params, sigma)).abs() < 1e-6,
            "{label}: actual={}, expected={}",
            effects.effect()[0],
            ame(params, sigma)
        );

        let h_param = 1e-3;
        let mut jac_row = vec![0.0; k + 1];
        for (m_idx, jac) in jac_row.iter_mut().take(k).enumerate() {
            let mut plus_p = params.to_vec();
            plus_p[m_idx] += h_param;
            let mut minus_p = params.to_vec();
            minus_p[m_idx] -= h_param;
            *jac = (ame(&plus_p, sigma) - ame(&minus_p, sigma)) / (2.0 * h_param);
        }
        jac_row[k] =
            (ame(params, sigma + h_param) - ame(params, sigma - h_param)) / (2.0 * h_param);
        let mut var = 0.0;
        for a in 0..=k {
            for b in 0..=k {
                var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
            }
        }
        let expected_se = var.sqrt();
        assert!(
            (effects.std_errors()[0] - expected_se).abs() < 1e-5 * (1.0 + expected_se),
            "{label}: actual SE={}, expected SE={expected_se}",
            effects.std_errors()[0]
        );
    }
}

/// 切片のみモデル（k=1）は限界効果の出力対象となる説明変数が存在しない
/// （定数項は出力から除外するため）。`marginal_effects`が空の結果を返す
/// （パニックしない）ことを確認する境界ケース（Logitの同名テストと同じ理由）。
#[test]
fn fit_marginal_effects_returns_empty_result_for_intercept_only_model() {
    let estimator = TobitEstimator::fit(
        intercept_only_uncensored_input(&[1.0, 2.0, 3.0, 4.0]),
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

    let effects = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            0.95,
        )
        .unwrap();
    assert!(effects.param_names().is_empty());
    assert!(effects.effect().is_empty());
}

#[test]
fn fit_marginal_effects_returns_invalid_confidence_level_error_out_of_range() {
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

    let result = estimator.marginal_effects(
        MarginalEffectsAt::Overall,
        MarginalEffectsTarget::ExpectedObserved,
        1.5,
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5
        })
    );
}

/// `target="E[y*|x]"`（潜在変数の期待値）は`dydx_j=βⱼ`という自明な形（評価点`x`に
/// 依存しない）ため、`at`（overall/mean/median）に関わらず常に同じ値になり、
/// その標準誤差も`β`自体の標準誤差（`std_errors()`の対応する要素）と厳密に一致する
/// （ヤコビアンが`β`ブロックで恒等写像・`σ`列がゼロに退化するため、数値微分に頼らず
/// 厳密等式として検証できる）。
#[test]
fn fit_marginal_effects_expected_latent_equals_beta_and_matches_beta_std_error() {
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

    for at in [
        MarginalEffectsAt::Overall,
        MarginalEffectsAt::Mean,
        MarginalEffectsAt::Median,
    ] {
        let effects = estimator
            .marginal_effects(at, MarginalEffectsTarget::ExpectedLatent, 0.95)
            .unwrap();
        assert_eq!(effects.param_names(), ["x1".to_string(), "x2".to_string()]);
        for (idx, j) in (1..3).enumerate() {
            assert!((effects.effect()[idx] - estimator.params()[j]).abs() < 1e-12);
            assert!(
                (effects.std_errors()[idx] - estimator.std_errors()[j]).abs() < 1e-9,
                "at={at:?}, idx={idx}, actual={}, expected={}",
                effects.std_errors()[idx],
                estimator.std_errors()[j]
            );
        }
    }
}

/// `target="E[y|x]"`（既定）の`at="overall"`（AME）を、実装内部（`target_w_and_s`）とは
/// 別に書いた閉形式`expected_observed_closed_form`を各観測で平均する方法で独立に
/// 再計算し、突き合わせる。標準誤差は、`dydx_j`を`prob_uncensored_closed_form`
/// （非打ち切り確率の閉形式）×`βⱼ`という独立な式で表し、fit済みパラメータの周りで
/// 数値微分したヤコビアンと`cov_params`の二次形式で計算した分散の平方根と突き合わせる
/// （Logitの`marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_
/// method_se`と同じ検算パターン）。
#[test]
fn fit_marginal_effects_expected_observed_overall_matches_independently_recomputed_dydx_and_delta_method_se()
 {
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
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    let x = estimator.input().x();
    let n = x.nrows();
    let k = 2;

    let effects = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            0.95,
        )
        .unwrap();
    assert_eq!(effects.param_names(), ["x1".to_string()]);

    // effectの独立再計算: 各観測でE[y|x]を数値微分してx1の限界効果を求め、平均する
    // （`target_w_and_s`の`P(uncensored)*βⱼ`という閉形式は経由しない）。
    let h = 1e-6;
    let dydx_at_row = |params: &[f64], sigma: f64, i: usize| -> f64 {
        let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
        let x1 = *x.get(i, 1);
        let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
        let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
        (plus - minus) / (2.0 * h)
    };
    let dydx_ame = |params: &[f64], sigma: f64| -> f64 {
        (0..n).map(|i| dydx_at_row(params, sigma, i)).sum::<f64>() / (n as f64)
    };
    let params = estimator.params();
    let sigma = estimator.sigma();
    assert!(
        (effects.effect()[0] - dydx_ame(params, sigma)).abs() < 1e-6,
        "actual={}, expected={}",
        effects.effect()[0],
        dydx_ame(params, sigma)
    );

    // 数値微分とは別の、McDonald-Moffitt(1980)の厳密な恒等式
    // `∂E[y|x]/∂xⱼ = P(uncensored|x)*βⱼ`（`target_w_and_s`のdocコメント「数式」参照）
    // による検算。`prob_uncensored_closed_form`は数値微分を一切経由しない厳密な閉形式
    // のため、数値誤差なしで一致するはず。
    let expected_dydx_exact: f64 = (0..n)
        .map(|i| {
            let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
            prob_uncensored_closed_form(mu, sigma, lower, upper) * params[1]
        })
        .sum::<f64>()
        / (n as f64);
    assert!(
        (effects.effect()[0] - expected_dydx_exact).abs() < 1e-9,
        "actual={}, expected={expected_dydx_exact}",
        effects.effect()[0]
    );

    // 標準誤差の独立検証: `dydx_ame`をfit済み`(β,σ)`の周りで数値微分してヤコビアン行を
    // 求め、`cov_params`との二次形式で分散を計算する。
    let h_param = 1e-5;
    let mut jac_row = vec![0.0; k + 1];
    for m in 0..k {
        let mut plus_p = params.to_vec();
        plus_p[m] += h_param;
        let mut minus_p = params.to_vec();
        minus_p[m] -= h_param;
        jac_row[m] = (dydx_ame(&plus_p, sigma) - dydx_ame(&minus_p, sigma)) / (2.0 * h_param);
    }
    jac_row[k] =
        (dydx_ame(params, sigma + h_param) - dydx_ame(params, sigma - h_param)) / (2.0 * h_param);

    let mut var = 0.0;
    for a in 0..=k {
        for b in 0..=k {
            var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
        }
    }
    let expected_se = var.sqrt();
    assert!(
        (effects.std_errors()[0] - expected_se).abs() < 1e-4,
        "actual={}, expected={expected_se}",
        effects.std_errors()[0]
    );
}

/// `at="mean"`/`"median"`が`at="overall"`と異なる代表点で評価されるため、一般には
/// 異なる値になることを確認する（Logitの同名テストと同じ理由）。
#[test]
fn fit_marginal_effects_expected_observed_at_mean_and_median_differ_from_overall() {
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

    let overall = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            0.95,
        )
        .unwrap();
    let at_mean = estimator
        .marginal_effects(
            MarginalEffectsAt::Mean,
            MarginalEffectsTarget::ExpectedObserved,
            0.95,
        )
        .unwrap();
    let at_median = estimator
        .marginal_effects(
            MarginalEffectsAt::Median,
            MarginalEffectsTarget::ExpectedObserved,
            0.95,
        )
        .unwrap();

    assert!((overall.effect()[0] - at_mean.effect()[0]).abs() > 1e-9);
    assert!((at_mean.effect()[0] - at_median.effect()[0]).abs() < 1e-12);

    // x1の平均・中央値がともに4.5（[1..8]の対称なデータのため）であることを利用し、
    // `expected_observed_closed_form`から独立に再計算したdydxと突き合わせる。
    let params = estimator.params();
    let sigma = estimator.sigma();
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
    let h = 1e-6;
    let expected_dydx = (expected_observed_closed_form(mu(4.5 + h), sigma, lower, upper)
        - expected_observed_closed_form(mu(4.5 - h), sigma, lower, upper))
        / (2.0 * h);
    assert!((at_mean.effect()[0] - expected_dydx).abs() < 1e-6);
}

/// `target="E[y|x]"`が右打ち切りのみのデータでも正しく計算できることを、
/// `expected_observed_closed_form`（右打ち切り分岐）で独立に再計算した`effect`と
/// 突き合わせて検証する。
#[test]
fn fit_marginal_effects_expected_observed_matches_independent_recomputation_for_right_only_censoring()
 {
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
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    assert_eq!(lower, None);
    let x = estimator.input().x();
    let n = x.nrows();
    let params = estimator.params();
    let sigma = estimator.sigma();

    let effects = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            0.95,
        )
        .unwrap();

    let h = 1e-6;
    let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
    let expected_dydx: f64 = (0..n)
        .map(|i| {
            let x1 = *x.get(i, 1);
            let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
            let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
            (plus - minus) / (2.0 * h)
        })
        .sum::<f64>()
        / (n as f64);

    assert!(
        (effects.effect()[0] - expected_dydx).abs() < 1e-6,
        "actual={}, expected={expected_dydx}",
        effects.effect()[0]
    );
}

/// `target="E[y|x]"`が両側打ち切りのデータでも正しく計算できることを、
/// `expected_observed_closed_form`（両側打ち切り分岐）で独立に再計算した`effect`と
/// 突き合わせて検証する。
#[test]
fn fit_marginal_effects_expected_observed_matches_independent_recomputation_for_two_sided_censoring()
 {
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
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    assert_eq!(lower, Some(0.0));
    assert_eq!(upper, Some(6.0));
    let x = estimator.input().x();
    let n = x.nrows();
    let params = estimator.params();
    let sigma = estimator.sigma();

    let effects = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            0.95,
        )
        .unwrap();

    let h = 1e-6;
    let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
    let expected_dydx: f64 = (0..n)
        .map(|i| {
            let x1 = *x.get(i, 1);
            let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
            let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
            (plus - minus) / (2.0 * h)
        })
        .sum::<f64>()
        / (n as f64);

    assert!(
        (effects.effect()[0] - expected_dydx).abs() < 1e-6,
        "actual={}, expected={expected_dydx}",
        effects.effect()[0]
    );
}

/// `target="P(uncensored)"`（左打ち切りのみ）を、`prob_uncensored_closed_form`を
/// `mu`について数値微分した`dprob_uncensored_dmu_closed_form`×`βⱼ`という
/// （`target_w_and_s`の`ProbUncensored`分岐とは独立の）式で再計算し、effect・標準誤差
/// 双方を突き合わせる。
#[test]
fn fit_marginal_effects_prob_uncensored_overall_matches_independently_recomputed_dydx_and_delta_method_se()
 {
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
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    let x = estimator.input().x();
    let n = x.nrows();
    let k = 2;

    let effects = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ProbUncensored,
            0.95,
        )
        .unwrap();

    let dydx_ame = |params: &[f64], sigma: f64| -> f64 {
        (0..n)
            .map(|i| {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
            })
            .sum::<f64>()
            / (n as f64)
    };
    let params = estimator.params();
    let sigma = estimator.sigma();
    assert!(
        (effects.effect()[0] - dydx_ame(params, sigma)).abs() < 1e-9,
        "actual={}, expected={}",
        effects.effect()[0],
        dydx_ame(params, sigma)
    );

    let h_param = 1e-5;
    let mut jac_row = vec![0.0; k + 1];
    for m in 0..k {
        let mut plus_p = params.to_vec();
        plus_p[m] += h_param;
        let mut minus_p = params.to_vec();
        minus_p[m] -= h_param;
        jac_row[m] = (dydx_ame(&plus_p, sigma) - dydx_ame(&minus_p, sigma)) / (2.0 * h_param);
    }
    jac_row[k] =
        (dydx_ame(params, sigma + h_param) - dydx_ame(params, sigma - h_param)) / (2.0 * h_param);

    let mut var = 0.0;
    for a in 0..=k {
        for b in 0..=k {
            var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
        }
    }
    let expected_se = var.sqrt();
    assert!(
        (effects.std_errors()[0] - expected_se).abs() < 1e-4,
        "actual={}, expected={expected_se}",
        effects.std_errors()[0]
    );
}

/// `target="P(uncensored)"`は右打ち切りのみだと符号が反転する（`x`の増加が上側の
/// 打ち切り確率を高めるため非打ち切り確率は減少する。`target_w_and_s`のdocコメント
/// 「数式」参照）。この符号反転が正しく実装されていることを、右打ち切りのみの
/// データで独立再計算したdydxと突き合わせて確認する。
#[test]
fn fit_marginal_effects_prob_uncensored_matches_independent_recomputation_for_right_only_censoring()
{
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
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    let x = estimator.input().x();
    let n = x.nrows();
    let params = estimator.params();
    let sigma = estimator.sigma();

    let effects = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ProbUncensored,
            0.95,
        )
        .unwrap();

    let expected_dydx: f64 = (0..n)
        .map(|i| {
            let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
            dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
        })
        .sum::<f64>()
        / (n as f64);

    assert!(
        (effects.effect()[0] - expected_dydx).abs() < 1e-9,
        "actual={}, expected={expected_dydx}",
        effects.effect()[0]
    );
    // x1の真の係数は正（+2相当）なので、右打ち切りのみでは非打ち切り確率の限界効果は
    // 負になるはず（xが増えるほど上側で打ち切られやすくなるため）。
    assert!(effects.effect()[0] < 0.0, "effect={}", effects.effect()[0]);
}

/// `target="P(uncensored)"`が両側打ち切りのデータでも正しく計算できることを検証する。
#[test]
fn fit_marginal_effects_prob_uncensored_matches_independent_recomputation_for_two_sided_censoring()
{
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
    let lower = estimator.input().lower();
    let upper = estimator.input().upper();
    let x = estimator.input().x();
    let n = x.nrows();
    let params = estimator.params();
    let sigma = estimator.sigma();

    let effects = estimator
        .marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ProbUncensored,
            0.95,
        )
        .unwrap();

    let expected_dydx: f64 = (0..n)
        .map(|i| {
            let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
            dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
        })
        .sum::<f64>()
        / (n as f64);

    assert!(
        (effects.effect()[0] - expected_dydx).abs() < 1e-9,
        "actual={}, expected={expected_dydx}",
        effects.effect()[0]
    );
}
