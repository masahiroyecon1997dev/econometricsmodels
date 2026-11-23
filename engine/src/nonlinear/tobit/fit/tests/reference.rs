//! `fit`の結果を参照実装（statsmodels・linearmodels・plm・R）や閉形式・独立再計算と照合するテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::tobit::TobitEstimator;
use crate::nonlinear::tobit::test_support::*;
use statrs::distribution::ChiSquared;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// 切片のみ・打ち切りなしのTobit（`(β,s=logσ)`空間の対数尤度が通常の正規分布の
/// 対数尤度に一致する）は、観測情報行列も閉じた形で書ける: `∂²ℓ/∂β²=-n/σ²`・
/// `∂²ℓ/∂s²=-2n`・`∂²ℓ/∂β∂s=0`（MLE点で`Σ(yᵢ-β̂)=0`となるため）という対角行列になり、
/// `Var(β̂)=σ̂²/n`・`Var(ŝ)=1/(2n)`・`Cov(β̂,ŝ)=0`が導ける（`docs/spec/tobit-spec.md`
/// 3.1〜3.3節の一般形の特殊ケース）。
/// `σ`のデルタ法変換（ヤコビアン`diag(1,σ)`）を適用すると、この対角性はそのまま
/// 保たれ`Var(σ̂)≈σ̂²Var(ŝ)=σ̂²/(2n)`・`Cov(β̂,σ̂)≈σ̂Cov(β̂,ŝ)=0`になる。
/// Logitの`fit_computes_std_errors_test_stats_p_values_and_ci_matching_closed_form_for_
/// intercept_only_model`と同じ設計（本体実装と同じ式を繰り返すのではなく、独立に
/// 導出した閉じた形の解でSE・z値・p値・信頼区間を検算する）。
#[test]
fn fit_computes_std_errors_test_stats_p_values_and_ci_matching_closed_form_for_intercept_only_uncensored_model()
 {
    let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
    let input = intercept_only_uncensored_input(&y);

    let n = y.len() as f64;
    let y_bar: f64 = y.iter().sum::<f64>() / n;
    let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
    let sigma_hat = (sse / n).sqrt();

    let expected_var_beta = sigma_hat.powi(2) / n;
    let expected_var_sigma = sigma_hat.powi(2) / (2.0 * n);

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

    assert!((*estimator.cov_params().get(0, 0) - expected_var_beta).abs() < 1e-6);
    assert!((*estimator.cov_params().get(1, 1) - expected_var_sigma).abs() < 1e-6);
    assert!(
        estimator.cov_params().get(0, 1).abs() < 1e-6,
        "cov(beta,sigma)={}",
        estimator.cov_params().get(0, 1)
    );

    let expected_se_beta = expected_var_beta.sqrt();
    let expected_se_sigma = expected_var_sigma.sqrt();
    assert!((estimator.std_errors()[0] - expected_se_beta).abs() < 1e-6);
    assert!((estimator.std_errors()[1] - expected_se_sigma).abs() < 1e-6);

    // z値・p値・信頼区間はstatrsのNormalで独立に検算する（本体実装と同じ計算式を
    // 繰り返すのではなく、標準正規分布の性質から直接導出する）。
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_crit = normal.inverse_cdf(0.975);

    let expected_z_beta = estimator.params()[0] / expected_se_beta;
    let expected_p_beta = 2.0 * (1.0 - normal.cdf(expected_z_beta.abs()));
    assert!((estimator.test_stats()[0] - expected_z_beta).abs() < 1e-6);
    assert!((estimator.p_values()[0] - expected_p_beta).abs() < 1e-6);
    assert!(
        (estimator.conf_lower()[0] - (estimator.params()[0] - z_crit * expected_se_beta)).abs()
            < 1e-6
    );
    assert!(
        (estimator.conf_upper()[0] - (estimator.params()[0] + z_crit * expected_se_beta)).abs()
            < 1e-6
    );

    let expected_z_sigma = estimator.sigma() / expected_se_sigma;
    let expected_p_sigma = 2.0 * (1.0 - normal.cdf(expected_z_sigma.abs()));
    assert!((estimator.test_stats()[1] - expected_z_sigma).abs() < 1e-6);
    assert!((estimator.p_values()[1] - expected_p_sigma).abs() < 1e-6);
    assert!(
        (estimator.conf_lower()[1] - (estimator.sigma() - z_crit * expected_se_sigma)).abs() < 1e-6
    );
    assert!(
        (estimator.conf_upper()[1] - (estimator.sigma() + z_crit * expected_se_sigma)).abs() < 1e-6
    );
}

/// 切片のみ・打ち切りなしのTobit（通常の正規分布のMLEに一致する）の対数尤度は
/// 閉じた形の解析解`ℓ=-(n/2)(ln(2π)+ln(σ̂²)+1)`を持つ（OLSの`log_likelihood`と同じ式。
/// `σ̂²=Σ(y-ȳ)²/n`はMLE推定量で、OLSの不偏推定量（`n-1`除算）とは異なる）。
/// `aic`/`bic`はこの`llf`と総パラメータ数`k+1=2`（切片1+`logσ`1）から直接計算できる。
/// 切片のみモデルは傾き係数が無い（`df_model=0`）ため、Wald検定がスキップされ
/// `wald_statistic`/`wald_p_value`がNaNになることも合わせて検証する。
#[test]
fn fit_computes_log_likelihood_aic_bic_matching_closed_form_for_intercept_only_uncensored_model() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
    let input = intercept_only_uncensored_input(&y);

    let n = y.len() as f64;
    let y_bar: f64 = y.iter().sum::<f64>() / n;
    let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
    let sigma_sq_hat = sse / n;
    let expected_llf = -(n / 2.0) * ((2.0 * std::f64::consts::PI).ln() + sigma_sq_hat.ln() + 1.0);
    let expected_aic = -2.0 * expected_llf + 2.0 * 2.0;
    let expected_bic = -2.0 * expected_llf + n.ln() * 2.0;

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

    assert!((estimator.log_likelihood() - expected_llf).abs() < 1e-6);
    assert!((estimator.aic() - expected_aic).abs() < 1e-6);
    assert!((estimator.bic() - expected_bic).abs() < 1e-6);
    assert_eq!(estimator.n_obs(), 5);
    assert_eq!(estimator.df_model(), 0);
    assert_eq!(estimator.df_resid(), 3);
    assert!(estimator.wald_statistic().is_nan());
    assert!(estimator.wald_p_value().is_nan());
}

/// Wald検定統計量・p値が独立再計算（2×2部分行列の手動逆行列計算、`Llt`分解を経由
/// しない）と一致することを検証する。`multivariate_censored_input`（切片+`x1`+`x2`、
/// `k=3`、傾き係数`q=2`）を使う。
#[test]
fn fit_wald_statistic_and_p_value_match_independently_recomputed_values() {
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

    // 傾き係数（切片を除く`params[1]`・`params[2]`）と、対応する`cov_params`の
    // 2×2部分行列（`σ`次元は含まれない）を使い、2×2逆行列の公式で手動計算する。
    let beta1 = estimator.params()[1];
    let beta2 = estimator.params()[2];
    let v = estimator.cov_params();
    let (v11, v12, v21, v22) = (*v.get(1, 1), *v.get(1, 2), *v.get(2, 1), *v.get(2, 2));
    let det = v11 * v22 - v12 * v21;
    let (inv11, inv12, inv21, inv22) = (v22 / det, -v12 / det, -v21 / det, v11 / det);
    let expected_wald =
        beta1 * (inv11 * beta1 + inv12 * beta2) + beta2 * (inv21 * beta1 + inv22 * beta2);

    let chi2 = ChiSquared::new(2.0).unwrap();
    let expected_p = 1.0 - chi2.cdf(expected_wald);

    assert_eq!(estimator.df_model(), 2);
    assert!((estimator.wald_statistic() - expected_wald).abs() < 1e-6);
    assert!((estimator.wald_p_value() - expected_p).abs() < 1e-9);
}
