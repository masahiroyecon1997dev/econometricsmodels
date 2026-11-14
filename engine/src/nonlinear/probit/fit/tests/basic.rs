//! `fit`の基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::probit::ProbitEstimator;
use crate::nonlinear::probit::ProbitInput;
use crate::nonlinear::probit::test_support::*;
use statrs::distribution::ChiSquared;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// 切片のみモデルは`log_likelihood`と`log_likelihood_null`が定義上一致する
/// （どちらも同じ「切片のみ」モデルを参照するため）。この性質を使い、
/// `df_model=0`分岐（`lr_p_value=NaN`、OLSの`f_p_value`と同じ扱い）を含めた
/// 適合度統計量一式を検証する（`LogitEstimator`の対応するテストと同じ構成）。
#[test]
fn fit_computes_goodness_of_fit_statistics_for_intercept_only_model() {
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

    let n: f64 = 7.0;
    let y_bar: f64 = 4.0 / 7.0;
    // 切片のみモデルの対数尤度の閉じた形: n1*ln(ȳ) + n0*ln(1-ȳ)（n1=4, n0=3）
    let expected_ll = 4.0 * y_bar.ln() + 3.0 * (1.0 - y_bar).ln();

    assert!((estimator.log_likelihood() - expected_ll).abs() < 1e-6);
    assert!((estimator.log_likelihood_null() - expected_ll).abs() < 1e-9);
    assert!(estimator.lr_statistic().abs() < 1e-6);
    assert!(estimator.pseudo_r_squared().abs() < 1e-6);
    assert_eq!(estimator.df_model(), 0);
    assert!(estimator.lr_p_value().is_nan());
    assert_eq!(estimator.n_obs(), 7);
    assert_eq!(estimator.df_resid(), 6);

    let expected_aic = -2.0 * expected_ll + 2.0;
    let expected_bic = -2.0 * expected_ll + n.ln();
    assert!((estimator.aic() - expected_aic).abs() < 1e-6);
    assert!((estimator.bic() - expected_bic).abs() < 1e-6);
}

/// `include_intercept=false`のとき、`log_likelihood_null`が参照する「切片のみ」
/// モデルはフィット対象のモデルの部分集合（入れ子）にならない
/// （`ProbitEstimator`の`log_likelihood_null`フィールドdocコメント参照）。
/// この場合`lr_statistic`が負になりうる（statsmodels準拠の仕様上の挙動、
/// `LogitEstimator`の対応するテストと同じ回帰テスト）ことを固定する。
/// `df_model`/`df_resid`/`aic`/`bic`は`include_intercept`の値に関わらず
/// 同じ式（`k-1`/`n-k`/`-2ℓ+2k`/`-2ℓ+ln(n)k`）で計算されることも確認する。
#[test]
fn fit_lr_statistic_can_be_negative_when_include_intercept_is_false() {
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

    let n = 4usize;
    let k = 2usize; // 切片なし、x1・x2のみ
    assert_eq!(estimator.df_model(), k - 1);
    assert_eq!(estimator.df_resid(), n - k);

    let expected_aic = -2.0 * estimator.log_likelihood() + 2.0 * (k as f64);
    let expected_bic = -2.0 * estimator.log_likelihood() + (n as f64).ln() * (k as f64);
    assert!((estimator.aic() - expected_aic).abs() < 1e-9);
    assert!((estimator.bic() - expected_bic).abs() < 1e-9);

    // 非入れ子のため`lr_statistic`が負になりうる（NaN/Infにはならない）。
    assert!(estimator.lr_statistic().is_finite());
    assert!(estimator.lr_p_value().is_finite());
}

/// 切片のみモデルは観測情報行列も閉じた形で書ける。MLEの一階条件`Σλᵢ=0`が
/// 収束点で厳密に成り立つことを使うと、Hessian（`-ℓ`の2階微分）は
/// `H(θ̂) = Σᵢλᵢ² = n*φ(θ̂)²/(ȳ(1-ȳ))`という形に単純化できる
/// （`λᵢ(λᵢ+θ)`の`θ`の項が`θ*Σλᵢ=0`で消える。導出は`probit.rs`のモジュールdocコメント
/// 「数式」節の`λᵢ`の定義と合わせて`docs/spec/probit-spec.md`参照）。
/// `Var(θ̂) = H(θ̂)⁻¹ = ȳ(1-ȳ)/(n*φ(θ̂)²)`。z値・p値・信頼区間はこの分散から
/// 標準正規分布（独立に`statrs::Normal`で検算）で導出できる。
#[test]
fn fit_computes_std_errors_test_stats_p_values_and_ci_matching_closed_form_for_intercept_only_model()
 {
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

    let n: f64 = 7.0;
    let y_bar: f64 = 4.0 / 7.0;
    let theta_hat = Normal::standard().inverse_cdf(y_bar);
    let phi_theta_hat = Normal::standard().pdf(theta_hat);
    let expected_var = y_bar * (1.0 - y_bar) / (n * phi_theta_hat * phi_theta_hat);
    let expected_se = expected_var.sqrt();

    // Newtonの収束判定（勾配ノルム`tol=1e-6`）による数値誤差があるため、
    // 他の閉じた形テスト（`fit_newton_converges_to_closed_form_solution_...`）と
    // 同じ桁の許容誤差（1e-6）を使う。
    assert!((*estimator.cov_params().get(0, 0) - expected_var).abs() < 1e-6);
    assert!((estimator.std_errors()[0] - expected_se).abs() < 1e-6);

    let expected_z = estimator.params()[0] / expected_se;
    assert!((estimator.test_stats()[0] - expected_z).abs() < 1e-6);

    // p値・信頼区間はstatrsのNormalで独立に検算する（本体実装と同じ計算式を
    // 繰り返すのではなく、標準正規分布の性質から直接導出する）。
    let normal = Normal::standard();
    let expected_p = 2.0 * (1.0 - normal.cdf(expected_z.abs()));
    assert!((estimator.p_values()[0] - expected_p).abs() < 1e-6);

    let z_crit = normal.inverse_cdf(0.975);
    let expected_lower = estimator.params()[0] - z_crit * expected_se;
    let expected_upper = estimator.params()[0] + z_crit * expected_se;
    assert!((estimator.conf_lower()[0] - expected_lower).abs() < 1e-6);
    assert!((estimator.conf_upper()[0] - expected_upper).abs() < 1e-6);
}

/// 多変量（k=3）モデルでの適合度統計量を、実装（`clamped_pdf_cdf`ベース）とは異なる式
/// （標準正規CDF`Φ`から直接`Σ[y ln Φ(z) + (1-y) ln(1-Φ(z))]`を計算するベルヌーイ
/// 対数尤度の定義式そのもの）で独立に再計算し、突き合わせる。
/// `fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ
/// データセットを再利用する（`LogitEstimator`の対応するテストと同じ構成）。
#[test]
fn fit_computes_goodness_of_fit_statistics_matching_independently_recomputed_values() {
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

    let n = 4usize;
    let k = 3usize;
    let x = estimator.input().x();
    let params = estimator.params();
    let normal = Normal::standard();
    let expected_ll: f64 = (0..n)
        .map(|i| {
            let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let p = normal.cdf(z);
            y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln()
        })
        .sum();
    assert!((estimator.log_likelihood() - expected_ll).abs() < 1e-9);

    let y_bar: f64 = 2.0 / 4.0; // n1=2, n0=2
    let expected_llnull = 2.0 * y_bar.ln() + 2.0 * (1.0 - y_bar).ln();
    assert!((estimator.log_likelihood_null() - expected_llnull).abs() < 1e-9);

    let expected_lr = 2.0 * (expected_ll - expected_llnull);
    assert!((estimator.lr_statistic() - expected_lr).abs() < 1e-9);

    let expected_pseudo_r2 = 1.0 - expected_ll / expected_llnull;
    assert!((estimator.pseudo_r_squared() - expected_pseudo_r2).abs() < 1e-9);

    assert_eq!(estimator.df_model(), k - 1);
    assert_eq!(estimator.df_resid(), n - k);
    assert_eq!(estimator.n_obs(), n);

    let chi2 = ChiSquared::new((k - 1) as f64).unwrap();
    let expected_lr_p = 1.0 - chi2.cdf(expected_lr);
    assert!((estimator.lr_p_value() - expected_lr_p).abs() < 1e-9);

    let expected_aic = -2.0 * expected_ll + 2.0 * (k as f64);
    let expected_bic = -2.0 * expected_ll + (n as f64).ln() * (k as f64);
    assert!((estimator.aic() - expected_aic).abs() < 1e-9);
    assert!((estimator.bic() - expected_bic).abs() < 1e-9);
}
