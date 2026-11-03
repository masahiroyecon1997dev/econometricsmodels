//! `TobitEstimator::fit`の本体。検証 → 標準化 → 初期値 → 最適化 → `cov_type`別の共分散 →
//! 推論統計量 → 適合度統計量の組み立て。

use super::TobitEstimator;
use super::fitted::TobitFitted;
use super::input::TobitInput;
use super::problem::TobitProblem;
use super::scaling::TobitScaling;
use super::statistics::{
    log_likelihood, ols_initial_params, validate_has_uncensored_observations, wald_chi2_test,
};
use crate::nonlinear::common::{
    CovType, MleError, MleFitOptions, SandwichVariant, cluster_cov_params,
    observed_information_cov_params, opg_cov_params, run_solver, sandwich_cov_params,
    validate_cluster_cov_type, validate_mle_options, validate_sufficient_observations,
};
use crate::shared::inference;
use faer::Mat;
use statrs::distribution::Normal;

/// [`TobitEstimator::fit`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_tobit(
    input: TobitInput,
    options: MleFitOptions,
) -> Result<TobitEstimator, MleError> {
    let MleFitOptions {
        solver,
        max_iter,
        tol,
        raise_on_non_convergence,
        cov_type,
        confidence_level,
    } = options;

    // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
    crate::shared::parallelism::ensure_serial();

    validate_mle_options(confidence_level, max_iter, tol)?;

    let n = input.nobs();
    let k = input.k();
    validate_sufficient_observations(n, k + 1)?;
    validate_cluster_cov_type(&cov_type, n, k - usize::from(input.has_intercept()))?;
    validate_has_uncensored_observations(input.y(), input.lower(), input.upper())?;

    // `x`（切片ありなら列を平均センタリング＋スケーリング、切片なしはスケーリング
    // のみ）と`y`（と打ち切り境界、`y`の母集団標準偏差を2の冪に丸めた値で一律
    // スケーリング）を標準化してから最適化する（`TobitScaling`のdocコメント・
    // `fit`のdocコメント「標準化空間」節参照）。
    let scaling = TobitScaling::fit(input.x(), input.y(), input.has_intercept());
    let x_std = scaling.standardize_x(input.x());
    let y_std = scaling.standardize_y(input.y());
    let lower_std = scaling.standardize_bound(input.lower());
    let upper_std = scaling.standardize_bound(input.upper());
    let initial_params = ols_initial_params(&x_std, input.x(), &y_std, input.param_names())?;
    let problem = TobitProblem::from_standardized(x_std, y_std, lower_std, upper_std);
    // `cov_type`がOPG/サンドイッチ型/クラスターの場合、収束点でのスコア評価に
    // 標準化空間の`TobitProblem`が必要になる。`run_solver`は`problem`の所有権を
    // 取り込むため、事前にクローンしておく（`LogitEstimator::fit`と同じ理由）。
    let problem_for_scores = match &cov_type {
        CovType::Classical => None,
        CovType::Opg | CovType::Hc0 | CovType::Hc1 | CovType::Cluster { .. } => {
            Some(problem.clone())
        }
    };

    let output = run_solver(
        problem,
        solver,
        initial_params,
        max_iter as u64,
        tol,
        input.y().nrows(),
        raise_on_non_convergence,
    )?;
    // Tobitの真の分離は係数発散ではなく`σ→0`退化として現れるため、Logit/Probitの分離検出
    // （`nonlinear::separation`）は行わない。全件打ち切りは
    // `validate_has_uncensored_observations`（`NoUncensoredObservations`）、
    // 部分的な準完全分離は`NonConvergence`で捕捉される。

    // 標準化空間の内部パラメータ`(β̃, s̃)`を元のスケールの`(β, logσ)`へ逆変換する
    // （`TobitScaling`のdocコメントの式）。
    let params_full = scaling.destandardize_params(&output.params);
    let sigma = params_full[k].exp();
    let params = params_full[..k].to_vec();

    let hessian_std = Mat::from_fn(k + 1, k + 1, |i, j| output.hessian[i][j]);
    // `problem_for_scores.as_ref().expect(...)`は各非`Classical`分岐でのみ呼ばれ、
    // 直前の`match cov_type`により常に`Some`であることが保証されている内部契約
    // （`LogitEstimator::fit`と同じ防御的な扱い）。
    let cov_params_std = match &cov_type {
        CovType::Classical => observed_information_cov_params(&hessian_std, k + 1)?,
        CovType::Opg => {
            let problem = problem_for_scores
                .as_ref()
                .expect("problem_for_scores must be Some for CovType::Opg");
            opg_cov_params(&problem.scores(&output.params), k + 1)?
        }
        CovType::Hc0 => {
            let problem = problem_for_scores
                .as_ref()
                .expect("problem_for_scores must be Some for CovType::Hc0");
            sandwich_cov_params(
                &hessian_std,
                &problem.scores(&output.params),
                n,
                k + 1,
                SandwichVariant::Hc0,
            )?
        }
        CovType::Hc1 => {
            let problem = problem_for_scores
                .as_ref()
                .expect("problem_for_scores must be Some for CovType::Hc1");
            sandwich_cov_params(
                &hessian_std,
                &problem.scores(&output.params),
                n,
                k + 1,
                SandwichVariant::Hc1,
            )?
        }
        CovType::Cluster { groups } => {
            let problem = problem_for_scores
                .as_ref()
                .expect("problem_for_scores must be Some for CovType::Cluster");
            // `groups`のNone・クラスター数不足の検証はfit()冒頭の
            // `validate_cluster_cov_type`で完了済み。
            // ここでの`expect`はその契約を明記する防御的な扱い（`LogitEstimator::fit`と同じ）。
            let groups = groups
                .as_ref()
                .expect("groups are validated by validate_cluster_cov_type for Cluster");
            cluster_cov_params(
                &hessian_std,
                &problem.scores(&output.params),
                n,
                k + 1,
                groups,
            )?
        }
    };
    // 標準化空間`(β̃, s̃)`の分散共分散行列を、元のスケール`(β, σ)`空間へ
    // `Cov(β,σ) = J·Cov(β̃,s̃)·Jᵀ`で変換する（`J`は`TobitScaling::param_jacobian`。
    // `x`の列標準化・センタリングの解除と、`s̃→σ=c·exp(s̃)`のデルタ法`∂σ/∂s̃=σ`を
    // 1本のヤコビアンに合成済み。`cov_type`の種類に依存しない共通の後処理、
    // `tobit.rs`の`TobitEstimator::fit`のdocコメント「標準化空間」「σのSE」節参照）。
    let cov_params = scaling.destandardize_cov_params(&cov_params_std, sigma);

    let normal = Normal::standard();
    let z_crit = inference::critical_value(&normal, confidence_level);

    let mut std_errors = vec![0.0; k + 1];
    let mut test_stats = vec![0.0; k + 1];
    let mut p_values = vec![0.0; k + 1];
    let mut conf_lower = vec![0.0; k + 1];
    let mut conf_upper = vec![0.0; k + 1];

    for j in 0..=k {
        let coef = if j < k { params[j] } else { sigma };
        let se = (*cov_params.get(j, j)).sqrt();
        let stat = inference::compute_inference_stat(&normal, coef, se, z_crit);

        std_errors[j] = se;
        test_stats[j] = stat.stat;
        p_values[j] = stat.p_value;
        conf_lower[j] = stat.conf_low;
        conf_upper[j] = stat.conf_high;
    }

    let llf = log_likelihood(
        input.x(),
        input.y(),
        input.lower(),
        input.upper(),
        &params_full,
    );
    // `σ`は推定されるパラメータなので、`aic`/`bic`の罰則と`df_resid`はそれを含む総パラメータ数
    // `k+1`を使う。一方`df_model`（下）はWald検定の対象となる傾き係数の数で、`σ`は検定対象
    // ではないため`k - k_constant`になる（両者は意図的に別の数え方。
    // `docs/spec/tobit-spec.md`「適合度統計量」参照）。
    let aic = -2.0 * llf + 2.0 * ((k + 1) as f64);
    let bic = -2.0 * llf + (n as f64).ln() * ((k + 1) as f64);
    let df_resid = n - (k + 1);

    let k_constant = usize::from(input.has_intercept());
    let df_model = k - k_constant;
    let (wald_statistic, wald_p_value) = if df_model == 0 {
        // 検定対象の傾き係数が存在しない（切片のみ、または切片も無いモデル）。
        // OLSの`f_p_value`・Logit/Probitの`lr_p_value`と同じ扱いでNaNを返す
        // （0除算・自由度0のカイ二乗分布の構築を避ける）。
        (f64::NAN, f64::NAN)
    } else {
        wald_chi2_test(&params, &cov_params, k_constant, df_model)?
    };

    Ok(TobitEstimator {
        fitted: TobitFitted {
            input,
            params,
            sigma,
            cov_params,
        },
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        log_likelihood: llf,
        aic,
        bic,
        n_obs: n,
        df_model,
        df_resid,
        wald_statistic,
        wald_p_value,
        converged: output.converged,
        n_iter: output.n_iter,
    })
}

#[cfg(test)]
mod tests {

    use super::super::problem::TobitProblem;
    use super::super::scaling::TobitScaling;

    use super::super::test_support::*;

    use crate::nonlinear::common::CovType;

    use crate::nonlinear::common::MleError;
    use crate::nonlinear::common::MleFitOptions;
    use crate::nonlinear::common::SandwichVariant;
    use crate::nonlinear::common::SolverType;

    use crate::nonlinear::common::cluster_cov_params;

    use crate::nonlinear::common::opg_cov_params;

    use crate::nonlinear::common::sandwich_cov_params;

    use crate::nonlinear::tobit::TobitEstimator;

    use crate::nonlinear::tobit::TobitInput;
    use crate::shared::error::ClusterSetting;
    use crate::shared::error::CommonError;
    use crate::shared::error::ErrorInfo;

    use crate::shared::group_codes::GroupCodes;

    use argmin::core::Hessian;
    use faer::Mat;

    use statrs::distribution::ChiSquared;

    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;

    /// 切片のみ（説明変数なし）・打ち切りなしのTobitは、通常の正規分布の最尤推定
    /// （`β̂=ȳ`・`σ̂²=Σ(y-ȳ)²/n`という閉じた形の解析解、OLSの不偏推定量`n-1`除算とは
    /// 異なる`n`除算のML推定量）に一致するはず（「打ち切りが極端に
    /// 少ないデータで、Newton法がOLSの閉形式解に近い値に収束すること」という完了条件の境界ケース）。
    #[test]
    fn fit_newton_converges_to_closed_form_solution_for_intercept_only_uncensored_data() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        let input = intercept_only_uncensored_input(&y);

        let n = y.len() as f64;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
        let expected_sigma = (sse / n).sqrt();

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

        assert!(estimator.converged());
        assert_eq!(estimator.params().len(), 1);
        assert!(
            (estimator.params()[0] - y_bar).abs() < 1e-5,
            "params={:?}, expected={}",
            estimator.params(),
            y_bar
        );
        assert!(
            (estimator.sigma() - expected_sigma).abs() < 1e-5,
            "sigma={}, expected={}",
            estimator.sigma(),
            expected_sigma
        );
    }

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
            (estimator.conf_lower()[1] - (estimator.sigma() - z_crit * expected_se_sigma)).abs()
                < 1e-6
        );
        assert!(
            (estimator.conf_upper()[1] - (estimator.sigma() + z_crit * expected_se_sigma)).abs()
                < 1e-6
        );
    }

    /// 切片のみ・打ち切りなしのTobit（通常の正規分布のMLEに一致する）の対数尤度は
    /// 閉じた形の解析解`ℓ=-(n/2)(ln(2π)+ln(σ̂²)+1)`を持つ（OLSの`log_likelihood`と同じ式。
    /// `σ̂²=Σ(y-ȳ)²/n`はMLE推定量で、OLSの不偏推定量（`n-1`除算）とは異なる）。
    /// `aic`/`bic`はこの`llf`と総パラメータ数`k+1=2`（切片1+`logσ`1）から直接計算できる。
    /// 切片のみモデルは傾き係数が無い（`df_model=0`）ため、Wald検定がスキップされ
    /// `wald_statistic`/`wald_p_value`がNaNになることも合わせて検証する。
    #[test]
    fn fit_computes_log_likelihood_aic_bic_matching_closed_form_for_intercept_only_uncensored_model()
     {
        let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        let input = intercept_only_uncensored_input(&y);

        let n = y.len() as f64;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
        let sigma_sq_hat = sse / n;
        let expected_llf =
            -(n / 2.0) * ((2.0 * std::f64::consts::PI).ln() + sigma_sq_hat.ln() + 1.0);
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

    /// `df_model=1`（傾き係数が1個）のとき、`W=z²`（カイ二乗(1)統計量は標準正規のz値の
    /// 2乗に一致するという代数的恒等式）が成り立つことを検証する。`fit_wald_statistic_
    /// and_p_value_match_independently_recomputed_values`（`q=2`、手動2×2逆行列）とは
    /// 別角度の検算（OLSの`wald_test_last_columns_matches_squared_t_statistic_for_
    /// single_column`と同じ発想）。`censored_regression_input`（`q=1`）を使う。
    #[test]
    fn fit_wald_statistic_matches_squared_z_statistic_for_single_slope() {
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

        assert_eq!(estimator.df_model(), 1);
        let z_slope = estimator.test_stats()[1];
        assert!((estimator.wald_statistic() - z_slope.powi(2)).abs() < 1e-9);

        let chi2 = ChiSquared::new(1.0).unwrap();
        let expected_p = 1.0 - chi2.cdf(estimator.wald_statistic());
        assert!((estimator.wald_p_value() - expected_p).abs() < 1e-9);
    }

    /// `cov_type=Cluster`のクラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`は、クラスター寄与
    /// スコアの総和がゼロ（MLEの一次条件`Σ_i s_i = 0`）になるため`rank(Ŝ) ≤ G - 1`。
    /// `multivariate_censored_input`（傾き係数`q=2`）に対し`G=2`（`G=q`ちょうど）を
    /// 組み合わせると`q×q`部分行列は`rank ≤ 1 < 2`で構造的に特異になる。`G`・`q`は
    /// 入力だけから判定できるため、`fit()`冒頭のバリデーション
    /// （`validate_cluster_cov_type` → `validate_cluster_count_covers_slopes`）が
    /// `CommonError::InsufficientClustersForInference`で弾く（
    /// 検証が無ければ`wald_chi2_test`内の`NearSingularMatrix`になる）。`wald_chi2_test`の
    /// `ensure_well_conditioned_symmetric_matrix`側のbackstop（`g > q`だが悪条件で
    /// 数値的にほぼ特異なケース）は、`wald_f_test`と共有する純粋な線形代数
    /// ユーティリティであり`shared/linear_algebra.rs`の単体テスト・OLSの
    /// `fit_returns_computation_failed_for_extreme_scale_difference_in_f_test`で担保する
    /// （Tobitは`TobitScaling`が`x`を内部で列標準化するため、極端なスケール差による
    /// 悪条件をTobitの`fit()`経由で再現するのが困難で、Tobit固有テストは設けない）。
    #[test]
    fn fit_returns_validation_error_when_cluster_count_equals_slopes() {
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let result = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
        );
    }

    /// 多変量（説明変数が1つ、切片込みで`k=2`・`s`を含め合計3パラメータ）の場合、
    /// 閉じた形の解析解は無いため、`cov_params`の対称性・各種統計量の内部整合性
    /// （z値・信頼区間の定義式通りの関係）を検証する回帰テスト。Logitの
    /// `fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ設計。
    /// 特に`TobitScaling::destandardize_cov_params`（`J·Cov·Jᵀ`）の適用が非対角成分も
    /// 含めて正しく機能しているかを確認する（対角成分だけでは転置ミス等の一部のバグを
    /// 検出できない）。
    #[test]
    fn fit_cov_params_is_symmetric_and_stats_are_internally_consistent() {
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

        let k_plus_1 = estimator.params().len() + 1;
        for i in 0..k_plus_1 {
            for j in 0..k_plus_1 {
                assert!(
                    (*estimator.cov_params().get(i, j) - *estimator.cov_params().get(j, i)).abs()
                        < 1e-9,
                    "cov_params not symmetric at ({i},{j})"
                );
            }
            assert!(
                *estimator.cov_params().get(i, i) > 0.0,
                "diagonal[{i}] <= 0"
            );
        }

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal.inverse_cdf(0.975);
        let mut coefs = estimator.params().to_vec();
        coefs.push(estimator.sigma());

        for (j, &coef) in coefs.iter().enumerate().take(k_plus_1) {
            let se = estimator.std_errors()[j];
            assert!((se - estimator.cov_params().get(j, j).sqrt()).abs() < 1e-9);
            assert!((estimator.test_stats()[j] - coef / se).abs() < 1e-9);
            assert!(
                (estimator.p_values()[j]
                    - 2.0 * (1.0 - normal.cdf(estimator.test_stats()[j].abs())))
                .abs()
                    < 1e-9
            );
            assert!((estimator.conf_lower()[j] - (coef - z_crit * se)).abs() < 1e-9);
            assert!((estimator.conf_upper()[j] - (coef + z_crit * se)).abs() < 1e-9);
        }
    }

    /// `fit`は非収束（`raise_on_non_convergence=false`）でも常に収束点（打ち切り点）の
    /// Hessianから`cov_params`を計算しようとするため、その打ち切り点でHessianが
    /// 不定符号・特異な場合は`MleError::SingularHessian`が返る（Newtonの最適化過程
    /// 自体は成功したが、SE計算のための逆行列計算が失敗するケース。`ols_initial_params`
    /// 由来の`SingularDesignMatrix`（設計行列`X`自体の特異性、最適化前に発生）とは
    /// 異なるエラー経路であることに注意）。`censored_regression_input`で`max_iter=1`
    /// にすると、Newton初回ステップ後の打ち切り点で実際にこれが発生することを実測で
    /// 確認済み（`max_iter=2`以降は`cov_params`計算が安定する。真の収束は`max_iter=11`、
    /// `fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false`
    /// のコメント参照）。
    #[test]
    fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point() {
        let input = censored_regression_input();
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 1,
                tol: 1e-12,
                raise_on_non_convergence: false,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{result:?}"
        );
    }

    /// 上のテストは`cov_type=Classical`（`observed_information_cov_params`）のみで
    /// `SingularHessian`エラー伝播を検証しているが、`sandwich_cov_params`（`Hc0`/`Hc1`）も
    /// 内部で同じHessianの逆行列計算を行うため、同じ打ち切り点で同じエラーが伝播する
    /// はず。Logit/Probitの`fit_returns_singular_design_matrix_error_for_perfectly_
    /// collinear_design_matrix`（`solver`×`cov_type`を1テストに集約）と同じ
    /// 「cov_type分岐ごとのエラー伝播`?`」のギャップパターンを
    /// Tobitでも確認する（`cargo llvm-cov`で発覚）。
    #[test]
    fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point_with_hc0_and_hc1()
     {
        for cov_type in [CovType::Hc0, CovType::Hc1] {
            let result = TobitEstimator::fit(
                censored_regression_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 1,
                    tol: 1e-12,
                    raise_on_non_convergence: false,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            );
            assert!(
                matches!(result, Err(MleError::SingularHessian)),
                "cov_type={cov_type:?}, result={result:?}"
            );
        }
    }

    /// `cluster_cov_params`（`Cluster`）も同じ理由で`SingularHessian`が伝播するはず
    /// （上記テストのdocコメント参照）。
    #[test]
    fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point_with_cluster()
     {
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let result = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 1,
                tol: 1e-12,
                raise_on_non_convergence: false,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{result:?}"
        );
    }

    /// 説明変数ありのモデルでも、打ち切りが実質発生しないデータではNewton法がOLSの
    /// 閉じた形の解（正規方程式）に近い値に収束するはず（完了条件の本体、
    /// `expected_*`はOLSの公式から本テスト内で独立に計算する）。
    #[test]
    fn fit_newton_converges_near_ols_closed_form_when_censoring_is_negligible() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
        let input = TobitInput::from_columns(
            &y,
            std::slice::from_ref(&x),
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(-1000.0),
            Some(1000.0),
        )
        .unwrap();

        let n = y.len() as f64;
        let x_bar: f64 = x.iter().sum::<f64>() / n;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sxy: f64 = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| (xi - x_bar) * (yi - y_bar))
            .sum();
        let sxx: f64 = x.iter().map(|xi| (xi - x_bar).powi(2)).sum();
        let expected_slope = sxy / sxx;
        let expected_intercept = y_bar - expected_slope * x_bar;
        let expected_sse: f64 = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| (yi - (expected_intercept + expected_slope * xi)).powi(2))
            .sum();
        let expected_sigma = (expected_sse / n).sqrt();

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

        assert!(estimator.converged());
        assert!(
            (estimator.params()[0] - expected_intercept).abs() < 1e-4,
            "intercept={}, expected={}",
            estimator.params()[0],
            expected_intercept
        );
        assert!(
            (estimator.params()[1] - expected_slope).abs() < 1e-4,
            "slope={}, expected={}",
            estimator.params()[1],
            expected_slope
        );
        assert!(
            (estimator.sigma() - expected_sigma).abs() < 1e-4,
            "sigma={}, expected={}",
            estimator.sigma(),
            expected_sigma
        );
    }

    #[test]
    fn fit_returns_invalid_confidence_level_error_out_of_range() {
        let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 1.5,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    #[test]
    fn fit_returns_invalid_max_iter_error() {
        let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 0,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidMaxIter { max_iter: 0 }
        );
    }

    #[test]
    fn fit_returns_invalid_tol_error() {
        let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 0.0,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(result.unwrap_err(), MleError::InvalidTol { tol: 0.0 });
    }

    #[test]
    fn fit_returns_insufficient_observations_error() {
        // n=2, 総パラメータ数k+1=2(切片1+logσ1) → n<=k+1でエラー
        // （`validate_sufficient_observations`にx列数ではなくk+1を渡す設計）。
        let input = intercept_only_uncensored_input(&[1.0, 2.0]);
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        let err = result.unwrap_err();
        assert_eq!(
            err,
            MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
        // `n_params`は`β`の数に`log σ`を足した値になる。
        let report = err.report();
        assert_eq!(
            report.message(),
            "Not enough observations: 2 observation(s) for 2 parameter(s); at least 3 are needed."
        );
    }

    #[test]
    fn fit_returns_singular_design_matrix_error_for_perfectly_collinear_data() {
        // x2=2*x1（完全な多重共線性）。ols_initial_paramsのQR分解が特異性を検出し、
        // Newton法が一度も反復する前にエラーを返す（SingularHessian
        // とは区別されるバリアント`SingularDesignMatrix`の検証）。
        // n=6 > 総パラメータ数k+1=4（切片1+x1+x2+logσ1）で`InsufficientObservations`を
        // 回避しつつ、x2=2*x1の構造的な多重共線性を検証する。
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_columns = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        ];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(-1000.0),
            Some(1000.0),
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularDesignMatrix { .. })),
            "{:?}",
            result
        );
    }

    #[test]
    fn fit_newton_converges_for_data_with_actual_censoring() {
        let input = censored_regression_input();
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

        assert!(estimator.converged());
        assert_eq!(estimator.params().len(), 2);
        assert!(estimator.sigma() > 0.0 && estimator.sigma().is_finite());
        for &p in estimator.params() {
            assert!(p.is_finite());
        }
        // `n_iter()`が他のどのテストでも未使用だった（`cargo llvm-cov`で発覚）。
        // 収束時は`0 < n_iter <= max_iter`のはず。
        assert!(estimator.n_iter() > 0 && estimator.n_iter() <= 100);
    }

    #[test]
    fn fit_bfgs_and_lbfgs_converge_to_similar_solution_as_newton_for_censored_data() {
        let newton = TobitEstimator::fit(
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

        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let estimator = TobitEstimator::fit(
                censored_regression_input(),
                MleFitOptions {
                    solver,
                    max_iter: 200,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            assert!(estimator.converged(), "solver={:?}", solver);
            for (a, b) in estimator.params().iter().zip(newton.params()) {
                assert!((a - b).abs() < 1e-3, "solver={:?}, a={a}, b={b}", solver);
            }
            assert!(
                (estimator.sigma() - newton.sigma()).abs() < 1e-3,
                "solver={:?}, sigma={}, newton_sigma={}",
                solver,
                estimator.sigma(),
                newton.sigma()
            );
        }
    }

    /// `censored_regression_input`の`x`（`1..8`）は`TobitScaling`の列標準化が実質no-opに
    /// なるほど自明なスケールではないが、`fit_bfgs_and_lbfgs_agree_with_newton_when_design_
    /// matrix_has_nontrivial_scale`（`logit/fit.rs`/`probit/fit.rs`）と同じ観点で、桁が大きく離れた
    /// スケール（`std`が1から大きく離れた値）でも標準化・逆標準化の往復が壊れないことを
    /// 明示的に検証する。`x`を100倍しつつ係数を1/100にスケールして同じ潜在変数
    /// `y*=-5+2x+noise`を再現しているため、期待される`β`・`σ`は`censored_regression_input`と
    /// 同一だが、ここではLogit/Probitの当該テストと同じくNewtonの結果を参照値とする
    /// クロスメソッド一致検証として書く（標準化空間での差異のみを見るのが目的のため）。
    #[test]
    fn fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale() {
        let x = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
        let make_input = || {
            TobitInput::from_columns(
                &y,
                std::slice::from_ref(&x),
                vec!["x1".to_string()],
                true,
                "y".to_string(),
                Some(0.0),
                None,
            )
            .unwrap()
        };

        let newton = TobitEstimator::fit(
            make_input(),
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
        assert!(newton.converged());

        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let estimator = TobitEstimator::fit(
                make_input(),
                MleFitOptions {
                    solver,
                    max_iter: 200,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            assert!(estimator.converged(), "solver={:?}", solver);
            for (a, b) in estimator.params().iter().zip(newton.params()) {
                assert!((a - b).abs() < 1e-3, "solver={:?}, a={a}, b={b}", solver);
            }
            assert!(
                (estimator.sigma() - newton.sigma()).abs() < 1e-3,
                "solver={:?}, sigma={}, newton_sigma={}",
                solver,
                estimator.sigma(),
                newton.sigma()
            );
        }
    }

    /// `y`（および打ち切り境界）を大きなスケールに引き伸ばしても、`fit()`が生スケールの
    /// データと数学的に同値な推定値に収束することを確認する回帰テスト。
    ///
    /// `y`を標準化せず最適化していた頃は、`y`のスケールが大きいと健全なMLE解でも標準化
    /// パラメータ空間のL2ノルムが大きくなり、当時あった標準化パラメータのノルムによる
    /// (準)分離ヒューリスティックが誤発火して失敗していた（Wooldridge mroz `hours`の
    /// 生スケールTobitで発覚）。`TobitScaling`が`y`を2の冪に丸めたスケールでスケーリング
    /// してから最適化し収束後に逆変換することで解消した（本テストは`TobitScaling`による
    /// スケール同値性の回帰テストとして維持する）。
    ///
    /// `y' = c·y`・`lower' = c·lower`（`c`は2の冪）とすると`β' = c·β`・`σ' = c·σ`・
    /// `SE' = c·SE`が厳密に成り立ち（`TobitScaling`が`y_scale`も`c`倍の2の冪に丸めるため、
    /// 両者は完全に同一の標準化問題を解く）、z統計量・p値・Wald統計量はスケール不変。
    #[test]
    fn fit_converges_for_large_response_scale_without_false_separation() {
        let c = 1024.0;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y_base = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
        let y_scaled: Vec<f64> = y_base.iter().map(|v| v * c).collect();

        let make = |y: &[f64], lower: f64| {
            TobitInput::from_columns(
                y,
                std::slice::from_ref(&x),
                vec!["x1".to_string()],
                true,
                "y".to_string(),
                Some(lower),
                None,
            )
            .unwrap()
        };
        let fit = |y: &[f64], lower: f64| {
            TobitEstimator::fit(
                make(y, lower),
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
        };

        let base = fit(&y_base, 0.0);
        let scaled = fit(&y_scaled, 0.0);

        assert!(scaled.converged());

        let rtol = |a: f64, b: f64| (a - b).abs() <= 1e-8 * b.abs().max(1.0);
        for (b, s) in base.params().iter().zip(scaled.params()) {
            assert!(rtol(s / c, *b), "beta: base={b}, scaled/c={}", s / c);
        }
        assert!(
            rtol(scaled.sigma() / c, base.sigma()),
            "sigma: base={}, scaled/c={}",
            base.sigma(),
            scaled.sigma() / c
        );
        for (b, s) in base.std_errors().iter().zip(scaled.std_errors()) {
            assert!(rtol(s / c, *b), "se: base={b}, scaled/c={}", s / c);
        }
        for (b, s) in base.test_stats().iter().zip(scaled.test_stats()) {
            assert!(rtol(*s, *b), "z: base={b}, scaled={s}");
        }
        assert!(
            rtol(scaled.wald_statistic(), base.wald_statistic()),
            "wald: base={}, scaled={}",
            base.wald_statistic(),
            scaled.wald_statistic()
        );
        // 対数尤度は変数変換のヤコビアン分ずれる（非打ち切り観測ごとに `-ln c`）。
        let n_uncensored = y_base.iter().filter(|&&v| v != 0.0).count() as f64;
        assert!(
            rtol(
                scaled.log_likelihood() + n_uncensored * c.ln(),
                base.log_likelihood()
            ),
            "llf: base={}, scaled+adj={}",
            base.log_likelihood(),
            scaled.log_likelihood() + n_uncensored * c.ln()
        );
    }

    /// 上のテストは切片あり・`c`が2の冪（＝標準化問題が両者で完全一致）だが、こちらは
    /// (a) **切片なし**モデル、(b) **`y`の標準偏差が2の冪ちょうどではない**大スケール、で
    /// `fit()`が既定Newtonで収束し真の傾き（生成係数）を回復することを確認する
    /// （`TobitScaling`の切片なし経路の end-to-end カバレッジと、`y_scale`が恒等変換で
    /// ない一般ケースの分離ヒューリスティック誤発火回避のリグレッションガード）。
    #[test]
    fn fit_converges_for_large_non_power_of_two_response_scale_without_intercept() {
        // 潜在変数 y* = 40·x（切片なし）、生成 σ ≈ 300 → y の std は数百オーダーで
        // 2の冪（256 / 512）ちょうどにはならない。左打ち切り 0。
        let x: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let mut rng_state = 0x1234_5678_u64;
        let mut next = || {
            // 決定的な xorshift でノイズを作る（テストの再現性、外部 rng 非依存）。
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 600.0
        };
        let y: Vec<f64> = x.iter().map(|&xi| (40.0 * xi + next()).max(0.0)).collect();
        let input = TobitInput::from_columns(
            &y,
            std::slice::from_ref(&x),
            vec!["x1".to_string()],
            false,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let est = TobitEstimator::fit(
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

        assert!(est.converged());
        // 傾きは真値 40 の近傍（打ち切り＋ノイズがあるため緩め）。以前は`y`の
        // 大スケールで当時のノルムによる分離ヒューリスティックが誤発火し`.unwrap()`が
        // panicしていた（現在はこのヒューリスティック自体が無い）。
        assert!(
            (est.params()[0] - 40.0).abs() < 5.0,
            "slope={}, expected≈40",
            est.params()[0]
        );
        assert!(est.sigma() > 0.0 && est.sigma().is_finite());
    }

    #[test]
    fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
        let input = censored_regression_input();
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 1,
                tol: 1e-12,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::NonConvergence { .. })),
            "{:?}",
            result
        );
    }

    /// Tobitの(準)完全分離は`MleError::SeparationDetected`ではなく
    /// `MleError::NonConvergence`として現れることを固定する。
    ///
    /// `fit()`はLogit/Probit用の分離検出（`nonlinear::separation`）を呼ばない。Tobitの分離は
    /// 係数の発散ではなく`σ→0`への退化として現れるため、この失敗モードが`NonConvergence`で
    /// あること自体を、Tobitの分離の現れ方の正本として固定する（`SeparationDetected`と
    /// `NonConvergence`はどちらも`ComputationError`にマップされるため、Python API境界の
    /// テストでは区別できない）。
    ///
    /// DGP: `y* = 100·x1 + 0.5·x2`（ノイズ皆無）を`lower=0`で左打ち切り。非打ち切り
    /// 観測は存在する（`NoUncensoredObservations`は通過する）が、分離方向に沿って
    /// `σ→0`へ退化し、Newtonは`max_iter`まで収束しない。
    #[test]
    fn fit_returns_non_convergence_not_separation_detected_for_noise_free_separation() {
        let n = 120;
        let x1: Vec<f64> = (0..n)
            .map(|i| -2.0 + 4.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let x2: Vec<f64> = (0..n).map(|i| (0.7 * i as f64).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (100.0 * x1[i] + 0.5 * x2[i]).max(0.0))
            .collect();
        // 非打ち切り観測が1件以上あることを前提とする（無ければ
        // `NoUncensoredObservations`が先に返り、このテストの意図がずれる）。
        assert!(y.iter().any(|&v| v > 0.0));

        let input = TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::NonConvergence { .. })),
            "expected NonConvergence, got {result:?}"
        );
    }

    #[test]
    fn fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false() {
        // `max_iter=1`（このテストの元々の値）だと、`censored_regression_input`の
        // 打ち切り点（Newtonの初回ステップ、まだ真の最尤推定点から遠い）でHessianが
        // 不定符号になり、`fit`が非収束時でも`cov_params`を計算するようになったことで
        // `SingularHessian`が先に発生してしまう（実測で確認、
        // `max_iter=1`は`SingularHessian`、`max_iter=2`以降で`cov_params`計算が
        // 安定し`converged=false`が返るようになる。真の収束は`max_iter=11`）。
        // `max_iter=3`のまま（`max_iter=2`でも通るが余裕を持たせる）、「非収束だが
        // cov_params計算は成功する」ケースを踏む。
        let input = censored_regression_input();
        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 3,
                tol: 1e-12,
                raise_on_non_convergence: false,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert!(!estimator.converged());
    }

    /// 打ち切りが重い（60観測中51観測＝85%が0で左打ち切り）合成データ。外部の乱数に依存しない
    /// 線形合同法で作る（`tests/support/_nonlinear_helpers.py`の`heavily_censored_tobit_dataset`と同じ手順・
    /// 同じ値。Python側の同名テストと同じ入力で挙動を固定する）。潜在変数は
    /// `y* = -1.8 + x1 + 0.5·x2 + e`（`x1`・`x2`・`e`は12個の一様乱数の和−6で近似した正規乱数）。
    fn heavily_censored_input() -> TobitInput {
        let mut state: u64 = 167;
        let mut unit = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut normal = move || (0..12).map(|_| unit()).sum::<f64>() - 6.0;
        let n = 60;
        // 引数の評価順（`x1`→`x2`→`e`）がPython側と同じになるよう、1つのクロージャを
        // 3回続けて回す。
        let x1: Vec<f64> = (0..n).map(|_| normal()).collect();
        let x2: Vec<f64> = (0..n).map(|_| normal()).collect();
        let e: Vec<f64> = (0..n).map(|_| normal()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (-1.8 + x1[i] + 0.5 * x2[i] + e[i]).max(0.0))
            .collect();
        assert_eq!(y.iter().filter(|&&v| v == 0.0).count(), 51);
        TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap()
    }

    fn heavily_censored_groups() -> Vec<String> {
        (0..60).map(|i| (i % 4).to_string()).collect()
    }

    fn truncated_fit_options(solver: SolverType, cov_type: CovType) -> MleFitOptions {
        MleFitOptions {
            solver,
            max_iter: 1,
            tol: 1e-12,
            raise_on_non_convergence: false,
            cov_type,
            confidence_level: 0.95,
        }
    }

    fn all_cov_types_for_heavily_censored_input() -> Vec<CovType> {
        let groups = heavily_censored_groups();
        vec![
            CovType::Classical,
            CovType::Opg,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        ]
    }

    /// `max_iter=1`・`raise_on_non_convergence=false`で打ち切った点は、重く打ち切られたデータでは
    /// `(β, log σ)`尤度が凹でない領域の近くにある。`Newton`（LM減衰で凹な領域に留まる）・`Bfgs`・`Lbfgs`
    /// （初期点のHessianが正定値ならその厳密な逆行列から出発するため、1手目がNewtonステップと一致）は、
    /// 全`cov_type`で未収束の結果を返す（Hessianの逆行列を使う`Classical`/`Hc0`/`Hc1`/`Cluster`でも
    /// `SingularHessian`にならない）。かつて`Lbfgs`は初期点のHessianを使わず1手目が勾配方向だったため、
    /// このデータでは凹でない領域（費用関数Hessianの最小固有値が約-0.6）に着地し、Hessianの逆行列を使う
    /// `cov_type`が`SingularHessian`になっていた。
    #[test]
    fn fit_returns_unconverged_result_for_every_solver_and_cov_type_at_truncated_point() {
        for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
            for cov_type in all_cov_types_for_heavily_censored_input() {
                let result = TobitEstimator::fit(
                    heavily_censored_input(),
                    truncated_fit_options(solver, cov_type.clone()),
                );
                let estimator = result
                    .unwrap_or_else(|e| panic!("solver={solver:?}, cov_type={cov_type:?}: {e:?}"));
                assert!(
                    !estimator.converged(),
                    "solver={solver:?}, cov_type={cov_type:?}"
                );
            }
        }
    }

    /// `fit()`と同じ手順（`TobitScaling`による`x`のセンタリング＋スケーリングと`y`の
    /// スケーリング→収束点でのscores/Hessian評価→`common.rs`の`cov_type`別の共通行列
    /// 演算→`TobitScaling::destandardize_cov_params`）をテスト側で独立に再現し、
    /// `cov_params_std_fn`が計算する標準化空間の分散共分散行列から最終的な`(β,σ)`空間の
    /// `cov_params`を得る。`Logit`の対応テストと同じ技法（`fit()`本体のロジックを
    /// 再利用せず、独立に再現することで配線ミスを検出する。`TobitScaling`の各メソッドは
    /// 元々の`standardize_columns`/`destandardize_cov_params`と同じく共有ヘルパー扱い）。
    fn expected_cov_params(
        input: &TobitInput,
        classical: &TobitEstimator,
        cov_params_std_fn: impl FnOnce(&Mat<f64>, &Mat<f64>) -> Mat<f64>,
    ) -> Mat<f64> {
        let k = input.k();
        let scaling = TobitScaling::fit(input.x(), input.y(), input.has_intercept());
        let x_std = scaling.standardize_x(input.x());
        let y_std = scaling.standardize_y(input.y());
        let lower_std = scaling.standardize_bound(input.lower());
        let upper_std = scaling.standardize_bound(input.upper());

        // 元のスケールの`(β, σ)`から標準化空間の内部パラメータ`(β̃, s̃=logσ̃)`を再構成する
        // （`destandardize_params`の逆写像。`TobitScaling`のdocコメントの式を反転）。
        let c = scaling.y_scale;
        // `β/c`を`ColumnScale::standardize_params`（`x`側の逆写像）で標準化空間へ戻す。
        let beta_over_c: Vec<f64> = classical.params()[..k].iter().map(|b| b / c).collect();
        let mut params_std = scaling.x.standardize_params(&beta_over_c);
        params_std.push(classical.sigma().ln() - c.ln());

        let problem_std = TobitProblem::from_standardized(x_std, y_std, lower_std, upper_std);
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k + 1, k + 1, |i, j| -cost_hessian_std[i][j]);

        scaling.destandardize_cov_params(
            &cov_params_std_fn(&hessian_std, &scores_std),
            classical.sigma(),
        )
    }

    fn assert_cov_params_close(actual: &Mat<f64>, expected: &Mat<f64>, k_plus_1: usize, tol: f64) {
        for i in 0..k_plus_1 {
            for j in 0..k_plus_1 {
                assert!(
                    (*actual.get(i, j) - *expected.get(i, j)).abs() < tol,
                    "({i},{j}): actual={}, expected={}",
                    *actual.get(i, j),
                    *expected.get(i, j)
                );
            }
        }
    }

    #[test]
    fn fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values() {
        let k_plus_1 = 4;
        let n = 8;

        let classical = TobitEstimator::fit(
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

        let expected_opg =
            expected_cov_params(&multivariate_censored_input(), &classical, |_h, s| {
                opg_cov_params(s, k_plus_1).unwrap()
            });
        let expected_hc0 =
            expected_cov_params(&multivariate_censored_input(), &classical, |h, s| {
                sandwich_cov_params(h, s, n, k_plus_1, SandwichVariant::Hc0).unwrap()
            });
        let expected_hc1 =
            expected_cov_params(&multivariate_censored_input(), &classical, |h, s| {
                sandwich_cov_params(h, s, n, k_plus_1, SandwichVariant::Hc1).unwrap()
            });

        let cases = [
            (CovType::Opg, &expected_opg),
            (CovType::Hc0, &expected_hc0),
            (CovType::Hc1, &expected_hc1),
        ];
        for (cov_type, expected) in cases {
            let estimator = TobitEstimator::fit(
                multivariate_censored_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            assert_cov_params_close(estimator.cov_params(), expected, k_plus_1, 1e-6);
        }
    }

    /// `solver`（`bfgs`/`lbfgs`）と`cov_type`（`Opg`/`Hc0`/`Hc1`）の組み合わせが正しく
    /// 機能することを確認する（Logitの`fit_non_classical_cov_types_work_with_bfgs_and_
    /// lbfgs`と同じ理由。`scores_std`の評価は収束点のパラメータにのみ依存し最適化
    /// アルゴリズムの種類に依存しない設計のため、`newton`で計算した`cov_params`
    /// （上のテストで既に正しさを検証済み）と`bfgs`/`lbfgs`の結果が一致するはず）。
    ///
    /// クラスターのグループ数は`G=4`（2件ずつ）にする。`multivariate_censored_input`は
    /// 傾き係数`q=2`（`x1`・`x2`、切片を除く）を持ち、Wald検定が`fit()`に
    /// 常時組み込まれたことで、クラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`の構造的な制約
    /// （`rank(Ŝ)≤G`、`engine/src/linear/CLAUDE.md`「クラスター数`G`と傾き係数の数`q`の
    /// 関係」参照）がWald検定の`q×q`部分行列にも及ぶことが判明した。`G=2`（`q`と同数）
    /// では実測でこの部分行列が特異になり`fit()`全体が`NearSingularMatrix`
    /// （`ComputationError`）になったため、
    /// `G>q`を満たす`G=4`に変更した（OLSの既存ガイドライン「境界の成功パスのテストでは
    /// `q`を`G`以下に保つ」をTobitでも踏襲）。
    #[test]
    fn fit_non_classical_cov_types_work_with_bfgs_and_lbfgs() {
        let k_plus_1 = 4;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "c".to_string(),
            "c".to_string(),
            "d".to_string(),
            "d".to_string(),
        ];

        for cov_type in [
            CovType::Opg,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        ] {
            let newton = TobitEstimator::fit(
                multivariate_censored_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            )
            .unwrap();

            for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
                let estimator = TobitEstimator::fit(
                    multivariate_censored_input(),
                    MleFitOptions {
                        solver,
                        max_iter: 300,
                        tol: 1e-8,
                        raise_on_non_convergence: true,
                        cov_type: cov_type.clone(),
                        confidence_level: 0.95,
                    },
                )
                .unwrap();
                assert!(estimator.converged(), "cov_type={cov_type:?}, {solver:?}");
                assert_cov_params_close(
                    estimator.cov_params(),
                    newton.cov_params(),
                    k_plus_1,
                    1e-4,
                );
            }
        }
    }

    /// `multivariate_censored_input`ではなく`censored_regression_input`（傾き係数
    /// `q=1`、`x1`のみ）を使う。Wald検定が`fit()`に常時組み込まれたことで、
    /// クラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`の構造的な制約（`rank(Ŝ)≤G`、
    /// `engine/src/linear/CLAUDE.md`「クラスター数`G`と傾き係数の数`q`の関係」参照）が
    /// Wald検定の`q×q`部分行列にも及ぶことが判明した。`multivariate_censored_input`
    /// （`q=2`）に対し`G=2`（G=2の境界値）を組み合わせると`q`と
    /// 同数になり、実測でこの部分行列が特異になり`fit()`全体が`NearSingularMatrix`
    /// （`ComputationError`）になった。`q=1`のデータセットなら`G=2>q=1`を満たしたまま「G=2の境界値」を検証できる
    /// ため、こちらに切り替えた（OLSの既存ガイドライン「境界の成功パスのテストでは`q`を
    /// `G`以下に保つ」をTobitでも踏襲）。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values() {
        let k_plus_1 = 3;
        let n = 8;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        let classical = TobitEstimator::fit(
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

        let groups_for_expected = groups.clone();
        let expected_cluster =
            expected_cov_params(&censored_regression_input(), &classical, |h, s| {
                cluster_cov_params(
                    h,
                    s,
                    n,
                    k_plus_1,
                    &crate::shared::group_codes::GroupCodes::from_labels(&groups_for_expected),
                )
                .unwrap()
            });

        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert_cov_params_close(estimator.cov_params(), &expected_cluster, k_plus_1, 1e-6);
    }

    /// 上のテストは4:4の均等サイズのグループのみを検証しているが、
    /// `testing-policy.md`が指摘する通り均等サイズのみのテストは実務で起こりやすい
    /// 偏った分布のグループサイズを見逃しうる。5:3の不均衡なグループ（G=2の境界値を
    /// 含む）でも同じ独立再計算の
    /// 技法で検証する（`fit_cov_type_cluster_matches_independently_recomputed_values`と
    /// 同じデータセット・同じ理由でq=1のデータセットを使う、グループ分割のみ変更）。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups() {
        let k_plus_1 = 3;
        let n = 8;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        let classical = TobitEstimator::fit(
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

        let groups_for_expected = groups.clone();
        let expected_cluster =
            expected_cov_params(&censored_regression_input(), &classical, |h, s| {
                cluster_cov_params(
                    h,
                    s,
                    n,
                    k_plus_1,
                    &crate::shared::group_codes::GroupCodes::from_labels(&groups_for_expected),
                )
                .unwrap()
            });

        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert_cov_params_close(estimator.cov_params(), &expected_cluster, k_plus_1, 1e-6);
    }

    #[test]
    fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
        let result = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster { groups: None },
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let groups = vec!["a".to_string(); 8];
        let result = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
        );
    }

    /// Logit（`fit_returns_singular_opg_matrix_error_for_perfectly_collinear_
    /// design_matrix`）はxの完全な多重共線性（x2=2*x1）でOPG行列`Σᵢsᵢsᵢ'`を特異にして
    /// いるが、Tobitでは`ols_initial_params`が最適化前にxのQRベースの階数検定を行うため
    /// （`fit_returns_singular_design_matrix_error_for_perfectly_collinear_data`参照）、
    /// xが完全に多重共線だと最適化に入る前に`SingularDesignMatrix`になり同じ手法は使えない。
    ///
    /// このテストのデータ（x1・x2は互いに無関係、多重共線性なし）は実測で見つけた
    /// 別の退化パターンで、`SingularOpgMatrix`の再現に多重共線性は必須ではないことが
    /// 分かっている（独立検証で、x2をx1と無関係な値に変えても収束点での
    /// OPG行列の最小固有値が機械精度オーダーまで落ちる同じ現象を確認済み）。原因は
    /// 観測数`n=6`が総パラメータ数`k+1=4`に対して少なく（`validate_sufficient_
    /// observations`の下限ぎりぎり）、かつ左打ち切り観測が2件という小標本特有の
    /// 収束点の数値配置に依存すると推測されるが、厳密な理論的必然性までは未解明
    /// （`n=7`など観測数を増やすと再現しなくなることを確認済み）。「なぜこの具体的な
    /// データで起こるか」を将来の保守者が誤って多重共線性のせいだと読み違えないよう、
    /// ここに正直に記録する。
    #[test]
    fn fit_returns_singular_opg_matrix_error_for_small_sample_degenerate_case() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85];
        let x2: Vec<f64> = vec![5.0, -3.0, 7.0, 1.0, -2.0, 9.0];
        let input = TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Opg,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularOpgMatrix)),
            "{result:?}"
        );
    }

    /// 全件が下限（`lower`）で打ち切られている（非打ち切り観測が1件も無い）場合は
    /// `MleError::NoUncensoredObservations`を返す。実測で確認済み: このバリデーションが
    /// 無いと`fit()`は`converged=true`のまま統計的に無意味な巨大SE（100万倍オーダー）を
    /// 返す退化収束を起こしていた（参照実装`survival::survreg`は同種のデータで
    /// エラーを返すことをdevcontainer内で実際に確認している）。
    #[test]
    fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_at_lower() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0; 8];
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::NoUncensoredObservations {
                lower: Some(0.0),
                upper: None
            }
        );
    }

    /// 上のテストは`lower`側のみの片側打ち切りを検証しているが、`at_lower`/`at_upper`の
    /// 判定ロジックが対称的に実装されていることを、`upper`側のみの片側打ち切りでも
    /// 独立に確認する。
    #[test]
    fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_at_upper() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![6.0; 8];
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            None,
            Some(6.0),
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::NoUncensoredObservations {
                lower: None,
                upper: Some(6.0)
            }
        );
    }

    /// 両側打ち切りで、非打ち切り観測が1件も無い（全観測が`lower`または`upper`ちょうど）
    /// 場合も同じエラーになることを確認する（`lower`側だけの判定に留まらないことの検証）。
    #[test]
    fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_two_sided() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0];
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(6.0),
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::NoUncensoredObservations {
                lower: Some(0.0),
                upper: Some(6.0)
            }
        );
    }

    /// 非打ち切り観測がちょうど1件ある場合は`NoUncensoredObservations`を返さない
    /// （「1件以上」という条件の境界値、`validate_sufficient_observations`と同じ
    /// `>=`境界の考え方）。このデータは病理的に近い（実際にNewtonは収束しない）ため
    /// `raise_on_non_convergence=false`で受け、`NonConvergence`ではなく
    /// バリデーション自体は通過することだけを確認する。
    #[test]
    fn fit_does_not_return_no_uncensored_observations_error_when_exactly_one_observation_is_uncensored()
     {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.95];
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: false,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            !matches!(result, Err(MleError::NoUncensoredObservations { .. })),
            "{result:?}"
        );
    }

    /// property-basedテスト。`logit/fit.rs`/`probit/fit.rs`の`mod proptests`と同型の設計だが、
    /// Tobitは打ち切り回帰でありLogit/Probitと異なり(a)説明変数の係数`β`に加え誤差項の
    /// 標準偏差`σ`（`s=logσ`でパラメータ化）を持つ、(b)観測ごとの尤度が非打ち切り/左
    /// 打ち切り/右打ち切りの3分岐になる、という構造上の違いがある。そのため
    /// `score_is_near_zero_at_converged_params`は3分岐+σパラメータ化のFOCを独立に
    /// 再導出せず、`TobitProblem::scores()`（本番のOPG/サンドイッチSE計算が使うのと
    /// 同じ観測ごとのスコア行列）を再利用する（Probitの
    /// `clamped_pdf_cdf`共有と同種のトレードオフで、`scores()`自体のバグはこの
    /// プロパティでは検出できない）。
    ///
    /// **単なる重複ではない**: `fit()`は`TobitScaling`で標準化した空間で最適化し、
    /// `est.params()`/`est.sigma()`は逆変換（destandardization）後の値。このプロパティは
    /// `TobitProblem::new(est.input())`（生スケール）に`est.params()`/`est.sigma().ln()`を
    /// 渡してFOCを評価するため、多数のランダム構成で標準化⇔生スケールの逆変換自体を
    /// 実質的に検証する（既存の固定テスト`gradient_matches_numerical_differentiation_of_
    /// cost`/`scores_sum_to_negative_gradient`はいずれも`TobitScaling`を経由しないため、
    /// この往復変換を検証できていなかった）。標準化パラメータ空間でスコアが0なら生
    /// スケールでもスコアが0になることは、標準化がθに依存しないアフィン変換であり
    /// 対数尤度の差が定数項（θ非依存のヤコビアン項）のみであることから理論的に導ける。
    ///
    /// ケース生成は左打ち切り（`lower=0`のみ）・右打ち切り（`upper=0`のみ）・両側打ち切り
    /// （`lower=-1`・`upper=1`）の3通りを`censoring_strategy`でランダムに選ぶ
    /// （`censored_contribution`の`direction=-1.0`分岐や左右混在のデータセットも通る）。
    /// `y* = β₀ + Σxⱼβⱼ + σε`（`ε`は標準正規、`u∈(1e-6, 1-1e-6)`の逆CDFでサンプリング）
    /// を計算し、下限未満・上限超過の観測をそれぞれの境界値で打ち切る。
    ///
    /// 較正値（`n=k+30..=80`, `beta∈[-1,1]`, `sigma∈[0.5,2]`）は`PROPTEST_CASES=5000`
    /// （デフォルト256の約20倍）まで増やしても`NoUncensoredObservations`等による
    /// `prop_assume`棄却が原因の失敗（"too many global rejects"）を起こさないことを
    /// 実測済み。
    ///
    /// プロパティの有効性検証（バグ注入→検出確認→元に戻す）は当初の3件について実施済み
    /// （4件目の`negating_y_and_swapping_bounds_negates_coefficients_only`は右打ち切り・
    /// 両側打ち切りへの拡張と同時に追加し、右打ち切り分岐へのバグ注入で検出を確認した）:
    /// `score_is_near_zero_at_converged_params`は`TobitProblem::gradient`の`grad[j]`
    /// 計算に定数オフセットを注入、`coefficients_and_se_are_invariant_to_column_order`は
    /// `TobitInput::from_columns`の`param_names`を逆順にするバグを注入、
    /// `hc0_std_errors_are_at_most_hc1_std_errors`は`nonlinear::common::
    /// sandwich_cov_params`のHC1補正係数を反転（`n/(n-k)`→`(n-k)/n`、Logit/Probitと
    /// 共有するロジックのため同一のバグ注入で確認）するバグを注入し、いずれも検出
    /// できることを確認した。
    ///
    /// `ExpectedUncensored`の3件（`decomposition_identity_holds_for_random_inputs`・
    /// `expected_uncensored_stays_within_bounds_and_w_in_unit_interval`・
    /// `mirror_symmetry_negates_mean_and_derivative_and_keeps_weight`）は`fit()`を介さず
    /// `predicted_value`・`target_w_and_s`を乱数の`(μ,σ,境界)`で直接検証する。バグ注入で
    /// 有効性を確認済み: 下側の裾の鏡像で`(Pa,Pb)`・`(A,B)`の入れ替えを外す→3件とも失敗、
    /// 平均を`μ+σ(Pa+Pb)`にする→3件とも失敗、`∂w/∂μ`の符号を誤る→鏡像対称性が失敗。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        // 高kはbenchmarkのmany_regressorsシナリオでカバー済みのため小さく保つ
        // （logit/fit.rs/probit/fit.rsと同じ理由）。
        const MAX_K: usize = 3;

        /// `tobit_case_strategy`が生成するタプル: `(n, k, x_cols, beta, sigma, u, keys)`。
        /// `k`は切片を除いた説明変数の数（`beta`は切片込みで`k+1`要素）。
        type TobitCase = (
            usize,
            usize,
            Vec<Vec<f64>>,
            Vec<f64>,
            f64,
            Vec<f64>,
            Vec<u64>,
        );

        fn tobit_case_strategy() -> impl Strategy<Value = TobitCase> {
            (1..=MAX_K).prop_flat_map(|k| {
                (k + 30..=80usize).prop_flat_map(move |n| {
                    (
                        Just(n),
                        Just(k),
                        collection::vec(collection::vec(-2.0f64..2.0, n), k),
                        collection::vec(-1.0f64..1.0, k + 1),
                        0.5f64..2.0,
                        collection::vec(1e-6f64..1.0 - 1e-6, n),
                        collection::vec(any::<u64>(), k),
                    )
                })
            })
        }

        fn x_names(k: usize) -> Vec<String> {
            (1..=k).map(|i| format!("x{i}")).collect()
        }

        /// 左打ち切りの下限（打ち切りなしのTobitの意味が薄れない程度に、真の`y*`分布の
        /// 中心付近を狙う）。右打ち切りの上限はこの値と対称（`beta`の分布が0周りに対称な
        /// ため、左・右で打ち切り割合が同程度になる）。
        const LOWER_BOUND: f64 = 0.0;

        /// 打ち切り境界`(lower, upper)`。
        type Bounds = (Option<f64>, Option<f64>);

        /// 左打ち切り・右打ち切り・両側打ち切りをランダムに選ぶ。両側打ち切りの幅
        /// （`[-1, 1]`）は、`NoUncensoredObservations`等の`prop_assume`棄却が過剰にならない
        /// 程度に真の`y*`分布の中心付近を含める。
        fn censoring_strategy() -> impl Strategy<Value = Bounds> {
            prop_oneof![
                Just((Some(LOWER_BOUND), None)),
                Just((None, Some(LOWER_BOUND))),
                Just((Some(-1.0), Some(1.0))),
            ]
        }

        /// 真の`beta`・`sigma`から`y* = β₀+Σxⱼβⱼ+σε`を計算し、`bounds`で打ち切る
        /// （下限未満は下限、上限超過は上限に置き換える）。
        fn simulate_y(
            n: usize,
            x_cols: &[Vec<f64>],
            beta: &[f64],
            sigma: f64,
            u: &[f64],
            bounds: Bounds,
        ) -> Vec<f64> {
            let normal = Normal::standard();
            (0..n)
                .map(|i| {
                    let mut y_star = beta[0];
                    for (j, x_col) in x_cols.iter().enumerate() {
                        y_star += x_col[i] * beta[j + 1];
                    }
                    y_star += sigma * normal.inverse_cdf(u[i]);
                    if let Some(lower) = bounds.0
                        && y_star < lower
                    {
                        return lower;
                    }
                    if let Some(upper) = bounds.1
                        && y_star > upper
                    {
                        return upper;
                    }
                    y_star
                })
                .collect()
        }

        /// `solver`ごとの`tol`既定値の解決は`logit/fit.rs`/`probit/fit.rs`の`default_options`と
        /// 同じ（`docs/spec/tobit-spec.md`、Logit/Probitと共有する`run_solver`のため
        /// 同じ意味論）。
        fn default_options(cov_type: CovType, solver: SolverType) -> MleFitOptions {
            let tol = match solver {
                SolverType::Newton => 1e-6,
                SolverType::Bfgs | SolverType::Lbfgs => 1e-8,
            };
            MleFitOptions {
                solver,
                max_iter: 50,
                tol,
                raise_on_non_convergence: true,
                cov_type,
                confidence_level: 0.95,
            }
        }

        fn solver_strategy() -> impl Strategy<Value = SolverType> {
            prop_oneof![
                Just(SolverType::Newton),
                Just(SolverType::Bfgs),
                Just(SolverType::Lbfgs),
            ]
        }

        /// `logit/fit.rs`/`probit/fit.rs`と同じ理由で`1e-6`より緩い`1e-4`を使う
        /// （反復最適化の収束経路が列順序で変わりうるため）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-4 * expected.abs().max(1.0);
            let diff = (actual - expected).abs();
            assert!(
                diff <= tol,
                "{msg}: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
            );
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// MLEの一次条件（スコア方程式）: 収束点で`Σᵢsᵢ ≈ 0`（`sᵢ`は
            /// `TobitProblem::scores()`が返す観測ごとのスコア行、`β`の`k+1`列＋
            /// `s=logσ`の1列、計`k+2`列）が全パラメータ列で成り立つ。許容誤差の根拠は
            /// `logit/fit.rs`の同名プロパティのdocコメント参照（`n`非依存の絶対閾値）。
            #[test]
            fn score_is_near_zero_at_converged_params(
                (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
                bounds in censoring_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
                let names = x_names(k);
                let input = TobitInput::from_columns(
                    &y, &x_cols, names, true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result = TobitEstimator::fit(input, default_options(CovType::Classical, solver));
                prop_assume!(result.is_ok());
                let est = result.unwrap();

                let mut full_params: Vec<f64> = est.params().to_vec();
                full_params.push(est.sigma().ln());

                let problem = TobitProblem::new(est.input());
                let scores = problem.scores(&full_params);
                for j in 0..=(k + 1) {
                    let score: f64 = (0..n).map(|i| *scores.get(i, j)).sum();
                    prop_assert!(
                        score.abs() <= 1e-4,
                        "score[{j}] should be ~0, got {score} (solver={solver:?})"
                    );
                }
            }

            /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
            /// `σ`は列順序に依存しない別パラメータのため、`std_errors()`の末尾要素
            /// （インデックス`k+1`）で個別に比較する。
            #[test]
            fn coefficients_and_se_are_invariant_to_column_order(
                (n, k, x_cols, beta, sigma, u, keys) in tobit_case_strategy()
                    .prop_filter("need >=2 columns to permute", |(_, k, _, _, _, _, _)| *k >= 2),
                bounds in censoring_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
                let names = x_names(k);
                let input1 = TobitInput::from_columns(
                    &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result1 = TobitEstimator::fit(input1, default_options(CovType::Classical, solver));
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let mut order: Vec<usize> = (0..k).collect();
                order.sort_by_key(|&i| keys[i]);
                let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
                let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

                let input2 = TobitInput::from_columns(
                    &y, &permuted_x, permuted_names, true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result2 = TobitEstimator::fit(input2, default_options(CovType::Classical, solver));
                prop_assume!(result2.is_ok());
                let est2 = result2.unwrap();

                let names1 = est1.input().param_names().to_vec();
                let names2 = est2.input().param_names().to_vec();
                let (params1, params2) = (est1.params(), est2.params());
                let (se1, se2) = (est1.std_errors(), est2.std_errors());
                for (idx1, name) in names1.iter().enumerate() {
                    let idx2 = names2.iter().position(|n| n == name)
                        .expect("name should exist in permuted result");
                    assert_approx_eq(params2[idx2], params1[idx1], &format!("param[{name}] under column permutation"));
                    assert_approx_eq(se2[idx2], se1[idx1], &format!("std_error[{name}] under column permutation"));
                }
                assert_approx_eq(est2.sigma(), est1.sigma(), "sigma under column permutation");
                assert_approx_eq(se2[k + 1], se1[k + 1], "std_error[sigma] under column permutation");
            }

            /// HC0の標準誤差は常にHC1以下（`docs/spec/tobit-spec.md`）。
            #[test]
            fn hc0_std_errors_are_at_most_hc1_std_errors(
                (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
                bounds in censoring_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
                let names = x_names(k);
                let input1 = TobitInput::from_columns(
                    &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result1 = TobitEstimator::fit(input1, default_options(CovType::Hc0, solver));
                prop_assume!(result1.is_ok());
                let est_hc0 = result1.unwrap();

                let input2 = TobitInput::from_columns(
                    &y, &x_cols, names, true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result2 = TobitEstimator::fit(input2, default_options(CovType::Hc1, solver));
                prop_assume!(result2.is_ok());
                let est_hc1 = result2.unwrap();

                for j in 0..=(k + 1) {
                    prop_assert!(
                        est_hc0.std_errors()[j] <= est_hc1.std_errors()[j] + 1e-9,
                        "HC0 se[{j}]={} should be <= HC1 se[{j}]={}",
                        est_hc0.std_errors()[j], est_hc1.std_errors()[j]
                    );
                }
            }

            /// 鏡像対称性: `y`と境界を符号反転して`(lower, upper)`→`(-upper, -lower)`にすると
            /// （`-y* = x(-β) + σ(-ε)`、`-ε`も標準正規）、係数は符号反転・`σ`と標準誤差は不変。
            /// 左打ち切りと右打ち切りで別々のコード分岐（`censored_contribution`の
            /// `direction`）を通るため、片方の分岐の符号・スケールのバグがここで顕在化する。
            #[test]
            fn negating_y_and_swapping_bounds_negates_coefficients_only(
                (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
                bounds in censoring_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
                let names = x_names(k);
                let input1 = TobitInput::from_columns(
                    &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result1 = TobitEstimator::fit(input1, default_options(CovType::Classical, solver));
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let y_neg: Vec<f64> = y.iter().map(|v| -v).collect();
                let input2 = TobitInput::from_columns(
                    &y_neg, &x_cols, names, true, "y".to_string(),
                    bounds.1.map(|v| -v), bounds.0.map(|v| -v),
                ).unwrap();
                let result2 = TobitEstimator::fit(input2, default_options(CovType::Classical, solver));
                prop_assume!(result2.is_ok());
                let est2 = result2.unwrap();

                for j in 0..=k {
                    assert_approx_eq(est2.params()[j], -est1.params()[j], &format!("param[{j}] under y negation"));
                }
                assert_approx_eq(est2.sigma(), est1.sigma(), "sigma under y negation");
                for j in 0..=(k + 1) {
                    assert_approx_eq(est2.std_errors()[j], est1.std_errors()[j], &format!("std_error[{j}] under y negation"));
                }
            }
        }

        /// 大規模ランダム探索で捕捉した具体的な入力の1つを固定値化した回帰テスト。
        ///
        /// **調査方法**: 評価回数バジェット方式の導入後は、line
        /// searchの暴走が数秒程度の有限時間で`Err`に変換されるようになったため、
        /// `kill`前提の外部タイムアウト無しで大量試行を安全に回せることを利用し、
        /// `tobit_case_strategy()`・`solver_strategy()`を`TestRunner`で直接
        /// サンプリングして30万回試行する使い捨ての探索ハーネスを一時的に実装し
        /// 実行した（コミット履歴には残さず、本テストのみを結果として残す）。
        ///
        /// **判明した事実（当時）**: 30万試行中8件が`MleError::EvaluationBudgetExceeded`
        /// 相当のエラー（`SolverType::Lbfgs`は当時のargmin組み込みLBFGSの`SolverExit`経由で
        /// 同じメッセージを含む計算エラーになっていた）を引き起こした。**8件
        /// 全てが`solver=Lbfgs`**で、`solver=Bfgs`・`solver=Newton`は1件もヒットしな
        /// かった。打ち切り割合（`n_uncensored/n`）は0.38〜0.80、`n`は41〜75、`k`は
        /// 1〜3、`sigma`は0.53〜1.85とヒット全体に幅広く分布しており、「極端な打ち切り・
        /// 特定の狭いパラメータ域が必要」という仮説は支持されなかった（8件中7件は
        /// 0.65〜0.80という中程度の打ち切り）。真因は当時未確定で、argmin組み込み
        /// `LBFGS`固有の何か（limited-memory two-loop recursionの初期スケーリング等）に
        /// 起因すると推測されていたが、真因の特定は後続の調査に持ち越された。
        ///
        /// **判明した実際の原因と解消（本テストの現在の主眼）**: `SolverType::Lbfgs`
        /// を`FaerLbfgs`（`FaerBfgs`と同型のself-scaling初期化を持つ自前実装）に置き換えた
        /// 結果、この入力は**もはや退化せず正常に収束する**（実測: `n_iter=11`、
        /// `converged=true`）。`FaerBfgs`で既に確立していた「1回目の
        /// 反復専用のline search初期ステップ幅`min(1,1/‖g₀‖)`」を、当時のargmin組み込み
        /// `LBFGS`の公開APIでは適用できていなかったこと（1回目の反復は履歴が空で
        /// `γ=1.0`固定のため、単位行列の逆Hessianを使う`FaerBfgs`の1回目と同型の
        /// 暴走リスクを抱えたままだった）が実際の原因だったと裏付けられた。本テストは
        /// 「境界の有限時間`Err`」から「実際に収束する」への挙動変化そのものを固定する
        /// 回帰ガードとして残す。
        #[test]
        fn fit_lbfgs_converges_for_a_previously_stalling_case() {
            let beta = vec![0.9873339499036579, -0.12291865368107788];
            let sigma = 1.4304964761886263;
            let x_cols: Vec<Vec<f64>> = vec![vec![
                0.09730587261408116,
                -0.44608585527013445,
                -1.1780876932682702,
                -1.7218664313287149,
                -1.5668563688839525,
                1.569383650134455,
                -0.8904079788143034,
                -0.38220417882649177,
                -0.6650396911368611,
                1.5455987143552725,
                1.0914058868351664,
                1.5608174706710314,
                -1.0457715928011817,
                0.8460357774450576,
                -1.6746201868364683,
                -0.10662833055410269,
                -1.3549266627206265,
                1.802095632400922,
                0.22454103534302314,
                0.9226514870462461,
                -1.5960509851353555,
                -0.06299762418526257,
                -0.7361961389207361,
                -0.6159276360973034,
                1.6276092545963188,
                1.8777781260428934,
                1.4040046871028768,
                1.6191183983983046,
                0.7928232369566655,
                -1.2215837625042971,
                0.43098012504668753,
                0.6437939214254764,
                1.8726228041330524,
                0.18531824642137915,
                -1.2889799350480626,
                1.3893942618908828,
                0.9546063410017254,
                1.6709682203826888,
                -0.6385965449169932,
                1.801002621573055,
                1.89685718642964,
            ]];
            let u = vec![
                0.3617740489249382,
                0.8638655369835531,
                0.5854529043833009,
                0.17096769280771643,
                0.27549992513298815,
                0.29109304112930307,
                0.23572824129100706,
                0.6413758528582494,
                0.015031380540545333,
                0.9591408659161148,
                0.03755478548383856,
                0.5531593343087735,
                0.00290634638539887,
                0.6500237097120347,
                0.41895952824744176,
                0.6246296073538906,
                0.672262276485462,
                0.072963646484062,
                0.8961814455371079,
                0.37114140842146587,
                0.07134975672937831,
                0.42084724899948256,
                0.612656348188821,
                0.03682767217809841,
                0.7090082364949691,
                0.30669875984016237,
                0.9155739980338565,
                0.022268595803044702,
                0.9400939907877396,
                0.922790135057218,
                0.05561252335283864,
                0.9567466460014393,
                0.1117763895368753,
                0.16362361406720183,
                0.41560137699383287,
                0.08196797976110878,
                0.25881610622077167,
                0.33012720695491854,
                0.6634215047000254,
                0.23962574010778723,
                0.9444306832294833,
            ];
            let n = 41;

            let y = simulate_y(n, &x_cols, &beta, sigma, &u, (Some(LOWER_BOUND), None));
            let names = x_names(1);
            let input = TobitInput::from_columns(
                &y,
                &x_cols,
                names,
                true,
                "y".to_string(),
                Some(LOWER_BOUND),
                None,
            )
            .unwrap();

            let start = std::time::Instant::now();
            let result = TobitEstimator::fit(
                input,
                default_options(CovType::Classical, SolverType::Lbfgs),
            );
            let elapsed = start.elapsed();

            assert!(
                result.is_ok(),
                "expected FaerLbfgs's self-scaling to converge on this input, got {result:?}"
            );
            assert!(
                elapsed < std::time::Duration::from_secs(30),
                "fit() took {elapsed:?}, expected a bounded run within seconds"
            );
            // 実測`n_iter=11`（緩い上限）。`FaerLbfgs`の1回目line
            // search初期ステップ幅の設計変更（`engine/src/nonlinear/CLAUDE.md`
            // 「実装時に踏んだ罠その2」参照）時に、収束はするが反復回数が
            // 大きく増える形の劣化が紛れ込んでいないかを検知するためのガード。
            let n_iter = result.unwrap().n_iter();
            assert!(
                n_iter < 30,
                "n_iter={n_iter}, expected roughly the same order as the observed 11 \
                 iterations (a large increase would indicate a regression in FaerLbfgs's \
                 first-iteration self-scaling, even though it still converges)"
            );
        }
    }
}
