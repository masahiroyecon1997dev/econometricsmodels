//! `LogitEstimator::fit`の本体。検証 → 標準化 → 初期値 → 最適化 → 分離検出 → `cov_type`別の
//! 共分散 → 推論統計量 → 適合度統計量の組み立て。

use super::LogitEstimator;
use super::fitted::LogitFitted;
use super::input::LogitInput;
use super::model::log_likelihood;
use super::problem::LogitProblem;
use crate::nonlinear::common::{
    CovType, GoodnessOfFit, MleError, MleFitOptions, SandwichVariant, cluster_cov_params,
    destandardize_cov_params, destandardize_params, goodness_of_fit, log_likelihood_null,
    observed_information_cov_params, ols_based_initial_params, opg_cov_params, run_solver,
    sandwich_cov_params, standardize_columns, validate_fit_preconditions,
};
use crate::nonlinear::separation::verify_no_separation;
use crate::shared::error::CommonError;
use crate::shared::inference;
use faer::Mat;
use statrs::distribution::Normal;

/// [`LogitEstimator::fit`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_logit(
    input: LogitInput,
    options: MleFitOptions,
) -> Result<LogitEstimator, MleError> {
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

    let n = input.nobs();
    let k = input.k();
    validate_fit_preconditions(
        confidence_level,
        max_iter,
        tol,
        input.y(),
        k,
        input.has_intercept(),
        &cov_type,
    )?;

    let (x_std, scale) = standardize_columns(input.x(), input.has_intercept());
    // `solver`に関わらず、標準化空間でLPMのIRLS 1ステップ相当を初期値（warm start）に
    // する（`ols_based_initial_params`）。前段の列ピボットQRランクチェックにより、
    // 完全な多重共線性は`solver`によらず単一経路で`SingularDesignMatrix`として検出される
    // （従来はゼロベクトル初期値で、`newton`は`newton_step`内のQR、
    // `bfgs`/`lbfgs`は収束後の`observed_information_cov_params`という`solver`依存の
    // 別経路に分かれていた）。
    let initial_params = ols_based_initial_params(
        &x_std,
        input.x(),
        input.y(),
        input.has_intercept(),
        input.param_names(),
        // logit: IRLS重み `w = p̄(1-p̄)`、リンク値 `η₀ = ln(p̄/(1-p̄))`。
        |p_bar| (p_bar * (1.0 - p_bar), (p_bar / (1.0 - p_bar)).ln()),
    )?;
    let problem = LogitProblem::from_standardized(x_std, input.y().clone());
    // `cov_type`がOPG/サンドイッチ型の場合、収束点でのスコア評価に元の
    // `LogitProblem`（標準化空間のx_std）が必要になる。`run_solver`は`problem`の
    // 所有権を取り込む（内部で保持していたモデルを呼び出し元へ返さない設計）ため、
    // 事前にクローンしておく必要がある（`LogitProblem`は`argmin::core::Executor`
    // 向けに元々`Clone`を要求しているため、この用途のための追加のtraitではない）。
    // `Classical`はスコアを使わないため、無駄な複製（設計行列を含む）を避けるために
    // `cov_type`に応じて条件付きで行う。
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

    let params = destandardize_params(&output.params, &scale);

    let hessian_std = Mat::from_fn(k, k, |i, j| output.hessian[i][j]);
    // 収束後に完全分離・準完全分離を厳密に判定する（`nonlinear::separation`）。
    // 分離なら`raise_on_non_convergence`に応じてエラー、または`converged=false`で続行する。
    let converged = verify_no_separation(
        input.x(),
        &scale,
        input.y(),
        output.converged,
        raise_on_non_convergence,
    )?;
    // `problem_for_scores.as_ref().expect(...)`は各非`Classical`分岐でのみ呼ばれ、
    // 直前の`match cov_type { CovType::Classical => None, _ => Some(...) }`により
    // 常に`Some`であることが保証されている内部契約（`cov_type`という同じ値で
    // 2回目のmatchを行うことになるが、パニックしないことをコンパイラの型システムでは
    // 表現できないため、`expect`のメッセージで契約を明記して防御的に扱う）。
    let cov_params_std = match &cov_type {
        CovType::Classical => observed_information_cov_params(&hessian_std, k)?,
        CovType::Opg => {
            let problem = problem_for_scores
                .as_ref()
                .expect("problem_for_scores must be Some for CovType::Opg");
            opg_cov_params(&problem.scores(&output.params), k)?
        }
        CovType::Hc0 => {
            let problem = problem_for_scores
                .as_ref()
                .expect("problem_for_scores must be Some for CovType::Hc0");
            sandwich_cov_params(
                &hessian_std,
                &problem.scores(&output.params),
                n,
                k,
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
                k,
                SandwichVariant::Hc1,
            )?
        }
        CovType::Cluster { groups } => {
            let problem = problem_for_scores
                .as_ref()
                .expect("problem_for_scores must be Some for CovType::Cluster");
            // `groups`のNone・クラスター数不足の検証はfit()冒頭で完了済み
            // （MissingClusterColumn/InsufficientClustersを最適化前に早期に返す
            // ため）。ここでの`expect`はその契約を明記する防御的な扱い。
            let groups = groups
                .as_ref()
                .expect("groups are validated by validate_fit_preconditions for Cluster");
            cluster_cov_params(&hessian_std, &problem.scores(&output.params), n, k, groups)?
        }
    };
    let cov_params = destandardize_cov_params(&cov_params_std, &scale);

    // `Normal::new(0.0, 1.0)`は標準正規分布であり、標準偏差が正であることを
    // 要求するstatrsの検証を常に満たすため、この`map_err`分岐は理論上到達不能
    // （`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照）。
    let normal = Normal::new(0.0, 1.0).map_err(|e| CommonError::internal(e.to_string()))?;
    let z_crit = inference::critical_value(&normal, confidence_level);

    let mut std_errors = vec![0.0; k];
    let mut test_stats = vec![0.0; k];
    let mut p_values = vec![0.0; k];
    let mut conf_lower = vec![0.0; k];
    let mut conf_upper = vec![0.0; k];

    for j in 0..k {
        let se = (*cov_params.get(j, j)).sqrt();
        let stat = inference::compute_inference_stat(&normal, params[j], se, z_crit);

        std_errors[j] = se;
        test_stats[j] = stat.stat;
        p_values[j] = stat.p_value;
        conf_lower[j] = stat.conf_low;
        conf_upper[j] = stat.conf_high;
    }

    let llf = log_likelihood(input.x(), input.y(), &params);
    let llnull = log_likelihood_null(input.y());
    // `k=0`（`include_intercept=false`かつ説明変数も無い病的な入力）は
    // `validate_fit_preconditions`が`NoRegressors`として弾くため、ここには到達しない。
    let gof = goodness_of_fit(llf, llnull, n, k)?;
    let GoodnessOfFit {
        lr_statistic,
        lr_p_value,
        pseudo_r_squared,
        aic,
        bic,
        df_model,
        df_resid,
    } = gof;

    Ok(LogitEstimator {
        fitted: LogitFitted {
            input,
            params,
            cov_params,
        },
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        converged,
        n_iter: output.n_iter,
        log_likelihood: llf,
        log_likelihood_null: llnull,
        lr_statistic,
        lr_p_value,
        pseudo_r_squared,
        aic,
        bic,
        df_model,
        df_resid,
    })
}

#[cfg(test)]
mod tests {

    use super::super::model::logistic;
    use super::super::problem::LogitProblem;
    use super::super::test_support::*;

    use crate::nonlinear::common::CovType;

    use crate::nonlinear::common::MleError;
    use crate::nonlinear::common::MleFitOptions;
    use crate::nonlinear::common::SandwichVariant;
    use crate::nonlinear::common::SolverType;
    use crate::nonlinear::common::cluster_cov_params;

    use crate::nonlinear::common::destandardize_cov_params;

    use crate::nonlinear::common::opg_cov_params;

    use crate::nonlinear::common::sandwich_cov_params;
    use crate::nonlinear::common::standardize_columns;

    use crate::nonlinear::logit::LogitEstimator;

    use crate::nonlinear::logit::LogitInput;

    use crate::shared::error::ClusterSetting;
    use crate::shared::error::CommonError;
    use crate::shared::group_codes::GroupCodes;

    use argmin::core::Hessian;
    use faer::Mat;
    use statrs::distribution::ChiSquared;
    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;

    #[test]
    fn fit_newton_converges_to_closed_form_solution_for_intercept_only_model() {
        let input = intercept_only_input();
        let estimator = LogitEstimator::fit(
            input,
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
        let expected = (y_bar / (1.0 - y_bar)).ln();

        assert!(estimator.converged());
        assert_eq!(estimator.params().len(), 1);
        assert!(
            (estimator.params()[0] - expected).abs() < 1e-6,
            "params={:?}, expected={}",
            estimator.params(),
            expected
        );
        // 切片のみの1次元凹関数のNewton法は数回で収束するはず
        assert!(estimator.n_iter() <= 10, "n_iter={}", estimator.n_iter());
    }

    #[test]
    fn fit_pins_faer_global_parallelism_to_seq() {
        // `fit()` 冒頭の `crate::shared::parallelism::ensure_serial()` が faer の
        // グローバル並列度を `Par::Seq` へ引き戻すことの回帰ガード（nonlinear 系統代表）。
        // 別テストが `Seq` にしている可能性があるため、まず `Rayon` に戻してから通す。
        // 設計行列は極小なので一時的な `Rayon` 設定は病理を招かない。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let _ = LogitEstimator::fit(
            small_input(),
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

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    /// 切片のみモデルは観測情報行列も閉じた形で書ける: 全観測で`p_i=ȳ`（closed form）
    /// なので、Hessianは`-Σp_i(1-p_i) = -n*ȳ*(1-ȳ)`というスカラーになり、
    /// `Var(θ̂) = -H⁻¹ = 1/(n*ȳ*(1-ȳ))`。z値・p値・信頼区間はこの分散から
    /// 標準正規分布（統計独立に`statrs::Normal`で検算）で導出できる。
    #[test]
    fn fit_computes_std_errors_test_stats_p_values_and_ci_matching_closed_form_for_intercept_only_model()
     {
        let estimator = LogitEstimator::fit(
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
        let expected_var = 1.0 / (n * y_bar * (1.0 - y_bar));
        let expected_se = expected_var.sqrt();

        // Newtonの収束判定（勾配ノルム`tol=1e-6`）による数値誤差があるため、
        // 他の閉じた形テスト（`fit_newton_converges_to_closed_form_solution_...`）と
        // 同じ桁の許容誤差（1e-6）を使う（1e-9のような厳しすぎる許容誤差は、収束点が
        // 解析解からわずかにズレることに起因する誤検出を招く）。
        assert!((*estimator.cov_params().get(0, 0) - expected_var).abs() < 1e-6);
        assert!((estimator.std_errors()[0] - expected_se).abs() < 1e-6);

        let expected_z = estimator.params()[0] / expected_se;
        assert!((estimator.test_stats()[0] - expected_z).abs() < 1e-6);

        // p値・信頼区間はstatrsのNormalで独立に検算する（本体実装と同じ計算式を
        // 繰り返すのではなく、標準正規分布の性質から直接導出する）。
        let normal = Normal::new(0.0, 1.0).unwrap();
        let expected_p = 2.0 * (1.0 - normal.cdf(expected_z.abs()));
        assert!((estimator.p_values()[0] - expected_p).abs() < 1e-6);

        let z_crit = normal.inverse_cdf(0.975);
        let expected_lower = estimator.params()[0] - z_crit * expected_se;
        let expected_upper = estimator.params()[0] + z_crit * expected_se;
        assert!((estimator.conf_lower()[0] - expected_lower).abs() < 1e-6);
        assert!((estimator.conf_upper()[0] - expected_upper).abs() < 1e-6);
    }

    /// 多変量（説明変数が2つ以上）の場合、標準誤差に閉じた形の解析解は無いため、
    /// `cov_params`の対称性・各種統計量の内部整合性（z値・信頼区間の定義式通りの関係）を
    /// 検証する回帰テスト。特に`destandardize_cov_params`が非対角成分も含めて
    /// 正しく元のスケールへ逆変換できているかを確認する（対角成分だけでは
    /// `destandardize_cov_params`のヤコビアン`J`の非対角（切片行）の取り違えや転置ミス等を検出できない）。
    #[test]
    fn fit_cov_params_is_symmetric_and_stats_are_internally_consistent() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
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
        let k = 3;

        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *estimator.cov_params().get(j, i)).abs()
                        < 1e-9,
                    "cov_params is not symmetric at ({i},{j})"
                );
            }
            assert!(
                *estimator.cov_params().get(i, i) > 0.0,
                "diagonal[{i}] <= 0"
            );
        }

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal.inverse_cdf(0.975);
        for j in 0..k {
            let se = estimator.std_errors()[j];
            assert!((se * se - *estimator.cov_params().get(j, j)).abs() < 1e-9);
            assert!((estimator.test_stats()[j] - estimator.params()[j] / se).abs() < 1e-9);
            assert!(
                (estimator.conf_upper()[j] - estimator.conf_lower()[j] - 2.0 * z_crit * se).abs()
                    < 1e-9
            );
        }
    }

    /// 切片のみモデルは`log_likelihood`と`log_likelihood_null`が定義上一致する
    /// （どちらも同じ「切片のみ」モデルを参照するため）。この性質を使い、
    /// `df_model=0`分岐（`lr_p_value=NaN`、OLSの`f_p_value`と同じ扱い）を含めた
    /// 適合度統計量一式を検証する。
    #[test]
    fn fit_computes_goodness_of_fit_statistics_for_intercept_only_model() {
        let estimator = LogitEstimator::fit(
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

    /// 多変量（k=3）モデルでの適合度統計量を、実装（`softplus`ベース）とは異なる式
    /// （`logistic`から直接`Σ[y ln(p) + (1-y) ln(1-p)]`を計算するベルヌーイ対数尤度の
    /// 定義式そのもの）で独立に再計算し、突き合わせる。`fit_cov_params_is_symmetric_
    /// and_stats_are_internally_consistent`と同じデータセットを再利用する。
    #[test]
    fn fit_computes_goodness_of_fit_statistics_matching_independently_recomputed_values() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
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
        let expected_ll: f64 = (0..n)
            .map(|i| {
                let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                let p = logistic(z);
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

    /// `include_intercept=false`のとき、`log_likelihood_null`が参照する「切片のみ」
    /// モデルはフィット対象のモデルの部分集合（入れ子）にならない
    /// （`LogitEstimator`の`log_likelihood_null`フィールドdocコメント参照）。
    /// この場合`lr_statistic`が負になりうる（statsmodels準拠の仕様上の挙動、
    /// 実測で確認済み）ことを回帰テストとして固定する。
    /// `df_model`/`df_resid`/`aic`/`bic`は`include_intercept`の値に関わらず
    /// 同じ式（`k-1`/`n-k`/`-2ℓ+2k`/`-2ℓ+ln(n)k`）で計算されることも確認する。
    #[test]
    fn fit_lr_statistic_can_be_negative_when_include_intercept_is_false() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
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

    /// `cov_type=Opg`/`Hc0`/`Hc1`が返す`cov_params`を、`fit()`と同じ手順
    /// （標準化→収束点でのscores/Hessian評価→`common/cov.rs`の共通行列演算→
    /// `destandardize_cov_params`）をテスト側で独立に再現した値と突き合わせる。
    /// 多変量（k=3）データセットを使う理由: 切片のみモデルでは情報行列の等式
    /// （`Σᵢsᵢsᵢ' = -H`）が有限標本で厳密に成り立ってしまい、`classical`/`opg`/`hc0`
    /// が偶然同じ値になるため、`fit()`の`match cov_type`の配線ミス（例えば`Opg`の
    /// 枝で誤って`observed_information_cov_params`を呼んでいた場合等）を検出できない
    /// （`fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ
    /// データセットを再利用する）。
    #[test]
    fn fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 4;

        let classical = LogitEstimator::fit(
            make_input(),
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

        // `fit()`と同じ手順を独立に再現する: 標準化→θ_stdへ変換→収束点でのscores/
        // Hessian評価→destandardize_cov_params。`LogitProblem::hessian`（argminトレイト）は
        // コスト関数（負の対数尤度）のHessianを返す符号規約のため、`run_solver`が
        // `SolverOutput.hessian`に格納する対数尤度そのもののHessianに合わせて1回符号反転する
        // （`run_solver`のdocコメント「Hessianトレイトの符号規約」と同じ変換）。
        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std = scale.standardize_params(classical.params());
        let problem_std =
            LogitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_opg =
            destandardize_cov_params(&opg_cov_params(&scores_std, k).unwrap(), &scale);
        let expected_hc0 = destandardize_cov_params(
            &sandwich_cov_params(&hessian_std, &scores_std, n, k, SandwichVariant::Hc0).unwrap(),
            &scale,
        );
        let expected_hc1 = destandardize_cov_params(
            &sandwich_cov_params(&hessian_std, &scores_std, n, k, SandwichVariant::Hc1).unwrap(),
            &scale,
        );

        let cases = [
            (CovType::Opg, &expected_opg),
            (CovType::Hc0, &expected_hc0),
            (CovType::Hc1, &expected_hc1),
        ];
        for (cov_type, expected) in cases {
            let estimator = LogitEstimator::fit(
                make_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 35,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            for i in 0..k {
                for j in 0..k {
                    assert!(
                        (*estimator.cov_params().get(i, j) - *expected.get(i, j)).abs() < 1e-8,
                        "cov_type={:?}, ({i},{j}): actual={}, expected={}",
                        cov_type,
                        *estimator.cov_params().get(i, j),
                        *expected.get(i, j)
                    );
                }
            }
        }
    }

    /// `cov_type=Cluster`が返す`cov_params`を、`fit()`と同じ手順をテスト側で独立に
    /// 再現した値と突き合わせる（上の`fit_cov_type_opg_hc0_hc1_match_independently_
    /// recomputed_values`と同じ技法・同じ多変量データセット。情報行列の等式が
    /// 厳密に成り立つ切片のみモデルでは配線ミスを検出できないため）。
    /// クラスター数`G`は傾き係数の数`q=2`より多くする必要がある（`G <= q`は
    /// `InsufficientClustersForInference`で弾かれる）ため`G=3`（2:1:1）。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 4;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];

        let classical = LogitEstimator::fit(
            make_input(),
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

        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std = scale.standardize_params(classical.params());
        let problem_std =
            LogitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_cluster = destandardize_cov_params(
            &cluster_cov_params(
                &hessian_std,
                &scores_std,
                n,
                k,
                &crate::shared::group_codes::GroupCodes::from_labels(&groups),
            )
            .unwrap(),
            &scale,
        );

        let estimator = LogitEstimator::fit(
            make_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        )
        .unwrap();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *expected_cluster.get(i, j)).abs() < 1e-8,
                    "({i},{j}): actual={}, expected={}",
                    *estimator.cov_params().get(i, j),
                    *expected_cluster.get(i, j)
                );
            }
        }
    }

    /// 上のテストは均等サイズのグループのみを検証しているが、
    /// `testing-policy.md`が指摘する通り均等サイズのみのテストは実務で起こりやすい
    /// 偏った分布のグループサイズ（クラスター内の観測数がクラスターごとに異なる場合）
    /// を見逃しうる。OLS側の対応するテスト（`fit_computes_cluster_std_errors_test_stats_
    /// p_values_conf_int_and_f_test`、2:3の不均衡）に倣い、3:1:1の不均衡なグループでも
    /// 同じ独立再計算の技法で検証する（`G=3 > q=2`）。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups() {
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let x_columns = vec![
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![-5.0, 2.0, 8.0, -1.0, 3.0],
        ];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 5;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];

        let classical = LogitEstimator::fit(
            make_input(),
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

        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std = scale.standardize_params(classical.params());
        let problem_std =
            LogitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_cluster = destandardize_cov_params(
            &cluster_cov_params(
                &hessian_std,
                &scores_std,
                n,
                k,
                &crate::shared::group_codes::GroupCodes::from_labels(&groups),
            )
            .unwrap(),
            &scale,
        );

        let estimator = LogitEstimator::fit(
            make_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        )
        .unwrap();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *expected_cluster.get(i, j)).abs() < 1e-8,
                    "({i},{j}): actual={}, expected={}",
                    *estimator.cov_params().get(i, j),
                    *expected_cluster.get(i, j)
                );
            }
        }
    }

    #[test]
    fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
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
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let groups = vec!["a".to_string(); 4];

        let result = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
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

    /// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下だと、
    /// クラスターロバスト共分散`Ŝ`はクラスター寄与スコアの総和がゼロ（MLEの一次条件
    /// `Σ_i s_i = 0`）で`rank(Ŝ) ≤ g - 1`となり退化する。Logit/Probitは全体検定がLR
    /// のため`q×q`反転こそ通らないが、縮退した共分散から読んだSEを無警告で返すのは
    /// 実質バグのため、`fit()`冒頭（Newton反復の前）のバリデーションで
    /// `CommonError::InsufficientClustersForInference`として弾く（
    /// Logit/Probitでは新規制約）。説明変数2個（`q = 3 - 1 = 2`）に対し`g = 2`。
    #[test]
    fn fit_returns_validation_error_when_cluster_count_at_most_slopes() {
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let x_columns = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        ];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let groups: Vec<String> = (0..6)
            .map(|i| if i < 3 { "a" } else { "b" }.to_string())
            .collect();

        let result = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
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

    /// `solver`（`bfgs`/`lbfgs`）と`cov_type`（`Opg`/`Hc0`/`Hc1`/`Cluster`）の組み合わせが
    /// 正しく機能することを確認する（他のテストは`solver`横断が
    /// `CovType::Classical`のみ、`cov_type`横断が`SolverType::Newton`のみで、両方を
    /// 同時に変える組み合わせはここで検証する）。`scores_std`の評価は収束点の
    /// パラメータにのみ依存し最適化アルゴリズムの種類に依存しない設計のため、
    /// `newton`で計算した`cov_params`（既に上のテストで正しさを検証済み）と
    /// `bfgs`/`lbfgs`の結果が一致するはず。
    #[test]
    fn fit_non_classical_cov_types_work_with_bfgs_and_lbfgs() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        // `G=3 > q=2`（`G <= q`は`InsufficientClustersForInference`で弾かれる）。
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];

        for cov_type in [
            CovType::Opg,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        ] {
            let newton = LogitEstimator::fit(
                make_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 35,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            )
            .unwrap();

            for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
                let estimator = LogitEstimator::fit(
                    make_input(),
                    MleFitOptions {
                        solver,
                        max_iter: 200,
                        tol: 1e-8,
                        raise_on_non_convergence: true,
                        cov_type: cov_type.clone(),
                        confidence_level: 0.95,
                    },
                )
                .unwrap();

                assert!(
                    estimator.converged(),
                    "cov_type={:?}, {:?}",
                    cov_type,
                    solver
                );
                for i in 0..k {
                    for j in 0..k {
                        // 相対許容: 4観測・近分離のためcovの成分は`O(10²)`と大きく、絶対許容
                        // `1e-4`は相対`1e-7`に当たりbfgsの収束点の僅かな差で落ちる。
                        let expected = *newton.cov_params().get(i, j);
                        assert!(
                            (*estimator.cov_params().get(i, j) - expected).abs()
                                < 1e-4 * (1.0 + expected.abs()),
                            "cov_type={:?}, solver={:?}, ({i},{j}): actual={}, newton={}",
                            cov_type,
                            solver,
                            *estimator.cov_params().get(i, j),
                            *newton.cov_params().get(i, j)
                        );
                    }
                }
            }
        }
    }

    /// `newton`と同じデータセット（既知の解析解を持つ切片のみモデル）で`bfgs`/`lbfgs`を
    /// 実行し、いずれも同じ解析解へ収束することを検証する。
    #[test]
    fn fit_bfgs_and_lbfgs_converge_to_same_solution_as_newton() {
        let y_bar: f64 = 4.0 / 7.0;
        let expected = (y_bar / (1.0 - y_bar)).ln();

        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let estimator = LogitEstimator::fit(
                intercept_only_input(),
                MleFitOptions {
                    solver,
                    max_iter: 100,
                    tol: 1e-6,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();

            assert!(estimator.converged(), "{:?}", solver);
            assert!(
                (estimator.params()[0] - expected).abs() < 1e-4,
                "solver={:?}, params={:?}, expected={}",
                solver,
                estimator.params(),
                expected
            );
        }
    }

    /// 切片のみモデル（`intercept_only_input`）は`x`が定数列（切片）だけのため、
    /// `standardize_columns`のスケーリングが実質no-op（`stds`が全て`1.0`のまま）になり、
    /// 標準化・逆標準化の往復ロジックを通らない。このテストは
    /// 非自明なスケール（`std`が1から離れた値）を持つ説明変数を含むデータセットで
    /// `newton`/`bfgs`/`lbfgs`を実行し、3手法が同じ解へ収束することを検証する
    /// （閉じた形の解析解は存在しないため、`newton`の結果を参照値として使う
    /// クロスメソッド一致検証。標準化空間でのBFGSの初期逆Hessian
    /// （`identity_matrix(k)`）・`destandardize_params`が正しく機能していることの
    /// 間接的な検証になる）。
    #[test]
    fn fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0]];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let newton = LogitEstimator::fit(
            make_input(),
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
        assert!(newton.converged());

        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let estimator = LogitEstimator::fit(
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

            assert!(estimator.converged(), "{:?}", solver);
            for j in 0..2 {
                assert!(
                    (estimator.params()[j] - newton.params()[j]).abs() < 1e-4,
                    "solver={:?}, j={j}, params={:?}, newton_params={:?}",
                    solver,
                    estimator.params(),
                    newton.params()
                );
            }
        }
    }

    #[test]
    fn fit_returns_invalid_confidence_level_error_out_of_range() {
        let result = LogitEstimator::fit(
            intercept_only_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-6,
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
    fn fit_returns_invalid_max_iter_error_for_non_positive_max_iter() {
        let result = LogitEstimator::fit(
            intercept_only_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 0,
                tol: 1e-6,
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
    fn fit_returns_insufficient_observations_error_when_n_less_equal_k() {
        let y = vec![0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
    }

    #[test]
    fn fit_returns_no_regressors_error_when_k_is_zero() {
        // `include_intercept=false`かつ説明変数も無い（k=0）病的な入力。
        // 以前はこの経路が`fit()`冒頭の`n<=k`チェック（n>=1なら常に通過してしまう）を
        // すり抜け、後段の`cov_params`計算（0×0行列に対する
        // `ensure_well_conditioned_symmetric_matrix`）で`faer`がpanicしていた
        // すり抜けていた。`k==0`を明示的に検証することでグレースフルなエラーに
        // なることを確認する。`n<=k`（観測数不足）とは
        // 別の不正のため、`InsufficientObservations`ではなく専用の`NoRegressors`を返す
        // （`k=0`だと`n>k`は常に成立してしまい、`InsufficientObservations`のメッセージ
        // だと「条件を満たしているのにエラーになる」という誤解を招くため）。
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let input = LogitInput::from_columns(&y, &[], vec![], false, "y".to_string()).unwrap();
        assert_eq!(input.k(), 0);

        let result = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::NoRegressors { n: 5 })
        );
    }

    #[test]
    fn fit_returns_invalid_tol_error_for_non_positive_tol() {
        for tol in [0.0, -1.0] {
            let result = LogitEstimator::fit(
                intercept_only_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 35,
                    tol,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            );
            assert_eq!(result.unwrap_err(), MleError::InvalidTol { tol });
        }
    }

    #[test]
    fn fit_returns_invalid_binary_y_error_for_non_binary_values() {
        // 0.5（連続値）・2.0（0/1ではない2値）・statsmodelsなら許容する単位区間内の
        // 中間値も含め、{0.0, 1.0}の完全一致以外は全て弾かれることを確認する
        // （statsmodelsより厳格な`{0.0, 1.0}`のみ許容という設計判断）。
        for (bad_value, bad_row) in [(0.5, 1), (2.0, 2)] {
            let mut y = vec![0.0, 1.0, 0.0, 1.0];
            y[bad_row] = bad_value;
            let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
            let input = LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result = LogitEstimator::fit(
                input,
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 35,
                    tol: 1e-6,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            );
            assert_eq!(
                result.unwrap_err(),
                MleError::InvalidBinaryY {
                    row: bad_row,
                    value: bad_value
                }
            );
        }
    }

    /// 標準偏差0の列（定数列）は、切片ありでは標準化で平均が引かれて全ゼロ列になり、
    /// 設計行列がランク落ちして`SingularDesignMatrix`で弾かれる（`solver`に依らない）。
    #[test]
    fn fit_returns_singular_design_matrix_error_for_constant_column_with_intercept() {
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let constant = vec![0.1; 8];

        for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
            let input = LogitInput::from_columns(
                &y,
                &[x1.clone(), constant.clone()],
                vec!["x1".to_string(), "c".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            let result = LogitEstimator::fit(
                input,
                MleFitOptions {
                    solver,
                    max_iter: 35,
                    tol: 1e-6,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            );
            assert!(
                matches!(result, Err(MleError::SingularDesignMatrix { .. })),
                "solver={solver:?}, result={result:?}"
            );
        }
    }

    /// 完全な多重共線性（`x2 = 2·x1`）の設計行列は、`fit()`冒頭の列ピボットQR
    /// ランクチェック（`nonlinear::common::checked_design_matrix_qr`）で`solver`・
    /// `cov_type`に関わらず単一経路で`SingularDesignMatrix`として弾かれる。
    ///
    /// 従来はゼロベクトル初期値で、`newton`は`newton_step`内の列ピボットQR、`bfgs`/`lbfgs`は
    /// 収束後の`observed_information_cov_params`（`cov_type=Opg`なら`opg_cov_params`）と
    /// いう`solver`/`cov_type`依存の別経路で検出しており、過去に`bfgs`だけ検出漏れして
    /// 桁違いに巨大なSEを含む`Ok`が返る実バグがあった（`tests/nonlinear/
    /// test_logit_validation.py`の履歴参照）。前段QRへの一本化でそのバグクラスを
    /// 構造的に排除したため、`solver`×`cov_type`を網羅していた旧5テスト
    /// （`..._with_bfgs_and_lbfgs` / `..._with_hc0_and_hc1` / `fit_returns_singular_opg_
    /// matrix_error_...` / `..._with_cluster`）を本1テストへ集約した。`x2`は`x1`から
    /// 生成し関係を自明にする。
    ///
    /// 旧5テストが検証していた「`fit()`の各`cov_type`分岐で`SingularHessian`/
    /// `SingularOpgMatrix`が`?`で伝播する」経路は、本入力では前段QRで先に弾かれるため
    /// もう通らない。この伝播経路のカバレッジは (1) `common/cov.rs`の関数レベルテスト
    /// （`observed_information_cov_params_returns_singular_hessian_error_*` /
    /// `opg_cov_params_*` / `sandwich_cov_params_*` / `cluster_cov_params_*`）と、
    /// (2) 共有インフラを`fit()`レベルで踏む`tobit/fit.rs`の
    /// `fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_
    /// truncated_point_with_hc0_and_hc1` / `..._with_cluster` が担う（`cov_params`計算は
    /// Logit/Probit/Tobitで同一コード）。`?`自体は分岐ロジックを持たないため
    /// `testing-policy.md`のカバレッジ方針上これで足りる。
    ///
    /// `Cluster`は`G=3 > q=2`（`q = k - k_constant`）にして`fit()`冒頭の
    /// `InsufficientClustersForInference`（`G <= q`）より手前を通す。
    #[test]
    fn fit_returns_singular_design_matrix_error_for_perfectly_collinear_design_matrix() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let x2: Vec<f64> = x1.iter().map(|v| v * 2.0).collect();
        let groups = vec![
            "g1".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g3".to_string(),
        ];

        for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
            for cov_type in [
                CovType::Classical,
                CovType::Hc0,
                CovType::Hc1,
                CovType::Opg,
                CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
            ] {
                let input = LogitInput::from_columns(
                    &y,
                    &[x1.clone(), x2.clone()],
                    vec!["x1".to_string(), "x2".to_string()],
                    true,
                    "y".to_string(),
                )
                .unwrap();

                let result = LogitEstimator::fit(
                    input,
                    MleFitOptions {
                        solver,
                        max_iter: 100,
                        tol: 1e-6,
                        raise_on_non_convergence: true,
                        cov_type: cov_type.clone(),
                        confidence_level: 0.95,
                    },
                );
                assert!(
                    matches!(result, Err(MleError::SingularDesignMatrix { .. })),
                    "solver={solver:?}, cov_type={cov_type:?}, result={result:?}"
                );
            }
        }
    }

    // `max_iter`打ち切りのテストは`intercept_only_input()`を使わない: warm start
    // （`ols_based_initial_params`）は切片のみモデルでは初期値がそのまま
    // 厳密なMLE（`η₀=ln(ȳ/(1-ȳ))`）になり1反復以内で収束してしまうため。代わりに
    // 多変量かつ収束に多反復を要する`near_separation_input_with_beta1(20.0)`を使う。
    // `beta1=20`への暗黙の依存が2つある:
    //   1. 収束の遅さ: `beta1`が大きいほど収束に多反復かかる。`20`は
    //      `fit_converges_normally_for_mild_near_separation_data_across_all_methods`
    //      （`max_iter=35`で3手法とも収束）で正常収束が保証されており、かつ warm start
    //      からでも1反復では`tol=1e-12`に到底届かない。
    //   2. 分離チェックとの相互作用: 分離の検出（`verify_no_separation`）は収束した場合にだけ
    //      走る。`max_iter=1`では未収束のため必ず`NonConvergence`（`raise=true`）／
    //      `converged=false`（`raise=false`）になる。`beta1=100`
    //      （`fit_returns_separation_detected_error_for_separated_data`）は完全分離になる
    //      ため、境界側の`20`である必要がある。
    #[test]
    fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
        let result = LogitEstimator::fit(
            near_separation_input_with_beta1(20.0),
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

    #[test]
    fn fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false() {
        let estimator = LogitEstimator::fit(
            near_separation_input_with_beta1(20.0),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 1,
                tol: 1e-12,
                raise_on_non_convergence: false,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert!(!estimator.converged());
    }

    /// `beta1`（`x1`の真の係数）を指定して、ほぼ分離したデータを生成する。`beta1`が大きいほど
    /// `y`が`x1`の符号だけで決まりやすくなり、`beta1=100`では`x1`の符号で全観測が完全に分類
    /// できる（完全分離、`fit_returns_separation_detected_error_for_separated_data`）。
    /// 疑似乱数生成は`benchmark/nonlinear/datasets.py`の`near_separation`シナリオと同じ設計
    /// 思想（真のlogit DGPで生成）。小さい`beta1`（例: `20`）は、そのシナリオと同様に意図的に
    /// 境界ケースに留めた、正常に収束するデータになる。
    fn near_separation_input_with_beta1(beta1: f64) -> LogitInput {
        fn lcg(seed: &mut u64) -> f64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
        }

        let n = 200;
        let mut seed = 42u64;
        let beta = [0.0, beta1, 0.5];
        let x1: Vec<f64> = (0..n).map(|_| lcg(&mut seed) * 4.0 - 2.0).collect();
        let x2: Vec<f64> = (0..n).map(|_| lcg(&mut seed) * 2.0 - 1.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let z = beta[0] + beta[1] * x1[i] + beta[2] * x2[i];
                let p = 1.0 / (1.0 + (-z).exp());
                if lcg(&mut seed) < p { 1.0 } else { 0.0 }
            })
            .collect();

        LogitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    }

    fn near_separation_input() -> LogitInput {
        near_separation_input_with_beta1(100.0)
    }

    #[test]
    fn fit_returns_separation_detected_error_for_separated_data() {
        for (solver, max_iter) in [
            (SolverType::Newton, 100),
            (SolverType::Bfgs, 100),
            (SolverType::Lbfgs, 100),
        ] {
            let result = LogitEstimator::fit(
                near_separation_input(),
                MleFitOptions {
                    solver,
                    max_iter,
                    tol: 1e-6,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            );
            assert!(
                matches!(result, Err(MleError::SeparationDetected)),
                "solver={:?}, result={:?}",
                solver,
                result
            );
        }
    }

    #[test]
    fn fit_returns_unconverged_result_for_separated_data_without_raising() {
        let estimator = LogitEstimator::fit(
            near_separation_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: false,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert!(!estimator.converged());
    }

    /// 回帰変数どうしの相関が強い（等相関`ρ=0.9`・`k=20`）データでも、`lbfgs`が既定の
    /// `max_iter=35`以内に`newton`と同じ解へ収束する回帰テスト。標準化後Hessianの条件数が
    /// 数百になり、`γI`を初期行列とするL-BFGSは40反復前後を要し既定を超えた。warm start点の
    /// 逆Hessianを初期行列にすると4反復前後になる（centeringでは直らない、相関による悪条件）。
    #[test]
    fn fit_lbfgs_converges_within_default_max_iter_for_strongly_correlated_regressors() {
        let (y, columns) =
            crate::nonlinear::common::test_support::equicorrelated_binary_data(1000, 20, 0.9, 1);
        let names: Vec<String> = (1..=20).map(|j| format!("x{j}")).collect();
        let fit = |solver: SolverType| {
            let input =
                LogitInput::from_columns(&y, &columns, names.clone(), true, "y".to_string())
                    .unwrap();
            LogitEstimator::fit(
                input,
                MleFitOptions {
                    solver,
                    max_iter: 35,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap()
        };

        let newton = fit(SolverType::Newton);
        let lbfgs = fit(SolverType::Lbfgs);

        assert!(lbfgs.converged());
        // 初期行列が`γI`だと40反復前後、warm start点の逆Hessianだと4反復前後（実測）。
        assert!(lbfgs.n_iter() <= 15, "n_iter={}", lbfgs.n_iter());
        for (a, b) in lbfgs.params().iter().zip(newton.params()) {
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "lbfgs={:?}, newton={:?}",
                lbfgs.params(),
                newton.params()
            );
        }
    }

    /// 平均が標準偏差に比べて大きい列（平均÷標準偏差が約12〜16）を持つ設計行列でも、標準化後の
    /// Hessianは良条件（条件数が小さい）になる。標準化で平均を引かないと列が切片列とほぼ共線に
    /// なり、条件数が数千になる（Wooldridge appleで約7,300）。ソルバーの初期化（`FaerBfgs`/
    /// `FaerLbfgs`の逆Hessian）はこの悪条件を反復数の面で隠すため、反復数ではなく条件数を直接見る。
    #[test]
    fn standardized_hessian_is_well_conditioned_for_columns_with_large_means() {
        let (y, columns) = crate::nonlinear::common::test_support::large_means_binary_data();
        let names: Vec<String> = (1..=columns.len()).map(|j| format!("x{j}")).collect();
        let input = LogitInput::from_columns(&y, &columns, names, true, "y".to_string()).unwrap();
        let k = input.k();

        let (x_std, _) = standardize_columns(input.x(), input.has_intercept());
        let problem = LogitProblem::from_standardized(x_std, input.y().clone());
        let hessian = problem.hessian(&vec![0.0; k]).unwrap();

        let cond = crate::nonlinear::common::test_support::condition_number(&hessian);
        // 平均を引く標準化では列が互いにほぼ直交し1〜2程度、引かないと数百〜数千（実測）。
        assert!(cond < 10.0, "condition number={cond}");
    }

    /// `beta1=20`（分離と誤検知してはいけない境界ケース、`tests/
    /// api_tests/fixtures/benchmarks/data/logit_near_separation.csv`と同じ設計思想）で
    /// 3手法とも正常に収束することを固定するリグレッションテスト。
    /// 分離検出の変更が既存の合格ケースを誤検知させないことを保証する。
    #[test]
    fn fit_converges_normally_for_mild_near_separation_data_across_all_methods() {
        for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
            let result = LogitEstimator::fit(
                near_separation_input_with_beta1(20.0),
                MleFitOptions {
                    solver,
                    max_iter: 35,
                    tol: 1e-6,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            );
            assert!(result.is_ok(), "solver={:?}, result={:?}", solver, result);
            assert!(result.unwrap().converged(), "solver={:?}", solver);
        }
    }

    /// property-basedテスト。`ols/fit.rs`の`mod proptests`と同型の設計だが、Logitは
    /// MLEベースのため不変条件が異なる（`testing-policy.md`「property-basedテスト」参照）。
    /// OLSの`coefficients_scale_linearly_with_y`（yに関する線形性）に相当する不変条件は
    /// 無い（yは0/1のためスケール自体に意味が無い）代わりに、MLEの一次条件
    /// （スコア方程式）が収束点で恒等的に成り立つことを検証する。
    ///
    /// プロパティの有効性検証（バグ注入→検出確認→元に戻す）は3件とも実施済み:
    /// `score_is_near_zero_at_converged_params`は`LogitProblem::gradient`の
    /// `diff`計算に定数オフセットを注入、`coefficients_and_se_are_invariant_to_
    /// column_order`は`LogitInput::from_columns`の`param_names`を逆順にする
    /// バグを注入、`hc0_std_errors_are_at_most_hc1_std_errors`は
    /// `nonlinear::common::sandwich_cov_params`のHC1補正係数を反転（`n/(n-k)`→
    /// `(n-k)/n`）するバグを注入し、いずれも検出できることを確認した。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        // 高k（列数依存バグ・数値的頑健性）はbenchmarkのmany_regressorsシナリオで
        // 別途カバーしているため、ここでは
        // MAX_Kを小さく保つ（分離を避けるための較正、下記`logit_case_strategy`参照）。
        const MAX_K: usize = 4;

        /// `logit_case_strategy`が生成するタプル: `(n, k, x_cols, beta, u, keys)`。
        type LogitCase = (usize, usize, Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<u64>);

        /// `(n, k, x_cols, beta, u, keys)`を生成する共通ストラテジ。
        ///
        /// `x_cols`は`-2.0..2.0`、`beta`（切片含むk+1個）は`-1.0..1.0`の一様分布
        /// （OLSの`-100.0..100.0`よりずっと狭い範囲。分離を避けるための較正、
        /// `n=100・seed多数で実測しComputationError/NonConvergenceが十分低頻度に
        /// 収まることを確認済み）。`u`はBernoulliサンプリング用の一様乱数
        /// （`0.0..1.0`）で、`y_i = u_i < p_i ? 1.0 : 0.0`として`simulate_y`内で計算する
        /// （proptestのstrategyには二値分布が無いため、閾値判定で代用）。
        fn logit_case_strategy() -> impl Strategy<Value = LogitCase> {
            (1..=MAX_K).prop_flat_map(|k| {
                (k + 20..=100usize).prop_flat_map(move |n| {
                    (
                        Just(n),
                        Just(k),
                        collection::vec(collection::vec(-2.0f64..2.0, n), k),
                        collection::vec(-1.0f64..1.0, k + 1),
                        collection::vec(0.0f64..1.0, n),
                        collection::vec(any::<u64>(), k),
                    )
                })
            })
        }

        fn x_names(k: usize) -> Vec<String> {
            (1..=k).map(|i| format!("x{i}")).collect()
        }

        /// 真の`beta`から線形予測子`z`・ロジスティック確率`p`を計算し、`u`との比較で
        /// 二値`y`をサンプリングする。
        fn simulate_y(n: usize, x_cols: &[Vec<f64>], beta: &[f64], u: &[f64]) -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let mut z = beta[0];
                    for (j, x_col) in x_cols.iter().enumerate() {
                        z += x_col[i] * beta[j + 1];
                    }
                    let p = 1.0 / (1.0 + (-z).exp());
                    if u[i] < p { 1.0 } else { 0.0 }
                })
                .collect()
        }

        /// `solver`ごとに`tol`の意味論が異なる（`docs/spec/logit-spec.md`3.2節）:
        /// `newton`は総和勾配に対する絶対閾値（既定`1e-6`）、
        /// `bfgs`/`lbfgs`は観測数`n_obs`で正規化した基準（既定`1e-8`、`run_solver`が
        /// 内部で`tol*n_obs`を実効的な絶対閾値に変換する）。`engine_pybind`側の
        /// `LogitOptions`と同じ既定値の解決をここでも行う（呼び出し元がこの対応を
        /// 誤ると実質的に閾値が数桁ずれるため、テストコード側でも明示する）。
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

        /// `SolverType::Newton`/`Bfgs`/`Lbfgs`を等確率で生成する。3手法とも収束経路・
        /// `tol`の意味論が異なるため（上記`default_options`参照）、各プロパティを
        /// solver非依存で1種類だけ検証するとBFGS/L-BFGS固有の経路がproptestの
        /// 恩恵を受けない。
        fn solver_strategy() -> impl Strategy<Value = SolverType> {
            prop_oneof![
                Just(SolverType::Newton),
                Just(SolverType::Bfgs),
                Just(SolverType::Lbfgs),
            ]
        }

        /// `ols/fit.rs`側の同名ヘルパー（`RTOL=1e-6`）より緩い`1e-4`を使う理由: OLSの
        /// 列順序不変性は閉形式解（col_piv_qr）の数値誤差のみが要因だが、Logitは
        /// 反復最適化（Newton/BFGS/L-BFGS）を経るため、列順序の違いで収束経路
        /// （標準化空間での列スケールが変わる）が変わり、`tol`ちょうどで停止する
        /// 反復回数がわずかにずれうる。この経路依存の誤差蓄積を吸収するため。
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

            /// MLEの一次条件（スコア方程式）: 収束点で`Σᵢ(yᵢ-pᵢ)xᵢ ≈ 0`が全パラメータ列で
            /// 成り立つ（`docs/spec/logit-spec.md`3.1節のスコア`∂ℓ/∂θ=X'(y-p)`が
            /// 収束点でゼロになるという、MLEの定義そのものに由来する不変条件）。
            /// 許容誤差`1e-4`は`n`に依存しない絶対閾値（`newton`の収束判定`tol=1e-6`・
            /// `bfgs`/`lbfgs`の実効閾値`tol*n_obs=1e-8*n`のいずれも、標準化空間の
            /// 勾配ノルムに対する基準であり原スケールのスコア自体は`n`でスケールしない。
            /// 実測（n<=100の256ケース）でも最大7.5e-7程度に収まることを確認済み、
            /// `1e-4`は2桁以上の安全マージンを持つ）。
            #[test]
            fn score_is_near_zero_at_converged_params(
                (n, k, x_cols, beta, u, _keys) in logit_case_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, &u);
                let names = x_names(k);
                let input = LogitInput::from_columns(&y, &x_cols, names, true, "y".to_string()).unwrap();
                let result = LogitEstimator::fit(input, default_options(CovType::Classical, solver));
                prop_assume!(result.is_ok());
                let est = result.unwrap();

                let p = est.predict();
                let x = est.input().x();
                for j in 0..=k {
                    let score: f64 = (0..n).map(|i| (y[i] - p[i]) * x.get(i, j)).sum();
                    prop_assert!(
                        score.abs() <= 1e-4,
                        "score[{j}] should be ~0, got {score} (solver={solver:?})"
                    );
                }
            }

            /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
            #[test]
            fn coefficients_and_se_are_invariant_to_column_order(
                (n, k, x_cols, beta, u, keys) in logit_case_strategy()
                    .prop_filter("need >=2 columns to permute", |(_, k, _, _, _, _)| *k >= 2),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, &u);
                let names = x_names(k);
                let input1 = LogitInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
                let result1 = LogitEstimator::fit(input1, default_options(CovType::Classical, solver));
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let mut order: Vec<usize> = (0..k).collect();
                order.sort_by_key(|&i| keys[i]);
                let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
                let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

                let input2 = LogitInput::from_columns(&y, &permuted_x, permuted_names, true, "y".to_string()).unwrap();
                let result2 = LogitEstimator::fit(input2, default_options(CovType::Classical, solver));
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
            }

            /// HC0の標準誤差は常にHC1以下（`hc1 = hc0`にn/(n-k)の小標本補正を乗じたもの、
            /// `docs/spec/logit-spec.md`3.3節）。
            #[test]
            fn hc0_std_errors_are_at_most_hc1_std_errors(
                (n, k, x_cols, beta, u, _keys) in logit_case_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, &u);
                let names = x_names(k);
                let input1 = LogitInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
                let result1 = LogitEstimator::fit(input1, default_options(CovType::Hc0, solver));
                prop_assume!(result1.is_ok());
                let est_hc0 = result1.unwrap();

                let input2 = LogitInput::from_columns(&y, &x_cols, names, true, "y".to_string()).unwrap();
                let result2 = LogitEstimator::fit(input2, default_options(CovType::Hc1, solver));
                prop_assume!(result2.is_ok());
                let est_hc1 = result2.unwrap();

                let (se_hc0, se_hc1) = (est_hc0.std_errors(), est_hc1.std_errors());
                for i in 0..=k {
                    prop_assert!(
                        se_hc0[i] <= se_hc1[i] + 1e-9,
                        "HC0 se[{i}]={} should be <= HC1 se[{i}]={}", se_hc0[i], se_hc1[i]
                    );
                }
            }
        }
    }
}
