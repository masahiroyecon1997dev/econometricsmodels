//! `ProbitEstimator::fit`の本体。検証 → 標準化 → 初期値 → 最適化 → 分離検出 → `cov_type`別の
//! 共分散 → 推論統計量 → 適合度統計量の組み立て。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use super::ProbitEstimator;
use super::fitted::ProbitFitted;
use super::input::ProbitInput;
use super::model::log_likelihood;
use super::problem::ProbitProblem;
use crate::nonlinear::common::{
    GoodnessOfFit, MleError, MleFitOptions, cov_params_for_type, destandardize_cov_params,
    destandardize_params, goodness_of_fit, log_likelihood_null, ols_based_initial_params,
    run_solver, scores_needed, standardize_columns, validate_fit_preconditions,
};
use crate::nonlinear::separation::verify_no_separation;
use crate::shared::inference;
use faer::Mat;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// [`ProbitEstimator::fit`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_probit(
    input: ProbitInput,
    options: MleFitOptions,
) -> Result<ProbitEstimator, MleError> {
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
    // （Logitと同じ経緯・同じ設計）。
    let initial_params = {
        let normal = Normal::standard();
        ols_based_initial_params(
            &x_std,
            input.x(),
            input.y(),
            input.has_intercept(),
            input.param_names(),
            // probit: IRLS重み `w = φ(Φ⁻¹(p̄))`、リンク値 `η₀ = Φ⁻¹(p̄)`。
            |p_bar| {
                let eta0 = normal.inverse_cdf(p_bar);
                (normal.pdf(eta0), eta0)
            },
        )?
    };
    let problem = ProbitProblem::from_standardized(x_std, input.y().clone());
    // `cov_type`がスコアを使う場合（`scores_needed`）、収束点でのスコア評価に標準化空間の
    // `ProbitProblem`が必要になる。`run_solver`は`problem`の所有権を取り込むため、事前に複製しておく。
    let problem_for_scores = scores_needed(&cov_type).then(|| problem.clone());

    let output = run_solver(
        problem,
        solver,
        initial_params,
        max_iter as u64,
        tol,
        n,
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
    let scores = problem_for_scores.map(|problem| problem.scores(&output.params));
    let cov_params_std = cov_params_for_type(&cov_type, &hessian_std, scores.as_ref(), n, k)?;
    let cov_params = destandardize_cov_params(&cov_params_std, &scale);

    let normal = Normal::standard();
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
    // 切片のみモデルの対数尤度: `ȳ=n1/n`の閉じた形の解析解（`fit`の再帰呼び出しは
    // 経由しない）。この式はリンク関数（標準正規CDF）に依存しないため、
    // `nonlinear/common/goodness_of_fit.rs`の`log_likelihood_null`（Logitと共通）をそのまま使う。
    let llnull = log_likelihood_null(input.y());
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

    Ok(ProbitEstimator {
        fitted: ProbitFitted {
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
mod tests;
