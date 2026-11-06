//! `TobitEstimator::fit`の本体。検証 → 標準化 → 初期値 → 最適化 → `cov_type`別の共分散 →
//! 推論統計量 → 適合度統計量の組み立て。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use super::TobitEstimator;
use super::fitted::TobitFitted;
use super::input::TobitInput;
use super::problem::TobitProblem;
use super::scaling::TobitScaling;
use super::statistics::{
    log_likelihood, ols_initial_params, validate_has_uncensored_observations, wald_chi2_test,
};
use crate::nonlinear::common::{
    MleError, MleFitOptions, cov_params_for_type, run_solver, scores_needed,
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
    // `cov_type`がスコアを使う場合（`scores_needed`）、収束点でのスコア評価に標準化空間の
    // `TobitProblem`が必要になる。`run_solver`は`problem`の所有権を取り込むため、事前に複製しておく。
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
    let scores = problem_for_scores.map(|problem| problem.scores(&output.params));
    let cov_params_std = cov_params_for_type(&cov_type, &hessian_std, scores.as_ref(), n, k + 1)?;
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
mod tests;
