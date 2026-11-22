//! `TobitEstimator::fit`の本体。検証 → 標準化 → 初期値 → 最適化 → `cov_type`別の共分散 →
//! 推論統計量 → 適合度統計量の組み立て。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use super::TobitEstimator;
use super::coordinates::DesignCoordinates;
use super::fitted::TobitFitted;
use super::input::TobitInput;
use super::problem::TobitProblem;
use super::scaling::TobitScaling;
use super::statistics::{log_likelihood, validate_has_uncensored_observations};
use crate::nonlinear::common::{
    MleError, MleFitOptions, cov_params_for_type, run_solver_keeping_problem, scores_needed,
    validate_cluster_cov_type, validate_mle_options, validate_outcome_not_constant,
    validate_sufficient_observations,
};
use crate::shared::inference;
use crate::shared::normal::StandardNormal;
use crate::shared::wald::wald_chi2_test;
use faer::Mat;

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
    // 打ち切り境界自体と、`y`と境界の整合性（`TobitInput::from_columns`は次元だけを検証する）。
    input.validate_censoring_bounds()?;

    let n = input.nobs();
    let k = input.k();
    validate_sufficient_observations(n, k + 1)?;
    validate_cluster_cov_type(&cov_type, n, k - usize::from(input.has_intercept()))?;
    validate_has_uncensored_observations(input.y(), input.lower(), input.upper())?;
    // 非打ち切り観測がある（全行が境界に一致する場合は上で`NoUncensoredObservations`になる）うえで、
    // 切片ありなのに`y`が全行で同じ値だと`σ→0`に退化して`NOT_CONVERGED`になるため、先に弾く。
    validate_outcome_not_constant(input.y(), input.has_intercept())?;

    // `x`（切片ありなら列を平均センタリング＋スケーリング、切片なしはスケーリング
    // のみ）と`y`（と打ち切り境界、`y`の母集団標準偏差を2の冪に丸めた値で一律
    // スケーリング）を標準化してから最適化する（`TobitScaling`のdocコメント・
    // `fit`のdocコメント「標準化空間」節参照）。
    let scaling = TobitScaling::fit(input.x(), input.y(), input.has_intercept());
    let x_std = scaling.standardize_x(input.x());
    let y_std = scaling.standardize_y(input.y());
    let lower_std = scaling.standardize_bound(input.lower());
    let upper_std = scaling.standardize_bound(input.upper());
    // 標準化した設計行列を直交座標`Q = X̃R⁻¹`に移し、`Q`の座標`b = Rβ̃`で最適化する
    // （`DesignCoordinates`のdocコメント）。ランク判定と条件数のガードもここで行う。
    let (coordinates, q) = DesignCoordinates::new(x_std, input.x(), input.param_names())?;
    let initial_params = coordinates.initial_params(&q, &y_std);
    let problem = TobitProblem::from_standardized(q, y_std, lower_std, upper_std);

    let (output, problem) = run_solver_keeping_problem(
        problem,
        solver,
        initial_params,
        max_iter as u64,
        tol,
        n,
        raise_on_non_convergence,
    )?;
    // `cov_type`がスコアを使う場合（`scores_needed`）、収束点でのスコア評価に標準化空間の問題が
    // 要る。`run_solver_keeping_problem`が返す問題をそのまま使い（最適化の前に複製しない）、
    // 評価したらすぐ手放す。
    let scores = scores_needed(&cov_type).then(|| problem.scores(&output.params));
    drop(problem);
    // Tobitの真の分離は係数発散ではなく`σ→0`退化として現れるため、Logit/Probitの分離検出
    // （`nonlinear::separation`）は行わない。全件打ち切りは
    // `validate_has_uncensored_observations`（`NoUncensoredObservations`）、
    // 部分的な準完全分離は`NonConvergence`で捕捉される。

    // `Q`の座標の最適解`(b, logσ̃)`を標準化空間`(β̃, s̃)`（`β̃ = R⁻¹b`）に戻し、元のスケールの
    // `(β, logσ)`へ逆変換する（`TobitScaling`のdocコメントの式）。
    let params_std = coordinates.standardized_params(&output.params);
    let params_full = scaling.destandardize_params(&params_std);
    let sigma = params_full[k].exp();
    let params = params_full[..k].to_vec();

    // `cov_type`別の共分散は`Q`の座標（`H`・スコアが良条件）で求め、`R⁻¹`で標準化空間に戻す。
    let hessian_q = Mat::from_fn(k + 1, k + 1, |i, j| output.hessian[i][j]);
    let cov_q = cov_params_for_type(&cov_type, &hessian_q, scores.as_ref(), n, k + 1)?;
    let cov_params_std = coordinates.standardized_covariance(&cov_q);
    // 標準化空間`(β̃, s̃)`の分散共分散行列を、元のスケール`(β, σ)`空間へ
    // `Cov(β,σ) = J·Cov(β̃,s̃)·Jᵀ`で変換する（`J`は`TobitScaling::param_jacobian`。
    // `x`の列標準化・センタリングの解除と、`s̃→σ=c·exp(s̃)`のデルタ法`∂σ/∂s̃=σ`を
    // 1本のヤコビアンに合成済み。`cov_type`の種類に依存しない共通の後処理、
    // `tobit.rs`の`TobitEstimator::fit`のdocコメント「標準化空間」「σのSE」節参照）。
    let cov_params = scaling.destandardize_cov_params(&cov_params_std, sigma);

    let normal = StandardNormal;
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
        // 検定対象は`β`の傾き係数で、`σ`は含まれない。`Q`の座標`b = Rβ̃`は`R`が上三角で、先頭が
        // 切片・末尾が傾きなので、末尾`df_model`個の傾きの統計量は`b_S'V_b,SS⁻¹b_S`（`V_b`は
        // `Q`の座標の共分散）。標準化は傾きを対角行列で変換するだけなので、元のスケールの
        // `β_S'V_SS⁻¹β_S`と同じ値になる。この等価性は、`R`が上三角で、傾きが末尾に並び、切片列が
        // 標準化後も先頭で傾きと混ざらない（切片列は平均を引かない）ことに依る。列の並びや切片の
        // 標準化の扱いを変えるときは、ここも見直す。
        let b = Mat::from_fn(k, 1, |i, _| output.params[i]);
        wald_chi2_test(&b, &cov_q, k_constant, df_model)?
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
