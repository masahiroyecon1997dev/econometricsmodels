//! `OlsEstimator::fit`の本体。検証（回帰子の有無・信頼水準・観測数・クラスター数・被説明変数）→
//! 最小二乗 → cov_type別の共分散 → t検定・信頼区間 → 適合度 → 全体F検定の順に組み立てる。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use faer::Mat;
use statrs::distribution::StudentsT;

use super::OlsEstimator;
use super::cov_params::{CovParams, compute_cov_params, validate_cluster_count};
use super::cov_type::CovType;
use super::input::OlsInput;
use crate::linear::common::LeastSquaresError;
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{
    GaussianGoodnessOfFit, gaussian_goodness_of_fit, is_perfect_fit,
};
use crate::shared::inference;
use crate::shared::least_squares::{LeastSquaresFit, least_squares, residual_sum_of_squares};
use crate::shared::validation::validate_has_regressors;
use crate::shared::wald::wald_f_test;

/// `OlsEstimator::fit`/`fit_with_optional_f_test`/`fit_auxiliary`の共通実装。契約（エラー条件）は`fit`のdocコメント参照。
///
/// `allow_undefined_f_test`が真のとき、クラスター数が傾き係数の数以下でも全体F検定を`None`にして続行する。
/// `check_outcome`が真のとき、被説明変数が定数（切片ありのみ）か完全フィットなら失敗させる。
pub(super) fn fit_ols(
    input: OlsInput,
    cov_type: CovType,
    confidence_level: f64,
    allow_undefined_f_test: bool,
    check_outcome: bool,
) -> Result<OlsEstimator, LeastSquaresError> {
    validate_has_regressors(input.nobs(), input.k())?;

    // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
    crate::shared::parallelism::ensure_serial();

    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
    }

    let n = input.nobs();
    let k = input.k();

    if n <= k {
        return Err(CommonError::InsufficientObservations { n, k }.into());
    }

    // 入力だけから判定できるクラスター数の検証は、QR分解・残差計算より前に行う
    // （`cov_params::validate_cluster_count`参照）。
    let f_test_defined = validate_cluster_count(
        &cov_type,
        n,
        k - usize::from(input.has_intercept()),
        allow_undefined_f_test,
    )?;

    // 入力だけから分かる被説明変数の検証は、QR分解より前に行う。
    if check_outcome && input.has_intercept() && input.outcome_constant() {
        return Err(CommonError::OutcomeConstant.into());
    }

    let LeastSquaresFit {
        params,
        residuals,
        xtx_inv,
    } = least_squares(input.x(), input.y()).map_err(|err| {
        LeastSquaresError::from_rank_deficient(err, input.x(), input.param_names())
    })?;

    let df_resid = n - k;
    let ssr = residual_sum_of_squares(&residuals);
    // 共分散・F検定より前に止める（残差が0だと`sigma^2`が0で、後段は0割りや無意味な桁の値になる）。
    if check_outcome && is_perfect_fit(ssr, input.outcome_sst(), k) {
        return Err(CommonError::PerfectFit.into());
    }
    let sigma2 = ssr / (df_resid as f64);

    let CovParams {
        cov_params,
        df_inference,
        hac_lags_used,
    } = compute_cov_params(&cov_type, input.x(), &residuals, &xtx_inv, sigma2, df_resid)?;

    let mut std_errors = Mat::zeros(k, 1);
    for j in 0..k {
        *std_errors.get_mut(j, 0) = (*cov_params.get(j, j)).sqrt();
    }

    let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let t_crit = inference::critical_value(&t_dist, confidence_level);

    let mut test_stats = Mat::zeros(k, 1);
    let mut p_values = Mat::zeros(k, 1);
    let mut conf_lower = Mat::zeros(k, 1);
    let mut conf_upper = Mat::zeros(k, 1);

    for j in 0..k {
        let coef = *params.get(j, 0);
        let se = *std_errors.get(j, 0);
        let stat = inference::compute_inference_stat(&t_dist, coef, se, t_crit);

        *test_stats.get_mut(j, 0) = stat.stat;
        *p_values.get_mut(j, 0) = stat.p_value;
        *conf_lower.get_mut(j, 0) = stat.conf_low;
        *conf_upper.get_mut(j, 0) = stat.conf_high;
    }

    let k_constant = usize::from(input.has_intercept());
    let GaussianGoodnessOfFit {
        r_squared,
        adj_r_squared,
        log_likelihood,
        aic,
        bic,
    } = gaussian_goodness_of_fit(input.y(), ssr, k, input.has_intercept());

    let df_model = k - k_constant;
    let f_test = if df_model == 0 || !f_test_defined {
        // 説明変数が定数項のみ（傾き係数が無い）モデルは検定対象が存在しない。
        // クラスター数が傾き係数の数以下（`fit_with_optional_f_test`のみ）では
        // `q×q`部分行列が構造的に特異で検定できない。どちらも定義なし（`None`）。
        None
    } else {
        Some(wald_f_test(
            &params,
            &cov_params,
            k_constant,
            df_model,
            df_inference,
        )?)
    };

    Ok(OlsEstimator {
        input,
        cov_type,
        params,
        residuals,
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        r_squared,
        adj_r_squared,
        f_test,
        log_likelihood,
        aic,
        bic,
        cov_params,
        df_inference,
        hac_lags_used,
    })
}

#[cfg(test)]
mod tests;
