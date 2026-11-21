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
use crate::shared::least_squares::residual_sum_of_squares;
use crate::shared::orthonormal::{OrthonormalDesign, WaldCoordinates};
use crate::shared::validation::validate_has_regressors;
use crate::shared::wald::wald_f_test;

/// QR分解より前に、入力だけから判定できる検査を行う（回帰変数の有無・信頼水準・観測数・クラスター数・
/// 被説明変数の順）。`fit_ols`と`AuxiliaryFit::fit`が同じ検査を同じ順序で行うための共通部分で、
/// どちらかだけ検査や順序を変えてしまうことを避ける。
///
/// 戻り値は、全体F検定が定義できるか（`validate_cluster_count`参照。`allow_undefined_f_test`が偽なら
/// 常に真）。
///
/// # Errors
/// `OlsEstimator::fit`が共分散より前に返すエラー。
pub(super) fn preflight(
    input: &OlsInput,
    cov_type: &CovType,
    confidence_level: f64,
    allow_undefined_f_test: bool,
    check_outcome: bool,
) -> Result<bool, LeastSquaresError> {
    validate_has_regressors(input.nobs(), input.k())?;

    // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
    crate::shared::parallelism::ensure_serial();

    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
    }

    let (n, k) = (input.nobs(), input.k());
    if n <= k {
        return Err(CommonError::InsufficientObservations { n, k }.into());
    }

    // 入力だけから判定できるクラスター数の検証は、QR分解・残差計算より前に行う
    // （`cov_params::validate_cluster_count`参照）。
    let f_test_defined = validate_cluster_count(
        cov_type,
        n,
        k - usize::from(input.has_intercept()),
        allow_undefined_f_test,
    )?;

    // 入力だけから分かる被説明変数の検証も、QR分解より前に行う。
    if check_outcome && input.has_intercept() && input.outcome_constant() {
        return Err(CommonError::OutcomeConstant.into());
    }
    Ok(f_test_defined)
}

/// `OlsEstimator::fit`/`fit_with_optional_f_test`/`fit_auxiliary`の共通実装。契約（エラー条件）は`fit`のdocコメント参照。
///
/// `allow_undefined_f_test`が真のとき、クラスター数が傾き係数の数以下でも全体F検定を`None`にして続行する。
/// `check_outcome`が真のとき、被説明変数が定数（切片ありのみ）か完全フィットなら失敗させる。
/// `release_design`が真のとき、設計行列を複製せずその場でQR分解し、推定量に設計行列を持たせない
/// （`OlsEstimator::fit_releasing_design`）。
pub(super) fn fit_ols(
    input: OlsInput,
    cov_type: CovType,
    confidence_level: f64,
    allow_undefined_f_test: bool,
    check_outcome: bool,
    release_design: bool,
) -> Result<OlsEstimator, LeastSquaresError> {
    let mut input = input;
    let (n, k) = (input.nobs(), input.k());
    let f_test_defined = preflight(
        &input,
        &cov_type,
        confidence_level,
        allow_undefined_f_test,
        check_outcome,
    )?;

    // 直交座標（`X = QR`、`b̃ = Q'y`）を係数・残差・共分散・F検定の全てに使う。ランク落ちは`R`の
    // 列ピボットQRで判定する（n×kの列ピボットQRと`(X'X)⁻¹`を別に求めない）。
    let design = if release_design {
        let x = input.take_x();
        OrthonormalDesign::new_checked_owned(x).map_err(|err| {
            LeastSquaresError::from_rank_deficient(err.kind, &err.x, input.param_names())
        })?
    } else {
        OrthonormalDesign::new_checked(input.x()).map_err(|err| {
            LeastSquaresError::from_rank_deficient(err, input.x(), input.param_names())
        })?
    };
    let coordinates = design.coordinates(input.y());
    let params = design.coefficients(&coordinates);
    // 設計行列を手放すときは、当てはめ値`Q·b̃`をここで求めて持つ（残差は`y - Q·b̃`で、
    // `design.residuals`と同じ演算）。
    let (residuals, released_fitted_values) = if release_design {
        let fitted = design.q() * &coordinates;
        (input.y() - &fitted, Some(fitted))
    } else {
        (design.residuals(input.y(), &coordinates), None)
    };

    let df_resid = n - k;
    let ssr = residual_sum_of_squares(&residuals);
    // 共分散・F検定より前に止める（残差が0だと`sigma^2`が0で、後段は0割りや無意味な桁の値になる）。
    if check_outcome && is_perfect_fit(ssr, input.outcome_sst(), k) {
        return Err(CommonError::PerfectFit.into());
    }
    let sigma2 = ssr / (df_resid as f64);

    // 係数の共分散と全体Wald/F検定は、直交座標（`X = QR`、`b̃ = Q'y`、ミート`M̃`）で求める。
    // `V = B M B`（`B = (X'X)⁻¹`）を作ってから反転すると、傾きの相関が極端に高いときに`B`の
    // 明示的な逆行列で桁落ちして数値的に正定値でなくなる。`M̃`は`x = Q`・`xtx_inv = 単位行列`で
    // 既存のmeat構築部品を呼んで得て、`V = R⁻¹ M̃ R⁻ᵀ`と統計量`b̃_S' M̃_SS⁻¹ b̃_S`の両方に使う
    // （`shared::orthonormal`）。`Q`は共分散の計算後に捨てる。
    let CovParams {
        cov_params: meat,
        df_inference,
        hac_lags_used,
    } = compute_cov_params(
        &cov_type,
        design.q(),
        &residuals,
        &design.identity(),
        sigma2,
        df_resid,
    )?;
    let cov_params = design.coefficient_covariance(&meat);
    drop(design);
    let wald = WaldCoordinates { coordinates, meat };

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
            &wald.coordinates,
            &wald.meat,
            k_constant,
            df_model,
            df_inference,
        )?)
    };

    Ok(OlsEstimator {
        input,
        released_fitted_values,
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
        wald,
        df_inference,
        hac_lags_used,
    })
}

#[cfg(test)]
mod tests;
