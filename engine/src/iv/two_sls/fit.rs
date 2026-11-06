//! `TwoSlsEstimator::fit`の本体（第一段階・第二段階・共分散・検定をつなぐオーケストレーション）。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use super::TwoSlsEstimator;
use super::cov::{CovStage, covariance_stage};
use super::sargan::sargan_test;
use super::wu_hausman::wu_hausman_test;
use crate::iv::common::{
    IvError, IvInput, compute_first_stage, hcat, mat_column_to_vec, mat_to_columns,
    validate_iv_fit_inputs, without_baked_in_intercept,
};
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::CovType;
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{is_perfect_fit, total_sum_of_squares};
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::instrumental_variables::{TwoSlsPointEstimate, two_sls_point_estimate};
use crate::shared::validation::validate_has_regressors;
use crate::shared::wald::wald_f_test;
use faer::Mat;
use statrs::distribution::StudentsT;

/// 第二段階の設計行列が回帰できる入力かを検査する。`OlsEstimator::fit`が回帰の前に行う検査
/// （回帰変数の有無・`confidence_level`の範囲）と同じ条件で、失敗は`SecondStageFailed`にする。
/// `x_endog=[]`のときは第一段階が一度も回らないので、`confidence_level`の範囲違反を検知する
/// 唯一の経路になる。観測数は`validate_iv_fit_inputs`が`n > k`を検証済み。
fn validate_second_stage(n: usize, k: usize, confidence_level: f64) -> Result<(), IvError> {
    validate_has_regressors(n, k).map_err(|source| IvError::SecondStageFailed {
        source: source.into(),
    })?;
    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(IvError::SecondStageFailed {
            source: CommonError::InvalidConfidenceLevel { confidence_level }.into(),
        });
    }
    Ok(())
}

/// [`TwoSlsEstimator::fit`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_two_sls(
    input: IvInput,
    cov_type: CovType,
    confidence_level: f64,
) -> Result<TwoSlsEstimator, IvError> {
    // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
    crate::shared::parallelism::ensure_serial();

    // 識別の順序条件、観測数`n`が構造方程式の係数の数`k`・第一段階の回帰変数の数`l`以下の入力、
    // `cov_type=Cluster`でクラスター数`g`が構造方程式の傾き係数の数`q`以下の入力、
    // 被説明変数が定数の入力を、第一段階の`OlsEstimator::fit`
    // （`FirstStageFailed`にラップされ、`k`・`q`も構造方程式のものと食い違う）より先に弾く（`compute_first_stage`より前）。
    validate_iv_fit_inputs(&input, &cov_type)?;

    // Wu-Hausman拡張回帰の`OlsInput::from_columns`が列ごとの`Vec`を受け取るため、
    // `x_exog`の列への変換は一度だけ行い使い回す。
    let x_exog_columns = mat_to_columns(input.x_exog());
    // 定数項列を新規に`OlsInput::from_columns`へ渡す際は、焼き込み済みの定数項列を
    // 除いた「素の」x_exog列を使う（`common::without_baked_in_intercept`のdoc
    // コメント参照。Wu-Hausman拡張回帰の列の構築で使う）。
    let (x_exog_bare, x_exog_names_bare) =
        without_baked_in_intercept(&x_exog_columns, input.x_exog_names(), input.has_intercept());

    // 第一段階回帰・弱操作変数診断（部分F統計量、docs/spec/iv-spec.md 3.4節）は
    // 2SLS/GMM間で共有するロジック（`common::compute_first_stage`、`iv/CLAUDE.md`
    // 「2SLSとGMMの独立実装方針」参照——GMM自体は第一段階回帰を必要としないが、
    // `engine_pybind`が`estimator="gmm"`でも同じ診断情報を独立に提供するために使う）。
    let (first_stage, weak_instrument_f_statistics) =
        compute_first_stage(&input, &cov_type, confidence_level)?;

    // 第二段階: y ~ x_exog + x̂_endog。係数と構造残差は行列版の点推定
    // （`shared::instrumental_variables::two_sls_point_estimate`）で求める。
    // `X = [x_exog, x_endog]`（実際の内生変数、推定値ではない）と`X̂ = [x_exog, x̂_endog]`は
    // 同じ列の並び。`x_exog`は定数項込み（`IvInput::x_exog`）。
    let n = input.nobs();
    let y = mat_column_to_vec(input.y(), 0);
    // `fitted_values()`は呼ぶたびに`n×1`を計算するので、列ごとに1回だけ呼ぶ。
    let fitted_columns: Vec<Mat<f64>> = first_stage
        .iter()
        .map(|(_, estimator)| estimator.fitted_values())
        .collect();
    let x_structural = hcat(input.x_exog(), input.x_endog());
    let k_exog = input.x_exog().ncols();
    let x_hat = Mat::from_fn(n, k_exog + input.k_endog(), |i, j| {
        if j < k_exog {
            *input.x_exog().get(i, j)
        } else {
            *fitted_columns[j - k_exog].get(i, 0)
        }
    });
    let k = x_hat.ncols();
    let mut param_names: Vec<String> = input.x_exog_names().to_vec();
    param_names.extend(input.x_endog_names().iter().cloned());

    validate_second_stage(n, k, confidence_level)?;
    let TwoSlsPointEstimate {
        params: beta,
        xtx_hat_inv: xtx_inv,
        residuals,
        ssr,
    } = two_sls_point_estimate(&x_structural, &x_hat, input.y()).map_err(|err| {
        IvError::SecondStageFailed {
            source: LeastSquaresError::from_rank_deficient(err, &x_hat, &param_names),
        }
    })?;

    let df_resid = n - k;

    // 構造残差が丸め誤差の範囲で0なら、標準誤差とそれを使う検定統計量は定義できない。
    // 第二段階回帰の残差ではなく構造残差で判定する（第二段階は`X̂`への回帰で別の量）。
    let sst = total_sum_of_squares(&y, None, input.has_intercept());
    if is_perfect_fit(ssr, sst, k) {
        return Err(CommonError::PerfectFit.into());
    }

    // 標準誤差の種別ごとの係数共分散行列と、t検定・信頼区間・F検定の自由度。
    let CovStage {
        cov_params,
        df_inference,
        hac_lags_used,
    } = covariance_stage(&cov_type, &x_hat, &residuals, &xtx_inv, ssr, df_resid)?;
    let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(&beta, &cov_params, &t_dist, confidence_level);

    let k_constant = usize::from(input.has_intercept());
    let r_squared = 1.0 - ssr / sst;
    let adj_r_squared = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

    let df_model = k - k_constant;
    let (wald_statistic, wald_p_value) = if df_model == 0 {
        // 説明変数が定数項のみ（傾き係数が無い）モデル。検定対象が存在しないため
        // OLSと同様NaNを返す（0除算を避ける）。
        (f64::NAN, f64::NAN)
    } else {
        wald_f_test(&beta, &cov_params, k_constant, df_model, df_inference)?
    };

    let wu_hausman = wu_hausman_test(
        &input,
        &first_stage,
        x_exog_bare,
        x_exog_names_bare,
        &y,
        &cov_type,
        confidence_level,
    )?;
    let sargan = sargan_test(&input, &residuals, ssr)?;

    Ok(TwoSlsEstimator {
        first_stage,
        weak_instrument_f_statistics,
        params: beta,
        param_names,
        dep_var_name: input.dep_var_name().to_string(),
        cov_type,
        residuals,
        std_errors,
        cov_params,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        df_resid,
        df_inference,
        hac_lags_used,
        df_model,
        r_squared,
        adj_r_squared,
        wald_statistic,
        wald_p_value,
        wu_hausman_statistic: wu_hausman.as_ref().map(|t| t.statistic),
        wu_hausman_p_value: wu_hausman.as_ref().map(|t| t.p_value),
        wu_hausman_df: wu_hausman.as_ref().map(|t| t.df),
        sargan_statistic: sargan.as_ref().map(|t| t.statistic),
        sargan_p_value: sargan.as_ref().map(|t| t.p_value),
        sargan_df: sargan.as_ref().map(|t| t.df),
    })
}

#[cfg(test)]
mod tests;
