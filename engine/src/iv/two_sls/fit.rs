//! `TwoSlsEstimator::fit`の本体（第一段階・第二段階・共分散・検定をつなぐオーケストレーション）。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use super::TwoSlsEstimator;
use super::cov::{CovStage, covariance_stage, wald_f_test, xtx_inverse};
use super::sargan::sargan_test;
use super::wu_hausman::wu_hausman_test;
use crate::iv::common::{
    IvError, IvInput, compute_first_stage, hcat, mat_column_to_vec, mat_to_columns,
    validate_iv_fit_inputs, without_baked_in_intercept,
};
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{is_perfect_fit, total_sum_of_squares};
use crate::shared::inference::{InferenceTable, inference_table};
use faer::Mat;
use statrs::distribution::StudentsT;

/// `second_stage`（`β̂`・`X̂`を得るためだけの内部委譲フィット）にのみ使う固定`cov_type`。
/// `two_sls.rs`のモジュールdoc「`second_stage`フィールドの位置づけ」参照。呼び出し元が指定した
/// `cov_type`は、独立実装のサンドイッチ計算（`fit()`本体）側で使う。
const SECOND_STAGE_COV_TYPE: CovType = CovType::Classical;

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
    // 被説明変数が定数の入力を、第一段階・第二段階の`OlsEstimator::fit`
    // （`FirstStageFailed`/`SecondStageFailed`にラップされ、`k`・`q`も構造方程式の
    // ものと食い違う）より先に弾く（`compute_first_stage`より前）。
    validate_iv_fit_inputs(&input, &cov_type)?;

    // 第二段階・Wu-Hausman拡張回帰の`OlsInput::from_columns`が列ごとの`Vec`を受け取るため、
    // `x_exog`の列への変換は一度だけ行い使い回す。
    let x_exog_columns = mat_to_columns(input.x_exog());
    // 定数項列を新規に`OlsInput::from_columns`へ渡す際は、焼き込み済みの定数項列を
    // 除いた「素の」x_exog列を使う（`common::without_baked_in_intercept`のdoc
    // コメント参照。`second_stage_columns`/`hausman_columns`の構築で使う）。
    let (x_exog_bare, x_exog_names_bare) =
        without_baked_in_intercept(&x_exog_columns, input.x_exog_names(), input.has_intercept());

    // 第一段階回帰・弱操作変数診断（部分F統計量、docs/spec/iv-spec.md 3.4節）は
    // 2SLS/GMM間で共有するロジック（`common::compute_first_stage`、`iv/CLAUDE.md`
    // 「2SLSとGMMの独立実装方針」参照——GMM自体は第一段階回帰を必要としないが、
    // `engine_pybind`が`estimator="gmm"`でも同じ診断情報を独立に提供するために使う）。
    let (first_stage, weak_instrument_f_statistics) =
        compute_first_stage(&input, &cov_type, confidence_level)?;
    let x_endog_hat_columns: Vec<Vec<f64>> = first_stage
        .iter()
        .map(|(_, estimator)| {
            let fitted: Mat<f64> = estimator.fitted_values();
            mat_column_to_vec(&fitted, 0)
        })
        .collect();

    // 第二段階: y ~ x_exog + x̂_endog
    let mut second_stage_columns: Vec<Vec<f64>> = x_exog_bare.to_vec();
    second_stage_columns.extend(x_endog_hat_columns);
    let mut second_stage_names: Vec<String> = x_exog_names_bare.to_vec();
    second_stage_names.extend(input.x_endog_names().iter().cloned());

    let y = mat_column_to_vec(input.y(), 0);
    // 第一段階と同じ理由（`y`・`second_stage_columns`いずれも`IvInput`由来かつ
    // `x_endog_hat_columns`は`n`行の`fitted_values()`から取り出しているため）で、
    // `DimensionMismatch`は理論上到達不能。
    let second_stage_input = OlsInput::from_columns(
        &y,
        &second_stage_columns,
        second_stage_names,
        input.has_intercept(),
        input.dep_var_name().to_string(),
    )
    .map_err(|source| IvError::SecondStageFailed { source })?;
    let second_stage =
        OlsEstimator::fit_auxiliary(second_stage_input, SECOND_STAGE_COV_TYPE, confidence_level)
            .map_err(|source| IvError::SecondStageFailed { source })?;

    // ── ここから独立実装のサンドイッチ型SE計算（`two_sls.rs`のモジュールdoc参照）──
    let x_hat = second_stage.input().x(); // X̂ = [x_exog, x̂_endog]
    let beta = second_stage.params();
    let n = x_hat.nrows();
    let k = x_hat.ncols();

    // 構造残差 e = y - Xβ̂。X = [x_exog, x_endog]（実際の内生変数、推定値ではない）。
    // 列の並びは第二段階の設計行列（x_exog ++ x̂_endog）と揃える必要があるため、
    // 同じ順序（x_exog ++ x_endog）で組み立てる。
    let x_structural = hcat(input.x_exog(), input.x_endog());
    let residuals = input.y() - &x_structural * beta;

    let df_resid = n - k;
    let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();

    // 構造残差が丸め誤差の範囲で0なら、標準誤差とそれを使う検定統計量は定義できない。
    // 第二段階回帰の残差ではなく構造残差で判定する（第二段階は`X̂`への回帰で別の量）。
    let sst = total_sum_of_squares(&y, None, input.has_intercept());
    if is_perfect_fit(ssr, sst, k) {
        return Err(CommonError::PerfectFit.into());
    }

    let xtx_inv = xtx_inverse(x_hat, k)?;

    // 標準誤差の種別ごとの係数共分散行列と、t検定・信頼区間・F検定の自由度。
    let CovStage {
        cov_params,
        df_inference,
        hac_lags_used,
    } = covariance_stage(&cov_type, x_hat, &residuals, &xtx_inv, ssr, df_resid)?;
    let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(beta, &cov_params, &t_dist, confidence_level);

    let k_constant = usize::from(input.has_intercept());
    let r_squared = 1.0 - ssr / sst;
    let adj_r_squared = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

    let df_model = k - k_constant;
    let (wald_statistic, wald_p_value) = if df_model == 0 {
        // 説明変数が定数項のみ（傾き係数が無い）モデル。検定対象が存在しないため
        // OLSと同様NaNを返す（0除算を避ける）。
        (f64::NAN, f64::NAN)
    } else {
        wald_f_test(beta, &cov_params, k_constant, df_model, df_inference)?
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
        second_stage,
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
