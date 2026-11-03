//! `TwoSlsEstimator::fit`の本体（第一段階・第二段階・共分散・検定をつなぐオーケストレーション）。

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
mod tests {
    use crate::iv::common::{IvError, IvEstimate, IvInput};
    use crate::iv::two_sls::TwoSlsEstimator;
    use crate::iv::two_sls::test_support::*;
    use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
    use crate::shared::error::CommonError;
    use crate::shared::group_codes::GroupCodes;
    use faer::Mat;

    #[allow(clippy::type_complexity)]
    fn nontrivial_x_exog_fitted_estimator() -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        TwoSlsEstimator,
    ) {
        let (x1, x_endog, z1, z2, y) = nontrivial_x_exog_columns();
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        (x1, x_endog, z1, z2, y, estimator)
    }

    /// 操作変数`z`が内生変数`x_endog`を完全に予測するデータ（`x_endog = z`、`y = 1 + 2*x_endog`）。
    /// 第一段階の残差が全て0になるため、第一段階は完全フィットとして拒否される。
    fn perfectly_predicted_endog_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = z.clone();
        let y: Vec<f64> = x_endog.iter().map(|&x| 1.0 + 2.0 * x).collect();
        (y, x_endog, z)
    }

    #[test]
    fn fit_matches_the_closed_form_iv_estimator_when_just_identified() {
        // 丁度識別・単一の操作変数では2SLSは`beta1 = cov(z, y) / cov(z, x)`、
        // `beta0 = mean(y) - beta1 * mean(x)`の閉形式解と一致する。
        let (y, x_endog, z) = just_identified_data();
        let n = y.len() as f64;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
        let (y_bar, x_bar, z_bar) = (mean(&y), mean(&x_endog), mean(&z));
        let cov_zy: f64 = z
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - z_bar) * (b - y_bar))
            .sum();
        let cov_zx: f64 = z
            .iter()
            .zip(&x_endog)
            .map(|(a, b)| (a - z_bar) * (b - x_bar))
            .sum();
        let beta1 = cov_zy / cov_zx;
        let beta0 = y_bar - beta1 * x_bar;

        let estimator =
            TwoSlsEstimator::fit(just_identified_input(), CovType::Classical, 0.95).unwrap();

        assert_eq!(estimator.param_names(), ["const", "x_endog"]);
        assert!((*estimator.params().get(0, 0) - beta0).abs() < 1e-8);
        assert!((*estimator.params().get(1, 0) - beta1).abs() < 1e-8);
    }

    #[test]
    fn fit_returns_the_first_stage_perfect_fit_when_instrument_perfectly_predicts_endog() {
        // 操作変数が内生変数を完全に予測する（第一段階の残差が全て0）と、第一段階の標準誤差と
        // 部分F統計量が定義できない。第一段階の文脈つきで完全フィットとして拒否する。
        let (y, x_endog, z) = perfectly_predicted_endog_data();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            IvError::FirstStageFailed {
                endog_name: "x_endog".to_string(),
                source: CommonError::PerfectFit.into(),
            }
        );
    }

    #[test]
    fn fit_returns_perfect_fit_when_the_structural_residuals_are_zero() {
        // 構造式`y = 1 + 2*x_endog`が厳密に成り立つと（第一段階は完全フィットではない）、
        // 構造残差が全て0で標準誤差が定義できない。第二段階回帰ではなく構造残差で判定する。
        let (_, x_endog, z) = just_identified_data();
        let y: Vec<f64> = x_endog.iter().map(|&x| 1.0 + 2.0 * x).collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::PerfectFit)
        );
    }

    #[test]
    fn fit_does_not_judge_the_second_stage_regression_for_a_perfect_fit() {
        // y = 1 + 2*x̂（x̂は第一段階の当てはめ値）は、第二段階の回帰`y ~ const + x̂`が厳密に成り立つ
        // が、構造残差`y - 1 - 2*x_endog = -2v`は0ではない。標準誤差の基準は構造残差なので、
        // 第二段階の回帰は完全フィットの判定をしない（`fit_auxiliary`）。
        let base = TwoSlsEstimator::fit(just_identified_input(), CovType::Classical, 0.95).unwrap();
        let v = base.first_stage_estimators()[0].1.residuals();
        let (_, x_endog, z) = just_identified_data();
        let y: Vec<f64> = (0..5)
            .map(|i| 1.0 + 2.0 * (x_endog[i] - *v.get(i, 0)))
            .collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!((*estimator.params().get(1, 0) - 2.0).abs() < 1e-8);
        assert!((0..2).all(|j| estimator.std_errors().get(j, 0).is_finite()));
    }

    #[test]
    fn fit_returns_outcome_constant_for_a_constant_endogenous_regressor_in_its_first_stage() {
        // 定数の内生変数は、その第一段階の被説明変数が定数。第一段階の文脈つきで報告する。
        let (y, _, z) = just_identified_data();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[vec![2.0; 5]],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            IvError::FirstStageFailed {
                endog_name: "x_endog".to_string(),
                source: CommonError::OutcomeConstant.into(),
            }
        );
    }

    #[test]
    fn fit_returns_outcome_constant_before_the_first_stage() {
        let (_, x_endog, z) = just_identified_data();
        let input = IvInput::from_columns(
            &[3.0; 5],
            &[],
            vec![],
            &[x_endog],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::OutcomeConstant)
        );
    }

    #[test]
    fn fit_pins_faer_global_parallelism_to_seq() {
        // `fit()` 冒頭の `crate::shared::parallelism::ensure_serial()` が faer の
        // グローバル並列度を `Par::Seq` へ引き戻すことの回帰ガード（iv 系統代表）。
        // 別テストが `Seq` にしている可能性があるため、まず `Rayon` に戻してから通す。
        // 設計行列は極小なので一時的な `Rayon` 設定は病理を招かない。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let _ = TwoSlsEstimator::fit(just_identified_input(), CovType::Classical, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    #[test]
    fn fit_matches_ols_when_x_endog_and_instruments_are_empty() {
        // `x_endog=[]`かつ`instruments=[]`の退化ケース（`IvInput`のdocコメント参照）は
        // 2SLSが素のOLSと数値的に一致するはず。
        let y = vec![2.1, 3.8, 6.3, 7.9, 10.2];
        let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        assert_eq!(estimator.param_names(), ["const", "x1"]);
        assert!(estimator.first_stage_estimators().is_empty());
        assert!((*estimator.params().get(0, 0) - 0.0).abs() < 0.5);
        assert!((*estimator.params().get(1, 0) - 2.0).abs() < 0.2);

        // `x_endog=[]`だと`X̂ = X`（推定値=実際の値）のため、SE・適合度統計量も
        // 素のOLSと数値的に一致するはず（`OlsEstimator`への直接fitと照合）。
        let ols_input =
            OlsInput::from_columns(&y, &x_exog, vec!["x1".to_string()], true, "y".to_string())
                .unwrap();
        let ols_estimator = OlsEstimator::fit(ols_input, CovType::Classical, 0.95).unwrap();
        for j in 0..2 {
            assert!(
                (*estimator.std_errors().get(j, 0) - *ols_estimator.std_errors().get(j, 0)).abs()
                    < 1e-8
            );
            assert!(
                (*estimator.test_stats().get(j, 0) - *ols_estimator.test_stats().get(j, 0)).abs()
                    < 1e-8
            );
        }
        assert!((estimator.r_squared() - ols_estimator.r_squared()).abs() < 1e-8);
        assert!((estimator.wald_statistic() - ols_estimator.f_statistic().unwrap()).abs() < 1e-8);
        assert!((estimator.wald_p_value() - ols_estimator.f_p_value().unwrap()).abs() < 1e-8);
        for j in 0..2 {
            assert!(
                (*estimator.p_values().get(j, 0) - *ols_estimator.p_values().get(j, 0)).abs()
                    < 1e-8
            );
            assert!(
                (*estimator.conf_lower().get(j, 0) - *ols_estimator.conf_lower().get(j, 0)).abs()
                    < 1e-8
            );
            assert!(
                (*estimator.conf_upper().get(j, 0) - *ols_estimator.conf_upper().get(j, 0)).abs()
                    < 1e-8
            );
        }
        assert_eq!(estimator.df_resid(), 3);
        assert_eq!(estimator.df_model(), 1);
    }

    /// 説明変数が定数項のみ（傾き係数が無い、`df_model=0`）の退化モデルでは、F検定の
    /// 対象が存在しないため`wald_statistic`/`wald_p_value`は`NaN`になる（`shared/wald.rs`の
    /// `wald_f_test`と同じ0除算回避の扱い、`two_sls.rs`のモジュールdoc参照）。
    #[test]
    fn fit_sets_f_statistic_and_f_p_value_to_nan_for_const_only_model() {
        let y = vec![2.1, 3.8, 6.3, 7.9, 10.2];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        assert_eq!(estimator.df_model(), 0);
        assert!(estimator.wald_statistic().is_nan());
        assert!(estimator.wald_p_value().is_nan());
    }

    #[test]
    fn fit_returns_second_stage_failed_when_confidence_level_is_invalid_and_x_endog_is_empty() {
        // `x_endog=[]`だと第一段階ループが一度も回らないため、`confidence_level`の
        // 範囲チェックは第二段階の`OlsEstimator::fit`経由でのみ働く（`two_sls.rs`の
        // モジュールdoc参照）。
        let y = vec![2.1, 3.8, 6.3, 7.9, 10.2];
        let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 1.5);
        assert_eq!(
            result.unwrap_err(),
            IvError::SecondStageFailed {
                source: crate::linear::common::LeastSquaresError::Common(
                    CommonError::InvalidConfidenceLevel {
                        confidence_level: 1.5
                    }
                ),
            }
        );
    }

    #[test]
    fn fit_returns_insufficient_instruments_error_when_under_identified() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![1.0, 1.0, 2.0, 2.0, 3.0]];
        let instruments = vec![vec![2.0, 1.0, 4.0, 3.0, 6.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string(), "endog2".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            }
        );
    }

    #[test]
    fn fit_succeeds_when_over_identified() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0]];
        let instruments = vec![
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        assert_eq!(estimator.nobs(), 6);
        assert_eq!(estimator.k(), 2);
        assert_eq!(estimator.dep_var_name(), "y");
        assert_eq!(estimator.first_stage_estimators().len(), 1);
        assert_eq!(estimator.first_stage_estimators()[0].0, "endog1");
    }

    /// 呼び出し元が指定した`cov_type`は第一段階（`OlsEstimator`委譲）・第二段階
    /// （`TwoSlsEstimator`自身が独立に計算する正しいSE）の両方に反映される
    /// （`two_sls.rs`のモジュールdoc「第一段階・第二段階での`cov_type`/
    /// `confidence_level`の扱い」参照）。内部実装専用の`second_stage`フィールド自身の
    /// 委譲フィットだけは常に`Classical`のまま（`two_sls.rs`のモジュールdoc
    /// 「`second_stage`フィールドの位置づけ」参照）。`second_stage`は非公開フィールド
    /// だが、このテストは同一モジュールの子モジュールのため直接参照できる。
    #[test]
    fn fit_uses_caller_provided_cov_type_for_first_stage_and_second_stage_se() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0]];
        let instruments = vec![
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Hc0, 0.95).unwrap();
        assert_eq!(
            estimator.first_stage_estimators()[0].1.cov_type(),
            &CovType::Hc0
        );
        assert_eq!(estimator.cov_type(), &CovType::Hc0);
        assert_eq!(estimator.second_stage.cov_type(), &CovType::Classical);
    }

    /// `confidence_level`が第一段階に反映されていることを、信頼区間の幅の変化で間接的に
    /// 確認する（`OlsEstimator`は`confidence_level`自体を公開するgetterを持たないため）。
    #[test]
    fn fit_uses_caller_provided_confidence_level_for_first_stage() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0]];
        let instruments = vec![
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        ];
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                &x_endog,
                vec!["endog1".to_string()],
                &instruments,
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let narrow = TwoSlsEstimator::fit(build_input(), CovType::Classical, 0.80).unwrap();
        let wide = TwoSlsEstimator::fit(build_input(), CovType::Classical, 0.99).unwrap();

        let narrow_first_stage = &narrow.first_stage_estimators()[0].1;
        let wide_first_stage = &wide.first_stage_estimators()[0].1;
        let narrow_width =
            *narrow_first_stage.conf_upper().get(0, 0) - *narrow_first_stage.conf_lower().get(0, 0);
        let wide_width =
            *wide_first_stage.conf_upper().get(0, 0) - *wide_first_stage.conf_lower().get(0, 0);
        assert!(
            wide_width > narrow_width,
            "wide={wide_width}, narrow={narrow_width}"
        );
    }

    /// 2SLSの射影公式`β̂ = (X'PzX)⁻¹X'Pzy`（`Pz=Z(Z'Z)⁻¹Z'`）を`faer`の行列演算で直接計算する。
    /// `TwoSlsEstimator::fit`（二段階回帰による構成）とは独立した実装で、両者が数値的に
    /// 一致することを確認するために使う（`fit_matches_independently_recomputed_
    /// projection_formula_*`のテスト群参照）。
    fn recompute_2sls_params_via_projection_formula(
        z: &Mat<f64>,
        x: &Mat<f64>,
        y: &Mat<f64>,
    ) -> Mat<f64> {
        use faer::Side;
        use faer::prelude::Solve;

        let k_z = z.ncols();
        let k_x = x.ncols();

        let ztz = z.transpose() * z;
        let ztx = z.transpose() * x;
        let zty = z.transpose() * y;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k_z, k_z));

        let pz_x = &ztz_inv * &ztx;
        let pz_y = &ztz_inv * &zty;
        let xt_pz_x = ztx.transpose() * &pz_x;
        let xt_pz_y = ztx.transpose() * &pz_y;
        let xt_pz_x_inv = xt_pz_x
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k_x, k_x));
        &xt_pz_x_inv * &xt_pz_y
    }

    fn assert_params_close(actual: &Mat<f64>, expected: &Mat<f64>) {
        assert_eq!(actual.nrows(), expected.nrows());
        for i in 0..actual.nrows() {
            assert!(
                (*actual.get(i, 0) - *expected.get(i, 0)).abs() < 1e-8,
                "param {i}: got {}, expected {}",
                *actual.get(i, 0),
                *expected.get(i, 0)
            );
        }
    }

    /// 他のテストは`x_endog`を操作変数が完全予測する退化ケースのみのため、内生性が残る
    /// 一般的なケース（`x_exog`が定数項のみ）をこちらでカバーする。
    #[test]
    fn fit_matches_independently_recomputed_projection_formula_when_over_identified() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0]];
        let instruments = vec![
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let n = y.len();
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => instruments[0][i],
            _ => instruments[1][i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[0][i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let expected_beta = recompute_2sls_params_via_projection_formula(&z, &x, &y_mat);
        assert_params_close(estimator.params(), &expected_beta);
    }

    /// `r_squared`/`adj_r_squared`/`df_resid`/`df_model`を、構造残差のSSR・元の`y`のTSSから
    /// 素朴な式で独立計算し照合する。
    #[test]
    fn fit_computes_r_squared_matching_manual_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        let (_x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let (_x1, _x_endog, _z1, _z2, y) = nontrivial_x_exog_columns();
        let n = y.len();
        let k = 3;

        let ssr: f64 = (0..n).map(|i| (*e.get(i, 0)).powi(2)).sum();
        let y_mean: f64 = y.iter().sum::<f64>() / (n as f64);
        let sst: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
        let expected_r_squared = 1.0 - ssr / sst;
        let expected_df_resid = n - k;
        let expected_adj_r_squared =
            1.0 - ((n - 1) as f64 / expected_df_resid as f64) * (1.0 - expected_r_squared);

        assert!((estimator.r_squared() - expected_r_squared).abs() < 1e-8);
        assert!((estimator.adj_r_squared() - expected_adj_r_squared).abs() < 1e-8);
        assert_eq!(estimator.df_resid(), expected_df_resid);
        assert_eq!(estimator.df_model(), k - 1);
    }

    /// 第一段階回帰（`x_endog[j] ~ x_exog + instruments`）の係数を、通常のOLS閉形式解
    /// `β̂=(Z'Z)⁻¹Z'x_endog`（`Z=[x_exog, instruments]`）で独立に計算し、
    /// `first_stage_estimators()`が返す`OlsEstimator`の`params()`と数値一致することを
    /// 確認する（既存テストは操作変数が内生変数を完全予測する退化ケースでしか第一段階を
    /// 検証しておらず、一般的な非退化ケースでの独立検証が無かったため追加）。
    #[test]
    fn first_stage_estimators_match_independently_recomputed_ols_closed_form() {
        use faer::Side;
        use faer::prelude::Solve;

        let (x1, x_endog, z1, z2, _y, estimator) = nontrivial_x_exog_fitted_estimator();

        let n = x1.len();
        // 第一段階の設計行列 Z = [const, x1, z1, z2]（`x_exog ++ instruments`のunion）
        let z = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let x_endog_mat = Mat::from_fn(n, 1, |i, _| x_endog[i]);

        let ztz = z.transpose() * &z;
        let zt_x_endog = z.transpose() * &x_endog_mat;
        let expected_beta = ztz.llt(Side::Lower).unwrap().solve(zt_x_endog);

        assert_eq!(estimator.first_stage_estimators().len(), 1);
        let (name, first_stage) = &estimator.first_stage_estimators()[0];
        assert_eq!(name, "endog1");
        assert_params_close(first_stage.params(), &expected_beta);
    }

    /// `x_exog`が定数項のみではなく実変数を含む、教科書的な2SLSの典型ケース
    /// （外生変数＋内生変数＋操作変数が同時に存在する）を射影公式で独立検証する。
    #[test]
    fn fit_matches_independently_recomputed_projection_formula_with_nontrivial_x_exog() {
        let (x1, x_endog, z1, z2, y, estimator) = nontrivial_x_exog_fitted_estimator();
        assert_eq!(estimator.param_names(), ["const", "x1", "endog1"]);

        let n = y.len();
        // Z = [const, x1, z1, z2]（全操作変数）, X = [const, x1, x_endog]
        let z = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            _ => x_endog[i],
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let expected_beta = recompute_2sls_params_via_projection_formula(&z, &x, &y_mat);
        assert_params_close(estimator.params(), &expected_beta);
    }

    #[test]
    fn fit_returns_second_stage_failed_when_second_stage_design_matrix_is_singular() {
        // 操作変数`z`と内生変数`x_endog`の（中心化後）共分散が0だと、第一段階の予測値
        // `x̂_endog`は定数（`x_endog`の標本平均）になる。すると第二段階の設計行列
        // `[const, x̂_endog]`が定数列同士の完全な多重共線性になり特異行列エラーになる
        // （弱操作変数の極端なケース。第一段階自体は特異にならない、`z`は`const`とは
        // 独立に変動するため）。
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let x_endog = vec![1.0, 4.0, 4.0, 1.0]; // mean 2.5, cov(z, x_endog) = 0
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["endog1".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::SecondStageFailed {
                source: crate::linear::common::LeastSquaresError::SingularMatrix {
                    columns: vec!["const".to_string(), "endog1".to_string()]
                },
            }
        );
    }

    #[test]
    fn fit_returns_first_stage_failed_when_first_stage_design_matrix_is_singular() {
        // 操作変数が全て同一値（分散ゼロ）だと、第一段階の設計行列（定数項+その列）が
        // 完全な多重共線性になり特異行列エラーになる。
        let y = vec![1.0, 2.0, 3.0];
        let x_endog = vec![vec![1.0, 2.0, 3.0]];
        let instruments = vec![vec![5.0, 5.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::FirstStageFailed {
                endog_name: "endog1".to_string(),
                source: crate::linear::common::LeastSquaresError::SingularMatrix {
                    columns: vec!["const".to_string(), "z1".to_string()]
                },
            }
        );
    }

    /// 観測数が構造方程式の係数の数以下なら、第一段階の`FirstStageFailed`ではなく
    /// 構造方程式の`n`・`k`で`CommonError::InsufficientObservations`を返す。
    #[test]
    fn fit_reports_structural_insufficient_observations_before_the_first_stage() {
        // n=2、構造方程式は切片+endog1でk=2、第一段階は切片+z1..z3でl=4。
        let y = vec![1.0, 2.0];
        let x_endog = vec![vec![1.0, 2.0]];
        let instruments = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 4.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
    }

    /// 構造方程式は推定できる（`n > k`）が第一段階の回帰変数の数`l`以下の観測数なら、
    /// 第一段階の観測数不足として専用エラーで拒否する（`n = l`では`X̂ = X`で2SLSがOLSに縮退する）。
    #[test]
    fn fit_returns_insufficient_observations_for_first_stage_when_n_is_at_most_l() {
        // n=3、構造方程式は切片+endog1でk=2（n>k）、第一段階は切片+z1+z2でl=3（n=l）。
        let y = vec![1.0, 2.0, 4.0];
        let x_endog = vec![vec![1.0, 2.5, 2.0]];
        let instruments = vec![vec![1.0, 2.0, 4.0], vec![2.0, 1.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientObservationsForFirstStage { n: 3, l: 3 }
        );
    }

    /// `n < l`（`k < n`）でも同じ専用エラーで拒否する。
    #[test]
    fn fit_returns_insufficient_observations_for_first_stage_when_n_is_below_l() {
        // n=3、k=2（切片+endog1）、l=4（切片+z1..z3）。
        let y = vec![1.0, 2.0, 4.0];
        let x_endog = vec![vec![1.0, 2.5, 2.0]];
        let instruments = vec![
            vec![1.0, 2.0, 4.0],
            vec![2.0, 1.0, 5.0],
            vec![0.5, 3.0, 1.0],
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientObservationsForFirstStage { n: 3, l: 4 }
        );
    }

    /// 観測数の検証はクラスター数の検証より先に走る（クラスターが1つだけでも、
    /// `n <= l`なら観測数不足を返す）。
    #[test]
    fn fit_checks_observation_count_before_cluster_count() {
        let y = vec![1.0, 2.0, 4.0];
        let x_endog = vec![vec![1.0, 2.5, 2.0]];
        let instruments = vec![vec![1.0, 2.0, 4.0], vec![2.0, 1.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let one_cluster = CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&[
                "a".to_string(),
                "a".to_string(),
                "a".to_string(),
            ])),
        };

        let result = TwoSlsEstimator::fit(input, one_cluster, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientObservationsForFirstStage { n: 3, l: 3 }
        );
    }

    /// 弱操作変数診断（部分F統計量）が、`TwoSlsEstimator::fit`・`partial_wald_statistic`
    /// （SUT）とは独立に手計算したネストF検定のオラクルと数値一致することを確認する。
    /// `nontrivial_x_exog_columns()`（過剰識別、x_exog=[x1]・instruments=[z1,z2]）を使う。
    #[test]
    fn fit_computes_weak_instrument_f_statistic_matching_manual_nested_f_test() {
        let (x1, x_endog, z1, z2, _y, estimator) = nontrivial_x_exog_fitted_estimator();
        let n = x1.len();
        let y_endog = Mat::from_fn(n, 1, |i, _| x_endog[i]);

        // 制限なしモデル: x_endog ~ const + x1 + z1 + z2（k_u=4）。
        let x_u = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let beta_u = manual_ols_beta(&x_u, &y_endog);
        let resid_u = &y_endog - &x_u * &beta_u;
        let ssr_u: f64 = (0..n).map(|i| (*resid_u.get(i, 0)).powi(2)).sum();

        // 制限モデル: x_endog ~ const + x1（k_r=2、instruments=[z1,z2]を除く）。
        let x_r = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x1[i] });
        let beta_r = manual_ols_beta(&x_r, &y_endog);
        let resid_r = &y_endog - &x_r * &beta_r;
        let ssr_r: f64 = (0..n).map(|i| (*resid_r.get(i, 0)).powi(2)).sum();

        let q = 2.0; // z1, z2
        let df_u = (n - 4) as f64;
        let expected_f = ((ssr_r - ssr_u) / q) / (ssr_u / df_u);

        let got = estimator.weak_instrument_f_statistics();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "endog1");
        assert!(
            (got[0].1 - expected_f).abs() < 1e-8,
            "got={}, expected={}",
            got[0].1,
            expected_f
        );
    }

    /// 完了条件「単体テストで既知のケース（強い操作変数・弱い操作変数）を確認」に対応する。
    /// 同じ`x_exog`（定数項のみ）・同じ`z1`/`z2`の下で、内生変数の生成方法だけを変え、
    /// 操作変数が強く効く場合とほぼ無関係な場合とで部分F統計量が大きく異なることを確認する
    /// （弱操作変数の経験則である閾値10を跨いだ値になっていることも合わせて確認する。
    /// Stock-Yogo臨界値との正式な照合はv1スコープ外、`docs/spec/iv-spec.md`3.4節）。
    #[test]
    fn fit_weak_instrument_f_statistic_is_large_for_strong_instruments_and_small_for_weak_instruments()
     {
        let n = 30;
        let z1: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let z2: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 1.7).sin() * 5.0 + (i as f64))
            .collect();
        // 小さく符号が交互に振れるだけの摂動（分散を極端に大きくしない程度のノイズ）。
        let small_noise: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        // z1・z2とはほぼ無関係な、大きな分散を持つノイズ（弱操作変数シナリオ用）。
        let large_noise: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 2.3).sin() * 50.0 + ((i as f64) * 0.7).cos() * 40.0)
            .collect();

        let y: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.1).collect();

        let build_input = |x_endog: &[f64]| {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog.to_vec()),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let strong_x_endog: Vec<f64> = (0..n)
            .map(|i| 2.0 * z1[i] + 1.5 * z2[i] + small_noise[i])
            .collect();
        let strong =
            TwoSlsEstimator::fit(build_input(&strong_x_endog), CovType::Classical, 0.95).unwrap();
        let strong_f = strong.weak_instrument_f_statistics()[0].1;

        let weak_x_endog: Vec<f64> = (0..n)
            .map(|i| 0.001 * z1[i] + 0.001 * z2[i] + large_noise[i])
            .collect();
        let weak =
            TwoSlsEstimator::fit(build_input(&weak_x_endog), CovType::Classical, 0.95).unwrap();
        let weak_f = weak.weak_instrument_f_statistics()[0].1;

        assert!(strong_f > 100.0, "strong_f={strong_f}");
        assert!(weak_f < 5.0, "weak_f={weak_f}");
        assert!(strong_f > weak_f, "strong_f={strong_f}, weak_f={weak_f}");
    }

    /// `x_endog=[]`の退化ケース（第一段階ループが一度も回らない）では、
    /// `weak_instrument_f_statistics()`も空になる。
    #[test]
    fn weak_instrument_f_statistics_is_empty_when_x_endog_is_empty() {
        let y = vec![2.1, 3.8, 6.3, 7.9, 10.2];
        let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        assert!(estimator.weak_instrument_f_statistics().is_empty());
    }

    /// `x_exog=[]`かつ`include_intercept=false`（制限モデルに回帰変数が1つも無い退化ケース、
    /// `partial_wald_statistic`の`x_exog_columns.is_empty()`分岐）でも計算できることを、
    /// 手計算したネストF検定のオラクル（制限モデルのSSRを`y_endog`自体の二乗和として
    /// 直接計算）と数値照合して確認する。
    #[test]
    fn fit_computes_weak_instrument_f_statistic_when_x_exog_is_empty_and_no_intercept() {
        let n = 10;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        let z2: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 1.3).sin() * 3.0 + (i as f64))
            .collect();
        let x_endog: Vec<f64> = (0..n)
            .map(|i| 1.5 * z1[i] + 0.5 * z2[i] + if i % 2 == 0 { 0.2 } else { -0.2 })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 * x_endog[i] + (i as f64) * 0.05)
            .collect();

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            false, // include_intercept=false かつ x_exog=[]
            "y".to_string(),
        )
        .unwrap();
        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let y_endog = Mat::from_fn(n, 1, |i, _| x_endog[i]);
        let x_u = Mat::from_fn(n, 2, |i, j| if j == 0 { z1[i] } else { z2[i] });
        let beta_u = manual_ols_beta(&x_u, &y_endog);
        let resid_u = &y_endog - &x_u * &beta_u;
        let ssr_u: f64 = (0..n).map(|i| (*resid_u.get(i, 0)).powi(2)).sum();
        // 制限モデル（回帰変数なし、予測値は常に0）: SSR_r = Σ x_endog_i²。
        let ssr_r: f64 = x_endog.iter().map(|v| v.powi(2)).sum();

        let q = 2.0;
        let df_u = (n - 2) as f64;
        let expected_f = ((ssr_r - ssr_u) / q) / (ssr_u / df_u);

        let got = estimator.weak_instrument_f_statistics();
        assert_eq!(got.len(), 1);
        assert!(
            (got[0].1 - expected_f).abs() < 1e-8,
            "got={}, expected={}",
            got[0].1,
            expected_f
        );
    }

    /// property-basedテスト。`ols/fit.rs`の`mod proptests`と同型の設計（配置・許容誤差`RTOL=1e-6`・
    /// `prop_assume!`によるフルランク安全弁）だが、IV固有の不変条件（操作変数の張る空間のみへの
    /// 依存・丁度識別の閉形式解）を検証する。ケース生成は`common::proptest_support`。
    mod proptests {
        use super::*;
        use crate::iv::common::proptest_support::{assert_approx_eq, iv_case_strategy};
        use proptest::prelude::*;
        use std::collections::HashMap;

        fn by_name(est: &TwoSlsEstimator) -> HashMap<String, (f64, f64)> {
            est.param_names()
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    (
                        name.clone(),
                        (*est.params().get(i, 0), *est.std_errors().get(i, 0)),
                    )
                })
                .collect()
        }

        /// 部分ピボット付きガウス消去で`a * x = b`（`a`は`m x m`）を解く。`gmm_point_estimate`等の
        /// 実装コード（Cholesky）とは独立な経路で閉形式解のオラクルを作るための自前実装。
        fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
            let m = b.len();
            for col in 0..m {
                let pivot = (col..m)
                    .max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())?;
                if a[pivot][col].abs() < 1e-10 {
                    return None;
                }
                a.swap(col, pivot);
                b.swap(col, pivot);
                for row in (col + 1)..m {
                    let f = a[row][col] / a[col][col];
                    let pivot_row = a[col].clone();
                    for (target, src) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                        *target -= f * src;
                    }
                    b[row] -= f * b[col];
                }
            }
            let mut x = vec![0.0; m];
            for row in (0..m).rev() {
                let tail: f64 = ((row + 1)..m).map(|c| a[row][c] * x[c]).sum();
                x[row] = (b[row] - tail) / a[row][row];
            }
            Some(x)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// `x_exog`の列順序を入れ替えても、名前で対応付けた係数・SEは変わらない。
            #[test]
            fn coefficients_and_se_are_invariant_to_x_exog_column_order(
                case in iv_case_strategy(false),
                use_hc1 in any::<bool>(),
            ) {
                let cov = || if use_hc1 { CovType::Hc1 } else { CovType::Classical };
                let base = TwoSlsEstimator::fit(case.input().unwrap(), cov(), 0.95);
                let permuted = case
                    .build_input(&case.shuffled_order(), &case.z, &case.y)
                    .map(|input| TwoSlsEstimator::fit(input, cov(), 0.95));
                prop_assume!(base.is_ok());
                prop_assume!(permuted.as_ref().is_ok_and(|r| r.is_ok()));
                let (base, permuted) = (by_name(&base.unwrap()), by_name(&permuted.unwrap().unwrap()));
                for (name, (beta, se)) in &base {
                    let (p_beta, p_se) = permuted[name];
                    assert_approx_eq(p_beta, *beta, &format!("beta[{name}]"));
                    assert_approx_eq(p_se, *se, &format!("se[{name}]"));
                }
            }

            /// 操作変数を可逆な線形変換（単位上三角行列）で置き換えても、2SLSの点推定・SEは
            /// 変わらない（第一段階の射影は操作変数の張る空間にのみ依存するため）。
            #[test]
            fn estimates_are_invariant_to_invertible_instrument_transform(
                case in iv_case_strategy(false),
            ) {
                let base = TwoSlsEstimator::fit(case.input().unwrap(), CovType::Classical, 0.95);
                let mixed_z = case.mixed_instruments();
                let mixed = case
                    .build_input(&case.identity_order(), &mixed_z, &case.y)
                    .map(|input| TwoSlsEstimator::fit(input, CovType::Classical, 0.95));
                prop_assume!(base.is_ok());
                prop_assume!(mixed.as_ref().is_ok_and(|r| r.is_ok()));
                let (base, mixed) = (base.unwrap(), mixed.unwrap().unwrap());
                for j in 0..base.params().nrows() {
                    assert_approx_eq(*mixed.params().get(j, 0), *base.params().get(j, 0), &format!("beta[{j}]"));
                    assert_approx_eq(*mixed.std_errors().get(j, 0), *base.std_errors().get(j, 0), &format!("se[{j}]"));
                }
            }

            /// 丁度識別（`len(z) == len(x_endog)`）では、2SLSの点推定は閉形式の操作変数推定量
            /// `(Z'X)⁻¹Z'y`（`Z=[const, x_exog, z]`、`X=[const, x_exog, x_endog]`）と一致する。
            #[test]
            fn just_identified_matches_closed_form_iv_estimator(
                case in iv_case_strategy(true),
            ) {
                let est = TwoSlsEstimator::fit(case.input().unwrap(), CovType::Classical, 0.95);
                prop_assume!(est.is_ok());
                let est = est.unwrap();

                let n = case.nobs();
                let ones = vec![1.0; n];
                let regressors: Vec<&Vec<f64>> = std::iter::once(&ones)
                    .chain(case.x_exog.iter())
                    .chain(case.x_endog.iter())
                    .collect();
                let instruments: Vec<&Vec<f64>> = std::iter::once(&ones)
                    .chain(case.x_exog.iter())
                    .chain(case.z.iter())
                    .collect();
                let dot = |a: &Vec<f64>, b: &Vec<f64>| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
                let zx: Vec<Vec<f64>> = instruments
                    .iter()
                    .map(|zc| regressors.iter().map(|xc| dot(zc, xc)).collect())
                    .collect();
                let zy: Vec<f64> = instruments.iter().map(|zc| dot(zc, &case.y)).collect();
                let solved = solve_linear_system(zx, zy);
                prop_assume!(solved.is_some());
                for (j, expected) in solved.unwrap().iter().enumerate() {
                    assert_approx_eq(*est.params().get(j, 0), *expected, &format!("beta[{j}]"));
                }
            }

            /// `y`を定数倍すると係数も同じ定数倍になる（SEは`|c|`倍）。
            #[test]
            fn coefficients_scale_linearly_with_y(
                case in iv_case_strategy(false),
                c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
            ) {
                let scaled_y: Vec<f64> = case.y.iter().map(|v| c * v).collect();
                let base = TwoSlsEstimator::fit(case.input().unwrap(), CovType::Classical, 0.95);
                let scaled = case
                    .build_input(&case.identity_order(), &case.z, &scaled_y)
                    .map(|input| TwoSlsEstimator::fit(input, CovType::Classical, 0.95));
                prop_assume!(base.is_ok());
                prop_assume!(scaled.as_ref().is_ok_and(|r| r.is_ok()));
                let (base, scaled) = (base.unwrap(), scaled.unwrap().unwrap());
                for j in 0..base.params().nrows() {
                    assert_approx_eq(*scaled.params().get(j, 0), c * *base.params().get(j, 0), &format!("beta[{j}]"));
                    assert_approx_eq(*scaled.std_errors().get(j, 0), c.abs() * *base.std_errors().get(j, 0), &format!("se[{j}]"));
                }
            }
        }
    }
}
