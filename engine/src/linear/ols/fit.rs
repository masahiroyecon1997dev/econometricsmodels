//! `OlsEstimator::fit`の本体。検証（回帰子の有無・信頼水準・観測数・クラスター数・被説明変数）→
//! 最小二乗 → cov_type別の共分散 → t検定・信頼区間 → 適合度 → 全体F検定の順に組み立てる。

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
mod tests {
    use super::*;
    use crate::linear::common::row_time_order;
    use crate::shared::error::MatrixKind;

    /// y = 1 + 2*x、ノイズなしの厳密解を持つデータ。
    fn exact_fit_input() -> OlsInput {
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let x_columns = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
        OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn fit_returns_perfect_fit_for_exact_fit_data() {
        // 残差が全て0に近いと`sigma^2`が0で標準誤差・F検定が定義できないため、共分散を計算する前に止める。
        let result = OlsEstimator::fit(exact_fit_input(), CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::PerfectFit)
        );
    }

    #[test]
    fn fit_auxiliary_recovers_known_coefficients_for_exact_fit_data() {
        // 補助回帰は完全フィットの判定をしないので、係数は厳密解のまま返る。
        let estimator =
            OlsEstimator::fit_auxiliary(exact_fit_input(), CovType::Classical, 0.95).unwrap();
        let params = estimator.params();

        assert!(
            (*params.get(0, 0) - 1.0).abs() < 1e-9,
            "const: {}",
            *params.get(0, 0)
        );
        assert!(
            (*params.get(1, 0) - 2.0).abs() < 1e-9,
            "x1: {}",
            *params.get(1, 0)
        );
    }

    #[test]
    fn fit_recovers_known_coefficients_for_nearly_exact_data() {
        // y = 1 + 2*x に小さな交互のノイズ。係数は厳密解の近くに返る。
        let y = vec![1.1, 2.9, 5.1, 6.9, 9.1];
        let x_columns = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        let params = estimator.params();

        assert!((*params.get(0, 0) - 1.0).abs() < 0.2);
        assert!((*params.get(1, 0) - 2.0).abs() < 0.1);
    }

    #[test]
    fn fit_returns_outcome_constant_when_the_outcome_is_the_same_in_every_row() {
        let input = OlsInput::from_columns(
            &[3.0; 5],
            &[vec![0.0, 1.0, 2.0, 3.0, 5.0]],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::OutcomeConstant)
        );
    }

    #[test]
    fn fit_treats_an_outcome_that_differs_by_one_ulp_as_constant() {
        // 計算で作った定数（`0.1 + 0.2`と`0.3`）は1 ulp違うが、定数として扱う。
        let y = [0.1 + 0.2, 0.3, 0.3, 0.1 + 0.2, 0.3];
        let input = OlsInput::from_columns(
            &y,
            &[vec![0.0, 1.0, 2.0, 3.0, 5.0]],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::OutcomeConstant)
        );
    }

    #[test]
    fn fit_does_not_treat_a_small_but_real_variation_as_constant() {
        let y = [3.0, 3.000_001, 3.0, 3.000_002, 3.000_001];
        let input = OlsInput::from_columns(
            &y,
            &[vec![0.0, 1.0, 2.0, 3.0, 5.0]],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        assert!(OlsEstimator::fit(input, CovType::Classical, 0.95).is_ok());
    }

    #[test]
    fn fit_accepts_a_constant_outcome_without_an_intercept() {
        // 切片なしでは定数の非中心化の総平方和が正なので、説明するものがある。
        let input = OlsInput::from_columns(
            &[3.0; 5],
            &[vec![1.0, 1.1, 0.9, 1.0, 1.05]],
            vec!["x1".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        assert!(OlsEstimator::fit(input, CovType::Classical, 0.95).is_ok());
    }

    #[test]
    fn perfect_fit_is_judged_relative_to_the_total_sum_of_squares() {
        // 同じ形の関係でも、ノイズが総平方和に対して`k * epsilon`を超えれば完全フィットではない
        // （ここでは相対的な大きさ約3e-4）。
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| 1.0 + 2.0 * v + if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let input = OlsInput::from_columns(&y, &[x], vec!["x1".to_string()], true, "y".to_string())
            .unwrap();

        assert!(OlsEstimator::fit(input, CovType::Classical, 0.95).is_ok());
    }

    #[test]
    fn fit_returns_no_regressors_error_when_k_is_zero() {
        // include_intercept=falseかつx_columns=[]の病的な入力。
        // Logit/Probitと同じ`CommonError::NoRegressors`で早期リジェクトされる。
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = OlsInput::from_columns(&y, &[], vec![], false, "y".to_string()).unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::NoRegressors { n: 5 })
        );
    }

    #[test]
    fn fit_returns_insufficient_observations_when_n_le_k() {
        let y = vec![1.0, 2.0];
        let x_columns = vec![vec![1.0, 2.0]];
        // include_intercept=trueでk=2、n=2 (n<=k)
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
    }

    #[test]
    fn fit_returns_singular_matrix_for_perfectly_collinear_columns() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0]; // x2 = 2 * x1 (完全な多重共線性)
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::SingularMatrix {
                columns: vec!["x1".to_string(), "x2".to_string()]
            }
        );
    }

    #[test]
    fn fit_returns_computation_failed_for_extreme_scale_difference_in_f_test() {
        // x1は1e6オーダー、x2は1e-3オーダーとスケールが極端に異なる（x3は通常
        // スケール）。x1・x2・x3は互いに線形従属ではないため設計行列自体は
        // フルランク（SingularMatrixにはならない）だが、傾き係数の同時共分散
        // 部分行列（wald_f_testが使う3x3部分行列）の条件数がスケール比の2乗
        // （≈1e18）相当となり倍精度の限界を超える
        // （ensure_well_conditioned_symmetric_matrixで検出）。
        let n = 10;
        let x1: Vec<f64> = (1..=n).map(|i| 1e6 * (i as f64)).collect();
        let x2: Vec<f64> = (1..=n).map(|i| 1e-3 * (i as f64).powi(2)).collect();
        let x3: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = if i % 2 == 0 { 1e3 } else { -1e3 };
                1.0 + 2.0 * x1[i] + 3.0 * x2[i] + 0.5 * x3[i] + noise
            })
            .collect();

        let input = OlsInput::from_columns(
            &y,
            &[x1, x2, x3],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::NearSingularMatrix {
                matrix: MatrixKind::CoefficientCovariance
            })
        ));
    }

    #[test]
    fn fit_pins_faer_global_parallelism_to_seq() {
        // `fit()` 冒頭の `crate::shared::parallelism::ensure_serial()` が faer の
        // グローバル並列度を `Par::Seq` へ引き戻すことの回帰ガード（linear 系統代表）。
        // 別テストが既に `Seq` にしている可能性があるため、まず `Rayon` に戻してから
        // `fit()` を通す。ここで扱う設計行列は極小なので、この一時的な `Rayon` 設定が
        // その病理（大標本 tall-skinny での不安定化）を招くことはない。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let input =
            OlsInput::from_columns(&y, &[x1], vec!["x1".to_string()], true, "y".to_string())
                .unwrap();

        let _ = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    #[test]
    fn fit_returns_invalid_confidence_level_when_out_of_range() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 1.5);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    /// x = [1,2,3,4,5], y = [2,4,5,4,5] の教科書的データセット。
    /// 期待値はscipy.stats（`scipy.stats.t`、`ppf`/`cdf`）で独立に計算・検算済み
    /// （手計算: b0=2.2, b1=0.6, SSR=2.4, df=3, sigma2=0.8）。
    #[test]
    fn fit_computes_classical_std_errors_test_stats_p_values_and_conf_int() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let params = estimator.params();
        assert!((*params.get(0, 0) - 2.2).abs() < 1e-9);
        assert!((*params.get(1, 0) - 0.6).abs() < 1e-9);

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.938_083_151_964_686).abs() < 1e-9);
        assert!((*se.get(1, 0) - 0.282_842_712_474_619).abs() < 1e-9);

        let t = estimator.test_stats();
        assert!((*t.get(0, 0) - 2.345_207_879_911_715).abs() < 1e-9);
        assert!((*t.get(1, 0) - 2.121_320_343_559_642_4).abs() < 1e-9);

        let p = estimator.p_values();
        assert!((*p.get(0, 0) - 0.100_743_456_085_420_12).abs() < 1e-6);
        assert!((*p.get(1, 0) - 0.124_027_062_657_554_59).abs() < 1e-6);

        let lower = estimator.conf_lower();
        let upper = estimator.conf_upper();
        assert!((*lower.get(0, 0) - (-0.785_399_261_018_909_6)).abs() < 1e-6);
        assert!((*upper.get(0, 0) - 5.185_399_261_018_91).abs() < 1e-6);
        assert!((*lower.get(1, 0) - (-0.300_131_745_291_273_4)).abs() < 1e-6);
        assert!((*upper.get(1, 0) - 1.500_131_745_291_273_2).abs() < 1e-6);
    }

    /// `confidence_level`の境界値（0.0・1.0ちょうど）が範囲外として拒否されることを確認する
    /// （`!(level > 0.0 && level < 1.0)`という判定式の境界そのものの検証）。
    #[test]
    fn fit_returns_invalid_confidence_level_at_exact_boundaries() {
        for level in [0.0, 1.0, -0.1] {
            let y = vec![1.0, 2.0, 3.0];
            let x_columns = vec![vec![1.0, 2.0, 3.0]];
            let input = OlsInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result = OlsEstimator::fit(input, CovType::Classical, level);

            assert_eq!(
                result.unwrap_err(),
                LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: level
                }),
                "level={level}"
            );
        }
    }

    /// x=[1..5], y=[2,4,5,4,5]（切片あり、classical）の適合度統計量。
    /// 期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(use_t=True)`。`fvalue`/`f_pvalue`は古典的F検定と
    /// ロバストWald検定の式が代数的に一致することも別途手計算で確認済み）。
    #[test]
    fn fit_computes_r_squared_and_information_criteria_with_intercept() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!((estimator.r_squared() - 0.599_999_999_999_999_9).abs() < 1e-9);
        assert!((estimator.adj_r_squared() - 0.466_666_666_666_666_56).abs() < 1e-9);
        assert!((estimator.log_likelihood() - (-5.259_769_728_322_863)).abs() < 1e-9);
        assert!((estimator.aic() - 14.519_539_456_645_726).abs() < 1e-9);
        assert!((estimator.bic() - 13.738_415_281_513_927).abs() < 1e-9);
        assert!((estimator.f_statistic().unwrap() - 4.499_999_999_999_999).abs() < 1e-6);
        assert!((estimator.f_p_value().unwrap() - 0.124_027_062_657_554_59).abs() < 1e-6);
    }

    /// 同じ(x, y)を切片なしで推定した場合。R²・調整済みR²がuncentered TSS
    /// （`Σy_i²`）を基準に計算されることを確認する（statsmodelsの`k_constant=0`の
    /// 挙動と一致。`docs/spec/ols-spec.md`「適合度統計量」参照）。
    #[test]
    fn fit_computes_r_squared_without_intercept_uses_uncentered_tss() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!((estimator.r_squared() - 0.920_930_232_558_139_5).abs() < 1e-9);
        assert!((estimator.adj_r_squared() - 0.901_162_790_697_674_5).abs() < 1e-9);
        assert!((estimator.log_likelihood() - (-7.863_404_415_393_264)).abs() < 1e-9);
        assert!((estimator.aic() - 17.726_808_830_786_528).abs() < 1e-9);
        assert!((estimator.bic() - 17.336_246_743_220_627).abs() < 1e-9);
        assert!((estimator.f_statistic().unwrap() - 46.588_235_294_117_66).abs() < 1e-6);
        assert!((estimator.f_p_value().unwrap() - 0.002_409_205_984_197_115_5).abs() < 1e-6);
    }

    /// HC1・HAC(maxlags=1)でのF統計量がロバストWald検定になることを確認する
    /// （R²・AIC/BIC・対数尤度は`cov_type`に依存しないため、ここではF統計量のみ検証）。
    /// 期待値はstatsmodelsで独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type=..., use_t=True)`）。
    #[test]
    fn fit_computes_robust_wald_f_test_for_hc_and_hac() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

        let input_hc1 = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator_hc1 = OlsEstimator::fit(input_hc1, CovType::Hc1, 0.95).unwrap();
        assert!((estimator_hc1.f_statistic().unwrap() - 6.279_069_767_441_904).abs() < 1e-6);
        assert!((estimator_hc1.f_p_value().unwrap() - 0.087_259_022_565_828_96).abs() < 1e-6);

        let input_hac = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let cov_type_hac = CovType::Hac {
            lags: Some(1),
            time_order: row_time_order(5),
        };
        let estimator_hac = OlsEstimator::fit(input_hac, cov_type_hac, 0.95).unwrap();
        assert!((estimator_hac.f_statistic().unwrap() - 13.235_294_117_647_193).abs() < 1e-6);
        assert!((estimator_hac.f_p_value().unwrap() - 0.035_791_053_269_350_51).abs() < 1e-6);
    }

    /// 説明変数が定数項のみ（傾き係数が無い）モデル。F検定は検定対象が存在しないため、
    /// 定義なし（`None`）を返す（`OlsEstimator::fit`の`df_model == 0`分岐）。
    #[test]
    fn fit_returns_no_f_test_when_model_has_no_slope_regressors() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = OlsInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!(estimator.f_statistic().is_none());
        assert!(estimator.f_p_value().is_none());
    }

    #[test]
    fn fit_exposes_input_cov_type_and_residuals_via_getters() {
        // y = 1 + 2*x に小さな交互のノイズ（残差が小さいことを確認しやすい）
        let y = vec![1.01, 2.99, 5.01, 6.99, 9.01];
        let x_columns = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert_eq!(estimator.input().nobs(), 5);
        assert_eq!(estimator.cov_type(), &CovType::Classical);
        let residuals = estimator.residuals();
        for i in 0..5 {
            assert!((*residuals.get(i, 0)).abs() < 0.05);
        }
    }

    #[test]
    fn fitted_values_equals_y_minus_residuals() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let fitted = estimator.fitted_values();
        for (i, &y_i) in y.iter().enumerate() {
            let expected = y_i - *estimator.residuals().get(i, 0);
            assert!((*fitted.get(i, 0) - expected).abs() < 1e-9);
        }
    }

    /// property-basedテスト。固定シナリオでの値の一致確認（上記）とは別に、
    /// OLSが満たすべき不変条件をランダムなデータで検証する（`testing-policy.md`
    /// 「将来的に検討する技術」参照）。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        const MAX_K: usize = 20;

        /// `(n, k, y, x_cols, keys)`を生成する共通ストラテジ。
        ///
        /// `n = k+10..=60`という十分なマージンを取り、値は独立な連続一様分布
        /// （`-100.0..100.0`）からサンプリングする。この条件下では生成される設計行列は
        /// 実務上ほぼ確実にフルランクになるため、各プロパティ側では追加の制約は課さず、
        /// `prop_assume!(result.is_ok())`で非フルランクになるレアケース（丸め誤差起因の
        /// 境界事例等）のみを除外する（「ランダムに生成する設計行列は
        /// SingularMatrixにならない範囲に制約する」という方針に対応）。
        /// `MAX_K=20`（旧4）はbenchmarkの高次元シナリオ`many_regressors`と揃えた値
        /// （列数依存バグ・高kでの数値的挙動の検証）。
        /// `k`が最大でも`n-k>=10`のマージンは保たれる。
        ///
        /// `keys`は列順序入れ替えテスト専用の補助データ（他のプロパティでは未使用）。
        fn ols_case_strategy()
        -> impl Strategy<Value = (usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<u64>)> {
            (1..=MAX_K).prop_flat_map(|k| {
                (k + 10..=60usize).prop_flat_map(move |n| {
                    (
                        Just(n),
                        Just(k),
                        collection::vec(-100.0f64..100.0, n),
                        collection::vec(collection::vec(-100.0f64..100.0, n), k),
                        collection::vec(any::<u64>(), k),
                    )
                })
            })
        }

        fn x_names(k: usize) -> Vec<String> {
            (1..=k).map(|i| format!("x{i}")).collect()
        }

        /// 相対誤差ベースの近似比較（絶対誤差フロア込み）。乱数生成で値のスケールが
        /// 揃わないため、固定フィクスチャ比較（`testing-policy.md`のRTOL=1e-8）より
        /// 緩めた閾値にしている（col_piv_qrの数値誤差の蓄積を考慮）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-6 * expected.abs().max(1.0);
            let diff = (actual - expected).abs();
            assert!(
                diff <= tol,
                "{msg}: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
            );
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// 定数項ありOLSでの残差和は常に0（正規方程式`X'e=0`のうち切片列に対応する
            /// 行から従う）。
            #[test]
            fn residuals_sum_to_zero_when_intercept_included(
                (n, k, y, x_cols, _keys) in ols_case_strategy()
            ) {
                let input = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result = OlsEstimator::fit(input, CovType::Classical, 0.95);
                prop_assume!(result.is_ok());
                let est = result.unwrap();

                let sum: f64 = (0..n).map(|i| *est.residuals().get(i, 0)).sum();
                let scale = y.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
                prop_assert!(
                    sum.abs() <= 1e-6 * scale * (n as f64),
                    "residual sum should be ~0, got {sum} (scale={scale}, n={n})"
                );
            }

            /// yをc倍すると、切片を含む全ての係数がc倍にスケールする
            /// （OLSはyに関して線形なため、切片も例外ではない）。
            #[test]
            fn coefficients_scale_linearly_with_y(
                (_n, k, y, x_cols, _keys) in ols_case_strategy(),
                c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
            ) {
                let input1 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result1 = OlsEstimator::fit(input1, CovType::Classical, 0.95);
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let y_scaled: Vec<f64> = y.iter().map(|v| v * c).collect();
                let input2 =
                    OlsInput::from_columns(&y_scaled, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result2 = OlsEstimator::fit(input2, CovType::Classical, 0.95);
                prop_assume!(result2.is_ok());
                let est2 = result2.unwrap();

                for i in 0..=k {
                    let expected = c * *est1.params().get(i, 0);
                    let actual = *est2.params().get(i, 0);
                    assert_approx_eq(actual, expected, &format!("param[{i}] scaled by c={c}"));
                }
            }

            /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
            #[test]
            fn coefficients_and_se_are_invariant_to_column_order(
                (_n, k, y, x_cols, keys) in ols_case_strategy()
                    .prop_filter("need >=2 columns to permute", |(_, k, _, _, _)| *k >= 2)
            ) {
                let names = x_names(k);
                let input1 =
                    OlsInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
                let result1 = OlsEstimator::fit(input1, CovType::Classical, 0.95);
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let mut order: Vec<usize> = (0..k).collect();
                order.sort_by_key(|&i| keys[i]);
                let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
                let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

                let input2 =
                    OlsInput::from_columns(&y, &permuted_x, permuted_names, true, "y".to_string()).unwrap();
                let result2 = OlsEstimator::fit(input2, CovType::Classical, 0.95);
                prop_assume!(result2.is_ok());
                let est2 = result2.unwrap();

                let names1 = est1.input().param_names().to_vec();
                let names2 = est2.input().param_names().to_vec();
                for (idx1, name) in names1.iter().enumerate() {
                    let idx2 = names2
                        .iter()
                        .position(|n| n == name)
                        .expect("name should exist in permuted result");
                    let p1 = *est1.params().get(idx1, 0);
                    let p2 = *est2.params().get(idx2, 0);
                    assert_approx_eq(p2, p1, &format!("param[{name}] under column permutation"));
                    let se1 = *est1.std_errors().get(idx1, 0);
                    let se2 = *est2.std_errors().get(idx2, 0);
                    assert_approx_eq(se2, se1, &format!("std_error[{name}] under column permutation"));
                }
            }

            /// HC0の標準誤差は常にHC1以下（`HC1 = HC0 * n/(n-k)`で`n/(n-k) >= 1`のため）。
            #[test]
            fn hc0_std_errors_are_at_most_hc1_std_errors(
                (_n, k, y, x_cols, _keys) in ols_case_strategy()
            ) {
                let input1 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result1 = OlsEstimator::fit(input1, CovType::Hc0, 0.95);
                prop_assume!(result1.is_ok());
                let est_hc0 = result1.unwrap();

                let input2 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result2 = OlsEstimator::fit(input2, CovType::Hc1, 0.95);
                prop_assume!(result2.is_ok());
                let est_hc1 = result2.unwrap();

                for i in 0..=k {
                    let se_hc0 = *est_hc0.std_errors().get(i, 0);
                    let se_hc1 = *est_hc1.std_errors().get(i, 0);
                    // 浮動小数点の丸め誤差の余地として小さな絶対許容誤差を加える。
                    prop_assert!(
                        se_hc0 <= se_hc1 + 1e-9,
                        "HC0 se[{i}]={se_hc0} should be <= HC1 se[{i}]={se_hc1}"
                    );
                }
            }
        }
    }
}
