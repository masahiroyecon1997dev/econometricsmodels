//! `GmmEstimator::fit`の本体（検証・反復・Hansen J・共分散・検定をつなぐオーケストレーション）。

use faer::Mat;
use statrs::distribution::Normal;

use super::GmmEstimator;
use super::hansen_j::hansen_j_test;
use super::iterate::{GmmIteration, GmmType, gmm_iterate, gmm_point_estimate};
use super::sandwich::{GmmCovariance, gmm_wald_chi2_test, sandwich_covariance};
use super::weight::{WeightType, validate_weight_type};
use crate::iv::common::{IvError, IvInput, hcat, mat_column_to_vec, validate_iv_fit_inputs};
use crate::linear::ols::CovType;
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{is_perfect_fit, total_sum_of_squares};
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::validation::{GMM_MAX_ITER_MIN, MAX_ITER_LIMIT};

/// [`GmmEstimator::fit`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_gmm(
    input: IvInput,
    gmm_type: GmmType,
    raise_on_non_convergence: bool,
    cov_type: CovType,
    confidence_level: f64,
) -> Result<GmmEstimator, IvError> {
    // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
    crate::shared::parallelism::ensure_serial();

    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
    }
    if let GmmType::Iterated { max_iter, tol, .. } = &gmm_type {
        if *max_iter < GMM_MAX_ITER_MIN as usize || *max_iter as u64 > MAX_ITER_LIMIT as u64 {
            return Err(IvError::InvalidGmmMaxIter {
                max_iter: *max_iter as i64,
            });
        }
        if !(tol.is_finite() && *tol > 0.0) {
            return Err(IvError::InvalidGmmTol { gmm_tol: *tol });
        }
    }
    let n = input.nobs();
    // 観測数（`n <= k`はサンドイッチSE計算のNaN/Infinity、`n <= l`は`Z'Z`の特異または
    // `W₀ = (Z'Z)⁻¹`のGMMがOLSへ縮退するのを防ぐ）・`cov_type=Cluster`のクラスター数
    // （構造方程式の`G <= q`）・被説明変数の定数を、点推定・SE計算より前に弾く
    // （`TwoSlsEstimator::fit`・`engine_pybind`と同じ`validate_iv_fit_inputs`）。
    // `weight_type=Cluster`の重み行列`S`（l×l）の`G`と`l`の関係は別軸のため
    // `validate_weight_type`側で検証する。
    validate_iv_fit_inputs(&input, &cov_type)?;
    let k = input.k_exog() + input.k_endog();

    // 1段階GMMは`weight_type`を持たない（点推定・Hansen Jとも`(Z'Z)⁻¹`のみで計算する）
    // ため検証もしない。2段階・反復では点推定に使う重みの妥当性を検証する。
    let weight_type: Option<&WeightType> = match &gmm_type {
        GmmType::OneStep => None,
        GmmType::TwoStep { weight } | GmmType::Iterated { weight, .. } => Some(weight),
    };
    let l_instruments = input.k_exog() + input.instruments().ncols();
    // `weight_type=Cluster`のグループ（`validate_weight_type`が検証して返す）。`cov_type=Cluster`
    // のグループとは別の入力で、同じ列を指していても別々に渡される。
    let weight_cluster_codes = match weight_type {
        Some(weight_type) => validate_weight_type(weight_type, n, k, l_instruments)?,
        None => None,
    };

    // Z = x_exog ++ instruments（全操作変数、docs/spec/iv-spec.md 1.1節）。
    let z = hcat(input.x_exog(), input.instruments());
    let l = z.ncols();

    // X = x_exog ++ x_endog（実際の内生変数。2SLSの第二段階のようなX̂への
    // 置き換えはしない、`gmm.rs`のモジュールdoc参照）。
    let x = hcat(input.x_exog(), input.x_endog());

    let mut param_names = input.x_exog_names().to_vec();
    param_names.extend(input.x_endog_names().iter().cloned());

    let y = input.y();

    // 初期推定: W₀ = (Z'Z)⁻¹（weight_typeによらず共通、2SLSと同じ重み）。
    let ztz = z.transpose() * &z;
    let beta0 = gmm_point_estimate(&z, &x, y, &ztz)?;
    let residuals0 = y - &x * &beta0;

    // 初期推定の残差が丸め誤差の範囲で0なら完全フィット。重み行列`S`が0になって後段の逆行列が
    // 特異になる前に、原因が分かるエラーで止める（`TwoSlsEstimator::fit`と同じ判定）。
    let sst = total_sum_of_squares(&mat_column_to_vec(y, 0), None, input.has_intercept());
    let ssr0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum();
    if is_perfect_fit(ssr0, sst, k) {
        return Err(CommonError::PerfectFit.into());
    }

    // weight_type=ClassicalのHansen J検定専用の重み行列`σ̂²₀・Z'Z`（`gmm.rs`の
    // モジュールdoc「Hansen J過剰識別検定」参照。`σ̂²`によるスケーリングは点推定には
    // 不要だがHansen Jには必須）。`σ̂²₀`はstep-0残差から計算する
    // （`OneStep`ではstep-0残差＝最終残差のため、この一箇所の計算で両方の
    // ケースをカバーできる）。
    let sigma2_0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
    let classical_s = Mat::from_fn(l, l, |i, j| sigma2_0 * (*ztz.get(i, j)));

    // 方式（`gmm_type`）に応じた回数の再推定（詳細は`gmm_iterate`のdocコメント）。
    let GmmIteration {
        beta,
        residuals,
        s_used,
        n_iter,
        converged,
        hac_lags_used,
    } = gmm_iterate(
        &gmm_type,
        weight_type,
        weight_cluster_codes,
        &z,
        &x,
        y,
        &ztz,
        beta0,
        residuals0,
        classical_s,
    )?;

    if !converged && raise_on_non_convergence {
        return Err(IvError::GmmNonConvergence {
            n_iter: n_iter as usize,
        });
    }

    let hansen_j = hansen_j_test(&input, &z, &residuals, &s_used)?;

    // ── ここから独立実装のサンドイッチ型SE計算（`gmm.rs`のモジュールdoc
    // 「標準誤差・検定統計量（cov_type対応）」参照）──
    let k_constant = usize::from(input.has_intercept());
    let df_resid = n - k;
    let df_model = k - k_constant;
    let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();

    let GmmCovariance {
        cov_params,
        hac_lags_used: cov_hac_lags,
    } = sandwich_covariance(&cov_type, &z, &x, &residuals, &ztz, &s_used, df_resid)?;
    // `cov_type=Hac`のラグ数は`weight_type=Hac`のラグ数より優先する。
    let hac_lags_used = cov_hac_lags.or(hac_lags_used);

    let normal_dist = Normal::new(0.0, 1.0).map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(&beta, &cov_params, &normal_dist, confidence_level);

    let r_squared = 1.0 - ssr / sst;
    let adj_r_squared = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

    let (wald_statistic, wald_p_value) = if df_model == 0 {
        // 説明変数が定数項のみ（傾き係数が無い）モデル。検定対象が存在しないため
        // 2SLSと同様NaNを返す（0除算を避ける）。
        (f64::NAN, f64::NAN)
    } else {
        gmm_wald_chi2_test(&beta, &cov_params, k_constant, df_model)?
    };

    Ok(GmmEstimator {
        params: beta,
        param_names,
        dep_var_name: input.dep_var_name().to_string(),
        residuals,
        gmm_type,
        n_iter,
        converged,
        nobs: n,
        k,
        cov_type,
        hac_lags_used,
        std_errors,
        cov_params,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        df_resid,
        df_model,
        r_squared,
        adj_r_squared,
        wald_statistic,
        wald_p_value,
        hansen_j_statistic: hansen_j.as_ref().map(|t| t.statistic),
        hansen_j_p_value: hansen_j.as_ref().map(|t| t.p_value),
        hansen_j_df: hansen_j.as_ref().map(|t| t.df),
    })
}

#[cfg(test)]
mod tests {
    use crate::iv::common::{IvError, IvEstimate, IvInput};
    use crate::iv::gmm::GmmEstimator;
    use crate::iv::gmm::GmmType;
    use crate::iv::gmm::WeightType;
    use crate::iv::gmm::iterate::gmm_point_estimate;
    use crate::iv::gmm::test_support::*;
    use crate::linear::ols::CovType;
    use crate::linear::ols::CovType as OlsCovType;
    use crate::shared::error::{CommonError, MatrixKind};
    use faer::Mat;

    /// 2SLSと同じデータ（`x_exog`に実変数を含む、過剰識別）で、`WeightType::Classical`の
    /// GMM点推定が`TwoSlsEstimator`の点推定と厳密に一致することを確認する（`gmm.rs`の
    /// モジュールdoc「2SLSとの共通化の判断」の根拠）。
    #[test]
    fn fit_matches_two_sls_point_estimate_when_weight_type_is_classical() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];

        let build_input = || {
            IvInput::from_columns(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let gmm = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let two_sls =
            crate::iv::two_sls::TwoSlsEstimator::fit(build_input(), OlsCovType::Classical, 0.95)
                .unwrap();

        assert_eq!(gmm.param_names(), two_sls.param_names());
        for j in 0..gmm.params().nrows() {
            assert!(
                (*gmm.params().get(j, 0) - *two_sls.params().get(j, 0)).abs() < 1e-8,
                "param {j}: gmm={}, two_sls={}",
                *gmm.params().get(j, 0),
                *two_sls.params().get(j, 0)
            );
        }
    }

    #[test]
    fn fit_matches_two_sls_when_just_identified() {
        let gmm = GmmEstimator::fit(
            just_identified_input(),
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let two_sls = crate::iv::two_sls::TwoSlsEstimator::fit(
            just_identified_input(),
            CovType::Classical,
            0.95,
        )
        .unwrap();

        assert_eq!(gmm.param_names(), ["const", "x_endog"]);
        for j in 0..2 {
            assert!((*gmm.params().get(j, 0) - *two_sls.params().get(j, 0)).abs() < 1e-8);
        }
    }

    #[test]
    fn fit_returns_perfect_fit_when_the_structural_residuals_are_zero() {
        // 構造式が厳密に成り立つと、初期推定の残差が全て0で重み行列が0になる。後段の逆行列が
        // 特異になる前に、原因の分かる完全フィットとして止める。
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = vec![1.2, 1.9, 3.3, 3.8, 5.1];
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

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Classical,
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::PerfectFit)
        );
    }

    #[test]
    fn fit_returns_outcome_constant_for_a_constant_outcome() {
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = vec![1.2, 1.9, 3.3, 3.8, 5.1];
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

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::OutcomeConstant)
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

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            }
        );
    }

    /// 操作変数が完全な線形従属（`z2 = 2*z1`）だと`Z'Z`が特異になり
    /// `CommonError::NearSingularMatrix { matrix: GmmWeight }`（`MATRIX_NEAR_SINGULAR`）
    /// になることを確認する（`.claude/rules/testing-policy.md`「テストの3系統」の
    /// `ComputationError`パス）。
    #[test]
    fn fit_returns_computation_error_when_instruments_are_perfectly_collinear() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0];
        let z1 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let z2: Vec<f64> = z1.iter().map(|v| 2.0 * v).collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::NearSingularMatrix {
                matrix: MatrixKind::GmmWeight
            })
        );
    }

    /// `dep_var_name()`/`weight_type()`/`gmm_tol()`/`nobs()`/`k()`が、
    /// `fit()`に渡した値・入力データと整合することを確認する（`two_sls/fit.rs`の
    /// `fit_succeeds_when_over_identified`と同じ「基本メタデータgetter群」の検証方針）。
    #[test]
    fn fit_exposes_basic_metadata_getters() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            GmmType::Iterated {
                weight: WeightType::Robust,
                max_iter: 5,
                tol: 1e-6,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        assert_eq!(estimator.dep_var_name(), "y");
        assert_eq!(estimator.weight_type(), Some(&WeightType::Robust));
        assert_eq!(
            estimator.gmm_type(),
            &GmmType::Iterated {
                weight: WeightType::Robust,
                max_iter: 5,
                tol: 1e-6,
            }
        );
        assert_eq!(estimator.nobs(), n);
        assert_eq!(estimator.k(), 2);
    }

    /// `residuals()`が最終推定値に基づく`y - Xβ̂`であることを確認する。
    #[test]
    fn residuals_are_consistent_with_final_params() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        for (i, &y_i) in y.iter().enumerate() {
            let expected = y_i
                - (*estimator.params().get(0, 0)) * (*x.get(i, 0))
                - (*estimator.params().get(1, 0)) * (*x.get(i, 1));
            assert!(
                (*estimator.residuals().get(i, 0) - expected).abs() < 1e-8,
                "row {i}"
            );
        }
    }

    /// `confidence_level`が`(0, 1)`の範囲外なら`InvalidConfidenceLevel`
    /// （`OlsEstimator::fit`/`TwoSlsEstimator::fit`と同じ規約）。
    #[test]
    fn fit_returns_invalid_confidence_level_error_out_of_range() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        for invalid in [0.0_f64, 1.0, -0.1, 1.5] {
            let input = IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result = GmmEstimator::fit(
                input,
                GmmType::TwoStep {
                    weight: WeightType::Robust,
                },
                true,
                CovType::Classical,
                invalid,
            );
            assert_eq!(
                result.unwrap_err(),
                IvError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: invalid
                }),
                "confidence_level={invalid}"
            );
        }
    }

    /// 観測数`n`が`k`（構造方程式の係数の数）以下なら`InsufficientObservations`
    /// （`n<=k`のままだと`df_resid=n-k`による除算
    /// 等がNaN/Infinityを静かに生成しうる、`gmm_hc_omega`のdocコメント参照）。
    #[test]
    fn fit_returns_insufficient_observations_error_when_n_is_at_most_k() {
        // n=2, k=2（const + endog1）: 丁度識別だが観測数が係数の数と等しい境界。
        let y = vec![1.0, 2.0];
        let x_endog = vec![2.0, 4.0];
        let z = vec![1.0, 3.0];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            std::slice::from_ref(&z),
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
    }

    /// 構造方程式は推定できる（`n > k`）が全操作変数の数`l`以下の観測数なら、
    /// `InsufficientObservationsForFirstStage`で拒否する（`n < l`では`Z'Z`が特異、
    /// `n = l`では`W₀ = (Z'Z)⁻¹`のGMMがOLSへ縮退する）。`2sls`と同じ検証を共有する。
    #[test]
    fn fit_returns_insufficient_observations_for_first_stage_when_n_is_at_most_l() {
        // n=3、k=2（切片+endog1）、l=3（切片+z1+z2）。
        let y = vec![1.0, 2.0, 4.0];
        let x_endog = vec![1.0, 2.5, 2.0];
        let z = vec![vec![1.0, 2.0, 4.0], vec![2.0, 1.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &z,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientObservationsForFirstStage { n: 3, l: 3 }
        );
    }

    /// `n < l`（`k < n`）でも同じ専用エラーで拒否する（検証が無ければ`Z'Z`が特異になり
    /// `NearSingularMatrix`の`ComputationError`になる経路）。
    #[test]
    fn fit_returns_insufficient_observations_for_first_stage_when_n_is_below_l() {
        // n=3、k=2（切片+endog1）、l=4（切片+z1..z3）。
        let y = vec![1.0, 2.0, 4.0];
        let x_endog = vec![1.0, 2.5, 2.0];
        let z = vec![
            vec![1.0, 2.0, 4.0],
            vec![2.0, 1.0, 5.0],
            vec![0.5, 3.0, 1.0],
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &z,
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientObservationsForFirstStage { n: 3, l: 4 }
        );
    }

    /// property-basedテスト。`two_sls/fit.rs`の`mod proptests`と同じケース生成器
    /// （`common::proptest_support`）・許容誤差（`RTOL=1e-6`）・`prop_assume!`によるフルランク
    /// 安全弁を使う。GMM固有の不変条件（`weight_type=Classical`と2SLSの一致、`S`のスカラー倍
    /// に対する点推定の不変性、丁度識別での`W`への非依存）を検証する。
    mod proptests {
        use super::*;
        use crate::iv::common::proptest_support::{IvCase, assert_approx_eq, iv_case_strategy};
        use crate::iv::two_sls::TwoSlsEstimator;
        use proptest::prelude::*;
        use std::collections::HashMap;

        fn fit_gmm(
            input: IvInput,
            weight_type: WeightType,
            cov_type: OlsCovType,
        ) -> Result<GmmEstimator, IvError> {
            GmmEstimator::fit(
                input,
                GmmType::TwoStep {
                    weight: weight_type,
                },
                true,
                cov_type,
                0.95,
            )
        }

        fn by_name(est: &GmmEstimator) -> HashMap<String, (f64, f64)> {
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

        /// `gmm_point_estimate`に渡す`(Z, X, y)`（`Z=[const, x_exog, z]`、`X=[const, x_exog,
        /// x_endog]`）を`IvCase`から直接組み立てる。
        fn design_matrices(case: &IvCase) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
            let n = case.nobs();
            let z_cols: Vec<&Vec<f64>> = case.x_exog.iter().chain(case.z.iter()).collect();
            let x_cols: Vec<&Vec<f64>> = case.x_exog.iter().chain(case.x_endog.iter()).collect();
            let z = Mat::from_fn(n, z_cols.len() + 1, |i, j| {
                if j == 0 { 1.0 } else { z_cols[j - 1][i] }
            });
            let x = Mat::from_fn(n, x_cols.len() + 1, |i, j| {
                if j == 0 { 1.0 } else { x_cols[j - 1][i] }
            });
            let y = Mat::from_fn(n, 1, |i, _| case.y[i]);
            (z, x, y)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// `weight_type=Classical`のGMM点推定は2SLSの点推定と一致する（`W₁∝(Z'Z)⁻¹`）。
            #[test]
            fn unadjusted_gmm_matches_two_sls_point_estimate(
                case in iv_case_strategy(false),
            ) {
                let gmm = fit_gmm(case.input().unwrap(), WeightType::Classical, OlsCovType::Classical);
                let two_sls = TwoSlsEstimator::fit(case.input().unwrap(), OlsCovType::Classical, 0.95);
                prop_assume!(gmm.is_ok() && two_sls.is_ok());
                let (gmm, two_sls) = (gmm.unwrap(), two_sls.unwrap());
                prop_assert_eq!(gmm.param_names(), two_sls.param_names());
                for j in 0..gmm.params().nrows() {
                    assert_approx_eq(*gmm.params().get(j, 0), *two_sls.params().get(j, 0), &format!("beta[{j}]"));
                }
            }

            /// 重み行列の`S`を正のスカラー倍しても、点推定`β̂(W)`は変わらない。
            #[test]
            fn point_estimate_is_invariant_to_positive_scaling_of_s(
                case in iv_case_strategy(false),
                c in 0.01f64..100.0,
            ) {
                let (z, x, y) = design_matrices(&case);
                let s = z.transpose() * &z;
                let s_scaled = Mat::from_fn(s.nrows(), s.ncols(), |i, j| c * *s.get(i, j));
                let unscaled = gmm_point_estimate(&z, &x, &y, &s);
                let scaled = gmm_point_estimate(&z, &x, &y, &s_scaled);
                prop_assume!(unscaled.is_ok() && scaled.is_ok());
                let (unscaled, scaled) = (unscaled.unwrap(), scaled.unwrap());
                for j in 0..unscaled.nrows() {
                    assert_approx_eq(*scaled.get(j, 0), *unscaled.get(j, 0), &format!("beta[{j}]"));
                }
            }

            /// `x_exog`の列順序を入れ替えても、名前で対応付けた係数・SEは変わらない
            /// （`weight_type=Robust`の2-step GMM、`cov_type=Hc1`）。
            #[test]
            fn coefficients_and_se_are_invariant_to_x_exog_column_order(
                case in iv_case_strategy(false),
            ) {
                let base = fit_gmm(case.input().unwrap(), WeightType::Robust, OlsCovType::Hc1);
                let permuted = case
                    .build_input(&case.shuffled_order(), &case.z, &case.y)
                    .map(|input| fit_gmm(input, WeightType::Robust, OlsCovType::Hc1));
                prop_assume!(base.is_ok());
                prop_assume!(permuted.as_ref().is_ok_and(|r| r.is_ok()));
                let (base, permuted) = (by_name(&base.unwrap()), by_name(&permuted.unwrap().unwrap()));
                for (name, (beta, se)) in &base {
                    let (p_beta, p_se) = permuted[name];
                    assert_approx_eq(p_beta, *beta, &format!("beta[{name}]"));
                    assert_approx_eq(p_se, *se, &format!("se[{name}]"));
                }
            }

            /// 丁度識別では、点推定は重み行列`W`（`weight_type`）に依存しない。
            #[test]
            fn just_identified_point_estimate_does_not_depend_on_weight_type(
                case in iv_case_strategy(true),
            ) {
                let classical = fit_gmm(case.input().unwrap(), WeightType::Classical, OlsCovType::Classical);
                let robust = fit_gmm(case.input().unwrap(), WeightType::Robust, OlsCovType::Classical);
                prop_assume!(classical.is_ok() && robust.is_ok());
                let (classical, robust) = (classical.unwrap(), robust.unwrap());
                for j in 0..classical.params().nrows() {
                    assert_approx_eq(*robust.params().get(j, 0), *classical.params().get(j, 0), &format!("beta[{j}]"));
                }
            }
        }
    }
}
