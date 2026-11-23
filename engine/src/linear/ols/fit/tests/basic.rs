//! `fit`の基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）のテスト。

use super::exact_fit_input;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;

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
    let input =
        OlsInput::from_columns(&y, &[x], vec!["x1".to_string()], true, "y".to_string()).unwrap();

    assert!(OlsEstimator::fit(input, CovType::Classical, 0.95).is_ok());
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
        OlsInput::from_columns(&y, &[x1], vec!["x1".to_string()], true, "y".to_string()).unwrap();

    let _ = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

    assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
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
