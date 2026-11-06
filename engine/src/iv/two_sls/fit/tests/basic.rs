//! `fit`の基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）のテスト。

use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::iv::two_sls::test_support::*;
use crate::linear::ols::CovType;
use faer::Mat;

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
fn fit_pins_faer_global_parallelism_to_seq() {
    // `fit()` 冒頭の `crate::shared::parallelism::ensure_serial()` が faer の
    // グローバル並列度を `Par::Seq` へ引き戻すことの回帰ガード（iv 系統代表）。
    // 別テストが `Seq` にしている可能性があるため、まず `Rayon` に戻してから通す。
    // 設計行列は極小なので一時的な `Rayon` 設定は病理を招かない。
    faer::set_global_parallelism(faer::Par::rayon(0));

    let _ = TwoSlsEstimator::fit(just_identified_input(), CovType::Classical, 0.95).unwrap();

    assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
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
    let weak = TwoSlsEstimator::fit(build_input(&weak_x_endog), CovType::Classical, 0.95).unwrap();
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
