//! `fit`の結果を参照実装（statsmodels・linearmodels・plm・R）や閉形式・独立再計算と照合するテスト。

use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::iv::two_sls::test_support::*;
use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
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
        OlsInput::from_columns(&y, &x_exog, vec!["x1".to_string()], true, "y".to_string()).unwrap();
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
            (*estimator.p_values().get(j, 0) - *ols_estimator.p_values().get(j, 0)).abs() < 1e-8
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
