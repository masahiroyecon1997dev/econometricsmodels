//! 手計算の公式との照合のテスト。

use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::WeightType;
use crate::iv::gmm::test_support::*;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use crate::shared::group_codes::GroupCodes;
use faer::Mat;
use faer::Side;
use faer::prelude::Solve;
use statrs::distribution::ChiSquared;
use statrs::distribution::ContinuousCDF;

/// `cov_type=Classical`/`Hc0`〜`Hc3`のSEを、`GmmEstimator`とは独立に手計算した
/// サンドイッチ公式（`gmm.rs`のモジュールdoc「標準誤差・検定統計量（cov_type
/// 対応）」参照）と数値照合する。点推定（`weight_type=Robust`・`TwoStep`）は
/// `fit_computes_robust_weighted_estimate_matching_manual_formula`と同じオラクルを
/// 再利用し、そこから独立に`s_used`・最終残差を再現したうえで、cov_typeごとの
/// `Ω̂`・サンドイッチを別コードで組み立てる。
#[test]
fn fit_computes_classical_and_hc_std_errors_matching_manual_sandwich_formula() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let l = 3;
    let k = 2;

    let z = Mat::from_fn(n, l, |i, j| match j {
        0 => 1.0,
        1 => z1[i],
        _ => z2[i],
    });
    let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

    let ztz = z.transpose() * &z;
    let ztz_inv = ztz
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(l, l));
    let zty = z.transpose() * &y_mat;
    let ztx = z.transpose() * &x;
    let bread0 = ztx.transpose() * &ztz_inv * &ztx;
    let meat0 = ztx.transpose() * &ztz_inv * &zty;
    let beta0 = bread0
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
        * &meat0;
    let e0 = &y_mat - &x * &beta0;

    // weight_type=Robustの2-step: S_used = Σê₀² zᵢzᵢ'（点推定用、小標本補正無し）。
    let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
    let s_used = z_scaled0.transpose() * &z_scaled0;
    let s_inv = s_used
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(l, l));
    let bread = ztx.transpose() * &s_inv * &ztx;
    let meat_beta = ztx.transpose() * &s_inv * &zty;
    let beta1 = bread
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
        * &meat_beta;
    let residuals = &y_mat - &x * &beta1;
    let bread_inv = bread
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k));

    let df_resid = n - k;

    let input_for = || {
        IvInput::from_columns(
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
        .unwrap()
    };

    let assert_se_matches = |cov_type: CovType, omega: &Mat<f64>| {
        let meat_z = &s_inv * omega * &s_inv;
        let meat = ztx.transpose() * &meat_z * &ztx;
        let cov_params = &bread_inv * &meat * &bread_inv;

        let estimator = GmmEstimator::fit(
            input_for(),
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            cov_type.clone(),
            0.95,
        )
        .unwrap();
        for j in 0..k {
            let expected_se = (*cov_params.get(j, j)).sqrt();
            let got_se = *estimator.std_errors().get(j, 0);
            assert!(
                (got_se - expected_se).abs() < 1e-8,
                "{cov_type:?} se {j}: got {got_se}, expected {expected_se}"
            );
        }
    };

    // classical: Ω̂ = σ̂²・Z'Z（σ̂² = Σ(ê-ē)²/df_resid、中心化。この専用テストは
    // weight_type=Robustを使っており、GMMの一次条件はē=0を保証しないため
    // 中心化の有無が実際に効く。`gmm.rs`のモジュールdoc「classical」の項参照）。
    let mean_residual: f64 = (0..n).map(|i| *residuals.get(i, 0)).sum::<f64>() / (n as f64);
    let centered_ssr: f64 = (0..n)
        .map(|i| (*residuals.get(i, 0) - mean_residual).powi(2))
        .sum();
    let sigma2 = centered_ssr / (df_resid as f64);
    let omega_classical = Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)));
    assert_se_matches(CovType::Classical, &omega_classical);

    // hc0/hc1: Ω̂ = Σscaleᵢ² zᵢzᵢ'（hc1は sqrt(n/(n-k)) 補正、gmm_hc_omegaのdocコメント
    // 参照——kは構造方程式の係数の数、lではない）。
    let hc1_scale = ((n as f64) / ((n - k) as f64)).sqrt();
    for (cov_type, scale) in [(CovType::Hc0, 1.0), (CovType::Hc1, hc1_scale)] {
        let z_scaled = Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * scale * (*z.get(i, j)));
        let omega = z_scaled.transpose() * &z_scaled;
        assert_se_matches(cov_type, &omega);
    }

    // hc2/hc3: レバレッジ h_ii = zᵢ'(Z'Z)⁻¹zᵢ（Zベースの自己拡張、`gmm.rs`のモジュールdoc
    // のdocコメント参照）。
    let leverage: Vec<f64> = (0..n)
        .map(|i| {
            (0..l)
                .map(|a| {
                    (0..l)
                        .map(|b| (*z.get(i, a)) * (*ztz_inv.get(a, b)) * (*z.get(i, b)))
                        .sum::<f64>()
                })
                .sum()
        })
        .collect();
    for (cov_type, is_hc3) in [(CovType::Hc2, false), (CovType::Hc3, true)] {
        let z_scaled = Mat::from_fn(n, l, |i, j| {
            let h = leverage[i];
            let scale = if is_hc3 {
                1.0 / (1.0 - h)
            } else {
                1.0 / (1.0 - h).sqrt()
            };
            (*residuals.get(i, 0)) * scale * (*z.get(i, j))
        });
        let omega = z_scaled.transpose() * &z_scaled;
        assert_se_matches(cov_type, &omega);
    }
}

#[test]
fn fit_computes_cluster_std_errors_matching_manual_sandwich_formula() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let l = 3;
    let k = 2;
    let groups: Vec<String> = (0..n).map(|i| format!("g{}", i / (n / 4))).collect();
    let n_groups = 4.0_f64;

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
    let estimator = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Robust,
        },
        true,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    )
    .unwrap();

    let z = Mat::from_fn(n, l, |i, j| match j {
        0 => 1.0,
        1 => z1[i],
        _ => z2[i],
    });
    let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
    let ztz_inv = (z.transpose() * &z)
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(l, l));
    let zty = z.transpose() * &y_mat;
    let ztx = z.transpose() * &x;
    let bread0 = ztx.transpose() * &ztz_inv * &ztx;
    let meat0 = ztx.transpose() * &ztz_inv * &zty;
    let beta0 = bread0
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
        * &meat0;
    let e0 = &y_mat - &x * &beta0;

    let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
    let s_used = z_scaled0.transpose() * &z_scaled0;
    let s_inv = s_used
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(l, l));
    let bread = ztx.transpose() * &s_inv * &ztx;
    let meat_beta = ztx.transpose() * &s_inv * &zty;
    let beta1 = bread
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
        * &meat_beta;
    let residuals = &y_mat - &x * &beta1;
    let bread_inv = bread
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k));

    let mut s_omega = Mat::<f64>::zeros(l, l);
    for g in ["g0", "g1", "g2", "g3"] {
        let mut s_g = [0.0_f64; 3];
        for (i, group) in groups.iter().enumerate().take(n) {
            if group == g {
                let e = *residuals.get(i, 0);
                for (a, s_g_a) in s_g.iter_mut().enumerate() {
                    *s_g_a += e * (*z.get(i, a));
                }
            }
        }
        for a in 0..l {
            for b in 0..l {
                *s_omega.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }
    let correction = (n_groups / (n_groups - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
    let omega = Mat::from_fn(l, l, |i, j| correction * (*s_omega.get(i, j)));

    let meat_z = &s_inv * &omega * &s_inv;
    let meat = ztx.transpose() * &meat_z * &ztx;
    let cov_params = &bread_inv * &meat * &bread_inv;

    for j in 0..k {
        let expected_se = (*cov_params.get(j, j)).sqrt();
        let got_se = *estimator.std_errors().get(j, 0);
        assert!(
            (got_se - expected_se).abs() < 1e-8,
            "cluster se {j}: got {got_se}, expected {expected_se}"
        );
    }
}

/// `cov_type=Hac`のSEを、Bartlettカーネル（lags=2）を要素ごとのループで素朴に
/// 計算したオラクルと数値照合する。小標本補正が無いため、点推定用の
/// `hac_moment_covariance`と同じ計算式になるはず（`gmm.rs`のモジュールdoc
/// 参照）——本テストは`hac_moment_covariance`を呼ばず独立に再計算することで、
/// その主張自体を検証する。
#[test]
fn fit_computes_hac_std_errors_matching_manual_sandwich_formula() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let l = 3;
    let k = 2;

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
    let estimator = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Robust,
        },
        true,
        CovType::Hac {
            lags: Some(2),
            time_order: row_time_order(n),
        },
        0.95,
    )
    .unwrap();

    let z = Mat::from_fn(n, l, |i, j| match j {
        0 => 1.0,
        1 => z1[i],
        _ => z2[i],
    });
    let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
    let ztz_inv = (z.transpose() * &z)
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(l, l));
    let zty = z.transpose() * &y_mat;
    let ztx = z.transpose() * &x;
    let bread0 = ztx.transpose() * &ztz_inv * &ztx;
    let meat0 = ztx.transpose() * &ztz_inv * &zty;
    let beta0 = bread0
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
        * &meat0;
    let e0 = &y_mat - &x * &beta0;

    let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
    let s_used = z_scaled0.transpose() * &z_scaled0;
    let s_inv = s_used
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(l, l));
    let bread = ztx.transpose() * &s_inv * &ztx;
    let meat_beta = ztx.transpose() * &s_inv * &zty;
    let beta1 = bread
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
        * &meat_beta;
    let residuals = &y_mat - &x * &beta1;
    let bread_inv = bread
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k));

    let ze = Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * (*z.get(i, j)));
    let mut omega = Mat::<f64>::zeros(l, l);
    for a in 0..l {
        for b in 0..l {
            let mut acc = 0.0;
            for t in 0..n {
                acc += (*ze.get(t, a)) * (*ze.get(t, b));
            }
            *omega.get_mut(a, b) = acc;
        }
    }
    for lag in 1..=2usize {
        let weight = 1.0 - (lag as f64) / 3.0;
        let mut s_l = Mat::<f64>::zeros(l, l);
        for a in 0..l {
            for b in 0..l {
                let mut acc = 0.0;
                for t in lag..n {
                    acc += (*ze.get(t, a)) * (*ze.get(t - lag, b));
                }
                *s_l.get_mut(a, b) = acc;
            }
        }
        for a in 0..l {
            for b in 0..l {
                *omega.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    let meat_z = &s_inv * &omega * &s_inv;
    let meat = ztx.transpose() * &meat_z * &ztx;
    let cov_params = &bread_inv * &meat * &bread_inv;

    for j in 0..k {
        let expected_se = (*cov_params.get(j, j)).sqrt();
        let got_se = *estimator.std_errors().get(j, 0);
        assert!(
            (got_se - expected_se).abs() < 1e-8,
            "hac se {j}: got {got_se}, expected {expected_se}"
        );
    }
}

/// `gmm_wald_chi2_test`（F統計量、常にロバストWald検定・χ²）を、`GmmEstimator`とは
/// 独立に手計算したオラクルと数値照合する。
#[test]
fn fit_computes_f_statistic_matching_manual_wald_chi2_formula() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
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
        CovType::Hc1,
        0.95,
    )
    .unwrap();

    // df_model=1（const以外はendog1のみ）なので手計算は単純なスカラー計算で済む:
    // wald = β_slope² / Var(β_slope)、χ²(1)のCDFからp値を求める。
    let beta_slope = *estimator.params().get(1, 0);
    let se_slope = *estimator.std_errors().get(1, 0);
    let expected_wald = (beta_slope / se_slope).powi(2);
    let chi2 = ChiSquared::new(1.0).unwrap();
    let expected_p = 1.0 - chi2.cdf(expected_wald);

    assert!(
        (estimator.wald_statistic() - expected_wald).abs() < 1e-8,
        "got {}, expected {}",
        estimator.wald_statistic(),
        expected_wald
    );
    assert!(
        (estimator.wald_p_value() - expected_p).abs() < 1e-8,
        "got {}, expected {}",
        estimator.wald_p_value(),
        expected_p
    );
}

/// `OneStep`（ループが一度も回らず`s_used=classical_s=σ̂²₀・Z'Z`のまま）でも、
/// `cov_type`（`Hc1`、非classicalで補正が効くケース）のSEが正しく計算されることを、
/// 独立手計算オラクルと数値照合する。`gmm.rs`のモジュールdoc「標準誤差・検定統計量
/// （cov_type対応）」が主張する「`s_used`の正のスカラー倍不変性によりcov_paramsは
/// 変わらない」という設計判断が、`TwoStep`以外の経路でも成立することを検証する。
#[test]
fn fit_computes_hc1_std_errors_matching_manual_formula_with_one_step_gmm() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let l = 3;
    let k = 2;

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
    let estimator = GmmEstimator::fit(input, GmmType::OneStep, true, CovType::Hc1, 0.95).unwrap();

    let z = Mat::from_fn(n, l, |i, j| match j {
        0 => 1.0,
        1 => z1[i],
        _ => z2[i],
    });
    let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

    // OneStep: β̂はW₀=(Z'Z)⁻¹による初期推定のみ（`gmm.rs`のモジュールdoc
    // 「`GmmType`と点推定への影響」参照）。
    let ztz = z.transpose() * &z;
    let ztz_inv = ztz
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(l, l));
    let zty = z.transpose() * &y_mat;
    let ztx = z.transpose() * &x;
    let bread = ztx.transpose() * &ztz_inv * &ztx;
    let beta = bread
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
        * (ztx.transpose() * &ztz_inv * &zty);
    let residuals = &y_mat - &x * &beta;

    // s_used = classical_s = σ̂²₀・Z'Z（step-0残差=最終残差、Hansen J用と同じ計算）。
    let sigma2_0: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
    let s_used = Mat::from_fn(l, l, |i, j| sigma2_0 * (*ztz.get(i, j)));
    let s_inv = s_used
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(l, l));
    let bread_w = ztx.transpose() * &s_inv * &ztx;
    let bread_inv = bread_w
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k));

    // Ω̂_hc1 = n/(n-k)補正込みのΣêᵢ²zᵢzᵢ'。
    let hc1_scale = ((n as f64) / ((n - k) as f64)).sqrt();
    let z_scaled = Mat::from_fn(n, l, |i, j| {
        (*residuals.get(i, 0)) * hc1_scale * (*z.get(i, j))
    });
    let omega = z_scaled.transpose() * &z_scaled;

    let meat_z = &s_inv * &omega * &s_inv;
    let meat = ztx.transpose() * &meat_z * &ztx;
    let cov_params = &bread_inv * &meat * &bread_inv;

    for j in 0..k {
        let expected_se = (*cov_params.get(j, j)).sqrt();
        let got_se = *estimator.std_errors().get(j, 0);
        assert!(
            (got_se - expected_se).abs() < 1e-8,
            "one_step hc1 se {j}: got {got_se}, expected {expected_se}"
        );
    }
}
