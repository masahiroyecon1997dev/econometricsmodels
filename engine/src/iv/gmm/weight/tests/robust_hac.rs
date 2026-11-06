//! ロバスト重み・HAC重み（ラグの解決・時間順序・手計算との照合）のテスト。

use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::test_support::*;
use crate::iv::gmm::weight::WeightType;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use faer::Mat;
use faer::Side;
use faer::prelude::Solve;

/// 過剰識別・不均一分散なデータで、`WeightType::Robust`の点推定が
/// `Classical`（=2SLS）と異なることを確認する（`weight_type`が実際に点推定へ
/// 影響することの動作確認。手計算オラクルとの数値一致は
/// `fit_computes_robust_weighted_estimate_matching_manual_formula`で検証する）。
#[test]
fn fit_with_robust_weight_type_differs_from_classical_when_heteroskedastic() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let build_input = || {
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

    let classical = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();
    let robust = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Robust,
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    let diff = (*classical.params().get(1, 0) - *robust.params().get(1, 0)).abs();
    assert!(
        diff > 1e-4,
        "expected robust weighting to change the point estimate, diff={diff}"
    );
}

/// `WeightType::Robust`の点推定を、`GmmEstimator::fit`とは独立に`faer`演算で
/// 手計算したオラクルと数値照合する（`two_sls/fit/tests/reference.rs`の
/// `fit_matches_independently_recomputed_projection_formula_*`と同じ検証方針）。
#[test]
fn fit_computes_robust_weighted_estimate_matching_manual_formula() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
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
        CovType::Classical,
        0.95,
    )
    .unwrap();

    // オラクル: 手作業でZ・X・yを組み立て、W₀=(Z'Z)⁻¹で初期推定→残差→
    // S=Σêᵢ²zᵢzᵢ'→W₁=S⁻¹で最終推定、という同じ2段階を`gmm_point_estimate`とは
    // 別コードで再現する。
    let z = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => z1[i],
        _ => z2[i],
    });
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

    let ztz = z.transpose() * &z;
    let ztz_inv = ztz
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(3, 3));
    let zty = z.transpose() * &y_mat;
    let ztx = z.transpose() * &x;
    let bread0 = ztx.transpose() * &ztz_inv * &ztx;
    let meat0 = ztx.transpose() * &ztz_inv * &zty;
    let beta0 = bread0
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(2, 2))
        * &meat0;
    let e0 = &y_mat - &x * &beta0;

    let z_scaled = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
    let s = z_scaled.transpose() * &z_scaled;
    let s_inv = s
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(3, 3));
    let bread1 = ztx.transpose() * &s_inv * &ztx;
    let meat1 = ztx.transpose() * &s_inv * &zty;
    let expected_beta = bread1
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(2, 2))
        * &meat1;

    for j in 0..2 {
        assert!(
            (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
            "param {j}: got {}, expected {}",
            *estimator.params().get(j, 0),
            *expected_beta.get(j, 0)
        );
    }
}

/// `weight_type=Hac`（`lags=0`）は`Robust`と一致するはず（HAC0=HC0と同じ関係、
/// `two_sls/cov/tests/hac.rs`の`fit_hac_with_zero_lags_matches_hc0`と同型の検証）。
#[test]
fn fit_hac_with_zero_lags_matches_robust() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let build_input = || {
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

    let robust = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Robust,
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();
    let hac = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Hac {
                lags: Some(0),
                time_order: row_time_order(16),
            },
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    for j in 0..2 {
        assert!(
            (*robust.params().get(j, 0) - *hac.params().get(j, 0)).abs() < 1e-8,
            "param {j}: robust={}, hac={}",
            *robust.params().get(j, 0),
            *hac.params().get(j, 0)
        );
    }
}

/// `weight_type=Hac`（`lags=2`）の点推定を、`hac_moment_covariance`（`matmul`
/// ベース）とは独立に、素朴なループでBartlettカーネル重み付き和を手計算したオラクルと
/// 数値照合する（`fit_hac_with_zero_lags_matches_robust`は`lags=0`（ラグ項ループが
/// 空になる自明ケース）のみで、ラグ重み付け本体は未検証だったため追加）。
#[test]
fn fit_computes_hac_weighted_estimate_matching_manual_formula_with_lags_two() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
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
            weight: WeightType::Hac {
                lags: Some(2),
                time_order: row_time_order(n),
            },
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    let z = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => z1[i],
        _ => z2[i],
    });
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

    let ztz = z.transpose() * &z;
    let ztz_inv = ztz
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(3, 3));
    let zty = z.transpose() * &y_mat;
    let ztx = z.transpose() * &x;
    let bread0 = ztx.transpose() * &ztz_inv * &ztx;
    let meat0 = ztx.transpose() * &ztz_inv * &zty;
    let beta0 = bread0
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(2, 2))
        * &meat0;
    let e0 = &y_mat - &x * &beta0;

    // Bartlettカーネル（lags=2、time_order無指定なので行順=時系列順）を、
    // `hac_moment_covariance`のmatmulベース実装とは別に、要素ごとの
    // ループで素朴に計算する。
    let ze = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
    let mut s = Mat::<f64>::zeros(3, 3);
    for a in 0..3 {
        for b in 0..3 {
            let mut acc = 0.0;
            for t in 0..n {
                acc += (*ze.get(t, a)) * (*ze.get(t, b));
            }
            *s.get_mut(a, b) = acc;
        }
    }
    for lag in 1..=2usize {
        let weight = 1.0 - (lag as f64) / 3.0;
        let mut s_l = Mat::<f64>::zeros(3, 3);
        for a in 0..3 {
            for b in 0..3 {
                let mut acc = 0.0;
                for t in lag..n {
                    acc += (*ze.get(t, a)) * (*ze.get(t - lag, b));
                }
                *s_l.get_mut(a, b) = acc;
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                *s.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    let s_inv = s
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(3, 3));
    let bread1 = ztx.transpose() * &s_inv * &ztx;
    let meat1 = ztx.transpose() * &s_inv * &zty;
    let expected_beta = bread1
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(2, 2))
        * &meat1;

    for j in 0..2 {
        assert!(
            (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
            "param {j}: got {}, expected {}",
            *estimator.params().get(j, 0),
            *expected_beta.get(j, 0)
        );
    }
}

/// `lags=None`（経験則自動計算`L = floor(4*(n/100)^(2/9))`）が、`heteroskedastic_test_
/// columns()`の`n=16`では`L=2`と一致するため、`lags=Some(2)`を明示指定した場合と
/// 数値的に一致するはず（`two_sls/cov/tests/hac.rs`の
/// `fit_computes_hac_std_errors_with_auto_lags_matching_explicit_lags`と同じ検証方針）。
#[test]
fn fit_with_hac_weight_type_and_auto_lags_matches_explicit_lags_two() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let build_input = || {
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

    let auto_estimator = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Hac {
                lags: None,
                time_order: row_time_order(16),
            },
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();
    let explicit_estimator = GmmEstimator::fit(
        build_input(),
        GmmType::TwoStep {
            weight: WeightType::Hac {
                lags: Some(2),
                time_order: row_time_order(16),
            },
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    for j in 0..2 {
        assert!(
            (*auto_estimator.params().get(j, 0) - *explicit_estimator.params().get(j, 0)).abs()
                < 1e-8
        );
    }
}

/// `hac_lags_used()`は`cov_type=Hac`または`weight_type=Hac`のどちらかが該当すれば
/// 解決後のラグ数（`n=16`の自動計算は2）、どちらも該当しなければ`None`。両方がHacで
/// `lags`が異なる場合は`cov_type`側を優先する。
#[test]
fn hac_lags_used_covers_cov_type_and_weight_type() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let hac = |lags: Option<i64>| CovType::Hac {
        lags,
        time_order: row_time_order(16),
    };
    let hac_weight = |lags: Option<i64>| WeightType::Hac {
        lags,
        time_order: row_time_order(16),
    };
    let lags_used = |gmm_type: GmmType, cov_type: CovType| {
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
        // 収束は検証対象外（反復GMMが`max_iter`内に収束しなくてもラグは報告される）
        GmmEstimator::fit(input, gmm_type, false, cov_type, 0.95)
            .unwrap()
            .hac_lags_used()
    };

    // どちらもHacでない
    assert_eq!(lags_used(GmmType::OneStep, CovType::Classical), None);
    assert_eq!(
        lags_used(
            GmmType::TwoStep {
                weight: WeightType::Robust
            },
            CovType::Hc1
        ),
        None
    );
    // `cov_type=Hac`のみ（1段階は`weight_type`を持たない）
    assert_eq!(lags_used(GmmType::OneStep, hac(None)), Some(2));
    assert_eq!(
        lags_used(
            GmmType::TwoStep {
                weight: WeightType::Robust
            },
            hac(Some(3))
        ),
        Some(3)
    );
    // `weight_type=Hac`のみ
    assert_eq!(
        lags_used(
            GmmType::TwoStep {
                weight: hac_weight(None)
            },
            CovType::Classical
        ),
        Some(2)
    );
    // 反復GMMでも`weight_type=Hac`のラグが報告される
    assert_eq!(
        lags_used(
            GmmType::Iterated {
                weight: hac_weight(None),
                max_iter: 5,
                tol: 1e-6
            },
            CovType::Classical
        ),
        Some(2)
    );
    // 両方Hacで`lags`が異なる場合は`cov_type`側
    assert_eq!(
        lags_used(
            GmmType::TwoStep {
                weight: hac_weight(Some(1))
            },
            hac(Some(3))
        ),
        Some(3)
    );
}

/// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
/// ラグ付きモーメント共分散を計算することを確認する（`two_sls/cov/tests/hac.rs`の
/// `fit_computes_hac_std_errors_respecting_time_order`と同じ検証方針）。
#[test]
fn fit_with_hac_weight_type_respects_time_order() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let shuffle: Vec<usize> = vec![3, 1, 6, 0, 5, 2, 7, 4, 11, 9, 14, 8, 13, 10, 15, 12];
    assert_eq!(shuffle.len(), n);
    let shuffled_time: Vec<f64> = shuffle.iter().map(|&i| i as f64).collect();
    let shuffled_y: Vec<f64> = shuffle.iter().map(|&i| y[i]).collect();
    let shuffled_x_endog: Vec<f64> = shuffle.iter().map(|&i| x_endog[i]).collect();
    let shuffled_z1: Vec<f64> = shuffle.iter().map(|&i| z1[i]).collect();
    let shuffled_z2: Vec<f64> = shuffle.iter().map(|&i| z2[i]).collect();

    let shuffled_input = IvInput::from_columns(
        &shuffled_y,
        &[],
        vec![],
        std::slice::from_ref(&shuffled_x_endog),
        vec!["endog1".to_string()],
        &[shuffled_z1, shuffled_z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let shuffled_estimator = GmmEstimator::fit(
        shuffled_input,
        GmmType::TwoStep {
            weight: WeightType::Hac {
                lags: Some(2),
                time_order: shuffled_time,
            },
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    let unshuffled_input = IvInput::from_columns(
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
    let unshuffled_estimator = GmmEstimator::fit(
        unshuffled_input,
        GmmType::TwoStep {
            weight: WeightType::Hac {
                lags: Some(2),
                time_order: row_time_order(n),
            },
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    for j in 0..2 {
        assert!(
            (*shuffled_estimator.params().get(j, 0) - *unshuffled_estimator.params().get(j, 0))
                .abs()
                < 1e-8
        );
    }
}
