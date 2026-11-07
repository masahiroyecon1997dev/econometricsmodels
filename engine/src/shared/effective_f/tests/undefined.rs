//! 統計量が定義できない入力(`None`になる条件)。

use super::*;

#[test]
fn it_is_undefined_when_the_statistic_cannot_be_formed() {
    let (endog, exog, z) = small();
    let classical = ScoreCovariance::Classical;
    let eff = effective_f;
    // 内生変数が2個・0個、操作変数が0個。
    let two = Mat::from_fn(12, 2, |i, j| {
        *endog.get(i, 0) * [1.0, 2.0][j] + j as f64 * (i as f64)
    });
    assert!(eff(&two, &exog, &z, 0, classical, 0.05).is_none());
    assert!(eff(&Mat::<f64>::zeros(12, 0), &exog, &z, 0, classical, 0.05).is_none());
    assert!(eff(&endog, &exog, &Mat::<f64>::zeros(12, 0), 0, classical, 0.05).is_none());
    // 水準が範囲外。
    for alpha in [0.0, 1.0, -0.1, f64::NAN] {
        assert!(eff(&endog, &exog, &z, 0, classical, alpha).is_none());
    }
    // 自由度が残らない(12 − 3 − 2 − 7 = 0)。
    assert!(eff(&endog, &exog, &z, 7, classical, 0.05).is_none());
    // HC2・HC3は吸収したパラメータがあると未定義。
    for variant in [HcVariant::Hc2, HcVariant::Hc3] {
        let cov = ScoreCovariance::Hc(variant);
        assert!(eff(&endog, &exog, &z, 1, cov, 0.05).is_none());
        assert!(eff(&endog, &exog, &z, 0, cov, 0.05).is_some());
    }
    // 操作変数が外生変数の線形結合・操作変数どうしが共線。
    let in_span = Mat::from_fn(12, 3, |i, j| {
        if j == 0 {
            *exog.get(i, 1) * 2.0 + 1.0
        } else {
            *z.get(i, j)
        }
    });
    assert!(eff(&endog, &exog, &in_span, 0, classical, 0.05).is_none());
    let collinear = Mat::from_fn(12, 3, |i, j| {
        if j == 2 {
            *z.get(i, 0) * 3.0
        } else {
            *z.get(i, j)
        }
    });
    assert!(eff(&endog, &exog, &collinear, 0, classical, 0.05).is_none());
    // 一次段階が完全フィット。
    let perfect = Mat::from_fn(12, 1, |i, _| {
        2.0 * *z.get(i, 0) - *z.get(i, 1) + *exog.get(i, 1)
    });
    assert!(eff(&perfect, &exog, &z, 0, classical, 0.05).is_none());
    // 非有限値。
    let mut nan_endog = endog.clone();
    *nan_endog.get_mut(3, 0) = f64::NAN;
    assert!(eff(&nan_endog, &exog, &z, 0, classical, 0.05).is_none());
    // クラスターが1つ・行数の合わないクラスター・時間順。
    let one = GroupCodes::from_labels_without_keys(&vec!["a".to_string(); 12]);
    assert!(eff(&endog, &exog, &z, 0, ScoreCovariance::Cluster(&one), 0.05).is_none());
    let short = [0usize, 1, 2];
    assert!(
        eff(
            &endog,
            &exog,
            &z,
            0,
            ScoreCovariance::Hac {
                lags: 1,
                order: &short
            },
            0.05
        )
        .is_none()
    );
}

#[test]
fn psd_eigenvalues_clamps_rounding_noise_and_rejects_indefinite_matrices() {
    let diagonal = |a: f64, b: f64| {
        Mat::from_fn(2, 2, |i, j| {
            if i != j {
                0.0
            } else if i == 0 {
                a
            } else {
                b
            }
        })
    };
    assert_eq!(
        psd_eigenvalues(&diagonal(1.0, -1e-14)).map(|v| v[0].max(v[1])),
        Some(1.0)
    );
    let clamped = psd_eigenvalues(&diagonal(1.0, -1e-14)).unwrap();
    assert!(clamped.iter().all(|v| *v >= 0.0));
    assert!(psd_eigenvalues(&diagonal(1.0, -0.5)).is_none());
    assert!(psd_eigenvalues(&diagonal(0.0, 0.0)).is_none());
    assert!(psd_eigenvalues(&diagonal(f64::NAN, 1.0)).is_none());
}

/// 自由度が残らないモデルは、自由度を使わないHC0でも未定義(Cragg-Donald・Kleibergen-Paapと同じ)。
#[test]
fn a_model_without_residual_degrees_of_freedom_is_undefined_for_every_covariance() {
    let (endog, exog, z) = small();
    // n = 12, K = 3, k_exog = 2: n_absorbed = 7 で n_params = n。
    for cov in [
        ScoreCovariance::Classical,
        ScoreCovariance::Hc(HcVariant::Hc0),
        ScoreCovariance::Hc(HcVariant::Hc1),
    ] {
        assert!(effective_f(&endog, &exog, &z, 7, cov, 0.05).is_none());
        assert!(effective_f(&endog, &exog, &z, 6, cov, 0.05).is_some());
    }
}

#[test]
fn infinite_inputs_and_unusable_lags_are_undefined() {
    let (endog, exog, z) = small();
    let classical = ScoreCovariance::Classical;
    let mut infinite = endog.clone();
    *infinite.get_mut(2, 0) = f64::INFINITY;
    assert!(effective_f(&infinite, &exog, &z, 0, classical, 0.05).is_none());
    let mut neg_infinite = z.clone();
    *neg_infinite.get_mut(5, 1) = f64::NEG_INFINITY;
    assert!(effective_f(&endog, &exog, &neg_infinite, 0, classical, 0.05).is_none());
    let order: Vec<usize> = (0..12).collect();
    let hac = |lags| ScoreCovariance::Hac {
        lags,
        order: &order,
    };
    assert!(effective_f(&endog, &exog, &z, 0, hac(11), 0.05).is_some());
    assert!(effective_f(&endog, &exog, &z, 0, hac(12), 0.05).is_none());
}

/// 完全フィットの境界は、Cragg-Donald・Kleibergen-Paapと同じ基準(`1 − R²`が約1.4e-14)。
#[test]
fn the_perfect_fit_boundary_matches_cragg_donald() {
    let (_, exog, z) = small();
    let noise = [
        0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.2, 0.05, -0.45, 0.3, 0.15, -0.1,
    ];
    for (scale, expect_some) in [(1e-5, true), (1e-9, false)] {
        let endog = Mat::from_fn(12, 1, |i, _| {
            2.0 * *z.get(i, 0) - *z.get(i, 1) + *exog.get(i, 1) + scale * noise[i]
        });
        let eff = effective_f(&endog, &exog, &z, 0, ScoreCovariance::Classical, 0.05);
        let cd = cragg_donald(&endog, &exog, &z, 0);
        assert_eq!(eff.is_some(), expect_some, "scale={scale}");
        assert_eq!(cd.is_some(), expect_some, "scale={scale}");
    }
}

/// 被説明変数の形が不正、または内生変数と完全に比例するときは、一般化版だけが未定義。
#[test]
fn the_generalized_part_is_undefined_for_unusable_dependent_variables() {
    let (endog, exog, z) = small();
    let classical = ScoreCovariance::Classical;
    let run = |y: &Mat<f64>| {
        crate::shared::effective_f::effective_f(&endog, &exog, &z, 0, classical, 0.05, Some(y))
            .unwrap()
    };
    let two_columns = Mat::from_fn(12, 2, |i, j| (i + j) as f64);
    let wrong_rows = Mat::from_fn(5, 1, |i, _| i as f64);
    let proportional = Mat::from_fn(12, 1, |i, _| 2.5 * *endog.get(i, 0));
    let in_exog_span = Mat::from_fn(12, 1, |i, _| 3.0 + 2.0 * *exog.get(i, 1));
    let mut with_nan = Mat::from_fn(12, 1, |i, _| (i as f64 * 0.9).sin());
    *with_nan.get_mut(4, 0) = f64::NAN;
    let mut with_infinity = with_nan.clone();
    *with_infinity.get_mut(4, 0) = f64::INFINITY;
    for y in [
        &two_columns,
        &wrong_rows,
        &proportional,
        &in_exog_span,
        &with_nan,
        &with_infinity,
    ] {
        let eff = run(y);
        assert!(eff.tsls.is_none() && eff.bias_bound.is_none());
        assert!(eff.statistic.is_finite());
    }
}

/// `Ŵ2`が特異(クラスター数が操作変数の数より少ない)でも、トレースと固有値で定義できる。
#[test]
fn a_singular_covariance_still_gives_a_value() {
    let (endog, exog, z) = small();
    let labels: Vec<String> = (0..12).map(|i| ["a", "b"][i / 6].to_string()).collect();
    let two = GroupCodes::from_labels_without_keys(&labels);
    let eff = effective_f(&endog, &exog, &z, 0, ScoreCovariance::Cluster(&two), 0.05).unwrap();
    assert!(eff.statistic.is_finite() && eff.statistic > 0.0);
    for cv in eff.simplified_tsls {
        assert!(cv.effective_df >= 1.0 && cv.effective_df <= 3.0);
    }
}
