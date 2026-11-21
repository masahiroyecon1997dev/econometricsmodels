//! 有効自由度とPatnaik臨界値の式(論文のTable 1・公表値を含む)。

use super::*;

// 論文のTable 1: `K_eff = 1..30`行、`x = 1/τ`が3.33・5・10・20の列(上側5%点)。
// 3.33の列は`x = 10/3`の値。
const TABLE_1: [[f64; 4]; 30] = [
    [12.05, 15.06, 23.11, 37.42],
    [9.57, 12.17, 19.29, 32.32],
    [8.53, 10.95, 17.67, 30.13],
    [7.92, 10.23, 16.72, 28.85],
    [7.51, 9.75, 16.08, 27.98],
    [7.21, 9.40, 15.62, 27.35],
    [6.98, 9.14, 15.26, 26.86],
    [6.80, 8.92, 14.97, 26.47],
    [6.65, 8.74, 14.73, 26.15],
    [6.52, 8.59, 14.53, 25.87],
    [6.41, 8.47, 14.36, 25.64],
    [6.32, 8.36, 14.21, 25.44],
    [6.24, 8.26, 14.08, 25.26],
    [6.16, 8.17, 13.96, 25.10],
    [6.10, 8.10, 13.86, 24.96],
    [6.04, 8.03, 13.77, 24.83],
    [5.99, 7.96, 13.68, 24.71],
    [5.94, 7.91, 13.60, 24.60],
    [5.89, 7.85, 13.53, 24.50],
    [5.85, 7.80, 13.46, 24.41],
    [5.81, 7.76, 13.40, 24.33],
    [5.78, 7.72, 13.35, 24.25],
    [5.74, 7.68, 13.29, 24.18],
    [5.71, 7.64, 13.24, 24.11],
    [5.68, 7.61, 13.20, 24.05],
    [5.66, 7.57, 13.15, 23.98],
    [5.63, 7.54, 13.11, 23.93],
    [5.61, 7.51, 13.07, 23.87],
    [5.58, 7.49, 13.04, 23.82],
    [5.56, 7.46, 13.00, 23.77],
];

/// `Ŵ2 = I_K`なら`K_eff = K`で、臨界値は論文の公表値(小数第2位)に一致する。
#[test]
fn patnaik_critical_values_reproduce_table_1_of_the_paper() {
    let xs = [10.0 / 3.0, 5.0, 10.0, 20.0];
    for (row, published) in TABLE_1.iter().enumerate() {
        let k = row + 1;
        for (x, expected) in xs.iter().zip(published) {
            let (df, cv) = patnaik_critical_value(&vec![1.0; k], *x, 0.05).unwrap();
            assert!((df - k as f64).abs() < 1e-12);
            assert!(
                (cv - expected).abs() <= 0.0051,
                "K={k} x={x}: {cv} vs published {expected}"
            );
        }
    }
}

/// 固有値(3, 1)の`Ŵ2`: `tr = 4`、`tr(Ŵ2²) = 10`、`λ_max = 3`なので、`x = 10`で
/// `K_eff = 16·21/(10 + 20·4·3) = 1.344`。臨界値の期待値はscipyの`ncx2.ppf`。
#[test]
fn effective_degrees_of_freedom_and_critical_value_follow_the_formula() {
    let (df, cv) = patnaik_critical_value(&[3.0, 1.0], 10.0, 0.05).unwrap();
    close(df, 1.344, 1e-12);
    close(cv, 21.295336706396494, 1e-9);
    let (df, cv) = patnaik_critical_value(&[3.0, 1.0], 3.0, 0.05).unwrap();
    close(df, 1.3658536585365855, 1e-12);
    close(cv, 10.219559398553013, 1e-9);
    // 固有値のスケールには依らない。
    let (df_scaled, cv_scaled) = patnaik_critical_value(&[300.0, 100.0], 10.0, 0.05).unwrap();
    close(df_scaled, 1.344, 1e-12);
    close(cv_scaled, 21.295336706396494, 1e-9);
}

/// 任意の半正定値の`Ŵ2`で`1 ≤ K_eff ≤ K`。
#[test]
fn effective_degrees_of_freedom_lies_between_one_and_the_number_of_instruments() {
    let spectra: [&[f64]; 4] = [
        &[1.0, 1.0, 1.0],
        &[10.0, 1.0, 0.1],
        &[5.0, 0.0, 0.0],
        &[2.0, 1.0, 1.0, 0.5],
    ];
    for eigenvalues in spectra {
        for x in [0.0, 3.0, 10.0, 20.0] {
            let (df, _) = patnaik_critical_value(eigenvalues, x, 0.05).unwrap();
            assert!(
                df >= 1.0 - 1e-12 && df <= eigenvalues.len() as f64 + 1e-12,
                "{df}"
            );
        }
    }
    assert!(patnaik_critical_value(&[0.0, 0.0], 10.0, 0.05).is_none());
}

/// `τ`が小さい(バイアスに厳しい)ほど臨界値は大きく、`α`が小さいほど大きい。
#[test]
fn critical_values_increase_with_stricter_thresholds_and_levels() {
    let (endog, exog, z) = large(4);
    let cov = ScoreCovariance::Hc(HcVariant::Hc1);
    let at = |alpha| effective_f(&endog, &exog, &z, 0, cov, alpha).unwrap();
    let (loose, strict) = (at(0.10), at(0.05));
    for window in strict.simplified_tsls.windows(2) {
        assert!(window[0].critical_value > window[1].critical_value);
        assert!(window[0].tau < window[1].tau);
    }
    for (a, b) in loose.simplified_tsls.iter().zip(strict.simplified_tsls) {
        assert!(b.critical_value > a.critical_value);
        assert_eq!(a.x, b.x);
    }
    assert_eq!(strict.n_instruments, 4);
    assert_eq!(strict.alpha, 0.05);
    let taus: Vec<f64> = strict.simplified_tsls.iter().map(|c| c.tau).collect();
    assert_eq!(taus, BIAS_THRESHOLDS);
    for cv in strict.simplified_tsls {
        close(cv.x, 1.0 / cv.tau, 1e-15);
    }
}

/// 論文2.2.3節の公表値: 同次分散で一般化TSLS、`τ = 0.10`、`α = 0.05`の臨界値は、`K = 3`で8.53、
/// `K = 30`で12.27(小数第2位)。
#[test]
fn generalized_critical_values_reproduce_the_published_homoskedastic_values() {
    for (k, published) in [(3usize, 8.53), (30, 12.27)] {
        let bound = (1.0 - 2.0 / k as f64).abs();
        let (df, cv) = patnaik_critical_value(&vec![1.0; k], bound / 0.10, 0.05).unwrap();
        assert!((df - k as f64).abs() < 1e-12);
        assert!(
            (cv - published).abs() <= 0.0051,
            "K={k}: {cv} vs {published}"
        );
    }
}
