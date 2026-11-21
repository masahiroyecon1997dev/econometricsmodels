//! `OlsEstimator::fit_releasing_design`のテスト: `fit`と同じ結果・同じエラーで、設計行列だけを持たない。

use crate::linear::common::LeastSquaresError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::shared::group_codes::GroupCodes;

const N: usize = 60;

/// 決定的な非自明データ（切片 + 4列、不均一な誤差）。
fn data(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let x: Vec<Vec<f64>> = (0..4)
        .map(|j| {
            (0..n)
                .map(|i| ((i * (3 + 2 * j) + 5 * j) % 17) as f64 - 8.0 + 0.1 * j as f64)
                .collect()
        })
        .collect();
    let y = (0..n)
        .map(|i| {
            let noise = (((i * 11) % 7) as f64 - 3.0) * (1.0 + (i % 4) as f64 * 0.5);
            1.0 + x[0][i] - 0.5 * x[1][i] + 0.3 * x[2][i] + 0.2 * x[3][i] + noise
        })
        .collect();
    (y, x)
}

fn names() -> Vec<String> {
    ["a", "b", "c", "d"].map(String::from).to_vec()
}

fn input(y: &[f64], x: &[Vec<f64>]) -> OlsInput {
    OlsInput::from_columns(y, x, names(), true, "y".to_string()).unwrap()
}

fn cov_types() -> Vec<CovType> {
    let groups: Vec<String> = (0..N).map(|i| format!("g{}", i % 6)).collect();
    vec![
        CovType::Classical,
        CovType::Hc0,
        CovType::Hc1,
        CovType::Hc2,
        CovType::Hc3,
        CovType::Hac {
            lags: Some(2),
            time_order: (0..N).map(|i| i as f64).collect(),
        },
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
    ]
}

#[test]
fn fit_releasing_design_matches_fit_for_every_cov_type() {
    let (y, x) = data(N);
    for cov_type in cov_types() {
        let borrowed = OlsEstimator::fit(input(&y, &x), cov_type.clone(), 0.95).unwrap();
        let released =
            OlsEstimator::fit_releasing_design(input(&y, &x), cov_type.clone(), 0.95).unwrap();
        let label = format!("{cov_type:?}");

        assert_eq!(released.params(), borrowed.params(), "{label}");
        assert_eq!(released.residuals(), borrowed.residuals(), "{label}");
        assert_eq!(released.std_errors(), borrowed.std_errors(), "{label}");
        assert_eq!(released.cov_params(), borrowed.cov_params(), "{label}");
        assert_eq!(released.test_stats(), borrowed.test_stats(), "{label}");
        assert_eq!(released.p_values(), borrowed.p_values(), "{label}");
        assert_eq!(released.conf_lower(), borrowed.conf_lower(), "{label}");
        assert_eq!(released.conf_upper(), borrowed.conf_upper(), "{label}");
        assert_eq!(released.r_squared(), borrowed.r_squared(), "{label}");
        assert_eq!(
            released.adj_r_squared(),
            borrowed.adj_r_squared(),
            "{label}"
        );
        assert_eq!(released.f_statistic(), borrowed.f_statistic(), "{label}");
        assert_eq!(released.f_p_value(), borrowed.f_p_value(), "{label}");
        assert_eq!(
            released.log_likelihood(),
            borrowed.log_likelihood(),
            "{label}"
        );
        assert_eq!(released.aic(), borrowed.aic(), "{label}");
        assert_eq!(released.bic(), borrowed.bic(), "{label}");
        assert_eq!(released.df_resid(), borrowed.df_resid(), "{label}");
        assert_eq!(released.df_inference(), borrowed.df_inference(), "{label}");
        assert_eq!(
            released.hac_lags_used(),
            borrowed.hac_lags_used(),
            "{label}"
        );
        assert_eq!(
            released.wald_test_last_columns(2).unwrap(),
            borrowed.wald_test_last_columns(2).unwrap(),
            "{label}"
        );
    }
}

#[test]
fn fit_releasing_design_fitted_values_are_q_times_the_coordinates() {
    let (y, x) = data(N);
    let borrowed = OlsEstimator::fit(input(&y, &x), CovType::Classical, 0.95).unwrap();
    let released =
        OlsEstimator::fit_releasing_design(input(&y, &x), CovType::Classical, 0.95).unwrap();

    // `Xβ̂`と丸め誤差の範囲（相対`1e-12`）で一致する。
    let (got, expected) = (released.fitted_values(), borrowed.fitted_values());
    assert_eq!((got.nrows(), got.ncols()), (N, 1));
    let scale = (0..N)
        .map(|i| expected.get(i, 0).abs())
        .fold(0.0_f64, f64::max);
    for i in 0..N {
        assert!(
            (got.get(i, 0) - expected.get(i, 0)).abs() <= 1e-12 * scale,
            "row {i}: {} vs {}",
            got.get(i, 0),
            expected.get(i, 0)
        );
    }
    // 残差は`y - 当てはめ値`と同じ演算で作る（当てはめ値と残差が同じ`Q·b̃`から出る）。
    for (i, yi) in y.iter().enumerate() {
        assert_eq!(*released.residuals().get(i, 0), yi - *got.get(i, 0));
    }
}

#[test]
fn fit_releasing_design_matches_fit_for_a_weighted_input() {
    let (y, x) = data(N);
    let weights: Vec<f64> = (0..N).map(|i| 0.5 + (i % 5) as f64 * 0.3).collect();
    let weighted = || {
        OlsInput::from_columns_weighted(&y, &x, names(), true, "y".to_string(), &weights).unwrap()
    };
    let borrowed = OlsEstimator::fit(weighted(), CovType::Hc1, 0.95).unwrap();
    let released = OlsEstimator::fit_releasing_design(weighted(), CovType::Hc1, 0.95).unwrap();
    assert_eq!(released.params(), borrowed.params());
    assert_eq!(released.std_errors(), borrowed.std_errors());
    assert_eq!(released.residuals(), borrowed.residuals());
}

#[test]
fn fit_releasing_design_keeps_the_metadata_and_refuses_to_expose_the_design() {
    let (y, x) = data(N);
    let released =
        OlsEstimator::fit_releasing_design(input(&y, &x), CovType::Classical, 0.95).unwrap();
    let released_input = released.input();
    assert_eq!(released_input.nobs(), N);
    assert_eq!(released_input.k(), 5);
    assert_eq!(released_input.param_names()[0], "const");
    assert_eq!(released_input.dep_var_name(), "y");
    assert!(released_input.has_intercept());
    assert_eq!(released_input.y().nrows(), N);
}

#[test]
#[should_panic(expected = "the design matrix was released")]
fn fit_releasing_design_makes_the_design_unavailable() {
    let (y, x) = data(N);
    let released =
        OlsEstimator::fit_releasing_design(input(&y, &x), CovType::Classical, 0.95).unwrap();
    let _ = released.input().x();
}

/// `fit`と`fit_releasing_design`が同じ入力に同じエラー（`Debug`の表示が同じ。`NaN`を含むので
/// `==`では比べない）を返す。
fn assert_same_error(
    build: impl Fn() -> OlsInput,
    cov_type: CovType,
    confidence_level: f64,
    label: &str,
) {
    let expected: LeastSquaresError =
        OlsEstimator::fit(build(), cov_type.clone(), confidence_level)
            .map(|_| ())
            .unwrap_err();
    let got = OlsEstimator::fit_releasing_design(build(), cov_type, confidence_level)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(format!("{got:?}"), format!("{expected:?}"), "{label}");
}

#[test]
fn fit_releasing_design_returns_the_same_errors_as_fit() {
    let (y, x) = data(N);

    for level in [0.0, 1.0, f64::NAN] {
        assert_same_error(
            || input(&y, &x),
            CovType::Classical,
            level,
            "confidence level",
        );
    }

    // 観測数が足りない。
    let (short_y, short_x) = data(5);
    assert_same_error(
        || input(&short_y, &short_x),
        CovType::Classical,
        0.95,
        "n <= k",
    );

    // 共線な列（列名つき。QRから復元した設計行列で特定する）。
    let mut collinear = x.clone();
    collinear[3] = collinear[1].iter().map(|v| 2.0 * v).collect();
    assert_same_error(
        || input(&y, &collinear),
        CovType::Classical,
        0.95,
        "collinear",
    );
    let error = OlsEstimator::fit_releasing_design(input(&y, &collinear), CovType::Classical, 0.95)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(error, LeastSquaresError::SingularMatrix { .. }));

    // 完全フィット・定数の被説明変数。
    let exact: Vec<f64> = (0..N)
        .map(|i| 1.0 + x[0][i] - 0.5 * x[1][i] + 0.3 * x[2][i] + 0.2 * x[3][i])
        .collect();
    assert_same_error(
        || input(&exact, &x),
        CovType::Classical,
        0.95,
        "perfect fit",
    );
    assert_same_error(
        || input(&vec![3.0; N], &x),
        CovType::Classical,
        0.95,
        "constant y",
    );
}
