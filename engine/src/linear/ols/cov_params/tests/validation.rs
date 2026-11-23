//! 入力検証・エラー返却のテスト。

use crate::linear::common::LeastSquaresError;
use crate::linear::common::row_time_order;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::cov_type::CovType;
use crate::linear::ols::input::OlsInput;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

#[test]
fn fit_returns_invalid_hac_lags_when_out_of_range() {
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

    let cov_type = CovType::Hac {
        lags: Some(-1),
        time_order: row_time_order(5),
    };
    let result = OlsEstimator::fit(input, cov_type, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::InvalidHacLags { hac_lags: -1, n: 5 })
    );
}

/// `hac_lags`の上限側の境界。`n=5`のとき`lags=5`（`n`自体）は範囲外（`[0, n)`）、
/// `lags=4`（`n-1`）は許容される最大値であることを確認する。
#[test]
fn fit_returns_invalid_hac_lags_when_equal_to_n() {
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

    let cov_type = CovType::Hac {
        lags: Some(5),
        time_order: row_time_order(5),
    };
    let result = OlsEstimator::fit(input, cov_type, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::InvalidHacLags { hac_lags: 5, n: 5 })
    );
}

#[test]
fn fit_accepts_hac_lags_at_upper_boundary_of_n_minus_one() {
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

    let cov_type = CovType::Hac {
        lags: Some(4), // n - 1、許容される最大値
        time_order: row_time_order(5),
    };
    let result = OlsEstimator::fit(input, cov_type, 0.95);

    assert!(result.is_ok());
}

#[test]
fn fit_returns_missing_cluster_column_when_groups_not_provided() {
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

    let result = OlsEstimator::fit(input, CovType::Cluster { groups: None }, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::MissingClusterColumn {
            setting: ClusterSetting::CovType,
        })
    );
}

#[test]
fn fit_returns_insufficient_clusters_when_only_one_group() {
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

    let groups = vec!["a".to_string(); 5];
    let cov_type = CovType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&groups)),
    };
    let result = OlsEstimator::fit(input, cov_type, 0.95);

    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::InsufficientClusters {
            setting: ClusterSetting::CovType,
            g: 1
        })
    );
}

/// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`以下だと、クラスター寄与
/// スコアの総和がゼロ（正規方程式`X'e = 0`）で`rank(Ŝ) ≤ g - 1`のため、ロバスト
/// Wald/F検定の`q×q`部分行列が構造的に特異になる。`g`・`q`は入力だけから判定
/// できるため`fit()`のバリデーションが`CommonError::InsufficientClustersForInference`
/// で弾く。切片＋説明変数3個（`q = 4 - 1 = 3`）に対し`g = 2`
/// （`g < q`）と`g = 3`（`g == q`、`rank(Ŝ) ≤ 2 < 3`で依然特異）の両方を確認する。
/// backstop（`g > q`だが準多重共線性で数値的にほぼ特異）は`cluster.rs`の
/// `fit_with_optional_f_test_does_not_hide_other_cluster_errors`。
#[test]
fn fit_returns_validation_error_when_cluster_count_at_most_slopes() {
    let n = 12;
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
    let x_columns = vec![
        (0..n).map(|i| i as f64).collect::<Vec<f64>>(),
        (0..n).map(|i| (i * i) as f64).collect::<Vec<f64>>(),
        (0..n).map(|i| ((i % 4) + 1) as f64).collect::<Vec<f64>>(),
    ];
    for g in [2_usize, 3_usize] {
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % g)).collect();
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let result = OlsEstimator::fit(
            input,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::InsufficientClustersForInference { g, q: 3 }),
            "g={g}"
        );
    }
}

/// `cov_type=Cluster`かつ`g > q`（`fit()`冒頭のバリデーションを通過）でも、傾き係数間の
/// スケールが極端に異なる（`x1`は1e6倍・`x2`は1e-3倍、`g=6 > q=3`）だけでは失敗しない。
/// ロバストWald/F検定は相関スケール化した部分行列で判定するため、F統計量・p値は単位を
/// 揃えたときと一致する（`fit_f_test_is_invariant_to_a_change_of_units`のcluster版）。
/// 準共線による失敗は`fit_with_optional_f_test_does_not_hide_other_cluster_errors`で固定する。
#[test]
fn fit_cluster_f_test_is_invariant_to_a_change_of_units() {
    let n = 12;
    let unit_x1: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let unit_x2: Vec<f64> = (1..=n).map(|i| (i as f64).powi(2)).collect();
    let x3: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let noise = if i % 2 == 0 { 1e3 } else { -1e3 };
            1.0 + 2.0 * 1e6 * unit_x1[i] + 3.0 * 1e-3 * unit_x2[i] + 0.5 * x3[i] + noise
        })
        .collect();
    let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 6)).collect();
    let fit = |x1_scale: f64, x2_scale: f64| {
        let x1: Vec<f64> = unit_x1.iter().map(|v| v * x1_scale).collect();
        let x2: Vec<f64> = unit_x2.iter().map(|v| v * x2_scale).collect();
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2, x3.clone()],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        OlsEstimator::fit(
            input,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        )
        .unwrap()
    };

    let scaled = fit(1e6, 1e-3);
    let unscaled = fit(1.0, 1.0);

    let (f_scaled, f_unscaled) = (
        scaled.f_statistic().unwrap(),
        unscaled.f_statistic().unwrap(),
    );
    assert!(f_scaled.is_finite() && f_scaled > 0.0);
    assert!((f_scaled / f_unscaled - 1.0).abs() < 1e-9);
    let (p_scaled, p_unscaled) = (scaled.f_p_value().unwrap(), unscaled.f_p_value().unwrap());
    assert!((p_scaled - p_unscaled).abs() <= 1e-9 * p_unscaled);
}

/// コードの行数が`n`と食い違う入力は、短くても長くても`Err`にする（releaseビルドでも）。
/// 短い場合に先頭の行だけで集計した標準誤差を静かに返さないことの確認を含む。
#[test]
fn fit_returns_cluster_dimension_mismatch_when_groups_length_differs_from_nobs() {
    for groups_rows in [2usize, 7] {
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

        let labels: Vec<String> = (0..groups_rows).map(|i| format!("g{}", i % 3)).collect();
        let cov_type = CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labels)),
        };
        assert_eq!(
            OlsEstimator::fit(input, cov_type, 0.95).unwrap_err(),
            LeastSquaresError::Common(CommonError::ClusterDimensionMismatch { groups_rows, n: 5 }),
            "groups_rows={groups_rows}"
        );
    }
}
