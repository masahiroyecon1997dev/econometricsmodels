//! クラスター共分散・`fit_with_optional_f_test`（クラスター数が傾き以下のときの全体F検定）のテスト。

use crate::linear::common::LeastSquaresError;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::cov_type::CovType;
use crate::linear::ols::input::OlsInput;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::error::MatrixKind;
use crate::shared::group_codes::GroupCodes;

/// 切片＋説明変数3個（`q = 3`）の12観測。`fit_with_optional_f_test`の境界テスト用。
fn cluster_boundary_input() -> OlsInput {
    let n = 12;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            // クラスター（`i % g`）と周期が揃わない不規則なノイズ
            let noise = [
                0.3, -0.2, 0.1, -0.4, 0.25, -0.15, 0.05, 0.35, -0.3, 0.2, -0.1, 0.4,
            ];
            1.0 + 0.5 * i as f64 + noise[i]
        })
        .collect();
    let x_columns = vec![
        (0..n).map(|i| i as f64).collect::<Vec<f64>>(),
        (0..n).map(|i| (i * i) as f64).collect::<Vec<f64>>(),
        (0..n).map(|i| ((i % 4) + 1) as f64).collect::<Vec<f64>>(),
    ];
    OlsInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

fn cluster_cov_type(n_groups: usize) -> CovType {
    let groups: Vec<String> = (0..12).map(|i| format!("g{}", i % n_groups)).collect();
    CovType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&groups)),
    }
}

/// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）を2クラスター（groups=["a","a","b","b","b"]）
/// でのクラスターロバスト標準誤差。期待値はstatsmodels 0.14.6で独立に計算・検算済み
/// （`sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)`。
/// 小標本補正はstatsmodelsの既定`use_correction=True`のまま、明示指定はしていない）。
#[test]
fn fit_computes_cluster_std_errors_test_stats_p_values_conf_int_and_f_test() {
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

    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
    ];
    let cov_type = CovType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&groups)),
    };
    let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

    let se = estimator.std_errors();
    assert!((*se.get(0, 0) - 0.785_196_366_097_886_8).abs() < 1e-6);
    assert!((*se.get(1, 0) - 0.230_940_107_675_849_05).abs() < 1e-6);

    let t = estimator.test_stats();
    assert!((*t.get(0, 0) - 2.801_846_894_596_724_5).abs() < 1e-6);
    assert!((*t.get(1, 0) - 2.598_076_211_353_332).abs() < 1e-6);

    let p = estimator.p_values();
    assert!((*p.get(0, 0) - 0.218_242_895_017_685_43).abs() < 1e-6);
    assert!((*p.get(1, 0) - 0.233_908_049_281_92).abs() < 1e-6);

    let lower = estimator.conf_lower();
    let upper = estimator.conf_upper();
    assert!((*lower.get(0, 0) - (-7.776_865_785_740_13)).abs() < 1e-6);
    assert!((*upper.get(0, 0) - 12.176_865_785_740_125).abs() < 1e-6);
    assert!((*lower.get(1, 0) - (-2.334_372_289_923_566_6)).abs() < 1e-6);
    assert!((*upper.get(1, 0) - 3.534_372_289_923_567_7).abs() < 1e-6);

    assert!((estimator.f_statistic().unwrap() - 6.750_000_000_000_083_5).abs() < 1e-6);
    assert!((estimator.f_p_value().unwrap() - 0.233_908_049_281_92).abs() < 1e-6);
}

/// クラスターロバスト標準誤差は、同じ入力に対して`fit()`を複数回呼んでも
/// ビット単位で同じ結果になること（`cluster_cov_params`内部の集約が
/// `HashMap`の反復順序に依存し、実行のたびに浮動小数点の丸め誤差レベルで
/// 結果がぶれていた回帰。`BTreeMap`化で修正した）。
#[test]
fn fit_cluster_std_errors_are_deterministic_across_repeated_fits() {
    let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 3.0, 6.0, 1.0, 7.0, 2.5];
    let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
    let groups: Vec<String> = (0..10).map(|i| format!("g{}", i % 4)).collect();

    let build = || {
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let cov_type = CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        };
        OlsEstimator::fit(input, cov_type, 0.95).unwrap()
    };

    let first = build();
    for _ in 0..20 {
        let repeat = build();
        assert_eq!(
            *repeat.std_errors().get(0, 0),
            *first.std_errors().get(0, 0)
        );
        assert_eq!(
            *repeat.std_errors().get(1, 0),
            *first.std_errors().get(1, 0)
        );
    }
}

/// `fit_with_optional_f_test`は`g <= q`（`g == q`も構造的に特異）でも失敗せず、全体F検定
/// （統計量・p値・自由度）だけが`None`になる。係数・標準誤差・信頼区間は`g - 1`自由度で
/// 有限に求まる（`fit`は同じ入力を`InsufficientClustersForInference`で拒否する）。
#[test]
fn fit_with_optional_f_test_leaves_f_test_undefined_when_cluster_count_at_most_slopes() {
    for g in [2_usize, 3_usize] {
        let estimator = OlsEstimator::fit_with_optional_f_test(
            cluster_boundary_input(),
            cluster_cov_type(g),
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.f_statistic(), None, "g={g}");
        assert_eq!(estimator.f_p_value(), None, "g={g}");
        assert_eq!(estimator.f_df(), None, "g={g}");
        assert_eq!(estimator.df_inference(), g - 1, "g={g}");
        for j in 0..estimator.input().k() {
            let se = *estimator.std_errors().get(j, 0);
            assert!(se.is_finite() && se > 0.0, "g={g} j={j} se={se}");
            assert!(estimator.p_values().get(j, 0).is_finite(), "g={g} j={j}");
        }
    }
}

/// `g == q + 1`では全体F検定が定義でき、`fit`と同じ値を返す（境界の反対側）。
#[test]
fn fit_with_optional_f_test_matches_fit_when_cluster_count_exceeds_slopes() {
    let optional =
        OlsEstimator::fit_with_optional_f_test(cluster_boundary_input(), cluster_cov_type(4), 0.95)
            .unwrap();
    let strict = OlsEstimator::fit(cluster_boundary_input(), cluster_cov_type(4), 0.95).unwrap();
    assert!(optional.f_statistic().is_some());
    assert_eq!(optional.f_statistic(), strict.f_statistic());
    assert_eq!(optional.f_p_value(), strict.f_p_value());
    assert_eq!(optional.f_df(), Some((3, 3)));
    assert_eq!(optional.f_df(), strict.f_df());
}

/// クラスター数の下限（`g >= 2`）は`fit_with_optional_f_test`でも検証される。
#[test]
fn fit_with_optional_f_test_still_rejects_a_single_cluster() {
    let result =
        OlsEstimator::fit_with_optional_f_test(cluster_boundary_input(), cluster_cov_type(1), 0.95);
    assert_eq!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::InsufficientClusters {
            setting: ClusterSetting::CovType,
            g: 1
        })
    );
}

/// 切片なし（`q = k`、`k_constant = 0`）でも境界は同じ形になる。`x`3列・`g`は`q = 3`以下で
/// `None`、`g = 4`で定義あり。
#[test]
fn fit_with_optional_f_test_handles_the_boundary_without_intercept() {
    let n = 12;
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
    let x_columns = vec![
        (0..n).map(|i| i as f64).collect::<Vec<f64>>(),
        (0..n).map(|i| (i * i) as f64).collect::<Vec<f64>>(),
        (0..n).map(|i| ((i % 4) + 1) as f64).collect::<Vec<f64>>(),
    ];
    let fit = |g: usize| {
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();
        OlsEstimator::fit_with_optional_f_test(input, cluster_cov_type(g), 0.95).unwrap()
    };
    for g in [2_usize, 3_usize] {
        let estimator = fit(g);
        assert_eq!(estimator.f_statistic(), None, "g={g}");
        assert_eq!(estimator.f_df(), None, "g={g}");
    }
    let estimator = fit(4);
    assert!(estimator.f_statistic().is_some());
    assert_eq!(estimator.f_df(), Some((3, 3)));
}

/// `fit_with_optional_f_test`が`None`にするのは`g <= q`だけで、他の失敗は`fit`と同じ
/// まま伝播する: `groups`未指定（`MissingClusterColumn`）、`groups`の長さが`n`と不一致
/// （`ClusterDimensionMismatch`）、`g > q`でのほぼ特異性（`NearSingularMatrix`）。
#[test]
fn fit_with_optional_f_test_does_not_hide_other_cluster_errors() {
    let missing = OlsEstimator::fit_with_optional_f_test(
        cluster_boundary_input(),
        CovType::Cluster { groups: None },
        0.95,
    );
    assert_eq!(
        missing.unwrap_err(),
        LeastSquaresError::Common(CommonError::MissingClusterColumn {
            setting: ClusterSetting::CovType,
        })
    );

    let short: Vec<String> = (0..7).map(|i| format!("g{}", i % 4)).collect();
    let mismatch = OlsEstimator::fit_with_optional_f_test(
        cluster_boundary_input(),
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&short)),
        },
        0.95,
    );
    assert!(matches!(
        mismatch.unwrap_err(),
        LeastSquaresError::Common(CommonError::ClusterDimensionMismatch { .. })
    ));

    // 2列目が1列目とほぼ同じ（説明変数の相関が`1 - 1e-13`程度）。単位を揃えても、クラスター
    // ロバストのサンドイッチ共分散`B M B`は`B = (X'X)⁻¹`の条件数（約1e12）が桁落ちを起こして
    // 数値的に正定値でなくなり（傾き係数の相関が1を超える）、Wald検定が止める。これは
    // サンドイッチ共分散の既知の限界に依存している: 共分散の計算経路を改善してこの入力が
    // 通るようになったら、このテストの入力（より強い準共線）を取り直すこと。
    let n = 40;
    let x1: Vec<f64> = (0..n).map(|i| (((i * 7) % 13) as f64) - 6.0).collect();
    let x2: Vec<f64> = (0..n)
        .map(|i| x1[i] + 1e-6 * ((((i * 5) % 11) as f64) - 5.0))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let noise = if i % 2 == 0 { 0.7 } else { -0.4 } + 0.1 * ((i % 5) as f64);
            1.0 + 2.0 * x1[i] + 3.0 * x2[i] + noise
        })
        .collect();
    let ill_conditioned = OlsInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 8)).collect();
    let result = OlsEstimator::fit_with_optional_f_test(
        ill_conditioned,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    );
    assert!(matches!(
        result.unwrap_err(),
        LeastSquaresError::Common(CommonError::NearSingularMatrix {
            matrix: MatrixKind::CoefficientCovariance
        })
    ));
}

/// クラスター以外の`cov_type`では`fit`と全く同じ結果になる。
#[test]
fn fit_with_optional_f_test_matches_fit_for_non_cluster_cov_types() {
    let optional =
        OlsEstimator::fit_with_optional_f_test(cluster_boundary_input(), CovType::Hc1, 0.95)
            .unwrap();
    let strict = OlsEstimator::fit(cluster_boundary_input(), CovType::Hc1, 0.95).unwrap();
    assert_eq!(optional.f_statistic(), strict.f_statistic());
    assert_eq!(optional.f_p_value(), strict.f_p_value());
    assert_eq!(optional.f_df(), strict.f_df());
}
