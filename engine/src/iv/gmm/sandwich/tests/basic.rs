//! getter・退化ケース・整合性のテスト。

use crate::iv::common::IvError;
use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::WeightType;
use crate::iv::gmm::test_support::*;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

/// `cov_type`と`weight_type`が同じ`fit()`で別々のグルーピングを使うとき、それぞれが
/// 自分のコードで計算される（片方のコードをもう片方に流用していない）ことの確認。
/// 点推定は`cov_type`に依存しないので`cov_type=Classical`のときとビット一致し、標準誤差は
/// 分割を入れ替えると変わる。ラベルだけ付け替えた同じ分割では（ラベルの辞書順が変わり
/// 加算順が変わるため丸め誤差の範囲で）一致する。
#[test]
fn fit_uses_separate_codes_for_cov_type_and_weight_type_clusters() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let weight_groups = labelled_groups(n, 4, "a");
    let cov_groups = labelled_groups(n, 5, "b");
    let fit = |weight: &[String], cov: Option<&[String]>| {
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
        GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(weight)),
                },
            },
            true,
            match cov {
                Some(groups) => CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(groups)),
                },
                None => CovType::Classical,
            },
            0.95,
        )
        .unwrap()
    };

    let mixed = fit(&weight_groups, Some(&cov_groups));
    let classical_cov = fit(&weight_groups, None);
    let swapped = fit(&cov_groups, Some(&weight_groups));
    let same = fit(&weight_groups, Some(&weight_groups));
    let relabelled: Vec<String> = cov_groups.iter().map(|g| format!("z{g}")).collect();
    let mixed_relabelled = fit(&weight_groups, Some(&relabelled));

    for j in 0..mixed.params().nrows() {
        assert_eq!(
            mixed.params().get(j, 0).to_bits(),
            classical_cov.params().get(j, 0).to_bits()
        );
        let se = *mixed.std_errors().get(j, 0);
        assert!(
            (se - *mixed_relabelled.std_errors().get(j, 0)).abs() < 1e-10 * se,
            "relabelling the cov_type clusters must not change the standard error"
        );
        assert_ne!(
            se.to_bits(),
            same.std_errors().get(j, 0).to_bits(),
            "cov_type clusters must not be replaced by the weight_type clusters"
        );
        assert_ne!(
            se.to_bits(),
            swapped.std_errors().get(j, 0).to_bits(),
            "swapping the two groupings must change the standard error"
        );
    }
}

/// 説明変数が定数項のみ（`x_exog=[]`・`x_endog=[]`、傾き係数`df_model=0`）のとき、
/// F統計量は0除算を避けてNaNになる（`two_sls/fit/tests/basic.rs`の
/// `fit_sets_f_statistic_and_f_p_value_to_nan_for_const_only_model`と同じ判断）。
#[test]
fn fit_sets_f_statistic_and_f_p_value_to_nan_for_const_only_model() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let z = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &[],
        vec![],
        std::slice::from_ref(&z),
        vec!["z1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();
    assert_eq!(estimator.df_model(), 0);
    assert!(estimator.wald_statistic().is_nan());
    assert!(estimator.wald_p_value().is_nan());
}

/// `cov_type`対応で追加した各getterが期待通りの次元・値を返すことを確認する
/// （`fit_exposes_basic_metadata_getters`と対になる、こちらは`cov_type`/SE系）。
#[test]
fn fit_exposes_cov_type_inference_getters() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
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

    assert_eq!(estimator.cov_type(), &CovType::Hc1);
    assert_eq!(estimator.std_errors().nrows(), 2);
    assert_eq!(estimator.test_stats().nrows(), 2);
    assert_eq!(estimator.p_values().nrows(), 2);
    assert_eq!(estimator.conf_lower().nrows(), 2);
    assert_eq!(estimator.conf_upper().nrows(), 2);
    assert_eq!(estimator.df_resid(), n - 2);
    assert_eq!(estimator.df_model(), 1);
    assert!(estimator.r_squared() <= 1.0);
    assert!(estimator.wald_statistic() >= 0.0);
    assert!((0.0..=1.0).contains(&estimator.wald_p_value()));

    for j in 0..2 {
        let coef = *estimator.params().get(j, 0);
        let se = *estimator.std_errors().get(j, 0);
        let expected_z = coef / se;
        assert!((estimator.test_stats().get(j, 0) - expected_z).abs() < 1e-10);
        assert!(*estimator.conf_lower().get(j, 0) < *estimator.conf_upper().get(j, 0));
    }
}

/// `OneStep`は`weight_type`を持たないため（`weight_cluster_codes`は
/// `None`）、`cov_type=Cluster`のグループは`validate_iv_fit_inputs`の構造方程式のクラスター数の検証
/// だけで検証され、`cov_type`から借りたコードで集計される。この経路が、点推定は`cov_type`に依らず、
/// 標準誤差はクラスター割り当てだけに依存し（ラベルの付け替えで変わらない）、
/// 欠落・長さ不一致のコードは`Err`になることを確認する。
#[test]
fn fit_with_one_step_gmm_uses_the_cov_type_cluster_groups_only() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let groups = labelled_groups(n, 5, "c");
    let fit = |cov_type: CovType| {
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
        GmmEstimator::fit(input, GmmType::OneStep, true, cov_type, 0.95)
    };
    let cluster = |labels: &[String]| CovType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(labels)),
    };

    let classical = fit(CovType::Classical).unwrap();
    let clustered = fit(cluster(&groups)).unwrap();
    let relabelled_groups: Vec<String> = groups.iter().map(|g| format!("z{g}")).collect();
    let relabelled = fit(cluster(&relabelled_groups)).unwrap();

    for j in 0..clustered.params().nrows() {
        // 点推定は`cov_type`に依存しない。
        assert_eq!(
            clustered.params().get(j, 0).to_bits(),
            classical.params().get(j, 0).to_bits()
        );
        let se = *clustered.std_errors().get(j, 0);
        assert_ne!(se.to_bits(), classical.std_errors().get(j, 0).to_bits());
        // ラベルの付け替え（コードの順序は同じ）では標準誤差は変わらない。
        assert!((se - *relabelled.std_errors().get(j, 0)).abs() < 1e-10 * se);
    }

    assert_eq!(
        fit(CovType::Cluster { groups: None }).unwrap_err(),
        IvError::Common(CommonError::MissingClusterColumn {
            setting: ClusterSetting::CovType,
        })
    );
    assert_eq!(
        fit(cluster(&groups[..n - 1])).unwrap_err(),
        IvError::Common(CommonError::ClusterDimensionMismatch {
            groups_rows: n - 1,
            n
        })
    );
}

/// 全`cov_type`で、`cov_params()`が対称で、対角の平方根が`std_errors()`と一致する
/// （`std_errors`は`cov_params`の対角から作るため、公開している行列が標準誤差の元そのもの
/// であることの確認）。`bread⁻¹ · meat · bread⁻¹`の形は丸め誤差で僅かに非対称になりうるため、
/// 許容は両係数の分散から決まる尺度（`sqrt(V_ii·V_jj)`）の`1e-12`倍。
#[test]
fn cov_params_is_symmetric_and_diagonal_matches_std_errors_for_every_cov_type() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let labels: Vec<String> = (0..16).map(|i| format!("g{}", i / 4)).collect();

    let cov_types = [
        CovType::Classical,
        CovType::Hc0,
        CovType::Hc1,
        CovType::Hc2,
        CovType::Hc3,
        CovType::Hac {
            lags: Some(2),
            time_order: row_time_order(16),
        },
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labels)),
        },
    ];
    for cov_type in cov_types {
        let name = format!("{cov_type:?}");
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
            cov_type,
            0.95,
        )
        .unwrap();

        let cov = estimator.cov_params();
        let k = estimator.params().nrows();
        assert_eq!((cov.nrows(), cov.ncols()), (k, k), "{name}: shape");
        for i in 0..k {
            for j in 0..k {
                let scale = (*cov.get(i, i) * *cov.get(j, j)).sqrt();
                assert!(
                    (*cov.get(i, j) - *cov.get(j, i)).abs() <= 1e-12 * scale,
                    "{name}: cov_params not symmetric at ({i},{j})"
                );
            }
            let se = *estimator.std_errors().get(i, 0);
            assert!(
                (cov.get(i, i).sqrt() - se).abs() <= 1e-12 * se,
                "{name}: sqrt(cov_params[{i}][{i}]) != std_errors[{i}]"
            );
        }
    }
}
