//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use super::dk_bandwidth_zero;
use super::dk_time_from;
use super::three_period_input;
use super::three_period_time;
use super::two_entity_input;
use super::two_period_input;
use crate::linear::common::LeastSquaresError;
use crate::panel::common::ColumnRole;
use crate::panel::common::PanelDimension;
use crate::panel::common::PanelError;
use crate::panel::common::TimeKeys;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::input::FeCovType;
use crate::panel::fe::input::FeEffects;
use crate::panel::fe::input::FeInput;
use crate::panel::fe::test_support::*;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::error_catalog::Stage;
use crate::shared::group_codes::GroupCodes;

/// 辞書順の`TimeKeys`（`FeCovType::Dk.time`に渡す）。
fn lex_time(values: &[&str]) -> TimeKeys {
    TimeKeys::lexicographic(strings(values))
}

/// 4エンティティ×2観測・`k=1`の1-way FE入力で、入力の`time`列を`time`で与える
/// （`Dk.time`上書きの経路を検証するため、時点ラベルの分け方だけを変えられるようにする）。
fn four_by_two_input(time: &[&str]) -> FeInput {
    let entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
    let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0];
    let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0];
    FeInput::from_columns(
        &y,
        &[x],
        vec!["x".to_string()],
        &entity,
        Some(&strings(time)),
        "y".into(),
    )
    .unwrap()
}

/// 3時点に分かれるが各エンティティ2観測のラベル（`t_periods=3`、退化しない）。
const THREE_PERIOD_LABELS: [&str; 8] = ["1", "2", "2", "3", "3", "1", "1", "2"];

const TWO_PERIOD_LABELS: [&str; 8] = ["1", "2", "1", "2", "1", "2", "1", "2"];

#[test]
fn fe_estimator_fit_returns_perfect_fit_for_exact_data() {
    // ノイズなしでは、within回帰の残差が全て0で標準誤差とF検定が定義できない。
    let entity = strings(&["a", "a", "a", "b", "b", "b"]);
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y: Vec<f64> = x1
        .iter()
        .zip(&entity)
        .map(|(x, e)| 2.0 * x + if e == "a" { 5.0 } else { 10.0 })
        .collect();
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::WithinRegressionFailed {
            source: CommonError::PerfectFit.into(),
        }
    );
}

#[test]
fn fe_estimator_fit_returns_outcome_constant_within_entity_for_an_entity_level_outcome() {
    // yが個体内で定数（個体間でだけ違う）なら、固定効果が全てを説明してwithin変換後のyは0になる。
    // 0.1のように3回足して3で割ると丸め誤差が残る値も、相対閾値で定数とみなす。
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
    let y = [0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 1.3, 1.3, 1.3];
    let x1 = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

    assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
}

#[test]
fn fe_estimator_fit_returns_outcome_constant_within_entity_when_effects_explain_the_outcome() {
    // 2-wayでは、y = 個体効果 + 時点効果 もwithin変換後に0になる。
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
    let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
    let alpha = [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, -2.0, -2.0, -2.0];
    let gamma = [0.0, 0.5, 1.5, 0.0, 0.5, 1.5, 0.0, 0.5, 1.5];
    let y: Vec<f64> = alpha.iter().zip(gamma).map(|(a, g)| a + g).collect();
    let x1 = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
    let input = FeInput::from_columns(
        &y,
        &[x1],
        vec!["x1".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

    assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
}

#[test]
fn fe_estimator_fit_propagates_singleton_error() {
    let entity = strings(&["a", "a", "c"]);
    let y = [1.0, 2.0, 3.0];
    let x1 = vec![1.0, 2.0, 3.0];
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::SingletonGroup {
            dimension: PanelDimension::ENTITY,
            column: "entity".to_string(),
            group_id: "c".to_string(),
            n_dropped_rows: 1,
        }
    );
}

#[test]
fn fe_estimator_fit_propagates_zero_variance_error() {
    // "female"は各エンティティ内で一定（時間不変）。n=6・n_entities=2・k=2で
    // df_model=4・n>df_modelとなるよう、各エンティティ3観測に拡張する
    // （n=4だとdf_model=2+2=4でn<=df_modelとなりInsufficientDegreesOfFreedomが
    // 先に発火してしまうため、自由度検証を踏まえたサイズにする）。
    let entity = strings(&["a", "a", "a", "b", "b", "b"]);
    let y = [1.0, 2.0, 3.0, 5.0, 6.0, 8.0];
    let x_varying = vec![10.0, 20.0, 15.0, 5.0, 10.0, 20.0];
    let female = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let input = FeInput::from_columns(
        &y,
        &[x_varying, female],
        vec!["x_varying".to_string(), "female".to_string()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::ZeroVarianceAfterDemeaning {
            role: ColumnRole::X,
            column: "female".to_string(),
        }
    );
}

#[test]
fn fe_estimator_fit_two_way_requires_time() {
    let entity = strings(&["a", "a", "b", "b"]);
    let y = [1.0, 2.0, 3.0, 4.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

    assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
}

#[test]
fn fe_estimator_fit_two_way_without_time_reports_requires_time_before_any_other_error() {
    // 2-wayで`time`が無いことは、他のどの欠陥（entityのsingleton・不正な
    // `confidence_level`・cov_type固有の検証）よりも先に報告される。優先順位は
    // `fit()`冒頭の`absorbed_levels`が守る。
    let entity = strings(&["a", "a", "c"]);
    let y = [1.0, 2.0, 3.0];
    let covs = || {
        vec![
            FeCovType::Classical,
            FeCovType::Hc2,
            FeCovType::Cluster { groups: None },
            FeCovType::Dk {
                bandwidth: Some(100),
                time: lex_time(&["1", "2", "3"]),
            },
        ]
    };
    for cov_type in covs() {
        let input = FeInput::from_columns(
            &y,
            &[vec![1.0, 2.0, 4.0]],
            vec!["x1".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, cov_type.clone(), 1.5);

        assert_eq!(
            result.unwrap_err(),
            PanelError::TwoWayRequiresTime,
            "cov_type = {cov_type:?}"
        );
    }
}

#[test]
fn fe_estimator_fit_two_way_propagates_singleton_error() {
    // entity "c"は時点"1"のみの1観測（singleton）。`fit()`が
    // `validate_no_singleton_groups_two_way`（entity/time双方を対称にチェックする方）を
    // 正しく呼んでいることの配線確認（1-way用の検証関数を誤って呼んでいないか）。
    let entity = strings(&["a", "a", "b", "b", "c"]);
    let time = strings(&["1", "2", "1", "2", "1"]);
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = FeInput::from_columns(
        &y,
        &[x1],
        vec!["x1".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::SingletonGroup {
            dimension: PanelDimension::ENTITY,
            column: "entity".to_string(),
            group_id: "c".to_string(),
            n_dropped_rows: 1,
        }
    );
}

#[test]
fn fe_estimator_fit_two_way_propagates_zero_variance_error() {
    // "year_dummy"はエンティティ間で変動しない（time FEと完全共線）。`fit()`が
    // within変換後に`validate_no_zero_variance_regressors`を呼んでいることの配線確認。
    // n=9（3エンティティ×3時点、n_entities=3・n_periods=3）に拡張し、
    // df_model=k(2)+neffects(3+3-1=5)=7・n>df_modelとなるサイズにする
    // （自由度検証を先に通過させるため）。"x_varying"は
    // entity×timeの交互作用項（entity_idx*time_idx）にして、加法分離可能な
    // 主効果のみの列（2-way demeanで機械的にゼロになる）にならないようにする
    // （Pythonで2-way demeanした結果、分散0.444...とゼロでないことを確認済み）。
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
    let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let x_varying = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
    let year_dummy = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
    let input = FeInput::from_columns(
        &y,
        &[x_varying, year_dummy],
        vec!["x_varying".to_string(), "year_dummy".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::ZeroVarianceAfterDemeaning {
            role: ColumnRole::X,
            column: "year_dummy".to_string(),
        }
    );
}

#[test]
fn fe_estimator_fit_propagates_invalid_confidence_level_error() {
    // `confidence_level`の範囲チェック
    // （`(0, 1)`の範囲外）が、`PanelError::WithinRegressionFailed`として正しく
    // 伝播することを確認する（`iv::two_sls`の同型テストに倣う）。
    let entity = strings(&["a", "a", "b", "b"]);
    let y = [1.0, 2.0, 3.0, 4.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 1.5);

    assert_eq!(
        result.unwrap_err(),
        PanelError::WithinRegressionFailed {
            source: LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5,
            }),
        }
    );
}

#[test]
fn fe_estimator_fit_wraps_ols_failure_as_within_regression_failed() {
    // 自由度検証（`n <= df_model`）が最小二乗側の`n <= k`相当の条件より常に厳しい
    // （`df_model = k + neffects > k`）ため、最小二乗側の観測数不足はこの経路では
    // 発生しえない（`fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error`が
    // 先に弾く）。そのため、ここでは最小二乗固有の別の失敗——完全な多重共線性（`LeastSquaresError::
    // SingularMatrix`）——を踏ませる: x2はx1のちょうど2倍で、within変換
    // （線形変換）後も比例関係`x2_demeaned = 2 * x1_demeaned`が保たれ完全共線になる。
    // n=6（3エンティティ、singletonではない）・k=2でdf_model=2+3=5、n=6>5と
    // 自由度検証は通過するようにする。
    let entity = strings(&["a", "a", "b", "b", "c", "c"]);
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x1 = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
    let x2: Vec<f64> = x1.iter().map(|v| 2.0 * v).collect();
    let input = FeInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

    assert!(matches!(
        result.unwrap_err(),
        PanelError::WithinRegressionFailed { .. }
    ));
}

#[test]
fn fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error() {
    // n=4・n_entities=2・k=2でdf_model=k+n_entities=4となり、n<=df_modelのため
    // 最小二乗へ進む前に`PanelError::InsufficientDegreesOfFreedom`で
    // 弾かれることを確認する。
    let entity = strings(&["a", "a", "b", "b"]);
    let y = [1.0, 2.0, 3.0, 4.0];
    let x1 = vec![1.0, 3.0, 2.0, 6.0];
    let x2 = vec![2.0, 5.0, 1.0, 9.0];
    let input = FeInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::InsufficientDegreesOfFreedom {
            n_obs: 4,
            n_groups: vec![2],
            k: 2,
        }
    );
}

/// エラーの`min_obs`が`fit`の実際の境界と一致する: 拒否される最大の`n`（`n = df_model`）の
/// 次の`n = min_obs`で成功する。1-wayの`n_effects`は個体数（ここでは2）。
#[test]
fn insufficient_degrees_of_freedom_min_obs_matches_the_fit_boundary() {
    use crate::shared::error::ErrorInfo;
    use crate::shared::error_catalog::ParamValue;

    let build = |n: usize, k: usize| {
        let entity: Vec<String> = (0..n)
            .map(|i| if i < n.div_ceil(2) { "a" } else { "b" }.to_string())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.9).sin() + 0.1 * i as f64)
            .collect();
        let x: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                (0..n)
                    .map(|i| ((i * (j + 2) + j) as f64 * 1.3).cos())
                    .collect()
            })
            .collect();
        let names = (0..k).map(|j| format!("x{j}")).collect();
        FeInput::from_columns(&y, &x, names, &entity, None, "y".into()).unwrap()
    };
    for k in 2..=4_usize {
        let n_effects = 2;
        let rejected_n = n_effects + k;
        let err = FeEstimator::fit(
            build(rejected_n, k),
            FeEffects::OneWay,
            FeCovType::Classical,
            0.95,
        )
        .unwrap_err();
        let min_obs = err
            .report()
            .all_params()
            .into_iter()
            .find_map(|(name, value)| match (name.as_str(), value) {
                ("min_obs", ParamValue::Int(v)) => Some(v as usize),
                _ => None,
            })
            .expect("min_obs");
        assert_eq!(min_obs, rejected_n + 1, "k={k}");
        assert!(
            FeEstimator::fit(
                build(min_obs, k),
                FeEffects::OneWay,
                FeCovType::Classical,
                0.95
            )
            .is_ok(),
            "k={k}"
        );
    }
}

#[test]
fn fe_estimator_fit_f_test_is_invariant_to_a_change_of_units() {
    // x1は1e6倍、x2は1e-3倍とスケールが極端に異なる（x3は通常スケール）。within変換後も3列は
    // 線形従属ではないため設計行列自体はフルランク。傾き係数の分散の比は1e18を超えるが、FE自身の
    // F検定（`wald_f_test`）は相関スケール化した部分行列で判定するため、F統計量・p値は
    // 単位を揃えたときと一致する。
    let n = 20;
    let entity: Vec<String> = (0..n).map(|i| format!("e{}", i % 5)).collect();
    let unit_x1: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
    let unit_x2: Vec<f64> = (0..n).map(|i| (((i * 5) % 13) as f64).powi(2)).collect();
    let x3: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let noise = if i % 2 == 0 { 1e3 } else { -1e3 };
            2.0 * 1e6 * unit_x1[i] + 3.0 * 1e-3 * unit_x2[i] + 0.5 * x3[i] + noise
        })
        .collect();
    let fit = |x1_scale: f64, x2_scale: f64| {
        let x1: Vec<f64> = unit_x1.iter().map(|v| v * x1_scale).collect();
        let x2: Vec<f64> = unit_x2.iter().map(|v| v * x2_scale).collect();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2, x3.clone()],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();
        FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap()
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

#[test]
fn fe_estimator_fit_cluster_propagates_insufficient_clusters_error() {
    // クラスター数2未満は`CommonError::InsufficientClusters`（`validate_cluster_groups`）
    // として伝播する。すべて同じクラスターに属する（g=1）データを使う。
    let entity = strings(&["a", "a", "b", "b", "c", "c"]);
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
    let single_cluster = strings(&["g", "g", "g", "g", "g", "g"]);
    let input =
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();

    let result = FeEstimator::fit(
        input,
        FeEffects::OneWay,
        FeCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&single_cluster)),
        },
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        PanelError::Common(CommonError::InsufficientClusters {
            setting: ClusterSetting::CovType,
            g: 1
        })
    );
}

#[test]
fn fe_estimator_fit_cluster_propagates_insufficient_clusters_for_inference_error() {
    // クラスター数g(=2)が傾き係数の数q(=k=2)以下（g<=q、境界は厳密不等号）は
    // `CommonError::InsufficientClustersForInference`として伝播する
    // （`validate_cluster_count_covers_slopes`、`.claude/rules/testing-policy.md`
    // 「G<=qで書く」の方針）。g=2自体は`InsufficientClusters`（g<2）は回避している。
    let entity = strings(&["a", "a", "b", "b", "c", "c"]);
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x1 = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
    let x2 = vec![2.0, 5.0, 1.0, 9.0, 3.0, 7.0];
    let two_clusters = strings(&["1", "1", "1", "2", "2", "2"]);
    let input = FeInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(
        input,
        FeEffects::OneWay,
        FeCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&two_clusters)),
        },
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        PanelError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
    );
}

#[test]
fn fe_estimator_fit_one_way_hac_with_single_time_period_is_rejected() {
    // `t_periods=1`（全観測が同一の`time`ラベル）という退化した境界ケース
    // （`resolve_dk_bandwidth`のNone分岐が`bandwidth=t_periods`を
    // 返しうる唯一のケース）。
    //
    // DKの小標本補正を`(t_periods/(t_periods-1))×((n-1)/(n-K))`
    // （fixestの`ssc()`、`t_periods`をclusterの`G`と同じ役割で使う）に変更した結果、
    // `t_periods=1`は`t_periods/(t_periods-1)=1/0`が発散し計算が成立しなくなった
    // （クラスターの`G=1`が`validate_cluster_groups`で拒否されるのと同じ理由）。
    // `resolve_dk_bandwidth`が`PanelError::InsufficientDkPeriods`で早期に拒否する
    // （`G/(G-1)`型の補正は`t_periods=1`で未定義になる）。
    let entity = strings(&["a", "a", "b", "b", "c", "c"]);
    let time = strings(&["1", "1", "1", "1", "1", "1"]);
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
    let input = FeInput::from_columns(
        &y,
        &[x],
        vec!["x".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap();

    let dk_time = dk_time_from(&input);
    let result = FeEstimator::fit(
        input,
        FeEffects::OneWay,
        FeCovType::Dk {
            bandwidth: None,
            time: dk_time,
        },
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        PanelError::InsufficientDkPeriods { t_periods: 1 }
    );
}

#[test]
fn fe_estimator_fit_hac_rejects_periods_not_covering_slopes() {
    // `t_periods=3`ではDK共分散のrankが`t-1=2`以下で、F検定の傾き`k=3`個の部分行列が
    // 構造的に特異になる。`wald_f_test`の数値的な特異性判定を待たず弾く。
    let time = three_period_time();
    let dk_time = TimeKeys::lexicographic(time.clone());
    let result = FeEstimator::fit(
        three_period_input(3, Some(&time)),
        FeEffects::OneWay,
        FeCovType::Dk {
            bandwidth: Some(0),
            time: dk_time,
        },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
            .in_stage(Stage::OverallFTest)
    );
}

#[test]
fn fe_estimator_fit_hac_counts_periods_from_time_override() {
    // `FeCovType::Dk.time`の上書きがあれば、時点数はその列で数える（入力の`time`が
    // `None`の1-way FEでも同じ判定になる）。
    let result = FeEstimator::fit(
        three_period_input(3, None),
        FeEffects::OneWay,
        FeCovType::Dk {
            bandwidth: Some(0),
            time: TimeKeys::lexicographic(three_period_time()),
        },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
            .in_stage(Stage::OverallFTest)
    );
}

#[test]
fn fe_estimator_fit_hac_rejects_degenerate_two_period_panel() {
    // 全エンティティが2時点に1観測ずつだと、within変換で時点スコアが`h_1 = h_2 = 0`に
    // 退化しDK共分散が恒等的にゼロになる（`k=1`でも`t > q`の検証は通ってしまう）。
    let (input, _) = two_period_input(1, false);
    let dk_cov = dk_bandwidth_zero(&input);
    let result = FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95);
    assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
}

#[test]
fn fe_estimator_fit_hac_rejects_degenerate_two_period_panel_two_way() {
    // 2-way（バランスパネルの二重デミーニング）でもエンティティ内の和がゼロになり同じ退化。
    let (input, _) = two_period_input(1, false);
    let dk_cov = dk_bandwidth_zero(&input);
    let result = FeEstimator::fit(input, FeEffects::TwoWay, dk_cov, 0.95);
    assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
}

#[test]
fn fe_estimator_fit_cluster_rejects_degenerate_two_group_split() {
    // 2時点のパネルを`time`でクラスタリング（`G=2 > k=1`の検証は通る）すると、
    // DKと同じ理由でクラスタースコアが恒等的にゼロになる。
    let (input, time) = two_period_input(1, false);
    let result = FeEstimator::fit(
        input,
        FeEffects::OneWay,
        FeCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&time)),
        },
        0.95,
    );
    assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
}

#[test]
fn fe_estimator_fit_hac_uses_resolved_dk_time_for_degeneracy_check() {
    // 入力の`time`は3時点だが、`Dk.time`上書きが2時点の退化パターン→拒否。
    let result = FeEstimator::fit(
        four_by_two_input(&THREE_PERIOD_LABELS),
        FeEffects::OneWay,
        FeCovType::Dk {
            bandwidth: Some(0),
            time: lex_time(&TWO_PERIOD_LABELS),
        },
        0.95,
    );
    assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);

    // 逆に入力の`time`が2時点の退化パターンでも、上書きが3時点なら通す。
    let fe = FeEstimator::fit(
        four_by_two_input(&TWO_PERIOD_LABELS),
        FeEffects::OneWay,
        FeCovType::Dk {
            bandwidth: Some(0),
            time: lex_time(&THREE_PERIOD_LABELS),
        },
        0.95,
    )
    .unwrap();
    assert!(*fe.std_errors().get(0, 0) > 1e-8);
}

#[test]
fn fe_estimator_fit_cluster_rejects_degenerate_split_by_non_time_column() {
    // `time`以外の任意のクラスター列でも、全エンティティが2クラスターに1観測ずつなら退化。
    let groups = strings(&["g2", "g1", "g1", "g2", "g2", "g1", "g1", "g2"]);
    let result = FeEstimator::fit(
        four_by_two_input(&THREE_PERIOD_LABELS),
        FeEffects::OneWay,
        FeCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    );
    assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
}

#[test]
fn fe_estimator_fit_two_way_cluster_by_entity_rejects_two_entity_panel() {
    // 2-wayのwithin変換は各時点内でも和をゼロにするため、エンティティ2つのパネルを
    // entityでクラスタリング（`groups: None`＝既定）すると、全時点が2クラスターに
    // 1観測ずつになりクラスタースコアが恒等的にゼロになる（time方向の退化）。
    let result = FeEstimator::fit(
        two_entity_input(),
        FeEffects::TwoWay,
        FeCovType::Cluster { groups: None },
        0.95,
    );
    assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
}

#[test]
fn fe_estimator_fit_two_way_hac_rejects_two_entity_split_override() {
    // 同じtime方向の退化は、`Dk.time`にentityと同じ分け方の2水準列を渡した場合にも起きる。
    let result = FeEstimator::fit(
        two_entity_input(),
        FeEffects::TwoWay,
        FeCovType::Dk {
            bandwidth: Some(0),
            time: lex_time(&["p", "p", "p", "p", "p", "q", "q", "q", "q", "q"]),
        },
        0.95,
    );
    assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
}

#[test]
fn fe_estimator_fit_hac_rejects_time_override_with_wrong_length() {
    // `FeCovType::Dk.time`の上書き列は`FeInput`の検証を通らないため、`fit()`が長さを
    // 検証する（Clusterの`groups`と同じ水準）。
    let (entity, time, x, y) = fixest_reference_input();
    let n = y.len();
    let input =
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();
    let result = FeEstimator::fit(
        input,
        FeEffects::OneWay,
        FeCovType::Dk {
            bandwidth: Some(0),
            time: TimeKeys::lexicographic(time[..n - 1].to_vec()),
        },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        PanelError::IdentifierDimensionMismatch {
            dimension: PanelDimension::TIME,
            y_rows: n,
            other_rows: n - 1,
        }
    );
}

#[test]
fn fe_estimator_fit_hac_rejects_bandwidth_out_of_range() {
    // n_periods=3のため`bandwidth`の許容範囲は`[0, 3)`。`bandwidth=3`（`t`自体）は
    // 範囲外（OLSの`hac_lags`と同型の`[0, n)`境界、`t`版）。
    let (entity, time, x, y) = fixest_reference_input();
    let input = FeInput::from_columns(
        &y,
        &[x],
        vec!["x".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap();

    let dk_time = dk_time_from(&input);
    let result = FeEstimator::fit(
        input,
        FeEffects::OneWay,
        FeCovType::Dk {
            bandwidth: Some(3),
            time: dk_time,
        },
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        PanelError::InvalidDkBandwidth { bandwidth: 3, t: 3 }
    );
}

#[test]
fn fe_estimator_fit_hac_rejects_negative_bandwidth() {
    let (entity, time, x, y) = fixest_reference_input();
    let input = FeInput::from_columns(
        &y,
        &[x],
        vec!["x".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap();

    let dk_time = dk_time_from(&input);
    let result = FeEstimator::fit(
        input,
        FeEffects::OneWay,
        FeCovType::Dk {
            bandwidth: Some(-1),
            time: dk_time,
        },
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        PanelError::InvalidDkBandwidth {
            bandwidth: -1,
            t: 3
        }
    );
}

#[test]
fn fe_estimator_fit_rejects_a_regressor_with_the_same_value_in_every_row() {
    // 生の値が全て同じ列は、固定効果を除く前に分かるので、除いた後の分散ゼロとは別のエラーにする。
    let (input, _) = two_period_input(1, false);
    let constant = vec![2.5; input.nobs()];
    let input = FeInput::from_group_codes(
        input.y(),
        &[constant],
        vec!["x".to_string()],
        input.entity_codes().clone(),
        None,
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::RegressorConstant {
            role: ColumnRole::X,
            column: "x".to_string()
        }
    );
}

#[test]
fn fe_estimator_fit_rejects_an_outcome_with_the_same_value_in_every_row() {
    let (input, _) = two_period_input(1, false);
    let y = vec![4.0; input.nobs()];
    let input = FeInput::from_group_codes(
        &y,
        input.x(),
        input.x_names().to_vec(),
        input.entity_codes().clone(),
        None,
        "y".into(),
    )
    .unwrap();

    let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::Common(CommonError::OutcomeConstant)
    );
}
