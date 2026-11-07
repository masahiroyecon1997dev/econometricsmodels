//! エラーの伝播・変換のテスト。

use crate::panel::common::PanelDimension;
use crate::panel::common::PanelError;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEstimator;
use crate::panel::re::input::ReEffects;
use crate::panel::re::input::ReInput;
use crate::panel::re::test_support::*;
use crate::panel::re::variance::swamy_arora_variance_components;
use crate::panel::re::variance::within_for_variance_component;
use crate::shared::error::CommonError;
use crate::shared::error::ErrorInfo;
use crate::shared::error_catalog::Stage;

#[test]
fn swamy_arora_variance_components_does_not_panic_when_entity_means_are_all_zero() {
    // 全エンティティの`ȳ_i.`が完全に一致し、かつその共通値がちょうど0のデータ。between回帰は
    // 「切片=0・傾き=0」という完全な当てはめ（`SSR_between=0`）になる。between回帰が
    // 係数の標準誤差・t統計量を計算していた頃は、切片・傾きの標準誤差が0で
    // `stat=0.0/0.0=NaN`となり`StudentsT::cdf(NaN)`がパニックしていた（`shared/inference.rs`の
    // `compute_inference_stat_returns_nan_p_value_without_panicking_when_coef_and_se_are_both_zero`
    // で別途固定）。現在のbetween回帰は残差平方和しか使わないため推論統計量を計算せず、
    // `σ_u² = max(0, 0 - σ_ε²/t_bar) = 0`で成功する。
    let entity = strings(&["a", "a", "b", "b", "c", "c"]);
    let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
    let y = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
    let input =
        ReInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();

    let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

    assert_eq!(vc.sigma2_u, 0.0);
}

#[test]
fn swamy_arora_variance_components_propagates_fe_singleton_error() {
    // entity "c"は1観測のみ（singleton）。内部FE推定（1-way）の
    // `PanelError::SingletonGroup`がそのまま伝播することを確認する。
    let entity = strings(&["a", "a", "c"]);
    let x1 = vec![1.0, 2.0, 3.0];
    let y = [1.0, 2.0, 3.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95);

    assert_eq!(
        result.unwrap_err(),
        PanelError::SingletonGroup {
            dimension: PanelDimension::Entity,
            column: "entity".to_string(),
            group_id: "c".to_string(),
            n_dropped_rows: 1,
        }
    );
}

#[test]
fn swamy_arora_variance_components_returns_between_regression_failed_when_entities_are_insufficient()
 {
    // between回帰は切片+傾き1個で2パラメータ。エンティティ数が2つだけだと
    // `n_entities <= k`（`OlsEstimator::fit`自身の`n<=k`検証）で失敗する。
    // singletonにならないよう各エンティティは2観測以上にする。
    let entity = strings(&["a", "a", "b", "b"]);
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let y = [1.0, 2.0, 3.0, 5.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95);

    assert!(matches!(
        result,
        Err(PanelError::BetweenRegressionFailed { .. })
    ));
}

#[test]
fn re_estimator_fit_propagates_between_regression_failed_error() {
    // `swamy_arora_variance_components_returns_between_regression_failed_when_
    // entities_are_insufficient`と同じデータ（n_entities=2, k=1で between回帰が
    // 特異になる）。`ReEstimator::fit`が分散成分推定の失敗をそのまま伝播することを
    // 確認する。
    let entity = strings(&["a", "a", "b", "b"]);
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let y = [1.0, 2.0, 3.0, 5.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

    assert!(matches!(
        result,
        Err(PanelError::BetweenRegressionFailed { .. })
    ));
}

/// 内部FEの失敗のうち、REの利用者に文脈が伝わらない3種だけが「分散成分を推定するwithin回帰」の
/// `stage`で包まれる。データの事実を述べるもの、`stage`を付けてはいけない`code`は包まない。
#[test]
fn within_for_variance_component_wraps_only_stage_capable_internal_fe_errors() {
    use crate::linear::common::LeastSquaresError;

    let stage = Stage::VarianceComponentWithinRegression;
    let wrapped: [fn() -> PanelError; 3] = [
        || PanelError::InsufficientDegreesOfFreedom {
            n_obs: 6,
            n_entities: 3,
            n_periods: None,
            k: 3,
        },
        || PanelError::WithinRegressionFailed {
            source: LeastSquaresError::SingularMatrix {
                columns: vec!["const".to_string()],
            },
        },
        || PanelError::FTestFailed {
            source: LeastSquaresError::Common(CommonError::NearSingularMatrix {
                matrix: crate::shared::error::MatrixKind::CoefficientCovariance,
            }),
        },
    ];
    for make in wrapped {
        let result = within_for_variance_component(make());
        assert_eq!(result, make().in_stage(stage.clone()));
        assert_eq!(result.report().stage(), Some(&stage));
    }
    let untouched: [fn() -> PanelError; 3] = [
        || PanelError::SingletonGroup {
            dimension: PanelDimension::Entity,
            column: "entity".to_string(),
            group_id: "a".to_string(),
            n_dropped_rows: 0,
        },
        || PanelError::UnbalancedPanelForTwoWay {
            n_obs: 5,
            entity_column: "entity".to_string(),
            n_entities: 2,
            time_column: "time".to_string(),
            n_periods: 3,
        },
        // どの計算で起きても同じ入力の誤り（`stage`を付けない`code`）。
        || PanelError::WithinRegressionFailed {
            source: LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5,
            }),
        },
    ];
    for make in untouched {
        assert_eq!(within_for_variance_component(make()), make());
    }
}
