//! 入力検証・エラー返却のテスト。

use super::all_rows;
use super::components;
use super::input;
use crate::panel::common::PanelError;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEstimator;
use crate::panel::re::input::ReEffects;
use crate::panel::re::input::ReInput;
use crate::panel::re::variance_method::VarianceMethod;
use crate::panel::re::variance_method::check_positive;
use crate::panel::re::variance_method::solve_two_by_two;

#[test]
fn amemiya_rejects_regressors_without_within_variation() {
    // 1-wayの`xi`（個体間でのみ変わる）と、2-wayの`xi`・`xt`（一次元でしか変わらない）。
    let one_way = input(&all_rows(), &["x1", "xi"], false);
    let err = components(&one_way, ReEffects::OneWay, VarianceMethod::Amemiya).unwrap_err();
    assert_eq!(
        err,
        PanelError::AmemiyaRequiresWithinVariation {
            columns: vec!["xi".to_string()]
        }
    );
    let two_way = input(&all_rows(), &["x1", "xi", "xt"], true);
    let err = components(&two_way, ReEffects::TwoWay, VarianceMethod::Amemiya).unwrap_err();
    assert_eq!(
        err,
        PanelError::AmemiyaRequiresWithinVariation {
            columns: vec!["xi".to_string(), "xt".to_string()]
        }
    );
    // 不均衡でも同じ。Wallace-Hussainは受け付ける。
    let rows: Vec<usize> = (0..24).filter(|r| *r != 5).collect();
    let unbalanced = input(&rows, &["x1", "xi"], false);
    assert!(matches!(
        components(&unbalanced, ReEffects::OneWay, VarianceMethod::Amemiya),
        Err(PanelError::AmemiyaRequiresWithinVariation { .. })
    ));
    assert!(
        components(
            &unbalanced,
            ReEffects::OneWay,
            VarianceMethod::WallaceHussain
        )
        .is_ok()
    );
}

#[test]
fn wallace_hussain_accepts_every_regressor_without_within_variation() {
    // 全列が個体間でのみ変わる。SwamyArora・Amemiyaは使えないがWHは推定できる。
    let inp = input(&all_rows(), &["xi"], false);
    let vc = components(&inp, ReEffects::OneWay, VarianceMethod::WallaceHussain).unwrap();
    assert!(vc.sigma2_eps > 0.0 && vc.time_varying.is_empty());
}

#[test]
fn pooled_regression_failure_is_reported_for_wallace_hussain() {
    // `xi`とその2倍は内部FEには渡らない（within変動なし）が、プールドOLSは特異になる。
    let base = input(&all_rows(), &["x1", "xi"], false);
    let doubled: Vec<f64> = base.x()[1].iter().map(|v| 2.0 * v).collect();
    let columns = vec![base.x()[0].clone(), base.x()[1].clone(), doubled];
    let entity: Vec<String> = (0..24).map(|r| format!("e{}", r / 4)).collect();
    let inp = ReInput::from_columns(
        base.y(),
        &columns,
        vec!["x1".into(), "xi".into(), "xi2".into()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();
    assert!(matches!(
        components(&inp, ReEffects::OneWay, VarianceMethod::WallaceHussain),
        Err(PanelError::PooledRegressionFailed { .. })
    ));
}

#[test]
fn non_positive_idiosyncratic_variance_is_rejected() {
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            check_positive(value, VarianceMethod::Amemiya).is_err(),
            "{value}"
        );
    }
    assert!(check_positive(1e-300, VarianceMethod::Amemiya).is_ok());
    let err = check_positive(0.0, VarianceMethod::WallaceHussain).unwrap_err();
    assert!(err.to_string().contains("wallace_hussain"));
}

#[test]
fn negative_sigma2_eps_from_the_trace_form_is_rejected_through_fit() {
    // 観測数2・6・2の不均衡パネル。y は個体間でほぼ決まり、プールドOLSの残差から作る
    // トレース形の連立が負の`σ_ε²`（約-1.09）を返す。`plm`は0に切り上げてθがNaNになるが、
    // 本実装は拒否する。閉形式（バランス）とAmemiya（`Q_w ≥ 0`を正の自由度で割る）は
    // 負にならず、0になるのは完全適合のときだけ。
    let y = [0.0, 1.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 6.0, 7.0];
    let x1 = [2.0, 1.0, 0.0, -2.0, -1.0, -3.0, -3.0, -3.0, -2.0, 2.0];
    let entity: Vec<String> = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2]
        .iter()
        .map(|i| format!("e{i}"))
        .collect();
    let build = || {
        ReInput::from_columns(
            &y,
            &[x1.to_vec()],
            vec!["x1".into()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap()
    };
    let err = ReEstimator::fit_with_variance_method(
        build(),
        ReEffects::OneWay,
        ReCovType::Classical,
        VarianceMethod::WallaceHussain,
        0.95,
    )
    .unwrap_err();
    match err {
        PanelError::NonPositiveIdiosyncraticVariance { method, sigma2_eps } => {
            assert_eq!(method, "wallace_hussain");
            assert!(sigma2_eps < 0.0, "{sigma2_eps}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    // 同じデータでもAmemiyaは推定できる。
    assert!(
        ReEstimator::fit_with_variance_method(
            build(),
            ReEffects::OneWay,
            ReCovType::Classical,
            VarianceMethod::Amemiya,
            0.95,
        )
        .is_ok()
    );
}

#[test]
fn singular_moment_system_is_a_computation_failure() {
    // 実入力でこの連立が特異になる例は見つかっていない（Amemiyaは`M_w,η = 0`で行列式が
    // `(O - N - K)(O - ΣT_i²/O) > 0`、Wallace-Hussainも一般の設計では非特異）。到達不能と
    // 断定せず、数値的に特異な連立を黙って解かないための防御として残す。
    // 2行が比例する連立（トレース形の`M`が特異）。
    let result = solve_two_by_two((1.0, 2.0, 2.0, 4.0), (1.0, 1.0), VarianceMethod::Amemiya);
    assert!(matches!(
        result,
        Err(PanelError::VarianceComponentsMomentsSingular { .. })
    ));
    let (eps, u) =
        solve_two_by_two((2.0, 0.0, 0.0, 3.0), (4.0, 9.0), VarianceMethod::Amemiya).unwrap();
    assert_eq!((eps, u), (2.0, 3.0));
}
