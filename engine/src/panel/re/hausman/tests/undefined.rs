//! 検定が定義されない場合（`None`）・除外される列・エラー返却のテスト。

use crate::linear::common::LeastSquaresError;
use crate::panel::common::PanelError;
use crate::panel::common::TimeKeys;
use crate::panel::fe::FeCovType;
use crate::panel::fe::FeEffects;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::FeInput;
use crate::panel::re::ReEffects;
use crate::panel::re::ReEstimator;
use crate::panel::re::ReInput;
use crate::panel::re::hausman::hausman_within_column_is_aliased;
use crate::panel::re::hausman::re_hausman_test;
use crate::panel::re::input::ReCovType;
use crate::panel::re::test_support::*;
use crate::panel::re::variance::swamy_arora_variance_components;
use crate::shared::error::CommonError;
use crate::shared::error_catalog::Stage;
use crate::shared::group_codes::GroupCodes;

#[test]
fn re_estimator_fit_with_only_time_invariant_regressors_has_no_hausman_test() {
    // 説明変数がすべて時間不変でも推定できる。内部FEの説明変数は0個で、
    // σ_ε²は`y`のwithin変動の平均二乗（自由度`n - N`）。比較対象の傾きが無いので
    // ハウスマン検定は`None`、within R²は0。
    let input = time_invariant_input(&["z"]);
    let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
    assert!(vc.time_varying.is_empty());
    assert_eq!(vc.fe.input().x().len(), 0);

    let y = time_invariant_columns().y;
    let ssr_within: f64 = y
        .chunks(4)
        .map(|c| {
            let m = c.iter().sum::<f64>() / 4.0;
            c.iter().map(|v| (v - m).powi(2)).sum::<f64>()
        })
        .sum();
    assert!((vc.sigma2_eps - ssr_within / (24.0 - 6.0)).abs() < 1e-12);

    let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
    assert_eq!(re.hausman_statistic(), None);
    assert_eq!(re.hausman_p_value(), None);
    assert_eq!(re.hausman_df(), None);
    assert_eq!(re.r_squared_within(), 0.0);
    assert_eq!(re.df_model(), 2);
}

#[test]
fn hausman_is_none_when_every_time_varying_regressor_varies_only_over_time() {
    // 時間変化のある列が`xt`（時点間でのみ変わる列）だけなら、`x̃t`は識別できず、
    // 検定できる列が残らないため`hausman_*`は`None`（`fit()`自体は成功する）。
    let c = two_way_columns();
    let input = ReInput::from_columns(
        &c.y,
        std::slice::from_ref(&c.xt),
        strings(&["xt"]),
        &c.entity,
        None,
        "y".into(),
    )
    .unwrap();

    let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

    assert_eq!(re.hausman_statistic(), None);
    assert_eq!(re.hausman_p_value(), None);
    assert_eq!(re.hausman_df(), None);
}

#[test]
fn two_way_without_regressors_has_zero_r_squared_and_no_hausman_test() {
    // 傾き係数が0個（定数項のみ）。Pythonからは`x=[]`が拒否されるためエンジンだけの経路。
    let c = two_way_columns();
    let input =
        ReInput::from_columns(&c.y, &[], vec![], &c.entity, Some(&c.time), "y".into()).unwrap();
    let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
    assert_eq!(
        (
            re.r_squared_within(),
            re.r_squared_between(),
            re.r_squared_overall()
        ),
        (0.0, 0.0, 0.0)
    );
    assert_eq!(re.hausman_statistic(), None);
}

#[test]
fn two_way_hausman_is_none_without_two_way_within_variation() {
    // 一次元でしか変わらない列だけなら、2-way within変換後の`X̃`が0列で検定できない。
    let re = ReEstimator::fit(
        two_way_input(&["xi", "xt"]),
        ReEffects::TwoWay,
        ReCovType::Classical,
        0.95,
    )
    .unwrap();
    assert_eq!(re.hausman_statistic(), None);
    assert_eq!(re.hausman_p_value(), None);
    assert_eq!(re.hausman_df(), None);
    // within R²は1-wayと同じく0（within変換後の当てはめ値が恒等的に0）。
    assert_eq!(re.r_squared_within(), 0.0);
}

#[test]
fn two_way_hausman_leaves_out_a_column_that_is_its_own_within_transform() {
    // 個体平均も時点平均も0の交互作用列`(e - ē)(t - t̄)`は、準偏差変換しても変わらず
    // （`x* = x̃`）、補助回帰で識別できない。検定対象から外し、`x1`だけを検定する。
    let c = two_way_columns();
    let interaction: Vec<f64> = (0..24)
        .map(|r| ((r / 4) as f64 - 2.5) * ((r % 4) as f64 - 1.5))
        .collect();
    let input = ReInput::from_columns(
        &c.y,
        &[c.x1.clone(), interaction],
        strings(&["x1", "z"]),
        &c.entity,
        Some(&c.time),
        "y".into(),
    )
    .unwrap();
    let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
    assert_eq!(re.hausman_df(), Some(1));
    assert!(re.hausman_statistic().unwrap().is_finite());
}

#[test]
fn hausman_within_column_is_aliased_only_when_the_difference_is_constant() {
    let within = [-1.0, 1.0, -2.0, 2.0];
    // x* = x̃ + 定数: 識別できない。
    let aliased: Vec<f64> = within.iter().map(|v| v + 0.37).collect();
    assert!(hausman_within_column_is_aliased(&aliased, &within, 2.0));
    // 差が個体ごとに違う（θ・平均が個体間で異なる）: 識別できる。
    let identified = [-1.0 + 0.3, 1.0 + 0.3, -2.0 + 0.9, 2.0 + 0.9];
    assert!(!hausman_within_column_is_aliased(&identified, &within, 2.9));
}

#[test]
fn re_estimator_fit_hausman_is_none_when_no_slope_regressors() {
    // 比較対象の傾き係数が0個（`x=[]`、定数項のみのモデル）の場合はNone
    // （`re.rs`のモジュールdoc「ハウスマン検定」参照）。`time`の有無によらない。
    let entity = strings(&["a", "a", "b", "b", "c", "c"]);
    let time = strings(&["1", "2", "1", "2", "1", "2"]);
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    for time in [None, Some(time.as_slice())] {
        let input = ReInput::from_columns(&y, &[], vec![], &entity, time, "y".into()).unwrap();
        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
    }
}

#[test]
fn re_hausman_test_returns_none_when_auxiliary_regression_is_rank_deficient() {
    // 補助回帰のランク落ち→None（`re.rs`のモジュールdoc「ハウスマン検定」参照）。
    // `ReEstimator::fit`経由では内部FE推定・RE本体が先に失敗するため自然な入力から
    // 再現しにくい。private関数`re_hausman_test`を直接呼び、準偏差変換済み側の
    // 列にwithin変換済み`x1`と同一の列を含めて完全共線にする。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    let fe_input =
        FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
            .unwrap();
    let fe = FeEstimator::fit(fe_input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();
    // `x*`に同じ列を2回入れて、補助回帰の設計行列を（`x̃`の識別可否とは無関係に）
    // 完全共線にする。
    let x_star = vec![fe.input().x()[0].clone(), fe.input().x()[0].clone()];

    let result = re_hausman_test(
        &fe,
        &[0],
        &y,
        &x_star,
        &["x1_star".to_string(), "x1_star2".to_string()],
        "y",
        &ReCovType::Classical,
        &GroupCodes::from_labels(&entity),
        0.95,
    );

    assert!(matches!(result, Err(PanelError::HausmanTestFailed { .. })));
}

#[test]
fn re_estimator_fit_hausman_returns_error_when_clusters_do_not_cover_auxiliary_slopes() {
    // RE本体は`G=3 > q=2`で成功するが、補助回帰の傾き係数は`2k=4 >= G`のため
    // ロバスト共分散が構造的に特異になり、`fit()`は`HausmanTestFailed`で失敗する。
    let groups = strings(&[
        "g1", "g1", "g1", "g2", "g2", "g3", "g3", "g3", "g1", "g2", "g3", "g3",
    ]);
    let result = ReEstimator::fit(
        hausman_two_slope_input(None),
        ReEffects::OneWay,
        ReCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    );
    assert!(matches!(
        result.unwrap_err(),
        PanelError::HausmanTestFailed {
            source: LeastSquaresError::Common(CommonError::InsufficientClustersForInference {
                g: 3,
                q: 4
            })
        }
    ));
}

#[test]
fn re_estimator_fit_returns_error_when_dk_periods_do_not_cover_hausman_slopes() {
    // 時点数`T=2`ではDK共分散のrankが`T-1=1`で、全体F検定の傾き係数`q=2`個の
    // 部分行列が構造的に特異になる。補助回帰の数値的な特異性判定を待たず、入力から
    // 判定できる`InsufficientDkPeriodsForInference`で弾く（`T=3 > k=2`で成功する
    // 境界は`re_estimator_fit_hausman_dk_default_bandwidth_matches_auto_resolved_value`）。
    let time = strings(&["1", "2", "1", "2", "1", "2", "1", "2", "1", "2", "1", "2"]);
    let result = ReEstimator::fit(
        hausman_two_slope_input(Some(&time)),
        ReEffects::OneWay,
        ReCovType::Dk {
            bandwidth: Some(0),
            time: TimeKeys::lexicographic(time.clone()),
        },
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        PanelError::InsufficientDkPeriodsForInference { t_periods: 2, q: 2 }
            .in_stage(Stage::OverallFTest)
    );
}
