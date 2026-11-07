//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use crate::linear::common::LeastSquaresError;
use crate::panel::common::BetweenMeans;
use crate::panel::common::PanelDimension;
use crate::panel::common::PanelError;
use crate::panel::common::TimeKeys;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEffects;
use crate::panel::re::ReEstimator;
use crate::panel::re::ReInput;
use crate::panel::re::test_support::*;
use crate::shared::error::CommonError;
use crate::shared::error::ErrorInfo;
use crate::shared::error_catalog::Stage;
use crate::shared::group_codes::GroupCodes;

#[test]
fn re_estimator_fit_rejects_time_invariant_regressor_collinear_with_intercept() {
    // 全エンティティで同じ値の列は切片と共線になる。between回帰ではこの列を落とすため
    // そこでは失敗せず、変換後の最終回帰の設計行列が特異になって計算エラーになる
    // （時間不変の列を許容しても、識別できない入力は通さない）。
    let TimeInvariantColumns { y, x1, entity, .. } = time_invariant_columns();
    let input = ReInput::from_columns(
        &y,
        &[x1, vec![2.0; 24]],
        strings(&["x1", "c"]),
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    assert!(matches!(
        ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95),
        Err(PanelError::QuasiDemeanedRegressionFailed { .. })
    ));
}

#[test]
fn mundlak_cluster_prevalidation_counts_the_identified_auxiliary_slopes() {
    // 補助回帰の傾き係数は`X*`の3個と別名を除いた`x̃`の1個で4個。クラスター数が5なら成功し、
    // 4なら補助回帰の事前検証で`HausmanTestFailed`になる（別名を数えると5個になり、
    // クラスター数5でも失敗してしまう）。RE本体の事前検証（傾き3個）は両方で通る。
    let fit = |labels: &[&str]| {
        ReEstimator::fit(
            mundlak_input(&["x1", "x2", "x1bar"], false),
            ReEffects::OneWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&strings(labels))),
            },
            0.95,
        )
    };
    let five = [
        "a", "a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "d", "d", "d", "d", "e",
        "e", "e", "e", "e", "e", "e",
    ];
    assert_eq!(fit(&five).unwrap().hausman_df(), Some(1));
    let four: Vec<&str> = five
        .iter()
        .map(|g| if *g == "e" { "d" } else { g })
        .collect();
    assert!(matches!(
        fit(&four),
        Err(PanelError::HausmanTestFailed { .. })
    ));
}

#[test]
fn re_estimator_fit_rejects_proportional_time_invariant_regressors() {
    // 定数ではない時間不変の列同士が比例していれば、between回帰では後ろの列が落ちるが、
    // 最終回帰（全列を残す）は特異になる。
    let TimeInvariantColumns { y, x1, entity, .. } = time_invariant_columns();
    let z: Vec<f64> = (0..24).map(|r| (r / 4) as f64).collect();
    let z2: Vec<f64> = z.iter().map(|v| 2.0 * v).collect();
    let input = ReInput::from_columns(
        &y,
        &[x1, z, z2],
        strings(&["x1", "z", "z2"]),
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    assert!(matches!(
        ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95),
        Err(PanelError::QuasiDemeanedRegressionFailed { .. })
    ));
}

#[test]
fn two_way_cluster_prevalidation_follows_the_auxiliary_slopes() {
    // 時点でクラスタリング（G = 4）。RE本体の傾き3個（`G > 3`）は満たすが、補助回帰の傾き
    // （`X*`の3個と`X̃`の1個で4個）には足りず、`fit()`はハウスマン検定の失敗で終わる。
    let c = two_way_columns();
    let by_time = || ReCovType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&c.time)),
    };
    let err = ReEstimator::fit(
        two_way_input(&["x1", "xi", "xt"]),
        ReEffects::TwoWay,
        by_time(),
        0.95,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            PanelError::HausmanTestFailed {
                source: LeastSquaresError::Common(CommonError::InsufficientClustersForInference {
                    g: 4,
                    q: 4
                })
            }
        ),
        "{err:?}"
    );
    // 傾きが3個（`x1`・`xi`と`X̃`の`x1`）なら通り、参照値（手計算）と一致する。
    let re = ReEstimator::fit(
        two_way_input(&["x1", "xi"]),
        ReEffects::TwoWay,
        by_time(),
        0.95,
    )
    .unwrap();
    assert_eq!(re.hausman_df(), Some(1));
    assert_rel(
        re.hausman_statistic().unwrap(),
        2.27570721529081,
        1e-9,
        "cluster time",
    );
}

#[test]
fn two_way_dk_prevalidation_requires_more_periods_than_slopes() {
    // `dk_time`を3時点にまとめる（`min(t, 3)`）。RE本体の傾きは`x1 + xi + xt`で3個のため
    // `T = 3 <= q`で事前検証に落ち、`x1 + xi`（q = 2）は通って手計算と一致する。
    let values: Vec<i128> = (0..24).map(|r| ((r % 4) as i128 + 1).min(3)).collect();
    let labels: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    let dk = || ReCovType::Dk {
        bandwidth: Some(1),
        time: TimeKeys::by_integer(labels.clone(), &values).unwrap(),
    };
    let err = ReEstimator::fit(
        two_way_input(&["x1", "xi", "xt"]),
        ReEffects::TwoWay,
        dk(),
        0.95,
    )
    .unwrap_err();
    assert!(
        matches!(
            &err,
            PanelError::InStage {
                stage: Stage::OverallFTest,
                source,
            } if **source == PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
        ),
        "{err:?}"
    );
    let re = ReEstimator::fit(two_way_input(&["x1", "xi"]), ReEffects::TwoWay, dk(), 0.95).unwrap();
    assert_rel(
        re.hausman_statistic().unwrap(),
        4.37954416942038,
        1e-9,
        "dk 3 periods",
    );
}

#[test]
fn two_way_fit_rejects_unbalanced_panel_singletons_and_missing_time() {
    let c = two_way_columns();
    let build = |keep: &dyn Fn(usize) -> bool, time: bool| {
        let rows: Vec<usize> = (0..24).filter(|&r| keep(r)).collect();
        let pick = |v: &[f64]| rows.iter().map(|&r| v[r]).collect::<Vec<f64>>();
        let entity: Vec<String> = rows.iter().map(|&r| c.entity[r].clone()).collect();
        let period: Vec<String> = rows.iter().map(|&r| c.time[r].clone()).collect();
        ReInput::from_columns(
            &pick(&c.y),
            &[pick(&c.x1)],
            strings(&["x1"]),
            &entity,
            time.then_some(period.as_slice()),
            "y".into(),
        )
        .unwrap()
    };
    let fit = |input| ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95);

    // 1行欠けた不均衡パネル。
    assert!(matches!(
        fit(build(&|r| r != 5, true)).unwrap_err(),
        PanelError::UnbalancedPanelForTwoWay { .. }
    ));
    // 先頭エンティティが1観測だけ（singleton）。
    assert!(matches!(
        fit(build(&|r| r >= 4 || r == 0, true)).unwrap_err(),
        PanelError::SingletonGroup {
            dimension: PanelDimension::Entity,
            ..
        }
    ));
    // `time`が無い2-way。
    assert_eq!(
        fit(build(&|_| true, false)).unwrap_err(),
        PanelError::TwoWayRequiresTime
    );
}

#[test]
fn two_way_fit_reports_computation_errors_by_the_stage_that_fails() {
    let c = two_way_columns();
    // 切片と共線な定数列: between回帰では落ち、最終回帰が特異になる。
    let constant = ReInput::from_columns(
        &c.y,
        &[c.x1.clone(), vec![2.0; 24]],
        strings(&["x1", "c"]),
        &c.entity,
        Some(&c.time),
        "y".into(),
    )
    .unwrap();
    assert!(matches!(
        ReEstimator::fit(constant, ReEffects::TwoWay, ReCovType::Classical, 0.95),
        Err(PanelError::QuasiDemeanedRegressionFailed { .. })
    ));
    // 定数ではない列同士の完全共線（`xi`と`2·xi`）は、between回帰（個体平均の回帰）では後ろの列が
    // 落ち、最終回帰が特異になる。
    let doubled: Vec<f64> = c.xi.iter().map(|v| 2.0 * v).collect();
    let collinear = ReInput::from_columns(
        &c.y,
        &[c.x1.clone(), c.xi.clone(), doubled],
        strings(&["x1", "xi", "xi_twice"]),
        &c.entity,
        Some(&c.time),
        "y".into(),
    )
    .unwrap();
    assert!(matches!(
        ReEstimator::fit(collinear, ReEffects::TwoWay, ReCovType::Classical, 0.95),
        Err(PanelError::QuasiDemeanedRegressionFailed { .. })
    ));
}

#[test]
fn two_way_between_and_fe_degrees_of_freedom_boundaries() {
    let fit = |rows: &[usize], names: &[&str]| {
        ReEstimator::fit(
            subset_input(rows, names, true),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
    };
    // 6エンティティ×2時点: 時点側のbetween回帰は`T = 2 <= K_b + 1 = 2`で失敗する。
    let two_periods: Vec<usize> = (0..24).filter(|r| r % 4 < 2).collect();
    assert!(matches!(
        fit(&two_periods, &["x1"]),
        Err(PanelError::BetweenRegressionFailed {
            means: BetweenMeans::Time,
            ..
        })
    ));
    // 2エンティティ×4時点: 個体平均側のbetween回帰が`N = 2 <= K_b + 1 = 2`で失敗する
    // （個体平均の回帰が先に走るため、メッセージはどちらの回帰かを区別する）。
    let two_entities: Vec<usize> = (0..24).filter(|r| r / 4 < 2).collect();
    let err = fit(&two_entities, &["x1"]).unwrap_err();
    assert!(matches!(
        err,
        PanelError::BetweenRegressionFailed {
            means: BetweenMeans::Entity,
            ..
        }
    ));
    assert_eq!(
        crate::shared::error::ErrorInfo::report(&err).stage(),
        Some(&crate::shared::error_catalog::Stage::BetweenRegressionEntity)
    );
    // 2エンティティ×2時点: 内部の2-way FEの自由度が足りない（`n = 4 <= k + N + T - 1 = 4`）。
    let tiny: Vec<usize> = (0..24).filter(|&r| r / 4 < 2 && r % 4 < 2).collect();
    let err = fit(&tiny, &["x1"]).unwrap_err();
    assert!(matches!(
        &err,
        PanelError::InStage {
            stage: Stage::VarianceComponentWithinRegression,
            source,
        } if matches!(**source, PanelError::InsufficientDegreesOfFreedom { .. })
    ));
    assert_eq!(
        crate::shared::error::ErrorInfo::report(&err).stage(),
        Some(&Stage::VarianceComponentWithinRegression)
    );
    // 3エンティティ×3時点（`df_resid = 3`）は最小規模で成功する。
    let small: Vec<usize> = (0..24).filter(|&r| r / 4 < 3 && r % 4 < 3).collect();
    let re = fit(&small, &["x1"]).unwrap();
    assert_eq!(re.df_resid(), 9 - 2);
}

#[test]
fn re_estimator_fit_dk_rejects_time_of_wrong_length() {
    // `ReCovType::Dk.time`は`ReInput`の検証を通らない公開引数のため、`fit()`が長さを
    // 検証する（`FeEstimator::fit`のDkアームと同じ）。
    let re = ReEstimator::fit(
        cov_type_reference_input(),
        ReEffects::OneWay,
        ReCovType::Dk {
            bandwidth: None,
            time: TimeKeys::lexicographic(strings(&["1", "2", "3"])),
        },
        0.95,
    );

    assert_eq!(
        re.unwrap_err(),
        PanelError::IdentifierDimensionMismatch {
            dimension: PanelDimension::Time,
            y_rows: 7,
            other_rows: 3,
        }
    );
}

#[test]
fn re_estimator_fit_hac_rejects_bandwidth_at_least_t_periods() {
    // `fit()`のdocコメントに明記した`PanelError::InvalidDkBandwidth`の伝播経路を固定する
    // （`resolve_dk_bandwidth`自体はFE側のテストでカバー済みだが、RE経由の配線は
    // 別途確認する。`fe_estimator_fit_hac_rejects_bandwidth_at_least_t_periods`と
    // 同型）。t_periods=3（time∈{1,2,3}）に対しbandwidth=3（`>=t`）を指定する。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    let input = ReInput::from_columns(
        &y,
        &[x1],
        vec!["x1".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap();

    let result = ReEstimator::fit(
        input,
        ReEffects::OneWay,
        ReCovType::Dk {
            bandwidth: Some(3),
            time: TimeKeys::lexicographic(time.clone()),
        },
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        PanelError::InvalidDkBandwidth { bandwidth: 3, t: 3 }
    );
}

#[test]
fn re_estimator_fit_cluster_returns_error_when_cluster_count_at_most_slopes() {
    // クラスター数`G`が傾き係数の数`q`（`df_model - 1`）以下だと構造的に特異になる
    // （OLS/FE共通の制約。REもここに合わせる）。ここではq=1（傾き1個）
    // に対しG=1（全観測が同一クラスター）にして発火させる。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let all_same_cluster = strings(&["g0", "g0", "g0", "g0", "g0", "g0", "g0"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = ReEstimator::fit(
        input,
        ReEffects::OneWay,
        ReCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&all_same_cluster)),
        },
        0.95,
    );

    assert!(matches!(
        result,
        Err(PanelError::Common(CommonError::InsufficientClusters { .. }))
    ));
}

#[test]
fn re_estimator_fit_propagates_fe_singleton_error_from_variance_components() {
    // entity "c"は1観測のみ（singleton）。`swamy_arora_variance_components`内部の
    // FE推定が`PanelError::SingletonGroup`を返し、そのまま伝播することを確認する
    // （`swamy_arora_variance_components_propagates_fe_singleton_error`と同じ配線）。
    let entity = strings(&["a", "a", "c"]);
    let x1 = vec![1.0, 2.0, 3.0];
    let y = [1.0, 2.0, 3.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

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
fn re_estimator_fit_propagates_invalid_confidence_level_error() {
    // `confidence_level`の範囲チェック（`(0, 1)`の範囲外）は、
    // `swamy_arora_variance_components`内部の最初の委譲（`FeEstimator::fit`）が
    // 真っ先に検証するため、`QuasiDemeanedRegressionFailed`ではなく
    // `WithinRegressionFailed`として伝播する（`FeEstimator::fit`自身の
    // `fe_estimator_fit_propagates_invalid_confidence_level_error`と同型）。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 1.5);

    assert_eq!(
        result.unwrap_err(),
        PanelError::WithinRegressionFailed {
            source: crate::linear::common::LeastSquaresError::Common(
                CommonError::InvalidConfidenceLevel {
                    confidence_level: 1.5,
                }
            ),
        }
    );
}

#[test]
fn re_estimator_fit_returns_perfect_fit_when_the_within_regression_is_exact() {
    // ノイズ無しの線形DGP（`y = 2*x1 + entity固有の切片`）では、分散成分を求める内部FEの
    // within回帰の残差が全て0になり、σ_ε²=0（θ=1で最終回帰の定数列が全て0になる）に
    // なる。最終回帰の特異性（`QuasiDemeanedRegressionFailed`）に落ちる前に、原因の分かる
    // 完全フィットとして、分散成分のwithin回帰の文脈つきで止まる。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0];
    let y = [7.0, 9.0, 11.0, 14.0, 16.0, 9.0, 11.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

    let err = result.unwrap_err();
    assert_eq!(
        err,
        PanelError::WithinRegressionFailed {
            source: crate::shared::error::CommonError::PerfectFit.into(),
        }
        .in_stage(Stage::VarianceComponentWithinRegression)
    );
    assert_eq!(
        err.report().code(),
        crate::shared::error_catalog::ErrorCode::PerfectFit
    );
}

#[test]
fn re_estimator_fit_passes_outcome_constant_within_entity_through_without_a_stage() {
    // 個体ごとに定数のyは、分散成分のwithin回帰が説明するものを持たない。入力の事実なので
    // 文脈句は付けない。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0];
    let y = [0.1, 0.1, 0.1, 0.7, 0.7, 1.3, 1.3];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

    assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
}
