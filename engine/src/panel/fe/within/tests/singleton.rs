//! シングルトン群の検出のテスト。

use crate::panel::common::PanelDimension;
use crate::panel::common::PanelError;
use crate::panel::common::{EffectNames, ensure_no_singleton_groups};
use crate::panel::fe::input::FeInput;
use crate::panel::fe::test_support::*;
use crate::panel::fe::within::validate_no_singleton_groups_one_way;
use crate::panel::fe::within::validate_no_singleton_groups_two_way;
use crate::shared::group_codes::GroupCodes;

#[test]
fn validate_no_singleton_groups_one_way_detects_entity_singleton() {
    // entity "c"は観測数1（singleton）。
    let y = [1.0, 2.0, 3.0];
    let entity = strings(&["a", "a", "c"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    let result = validate_no_singleton_groups_one_way(&input);

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
fn validate_no_singleton_groups_reports_first_singleton_in_observation_order() {
    // singletonが"b"と"a"の2つ。観測順では"b"が先、コード順（辞書順）では"a"が先。
    // 観測順で最初のもの（"b"）を報告する（整数コード化の前と同じ選び方）。
    let y = [1.0, 2.0, 3.0, 4.0];
    let entity = strings(&["b", "a", "c", "c"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    assert_eq!(
        validate_no_singleton_groups_one_way(&input).unwrap_err(),
        PanelError::SingletonGroup {
            dimension: PanelDimension::ENTITY,
            column: "entity".to_string(),
            group_id: "b".to_string(),
            n_dropped_rows: 2,
        }
    );
}

#[test]
fn validate_no_singleton_groups_one_way_accepts_no_singleton() {
    let y = [1.0, 2.0, 3.0, 4.0];
    let entity = strings(&["a", "a", "b", "b"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    assert_eq!(validate_no_singleton_groups_one_way(&input), Ok(()));
}

#[test]
fn validate_no_singleton_groups_two_way_detects_entity_singleton() {
    // entity "c"は時点"1"のみの1観測（singleton）。time側は各時点2観測ずつで
    // singletonではない。
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let entity = strings(&["a", "a", "b", "b", "c"]);
    let time = strings(&["1", "2", "1", "2", "1"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

    let result = validate_no_singleton_groups_two_way(&input);

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
fn validate_no_singleton_groups_two_way_detects_time_singleton() {
    // time "3"はentity "a"のみの1観測（singleton）。entity側はどちらも2観測ずつで
    // singletonではない。
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let entity = strings(&["a", "a", "a", "b", "b"]);
    let time = strings(&["1", "2", "3", "1", "2"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

    let result = validate_no_singleton_groups_two_way(&input);

    assert_eq!(
        result.unwrap_err(),
        PanelError::SingletonGroup {
            dimension: PanelDimension::TIME,
            column: "time".to_string(),
            group_id: "3".to_string(),
            n_dropped_rows: 1,
        }
    );
}

#[test]
fn validate_no_singleton_groups_two_way_reports_entity_before_time_when_both_present() {
    // entity "c"（1観測）とtime "3"（1観測、entity "c"自身の行）が両方singleton。
    // entityを先にチェックする方針（関数doc参照）により、entity側が報告される。
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let entity = strings(&["a", "a", "b", "b", "c"]);
    let time = strings(&["1", "2", "1", "2", "3"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

    let result = validate_no_singleton_groups_two_way(&input);

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
fn validate_no_singleton_groups_two_way_requires_time() {
    // entity側はsingletonではない（"a"が2観測）ため、`time`未指定の
    // `PanelError::TwoWayRequiresTime`が先に検出されることを確認する。
    let y = [1.0, 2.0];
    let entity = strings(&["a", "a"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    let result = validate_no_singleton_groups_two_way(&input);

    assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
}

#[test]
fn validate_no_singleton_groups_two_way_reports_missing_time_even_with_entity_singleton() {
    // entity "c"はsingletonだが`time`も未指定。`time`の存在チェックを先に行う方針
    // （関数doc、`within_transform_two_way`と同じ順序）により`TwoWayRequiresTime`が
    // 優先される。
    let y = [1.0, 2.0, 3.0];
    let entity = strings(&["a", "a", "c"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    let result = validate_no_singleton_groups_two_way(&input);

    assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
}

#[test]
fn ensure_no_singleton_groups_with_no_observations_succeeds() {
    // n=0境界（`validate_no_zero_variance_regressors_with_zero_observations_and_a_
    // regressor_succeeds`と同様の境界値テストの慣習に合わせる）。空配列にはsingleton
    // となりうる要素自体が存在しないため`Ok(())`になる。
    assert_eq!(
        ensure_no_singleton_groups(&EffectNames::default(), &GroupCodes::from_labels(&[]), None),
        Ok(())
    );
}

#[test]
fn validate_no_singleton_groups_two_way_accepts_no_singleton() {
    let y = [1.0, 2.0, 3.0, 4.0];
    let entity = strings(&["a", "a", "b", "b"]);
    let time = strings(&["1", "2", "1", "2"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

    assert_eq!(validate_no_singleton_groups_two_way(&input), Ok(()));
}

#[test]
fn singleton_group_error_reports_the_column_names_set_by_with_effect_names() {
    // 既定の名前は"entity"・"time"。`engine_pybind`が渡す列名がエラーの`column`に載る。
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let entity = strings(&["a", "a", "b", "b", "c"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();
    assert_eq!(
        validate_no_singleton_groups_one_way(&input),
        Err(PanelError::SingletonGroup {
            dimension: PanelDimension::ENTITY,
            column: "entity".to_string(),
            group_id: "c".to_string(),
            n_dropped_rows: 1,
        })
    );
    let input = input.with_effect_names(vec!["firm".to_string()]);
    assert_eq!(
        validate_no_singleton_groups_one_way(&input),
        Err(PanelError::SingletonGroup {
            dimension: PanelDimension::ENTITY,
            column: "firm".to_string(),
            group_id: "c".to_string(),
            n_dropped_rows: 1,
        })
    );

    // 時点側の単独グループ（時点"2"は1観測）は2つ目の名前で報告する。
    let y = [1.0, 2.0, 3.0, 4.0];
    let entity = strings(&["a", "a", "b", "b"]);
    let time = strings(&["1", "2", "1", "1"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into())
        .unwrap()
        .with_effect_names(vec!["firm".to_string(), "year".to_string()]);
    assert_eq!(
        validate_no_singleton_groups_two_way(&input),
        Err(PanelError::SingletonGroup {
            dimension: PanelDimension::TIME,
            column: "year".to_string(),
            group_id: "2".to_string(),
            // 時点2の行(a,2)を落とすと、個体aが(a,1)だけになって、それも落ちる。
            n_dropped_rows: 2,
        })
    );
}

#[test]
fn the_error_counts_the_rows_that_repeated_removal_of_singleton_groups_would_drop() {
    // (c,3)だけの個体cがsingleton。これを落とすと時点3が(b,3)だけになり、続けて個体bが(b,2)だけに、
    // 時点2が(a,2)だけに、時点1が(a,1)だけに…と、5行すべてが連鎖して落ちる。
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let entity = strings(&["a", "a", "b", "b", "c"]);
    let time = strings(&["1", "2", "2", "3", "3"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

    assert_eq!(
        validate_no_singleton_groups_two_way(&input),
        Err(PanelError::SingletonGroup {
            dimension: PanelDimension::ENTITY,
            column: "entity".to_string(),
            group_id: "c".to_string(),
            n_dropped_rows: 5,
        })
    );
}

#[test]
fn a_one_way_panel_drops_only_the_singleton_rows() {
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let entity = strings(&["a", "b", "c", "c", "d", "e"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    assert!(matches!(
        validate_no_singleton_groups_one_way(&input),
        Err(PanelError::SingletonGroup {
            n_dropped_rows: 4,
            ..
        })
    ));
}
