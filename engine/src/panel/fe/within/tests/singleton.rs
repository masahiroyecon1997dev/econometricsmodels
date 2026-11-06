//! シングルトン群の検出のテスト。

use crate::panel::common::PanelDimension;
use crate::panel::common::PanelError;
use crate::panel::common::reject_singleton_group;
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
            dimension: PanelDimension::Entity,
            column: "entity".to_string(),
            group_id: "c".to_string(),
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
            dimension: PanelDimension::Entity,
            column: "entity".to_string(),
            group_id: "b".to_string(),
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
            dimension: PanelDimension::Entity,
            column: "entity".to_string(),
            group_id: "c".to_string(),
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
            dimension: PanelDimension::Time,
            column: "time".to_string(),
            group_id: "3".to_string(),
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
            dimension: PanelDimension::Entity,
            column: "entity".to_string(),
            group_id: "c".to_string(),
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
fn reject_singleton_group_with_no_observations_succeeds() {
    // n=0境界（`validate_no_zero_variance_regressors_with_zero_observations_and_a_
    // regressor_succeeds`と同様の境界値テストの慣習に合わせる）。空配列にはsingleton
    // となりうる要素自体が存在しないため`Ok(())`になる。
    assert_eq!(
        reject_singleton_group(
            PanelDimension::Entity,
            "entity",
            &GroupCodes::from_labels(&[])
        ),
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
            dimension: PanelDimension::Entity,
            column: "entity".to_string(),
            group_id: "c".to_string(),
        })
    );
    let input = input.with_effect_names("firm".to_string(), None);
    assert_eq!(
        validate_no_singleton_groups_one_way(&input),
        Err(PanelError::SingletonGroup {
            dimension: PanelDimension::Entity,
            column: "firm".to_string(),
            group_id: "c".to_string(),
        })
    );

    // 時点側の単独グループ（時点"2"は1観測）は2つ目の名前で報告する。
    let y = [1.0, 2.0, 3.0, 4.0];
    let entity = strings(&["a", "a", "b", "b"]);
    let time = strings(&["1", "2", "1", "1"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into())
        .unwrap()
        .with_effect_names("firm".to_string(), Some("year".to_string()));
    assert_eq!(
        validate_no_singleton_groups_two_way(&input),
        Err(PanelError::SingletonGroup {
            dimension: PanelDimension::Time,
            column: "year".to_string(),
            group_id: "2".to_string(),
        })
    );
}
