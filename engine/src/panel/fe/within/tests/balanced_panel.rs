//! バランスパネルの検証のテスト。

use crate::panel::common::EffectNames;
use crate::panel::common::PanelError;
use crate::panel::common::validate_balanced_panel;
use crate::panel::fe::test_support::*;
use crate::shared::group_codes::GroupCodes;

fn codes(values: &[&str]) -> GroupCodes {
    GroupCodes::from_labels(&strings(values))
}

#[test]
fn validate_balanced_panel_accepts_shuffled_rows_when_entities_and_periods_differ() {
    // 3エンティティ×2時点（n_entities != n_periods）で、行を観測順に並べていない。
    let entity = codes(&["c", "a", "b", "a", "c", "b"]);
    let time = codes(&["2", "1", "2", "2", "1", "1"]);
    assert_eq!(
        validate_balanced_panel(&entity, &time, &EffectNames::default()),
        Ok(())
    );
}

#[test]
fn validate_balanced_panel_rejects_duplicate_pair_offsetting_a_missing_one() {
    // 3×2。(a,1)が重複し(a,2)が欠落するが、n_obs=6=3*2で件数は一致してしまう。
    let entity = codes(&["a", "a", "b", "b", "c", "c"]);
    let time = codes(&["1", "1", "1", "2", "1", "2"]);
    assert_eq!(
        validate_balanced_panel(&entity, &time, &EffectNames::default()),
        Err(PanelError::UnbalancedPanelForTwoWay {
            n_obs: 6,
            entity_column: "entity".to_string(),
            n_entities: 3,
            time_column: "time".to_string(),
            n_periods: 2,
        })
    );
}

#[test]
fn validate_balanced_panel_reports_the_column_names_set_by_effect_names() {
    let entity = codes(&["a", "a", "b"]);
    let time = codes(&["1", "2", "1"]);
    let names = EffectNames::new("firm".to_string(), Some("year".to_string()));
    let err = validate_balanced_panel(&entity, &time, &names).unwrap_err();
    assert_eq!(
        err,
        PanelError::UnbalancedPanelForTwoWay {
            n_obs: 3,
            entity_column: "firm".to_string(),
            n_entities: 2,
            time_column: "year".to_string(),
            n_periods: 2,
        }
    );
}
