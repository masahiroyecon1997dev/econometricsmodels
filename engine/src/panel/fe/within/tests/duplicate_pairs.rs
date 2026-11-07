//! 重複する(entity, time)の組合せの検証のテスト。

use crate::panel::common::EffectNames;
use crate::panel::common::PanelError;
use crate::panel::common::ensure_no_duplicate_entity_time_pairs;
use crate::panel::fe::test_support::*;
use crate::shared::group_codes::GroupCodes;

fn codes(values: &[&str]) -> GroupCodes {
    GroupCodes::from_labels(&strings(values))
}

#[test]
fn an_unbalanced_panel_without_repeated_pairs_is_accepted() {
    let entity = codes(&["a", "a", "a", "b", "b", "c"]);
    let time = codes(&["1", "2", "3", "1", "3", "2"]);
    assert_eq!(
        ensure_no_duplicate_entity_time_pairs(&entity, &time, &EffectNames::default()),
        Ok(())
    );
}

#[test]
fn a_repeated_pair_is_rejected_even_when_the_counts_match_a_balanced_panel() {
    // 3×2のグリッドと同じ6行だが、(a,1)が2回現れ(a,2)が欠ける。
    let entity = codes(&["a", "a", "b", "b", "c", "c"]);
    let time = codes(&["1", "1", "1", "2", "1", "2"]);
    let names = EffectNames::new("firm".to_string(), Some("year".to_string()));
    assert_eq!(
        ensure_no_duplicate_entity_time_pairs(&entity, &time, &names),
        Err(PanelError::DuplicateEntityTimePair {
            entity_column: "firm".to_string(),
            entity_id: "a".to_string(),
            time_column: "year".to_string(),
            time_id: "1".to_string(),
        })
    );
}

#[test]
fn it_reports_the_repeated_pair_whose_second_occurrence_comes_first() {
    // (b,2)は行1と行4、(a,1)は行2と行3。2回目の行が早い(a,1)を報告する。
    let entity = codes(&["c", "b", "a", "a", "b", "c"]);
    let time = codes(&["1", "2", "1", "1", "2", "3"]);
    let err =
        ensure_no_duplicate_entity_time_pairs(&entity, &time, &EffectNames::default()).unwrap_err();
    assert_eq!(
        err,
        PanelError::DuplicateEntityTimePair {
            entity_column: "entity".to_string(),
            entity_id: "a".to_string(),
            time_column: "time".to_string(),
            time_id: "1".to_string(),
        }
    );
}

#[test]
fn a_pair_repeated_three_times_is_reported_once() {
    let entity = codes(&["a", "a", "a", "b"]);
    let time = codes(&["1", "1", "1", "1"]);
    assert!(matches!(
        ensure_no_duplicate_entity_time_pairs(&entity, &time, &EffectNames::default()),
        Err(PanelError::DuplicateEntityTimePair { .. })
    ));
}

#[test]
fn a_sparse_panel_uses_the_sorting_path_and_reports_the_same_pair() {
    // 個体100×時点100のグリッドに対して行は6。グリッドが行数の8倍を超えるので並べ替えで調べる。
    // 行は102。(e000,t000)は行0と行100。
    let mut e_labels: Vec<String> = (0..100).map(|i| format!("e{i:03}")).collect();
    let mut t_labels: Vec<String> = (0..100).map(|i| format!("t{i:03}")).collect();
    e_labels.extend(["e000".to_string(), "e050".to_string()]);
    t_labels.extend(["t000".to_string(), "t007".to_string()]);
    let (entity, time) = (
        GroupCodes::from_labels(&e_labels),
        GroupCodes::from_labels(&t_labels),
    );
    assert!(entity.n_groups() * time.n_groups() > 8 * entity.nobs());
    assert_eq!(
        ensure_no_duplicate_entity_time_pairs(&entity, &time, &EffectNames::default()),
        Err(PanelError::DuplicateEntityTimePair {
            entity_column: "entity".to_string(),
            entity_id: "e000".to_string(),
            time_column: "time".to_string(),
            time_id: "t000".to_string(),
        })
    );
}
