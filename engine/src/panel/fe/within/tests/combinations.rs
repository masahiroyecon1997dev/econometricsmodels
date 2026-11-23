//! 全ての(entity, time)の組合せが過不足なく1回ずつ現れるか（`covers_every_combination`）のテスト。

use crate::panel::common::covers_every_combination;
use crate::panel::fe::test_support::*;
use crate::shared::group_codes::GroupCodes;

fn codes(values: &[&str]) -> GroupCodes {
    GroupCodes::from_labels(&strings(values))
}

#[test]
fn a_full_grid_covers_every_combination_in_any_row_order() {
    let entity = codes(&["a", "b", "a", "b", "a", "b"]);
    let time = codes(&["1", "1", "2", "2", "3", "3"]);
    assert!(covers_every_combination(&entity, &time));
}

#[test]
fn an_unbalanced_panel_does_not_cover_every_combination() {
    let entity = codes(&["a", "a", "a", "b", "b", "c"]);
    let time = codes(&["1", "2", "3", "1", "3", "2"]);
    assert!(!covers_every_combination(&entity, &time));
}

#[test]
fn a_repeated_pair_does_not_cover_every_combination_even_when_the_counts_match() {
    // 3×2のグリッドと同じ6行だが、(a,1)が2回現れ(a,2)が欠ける。
    let entity = codes(&["a", "a", "b", "b", "c", "c"]);
    let time = codes(&["1", "1", "1", "2", "1", "2"]);
    assert_eq!(entity.nobs(), entity.n_groups() * time.n_groups());
    assert!(!covers_every_combination(&entity, &time));
}

#[test]
fn a_pair_repeated_three_times_does_not_cover_every_combination() {
    let entity = codes(&["a", "a", "a", "b"]);
    let time = codes(&["1", "1", "1", "1"]);
    assert!(!covers_every_combination(&entity, &time));
}

#[test]
fn an_empty_panel_covers_every_combination() {
    assert!(covers_every_combination(&codes(&[]), &codes(&[])));
}
