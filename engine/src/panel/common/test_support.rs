//! 複数のファイルのテストが共有するフィクスチャ。

use crate::shared::group_codes::GroupCodes;

pub(super) fn labels(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| s.to_string()).collect()
}

pub(super) fn codes(values: &[&str]) -> GroupCodes {
    let labels: Vec<String> = values.iter().map(|s| s.to_string()).collect();
    GroupCodes::from_labels(&labels)
}
