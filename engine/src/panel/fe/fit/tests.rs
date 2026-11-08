//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `basic.rs`: 基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）
//! - `reference.rs`: 参照実装・閉形式・独立再計算との照合
//! - `cov_type.rs`: `cov_type`別の共分散・検定統計量
//! - `unbalanced.rs`: 不均衡パネルの2-way FE（交互射影）。ダミー変数回帰との照合
//! - `multi_way.rs`: 3次元以上のFE（交互射影）。ダミー変数回帰との照合
//! - `validation.rs`: 入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序
//! - `proptests.rs`: property-basedテスト

use crate::panel::common::TimeKeys;
use crate::panel::fe::input::FeCovType;
use crate::panel::fe::input::FeInput;
use crate::panel::fe::test_support::*;

mod basic;
mod cov_type;
mod multi_way;
mod proptests;
mod reference;
mod unbalanced;
mod validation;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

/// 4エンティティ×3期間の1-way FE入力（説明変数は`x1`〜`x3`の先頭`k`個）。DKの
/// `t_periods=3`に対し`k=3`（拒否）と`k=2`（成功）の境界を作るため。`T=2`は使わない:
/// 1-wayのwithin変換では`x̃_i1 = -x̃_i2`・`ẽ_i1 = -ẽ_i2`となり時点スコアが
/// `h_1 = h_2 = 0`に退化する（`rank(S) = 0`）ため、`t <= q`の規則を検証する入力として不適。
pub(super) fn three_period_input(k: usize, time: Option<&[String]>) -> FeInput {
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
    let columns = [
        vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.0, 2.0, 1.0, 3.0, 7.0, 4.0],
        vec![2.0, 1.0, 4.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0, 1.0, 1.0, 3.0],
        vec![4.0, 2.0, 1.0, 3.0, 3.0, 5.0, 2.0, 6.0, 3.0, 0.0, 4.0, 2.0],
    ];
    let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
    let names = ["x1", "x2", "x3"]
        .iter()
        .take(k)
        .map(|n| n.to_string())
        .collect();
    FeInput::from_columns(&y, &columns[..k], names, &entity, time, "y".into()).unwrap()
}

pub(super) fn three_period_time() -> Vec<String> {
    strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"])
}

/// 4エンティティ×2期間・`k=1`（`k = 0`なら説明変数なし）。`break_pattern`なら
/// エンティティ`a`に時点`1`の観測を1つ足し（不均衡な1-way）、「全エンティティが
/// 2時点に1観測ずつ」のパターンを崩す。
pub(super) fn two_period_input(k: usize, break_pattern: bool) -> (FeInput, Vec<String>) {
    let mut entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
    let mut time = strings(&["1", "2", "1", "2", "1", "2", "1", "2"]);
    let mut x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0];
    let mut y = vec![3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0];
    if break_pattern {
        entity.push("a".to_string());
        time.push("1".to_string());
        x.push(2.5);
        y.push(5.0);
    }
    let (columns, names) = if k == 0 {
        (vec![], vec![])
    } else {
        (vec![x], vec!["x".to_string()])
    };
    let input =
        FeInput::from_columns(&y, &columns, names, &entity, Some(&time), "y".into()).unwrap();
    (input, time)
}

/// `input`が持つ時点（ラベルの辞書順）を、DKの時点列としてそのまま渡す。時点のコードを辞書順の
/// `lexicographic`で作り直すので、`input`の時点がラベルの辞書順に作られている（`from_columns`）ことが前提。
pub(super) fn dk_time_from(input: &FeInput) -> TimeKeys {
    let Some(time) = input.time_periods() else {
        panic!("test input must have a time column");
    };
    // 辞書順のコードは、ラベルを辞書順に並べたものから作った`lexicographic`と同じ。
    TimeKeys::lexicographic(input.time_codes().map_or_else(Vec::new, |codes| {
        codes
            .codes()
            .iter()
            .map(|&code| time[code].clone())
            .collect()
    }))
}

pub(super) fn dk_bandwidth_zero(input: &FeInput) -> FeCovType {
    FeCovType::Dk {
        bandwidth: Some(0),
        time: dk_time_from(input),
    }
}

/// エンティティ2つ×5時点・`k=1`。2-wayではtime方向の各水準（時点）が2観測になる。
pub(super) fn two_entity_input() -> FeInput {
    let entity = strings(&["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]);
    let time = strings(&["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"]);
    let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 1.0, 4.0, 3.0, 6.0];
    let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5];
    FeInput::from_columns(
        &y,
        &[x],
        vec!["x".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap()
}
