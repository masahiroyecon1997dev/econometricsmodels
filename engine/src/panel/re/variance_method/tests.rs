//! `variance_method.rs`（REの分散成分推定法）のテスト。観点別に子モジュールへ分ける。
//!
//! - `validation.rs`: 入力検証・エラー返却
//! - `reference.rs`: plmの閉形式との照合・振り分け
//! - `proptests.rs`: property-basedテスト

use crate::panel::common::PanelError;
use crate::panel::re::input::ReEffects;
use crate::panel::re::input::ReInput;
use crate::panel::re::variance::VarianceComponents;
use crate::panel::re::variance_method::VarianceMethod;
use crate::panel::re::variance_method::variance_components;

mod proptests;
mod reference;
mod validation;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

const Y: [f64; 24] = [
    2.973, 0.102, -4.005, -1.69, 4.512, 2.697, 1.044, 0.361, 0.457, -1.063, -3.188, -3.244, 2.213,
    0.875, -0.479, -2.389, 5.936, 5.314, 2.163, -0.585, 2.081, 0.631, -1.679, -4.541,
];

const X1: [f64; 24] = [
    -0.828, -0.348, -1.538, -0.256, -1.15, 0.012, -0.223, 0.888, -0.592, -0.656, -0.683, -0.016,
    -0.443, 0.353, 0.073, 0.007, -0.188, -0.766, -0.221, -0.984, -1.104, -0.938, 0.679, -1.577,
];

const XI: [f64; 6] = [-0.87, 0.485, -0.186, 1.546, -0.611, -0.348];

const XT: [f64; 4] = [-1.637, 0.02, 0.892, -0.873];

/// 6エンティティ×4時点の手作りパネル（`plm`の参照値はこの値から作った）。`rows`は使う行
/// （0始まり）、`time`なら2-wayの時点も付ける。列は`names`（`x1`・`xi`・`xt`）。
pub(super) fn input(rows: &[usize], names: &[&str], two_way: bool) -> ReInput {
    let pick = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { rows.iter().map(|&r| f(r)).collect() };
    let columns: Vec<Vec<f64>> = names
        .iter()
        .map(|n| match *n {
            "x1" => pick(&|r| X1[r]),
            "xi" => pick(&|r| XI[r / 4]),
            "xt" => pick(&|r| XT[r % 4]),
            other => panic!("unknown column {other}"),
        })
        .collect();
    let entity: Vec<String> = rows.iter().map(|r| format!("e{}", r / 4)).collect();
    let time: Vec<String> = rows.iter().map(|r| format!("t{}", r % 4)).collect();
    ReInput::from_columns(
        &pick(&|r| Y[r]),
        &columns,
        names.iter().map(|n| n.to_string()).collect(),
        &entity,
        two_way.then_some(time.as_slice()),
        "y".into(),
    )
    .unwrap()
}

pub(super) fn all_rows() -> Vec<usize> {
    (0..24).collect()
}

pub(super) fn components(
    input: &ReInput,
    effects: ReEffects,
    method: VarianceMethod,
) -> Result<VarianceComponents, PanelError> {
    variance_components(input, effects, method, 0.95)
}
