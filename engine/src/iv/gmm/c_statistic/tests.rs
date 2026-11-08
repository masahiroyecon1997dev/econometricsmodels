//! `c_statistic.rs`（C統計量）のテスト。観点別に子モジュールへ分ける。
//!
//! - `reference.rs`: 独立な参照値との照合（内生変数の外生性はlinearmodelsの`c_stat`と、操作変数の
//!   直交性はそれに合わせて書いたnumpy版との照合。生成に使ったデータは`data.rs`）
//! - `validation.rs`: 入力検証・エラー返却・未定義
//! - `invariants.rs`: 恒等式・不変性

use crate::iv::common::IvInput;
use crate::iv::gmm::WeightType;
use crate::shared::group_codes::GroupCodes;

mod data;
mod invariants;
mod reference;
mod validation;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

pub(super) const N: usize = 24;

/// `x_exog=[x1]`（定数項あり）、`x_endog=[d1, d2]`、`instruments=[z1, z2, z3]`。
pub(super) fn input() -> IvInput {
    input_with_instruments(&[0, 1, 2])
}

/// `instruments`のうち`positions`の列だけを使う入力。
pub(super) fn input_with_instruments(positions: &[usize]) -> IvInput {
    let all = [("z1", data::Z1), ("z2", data::Z2), ("z3", data::Z3)];
    let columns: Vec<Vec<f64>> = positions.iter().map(|&p| all[p].1.to_vec()).collect();
    let names: Vec<String> = positions.iter().map(|&p| all[p].0.to_string()).collect();
    IvInput::from_columns(
        &data::Y,
        &[data::X1.to_vec()],
        vec!["x1".to_string()],
        &[data::D1.to_vec(), data::D2.to_vec()],
        vec!["d1".to_string(), "d2".to_string()],
        &columns,
        names,
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// `x_exog=[x1]`、`x_endog=[d1]`、`instruments=[z1, z2, z3, z4]`。検定する操作変数を外しても
/// 過剰識別のままになる（除外操作変数4本に対して内生変数1個）。
pub(super) fn input_one_endog(include_intercept: bool) -> IvInput {
    IvInput::from_columns(
        &data::Y,
        &[data::X1.to_vec()],
        vec!["x1".to_string()],
        &[data::D1.to_vec()],
        vec!["d1".to_string()],
        &[
            data::Z1.to_vec(),
            data::Z2.to_vec(),
            data::Z3.to_vec(),
            data::Z4.to_vec(),
        ],
        ["z1", "z2", "z3", "z4"].map(String::from).to_vec(),
        include_intercept,
        "y".to_string(),
    )
    .unwrap()
}

/// `n_groups`個のクラスター（行番号を`n_groups`で割った余り）。
pub(super) fn cluster_weight(n_groups: usize) -> WeightType {
    let labels: Vec<String> = (0..N).map(|i| format!("g{}", i % n_groups)).collect();
    WeightType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&labels)),
    }
}

/// ラグ2のNewey-West。時間順は行番号。
pub(super) fn hac_weight() -> WeightType {
    WeightType::Hac {
        lags: Some(2),
        time_order: (0..N).map(|i| i as f64).collect(),
    }
}
