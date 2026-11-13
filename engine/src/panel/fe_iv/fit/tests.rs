//! `fit`のテスト。テーマ別に子モジュールへ分ける。
//!
//! - `reference.rs`: fixestの`feols(y ~ x1 | 固定効果 | x2 ~ z1 + z2)`との照合（`data.rs`のデータ）
//! - `cov_type.rs`: Driscoll-Kraay（時点でクラスタリングした共分散との恒等式、自由度の境界）
//! - `lsdv.rs`: ダミー変数を明示した2SLS（LSDV）との照合（FWLにより、within変換した2SLSと係数・
//!   共分散が一致する）。不均衡な2-wayと3次元もここで確かめる
//! - `sargan.rs`: Sargan過剰識別検定（ダミー変数を明示した2SLSでの再計算、`σ̂²`の分母、帰無仮説下のサイズ）
//! - `structure.rs`: 構造残差・R²・固定効果・第一段階・不変性
//! - `wu_hausman.rs`: Wu-Hausman（fixestの補助回帰との照合、ダミー込みの補助回帰での再計算、`None`になる条件）
//! - `validation.rs`: 入力検証・エラー返却・検証順序

use crate::panel::common::PanelError;
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::{FeIvEstimator, FeIvInput, RoleColumns};
use crate::shared::group_codes::GroupCodes;

mod cov_type;
mod data;
mod lsdv;
mod reference;
mod sargan;
mod structure;
mod validation;
mod wu_hausman;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

const N_ENTITIES: usize = 10;
const N_PERIODS: usize = 4;

fn labels(values: impl IntoIterator<Item = usize>) -> Vec<String> {
    values.into_iter().map(|v| v.to_string()).collect()
}

/// 個体（行は個体ごと）。
pub(super) fn entity_labels() -> Vec<String> {
    labels((0..N_ENTITIES).flat_map(|i| std::iter::repeat_n(i, N_PERIODS)))
}

/// 時点。
pub(super) fn time_labels() -> Vec<String> {
    labels((0..N_ENTITIES).flat_map(|_| 0..N_PERIODS))
}

/// 個体と時点の両方と交わる3つ目の次元（3水準）。
pub(super) fn third_labels() -> Vec<String> {
    labels((0..N_ENTITIES).flat_map(|i| (0..N_PERIODS).map(move |t| (i + 2 * t) % 3)))
}

/// 追加の列（外生変数2つ目、内生変数2つ目、操作変数3つ目）。値は決定的な疑似データ。
fn extra_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let exog2 = (0..40).map(|r| (r as f64 * 0.9).sin()).collect();
    let endog2 = (0..40)
        .map(|r| 0.6 * data::X2[r] + 0.4 * data::Z1[r] + 0.5 * (r as f64 * 1.3).cos())
        .collect();
    let instrument3 = (0..40)
        .map(|r| 0.3 * data::X1[r] + (r as f64 * 2.1).sin())
        .collect();
    (exog2, endog2, instrument3)
}

/// `rows`の行だけを使った入力（外生変数1個、内生変数1個、操作変数2個）。`dimensions`は効果の各次元の
/// ラベル（全行分）。
pub(super) fn input_with(rows: &[usize], dimensions: &[Vec<String>]) -> FeIvInput {
    input_spec(rows, dimensions, 1, 1, 2)
}

/// 外生変数`k_exog`（0〜2）・内生変数`k_endog`（1〜2）・操作変数`k_inst`（2〜3）の入力。
/// 列の並びは外生`x1`・`w`、内生`x2`・`x3`、操作変数`z1`・`z2`・`z3`の先頭から。
pub(super) fn input_spec(
    rows: &[usize],
    dimensions: &[Vec<String>],
    k_exog: usize,
    k_endog: usize,
    k_inst: usize,
) -> FeIvInput {
    let pick = |column: &[f64]| -> Vec<f64> { rows.iter().map(|&r| column[r]).collect() };
    let (exog2, endog2, instrument3) = extra_columns();
    let exog: Vec<Vec<f64>> = [pick(&data::X1), pick(&exog2)]
        .into_iter()
        .take(k_exog)
        .collect();
    let endog: Vec<Vec<f64>> = [pick(&data::X2), pick(&endog2)]
        .into_iter()
        .take(k_endog)
        .collect();
    let instruments: Vec<Vec<f64>> = [pick(&data::Z1), pick(&data::Z2), pick(&instrument3)]
        .into_iter()
        .take(k_inst)
        .collect();
    let names = |all: &[&str], n: usize| all.iter().take(n).map(|s| s.to_string()).collect();
    let groups = dimensions
        .iter()
        .map(|labels| {
            let picked: Vec<String> = rows.iter().map(|&r| labels[r].clone()).collect();
            GroupCodes::from_labels(&picked)
        })
        .collect();
    FeIvInput::from_group_codes(
        &pick(&data::Y),
        "y".to_string(),
        RoleColumns {
            names: names(&["x1", "w"], k_exog),
            columns: &exog,
        },
        RoleColumns {
            names: names(&["x2", "x3"], k_endog),
            columns: &endog,
        },
        RoleColumns {
            names: names(&["z1", "z2", "z3"], k_inst),
            columns: &instruments,
        },
        groups,
    )
    .unwrap()
}

pub(super) fn all_rows() -> Vec<usize> {
    (0..N_ENTITIES * N_PERIODS).collect()
}

/// 全行の1-way（個体）入力。
pub(super) fn one_way_input() -> FeIvInput {
    input_with(&all_rows(), &[entity_labels()])
}

/// 全行の2-way（個体・時点）入力。
pub(super) fn two_way_input() -> FeIvInput {
    input_with(&all_rows(), &[entity_labels(), time_labels()])
}

pub(super) fn fit(
    input: FeIvInput,
    effects: FeEffects,
    cov_type: FeCovType,
) -> Result<FeIvEstimator, PanelError> {
    FeIvEstimator::fit(input, effects, cov_type, 0.95)
}

pub(super) fn entity_cluster() -> FeCovType {
    FeCovType::Cluster { groups: None }
}
