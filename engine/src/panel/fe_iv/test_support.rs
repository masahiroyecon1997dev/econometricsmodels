//! `fe_iv`のテストが共有するデータと入力の組み立て（`fit/tests/`と`gmm/fit/tests/`が使う）。
//!
//! 10個体×4時点の40行（行は個体ごと）。値は`data.rs`（fixestの参照値を作ったデータ）。

use super::input::{FeIvInput, RoleColumns};
use crate::shared::group_codes::GroupCodes;

pub(super) mod data;

pub(super) const N_ENTITIES: usize = 10;
pub(super) const N_PERIODS: usize = 4;

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

/// 次元ごとのダミー列（1つ目の次元は全水準、2つ目以降は最初の水準を落とす）。
pub(super) fn dummies(dimensions: &[GroupCodes]) -> Vec<Vec<f64>> {
    let mut columns = Vec::new();
    for (d, dimension) in dimensions.iter().enumerate() {
        for level in usize::from(d > 0)..dimension.n_groups() {
            columns.push(
                dimension
                    .codes()
                    .iter()
                    .map(|&code| f64::from(u8::from(code == level)))
                    .collect(),
            );
        }
    }
    columns
}
