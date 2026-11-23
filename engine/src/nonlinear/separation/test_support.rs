//! 複数のファイルのテストが共有するフィクスチャ。

use crate::shared::linear_program::BoxBoundedLp;
use crate::shared::linear_program::LinearProgramSolver;
use crate::shared::linear_program::LpError;
use faer::Mat;

/// `has_intercept=true`の設計行列（先頭列が1）を作る。
pub(super) fn design(columns: &[&[f64]]) -> Mat<f64> {
    let n = columns[0].len();
    Mat::from_fn(n, columns.len() + 1, |i, j| {
        if j == 0 { 1.0 } else { columns[j - 1][i] }
    })
}

pub(super) fn response(values: &[f64]) -> Mat<f64> {
    Mat::from_fn(values.len(), 1, |i, _| values[i])
}

pub(super) fn quasi_complete_dummy_data(n: usize) -> (Mat<f64>, Mat<f64>) {
    // d=1の観測は全てy=1、d=0の観測はyが交互（分離しない）。xは決定的な擬似乱数。
    let d: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 37 % 101) as f64) / 50.0 - 1.0)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| if d[i] == 1.0 || i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    (design(&[&d, &x]), response(&y))
}

/// 常にエラーを返すソルバー（LPの失敗が`SeparationCheckFailed`になることの検証用）。
pub(super) struct FailingSolver;

impl LinearProgramSolver for FailingSolver {
    fn maximize(&self, _: &BoxBoundedLp<'_>) -> Result<Vec<f64>, LpError> {
        Err(LpError::IterationLimit { pivots: 1 })
    }
}
