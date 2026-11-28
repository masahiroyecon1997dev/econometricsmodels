//! 分離検出のLPに渡す行。LPが行に求める操作（[`SeparationRows`]）と、2値モデルの行
//! （[`SignedRows`]。標準化後の符号付き設計行列の行を、行列を複製せず元の`x`と`1/std`から都度計算する）。

use faer::Mat;
use std::borrow::Cow;

use crate::nonlinear::common::ColumnScale;

/// 分離のLP（`lp.rs`）が制約の行に求める操作。行は単位長にそろえたベクトル`uᵣ`で、制約は
/// `uᵣ'θ ≥ 0`。行の番号は`margins`の添字と同じで、無効な行（零ベクトル・存在しない行）は
/// `is_active`が偽を返し、制約にもマージンの判定にも使わない。
pub(super) trait SeparationRows {
    /// 行`r`が制約になるか。
    fn is_active(&self, r: usize) -> bool;

    /// 全ての行の`uᵣ'θ`（無効な行は`0.0`）。
    fn margins(&self, theta: &[f64]) -> Vec<f64>;

    /// 行`r`の単位ベクトル`uᵣ`。有効な行（`is_active(r)`が真）にだけ呼ぶ。
    fn unit_row(&self, r: usize) -> Vec<f64>;

    /// LPの目的関数の係数`c = Σᵣ uᵣ`（有効な行の和）。
    fn objective(&self) -> Vec<f64>;
}

/// 設計行列の第`j`列の値。`Mat`は列方向に連続しているので通常は借用でき、要素ごとの境界チェックを
/// 避けられる（行列を走査する各パスが`O(nk)`で、大標本ではこのコストが目立つ）。連続でない場合
/// （通常は起きない）だけコピーする。
pub(super) fn column_values(x: &Mat<f64>, j: usize) -> Cow<'_, [f64]> {
    match x.col(j).try_as_col_major() {
        Some(col) => Cow::Borrowed(col.as_slice()),
        None => Cow::Owned((0..x.nrows()).map(|i| *x.get(i, j)).collect()),
    }
}

/// 標準化空間の符号付き設計行列の行`zᵢ = (2yᵢ-1)·x_stdᵢ`を、単位長にそろえて扱う。
/// 標準化後の行列は持たず、`x`・`1/std`から都度計算する。
pub(super) struct SignedRows<'a> {
    x: &'a Mat<f64>,
    inv_std: Vec<f64>,
    /// `(2yᵢ-1)/‖x_stdᵢ‖`。零ベクトルの行（制約にならない）は`0.0`。
    row_scale: Vec<f64>,
}

impl<'a> SignedRows<'a> {
    pub(super) fn new(x: &'a Mat<f64>, scale: &ColumnScale, y: &Mat<f64>) -> Self {
        let n = x.nrows();
        let k = x.ncols();
        let inv_std: Vec<f64> = scale.stds().iter().map(|s| 1.0 / s).collect();
        let mut sq_norm = vec![0.0_f64; n];
        for (j, &inv) in inv_std.iter().enumerate().take(k) {
            for (acc, &xv) in sq_norm.iter_mut().zip(column_values(x, j).iter()) {
                let v = xv * inv;
                *acc += v * v;
            }
        }
        let row_scale = sq_norm
            .iter()
            .enumerate()
            .map(|(i, &sq)| {
                let norm = sq.sqrt();
                if norm > 0.0 && norm.is_finite() {
                    let sign = if *y.get(i, 0) > 0.5 { 1.0 } else { -1.0 };
                    sign / norm
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            x,
            inv_std,
            row_scale,
        }
    }

    fn n(&self) -> usize {
        self.x.nrows()
    }

    fn k(&self) -> usize {
        self.x.ncols()
    }
}

impl SeparationRows for SignedRows<'_> {
    fn is_active(&self, i: usize) -> bool {
        self.row_scale[i] != 0.0
    }

    /// 各行の`uᵢ'θ`（単位長の行と`θ`の内積）。零ベクトルの行は`0.0`。
    fn margins(&self, theta: &[f64]) -> Vec<f64> {
        let mut acc = vec![0.0_f64; self.n()];
        for (j, (&inv, &t)) in self.inv_std.iter().zip(theta).enumerate() {
            let w = inv * t;
            if w == 0.0 {
                continue;
            }
            for (a, &xv) in acc.iter_mut().zip(column_values(self.x, j).iter()) {
                *a += xv * w;
            }
        }
        for (a, &s) in acc.iter_mut().zip(&self.row_scale) {
            *a *= s;
        }
        acc
    }

    /// 行`i`の単位ベクトル`uᵢ`。
    fn unit_row(&self, i: usize) -> Vec<f64> {
        (0..self.k())
            .map(|j| *self.x.get(i, j) * self.inv_std[j] * self.row_scale[i])
            .collect()
    }

    /// LPの目的関数の係数`c = Σᵢ uᵢ`。
    fn objective(&self) -> Vec<f64> {
        (0..self.k())
            .map(|j| {
                let sum: f64 = column_values(self.x, j)
                    .iter()
                    .zip(&self.row_scale)
                    .map(|(&xv, &s)| xv * s)
                    .sum();
                sum * self.inv_std[j]
            })
            .collect()
    }
}
