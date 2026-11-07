//! `effective_f`のテスト。複数のテーマが使うデータと照合用のヘルパーをここに置く。

use super::*;
use crate::shared::covariance::HcVariant;
use crate::shared::group_codes::GroupCodes;
use crate::shared::weak_instruments::cragg_donald;

mod bias_bound;
mod critical_values;
mod invariance;
mod reference;
mod undefined;

pub(super) type Data = (Mat<f64>, Mat<f64>, Mat<f64>);

/// 被説明変数なし(単純版だけ)の呼び出し。
pub(super) fn effective_f(
    y_endog: &Mat<f64>,
    x_exog: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
    covariance: ScoreCovariance<'_>,
    alpha: f64,
) -> Option<EffectiveF> {
    super::effective_f(y_endog, x_exog, z, n_absorbed, covariance, alpha, None)
}

pub(super) const SMALL_X: [f64; 12] = [
    0.3, -1.2, 0.8, 2.0, -0.5, 1.1, 0.7, -0.4, 1.6, 0.2, -1.0, 0.9,
];
pub(super) const SMALL_Z1: [f64; 12] = [
    1.0, 0.2, -0.7, 1.5, 0.9, -1.1, 0.3, 0.8, -0.6, 1.2, -0.3, 0.5,
];
pub(super) const SMALL_Z2: [f64; 12] = [
    0.4, -0.9, 1.3, 0.1, -0.2, 0.7, -1.4, 0.6, 0.0, -0.8, 1.1, 0.35,
];
pub(super) const SMALL_Z3: [f64; 12] = [
    -0.5, 0.7, 0.2, -1.1, 0.8, 0.4, 1.0, -0.3, 0.9, -0.6, 0.15, -0.2,
];
pub(super) const SMALL_U1: [f64; 12] = [
    0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.2, 0.05, -0.45, 0.3, 0.15, -0.1,
];
pub(super) const SMALL_U2: [f64; 12] = [
    -0.3, 0.4, 0.5, -0.2, 0.1, -0.6, 0.35, 0.2, -0.1, -0.4, 0.25, 0.3,
];

/// n=12。定数・外生`x`、除外操作変数`z1,z2,z3`、内生変数`Y = 0.9 z1 − 0.4 z2 + 0.3 x + u1`。
pub(super) fn small() -> Data {
    let endog = Mat::from_fn(12, 1, |i, _| {
        0.9 * SMALL_Z1[i] - 0.4 * SMALL_Z2[i] + 0.3 * SMALL_X[i] + SMALL_U1[i]
    });
    let exog = Mat::from_fn(12, 2, |i, j| if j == 0 { 1.0 } else { SMALL_X[i] });
    let z = Mat::from_fn(12, 3, |i, j| [SMALL_Z1[i], SMALL_Z2[i], SMALL_Z3[i]][j]);
    (endog, exog, z)
}

/// n=240の決定的な合成データ(異分散)。操作変数は`n_instruments`個(3または4)。
pub(super) fn large(n_instruments: usize) -> Data {
    let n = 240;
    let x = |i: usize| (0.7 * i as f64 + 0.3).sin() + 0.5 * (0.11 * i as f64).cos();
    let z = |i: usize, l: usize| {
        let (t, l) = (i as f64, l as f64);
        (1.3 * t * (l + 1.0) + l).sin() + 0.3 * (0.37 * t * (l + 2.0)).cos()
    };
    let h = |i: usize| 1.0 + 0.8 * (0.05 * i as f64).sin().abs();
    let endog = Mat::from_fn(n, 1, |i, _| {
        let fourth = if n_instruments == 4 {
            0.1 * z(i, 3)
        } else {
            0.0
        };
        0.5 * z(i, 0) + 0.3 * z(i, 1) - 0.2 * z(i, 2)
            + fourth
            + 0.4 * x(i)
            + (2.1 * i as f64 + 1.0).sin() * h(i)
    });
    let exog = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x(i) });
    let zs = Mat::from_fn(n, n_instruments, z);
    (endog, exog, zs)
}

pub(super) fn clusters(n: usize, size: usize) -> GroupCodes {
    let labels: Vec<String> = (0..n).map(|i| format!("g{:03}", i / size)).collect();
    GroupCodes::from_labels_without_keys(&labels)
}

pub(super) fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs(),
        "actual={actual} expected={expected}"
    );
}

pub(super) fn column(m: &Mat<f64>, j: usize) -> Mat<f64> {
    Mat::from_fn(m.nrows(), 1, |i, _| *m.get(i, j))
}

/// 被説明変数(n=240): `y = Y + 0.5x + cos(0.9 i)·h_i`。
pub(super) fn large_y(endog: &Mat<f64>, exog: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(240, 1, |i, _| {
        let h = 1.0 + 0.8 * (0.05 * i as f64).sin().abs();
        *endog.get(i, 0) + 0.5 * *exog.get(i, 1) + (0.9 * i as f64).cos() * h
    })
}
