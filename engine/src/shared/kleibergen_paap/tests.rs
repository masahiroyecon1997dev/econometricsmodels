//! `kleibergen_paap`のテスト。複数のテーマが使うデータと照合用のヘルパーをここに置く。

use super::*;
use crate::shared::covariance::HcVariant;
use crate::shared::group_codes::GroupCodes;

mod absorbed;
mod invariance;
mod reference;
mod undefined;

/// n=12。定数・外生`x`(`k_exog=2`)、除外操作変数`z1,z2,z3`(`q=3`)、内生変数`d1,d2`。
pub(super) fn data() -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let x = [
        0.3, -1.2, 0.8, 2.0, -0.5, 1.1, 0.7, -0.4, 1.6, 0.2, -1.0, 0.9,
    ];
    let z1 = [
        1.0, 0.2, -0.7, 1.5, 0.9, -1.1, 0.3, 0.8, -0.6, 1.2, -0.3, 0.5,
    ];
    let z2 = [
        0.4, -0.9, 1.3, 0.1, -0.2, 0.7, -1.4, 0.6, 0.0, -0.8, 1.1, 0.35,
    ];
    let z3 = [
        -0.5, 0.7, 0.2, -1.1, 0.8, 0.4, 1.0, -0.3, 0.9, -0.6, 0.15, -0.2,
    ];
    let u1 = [
        0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.2, 0.05, -0.45, 0.3, 0.15, -0.1,
    ];
    let u2 = [
        -0.3, 0.4, 0.5, -0.2, 0.1, -0.6, 0.35, 0.2, -0.1, -0.4, 0.25, 0.3,
    ];
    let exog = Mat::from_fn(12, 2, |i, j| if j == 0 { 1.0 } else { x[i] });
    let z = Mat::from_fn(12, 3, |i, j| [z1[i], z2[i], z3[i]][j]);
    let y = Mat::from_fn(12, 2, |i, j| {
        if j == 0 {
            0.9 * z1[i] - 0.4 * z2[i] + 0.3 * x[i] + u1[i]
        } else {
            0.2 * z1[i] + 0.7 * z3[i] - 0.2 * x[i] + u2[i]
        }
    });
    (y, exog, z)
}

pub(super) fn first_column(m: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(m.nrows(), 1, |i, _| *m.get(i, 0))
}

pub(super) fn four_clusters() -> GroupCodes {
    let labels: Vec<String> = ["a", "b", "c", "d"]
        .iter()
        .flat_map(|l| std::iter::repeat_n(l.to_string(), 3))
        .collect();
    GroupCodes::from_labels_without_keys(&labels)
}

pub(super) fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs(),
        "actual={actual} expected={expected}"
    );
}

pub(super) fn kleibergen_paap_lm(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
    cov: ScoreCovariance<'_>,
) -> Option<RankTest> {
    kleibergen_paap(y, x, z, n_absorbed, cov)?.rk_lm
}

pub(super) fn kleibergen_paap_f(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
    cov: ScoreCovariance<'_>,
) -> Option<f64> {
    kleibergen_paap(y, x, z, n_absorbed, cov)?.rk_wald_f
}

pub(super) fn lm(
    y: &Mat<f64>,
    exog: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
    cov: ScoreCovariance<'_>,
) -> RankTest {
    kleibergen_paap_lm(y, exog, z, n_absorbed, cov).unwrap()
}

pub(super) fn f(
    y: &Mat<f64>,
    exog: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
    cov: ScoreCovariance<'_>,
) -> f64 {
    kleibergen_paap_f(y, exog, z, n_absorbed, cov).unwrap()
}

pub(super) fn covariance<'a>(
    name: &str,
    groups: &'a GroupCodes,
    order: &'a [usize],
) -> ScoreCovariance<'a> {
    match name {
        "classical" => ScoreCovariance::Classical,
        "hc0" => ScoreCovariance::Hc(HcVariant::Hc0),
        "hc1" => ScoreCovariance::Hc(HcVariant::Hc1),
        "hc2" => ScoreCovariance::Hc(HcVariant::Hc2),
        "hc3" => ScoreCovariance::Hc(HcVariant::Hc3),
        "cluster" => ScoreCovariance::Cluster(groups),
        "hac" => ScoreCovariance::Hac { lags: 2, order },
        _ => unreachable!(),
    }
}

/// n=240の決定的な合成データ(内生変数2個・除外操作変数4個・外生変数は定数と`x`)。Pythonの`numpy`で
/// 同じ式から作り、linearmodels 7.1で計算した値と照合する。小標本の12行では丸め誤差が見えにくいので、
/// 大きめのデータでも照合する。
pub(super) fn large_sample() -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let n = 240;
    let x: Vec<f64> = (0..n)
        .map(|i| (0.7 * i as f64 + 0.3).sin() + 0.5 * (0.11 * i as f64).cos())
        .collect();
    let z = Mat::from_fn(n, 4, |i, l| {
        let (t, l) = (i as f64, l as f64);
        (1.3 * t * (l + 1.0) + l).sin() + 0.3 * (0.37 * t * (l + 2.0)).cos()
    });
    let scale = |i: usize| 1.0 + 0.8 * (0.05 * i as f64).sin().abs();
    let e1 = |i: usize| (2.1 * i as f64 + 1.0).sin() * scale(i);
    let e2 = |i: usize| (1.7 * i as f64 + 0.4).cos() * scale(i);
    let y = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            0.5 * z.get(i, 0) + 0.3 * z.get(i, 1) - 0.2 * z.get(i, 2) + 0.4 * x[i] + e1(i)
        } else {
            0.1 * z.get(i, 0) + 0.4 * z.get(i, 3) + 0.2 * z.get(i, 2) - 0.3 * x[i]
                + e2(i)
                + 0.3 * e1(i)
        }
    });
    let exog = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x[i] });
    (y, exog, z)
}
