//! Stock-Yogo(2005)の弱操作変数の臨界値表(2SLSの相対バイアスとサイズ歪み)。
//!
//! Cragg-Donald Wald F統計量(`shared::weak_instruments::cragg_donald`)がこの値を超えたら、
//! 5%水準で「操作変数は弱い」を棄却する(`g_min`が臨界値を超えると棄却、という論文の記述と同じ)。
//! 表は内生変数の数`n`(1〜)と除外操作変数の数`K2`(1〜30)の関数で、表に無い組み合わせは`None`。
//! 等分散・系列相関なしの誤差を前提にした値で、ロバストな統計量には使えない。
//!
//! 出典: Stock, J. H. and Yogo, M. (2005), "Testing for Weak Instruments in Linear IV Regression",
//! in Andrews, D. W. K. and Stock, J. H. (eds.), *Identification and Inference for Econometric Models:
//! Essays in Honor of Thomas Rothenberg*, Cambridge University Press。値は著者の
//! ワーキングペーパー(NBER Technical Working Paper 284, 2001年8月、2002年10月改訂版)の
//! Table 1(TSLSの相対バイアス)とTable 2(TSLSのサイズ)を転記した。
//!
//! 転記の検証: 欠損セルの位置(バイアスは`K2 >= n + 2`、サイズは`K2 >= n`)、水準に対する単調性
//! (厳しい水準ほど大きい)、`K2`に対する単調性(サイズ)、内生変数の数に対する単調性(バイアス)、
//! 隣り合う`K2`の差の滑らかさ、広く引用される値(`n=1`のサイズ10%で`K2=1,2,5`が16.38・19.93・26.87、
//! バイアス5%・10%・30%で`K2=3`が13.91・9.08・5.39 など)との一致を単体テストで固定している。
//! 第二の独立な出典(Stataの`ivreg2`のソース)は取得できなかったので、全セルの独立な照合はしていない。

/// 相対バイアスの水準`b`(OLSのバイアスに対する2SLSのバイアスの上限)。表の列の順。
pub(crate) const TSLS_BIAS_LEVELS: [f64; 4] = [0.05, 0.10, 0.20, 0.30];

/// 5%のWald検定のサイズ歪みの水準`r`(検定のサイズの上限)。表の列の順。
pub(crate) const TSLS_SIZE_LEVELS: [f64; 4] = [0.10, 0.15, 0.20, 0.25];

/// 論文の表が`K2`について与える最大値。
pub(crate) const MAX_EXCLUDED_INSTRUMENTS: usize = 30;

/// 2SLSの相対バイアスの臨界値(`TSLS_BIAS_LEVELS`の順)。`n_endog`は内生変数の数(1〜3)、
/// `n_excluded`は除外操作変数の数`K2`。表に無い組み合わせ(`K2 < n + 2`、`K2 > 30`、`n > 3`)は`None`。
pub(crate) fn tsls_bias_critical_values(n_endog: usize, n_excluded: usize) -> Option<[f64; 4]> {
    let table: &[Option<[f64; 4]>; MAX_EXCLUDED_INSTRUMENTS] = match n_endog {
        1 => &BIAS_N1,
        2 => &BIAS_N2,
        3 => &BIAS_N3,
        _ => return None,
    };
    n_excluded
        .checked_sub(1)
        .and_then(|i| table.get(i).copied().flatten())
}

/// 2SLSのサイズ歪みの臨界値(`TSLS_SIZE_LEVELS`の順)。`n_endog`は内生変数の数(1〜2)、
/// `n_excluded`は除外操作変数の数`K2`。表に無い組み合わせ(`K2 < n`、`K2 > 30`、`n > 2`)は`None`。
pub(crate) fn tsls_size_critical_values(n_endog: usize, n_excluded: usize) -> Option<[f64; 4]> {
    let table: &[Option<[f64; 4]>; MAX_EXCLUDED_INSTRUMENTS] = match n_endog {
        1 => &SIZE_N1,
        2 => &SIZE_N2,
        _ => return None,
    };
    n_excluded
        .checked_sub(1)
        .and_then(|i| table.get(i).copied().flatten())
}

/// Table 1(TSLSの相対バイアス)、内生変数1個。添字は`K2 - 1`。
// 表の値は論文の数値で、`6.28`などが`TAU`に近いのは偶然。
#[allow(clippy::approx_constant)]
const BIAS_N1: [Option<[f64; 4]>; MAX_EXCLUDED_INSTRUMENTS] = [
    None,                             // K2 = 1
    None,                             // K2 = 2
    Some([13.91, 9.08, 6.46, 5.39]),  // K2 = 3
    Some([16.85, 10.27, 6.71, 5.34]), // K2 = 4
    Some([18.37, 10.83, 6.77, 5.25]), // K2 = 5
    Some([19.28, 11.12, 6.76, 5.15]), // K2 = 6
    Some([19.86, 11.29, 6.73, 5.07]), // K2 = 7
    Some([20.25, 11.39, 6.69, 4.99]), // K2 = 8
    Some([20.53, 11.46, 6.65, 4.92]), // K2 = 9
    Some([20.74, 11.49, 6.61, 4.86]), // K2 = 10
    Some([20.90, 11.51, 6.56, 4.80]), // K2 = 11
    Some([21.01, 11.52, 6.53, 4.75]), // K2 = 12
    Some([21.10, 11.52, 6.49, 4.71]), // K2 = 13
    Some([21.18, 11.52, 6.45, 4.67]), // K2 = 14
    Some([21.23, 11.51, 6.42, 4.63]), // K2 = 15
    Some([21.28, 11.50, 6.39, 4.59]), // K2 = 16
    Some([21.31, 11.49, 6.36, 4.56]), // K2 = 17
    Some([21.34, 11.48, 6.33, 4.53]), // K2 = 18
    Some([21.36, 11.46, 6.31, 4.51]), // K2 = 19
    Some([21.38, 11.45, 6.28, 4.48]), // K2 = 20
    Some([21.39, 11.44, 6.26, 4.46]), // K2 = 21
    Some([21.40, 11.42, 6.24, 4.43]), // K2 = 22
    Some([21.41, 11.41, 6.22, 4.41]), // K2 = 23
    Some([21.41, 11.40, 6.20, 4.39]), // K2 = 24
    Some([21.42, 11.38, 6.18, 4.37]), // K2 = 25
    Some([21.42, 11.37, 6.16, 4.35]), // K2 = 26
    Some([21.42, 11.36, 6.14, 4.34]), // K2 = 27
    Some([21.42, 11.34, 6.13, 4.32]), // K2 = 28
    Some([21.42, 11.33, 6.11, 4.31]), // K2 = 29
    Some([21.42, 11.32, 6.09, 4.29]), // K2 = 30
];

/// Table 1、内生変数2個。
// 表の値は論文の数値で、`6.28`などが`TAU`に近いのは偶然。
#[allow(clippy::approx_constant)]
const BIAS_N2: [Option<[f64; 4]>; MAX_EXCLUDED_INSTRUMENTS] = [
    None,                             // K2 = 1
    None,                             // K2 = 2
    None,                             // K2 = 3
    Some([11.04, 7.56, 5.57, 4.73]),  // K2 = 4
    Some([13.97, 8.78, 5.91, 4.79]),  // K2 = 5
    Some([15.72, 9.48, 6.08, 4.78]),  // K2 = 6
    Some([16.88, 9.92, 6.16, 4.76]),  // K2 = 7
    Some([17.70, 10.22, 6.20, 4.73]), // K2 = 8
    Some([18.30, 10.43, 6.22, 4.69]), // K2 = 9
    Some([18.76, 10.58, 6.23, 4.66]), // K2 = 10
    Some([19.12, 10.69, 6.23, 4.62]), // K2 = 11
    Some([19.40, 10.78, 6.22, 4.59]), // K2 = 12
    Some([19.64, 10.84, 6.21, 4.56]), // K2 = 13
    Some([19.83, 10.89, 6.20, 4.53]), // K2 = 14
    Some([19.98, 10.93, 6.19, 4.50]), // K2 = 15
    Some([20.12, 10.96, 6.17, 4.48]), // K2 = 16
    Some([20.23, 10.99, 6.16, 4.45]), // K2 = 17
    Some([20.33, 11.00, 6.14, 4.43]), // K2 = 18
    Some([20.41, 11.02, 6.13, 4.41]), // K2 = 19
    Some([20.48, 11.03, 6.11, 4.39]), // K2 = 20
    Some([20.54, 11.04, 6.10, 4.37]), // K2 = 21
    Some([20.60, 11.05, 6.08, 4.35]), // K2 = 22
    Some([20.65, 11.05, 6.07, 4.33]), // K2 = 23
    Some([20.69, 11.05, 6.06, 4.32]), // K2 = 24
    Some([20.73, 11.06, 6.05, 4.30]), // K2 = 25
    Some([20.76, 11.06, 6.03, 4.29]), // K2 = 26
    Some([20.79, 11.06, 6.02, 4.27]), // K2 = 27
    Some([20.82, 11.05, 6.01, 4.26]), // K2 = 28
    Some([20.84, 11.05, 6.00, 4.24]), // K2 = 29
    Some([20.86, 11.05, 5.99, 4.23]), // K2 = 30
];

/// Table 1、内生変数3個。
// 表の値は論文の数値で、`6.28`などが`TAU`に近いのは偶然。
#[allow(clippy::approx_constant)]
const BIAS_N3: [Option<[f64; 4]>; MAX_EXCLUDED_INSTRUMENTS] = [
    None,                             // K2 = 1
    None,                             // K2 = 2
    None,                             // K2 = 3
    None,                             // K2 = 4
    Some([9.53, 6.61, 4.99, 4.30]),   // K2 = 5
    Some([12.20, 7.77, 5.35, 4.40]),  // K2 = 6
    Some([13.95, 8.50, 5.56, 4.44]),  // K2 = 7
    Some([15.18, 9.01, 5.69, 4.46]),  // K2 = 8
    Some([16.10, 9.37, 5.78, 4.46]),  // K2 = 9
    Some([16.80, 9.64, 5.83, 4.45]),  // K2 = 10
    Some([17.35, 9.85, 5.87, 4.44]),  // K2 = 11
    Some([17.80, 10.01, 5.90, 4.42]), // K2 = 12
    Some([18.17, 10.14, 5.92, 4.41]), // K2 = 13
    Some([18.47, 10.25, 5.93, 4.39]), // K2 = 14
    Some([18.73, 10.33, 5.94, 4.37]), // K2 = 15
    Some([18.94, 10.41, 5.94, 4.36]), // K2 = 16
    Some([19.13, 10.47, 5.94, 4.34]), // K2 = 17
    Some([19.29, 10.52, 5.94, 4.32]), // K2 = 18
    Some([19.44, 10.56, 5.94, 4.31]), // K2 = 19
    Some([19.56, 10.60, 5.93, 4.29]), // K2 = 20
    Some([19.67, 10.63, 5.93, 4.28]), // K2 = 21
    Some([19.77, 10.65, 5.92, 4.27]), // K2 = 22
    Some([19.86, 10.68, 5.92, 4.25]), // K2 = 23
    Some([19.94, 10.70, 5.91, 4.24]), // K2 = 24
    Some([20.01, 10.71, 5.90, 4.23]), // K2 = 25
    Some([20.07, 10.73, 5.90, 4.21]), // K2 = 26
    Some([20.13, 10.74, 5.89, 4.20]), // K2 = 27
    Some([20.18, 10.75, 5.88, 4.19]), // K2 = 28
    Some([20.23, 10.76, 5.88, 4.18]), // K2 = 29
    Some([20.27, 10.77, 5.87, 4.17]), // K2 = 30
];

/// Table 2(TSLSのサイズ)、内生変数1個。
// 表の値は論文の数値で、`6.28`などが`TAU`に近いのは偶然。
#[allow(clippy::approx_constant)]
const SIZE_N1: [Option<[f64; 4]>; MAX_EXCLUDED_INSTRUMENTS] = [
    Some([16.38, 8.96, 6.66, 5.53]),    // K2 = 1
    Some([19.93, 11.59, 8.75, 7.25]),   // K2 = 2
    Some([22.30, 12.83, 9.54, 7.80]),   // K2 = 3
    Some([24.58, 13.96, 10.26, 8.31]),  // K2 = 4
    Some([26.87, 15.09, 10.98, 8.84]),  // K2 = 5
    Some([29.18, 16.23, 11.72, 9.38]),  // K2 = 6
    Some([31.50, 17.38, 12.48, 9.93]),  // K2 = 7
    Some([33.84, 18.54, 13.24, 10.50]), // K2 = 8
    Some([36.19, 19.71, 14.01, 11.07]), // K2 = 9
    Some([38.54, 20.88, 14.78, 11.65]), // K2 = 10
    Some([40.90, 22.06, 15.56, 12.23]), // K2 = 11
    Some([43.27, 23.24, 16.35, 12.82]), // K2 = 12
    Some([45.64, 24.42, 17.14, 13.41]), // K2 = 13
    Some([48.01, 25.61, 17.93, 14.00]), // K2 = 14
    Some([50.39, 26.80, 18.72, 14.60]), // K2 = 15
    Some([52.77, 27.99, 19.51, 15.19]), // K2 = 16
    Some([55.15, 29.19, 20.31, 15.79]), // K2 = 17
    Some([57.53, 30.38, 21.10, 16.39]), // K2 = 18
    Some([59.92, 31.58, 21.90, 16.99]), // K2 = 19
    Some([62.30, 32.77, 22.70, 17.60]), // K2 = 20
    Some([64.69, 33.97, 23.50, 18.20]), // K2 = 21
    Some([67.07, 35.17, 24.30, 18.80]), // K2 = 22
    Some([69.46, 36.37, 25.10, 19.41]), // K2 = 23
    Some([71.85, 37.57, 25.90, 20.01]), // K2 = 24
    Some([74.24, 38.77, 26.71, 20.61]), // K2 = 25
    Some([76.62, 39.97, 27.51, 21.22]), // K2 = 26
    Some([79.01, 41.17, 28.31, 21.83]), // K2 = 27
    Some([81.40, 42.37, 29.12, 22.43]), // K2 = 28
    Some([83.79, 43.57, 29.92, 23.04]), // K2 = 29
    Some([86.17, 44.78, 30.72, 23.65]), // K2 = 30
];

/// Table 2、内生変数2個。
// 表の値は論文の数値で、`6.28`などが`TAU`に近いのは偶然。
#[allow(clippy::approx_constant)]
const SIZE_N2: [Option<[f64; 4]>; MAX_EXCLUDED_INSTRUMENTS] = [
    None,                               // K2 = 1
    Some([7.03, 4.58, 3.95, 3.63]),     // K2 = 2
    Some([13.43, 8.18, 6.40, 5.45]),    // K2 = 3
    Some([16.87, 9.93, 7.54, 6.28]),    // K2 = 4
    Some([19.45, 11.22, 8.38, 6.89]),   // K2 = 5
    Some([21.68, 12.33, 9.10, 7.42]),   // K2 = 6
    Some([23.72, 13.34, 9.77, 7.91]),   // K2 = 7
    Some([25.64, 14.31, 10.41, 8.39]),  // K2 = 8
    Some([27.51, 15.24, 11.03, 8.85]),  // K2 = 9
    Some([29.32, 16.16, 11.65, 9.31]),  // K2 = 10
    Some([31.11, 17.06, 12.25, 9.77]),  // K2 = 11
    Some([32.88, 17.95, 12.86, 10.22]), // K2 = 12
    Some([34.62, 18.84, 13.45, 10.68]), // K2 = 13
    Some([36.36, 19.72, 14.05, 11.13]), // K2 = 14
    Some([38.08, 20.60, 14.65, 11.58]), // K2 = 15
    Some([39.80, 21.48, 15.24, 12.03]), // K2 = 16
    Some([41.51, 22.35, 15.83, 12.49]), // K2 = 17
    Some([43.22, 23.22, 16.42, 12.94]), // K2 = 18
    Some([44.92, 24.09, 17.02, 13.39]), // K2 = 19
    Some([46.62, 24.96, 17.61, 13.84]), // K2 = 20
    Some([48.31, 25.82, 18.20, 14.29]), // K2 = 21
    Some([50.01, 26.69, 18.79, 14.74]), // K2 = 22
    Some([51.70, 27.56, 19.38, 15.19]), // K2 = 23
    Some([53.39, 28.42, 19.97, 15.64]), // K2 = 24
    Some([55.07, 29.29, 20.56, 16.10]), // K2 = 25
    Some([56.76, 30.15, 21.15, 16.55]), // K2 = 26
    Some([58.45, 31.02, 21.74, 17.00]), // K2 = 27
    Some([60.13, 31.88, 22.33, 17.45]), // K2 = 28
    Some([61.82, 32.74, 22.92, 17.90]), // K2 = 29
    Some([63.51, 33.61, 23.51, 18.35]), // K2 = 30
];

#[cfg(test)]
mod tests {
    use super::*;

    fn bias(n: usize, k2: usize) -> Option<[f64; 4]> {
        tsls_bias_critical_values(n, k2)
    }
    fn size(n: usize, k2: usize) -> Option<[f64; 4]> {
        tsls_size_critical_values(n, k2)
    }

    /// 広く引用される値(論文のTable 1・Table 2)。
    #[test]
    fn it_returns_the_published_values_for_well_known_cells() {
        assert_eq!(size(1, 1), Some([16.38, 8.96, 6.66, 5.53]));
        assert_eq!(size(1, 2).map(|v| v[0]), Some(19.93));
        assert_eq!(size(1, 5).map(|v| v[0]), Some(26.87));
        assert_eq!(size(2, 3).map(|v| v[0]), Some(13.43));
        assert_eq!(bias(1, 3), Some([13.91, 9.08, 6.46, 5.39]));
        assert_eq!(bias(1, 30).map(|v| v[0]), Some(21.42));
        assert_eq!(bias(2, 4).map(|v| v[0]), Some(11.04));
        assert_eq!(bias(3, 5).map(|v| v[0]), Some(9.53));
    }

    #[test]
    fn it_has_values_only_where_the_paper_tabulates_them() {
        for k2 in 1..=MAX_EXCLUDED_INSTRUMENTS {
            for n in 1..=3 {
                assert_eq!(bias(n, k2).is_some(), k2 >= n + 2, "bias n={n} K2={k2}");
            }
            for n in 1..=2 {
                assert_eq!(size(n, k2).is_some(), k2 >= n, "size n={n} K2={k2}");
            }
        }
        // 表の外。
        assert_eq!(bias(4, 10), None);
        assert_eq!(size(3, 10), None);
        assert_eq!(bias(0, 10), None);
        assert_eq!(size(1, 0), None);
        assert_eq!(size(1, 31), None);
        assert_eq!(bias(1, 31), None);
    }

    /// 厳しい水準ほど臨界値は大きい(行の中で単調に減る)。
    #[test]
    fn critical_values_decrease_as_the_tolerated_distortion_grows() {
        for k2 in 1..=MAX_EXCLUDED_INSTRUMENTS {
            for n in 1..=3 {
                if let Some(v) = bias(n, k2) {
                    assert!(
                        v.windows(2).all(|w| w[0] > w[1]),
                        "bias n={n} K2={k2} {v:?}"
                    );
                }
            }
            for n in 1..=2 {
                if let Some(v) = size(n, k2) {
                    assert!(
                        v.windows(2).all(|w| w[0] > w[1]),
                        "size n={n} K2={k2} {v:?}"
                    );
                }
            }
        }
    }

    /// サイズの臨界値は除外操作変数が増えるほど大きい。バイアスは、内生変数が増えるほど小さい。
    #[test]
    fn critical_values_are_monotone_in_the_number_of_instruments_and_endogenous_variables() {
        for n in 1..=2 {
            for k2 in (n + 1)..=MAX_EXCLUDED_INSTRUMENTS {
                let (lo, hi) = (size(n, k2 - 1).unwrap(), size(n, k2).unwrap());
                assert!((0..4).all(|c| hi[c] > lo[c]), "size n={n} K2={k2}");
            }
        }
        for k2 in 5..=MAX_EXCLUDED_INSTRUMENTS {
            let (a, b, c) = (
                bias(1, k2).unwrap(),
                bias(2, k2).unwrap(),
                bias(3, k2).unwrap(),
            );
            assert!((0..4).all(|i| a[i] > b[i] && b[i] > c[i]), "bias K2={k2}");
        }
    }

    /// 転記ミス(1桁の誤り)を拾うため、隣り合う`K2`の差の変化(2階差分)が滑らかであることを確かめる。
    /// 序盤の数点は曲率が大きいので許容を緩め、`K2`が大きい領域ほど絞る(実測の最大は、
    /// サイズが`K2 >= 11`で0.03・`K2 >= 16`で0.01、バイアスが`K2 >= 10`で0.15・`K2 >= 16`で0.05・
    /// `K2 >= 21`で0.01)。
    #[test]
    fn critical_values_vary_smoothly_in_the_number_of_instruments() {
        let second_difference = |f: &dyn Fn(usize) -> Option<[f64; 4]>, k2: usize, c: usize| {
            f(k2 - 1).unwrap()[c] - 2.0 * f(k2).unwrap()[c] + f(k2 + 1).unwrap()[c]
        };
        for c in 0..4 {
            for k2 in 6..MAX_EXCLUDED_INSTRUMENTS {
                let tolerance = match k2 {
                    ..=10 => 0.25,
                    11..=15 => 0.05,
                    _ => 0.02,
                };
                for n in 1..=2 {
                    let d = second_difference(&|k| size(n, k), k2, c);
                    assert!(d.abs() < tolerance, "size n={n} c={c} K2={k2} d2={d}");
                }
            }
            for k2 in 10..MAX_EXCLUDED_INSTRUMENTS {
                let tolerance = match k2 {
                    ..=15 => 0.2,
                    16..=20 => 0.07,
                    _ => 0.02,
                };
                for n in 1..=3 {
                    let d = second_difference(&|k| bias(n, k), k2, c);
                    assert!(d.abs() < tolerance, "bias n={n} c={c} K2={k2} d2={d}");
                }
            }
        }
    }
}
