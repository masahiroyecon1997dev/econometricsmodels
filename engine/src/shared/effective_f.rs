//! Montiel Olea and Pflueger(2013)の有効F統計量とPatnaik臨界値。行列だけを受け取る。
//!
//! 内生変数が1個のときの、ロバスト(不均一分散・クラスター・自己相関)な弱操作変数の診断。
//! `(Y, X_exog, Z)`と吸収したパラメータ数`n_absorbed`を受け取り、重みつきの回帰は呼び出し側が
//! `√w`を掛けた行列(定数列を含む)を渡す。共分散は[`ScoreCovariance`]で選び、meatと小標本の係数は
//! `shared::score_covariance`が定める(Kleibergen-Paapと同じ規約)。
//!
//! 参考文献:
//! - Montiel Olea, J. L. and Pflueger, C. (2013). A Robust Test for Weak Instruments. Journal of
//!   Business & Economic Statistics, 31(3), 358-369.

use faer::{Mat, Side};

use super::linear_algebra::inverse_cholesky_factor;
use super::noncentral_chi_squared::noncentral_chi_squared_quantile;
use super::score_covariance::{ScoreCovariance, covariance_multiplier, hc_row_scales, score_meat};
use super::weak_instruments::{EXACT_FIT_TOLERANCE, rms_orthogonalized};

/// 臨界値を出す相対バイアスの閾値`τ`(Nagarバイアスの、ベンチマークに対する割合の上限)。
pub(crate) const BIAS_THRESHOLDS: [f64; 4] = [0.05, 0.10, 0.20, 0.30];

/// `Ŵ2`の最小固有値が、最大固有値のこの割合より負なら、丸め誤差ではなく不定値とみなす。
const NEGATIVE_EIGENVALUE_TOLERANCE: f64 = 1e-10;

/// Patnaik臨界値と、その計算に使った量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatnaikCriticalValue {
    /// 相対バイアスの閾値`τ`。
    pub tau: f64,
    /// 非心度の入力`x`(単純版は`1/τ`)。
    pub x: f64,
    /// 有効自由度`K_eff`(小数)。
    pub effective_df: f64,
    /// 臨界値(`χ²_{K_eff}(x·K_eff)`の上側`α`点を`K_eff`で割ったもの)。有効Fがこれを超えれば、
    /// 水準`α`で「Nagarバイアスが閾値`τ`を超える」を棄却する。
    pub critical_value: f64,
}

/// [`effective_f`]の結果。
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveF {
    /// 有効F統計量。
    pub statistic: f64,
    /// 除外操作変数の数`K`。
    pub n_instruments: usize,
    /// 臨界値の水準`α`。
    pub alpha: f64,
    /// 単純版TSLSの臨界値(`x = 1/τ`)。`τ`は[`BIAS_THRESHOLDS`]の順。
    pub simplified_tsls: [PatnaikCriticalValue; 4],
}

/// Montiel Olea-Pfluegerの有効F統計量と、単純版TSLSのPatnaik臨界値を計算する。
///
/// 定義: 外生変数で`Y`と`Z`を直交化し、`Z̃'Z̃/n = I`となるように直交正規化する(統計量はこの変換に
/// 依らない)。`π̂ = Z̃'Y/n`、`V̂2`を`π̂`の共分散の推定値(`covariance`による。`Ŵ2 = n·V̂2`)として
///
/// ```text
/// F_eff = π̂'π̂ / tr(V̂2)                                           [論文 Eq. (3)]
/// K_eff(x) = tr(Ŵ2)²(1 + 2x) / [tr(Ŵ2²) + 2x·tr(Ŵ2)·λ_max(Ŵ2)]    [同 Eq. (4), (33)]
/// 臨界値 = χ²_{K_eff}(x·K_eff)の上側α点 / K_eff                    [同 Eq. (32)]
/// ```
///
/// 単純版TSLSは`x = 1/τ`。`K_eff`は`Ŵ2`のスケールに依らない。`Classical`では`Ŵ2 = ω̂_22·I`で
/// `K_eff = K`、`F_eff`はCragg-Donaldの`F`に一致する。
///
/// - `y_endog`: 内生変数(n×1。1個だけ)、`x_exog`: 外生変数(n×k_exog。定数項を使うなら定数列を含め、
///   平均は引かず他の列と同じく偏回帰で落とす)、`z`: 除外操作変数(n×K)、`alpha`: 臨界値の水準(0〜1)。
/// - `Ŵ2`は半正定値でよい(クラスターや観測が少なくてランク落ちでも、トレースと固有値だけを使う)。
///
/// 次のとき`None`: 内生変数が1個でない、`K = 0`、`alpha`が(0,1)の外、`X`がフルランクでない、`Z`の列が
/// `X`の張る空間にほぼ入っている、`Z̃`が共線、一次段階が完全フィット、HC2・HC3でレバレッジが1、
/// `n_absorbed > 0`のHC2・HC3、クラスターが2未満、`Hac`の時間順の長さが行数と違う・`lags >= n`、
/// 自由度`n − K − k_exog − n_absorbed`が1未満、入力に非有限値がある。
pub(crate) fn effective_f(
    y_endog: &Mat<f64>,
    x_exog: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
    covariance: ScoreCovariance<'_>,
    alpha: f64,
) -> Option<EffectiveF> {
    let (n, k) = (y_endog.nrows(), z.ncols());
    debug_assert_eq!(z.nrows(), n);
    debug_assert_eq!(x_exog.nrows(), n);
    if y_endog.ncols() != 1 || k == 0 || n == 0 || !(alpha > 0.0 && alpha < 1.0) {
        return None;
    }
    let first_stage = FirstStage::new(y_endog, x_exog, z)?;
    let n_f = n as f64;
    let n_params = k + x_exog.ncols() + n_absorbed;

    // スコア`ψ_i = ũ_i e_i`(n×K)から共分散の推定値`Ŵ2 = c · meat / n`を作る。
    let psi = Mat::from_fn(n, k, |i, a| {
        *first_stage.u.get(i, a) * first_stage.residuals[i]
    });
    let (meat, n_groups) = score_meat(
        &psi,
        covariance,
        |variant| hc_row_scales(variant, &first_stage.u, x_exog, n_params, n_absorbed),
        || {
            // 共分散がクロネッカー積なら`Ŵ2 ∝ I`。`Ũ'Ũ/n = I`なので`meat = (Σe²)·I`。
            let sum_squares: f64 = first_stage.residuals.iter().map(|e| e * e).sum();
            Mat::from_fn(k, k, |a, b| if a == b { sum_squares } else { 0.0 })
        },
    )?;
    let multiplier = covariance_multiplier(covariance, n, n_params, n_groups)?;
    let w2 = Mat::from_fn(k, k, |a, b| {
        multiplier * 0.5 * (*meat.get(a, b) + *meat.get(b, a)) / n_f
    });

    let eigenvalues = psd_eigenvalues(&w2)?;
    let trace: f64 = eigenvalues.iter().sum();
    // `V̂2 = Ŵ2/n`なので`tr(V̂2) = tr(Ŵ2)/n`。
    let statistic = first_stage.coefficients.iter().map(|p| p * p).sum::<f64>() / (trace / n_f);
    if !statistic.is_finite() {
        return None;
    }

    let mut critical_values = [PatnaikCriticalValue {
        tau: 0.0,
        x: 0.0,
        effective_df: 0.0,
        critical_value: 0.0,
    }; 4];
    for (slot, tau) in critical_values.iter_mut().zip(BIAS_THRESHOLDS) {
        let x = 1.0 / tau;
        let (effective_df, critical_value) = patnaik_critical_value(&eigenvalues, x, alpha)?;
        *slot = PatnaikCriticalValue {
            tau,
            x,
            effective_df,
            critical_value,
        };
    }
    Some(EffectiveF {
        statistic,
        n_instruments: k,
        alpha,
        simplified_tsls: critical_values,
    })
}

/// 半正定値の`Ŵ2`(対称)の固有値。丸め誤差の分の負の固有値は0にそろえる。最大固有値が正でない、
/// または最小固有値が最大固有値の`-1e-10`倍より小さい(不定値)とき`None`。
fn psd_eigenvalues(w2: &Mat<f64>) -> Option<Vec<f64>> {
    let values = w2.self_adjoint_eigenvalues(Side::Lower).ok()?;
    let largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
    if !(largest.is_finite() && largest > 0.0)
        || smallest < -NEGATIVE_EIGENVALUE_TOLERANCE * largest
    {
        return None;
    }
    Some(values.into_iter().map(|v| v.max(0.0)).collect())
}

/// 有効自由度`K_eff`とPatnaik臨界値(論文 Eq. (33), (32))。`eigenvalues`は`Ŵ2`の固有値(0以上)。
/// `x`は非心度の入力、`alpha`は水準。`Ŵ2`のスケールに依らない(最大固有値で割ってから計算する)。
/// 固有値が正でないとき、分位点が求まらないときは`None`。`alpha`が1e-8より小さいと、分位点の計算の
/// 精度が水準に対して足りない(下側確率`1 − α`が1に近すぎる)。
fn patnaik_critical_value(eigenvalues: &[f64], x: f64, alpha: f64) -> Option<(f64, f64)> {
    let largest = eigenvalues.iter().copied().fold(0.0, f64::max);
    if !(largest > 0.0 && largest.is_finite() && x >= 0.0 && x.is_finite()) {
        return None;
    }
    let scaled: Vec<f64> = eigenvalues.iter().map(|v| v / largest).collect();
    let trace: f64 = scaled.iter().sum();
    let trace_of_square: f64 = scaled.iter().map(|v| v * v).sum();
    // 最大固有値は1に正規化してある。
    let effective_df = trace * trace * (1.0 + 2.0 * x) / (trace_of_square + 2.0 * x * trace);
    let quantile = noncentral_chi_squared_quantile(1.0 - alpha, effective_df, x * effective_df)?;
    Some((effective_df, quantile / effective_df))
}

/// 外生変数で直交化した内生変数と、直交正規化した操作変数、一次段階の係数と残差。
struct FirstStage {
    /// 直交正規化した操作変数(n×K。`Ũ'Ũ/n = I`)。
    u: Mat<f64>,
    /// 一次段階の係数`π̂ = Ũ'Ỹ/n`。
    coefficients: Vec<f64>,
    /// 一次段階の残差`e = Ỹ − Ũπ̂`。
    residuals: Vec<f64>,
}

impl FirstStage {
    fn new(y_endog: &Mat<f64>, x_exog: &Mat<f64>, z: &Mat<f64>) -> Option<Self> {
        let (n, k) = (y_endog.nrows(), z.ncols());
        let (y, z) = rms_orthogonalized(y_endog, x_exog, z)?;

        let n_f = n as f64;
        let gram = z.transpose() * &z;
        let q_zz = Mat::from_fn(k, k, |a, b| *gram.get(a, b) / n_f);
        // `Ũ = Z̃ C⁻ᵀ`(`Q_zz = CC'`)。`Ũ'Ũ/n = C⁻¹ Q_zz C⁻ᵀ = I`。
        let c_inv = inverse_cholesky_factor(&q_zz)?;
        let u = &z * c_inv.transpose();

        let u_y = u.transpose() * &y;
        let coefficients: Vec<f64> = (0..k).map(|a| *u_y.get(a, 0) / n_f).collect();
        let residuals: Vec<f64> = (0..n)
            .map(|i| {
                let fitted: f64 = (0..k).map(|a| *u.get(i, a) * coefficients[a]).sum();
                *y.get(i, 0) - fitted
            })
            .collect();
        // `Ỹ`の二乗平均は1なので、残差の二乗平均は`1 − R²`。Cragg-Donald・Kleibergen-Paapと同じ基準で、
        // 完全フィットなら統計量は無意味。
        let mean_square_residual = residuals.iter().map(|e| e * e).sum::<f64>() / n_f;
        if !mean_square_residual.is_finite() || mean_square_residual <= EXACT_FIT_TOLERANCE {
            return None;
        }
        if coefficients.iter().any(|p| !p.is_finite()) {
            return None;
        }
        Some(Self {
            u,
            coefficients,
            residuals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::covariance::HcVariant;
    use crate::shared::group_codes::GroupCodes;
    use crate::shared::kleibergen_paap::kleibergen_paap;
    use crate::shared::weak_instruments::cragg_donald;

    type Data = (Mat<f64>, Mat<f64>, Mat<f64>);

    /// n=12。定数・外生`x`、除外操作変数`z1,z2,z3`、内生変数`d1`。
    fn small() -> Data {
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
        let endog = Mat::from_fn(12, 1, |i, _| 0.9 * z1[i] - 0.4 * z2[i] + 0.3 * x[i] + u1[i]);
        let exog = Mat::from_fn(12, 2, |i, j| if j == 0 { 1.0 } else { x[i] });
        let z = Mat::from_fn(12, 3, |i, j| [z1[i], z2[i], z3[i]][j]);
        (endog, exog, z)
    }

    /// n=240の決定的な合成データ(異分散)。操作変数は`n_instruments`個(3または4)。
    fn large(n_instruments: usize) -> Data {
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

    fn clusters(n: usize, size: usize) -> GroupCodes {
        let labels: Vec<String> = (0..n).map(|i| format!("g{:03}", i / size)).collect();
        GroupCodes::from_labels_without_keys(&labels)
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs(),
            "actual={actual} expected={expected}"
        );
    }

    fn column(m: &Mat<f64>, j: usize) -> Mat<f64> {
        Mat::from_fn(m.nrows(), 1, |i, _| *m.get(i, j))
    }

    // 論文のTable 1: `K_eff = 1..30`行、`x = 1/τ`が3.33・5・10・20の列(上側5%点)。
    // 3.33の列は`x = 10/3`の値。
    const TABLE_1: [[f64; 4]; 30] = [
        [12.05, 15.06, 23.11, 37.42],
        [9.57, 12.17, 19.29, 32.32],
        [8.53, 10.95, 17.67, 30.13],
        [7.92, 10.23, 16.72, 28.85],
        [7.51, 9.75, 16.08, 27.98],
        [7.21, 9.40, 15.62, 27.35],
        [6.98, 9.14, 15.26, 26.86],
        [6.80, 8.92, 14.97, 26.47],
        [6.65, 8.74, 14.73, 26.15],
        [6.52, 8.59, 14.53, 25.87],
        [6.41, 8.47, 14.36, 25.64],
        [6.32, 8.36, 14.21, 25.44],
        [6.24, 8.26, 14.08, 25.26],
        [6.16, 8.17, 13.96, 25.10],
        [6.10, 8.10, 13.86, 24.96],
        [6.04, 8.03, 13.77, 24.83],
        [5.99, 7.96, 13.68, 24.71],
        [5.94, 7.91, 13.60, 24.60],
        [5.89, 7.85, 13.53, 24.50],
        [5.85, 7.80, 13.46, 24.41],
        [5.81, 7.76, 13.40, 24.33],
        [5.78, 7.72, 13.35, 24.25],
        [5.74, 7.68, 13.29, 24.18],
        [5.71, 7.64, 13.24, 24.11],
        [5.68, 7.61, 13.20, 24.05],
        [5.66, 7.57, 13.15, 23.98],
        [5.63, 7.54, 13.11, 23.93],
        [5.61, 7.51, 13.07, 23.87],
        [5.58, 7.49, 13.04, 23.82],
        [5.56, 7.46, 13.00, 23.77],
    ];

    /// `Ŵ2 = I_K`なら`K_eff = K`で、臨界値は論文の公表値(小数第2位)に一致する。
    #[test]
    fn patnaik_critical_values_reproduce_table_1_of_the_paper() {
        let xs = [10.0 / 3.0, 5.0, 10.0, 20.0];
        for (row, published) in TABLE_1.iter().enumerate() {
            let k = row + 1;
            for (x, expected) in xs.iter().zip(published) {
                let (df, cv) = patnaik_critical_value(&vec![1.0; k], *x, 0.05).unwrap();
                assert!((df - k as f64).abs() < 1e-12);
                assert!(
                    (cv - expected).abs() <= 0.0051,
                    "K={k} x={x}: {cv} vs published {expected}"
                );
            }
        }
    }

    /// 固有値(3, 1)の`Ŵ2`: `tr = 4`、`tr(Ŵ2²) = 10`、`λ_max = 3`なので、`x = 10`で
    /// `K_eff = 16·21/(10 + 20·4·3) = 1.344`。臨界値の期待値はscipyの`ncx2.ppf`。
    #[test]
    fn effective_degrees_of_freedom_and_critical_value_follow_the_formula() {
        let (df, cv) = patnaik_critical_value(&[3.0, 1.0], 10.0, 0.05).unwrap();
        close(df, 1.344, 1e-12);
        close(cv, 21.295336706396494, 1e-9);
        let (df, cv) = patnaik_critical_value(&[3.0, 1.0], 3.0, 0.05).unwrap();
        close(df, 1.3658536585365855, 1e-12);
        close(cv, 10.219559398553013, 1e-9);
        // 固有値のスケールには依らない。
        let (df_scaled, cv_scaled) = patnaik_critical_value(&[300.0, 100.0], 10.0, 0.05).unwrap();
        close(df_scaled, 1.344, 1e-12);
        close(cv_scaled, 21.295336706396494, 1e-9);
    }

    /// 任意の半正定値の`Ŵ2`で`1 ≤ K_eff ≤ K`。
    #[test]
    fn effective_degrees_of_freedom_lies_between_one_and_the_number_of_instruments() {
        let spectra: [&[f64]; 4] = [
            &[1.0, 1.0, 1.0],
            &[10.0, 1.0, 0.1],
            &[5.0, 0.0, 0.0],
            &[2.0, 1.0, 1.0, 0.5],
        ];
        for eigenvalues in spectra {
            for x in [0.0, 3.0, 10.0, 20.0] {
                let (df, _) = patnaik_critical_value(eigenvalues, x, 0.05).unwrap();
                assert!(
                    df >= 1.0 - 1e-12 && df <= eigenvalues.len() as f64 + 1e-12,
                    "{df}"
                );
            }
        }
        assert!(patnaik_critical_value(&[0.0, 0.0], 10.0, 0.05).is_none());
    }

    /// 等分散なら`F_eff`はCragg-Donaldの`F`、`K_eff`は操作変数の数。
    #[test]
    fn classical_effective_f_equals_cragg_donald() {
        for (endog, exog, z) in [small(), large(3), large(4)] {
            for n_absorbed in [0, 1, 3] {
                let cd = cragg_donald(&endog, &exog, &z, n_absorbed).unwrap();
                let result = effective_f(
                    &endog,
                    &exog,
                    &z,
                    n_absorbed,
                    ScoreCovariance::Classical,
                    0.05,
                )
                .unwrap();
                close(result.statistic, cd, 1e-10);
                for cv in result.simplified_tsls {
                    close(cv.effective_df, z.ncols() as f64, 1e-10);
                }
            }
        }
    }

    /// 操作変数が1個なら`Ŵ2`はスカラーで、`F_eff`は頑健な`t²`(Kleibergen-PaapのWald F)に一致する
    /// (独立に書いた統計量どうしの照合)。
    #[test]
    fn with_one_instrument_it_equals_the_kleibergen_paap_wald_f() {
        let order: Vec<usize> = (0..240).collect();
        let groups = clusters(240, 12);
        let (endog, exog, z) = large(4);
        let z1 = column(&z, 0);
        let covariances = [
            ScoreCovariance::Classical,
            ScoreCovariance::Hc(HcVariant::Hc0),
            ScoreCovariance::Hc(HcVariant::Hc1),
            ScoreCovariance::Hc(HcVariant::Hc2),
            ScoreCovariance::Hc(HcVariant::Hc3),
            ScoreCovariance::Cluster(&groups),
            ScoreCovariance::Hac {
                lags: 4,
                order: &order,
            },
        ];
        for cov in covariances {
            let kp = kleibergen_paap(&endog, &exog, &z1, 0, cov)
                .unwrap()
                .rk_wald_f
                .unwrap();
            let eff = effective_f(&endog, &exog, &z1, 0, cov, 0.05).unwrap();
            close(eff.statistic, kp, 1e-9);
            for cv in eff.simplified_tsls {
                close(cv.effective_df, 1.0, 1e-12);
            }
        }
    }

    /// 異分散のデータでは、ロバスト版の`K_eff`は`K`より小さくなる。
    #[test]
    fn robust_effective_degrees_of_freedom_falls_below_the_number_of_instruments() {
        let (endog, exog, z) = large(4);
        let eff = effective_f(
            &endog,
            &exog,
            &z,
            0,
            ScoreCovariance::Hc(HcVariant::Hc1),
            0.05,
        )
        .unwrap();
        for cv in eff.simplified_tsls {
            assert!(
                cv.effective_df < 4.0 - 1e-6 && cv.effective_df >= 1.0,
                "{}",
                cv.effective_df
            );
        }
    }

    /// `τ`が小さい(バイアスに厳しい)ほど臨界値は大きく、`α`が小さいほど大きい。
    #[test]
    fn critical_values_increase_with_stricter_thresholds_and_levels() {
        let (endog, exog, z) = large(4);
        let cov = ScoreCovariance::Hc(HcVariant::Hc1);
        let at = |alpha| effective_f(&endog, &exog, &z, 0, cov, alpha).unwrap();
        let (loose, strict) = (at(0.10), at(0.05));
        for window in strict.simplified_tsls.windows(2) {
            assert!(window[0].critical_value > window[1].critical_value);
            assert!(window[0].tau < window[1].tau);
        }
        for (a, b) in loose.simplified_tsls.iter().zip(strict.simplified_tsls) {
            assert!(b.critical_value > a.critical_value);
            assert_eq!(a.x, b.x);
        }
        assert_eq!(strict.n_instruments, 4);
        assert_eq!(strict.alpha, 0.05);
        let taus: Vec<f64> = strict.simplified_tsls.iter().map(|c| c.tau).collect();
        assert_eq!(taus, BIAS_THRESHOLDS);
        for cv in strict.simplified_tsls {
            close(cv.x, 1.0 / cv.tau, 1e-15);
        }
    }

    /// 列のスケール・操作変数の正則な線形結合・行の並べ替えで、統計量も臨界値も変わらない。
    #[test]
    fn it_is_invariant_to_scaling_recombination_and_row_order() {
        let (endog, exog, z) = large(4);
        let groups = clusters(240, 12);
        let endog_scaled = Mat::from_fn(240, 1, |i, _| *endog.get(i, 0) * 1000.0);
        let exog_scaled = Mat::from_fn(240, 2, |i, j| *exog.get(i, j) * [3.0, 1e-3][j]);
        let z_mixed = Mat::from_fn(240, 4, |i, j| match j {
            0 => *z.get(i, 0) * 50.0 + *z.get(i, 1) - 2.0 * *z.get(i, 2),
            1 => *z.get(i, 1) * 0.01,
            2 => *z.get(i, 2) + 0.5 * *z.get(i, 0),
            _ => *z.get(i, 3) - *z.get(i, 1),
        });
        for cov in [
            ScoreCovariance::Classical,
            ScoreCovariance::Hc(HcVariant::Hc3),
            ScoreCovariance::Cluster(&groups),
        ] {
            let base = effective_f(&endog, &exog, &z, 0, cov, 0.05).unwrap();
            let other = effective_f(&endog_scaled, &exog_scaled, &z_mixed, 0, cov, 0.05).unwrap();
            close(other.statistic, base.statistic, 1e-8);
            for (a, b) in other.simplified_tsls.iter().zip(base.simplified_tsls) {
                close(a.effective_df, b.effective_df, 1e-8);
                close(a.critical_value, b.critical_value, 1e-8);
            }
        }
        // 行の並べ替え(HC・クラスターは行の並びに依らない)。
        let rows: Vec<usize> = (0..240).map(|i| (i * 7 + 3) % 240).collect();
        let permute = |m: &Mat<f64>| Mat::from_fn(240, m.ncols(), |i, j| *m.get(rows[i], j));
        let labels: Vec<String> = rows.iter().map(|&r| format!("g{:03}", r / 12)).collect();
        let permuted_groups = GroupCodes::from_labels_without_keys(&labels);
        let a = effective_f(
            &endog,
            &exog,
            &z,
            0,
            ScoreCovariance::Cluster(&groups),
            0.05,
        )
        .unwrap();
        let b = effective_f(
            &permute(&endog),
            &permute(&exog),
            &permute(&z),
            0,
            ScoreCovariance::Cluster(&permuted_groups),
            0.05,
        )
        .unwrap();
        close(b.statistic, a.statistic, 1e-10);
    }

    /// 個体ダミーを外生変数に入れた計算(`n_absorbed = 0`)と、個体ごとに平均除去して吸収した個体の数を
    /// `n_absorbed`で渡した計算は一致する(Frisch-Waugh-Lovell)。
    #[test]
    fn dummies_in_the_regressors_equal_the_demeaned_data_with_absorbed_parameters() {
        let (endog, exog, z) = large(4);
        let groups = clusters(240, 12);
        let order: Vec<usize> = (0..240).collect();
        let demean = |m: &Mat<f64>| {
            Mat::from_fn(240, m.ncols(), |i, j| {
                let start = i / 12 * 12;
                *m.get(i, j) - (start..start + 12).map(|r| *m.get(r, j)).sum::<f64>() / 12.0
            })
        };
        let x = column(&exog, 1);
        let dummies = Mat::from_fn(240, 21, |i, j| {
            if j < 20 {
                f64::from(u8::from(i / 12 == j))
            } else {
                *x.get(i, 0)
            }
        });
        let (endog_w, x_w, z_w) = (demean(&endog), demean(&x), demean(&z));
        for cov in [
            ScoreCovariance::Classical,
            ScoreCovariance::Hc(HcVariant::Hc0),
            ScoreCovariance::Hc(HcVariant::Hc1),
            ScoreCovariance::Cluster(&groups),
            ScoreCovariance::Hac {
                lags: 4,
                order: &order,
            },
        ] {
            let with_dummies = effective_f(&endog, &dummies, &z, 0, cov, 0.05).unwrap();
            let within = effective_f(&endog_w, &x_w, &z_w, 20, cov, 0.05).unwrap();
            close(within.statistic, with_dummies.statistic, 1e-8);
            for (a, b) in within
                .simplified_tsls
                .iter()
                .zip(with_dummies.simplified_tsls)
            {
                close(a.effective_df, b.effective_df, 1e-8);
                close(a.critical_value, b.critical_value, 1e-8);
            }
        }
    }

    #[test]
    fn it_is_undefined_when_the_statistic_cannot_be_formed() {
        let (endog, exog, z) = small();
        let classical = ScoreCovariance::Classical;
        let eff = effective_f;
        // 内生変数が2個・0個、操作変数が0個。
        let two = Mat::from_fn(12, 2, |i, j| {
            *endog.get(i, 0) * [1.0, 2.0][j] + j as f64 * (i as f64)
        });
        assert!(eff(&two, &exog, &z, 0, classical, 0.05).is_none());
        assert!(eff(&Mat::<f64>::zeros(12, 0), &exog, &z, 0, classical, 0.05).is_none());
        assert!(eff(&endog, &exog, &Mat::<f64>::zeros(12, 0), 0, classical, 0.05).is_none());
        // 水準が範囲外。
        for alpha in [0.0, 1.0, -0.1, f64::NAN] {
            assert!(eff(&endog, &exog, &z, 0, classical, alpha).is_none());
        }
        // 自由度が残らない(12 − 3 − 2 − 7 = 0)。
        assert!(eff(&endog, &exog, &z, 7, classical, 0.05).is_none());
        // HC2・HC3は吸収したパラメータがあると未定義。
        for variant in [HcVariant::Hc2, HcVariant::Hc3] {
            let cov = ScoreCovariance::Hc(variant);
            assert!(eff(&endog, &exog, &z, 1, cov, 0.05).is_none());
            assert!(eff(&endog, &exog, &z, 0, cov, 0.05).is_some());
        }
        // 操作変数が外生変数の線形結合・操作変数どうしが共線。
        let in_span = Mat::from_fn(12, 3, |i, j| {
            if j == 0 {
                *exog.get(i, 1) * 2.0 + 1.0
            } else {
                *z.get(i, j)
            }
        });
        assert!(eff(&endog, &exog, &in_span, 0, classical, 0.05).is_none());
        let collinear = Mat::from_fn(12, 3, |i, j| {
            if j == 2 {
                *z.get(i, 0) * 3.0
            } else {
                *z.get(i, j)
            }
        });
        assert!(eff(&endog, &exog, &collinear, 0, classical, 0.05).is_none());
        // 一次段階が完全フィット。
        let perfect = Mat::from_fn(12, 1, |i, _| {
            2.0 * *z.get(i, 0) - *z.get(i, 1) + *exog.get(i, 1)
        });
        assert!(eff(&perfect, &exog, &z, 0, classical, 0.05).is_none());
        // 非有限値。
        let mut nan_endog = endog.clone();
        *nan_endog.get_mut(3, 0) = f64::NAN;
        assert!(eff(&nan_endog, &exog, &z, 0, classical, 0.05).is_none());
        // クラスターが1つ・行数の合わないクラスター・時間順。
        let one = GroupCodes::from_labels_without_keys(&vec!["a".to_string(); 12]);
        assert!(eff(&endog, &exog, &z, 0, ScoreCovariance::Cluster(&one), 0.05).is_none());
        let short = [0usize, 1, 2];
        assert!(
            eff(
                &endog,
                &exog,
                &z,
                0,
                ScoreCovariance::Hac {
                    lags: 1,
                    order: &short
                },
                0.05
            )
            .is_none()
        );
    }

    /// 実装と別の道筋の手計算(K=4、異分散): 外生変数の偏回帰は正規方程式、直交正規化は対称平方根
    /// `Q^{-1/2} = V Λ^{-1/2} V'`、共分散は定義どおりの和。生のスケールのまま与える。
    #[test]
    fn it_matches_a_direct_computation_with_a_symmetric_square_root() {
        use faer::prelude::Solve;
        let (endog, exog, z) = large(4);
        let (n, k) = (240usize, 4usize);
        let n_f = n as f64;
        let partial_out = |m: &Mat<f64>| {
            let beta = (exog.transpose() * &exog)
                .partial_piv_lu()
                .solve(exog.transpose() * m);
            m - &exog * beta
        };
        let (y, zt) = (partial_out(&endog), partial_out(&z));
        let q = Mat::from_fn(k, k, |a, b| {
            (0..n).map(|i| *zt.get(i, a) * *zt.get(i, b)).sum::<f64>() / n_f
        });
        let eigen = q.self_adjoint_eigen(Side::Lower).unwrap();
        let (values, vectors) = (eigen.S().column_vector(), eigen.U());
        let inv_root = Mat::<f64>::from_fn(k, k, |a, b| {
            (0..k)
                .map(|c| *vectors.get(a, c) * *vectors.get(b, c) / values.get(c).sqrt())
                .sum::<f64>()
        });
        let u = &zt * &inv_root;
        let pi: Vec<f64> = (0..k)
            .map(|a| (0..n).map(|i| *u.get(i, a) * *y.get(i, 0)).sum::<f64>() / n_f)
            .collect();
        let e: Vec<f64> = (0..n)
            .map(|i| *y.get(i, 0) - (0..k).map(|a| *u.get(i, a) * pi[a]).sum::<f64>())
            .collect();
        let pi_sq: f64 = pi.iter().map(|p| p * p).sum();

        // 共分散`V2 = c/n² · meat`を定義どおりに作り、`F_eff`と`K_eff`を比べる。
        let groups = clusters(n, 12);
        let l = (k + 2) as f64;
        let hc0_meat = Mat::<f64>::from_fn(k, k, |a, b| {
            (0..n)
                .map(|i| e[i] * e[i] * *u.get(i, a) * *u.get(i, b))
                .sum::<f64>()
        });
        let cluster_meat = Mat::from_fn(k, k, |a, b| {
            (0..20)
                .map(|g| {
                    let sum = |c: usize| {
                        (12 * g..12 * g + 12)
                            .map(|i| *u.get(i, c) * e[i])
                            .sum::<f64>()
                    };
                    sum(a) * sum(b)
                })
                .sum()
        });
        let cluster_c = 20.0 / 19.0 * (n_f - 1.0) / (n_f - l);
        for (covariance, meat, c) in [
            (ScoreCovariance::Hc(HcVariant::Hc0), &hc0_meat, 1.0),
            (ScoreCovariance::Cluster(&groups), &cluster_meat, cluster_c),
        ] {
            let w2 = Mat::from_fn(k, k, |a, b| c * *meat.get(a, b) / n_f);
            let trace: f64 = (0..k).map(|a| *w2.get(a, a)).sum();
            let expected_f = pi_sq / (trace / n_f);
            let eigenvalues = w2.self_adjoint_eigenvalues(Side::Lower).unwrap();
            let (expected_df, expected_cv) =
                patnaik_critical_value(&eigenvalues, 10.0, 0.05).unwrap();

            let result = effective_f(&endog, &exog, &z, 0, covariance, 0.05).unwrap();
            close(result.statistic, expected_f, 1e-9);
            close(result.simplified_tsls[1].effective_df, expected_df, 1e-9);
            close(result.simplified_tsls[1].critical_value, expected_cv, 1e-9);
        }
    }

    #[test]
    fn psd_eigenvalues_clamps_rounding_noise_and_rejects_indefinite_matrices() {
        let diagonal = |a: f64, b: f64| {
            Mat::from_fn(2, 2, |i, j| {
                if i != j {
                    0.0
                } else if i == 0 {
                    a
                } else {
                    b
                }
            })
        };
        assert_eq!(
            psd_eigenvalues(&diagonal(1.0, -1e-14)).map(|v| v[0].max(v[1])),
            Some(1.0)
        );
        let clamped = psd_eigenvalues(&diagonal(1.0, -1e-14)).unwrap();
        assert!(clamped.iter().all(|v| *v >= 0.0));
        assert!(psd_eigenvalues(&diagonal(1.0, -0.5)).is_none());
        assert!(psd_eigenvalues(&diagonal(0.0, 0.0)).is_none());
        assert!(psd_eigenvalues(&diagonal(f64::NAN, 1.0)).is_none());
    }

    /// 自由度が残らないモデルは、自由度を使わないHC0でも未定義(Cragg-Donald・Kleibergen-Paapと同じ)。
    #[test]
    fn a_model_without_residual_degrees_of_freedom_is_undefined_for_every_covariance() {
        let (endog, exog, z) = small();
        // n = 12, K = 3, k_exog = 2: n_absorbed = 7 で n_params = n。
        for cov in [
            ScoreCovariance::Classical,
            ScoreCovariance::Hc(HcVariant::Hc0),
            ScoreCovariance::Hc(HcVariant::Hc1),
        ] {
            assert!(effective_f(&endog, &exog, &z, 7, cov, 0.05).is_none());
            assert!(effective_f(&endog, &exog, &z, 6, cov, 0.05).is_some());
        }
    }

    #[test]
    fn infinite_inputs_and_unusable_lags_are_undefined() {
        let (endog, exog, z) = small();
        let classical = ScoreCovariance::Classical;
        let mut infinite = endog.clone();
        *infinite.get_mut(2, 0) = f64::INFINITY;
        assert!(effective_f(&infinite, &exog, &z, 0, classical, 0.05).is_none());
        let mut neg_infinite = z.clone();
        *neg_infinite.get_mut(5, 1) = f64::NEG_INFINITY;
        assert!(effective_f(&endog, &exog, &neg_infinite, 0, classical, 0.05).is_none());
        let order: Vec<usize> = (0..12).collect();
        let hac = |lags| ScoreCovariance::Hac {
            lags,
            order: &order,
        };
        assert!(effective_f(&endog, &exog, &z, 0, hac(11), 0.05).is_some());
        assert!(effective_f(&endog, &exog, &z, 0, hac(12), 0.05).is_none());
    }

    /// 完全フィットの境界は、Cragg-Donald・Kleibergen-Paapと同じ基準(`1 − R²`が約1.4e-14)。
    #[test]
    fn the_perfect_fit_boundary_matches_cragg_donald() {
        let (_, exog, z) = small();
        let noise = [
            0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.2, 0.05, -0.45, 0.3, 0.15, -0.1,
        ];
        for (scale, expect_some) in [(1e-5, true), (1e-9, false)] {
            let endog = Mat::from_fn(12, 1, |i, _| {
                2.0 * *z.get(i, 0) - *z.get(i, 1) + *exog.get(i, 1) + scale * noise[i]
            });
            let eff = effective_f(&endog, &exog, &z, 0, ScoreCovariance::Classical, 0.05);
            let cd = cragg_donald(&endog, &exog, &z, 0);
            assert_eq!(eff.is_some(), expect_some, "scale={scale}");
            assert_eq!(cd.is_some(), expect_some, "scale={scale}");
        }
    }

    /// `Ŵ2`が特異(クラスター数が操作変数の数より少ない)でも、トレースと固有値で定義できる。
    #[test]
    fn a_singular_covariance_still_gives_a_value() {
        let (endog, exog, z) = small();
        let labels: Vec<String> = (0..12).map(|i| ["a", "b"][i / 6].to_string()).collect();
        let two = GroupCodes::from_labels_without_keys(&labels);
        let eff = effective_f(&endog, &exog, &z, 0, ScoreCovariance::Cluster(&two), 0.05).unwrap();
        assert!(eff.statistic.is_finite() && eff.statistic > 0.0);
        for cv in eff.simplified_tsls {
            assert!(cv.effective_df >= 1.0 && cv.effective_df <= 3.0);
        }
    }
}
