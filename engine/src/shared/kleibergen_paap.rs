//! Kleibergen-Paap(2006)のrkランク統計量(LMとWald F)。行列だけを受け取る。
//!
//! Cragg-Donald(`weak_instruments`)と同じく`(Y, X_exog, Z)`と吸収したパラメータ数`n_absorbed`を受け取り、
//! 重みつきの回帰は呼び出し側が`√w`を掛けた行列(定数列を含む)を渡す。共分散は[`ScoreCovariance`]で
//! 選び、meatは`shared::covariance`の関数を使う。
//!
//! 参考文献:
//! - Kleibergen, F., & Paap, R. (2006). Generalized reduced rank tests using the singular value
//!   decomposition. Journal of Econometrics, 133(1), 97-126.
//! - Baum, C. F., Schaffer, M. E., & Stillman, S. (2007). Enhanced routines for instrumental
//!   variables/GMM estimation and testing. Stata Journal, 7(4), 465-506(`ivreg2`のrk Wald F)。

use faer::Mat;
use faer::Side;
use faer::prelude::SolveLstsq;

use super::covariance::{
    HcScale, HcVariant, cluster_correction, cluster_meat, hac_meat, hc_meat, hc_scales, leverages,
    xtx_inverse,
};
use super::error::MatrixKind;
use super::group_codes::GroupCodes;
use super::linear_algebra::{
    checked_col_piv_qr, ensure_well_conditioned_symmetric_matrix, forward_substitute, unit_columns,
};
use super::weak_instruments::orthogonalize_unit;

/// 一次段階のスコアの共分散の推定方法。
#[derive(Debug, Clone, Copy)]
pub(crate) enum ScoreCovariance<'a> {
    /// 等分散・無相関(スコアの共分散がクロネッカー積になる)。LMは`n·r²_min`(Andersonの正準相関の
    /// LM統計量)、Wald FはCragg-Donaldの`F`に一致する。
    Classical,
    /// 観測ごとのスコアに行スケールを掛けたHC。`Hc1`は`√(n/(n−L−c−n_absorbed))`、`Hc2`・`Hc3`は
    /// `[X_exog, Z]`のレバレッジによる(`covariance::hc_scales`)。
    Hc(HcVariant),
    /// クラスターロバスト。`groups`の長さは行数に等しい。
    Cluster(&'a GroupCodes),
    /// Bartlett核のNewey-West。`order`は`0..n`の置換(時間順)、`lags < n`。
    Hac { lags: usize, order: &'a [usize] },
}

/// カイ二乗分布に従うランク検定の統計量と自由度。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankTest {
    /// 検定統計量。
    pub statistic: f64,
    /// `χ²`分布の自由度。
    pub df: usize,
}

/// 完全フィット(最小の正準相関が1)とみなす`1 − r²_min`の下限。
const EXACT_FIT_TOLERANCE: f64 = 1.4210854715202004e-14;

/// スコアの共分散の固有値が最大固有値のこの割合以下なら特異とみなす。Wald Fのスコアは合計が0なので、
/// クラスター数がちょうど`d`のときの共分散は厳密にはランク落ちだが、丸め誤差の分だけ正の固有値が残りうる
/// (機械イプシロンの数倍では拾えないことがある)。
const SINGULAR_COVARIANCE_TOLERANCE: f64 = 1e-10;

/// `1 − h_ii`がこれ以下の行があるとHC2・HC3のスケールが定義できない。
const UNIT_LEVERAGE_TOLERANCE: f64 = 1e-10;

/// [`kleibergen_paap`]の結果。LMとWald Fは共分散の階数の条件が違い(Wald Fのスコアは合計が0になるので、
/// クラスターがLMより1つ多く要る)、片方だけ定義できることがあるので、独立に`Option`にする。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KleibergenPaapStatistics {
    /// rk LM統計量。
    pub rk_lm: Option<RankTest>,
    /// rk Wald F統計量。
    pub rk_wald_f: Option<f64>,
}

/// Kleibergen-Paapのrk LM統計量とrk Wald F統計量を、`covariance`で選んだ共分散で計算する。
/// 直交化・特異値分解などの前処理は1回だけ行う。
///
/// - `y_endog`: 内生変数(n×k)、`x_exog`: 外生変数(n×k_exog。定数項を使うなら定数列を含め、平均は
///   引かず他の列と同じく偏回帰で落とす)、`z`: 除外操作変数(n×q)。
/// - **rk LM**: 一次段階の係数行列が列フルランクでない(過小識別)という帰無仮説の検定。統計量は漸近的に
///   `χ²(q − k + 1)`。自由度補正は無い。`n_absorbed`は`Hc(Hc1)`の行スケール
///   `√(n/(n−q−k_exog−n_absorbed))`にだけ入る(`Classical`・`Hc0`・`Cluster`・`Hac`は`n_absorbed`に
///   よらない)。
/// - **rk Wald F**: Cragg-Donald F統計量の、`covariance`に頑健な版。検定ではなくp値は無い。一次段階の
///   残差からスコアを作ったWald統計量`W`を、小標本の補正を含めて`q`で割ったもの(`l = q + k_exog`):
///   - `Classical`・`Hac`: `W · (n − l − n_absorbed) / n / q`(`Classical`はCragg-Donaldの`F`に一致)
///   - `Hc(Hc1)`: `W / q`(`W`の行スケールに`√(n/(n−l−n_absorbed))`を含む。linearmodelsの`robust`と
///     同じ値)。`Hc0`・`Hc2`・`Hc3`も`W / q`(行スケールだけ)
///   - `Cluster`: `W · (n − l − n_absorbed)(G − 1) / (G (n − 1)) / q`(`G`はクラスター数)
/// - `Classical`と`Hc0`のLMは、linearmodelsの`unadjusted`・`robust`に対応する。`Hc2`・`Hc3`の
///   レバレッジは`[X_exog, Z]`だけから計算し、吸収した効果の分を含まないので、`n_absorbed > 0`では
///   未定義とする。
///
/// 前処理で定義できないとき(内生変数が無い、`q < k`、`X`がフルランクでない、`Z`の列が`X`の張る空間に
/// ほぼ入っている、`Z̃`または`Ỹ`が共線、最小の正準相関が1(完全フィット)、入力に非有限値がある)は
/// 全体が`None`。それ以外は統計量ごとに`None`になりうる: HC2・HC3でレバレッジが1、`n_absorbed > 0`の
/// HC2・HC3、クラスターが2未満、`Hac`の`lags >= n`、スコアの共分散が特異(クラスターや観測が少なすぎる)。
/// Wald Fはさらに自由度`n − l − n_absorbed`が1未満のとき`None`。
pub(crate) fn kleibergen_paap(
    y_endog: &Mat<f64>,
    x_exog: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
    covariance: ScoreCovariance<'_>,
) -> Option<KleibergenPaapStatistics> {
    let first_stage = FirstStage::new(y_endog, x_exog, z)?;
    Some(KleibergenPaapStatistics {
        rk_lm: first_stage.lm(x_exog, n_absorbed, covariance),
        rk_wald_f: first_stage.wald_f(x_exog, n_absorbed, covariance),
    })
}

/// 外生変数で直交化した`Ỹ`・`Z̃`の正準相関の分解と、最小特異値の方向。
struct FirstStage {
    n: usize,
    /// 除外操作変数の数。
    q: usize,
    /// 直交化後の`Ỹ`(n×k)・`Z̃`(n×q)。各列は二乗平均が1(統計量は列のスケールによらない)。
    y: Mat<f64>,
    z: Mat<f64>,
    /// `Ỹ`を`Z̃`に回帰した残差。
    residuals: Mat<f64>,
    /// `Θ̂ = Q_zz^{-1/2}' Q_zx Q_xx^{-1/2}`の最小特異値に属する左特異ベクトル以降の基底`A⊥`と
    /// 右特異ベクトル`B⊥`で作る`λ = A⊥'Θ̂B⊥`(長さ`q − k + 1`)。
    lam: Vec<f64>,
    /// `A⊥' Q_zz^{-1/2}'`((q − k + 1)×q)と`B⊥' Q_xx^{-1/2}'`(長さk)。観測ごとのスコア`x_i ⊗ z_i`
    /// を`λ`の方向へ射影する係数で、射影すると`(B⊥ ⊗ A⊥)'(Q_xx^{-1/2} ⊗ Q_zz^{-1/2})`のクロネッカー積が
    /// 外積に分解できる。
    z_weights: Mat<f64>,
    y_weights: Vec<f64>,
}

impl FirstStage {
    fn new(y_endog: &Mat<f64>, x_exog: &Mat<f64>, z: &Mat<f64>) -> Option<Self> {
        let (n, k, q) = (y_endog.nrows(), y_endog.ncols(), z.ncols());
        debug_assert_eq!(z.nrows(), n);
        debug_assert_eq!(x_exog.nrows(), n);
        if k == 0 || q < k || n == 0 {
            return None;
        }
        let (y_unit, z_unit) = orthogonalize_unit(y_endog, x_exog, z)?;
        // 二乗平均が1になるようにそろえる(`Q = Z̃'Z̃/n`の対角が1)。
        let root_n = (n as f64).sqrt();
        let y = Mat::from_fn(n, k, |i, j| *y_unit.get(i, j) * root_n);
        let z = Mat::from_fn(n, q, |i, j| *z_unit.get(i, j) * root_n);

        let n_f = n as f64;
        let scaled = |m: Mat<f64>| Mat::from_fn(m.nrows(), m.ncols(), |a, b| *m.get(a, b) / n_f);
        let q_zz = scaled(z.transpose() * &z);
        let q_xx = scaled(y.transpose() * &y);
        let q_zx = scaled(z.transpose() * &y);
        // 非ピボットのCholeskyは共線を検出できないので、固有値でも確かめる。
        ensure_well_conditioned_symmetric_matrix(&q_zz, q, MatrixKind::DesignGram).ok()?;
        ensure_well_conditioned_symmetric_matrix(&q_xx, k, MatrixKind::DesignGram).ok()?;
        let llt_z = q_zz.llt(Side::Lower).ok()?;
        let llt_x = q_xx.llt(Side::Lower).ok()?;
        let (c_z, c_x) = (llt_z.L(), llt_x.L());
        let c_z_inv = forward_substitute(c_z, &Mat::<f64>::identity(q, q));
        let c_x_inv = forward_substitute(c_x, &Mat::<f64>::identity(k, k));

        // Θ̂ = C_z⁻¹ Q_zx C_x⁻ᵀ (q×k)
        let theta = &c_z_inv * &q_zx * c_x_inv.transpose();
        let svd = theta.svd().ok()?;
        let (u, v) = (svd.U(), svd.V());
        let s_min = *svd.S().column_vector().get(k - 1);
        if !s_min.is_finite() || 1.0 - s_min * s_min <= EXACT_FIT_TOLERANCE {
            return None;
        }

        // 最小特異値に属する右特異ベクトル(長さk)と、それ以降の左特異ベクトル(q − k + 1本)。
        let d = q - k + 1;
        let lam: Vec<f64> = (0..d)
            .map(|a| {
                (0..q)
                    .map(|l| {
                        (0..k)
                            .map(|j| *u.get(l, k - 1 + a) * *theta.get(l, j) * *v.get(j, k - 1))
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        // A⊥' C_z⁻¹ ((q − k + 1)×q) と B⊥' C_x⁻¹ (長さk)。
        let z_weights = Mat::from_fn(d, q, |a, l| {
            (0..q)
                .map(|r| *u.get(r, k - 1 + a) * *c_z_inv.get(r, l))
                .sum()
        });
        let y_weights: Vec<f64> = (0..k)
            .map(|j| (0..k).map(|r| *v.get(r, k - 1) * *c_x_inv.get(r, j)).sum())
            .collect();

        let qr_z = checked_col_piv_qr(&z).ok()?;
        let residuals = &y - &z * qr_z.solve_lstsq(&y);
        if lam.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(Self {
            n,
            q,
            y,
            z,
            residuals,
            lam,
            z_weights,
            y_weights,
        })
    }

    fn lm(
        &self,
        x_exog: &Mat<f64>,
        n_absorbed: usize,
        covariance: ScoreCovariance<'_>,
    ) -> Option<RankTest> {
        let (statistic, _) = self.chi2(&self.y, x_exog, n_absorbed, covariance)?;
        Some(RankTest {
            statistic,
            df: self.lam.len(),
        })
    }

    fn wald_f(
        &self,
        x_exog: &Mat<f64>,
        n_absorbed: usize,
        covariance: ScoreCovariance<'_>,
    ) -> Option<f64> {
        let (n, q) = (self.n as f64, self.q);
        let n_params = q + x_exog.ncols() + n_absorbed;
        let df = self.n.checked_sub(n_params).filter(|&d| d >= 1)? as f64;
        let (wald, n_groups) = self.chi2(&self.residuals, x_exog, n_absorbed, covariance)?;
        let factor = match covariance {
            ScoreCovariance::Classical | ScoreCovariance::Hac { .. } => df / n,
            ScoreCovariance::Hc(_) => 1.0,
            ScoreCovariance::Cluster(_) => 1.0 / cluster_correction(n_groups?, self.n, n_params),
        };
        Some(wald * factor / (q as f64))
    }

    /// `scores_from`(n×k)と`Z̃`の積のスコアを`λ`の方向へ射影し、`covariance`で共分散`V̂_λ`を作って
    /// `n · λ' V̂_λ⁻¹ λ`を返す(LMは`Ỹ`、Waldは一次段階の残差)。クラスターのときはクラスター数も返す。
    fn chi2(
        &self,
        scores_from: &Mat<f64>,
        x_exog: &Mat<f64>,
        n_absorbed: usize,
        covariance: ScoreCovariance<'_>,
    ) -> Option<(f64, Option<usize>)> {
        let (n, d) = (self.n, self.lam.len());
        let n_f = n as f64;
        // ψ_ia = (v_i · B⊥'C_x⁻¹) (z̃_i · A⊥_a'C_z⁻¹)。
        let projected_y: Vec<f64> = (0..n)
            .map(|i| {
                (0..self.y_weights.len())
                    .map(|j| *scores_from.get(i, j) * self.y_weights[j])
                    .sum()
            })
            .collect();
        let psi = Mat::from_fn(n, d, |i, a| {
            let projected_z: f64 = (0..self.q)
                .map(|l| *self.z.get(i, l) * *self.z_weights.get(a, l))
                .sum();
            projected_y[i] * projected_z
        });

        let (meat, n_groups) = match covariance {
            ScoreCovariance::Classical => {
                // 共分散がクロネッカー積なら、射影後は`ω · I`(`ω = Σ_v` の方向の分散)。
                let omega = projected_y.iter().map(|p| p * p).sum::<f64>();
                (
                    Mat::from_fn(d, d, |a, b| if a == b { omega } else { 0.0 }),
                    None,
                )
            }
            ScoreCovariance::Hc(variant) => {
                let scales = self.hc_row_scales(variant, x_exog, n_absorbed)?;
                (hc_meat(&psi, &scales), None)
            }
            ScoreCovariance::Cluster(groups) => {
                if groups.nobs() != n || groups.n_groups() < 2 {
                    return None;
                }
                let (meat, g) = cluster_meat(groups, d, |i, a| *psi.get(i, a));
                (meat, Some(g))
            }
            ScoreCovariance::Hac { lags, order } => {
                if order.len() != n || lags >= n {
                    return None;
                }
                (hac_meat(n, d, lags, order, |i, a| *psi.get(i, a)), None)
            }
        };

        // V̂_λ = meat / n。対称にそろえ、固有分解で二次形式を作る。正定値でない(ランク落ち)なら
        // 一般化逆行列は使わず未定義とする。
        let v_lambda = Mat::from_fn(d, d, |a, b| 0.5 * (*meat.get(a, b) + *meat.get(b, a)) / n_f);
        let eigen = v_lambda.self_adjoint_eigen(Side::Lower).ok()?;
        let (values, vectors) = (eigen.S().column_vector(), eigen.U());
        let largest = (0..d)
            .map(|a| *values.get(a))
            .fold(f64::NEG_INFINITY, f64::max);
        let tolerance = largest * SINGULAR_COVARIANCE_TOLERANCE;
        if !largest.is_finite() || (0..d).any(|a| *values.get(a) <= tolerance) {
            return None;
        }
        let statistic = n_f
            * (0..d)
                .map(|a| {
                    let coef: f64 = (0..d).map(|b| *vectors.get(b, a) * self.lam[b]).sum();
                    coef * coef / *values.get(a)
                })
                .sum::<f64>();
        statistic.is_finite().then_some((statistic, n_groups))
    }

    /// HCの行スケール(スコア全体に掛ける)。`Hc1`は`√(n/(n−q−k_exog−n_absorbed))`、`Hc2`・`Hc3`は
    /// 一次段階の回帰子`[X_exog, Z]`のレバレッジ(`Z̃`は`X`と直交なので`h(X) + h(Z̃)`)による。
    fn hc_row_scales(
        &self,
        variant: HcVariant,
        x_exog: &Mat<f64>,
        n_absorbed: usize,
    ) -> Option<Vec<f64>> {
        let n = self.n;
        let one = |_: usize| 1.0;
        match variant {
            HcVariant::Hc0 => Some(hc_scales(n, one, HcScale::Hc0)),
            HcVariant::Hc1 => {
                let df = n
                    .checked_sub(self.q + x_exog.ncols() + n_absorbed)
                    .filter(|&d| d >= 1)?;
                let factor = ((n as f64) / (df as f64)).sqrt();
                Some(hc_scales(n, one, HcScale::Hc1 { factor }))
            }
            HcVariant::Hc2 | HcVariant::Hc3 => {
                // レバレッジは吸収した効果の分を含まないので、`n_absorbed > 0`では個体ダミーを回帰子に
                // 入れた定義と食い違う。吸収した効果のレバレッジを受け取る口ができるまでは未定義とする。
                if n_absorbed > 0 {
                    return None;
                }
                let mut leverage = leverages(&self.z, &xtx_inverse(&self.z).ok()?);
                if x_exog.ncols() > 0 {
                    let x_unit = unit_columns(x_exog);
                    let x_leverage = leverages(&x_unit, &xtx_inverse(&x_unit).ok()?);
                    for (h, h_x) in leverage.iter_mut().zip(x_leverage) {
                        *h += h_x;
                    }
                }
                if leverage
                    .iter()
                    .any(|h| h.is_nan() || 1.0 - h <= UNIT_LEVERAGE_TOLERANCE)
                {
                    return None;
                }
                let scale = if variant == HcVariant::Hc2 {
                    HcScale::Hc2 {
                        leverage: &leverage,
                    }
                } else {
                    HcScale::Hc3 {
                        leverage: &leverage,
                    }
                };
                Some(hc_scales(n, one, scale))
            }
        }
    }
}

#[cfg(test)]
mod tests;
