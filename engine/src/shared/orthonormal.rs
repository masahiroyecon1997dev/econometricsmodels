//! ピボットなしQR分解 `X = QR` による、直交座標での推論の部品。
//!
//! サンドイッチ共分散`V = B M B`（`B = (X'X)⁻¹`）を作ってから反転すると、傾きの相関が極端に
//! 高いとき`B`の明示的な逆行列で桁落ちし、`V`が数値的に正定値でなくなる。`X = QR`とすれば
//! `β = R⁻¹Q'y`、`V = R⁻¹ M̃ R⁻ᵀ`で、`M̃`は`x_i`を`Q`の`i`行目に置き換えた直交座標のミート
//! （`Q'Q = I`なので条件数は`Ω`の条件数で決まり、`X`の条件数に依らない）。末尾`q`個の係数の
//! Wald統計量は`β_S' V_SS⁻¹ β_S = b̃_S' M̃_SS⁻¹ b̃_S`（`b̃ = Q'y = Rβ`、`R⁻¹`が上三角なので
//! `β_S`は`b̃_S`だけで決まる）。`V`を作らず、`M̃`の末尾ブロックだけを反転する。
//!
//! `M̃`は既存のmeat構築部品（[`super::covariance`]の`hc_cov_params`・`cluster_cov_params`・
//! `hac_cov_params`等）に`x = Q`・`xtx_inv = 単位行列`を渡せば得られる。小標本補正はスカラー倍
//! なのでそのまま効き、HC2/HC3のレバレッジは`‖q_i‖²`になる。統計量は`b̃`と`M̃`を
//! [`super::wald::wald_quadratic_form`]に渡せばよい（相関スケール化と悪条件の判定も`M̃`の
//! 末尾ブロックに対して働く）。
//!
//! **ピボットなしにする理由**: 列を並べ替えると「検定対象が末尾`q`成分」という性質が崩れる。
//! 列を入れ替えない上三角の`R`なら、切片が先頭・検定対象が末尾という既存の列の並びで
//! そのまま使える。ランク落ちの判定は、呼び出し側の列ピボットQR（`checked_col_piv_qr`）が
//! 先に行うか（[`OrthonormalDesign::new`]。ここでは対角が厳密に0または非有限のときだけ防御的に
//! 弾く）、`R`（k×k）の列ピボットQRで行う（[`OrthonormalDesign::new_checked`]。`X = QR`の
//! `R`は`X`と同じ特異値を持つので、n×kの列ピボットQRを別に走らせずに済む）。

use faer::{Mat, MatRef};

use super::linear_algebra::{RankDeficient, checked_col_piv_qr};

/// 全体Wald検定に使う直交座標の組。推定量が保持し、末尾`q`列の検定にも使う。
#[derive(Debug, Clone)]
pub(crate) struct WaldCoordinates {
    /// 係数の直交座標 `b̃ = Q'y`（k×1）。
    pub(crate) coordinates: Mat<f64>,
    /// 直交座標のミート`M̃`（k×k、小標本補正込み）。
    pub(crate) meat: Mat<f64>,
}

/// `X = QR`（`Q`はn×k、`R`はk×kの上三角）。
#[derive(Debug)]
pub(crate) struct OrthonormalDesign {
    q: Mat<f64>,
    r: Mat<f64>,
}

impl OrthonormalDesign {
    /// `x`（n×k）をピボットなしのHouseholder QRで分解する。`k = 0`（列が無い`x`、固定効果のみの
    /// モデルのwithin変換後の設計行列）は空の`Q`・`R`を返す。
    ///
    /// # Errors
    /// 行数が列数に満たない、または`R`の対角が厳密に0・非有限のとき[`RankDeficient::Columns`]。
    /// 数値的なランク落ちの判定ではない（呼び出し側の列ピボットQRが先に行う）。
    pub(crate) fn new(x: &Mat<f64>) -> Result<Self, RankDeficient> {
        Self::decompose(x, false)
    }

    /// [`Self::new`]に加えて、`R`（k×k）の列ピボットQRでランク落ちを判定する
    /// （`checked_col_piv_qr`と同じ閾値）。`Q`を作る前に判定するので、ランク落ちでは`Q`を作らない。
    ///
    /// # Errors
    /// `x`がフルカラムランクでないとき[`RankDeficient::Columns`]。
    pub(crate) fn new_checked(x: &Mat<f64>) -> Result<Self, RankDeficient> {
        Self::decompose(x, true)
    }

    fn decompose(x: &Mat<f64>, check_rank: bool) -> Result<Self, RankDeficient> {
        let (n, k) = (x.nrows(), x.ncols());
        if n < k {
            return Err(RankDeficient::Columns);
        }
        if k == 0 {
            return Ok(Self {
                q: Mat::zeros(n, 0),
                r: Mat::zeros(0, 0),
            });
        }
        let qr = x.qr();
        let r = qr.thin_R().to_owned();
        if (0..k).any(|i| {
            let diagonal = *r.get(i, i);
            !diagonal.is_finite() || diagonal == 0.0
        }) {
            return Err(RankDeficient::Columns);
        }
        if check_rank {
            checked_col_piv_qr(&r)?;
        }
        Ok(Self {
            q: qr.compute_thin_Q(),
            r,
        })
    }

    /// 直交化した設計行列`Q`（n×k）。meat構築部品に`x`として渡す。
    pub(crate) fn q(&self) -> &Mat<f64> {
        &self.q
    }

    /// k×kの単位行列。meat構築部品に`xtx_inv`として渡す（`Q'Q = I`）。
    pub(crate) fn identity(&self) -> Mat<f64> {
        Mat::identity(self.r.nrows(), self.r.nrows())
    }

    /// 係数の直交座標 `b̃ = Q'y`（= `Rβ̂`、k×1）。
    pub(crate) fn coordinates(&self, y: &Mat<f64>) -> Mat<f64> {
        self.q.transpose() * y
    }

    /// 係数 `β̂ = R⁻¹ b̃`（k×1）。`b̃`は[`Self::coordinates`]の値。
    pub(crate) fn coefficients(&self, coordinates: &Mat<f64>) -> Mat<f64> {
        solve_upper(&self.r, coordinates.as_ref())
    }

    /// 残差 `y - Q b̃`（n×1）。`y - Xβ̂`と同じ値で、`β̂`を介さない。
    pub(crate) fn residuals(&self, y: &Mat<f64>, coordinates: &Mat<f64>) -> Mat<f64> {
        y - &self.q * coordinates
    }

    /// 係数の共分散 `V = R⁻¹ M̃ R⁻ᵀ`（k×k）。`m_tilde`は対称なk×kの直交座標のミート。
    /// 丸めによるわずかな非対称は、上下三角の平均で取り除く。
    pub(crate) fn coefficient_covariance(&self, m_tilde: &Mat<f64>) -> Mat<f64> {
        let left = solve_upper(&self.r, m_tilde.as_ref());
        let v = solve_upper(&self.r, left.transpose());
        Mat::from_fn(v.nrows(), v.ncols(), |i, j| {
            0.5 * (*v.get(i, j) + *v.get(j, i))
        })
    }
}

/// 上三角`r`（k×k）で`r · x = b`を解く（後退代入。`b`はk×m）。
fn solve_upper(r: &Mat<f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let k = r.nrows();
    let mut x = Mat::<f64>::zeros(k, b.ncols());
    for c in 0..b.ncols() {
        for i in (0..k).rev() {
            let tail: f64 = (i + 1..k).map(|j| *r.get(i, j) * *x.get(j, c)).sum();
            *x.get_mut(i, c) = (*b.get(i, c) - tail) / *r.get(i, i);
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::covariance::{
        HcVariant, classical_cov_params, cluster_cov_params, hc_cov_params, xtx_inverse,
    };
    use crate::shared::group_codes::GroupCodes;
    use crate::shared::wald::wald_quadratic_form;

    /// 切片・`x1`・`x2 = x1 + eps·ノイズ`（相関は`1 - O(eps²)`）の決定的なデータ（n=40、8クラスター）。
    /// 厳密値（有理数演算）はこのf64の値に対して、演算の順序まで同じに計算したもの。
    fn near_collinear(eps: f64) -> (Mat<f64>, Mat<f64>, GroupCodes) {
        let n = 40;
        let x1: Vec<f64> = (0..n).map(|i| (((i * 7) % 13) as f64) - 6.0).collect();
        let x2: Vec<f64> = (0..n)
            .map(|i| x1[i] + eps * ((((i * 5) % 11) as f64) - 5.0))
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = if i % 2 == 0 { 0.7 } else { -0.4 } + 0.1 * ((i % 5) as f64);
                1.0 + 2.0 * x1[i] + 3.0 * x2[i] + noise
            })
            .collect();
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            _ => x2[i],
        });
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 8)).collect();
        (
            x,
            Mat::from_fn(n, 1, |i, _| y[i]),
            GroupCodes::from_labels_without_keys(&groups),
        )
    }

    /// 傾きの全体Wald統計量（直交座標のミートから。`V`を作らない）。
    fn slope_wald(
        x: &Mat<f64>,
        y: &Mat<f64>,
        meat: impl Fn(&OrthonormalDesign, &Mat<f64>) -> Mat<f64>,
    ) -> f64 {
        let design = OrthonormalDesign::new(x).unwrap();
        let b_tilde = design.coordinates(y);
        let residuals = design.residuals(y, &b_tilde);
        let m_tilde = meat(&design, &residuals);
        wald_quadratic_form(&b_tilde, &m_tilde, 1, 2).unwrap()
    }

    fn assert_relative(got: f64, expected: f64, rtol: f64) {
        assert!(
            (got / expected - 1.0).abs() < rtol,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn q_is_orthonormal_and_reconstructs_x() {
        let (x, _, _) = near_collinear(1e-3);
        let design = OrthonormalDesign::new(&x).unwrap();
        let k = x.ncols();

        let gram = design.q().transpose() * design.q();
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((*gram.get(i, j) - expected).abs() < 1e-13, "[{i},{j}]");
            }
        }
        let rebuilt = design.q() * &design.r;
        for i in 0..x.nrows() {
            for j in 0..k {
                assert!(
                    (*rebuilt.get(i, j) - *x.get(i, j)).abs() < 1e-12,
                    "[{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn coefficients_and_residuals_match_the_normal_equations() {
        let (x, y, _) = near_collinear(1e-3);
        let design = OrthonormalDesign::new(&x).unwrap();
        let b_tilde = design.coordinates(&y);
        let beta = solve_upper(&design.r, b_tilde.as_ref());
        let residuals = design.residuals(&y, &b_tilde);

        // 正規方程式 X'e = 0、および X β̂ + e = y。
        let xte = x.transpose() * &residuals;
        for j in 0..3 {
            assert!((*xte.get(j, 0)).abs() < 1e-8, "X'e[{j}]");
        }
        let fitted = &x * &beta;
        for i in 0..x.nrows() {
            assert!((*fitted.get(i, 0) + *residuals.get(i, 0) - *y.get(i, 0)).abs() < 1e-8);
        }
    }

    #[test]
    fn coefficient_covariance_matches_the_classical_formula() {
        // `M̃ = σ² I`なら`V = σ² R⁻¹R⁻ᵀ = σ²(X'X)⁻¹`。
        let (x, _, _) = near_collinear(1e-1);
        let design = OrthonormalDesign::new(&x).unwrap();
        let sigma2 = 2.5;
        let m_tilde = Mat::from_fn(3, 3, |i, j| if i == j { sigma2 } else { 0.0 });

        let got = design.coefficient_covariance(&m_tilde);
        let expected = classical_cov_params(sigma2, &xtx_inverse(&x).unwrap(), 3);

        for i in 0..3 {
            for j in 0..3 {
                let scale = (*expected.get(i, i) * *expected.get(j, j)).sqrt();
                assert!(
                    (*got.get(i, j) - *expected.get(i, j)).abs() <= 1e-9 * scale,
                    "[{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn coefficient_covariance_matches_the_sandwich_when_well_conditioned() {
        // 旧経路`B M B`と新経路`R⁻¹M̃R⁻ᵀ`は、悪条件でなければ丸めの範囲で一致する
        // （標準誤差が変わらないことの確認）。
        let (x, y, groups) = near_collinear(1e-1);
        let design = OrthonormalDesign::new(&x).unwrap();
        let b_tilde = design.coordinates(&y);
        let residuals = design.residuals(&y, &b_tilde);

        let m_tilde = cluster_cov_params(design.q(), &residuals, &design.identity(), &groups);
        let new = design.coefficient_covariance(&m_tilde);
        let old = cluster_cov_params(&x, &residuals, &xtx_inverse(&x).unwrap(), &groups);

        for i in 0..3 {
            for j in 0..3 {
                let scale = (*old.get(i, i) * *old.get(j, j)).sqrt();
                assert!(
                    (*new.get(i, j) - *old.get(i, j)).abs() <= 1e-9 * scale,
                    "[{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn slope_wald_from_orthonormal_coordinates_matches_exact_arithmetic() {
        // 厳密値は有理数演算（Python`fractions`）。傾きの相関が`1 - 5e-13`（eps=1e-6）・
        // `1 - 5e-17`（eps=1e-8）の準共線。旧経路（`V`を作って反転）はeps=1e-6で
        // 非正定値になり`NearSingularMatrix`、eps=1e-8でも誤差が1を超える。
        // 誤差の目安は相関行列の条件数×ε（実測: eps=1e-6で7e-11、eps=1e-8で2e-9）。
        for (eps, cluster_exact, hc1_exact) in [
            (1e-6, 85942.08422275676, 44541.013916605865),
            (1e-8, 85942.15366005209, 44541.039983000206),
        ] {
            let (x, y, groups) = near_collinear(eps);
            let cluster = slope_wald(&x, &y, |d, e| {
                cluster_cov_params(d.q(), e, &d.identity(), &groups)
            });
            let hc1 = slope_wald(&x, &y, |d, e| {
                hc_cov_params(d.q(), e, &d.identity(), HcVariant::Hc1)
            });
            assert_relative(cluster, cluster_exact, 1e-7);
            assert_relative(hc1, hc1_exact, 1e-7);
        }
    }

    #[test]
    fn rejects_columns_that_cannot_be_decomposed() {
        // 列数が行数を超える。
        assert_eq!(
            OrthonormalDesign::new(&Mat::<f64>::zeros(2, 3)).unwrap_err(),
            RankDeficient::Columns
        );
        // 全ゼロ列（`R`の対角が0）。
        let zero_column = Mat::from_fn(4, 2, |i, j| if j == 0 { 1.0 } else { 0.0 * (i as f64) });
        assert_eq!(
            OrthonormalDesign::new(&zero_column).unwrap_err(),
            RankDeficient::Columns
        );
    }

    #[test]
    fn accepts_a_design_without_columns() {
        let design = OrthonormalDesign::new(&Mat::<f64>::zeros(5, 0)).unwrap();
        assert_eq!(design.q().ncols(), 0);
        assert_eq!(design.identity().nrows(), 0);
    }
}
