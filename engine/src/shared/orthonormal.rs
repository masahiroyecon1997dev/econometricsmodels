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
//! そのまま使える。ランク落ちの判定は、列を揃えた`R`（k×k）の列ピボットQRで行う
//! （[`OrthonormalDesign::new_checked`]。`X = QR`の`R`は`X`と同じ特異値を持つので、n×kの
//! 列ピボットQRを別に走らせずに済む）。[`OrthonormalDesign::new`]は判定を持たず、先に
//! `new_checked`を通した同じ設計行列にだけ使う（ここでは対角が厳密に0または非有限のときだけ防御的に弾く）。
//!
//! **列を揃えた`R`でのランク判定と条件数のガード**: [`OrthonormalDesign::new_checked`]は、`R`の
//! 列を単位ノルムに揃えた行列（`R`の列ノルムは`X`の列ノルムに等しい）に対して、(1) 列ピボットQRの
//! ランク判定（相対閾値`k·ε`）、(2) 条件数が[`RANK_DEFICIENT_CONDITION`]以内であることの確認
//! （超えれば完全な共線とみなす）、(3) [`MAX_EQUILIBRATED_CONDITION`]以内であることの確認、を
//! この順に行う。ランク落ち（(1)と(2)）は[`RankDeficient::Columns`]、(3)の超過は
//! [`RankDeficient::GramMatrix`]で弾く。(2)があるのは、完全な共線の条件数が`1e14`〜`1e16`で
//! (1)の閾値の近くに来て、(1)だけではどちらに転ぶか決まらないため（列名つきの報告をそろえる）。QRなら`cond(X)`が`1e8`を超える設計も推定できてしまうが、
//! `X'X`のCholeskyで止めていた従来の受け入れ範囲（列のスケールによらず、列を揃えた`cond(X)`で
//! 約`5e7`）を、手法によらず保つため。**生の`R`でなく列を揃えた行列に対して判定する**のは、列の
//! 単位で合否が動かないようにするため（Choleskyの成否は列の一様な拡大縮小に依らない。生の`R`の
//! 相対閾値`max|R_ii|`だと、共線な2列を`1e-12`倍するだけでランク落ちとして弾かれる）。

use faer::{Mat, MatRef};

use super::linear_algebra::{RankDeficient, checked_col_piv_qr};

/// 列を単位ノルムに揃えた設計行列の条件数の上限。これを超える設計は[`RankDeficient::GramMatrix`]
/// で弾く（`OrthonormalDesign::new_checked`）。`X'X`のCholeskyは`cond² · ε ≈ 1`、つまり
/// `cond ≈ ε^(-1/2) ≈ 6.7e7`で失敗する。凍結した準共線データで実測した成否の境界（2.8e7までは
/// 成功・9.2e7から失敗）の間に置いた暫定値で、Rの`lm`の許容`1e-7`とも同程度。この値を緩める
/// のは、受け入れる設計が増えるだけで破壊的でない（厳しくするのは破壊的）。
pub(crate) const MAX_EQUILIBRATED_CONDITION: f64 = 5.0e7;

/// 列を単位ノルムに揃えた設計行列の条件数がこれを超えると、完全な共線（数値的なランク落ち）として
/// [`RankDeficient::Columns`]で弾く（列名を伝えられる）。倍精度では係数に有効数字が残らず、QRが
/// 崩れ始める範囲（`1e12`あたり）。完全な共線（ダミー変数トラップ・重複列）の条件数は
/// `1e14`〜`1e16`で、列ピボットQRのランク判定（相対`k·ε`）はその境界でどちらにも転びうるため、
/// この上限でランク落ちの報告をそろえる。`MAX_EQUILIBRATED_CONDITION`との間は[`RankDeficient::GramMatrix`]。
pub(crate) const RANK_DEFICIENT_CONDITION: f64 = 1.0e12;

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
    /// ランク・条件数のガードは**無い**。呼び出し側が、同じ行列に対して先に[`Self::new_checked`]
    /// （または`OlsEstimator::fit_auxiliary`など、それを呼ぶ推定）を通していること。
    ///
    /// # Errors
    /// 行数が列数に満たない、または`R`の対角が厳密に0・非有限のとき[`RankDeficient::Columns`]。
    /// 数値的なランク落ちの判定ではない（呼び出し側が先に行う）。
    pub(crate) fn new(x: &Mat<f64>) -> Result<Self, RankDeficient> {
        Self::decompose(x, false, true)
    }

    /// [`Self::new`]に加えて、列を単位ノルムに揃えた`R`（k×k）の列ピボットQRでランク落ちを判定し
    /// （`checked_col_piv_qr`と同じ相対閾値）、その条件数が[`RANK_DEFICIENT_CONDITION`]・
    /// [`MAX_EQUILIBRATED_CONDITION`]以内であることを確かめる。`Q`を作る前に判定するので、弾く入力では`Q`を作らない。
    ///
    /// # Errors
    /// `x`がフルカラムランクでないとき、または列を揃えた条件数が[`RANK_DEFICIENT_CONDITION`]を
    /// 超える（完全な共線とみなす）とき[`RankDeficient::Columns`]。それ以下でも
    /// [`MAX_EQUILIBRATED_CONDITION`]を超える（または特異値分解が収束しない）とき
    /// [`RankDeficient::GramMatrix`]。
    pub(crate) fn new_checked(x: &Mat<f64>) -> Result<Self, RankDeficient> {
        Self::decompose(x, true, true)
    }

    /// [`Self::new_checked`]と同じ判定で、`R`（k×k）だけを返す（`Q`を作らない。`Q`が要らない
    /// 呼び出し側が、`Q`の分のメモリと計算を省く）。
    ///
    /// # Errors
    /// [`Self::new_checked`]と同じ。
    pub(crate) fn checked_r_factor(x: &Mat<f64>) -> Result<Mat<f64>, RankDeficient> {
        Ok(Self::decompose(x, true, false)?.r)
    }

    fn decompose(x: &Mat<f64>, check_rank: bool, with_q: bool) -> Result<Self, RankDeficient> {
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
        // `R`に非有限の要素がある（入力のNaN・無限大や、オーバーフロー）か、対角が厳密に0なら、
        // 列のランク落ちとして弾く（後段の等化・特異値分解は有限な`R`を前提にする）。
        let degenerate =
            (0..k).any(|j| (0..=j).any(|i| !r.get(i, j).is_finite()) || *r.get(j, j) == 0.0);
        if degenerate {
            return Err(RankDeficient::Columns);
        }
        if check_rank {
            let equilibrated = equilibrate_columns(&r);
            checked_col_piv_qr(&equilibrated)?;
            let condition = condition_number(&equilibrated);
            if condition > RANK_DEFICIENT_CONDITION {
                return Err(RankDeficient::Columns);
            }
            if condition > MAX_EQUILIBRATED_CONDITION {
                return Err(RankDeficient::GramMatrix);
            }
        }
        let q = if with_q {
            qr.compute_thin_Q()
        } else {
            Mat::zeros(0, k)
        };
        Ok(Self { q, r })
    }

    /// 直交化した設計行列`Q`（n×k）。meat構築部品に`x`として渡す。
    pub(crate) fn q(&self) -> &Mat<f64> {
        &self.q
    }

    /// 各行のレバレッジ `h_ii = ‖q_i‖²`（`X(X'X)⁻¹X'`の対角）。`X'X`も`(X'X)⁻¹`も作らない。
    pub(crate) fn row_leverages(&self) -> Vec<f64> {
        (0..self.q.nrows())
            .map(|i| (0..self.q.ncols()).map(|j| self.q.get(i, j).powi(2)).sum())
            .collect()
    }

    /// `Q`と`R`に分解して取り出す（`Q`をコピーせず別の構造体へ移すとき）。
    pub(crate) fn into_parts(self) -> (Mat<f64>, Mat<f64>) {
        (self.q, self.r)
    }

    /// k×kの単位行列。meat構築部品に`xtx_inv`として渡す（`Q'Q = I`）。
    pub(crate) fn identity(&self) -> Mat<f64> {
        Mat::identity(self.r.nrows(), self.r.nrows())
    }

    /// 係数の直交座標 `b̃ = Q'y`（= `Rβ̂`、k×1）。
    pub(crate) fn coordinates(&self, y: &Mat<f64>) -> Mat<f64> {
        self.q.transpose() * y
    }

    /// `(X'X)⁻¹ = R⁻¹R⁻ᵀ`（k×k）。`X'X`のCholeskyで逆行列を作るより、`X`の条件数の2乗で失う精度が
    /// 小さい。分散成分のトレース形など、`(X'X)⁻¹`そのものが式に現れる計算が使う。
    pub(crate) fn gram_inverse(&self) -> Mat<f64> {
        let r_inverse = solve_upper(&self.r, self.identity().as_ref());
        &r_inverse * r_inverse.transpose()
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
        covariance_from_factor(&self.r, m_tilde)
    }
}

/// `V = R⁻¹ M̃ R⁻ᵀ`（k×k）。`r`は上三角の因子、`m_tilde`は対称なk×kの直交座標のミート。
/// 丸めによるわずかな非対称は、上下三角の平均で取り除く。
pub(crate) fn covariance_from_factor(r: &Mat<f64>, m_tilde: &Mat<f64>) -> Mat<f64> {
    let left = solve_upper(r, m_tilde.as_ref());
    let v = solve_upper(r, left.transpose());
    Mat::from_fn(v.nrows(), v.ncols(), |i, j| {
        0.5 * (*v.get(i, j) + *v.get(j, i))
    })
}

/// 上三角`r`（k×k、`X = QR`の`R`）の列を単位ノルムに揃える。`R`の列ノルムは`X`の列ノルムに
/// 等しいので、`X`の列を揃えた行列と同じ特異値を持つ。列のノルムは、最大の成分で割ってから
/// 二乗和を取り、極端なスケールでのオーバーフロー・アンダーフローを避ける（ノルムそのもの
/// `largest * root`は作らず、`(r / largest) / root`で割る。`largest`が`f64::MAX`の近くでも
/// 溢れない）。前提: `R`は有限で、各列に0でない成分がある（呼び出し側が先に弾く）。
fn equilibrate_columns(r: &Mat<f64>) -> Mat<f64> {
    let k = r.ncols();
    let scales: Vec<(f64, f64)> = (0..k)
        .map(|j| {
            let largest = (0..=j).map(|i| r.get(i, j).abs()).fold(0.0_f64, f64::max);
            let scaled_sum: f64 = (0..=j).map(|i| (*r.get(i, j) / largest).powi(2)).sum();
            (largest, scaled_sum.sqrt())
        })
        .collect();
    Mat::from_fn(k, k, |i, j| {
        let (largest, root) = scales[j];
        if i <= j {
            *r.get(i, j) / largest / root
        } else {
            0.0
        }
    })
}

/// 行列（k×k、`k ≥ 1`）の条件数（最大特異値/最小特異値）。特異値分解が失敗したとき、最小特異値が
/// 0のとき、結果が有限でないとき（NaNを含む）は`f64::INFINITY`（呼び出し側が弾く）。
fn condition_number(matrix: &Mat<f64>) -> f64 {
    let Ok(singular_values) = matrix.singular_values() else {
        return f64::INFINITY;
    };
    let (Some(&largest), Some(&smallest)) = (singular_values.first(), singular_values.last())
    else {
        return f64::INFINITY;
    };
    let condition = largest / smallest;
    if smallest > 0.0 && condition.is_finite() {
        condition
    } else {
        f64::INFINITY
    }
}

/// `r`（`X = QR`の`R`）の列を揃えた行列の条件数。
#[cfg(test)]
fn equilibrated_condition(r: &Mat<f64>) -> f64 {
    condition_number(&equilibrate_columns(r))
}

/// 上三角`r`（k×k）で`r · x = b`を解く（後退代入。`b`はk×m）。
pub(crate) fn solve_upper(r: &Mat<f64>, b: MatRef<'_, f64>) -> Mat<f64> {
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
    fn coefficient_covariance_matches_the_sandwich_for_every_hc_variant() {
        // HC0〜HC3（HC2・HC3のレバレッジは直交座標で`‖q_i‖²`になる）でも、旧経路`B M B`と新経路
        // `R⁻¹M̃R⁻ᵀ`が、悪条件でなければ丸めの範囲で一致する。
        let (x, y, _) = near_collinear(1e-1);
        let design = OrthonormalDesign::new(&x).unwrap();
        let residuals = design.residuals(&y, &design.coordinates(&y));
        let xtx_inv = xtx_inverse(&x).unwrap();
        for variant in [
            HcVariant::Hc0,
            HcVariant::Hc1,
            HcVariant::Hc2,
            HcVariant::Hc3,
        ] {
            let m_tilde = hc_cov_params(design.q(), &residuals, &design.identity(), variant);
            let new = design.coefficient_covariance(&m_tilde);
            let old = hc_cov_params(&x, &residuals, &xtx_inv, variant);
            for i in 0..3 {
                for j in 0..3 {
                    let scale = (*old.get(i, i) * *old.get(j, j)).sqrt();
                    assert!(
                        (*new.get(i, j) - *old.get(i, j)).abs() <= 1e-9 * scale,
                        "{variant:?} [{i},{j}]"
                    );
                }
            }
        }
    }

    #[test]
    fn slope_wald_from_orthonormal_coordinates_matches_exact_arithmetic() {
        // 厳密値は有理数演算（Python`fractions`）。傾きの相関が`1 - 5e-13`（eps=1e-6）・
        // `1 - 5e-17`（eps=1e-8）の準共線。旧経路（`V`を作って反転）はeps=1e-6で
        // 非正定値になり`NearSingularMatrix`、eps=1e-8でも誤差が1を超える。
        // 誤差の目安は相関行列の条件数×ε（実測: eps=1e-6で7e-11、eps=1e-8で2e-9）。
        // 許容誤差は実測の約10倍（eps=1e-6で1e-9、eps=1e-8で3e-8）。
        for (eps, rtol, cluster_exact, hc1_exact) in [
            (1e-6, 1e-9, 85942.08422275676, 44541.013916605865),
            (1e-8, 3e-8, 85942.15366005209, 44541.039983000206),
        ] {
            let (x, y, groups) = near_collinear(eps);
            let cluster = slope_wald(&x, &y, |d, e| {
                cluster_cov_params(d.q(), e, &d.identity(), &groups)
            });
            let hc1 = slope_wald(&x, &y, |d, e| {
                hc_cov_params(d.q(), e, &d.identity(), HcVariant::Hc1)
            });
            assert_relative(cluster, cluster_exact, rtol);
            assert_relative(hc1, hc1_exact, rtol);
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
    fn rejects_a_design_whose_r_is_not_finite() {
        // 列のノルムがオーバーフローして`R`に無限大が入る（対角だけでなく上三角の要素も見る）。
        let overflow = Mat::from_fn(4, 2, |i, j| if j == 0 { (i + 1) as f64 } else { 1e308 });
        assert_eq!(
            OrthonormalDesign::new(&overflow).unwrap_err(),
            RankDeficient::Columns
        );
        let with_nan = Mat::from_fn(4, 2, |i, j| {
            if i == 1 && j == 1 {
                f64::NAN
            } else {
                1.0 + i as f64
            }
        });
        assert_eq!(
            OrthonormalDesign::new_checked(&with_nan).unwrap_err(),
            RankDeficient::Columns
        );
    }

    #[test]
    fn accepts_a_design_without_columns() {
        let design = OrthonormalDesign::new(&Mat::<f64>::zeros(5, 0)).unwrap();
        assert_eq!(design.q().ncols(), 0);
        assert_eq!(design.identity().nrows(), 0);
    }

    /// 切片・`x1`・`x2 = x1 + eps·ノイズ`の設計（`near_collinear`と同じ列）で、傾きの2列だけを
    /// `scale`倍する。
    fn near_collinear_design(eps: f64, scale: f64) -> Mat<f64> {
        let (x, _, _) = near_collinear(eps);
        Mat::from_fn(x.nrows(), 3, |i, j| {
            if j == 0 {
                *x.get(i, j)
            } else {
                scale * *x.get(i, j)
            }
        })
    }

    #[test]
    fn equilibrated_condition_is_the_condition_number_of_the_unit_norm_columns() {
        // 列が単位ベクトル(1, 0)と(0.8, 0.6)なら、特異値は√(1±0.8)で条件数は√(1.8/0.2) = 3。
        let r = Mat::from_fn(2, 2, |i, j| [[1.0, 0.8], [0.0, 0.6]][i][j]);
        assert_relative(equilibrated_condition(&r), 3.0, 1e-12);
        // 列の一様な拡大縮小では変わらない。
        let scaled = Mat::from_fn(2, 2, |i, j| *r.get(i, j) * if j == 0 { 1e6 } else { 1e-6 });
        assert_relative(equilibrated_condition(&scaled), 3.0, 1e-12);
        // 直交する列は、列のスケールが違っても条件数1。
        let orthogonal = Mat::from_fn(2, 2, |i, j| if i == j { [1e6, 1e-6][j] } else { 0.0 });
        assert_relative(equilibrated_condition(&orthogonal), 1.0, 1e-12);
    }

    #[test]
    fn new_checked_reports_exact_collinearity_as_columns_not_as_the_condition_guard() {
        // ダミー変数トラップ（ダミーの和が切片）・重複列・和の列。条件数は1e14〜1e16で、列ピボットQRの
        // ランク判定（相対`k·ε`）の境界に来るため、条件数の上限（`RANK_DEFICIENT_CONDITION`）でも
        // 列のランク落ちとして弾く。大きなnでも同じ。
        for n in [1_000_usize, 100_000] {
            let group = |i: usize| (i * 7 + i / 3) % 4;
            let wave = |i: usize| ((i * 37 % 101) as f64 - 50.0) / 7.0;
            let trap = Mat::from_fn(n, 6, |i, j| match j {
                0 => 1.0,
                1 => wave(i),
                _ => f64::from(u8::from(group(i) == j - 2)),
            });
            let duplicated = Mat::from_fn(n, 3, |i, j| if j == 0 { 1.0 } else { wave(i) });
            let sum = Mat::from_fn(n, 3, |i, j| match j {
                0 => 1.0,
                1 => wave(i),
                _ => wave(i) + 1.0,
            });
            for (label, x) in [("trap", trap), ("duplicate", duplicated), ("sum", sum)] {
                assert_eq!(
                    OrthonormalDesign::new_checked(&x).unwrap_err(),
                    RankDeficient::Columns,
                    "n={n} {label}"
                );
            }
        }
    }

    #[test]
    fn new_checked_boundary_does_not_move_with_the_column_scale() {
        // 列を揃えた条件数は、eps=1e-7で約2.8e7（上限5e7の内側）、eps=3e-8で約9.2e7（外側）。
        // 傾きの列の単位（1e-6〜1e6）が変わっても、合否は変わらない。
        for scale in [1.0, 1e6, 1e-6] {
            assert!(
                OrthonormalDesign::new_checked(&near_collinear_design(1e-7, scale)).is_ok(),
                "scale={scale}: cond below the limit must be accepted"
            );
            assert_eq!(
                OrthonormalDesign::new_checked(&near_collinear_design(3e-8, scale)).unwrap_err(),
                RankDeficient::GramMatrix,
                "scale={scale}: cond above the limit must be rejected"
            );
        }
    }

    #[test]
    fn new_checked_rank_decision_does_not_move_with_the_column_scale() {
        // 条件数が上限の内側（eps=1e-7で約2.8e7）の設計は、共線な2列の単位を1e±12倍しても受け入れる
        // （生の`R`の相対閾値`max|R_ii|`だと、1e-12倍で最小の対角が閾値を下回りランク落ちとして
        // 弾かれていた）。完全に重複した列は、どの単位でもランク落ち。
        for scale in [1e-12, 1e-9, 1e9, 1e12] {
            assert!(
                OrthonormalDesign::new_checked(&near_collinear_design(1e-7, scale)).is_ok(),
                "scale={scale}: a well-inside design must be accepted"
            );
            let (x, _, _) = near_collinear(1.0);
            let duplicated = Mat::from_fn(x.nrows(), 3, |i, j| match j {
                0 => 1.0,
                1 => scale * *x.get(i, 1),
                _ => 3.0 * scale * *x.get(i, 1),
            });
            assert_eq!(
                OrthonormalDesign::new_checked(&duplicated).unwrap_err(),
                RankDeficient::Columns,
                "scale={scale}: duplicated columns are rank deficient at any unit"
            );
        }
    }

    #[test]
    fn equilibrate_columns_survives_extreme_magnitudes() {
        // 二乗和がオーバーフロー（1e200²）・アンダーフロー（1e-200²）する大きさでも、列は単位ノルムに揃う。
        for magnitude in [1e200, 1e-200, f64::MAX / 2.0] {
            let r = Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => magnitude,
                (0, 1) => 0.5 * magnitude,
                (1, 1) => 0.5 * magnitude,
                _ => 0.0,
            });
            let equilibrated = equilibrate_columns(&r);
            for j in 0..2 {
                let norm: f64 = (0..2)
                    .map(|i| equilibrated.get(i, j).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert_relative(norm, 1.0, 1e-12);
            }
        }
    }

    #[test]
    fn condition_number_handles_degenerate_matrices() {
        assert_relative(
            condition_number(&Mat::from_fn(1, 1, |_, _| 5.0)),
            1.0,
            1e-12,
        );
        assert_eq!(condition_number(&Mat::<f64>::zeros(2, 2)), f64::INFINITY);
        let singular = Mat::from_fn(2, 2, |_, j| if j == 0 { 1.0 } else { 0.0 });
        assert_eq!(condition_number(&singular), f64::INFINITY);
        let with_nan = Mat::from_fn(2, 2, |i, j| if i == 0 && j == 1 { f64::NAN } else { 1.0 });
        assert_eq!(condition_number(&with_nan), f64::INFINITY);
    }

    #[test]
    fn new_does_not_apply_the_condition_guard() {
        // 同じ設計を、ガードなしの`new`は受理する（呼び出し側が先にランクを保証する前提）。
        let x = near_collinear_design(3e-8, 1.0);
        assert!(OrthonormalDesign::new(&x).is_ok());
        assert!(OrthonormalDesign::new_checked(&x).is_err());
    }

    #[test]
    fn new_checked_accepts_a_well_conditioned_design_with_the_same_q_as_new() {
        let (x, y, _) = near_collinear(1.0);
        let checked = OrthonormalDesign::new_checked(&x).unwrap();
        let unchecked = OrthonormalDesign::new(&x).unwrap();
        let (b_checked, b_unchecked) = (checked.coordinates(&y), unchecked.coordinates(&y));
        for i in 0..x.ncols() {
            assert_eq!(*b_checked.get(i, 0), *b_unchecked.get(i, 0));
        }
    }

    #[test]
    fn new_checked_rejects_rank_deficient_designs_like_checked_col_piv_qr() {
        let n = 12;
        let x1 = |i: usize| ((i * 7) % 5) as f64 - 2.0;
        let designs: Vec<(&str, Mat<f64>)> = vec![
            (
                "duplicate column",
                Mat::from_fn(n, 3, |i, j| if j == 0 { 1.0 } else { x1(i) }),
            ),
            (
                "scaled duplicate",
                Mat::from_fn(n, 3, |i, j| match j {
                    0 => 1.0,
                    1 => x1(i),
                    _ => 3.0 * x1(i),
                }),
            ),
            (
                "dummy variable trap",
                Mat::from_fn(n, 3, |i, j| match j {
                    0 => 1.0,
                    1 => f64::from(u8::from(i % 2 == 0)),
                    _ => f64::from(u8::from(i % 2 == 1)),
                }),
            ),
            (
                "all-zero column",
                Mat::from_fn(n, 3, |i, j| match j {
                    0 => 1.0,
                    1 => x1(i),
                    _ => 0.0,
                }),
            ),
            (
                "NaN column",
                Mat::from_fn(n, 3, |i, j| match j {
                    0 => 1.0,
                    1 => x1(i),
                    _ => f64::NAN,
                }),
            ),
            ("more columns than rows", Mat::<f64>::zeros(2, 3)),
        ];
        for (name, x) in designs {
            assert!(checked_col_piv_qr(&x).is_err(), "{name}: precondition");
            assert_eq!(
                OrthonormalDesign::new_checked(&x).unwrap_err(),
                RankDeficient::Columns,
                "{name}"
            );
        }
    }

    #[test]
    fn gram_inverse_matches_the_cholesky_inverse_of_the_gram_matrix() {
        let (x, _, _) = near_collinear(1.0);
        let from_qr = OrthonormalDesign::new_checked(&x).unwrap().gram_inverse();
        let from_cholesky = xtx_inverse(&x).unwrap();
        for i in 0..x.ncols() {
            for j in 0..x.ncols() {
                assert_relative(
                    *from_qr.get(i, j) + 1.0,
                    *from_cholesky.get(i, j) + 1.0,
                    1e-12,
                );
            }
        }
        // 対称。
        for i in 0..x.ncols() {
            for j in 0..i {
                assert_relative(*from_qr.get(i, j) + 1.0, *from_qr.get(j, i) + 1.0, 1e-12);
            }
        }
    }

    #[test]
    fn new_checked_accepts_a_design_without_columns() {
        let design = OrthonormalDesign::new_checked(&Mat::<f64>::zeros(5, 0)).unwrap();
        assert_eq!(design.q().ncols(), 0);
    }
}
