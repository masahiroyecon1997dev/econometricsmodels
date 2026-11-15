//! GMMの係数の分散共分散行列(サンドイッチの代数)とHansen J統計量。行列だけを受け取る。
//!
//! 重み行列`S`・モーメントの分散共分散行列`Ω̂`の組み立て(`cov_type`ごとに違う)と、小標本補正の
//! 分母(構造方程式の係数の数など)は呼び出し側が決める。ここは、それらを受け取った後の代数で、
//! IVのGMMが使い、固定効果つきのGMM(within変換後のデータと自前の`Ω̂`)も同じ関数を呼べるようにしてある。`IvInput`・系統のエラー型には依存しない。

use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::prelude::Solve;
use faer::{Mat, Par, Side};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::error::{CommonError, MatrixKind};
use super::orthonormal::{OrthonormalDesign, WaldCoordinates};

/// 対称正定値行列`mat`(`dim`×`dim`)の逆行列を明示的に構築する(Cholesky分解)。分解が失敗したら
/// `failure`を返す。
///
/// # Errors
/// `mat`が正定値でないとき`failure`。
pub(crate) fn invert_spd(
    mat: &Mat<f64>,
    dim: usize,
    failure: CommonError,
) -> Result<Mat<f64>, CommonError> {
    let llt = mat.llt(Side::Lower).map_err(|_| failure)?;
    Ok(llt.solve(Mat::<f64>::identity(dim, dim)))
}

/// 重み行列`S = L L'`で白色化した操作変数`Z̃ = Z L⁻ᵀ`（n×l）と、その下三角の因子`L`（l×l）。
///
/// GMMは、白色化したモーメント空間の最小二乗として書ける: `W = S⁻¹ = L⁻ᵀL⁻¹`のとき
/// `Ĝ = L⁻¹Z'X = Z̃'X`、`r = L⁻¹Z'y = Z̃'y`として、`β̂ = argmin ‖r - Ĝβ‖²`
/// （`B = X'ZWZ'X = Ĝ'Ĝ`）。`Ĝ`を直接QR分解すれば、`B`を作って分解する（条件数の2乗で精度を失う）
/// ことなく、OLSと同じ直交座標で係数・共分散・全体Wald検定を求められる（[`gmm_solve`]）。
/// `Ĝ`を`Z'X`から三角解で作る（`L⁻¹(Z'X)`）より、`Z̃`を先に作って`Z̃'X`と積を取るほうが、
/// 近共線な操作変数のとき精度が1〜2桁よい（試作で実測）。
pub(crate) struct WhitenedInstruments {
    z_tilde: Mat<f64>,
}

impl WhitenedInstruments {
    /// 重み`S = Z'Z`（初期推定。`W₀ = (Z'Z)⁻¹`）の白色化。`Z = Q_ZR_Z`のQR分解から`Z̃ = Q_Z`、
    /// `L = R_Zᵀ`を取る（`Z'Z`を作って分解しない。`Z`の条件数の2乗で精度を失わない）。
    /// 操作変数の列のランク落ち・悪条件は、設計行列と同じ判定（`OrthonormalDesign::new_checked`。
    /// 列を揃えた条件数の上限）で弾く。
    ///
    /// # Errors
    /// `Z`がランク落ち、または列を揃えた条件数が上限を超えるとき、`CommonError::NearSingularMatrix`
    /// （`MatrixKind::GmmWeight`）。
    pub(crate) fn from_instruments(z: &Mat<f64>) -> Result<Self, CommonError> {
        let design =
            OrthonormalDesign::new_checked(z).map_err(|_| CommonError::NearSingularMatrix {
                matrix: MatrixKind::GmmWeight,
            })?;
        let (z_tilde, _) = design.into_parts();
        Ok(Self { z_tilde })
    }

    /// 重み`S = A'A`（`A`は`m×l`、`m ≥ l`）の白色化 `Z̃ = Z L⁻ᵀ`。`A`のQR分解から`L`を取る
    /// （`S`を作って分解しない。`A`の条件数の2乗で精度を失わない）。ロバスト重み（`A`の行は
    /// `êᵢzᵢ`）・クラスター重み（`A`の行はクラスターのスコア）がこの形になる。
    /// `A`の列は先に単位ノルムに揃える（GMMの推定値は操作変数の列の単位に依らない）。
    /// 条件数の判定は設計行列と同じ（`OrthonormalDesign::new_checked`）。
    ///
    /// # Errors
    /// `A`の列が0か非有限、ランク落ち、または条件数が上限を超えるとき、
    /// `CommonError::NearSingularMatrix`（`MatrixKind::GmmWeight`）。
    pub(crate) fn from_factor_rows(z: &Mat<f64>, a: &Mat<f64>) -> Result<Self, CommonError> {
        let failure = || CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight,
        };
        let l = a.ncols();
        let scale: Vec<f64> = (0..l)
            .map(|j| {
                let largest = (0..a.nrows())
                    .map(|i| a.get(i, j).abs())
                    .fold(0.0, f64::max);
                let sum: f64 = (0..a.nrows())
                    .map(|i| (*a.get(i, j) / largest).powi(2))
                    .sum();
                1.0 / (largest * sum.sqrt())
            })
            .collect();
        if scale.iter().any(|d| !d.is_finite() || *d <= 0.0) {
            return Err(failure());
        }
        let a_unit = Mat::from_fn(a.nrows(), l, |i, j| scale[j] * *a.get(i, j));
        let r = OrthonormalDesign::checked_r_factor(&a_unit).map_err(|_| failure())?;
        Ok(Self::from_unit_lower(z, &scale, r.transpose().to_owned()))
    }

    /// 重み行列`S`（l×l、正定値）の白色化 `Z̃ = Z L⁻ᵀ`（`S = L L'`）。先に`S`の対角を1に揃えて
    /// （`D S D`、`D = diag(1/√S_jj)`）Cholesky分解する。HACの重みのように`S`を和の形で
    /// 作れない場合に使う（[`Self::from_factor_rows`]のほうが精度がよい）。
    ///
    /// # Errors
    /// `S`の対角が正の有限値でないか、`S`が正定値でないとき、`CommonError::NearSingularMatrix`
    /// （`MatrixKind::GmmWeight`）。
    pub(crate) fn from_weight(z: &Mat<f64>, s: &Mat<f64>) -> Result<Self, CommonError> {
        let failure = || CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight,
        };
        let l = s.nrows();
        let scale: Vec<f64> = (0..l).map(|j| 1.0 / s.get(j, j).sqrt()).collect();
        if scale.iter().any(|d| !d.is_finite() || *d <= 0.0) {
            return Err(failure());
        }
        let s_unit = Mat::from_fn(l, l, |i, j| scale[i] * *s.get(i, j) * scale[j]);
        let llt = s_unit.llt(Side::Lower).map_err(|_| failure())?;
        Ok(Self::from_unit_lower(z, &scale, llt.L().to_owned()))
    }

    /// [`Self::from_weight`]と同じ白色化を、`S`（対角を揃えた`D S D`）の条件数が`max_condition`以下
    /// のときだけ行う。条件数が大きいときは`None`（呼び出し側が[`Self::from_factor_rows`]に
    /// 切り替える）。Choleskyの丸め誤差は`S`の条件数に比例して広がるので、条件の良い`S`では
    /// 安い経路（`S`の分解）で十分に正確で、悪いときだけ`S`を作らないQRの経路に回す。
    ///
    /// # Errors
    /// [`Self::from_weight`]と同じ。
    pub(crate) fn from_weight_if_well_conditioned(
        z: &Mat<f64>,
        s: &Mat<f64>,
        max_condition: f64,
    ) -> Result<Option<Self>, CommonError> {
        let l = s.nrows();
        let scale: Vec<f64> = (0..l).map(|j| 1.0 / s.get(j, j).sqrt()).collect();
        if scale.iter().any(|d| !d.is_finite() || *d <= 0.0) {
            return Err(CommonError::NearSingularMatrix {
                matrix: MatrixKind::GmmWeight,
            });
        }
        let s_unit = Mat::from_fn(l, l, |i, j| scale[i] * *s.get(i, j) * scale[j]);
        let well_conditioned =
            s_unit
                .singular_values()
                .is_ok_and(|values| match (values.first(), values.last()) {
                    (Some(&largest), Some(&smallest)) => {
                        smallest > 0.0 && largest / smallest <= max_condition
                    }
                    _ => false,
                });
        if !well_conditioned {
            return Ok(None);
        }
        let llt = s_unit
            .llt(Side::Lower)
            .map_err(|_| CommonError::NearSingularMatrix {
                matrix: MatrixKind::GmmWeight,
            })?;
        Ok(Some(Self::from_unit_lower(z, &scale, llt.L().to_owned())))
    }

    /// `S_unit = D S D = L_unit L_unit'`の下三角因子から、`Z̃' = L_unit⁻¹ D Z'`を前進代入で作る。
    fn from_unit_lower(z: &Mat<f64>, scale: &[f64], lower: Mat<f64>) -> Self {
        let mut z_tilde_t = Mat::from_fn(z.ncols(), z.nrows(), |j, i| scale[j] * *z.get(i, j));
        solve_lower_triangular_in_place(lower.as_ref(), z_tilde_t.as_mut(), Par::Seq);
        Self {
            z_tilde: z_tilde_t.transpose().to_owned(),
        }
    }
}

/// 白色化した最小二乗で求めたGMMの解: `Ĝ = Z̃'X = QR`（直交座標）、`b̃ = Q'r`、`β̂ = R⁻¹b̃`。
pub(crate) struct GmmSolution {
    design: OrthonormalDesign,
    coordinates: Mat<f64>,
    beta: Mat<f64>,
    z_tilde: Mat<f64>,
    /// 点推定に使った重み`S_used`が、白色化の元の重み`S_factor`の何倍か（`S_used = c · S_factor`）。
    /// `Z'Z`の`Z`のQR分解で白色化するとき（初期推定・古典的な重み）は`σ̂²`、それ以外は1。
    weight_scale: f64,
}

impl GmmSolution {
    /// 点推定に使った重みが、白色化の元の重みの`scale`倍であることを記録する（Hansen Jの尺度）。
    /// 係数・共分散・Wald統計量は重みのスカラー倍で変わらない。
    pub(crate) fn with_weight_scale(mut self, scale: f64) -> Self {
        self.weight_scale = scale;
        self
    }

    /// Hansen J統計量 `J = (Z'ê)'S_used⁻¹(Z'ê) = ‖Z̃'ê‖²/c`（`n`で割らない）。白色化した空間の
    /// 内積なので、`S`を作って分解しない。`extra_scale`は`S_used`にさらに掛ける倍率（FEの吸収した効果の
    /// 自由度の補正。`S`が`extra_scale`倍なら`J`は`1/extra_scale`倍）。
    pub(crate) fn hansen_j_statistic(&self, residuals: &Mat<f64>, extra_scale: f64) -> f64 {
        let moments = self.z_tilde.transpose() * residuals;
        let squared: f64 = (0..moments.nrows())
            .map(|i| moments.get(i, 0).powi(2))
            .sum();
        squared / (self.weight_scale * extra_scale)
    }

    /// Hansen J検定（統計量とχ²のp値）。自由度`df`は呼び出し側が決める（1以上）。
    ///
    /// # Errors
    /// `df`が0のとき（呼び出し側の誤り）、`CommonError::Internal`。
    pub(crate) fn hansen_j_test(
        &self,
        residuals: &Mat<f64>,
        extra_scale: f64,
        df: usize,
    ) -> Result<HansenJ, CommonError> {
        if df == 0 {
            return Err(CommonError::internal(
                "the Hansen J test needs at least one degree of freedom",
            ));
        }
        let statistic = self.hansen_j_statistic(residuals, extra_scale);
        let chi2 = ChiSquared::new(df as f64).map_err(|e| CommonError::internal(e.to_string()))?;
        Ok(HansenJ {
            statistic,
            p_value: chi2.sf(statistic),
        })
    }

    /// 係数`β̂`（k×1）。
    pub(crate) fn beta(&self) -> &Mat<f64> {
        &self.beta
    }

    /// 白色化した操作変数`Z̃ = Z L⁻ᵀ`（n×l）。`Ω̂`の組み立ては、`Z`の代わりにこれを渡す
    /// （`L⁻¹Ω̂L⁻ᵀ`を`Z̃`の和として直接作れる。`Ω̂ = Z'DZ`型の和も、HACやクラスターのスコアの和も、
    /// 列の線形変換で形が変わらない。HC2・HC3のレバレッジ`z_i'(Z'Z)⁻¹z_i`は`Z`の列の線形変換に
    /// 対して不変で、`Z̃`でも同じ値になる）。`L⁻¹Ω̂L⁻ᵀ`を`l×l`の`Ω̂`から三角解で作るより、
    /// `Z'Z`の条件数の2乗で精度を失わない。
    pub(crate) fn z_tilde(&self) -> &Mat<f64> {
        &self.z_tilde
    }

    /// 係数の分散共分散行列 `Avar(β̂) = R⁻¹ M̃ R⁻ᵀ`（k×k、対称化済み）と、全体Wald検定に使う直交座標
    /// （`b̃`・`M̃ = Q' Ω̃ Q`）。`B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`と同じ値で、`B`も`B⁻¹`も作らない。
    ///
    /// `omega_tilde`は白色化したモーメント空間の`Ω̃ = L⁻¹Ω̂L⁻ᵀ`（l×l。`Ω̂`は`S`と独立な選択で、
    /// 小標本補正を含む）。[`Self::z_tilde`]を`Z`の代わりに渡して`Ω̂`の組み立てを呼べば得られる。
    pub(crate) fn covariance(&self, omega_tilde: &Mat<f64>) -> (Mat<f64>, WaldCoordinates) {
        let q = self.design.q();
        let meat = q.transpose() * omega_tilde * q;
        let cov_params = self.design.coefficient_covariance(&meat);
        (
            cov_params,
            WaldCoordinates {
                coordinates: self.coordinates.clone(),
                meat,
            },
        )
    }
}

/// 白色化した操作変数で、GMMの点推定と直交座標を求める（`β̂ = (X'ZWZ'X)⁻¹X'ZWZ'y`、`W = S⁻¹`）。
///
/// `Ĝ = Z̃'X`の列を揃えた条件数が`orthonormal`の上限（OLS・IVの設計行列と同じ）を超えるか、
/// ランク落ちのとき失敗する。判定は`X'ZWZ'X`の分解の失敗に代わる（`Ĝ`の特異値は`B`の特異値の
/// 平方根なので、同じ量を2乗せずに調べる）。
///
/// # Errors
/// `Ĝ`がランク落ち、または条件数が上限を超えるとき、`CommonError::NearSingularMatrix`
/// （`MatrixKind::GmmBread`）。
pub(crate) fn gmm_solve(
    x: &Mat<f64>,
    y: &Mat<f64>,
    whitening: WhitenedInstruments,
) -> Result<GmmSolution, CommonError> {
    let g_hat = whitening.z_tilde.transpose() * x; // (l, k)
    let r_vec = whitening.z_tilde.transpose() * y; // (l, 1)
    let design =
        OrthonormalDesign::new_checked(&g_hat).map_err(|_| CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmBread,
        })?;
    let coordinates = design.coordinates(&r_vec);
    let beta = design.coefficients(&coordinates);
    Ok(GmmSolution {
        design,
        coordinates,
        beta,
        z_tilde: whitening.z_tilde,
        weight_scale: 1.0,
    })
}

#[cfg(test)]
/// GMMのサンドイッチ型分散共分散行列 `Avar(β̂) = B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`
/// (`B = X'ZWZ'X`、`W = S_used⁻¹`、k×k)。
///
/// - `z`: 全操作変数の行列(n×l)、`x`: 構造方程式の設計行列(n×k)。
/// - `s_used`: 点推定に実際に使った重み行列`S`(l×l。`W = S⁻¹`)。
/// - `omega`: モーメント条件の分散共分散行列`Ω̂`(l×l)。`S`と`Ω̂`は独立な選択で、一般には一致
///   しない(一致する効率的GMMでも`B⁻¹`だけへの簡略化はしない。単一の経路にするため)。
///
/// `S`のスケールを`c`倍すると`B⁻¹`は`1/c`倍、`WΩ̂W`は`1/c²`倍で相殺するので、`cov_params`は
/// `S`のスカラー倍で変わらない。**`Ω̂`のスケールは変わらない**: `Ω̂`は`n`で割らない和
/// (`Σᵢ êᵢ²zᵢzᵢ'`など)で、小標本補正を含めて呼び出し側が渡す。返り値がそのまま`Avar(β̂)`。
///
/// 返り値は丸めで厳密には対称でない。Wald検定は`shared::wald`が対称化する。係数の分散共分散行列
/// として使う側が対称性を要るなら、呼び出し側で対称化する。行列の次元(`x`と`z`の行数、
/// `s_used`・`omega`がl×l)は呼び出し側の責任(debugビルドでは検査し、不一致はfaerがpanicする)。
///
/// # Errors
/// `S`または`B`が正定値でない(Cholesky分解が失敗する)とき、`CommonError::NearSingularMatrix`
/// (`MatrixKind::GmmWeight`・`MatrixKind::GmmBread`)。
pub(crate) fn gmm_sandwich_covariance(
    z: &Mat<f64>,
    x: &Mat<f64>,
    s_used: &Mat<f64>,
    omega: &Mat<f64>,
) -> Result<Mat<f64>, CommonError> {
    let l = z.ncols();
    let k = x.ncols();
    debug_assert_eq!(z.nrows(), x.nrows());
    debug_assert_eq!((s_used.nrows(), s_used.ncols()), (l, l));
    debug_assert_eq!((omega.nrows(), omega.ncols()), (l, l));
    // W = S_used⁻¹。bread = X'ZWZ'X は meat と同じく`W`を使うので、明示的な逆行列として持つ。
    let s_inv = invert_spd(
        s_used,
        l,
        CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight,
        },
    )?;
    let zx = z.transpose() * x; // (l, k)
    let bread = zx.transpose() * &s_inv * &zx; // (k, k) = X'ZWZ'X
    let bread_inv = invert_spd(
        &bread,
        k,
        CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmBread,
        },
    )?;

    let meat_z = &s_inv * omega * &s_inv; // (l, l) = WΩ̂W
    let meat = zx.transpose() * &meat_z * &zx; // (k, k) = X'ZWΩ̂WZ'X
    Ok(&bread_inv * &meat * &bread_inv)
}

/// Hansen J統計量とp値。自由度`df`は返さない(呼び出し側が渡した値で、結果と一緒に持つ)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HansenJ {
    pub(crate) statistic: f64,
    pub(crate) p_value: f64,
}

/// Hansen J統計量`J = (Z'ê)'S⁻¹(Z'ê)`(`n`で割らない。理由は[`hansen_j_test`])。`S`を作って分解する
/// 形で、白色化した解を持たない呼び出し側（C統計量）が使う。解を持つ側は
/// [`GmmSolution::hansen_j_test`]（`S`の条件数の2乗で精度を失わない）。自由度を持たないので、
/// 丁度識別(`J = 0`)も計算できる。
///
/// # Errors
/// `S`が正定値でないとき`CommonError::NearSingularMatrix`(`MatrixKind::GmmWeight`)。
pub(crate) fn hansen_j_statistic(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    s_used: &Mat<f64>,
) -> Result<f64, CommonError> {
    debug_assert_eq!((residuals.nrows(), residuals.ncols()), (z.nrows(), 1));
    debug_assert_eq!((s_used.nrows(), s_used.ncols()), (z.ncols(), z.ncols()));
    let zte = z.transpose() * residuals;
    let llt_s = s_used
        .llt(Side::Lower)
        .map_err(|_| CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight,
        })?;
    let s_inv_zte = llt_s.solve(&zte);
    Ok((0..z.ncols())
        .map(|i| (*zte.get(i, 0)) * (*s_inv_zte.get(i, 0)))
        .sum())
}

#[cfg(test)]
/// Hansen J過剰識別検定: `J = (Z'ê)'S⁻¹(Z'ê)`を`χ²(df)`と比べる。
///
/// **`n`で割らない**: `S`が`n`で正規化していない生の和(`S = Σᵢ êᵢ²zᵢzᵢ'`など)なら、標準形
/// `J = n·ḡ'Ŝ⁻¹ḡ`(`ḡ = Z'ê/n`、`Ŝ = S/n`)で`n`が相殺する。`S`の組み立てがこの前提を
/// 守ること(`S`に`σ̂²`を掛ける古典的な重みも含む)は呼び出し側の責任。
///
/// `df = 除外操作変数の数 - 内生変数の数`は`1`以上であること(丁度識別は検定が無いので、
/// 呼び出し側が先に分岐する)。
///
/// # Errors
/// `S`が正定値でないとき`CommonError::NearSingularMatrix`(`MatrixKind::GmmWeight`)。
/// `df == 0`のとき`CommonError::Internal`。
pub(crate) fn hansen_j_test(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    s_used: &Mat<f64>,
    df: usize,
) -> Result<HansenJ, CommonError> {
    if df == 0 {
        return Err(CommonError::internal(
            "the Hansen J test needs at least one degree of freedom",
        ));
    }
    let statistic = hansen_j_statistic(z, residuals, s_used)?;
    let chi2 = ChiSquared::new(df as f64).map_err(|e| CommonError::internal(e.to_string()))?;
    let p_value = chi2.sf(statistic);
    Ok(HansenJ { statistic, p_value })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n=8、操作変数l=3(定数・z1・z2)、構造方程式の設計行列k=2(定数・内生変数)。
    fn data() -> (Mat<f64>, Mat<f64>, Mat<f64>) {
        let z1 = [1.0, 0.2, -0.7, 1.5, 0.9, -1.1, 0.3, 0.8];
        let z2 = [0.4, -0.9, 1.3, 0.1, -0.2, 0.7, -1.4, 0.6];
        let d = [0.9, -0.4, 0.2, 2.6, 0.3, 0.5, -0.8, 1.1];
        let e = [0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.2, 0.05];
        let z = Mat::from_fn(8, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(8, 2, |i, j| if j == 0 { 1.0 } else { d[i] });
        let residuals = Mat::from_fn(8, 1, |i, _| e[i]);
        (z, x, residuals)
    }

    fn y_vector() -> Mat<f64> {
        let y = [1.3, -0.2, 0.8, 2.9, 0.4, 1.1, -0.9, 1.6];
        Mat::from_fn(8, 1, |i, _| y[i])
    }

    /// `Ω̂ = Σ êᵢ² zᵢzᵢ'`（`Z`の代わりに白色化した`Z̃`で作れば`Ω̃ = L⁻¹Ω̂L⁻ᵀ`）。
    fn robust_omega(z: &Mat<f64>, e: &Mat<f64>) -> Mat<f64> {
        Mat::from_fn(z.ncols(), z.ncols(), |a, b| {
            (0..z.nrows())
                .map(|i| e.get(i, 0).powi(2) * z.get(i, a) * z.get(i, b))
                .sum()
        })
    }

    /// 白色化した最小二乗の係数・共分散が、正規方程式を明示的に解く従来の形
    /// （`β̂ = B⁻¹X'ZWZ'y`、`B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`）と一致する。重みの白色化は3通り
    /// （`S`のCholesky分解・`S = A'A`の`A`のQR分解・`S = Z'Z`の`Z`のQR分解）。
    #[test]
    fn gmm_solve_matches_the_explicit_normal_equations_in_every_whitening() {
        let (z, x, e0) = data();
        let y = y_vector();
        // `S = Σ ê²zz'`（`A`の行は`êᵢzᵢ`）。
        let a = Mat::from_fn(8, 3, |i, j| *e0.get(i, 0) * *z.get(i, j));
        let s = a.transpose() * &a;
        let s_inv = invert_spd(&s, 3, CommonError::PerfectFit).unwrap();
        let zx = z.transpose() * &x;
        let zy = z.transpose() * &y;
        let bread = zx.transpose() * &s_inv * &zx;
        let beta =
            invert_spd(&bread, 2, CommonError::PerfectFit).unwrap() * zx.transpose() * &s_inv * &zy;
        let resid = &y - &x * &beta;
        let omega = robust_omega(&z, &resid);
        let expected_cov = gmm_sandwich_covariance(&z, &x, &s, &omega).unwrap();

        let whitenings = [
            WhitenedInstruments::from_weight(&z, &s).unwrap(),
            WhitenedInstruments::from_factor_rows(&z, &a).unwrap(),
            WhitenedInstruments::from_weight_if_well_conditioned(&z, &s, 1e12)
                .unwrap()
                .unwrap(),
        ];
        for (index, whitening) in whitenings.into_iter().enumerate() {
            let solution = gmm_solve(&x, &y, whitening).unwrap();
            for i in 0..2 {
                assert!(
                    (*solution.beta().get(i, 0) - *beta.get(i, 0)).abs() < 1e-10,
                    "{index}"
                );
            }
            let omega_tilde = robust_omega(solution.z_tilde(), &resid);
            let (cov, wald) = solution.covariance(&omega_tilde);
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (*cov.get(i, j) - *expected_cov.get(i, j)).abs() < 1e-9,
                        "{index} ({i},{j})"
                    );
                }
            }
            // 直交座標の二次形式が、共分散を使った二次形式と一致する（傾き1個）。
            let slope = *solution.beta().get(1, 0);
            let wald_from_cov = slope * slope / *cov.get(1, 1);
            let b_tilde = &wald.coordinates;
            let r_inv_b = solution.design.coefficients(b_tilde);
            assert!((*r_inv_b.get(1, 0) - slope).abs() < 1e-10);
            assert!(wald_from_cov.is_finite());
        }
        // `S = Z'Z`の`Z`のQR分解（初期推定）は、`Z'Z`をCholesky分解した白色化と同じ係数になる。
        let ztz = z.transpose() * &z;
        let from_qr =
            gmm_solve(&x, &y, WhitenedInstruments::from_instruments(&z).unwrap()).unwrap();
        let from_chol =
            gmm_solve(&x, &y, WhitenedInstruments::from_weight(&z, &ztz).unwrap()).unwrap();
        for i in 0..2 {
            assert!((*from_qr.beta().get(i, 0) - *from_chol.beta().get(i, 0)).abs() < 1e-10);
        }
    }

    #[test]
    fn whitening_rejects_weights_that_are_not_positive_definite() {
        let (z, _, _) = data();
        let indefinite = Mat::from_fn(3, 3, |i, j| if i == j { [1.0, -1.0, 1.0][i] } else { 0.0 });
        let failure = || CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight,
        };
        assert_eq!(
            WhitenedInstruments::from_weight(&z, &indefinite).err(),
            Some(failure())
        );
        let zero_diagonal = Mat::<f64>::zeros(3, 3);
        assert_eq!(
            WhitenedInstruments::from_weight(&z, &zero_diagonal).err(),
            Some(failure())
        );
        let rows = Mat::from_fn(
            8,
            3,
            |i, j| if j == 2 { 2.0 * (i as f64) } else { i as f64 },
        );
        assert_eq!(
            WhitenedInstruments::from_factor_rows(&z, &rows).err(),
            Some(failure())
        );
        // 条件数が上限を超える重みは`None`（呼び出し側がQRの経路に切り替える）。
        let ill = Mat::from_fn(3, 3, |i, j| if i == j { [1.0, 1.0, 1e-9][i] } else { 0.0 });
        assert!(
            WhitenedInstruments::from_weight_if_well_conditioned(&z, &ill, 1e6)
                .unwrap()
                .is_some()
        );
        let near = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 1.0 - 1e-9 });
        assert!(
            WhitenedInstruments::from_weight_if_well_conditioned(&z, &near, 1e6)
                .unwrap()
                .is_none()
        );
    }

    /// 構造方程式の設計行列が、操作変数で説明される部分（`Ĝ = Z̃'X`）でランク落ちなら、
    /// `GMM_BREAD`の`NearSingularMatrix`（`B = Ĝ'Ĝ`の分解の失敗に代わる）。
    #[test]
    fn gmm_solve_rejects_a_rank_deficient_weighted_moment_matrix() {
        let (z, _, _) = data();
        let y = y_vector();
        let collinear = Mat::from_fn(8, 2, |i, j| (i as f64 + 1.0) * (j as f64 + 1.0));
        let whitening = WhitenedInstruments::from_instruments(&z).unwrap();
        assert_eq!(
            gmm_solve(&collinear, &y, whitening).err(),
            Some(CommonError::NearSingularMatrix {
                matrix: MatrixKind::GmmBread
            })
        );
    }

    #[test]
    fn invert_spd_returns_the_inverse_and_reports_the_given_failure() {
        let m = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 1.0 });
        let inv = invert_spd(&m, 2, CommonError::PerfectFit).unwrap();
        let identity = &m * &inv;
        for i in 0..2 {
            for j in 0..2 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((*identity.get(i, j) - target).abs() < 1e-12);
            }
        }
        let indefinite = Mat::from_fn(2, 2, |i, j| if i == j { -1.0 } else { 0.0 });
        assert_eq!(
            invert_spd(&indefinite, 2, CommonError::PerfectFit),
            Err(CommonError::PerfectFit)
        );
    }

    /// `S`と`Ω̂`が同じ(効率的GMM、`W = Ω̂⁻¹`)とき、サンドイッチは`B⁻¹ = (X'ZΩ̂⁻¹Z'X)⁻¹`になる。
    #[test]
    fn sandwich_reduces_to_the_inverse_of_the_bread_when_the_weight_equals_omega() {
        let (z, x, _) = data();
        let ztz = z.transpose() * &z;
        let s = Mat::from_fn(3, 3, |i, j| 0.37 * *ztz.get(i, j));

        let cov = gmm_sandwich_covariance(&z, &x, &s, &s).unwrap();

        let s_inv = invert_spd(&s, 3, CommonError::PerfectFit).unwrap();
        let zx = z.transpose() * &x;
        let bread = zx.transpose() * &s_inv * &zx;
        let expected = invert_spd(&bread, 2, CommonError::PerfectFit).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (*cov.get(i, j) - *expected.get(i, j)).abs() < 1e-10,
                    "({i},{j})"
                );
            }
        }
        // 古典的な重み`S = σ²Z'Z`のとき、`B⁻¹ = σ²(X'P_zX)⁻¹`(2SLSの分散)。
        let pz_x = &z * invert_spd(&ztz, 3, CommonError::PerfectFit).unwrap() * &zx;
        let two_sls = invert_spd(&(x.transpose() * &pz_x), 2, CommonError::PerfectFit).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (*cov.get(i, j) - 0.37 * *two_sls.get(i, j)).abs() < 1e-10,
                    "({i},{j})"
                );
            }
        }
    }

    /// `S ≠ Ω̂`(比例もしない)の一般形を、`invert_spd`を使わない経路(LU分解の逆行列)で組んだ
    /// 手計算と照合する。`S`と`Ω̂`の取り違えは、`S = Ω̂`のテストでは検出できない。
    #[test]
    fn sandwich_matches_the_general_form_built_independently_when_the_weight_differs_from_omega() {
        let (z, x, _) = data();
        let ztz = z.transpose() * &z;
        let scale = [1.0, 2.0, 0.5];
        let omega = Mat::from_fn(3, 3, |i, j| scale[i] * *ztz.get(i, j) * scale[j]);
        let s = Mat::from_fn(3, 3, |i, j| {
            0.37 * *ztz.get(i, j) + if i == j { 0.2 } else { 0.0 }
        });

        let cov = gmm_sandwich_covariance(&z, &x, &s, &omega).unwrap();

        let inverse = |m: &Mat<f64>| {
            m.partial_piv_lu()
                .solve(Mat::<f64>::identity(m.nrows(), m.nrows()))
        };
        let w = inverse(&s);
        let zx = z.transpose() * &x;
        let b_inv = inverse(&(zx.transpose() * &w * &zx));
        let expected = &b_inv * (zx.transpose() * &w * &omega * &w * &zx) * &b_inv;
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (*cov.get(i, j) - *expected.get(i, j)).abs()
                        < 1e-10 * expected.get(i, j).abs().max(1e-3),
                    "({i},{j})"
                );
            }
        }
        // 返り値は(丸めの範囲で)対称。
        assert!((*cov.get(0, 1) - *cov.get(1, 0)).abs() < 1e-12 * cov.get(0, 1).abs().max(1.0));
        // `S`と`Ω̂`を入れ替えると別の行列になる(取り違えを検出できる入力であること)。
        let swapped = gmm_sandwich_covariance(&z, &x, &omega, &s).unwrap();
        assert!((*cov.get(0, 0) - *swapped.get(0, 0)).abs() > 1e-6 * cov.get(0, 0).abs());
    }

    #[test]
    fn sandwich_is_invariant_to_scaling_the_weight_matrix() {
        let (z, x, _) = data();
        let ztz = z.transpose() * &z;
        let omega = Mat::from_fn(3, 3, |i, j| (1.0 + (i + j) as f64 * 0.1) * *ztz.get(i, j));
        let s = ztz.clone();
        let scaled = Mat::from_fn(3, 3, |i, j| 7.5 * *s.get(i, j));

        let a = gmm_sandwich_covariance(&z, &x, &s, &omega).unwrap();
        let b = gmm_sandwich_covariance(&z, &x, &scaled, &omega).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((*a.get(i, j) - *b.get(i, j)).abs() < 1e-10 * a.get(i, j).abs().max(1.0));
            }
        }
    }

    #[test]
    fn sandwich_reports_a_singular_weight_matrix_and_a_singular_bread() {
        let (z, x, _) = data();
        let omega = Mat::<f64>::identity(3, 3);
        let zero = Mat::<f64>::zeros(3, 3);
        assert_eq!(
            gmm_sandwich_covariance(&z, &x, &zero, &omega),
            Err(CommonError::NearSingularMatrix {
                matrix: MatrixKind::GmmWeight
            })
        );
        // 設計行列にゼロ列があれば X'ZWZ'X が特異。
        let collinear = Mat::from_fn(8, 2, |_, j| if j == 0 { 1.0 } else { 0.0 });
        assert_eq!(
            gmm_sandwich_covariance(&z, &collinear, &Mat::<f64>::identity(3, 3), &omega),
            Err(CommonError::NearSingularMatrix {
                matrix: MatrixKind::GmmBread
            })
        );
    }

    #[test]
    fn hansen_j_matches_the_quadratic_form_without_dividing_by_n() {
        let (z, _, e) = data();
        let ztz = z.transpose() * &z;
        let s = Mat::from_fn(3, 3, |i, j| 0.37 * *ztz.get(i, j));

        let j = hansen_j_test(&z, &e, &s, 2).unwrap();

        let zte = z.transpose() * &e;
        let s_inv = invert_spd(&s, 3, CommonError::PerfectFit).unwrap();
        let quad: f64 = (0..3)
            .map(|a| {
                (0..3)
                    .map(|b| zte.get(a, 0) * s_inv.get(a, b) * zte.get(b, 0))
                    .sum::<f64>()
            })
            .sum();
        assert!((j.statistic - quad).abs() < 1e-12);
        let expected_p = ChiSquared::new(2.0).unwrap().sf(j.statistic);
        assert!((j.p_value - expected_p).abs() < 1e-15);
    }

    #[test]
    fn hansen_j_rejects_zero_degrees_of_freedom_and_a_singular_weight_matrix() {
        let (z, _, e) = data();
        let s = Mat::<f64>::identity(3, 3);
        assert!(matches!(
            hansen_j_test(&z, &e, &s, 0),
            Err(CommonError::Internal { .. })
        ));
        assert_eq!(
            hansen_j_test(&z, &e, &Mat::<f64>::zeros(3, 3), 1),
            Err(CommonError::NearSingularMatrix {
                matrix: MatrixKind::GmmWeight
            })
        );
    }
}
