//! GMMの係数の分散共分散行列(サンドイッチの代数)とHansen J統計量。行列だけを受け取る。
//!
//! 重み行列`S`・モーメントの分散共分散行列`Ω̂`の組み立て(`cov_type`ごとに違う)と、小標本補正の
//! 分母(構造方程式の係数の数など)は呼び出し側が決める。ここは、それらを受け取った後の代数で、
//! IVのGMMが使い、固定効果つきのGMM(within変換後のデータと自前の`Ω̂`)も同じ関数を呼べるようにしてある。`IvInput`・系統のエラー型には依存しない。

use faer::prelude::Solve;
use faer::{Mat, Side};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::error::{CommonError, MatrixKind};

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

/// Hansen J統計量`J = (Z'ê)'S⁻¹(Z'ê)`(`n`で割らない。理由は[`hansen_j_test`])。自由度を持たないので、
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
