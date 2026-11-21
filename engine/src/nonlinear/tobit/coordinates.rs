//! 標準化した設計行列の直交座標（`Q = X̃R⁻¹`）。Tobitの最適化・共分散・全体Wald検定をこの座標で行う。
//!
//! `-H`は`X'WX`の形で、条件数が設計行列の2乗になる。ほぼ共線な傾きでは、`X`の座標で`H`の各要素を
//! 倍精度で組む丸めだけで統計量が桁違いにずれ、`-H`の分解も`cond(X)²`を保てない。`X̃ = QR`
//! （標準化した設計行列のピボットなしQR）として`Q`の座標`b = Rβ̃`で最適化すれば、`Q'Q`が定数倍の
//! 単位行列なので、`H`の条件数は共線性に依らず重み（打ち切りの度合い）だけで決まる。ニュートン法は
//! パラメータの線形変換に対して不変なので、厳密な演算では同じ反復になる。
//!
//! **`Q`は2の冪（約`√n`）倍する**: 標準化した列のノルムは`√n`で、`Q`の列は単位ノルムなので、そのままだと
//! 勾配が`√n`分小さくなり、`tol`（勾配のノルムとの比較。`bfgs`・`lbfgs`は`tol·n`）が`X̃`の座標より
//! `√n`倍ゆるむ。`√n`に近い2の冪を掛けて、勾配の大きさを`X̃`の座標に合わせる。2の冪は丸めなしで掛かる。
//!
//! 受け入れ範囲は他の手法と同じ（列を揃えた`R`の条件数、`OrthonormalDesign::new_checked`）。

use faer::Mat;

use crate::nonlinear::common::MleError;
use crate::shared::error::{CommonError, MatrixKind};
use crate::shared::linear_algebra::{RankDeficient, collinear_columns, column_names};
use crate::shared::orthonormal::{OrthonormalDesign, covariance_from_factor, solve_upper};

/// 打ち切りを無視したOLSの残差標準偏差が、`y`のスケールに対しこの割合以下なら、完全な当てはまりとして
/// `σ = 1`にフォールバックする。
const PERFECT_FIT_RELATIVE_SIGMA: f64 = 1e-8;

/// 標準化空間の係数`β̃`と`Q`の座標`b`の対応（`b = Rβ̃`、`R`は`Q`の倍率を含めた上三角のk×k）。
#[derive(Debug, Clone)]
pub(super) struct DesignCoordinates {
    /// `Q`の倍率`s`（2の冪）を含む`R/s`。`X̃ = (sQ)(R/s)`。
    r: Mat<f64>,
    /// `Q`に掛けた倍率`s`。
    scale: f64,
}

impl DesignCoordinates {
    /// 標準化した設計行列`x_std`（n×k、切片ありなら先頭が切片列）を`QR`に分解し、`s·Q`（n×k）と
    /// 座標の対応を返す。`x_std`は呼び出し側が先に標準化したもので、ランク判定・条件数のガードも
    /// この行列に対して行う。
    ///
    /// **共線な列は標準化前の`x`で特定する**: 切片ありの標準化は非切片列の平均を引くので、標準化後の
    /// 行列ではダミー変数の罠・定数列が切片を含まない形に見える。報告する列名は元の設計の列。
    ///
    /// # Errors
    /// - ランク落ち（完全な多重共線性、列を揃えた条件数が`1e12`超）: `MleError::SingularDesignMatrix`
    /// - 列を揃えた条件数が`5e7`超: `NearSingularMatrix`（`DESIGN_GRAM`。`CommonError`）
    pub(super) fn new(
        x_std: &Mat<f64>,
        x: &Mat<f64>,
        names: &[String],
    ) -> Result<(Self, Mat<f64>), MleError> {
        let design = OrthonormalDesign::new_checked(x_std).map_err(|err| match err {
            RankDeficient::Columns => MleError::SingularDesignMatrix {
                columns: column_names(&collinear_columns(x), names),
            },
            RankDeficient::GramMatrix => MleError::Common(CommonError::NearSingularMatrix {
                matrix: MatrixKind::DesignGram,
            }),
        })?;
        let n = x_std.nrows();
        let k = x_std.ncols();
        let (mut q, r) = design.into_parts();
        // `√n`に近い2の冪（`n ≥ 1`なので指数は0以上）。
        let exponent = (n as f64).sqrt().log2().round() as i32;
        let scale = 2.0_f64.powi(exponent);
        for j in 0..k {
            for i in 0..n {
                *q.get_mut(i, j) *= scale;
            }
        }
        let r = Mat::from_fn(k, k, |i, j| *r.get(i, j) / scale);
        Ok((Self { r, scale }, q))
    }

    /// 打ち切りを無視したOLS（`y`の`Q`への射影）から、最適化の初期値`(b, logσ)`を作る。
    /// `b = (sQ)'y / s²`、`σ`は残差の標準偏差。完全な当てはまり（残差が`y`のスケールに対して
    /// 極小）では`logσ`を0にフォールバックする（`log(0) = -∞`で最適化が壊れるのを避ける。
    /// あくまで初期値のヒューリスティックで、Tobitの推定値ではない）。
    pub(super) fn initial_params(&self, q: &Mat<f64>, y: &Mat<f64>) -> Vec<f64> {
        let n = q.nrows();
        let k = q.ncols();
        let s2 = self.scale * self.scale;
        let b: Vec<f64> = (0..k)
            .map(|j| (0..n).map(|i| *q.get(i, j) * *y.get(i, 0)).sum::<f64>() / s2)
            .collect();
        let sse: f64 = (0..n)
            .map(|i| {
                let fitted: f64 = (0..k).map(|j| *q.get(i, j) * b[j]).sum();
                (*y.get(i, 0) - fitted).powi(2)
            })
            .sum();
        let candidate_sigma = (sse / n as f64).sqrt();
        let y_scale = (0..n)
            .map(|i| y.get(i, 0).abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let sigma = if candidate_sigma > PERFECT_FIT_RELATIVE_SIGMA * y_scale {
            candidate_sigma
        } else {
            1.0
        };
        let mut params = b;
        params.push(sigma.ln());
        params
    }

    /// `Q`の座標のパラメータ`(b, logσ)`（k+1個）を、標準化空間の`(β̃, logσ̃)`に戻す
    /// （`β̃ = R⁻¹b`の後退代入。`logσ`はそのまま）。
    pub(super) fn standardized_params(&self, params: &[f64]) -> Vec<f64> {
        let k = self.r.nrows();
        let b = Mat::from_fn(k, 1, |i, _| params[i]);
        let beta = solve_upper(&self.r, b.as_ref());
        let mut out: Vec<f64> = (0..k).map(|i| *beta.get(i, 0)).collect();
        out.push(params[k]);
        out
    }

    /// `Q`の座標の共分散（`(k+1)×(k+1)`。`b`と`logσ`）を、標準化空間`(β̃, logσ̃)`の共分散に戻す:
    /// `V_β̃ = R⁻¹ V_b R⁻ᵀ`（`logσ`の行・列は変えない）。三角行列の後退代入で両側から解き、
    /// `R⁻¹`を明示的に作らない。
    pub(super) fn standardized_covariance(&self, cov: &Mat<f64>) -> Mat<f64> {
        let k = self.r.nrows();
        let full = Mat::from_fn(k + 1, k + 1, |i, j| {
            if i < k && j < k {
                *self.r.get(i, j)
            } else if i == j {
                1.0
            } else {
                0.0
            }
        });
        covariance_from_factor(&full, cov)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(n: usize) -> (Mat<f64>, Vec<String>) {
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => ((i * 7) % 11) as f64 - 5.0,
            _ => ((i * 5 + 3) % 13) as f64 * 0.5,
        });
        (x, ["const", "a", "b"].map(String::from).to_vec())
    }

    #[test]
    fn q_times_r_reconstructs_the_standardized_design() {
        let (x, names) = design(40);
        let (coordinates, q) = DesignCoordinates::new(&x, &x, &names).unwrap();
        let back = &q * &coordinates.r;
        for i in 0..40 {
            for j in 0..3 {
                assert!((back.get(i, j) - x.get(i, j)).abs() < 1e-12);
            }
        }
    }

    /// `Q`の倍率は`√n`に近い2の冪で、`Q'Q`はその2乗倍の単位行列（`X̃`の座標と勾配の大きさを合わせる）。
    #[test]
    fn q_is_scaled_by_a_power_of_two_near_the_square_root_of_n() {
        for n in [4, 40, 600, 100_000] {
            let (x, names) = design(n);
            let (coordinates, q) = DesignCoordinates::new(&x, &x, &names).unwrap();
            let exponent = coordinates.scale.log2();
            assert_eq!(exponent.fract(), 0.0, "n={n}");
            assert!(
                (coordinates.scale / (n as f64).sqrt()).ln().abs() < 0.5,
                "n={n}"
            );
            let gram = q.transpose() * &q;
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j {
                        coordinates.scale.powi(2)
                    } else {
                        0.0
                    };
                    assert!(
                        (gram.get(i, j) - expected).abs() < 1e-10 * coordinates.scale.powi(2),
                        "n={n} ({i}, {j})"
                    );
                }
            }
        }
    }

    #[test]
    fn standardized_params_and_covariance_invert_the_coordinate_map() {
        let (x, names) = design(40);
        let (coordinates, _) = DesignCoordinates::new(&x, &x, &names).unwrap();
        // `β̃`から`b = Rβ̃`を作り、戻すと`β̃`。
        let beta = [0.7, -1.3, 2.1];
        let b: Vec<f64> = (0..3)
            .map(|i| (i..3).map(|j| *coordinates.r.get(i, j) * beta[j]).sum())
            .chain([0.25])
            .collect();
        let back = coordinates.standardized_params(&b);
        for (value, expected) in back.iter().zip(beta) {
            assert!((value - expected).abs() < 1e-12);
        }
        assert_eq!(back[3], 0.25);
        // `V_b = R V_β̃ Rᵀ`（`logσ`との共分散も`R`だけ掛かる）を戻すと`V_β̃`。
        let v_beta = Mat::from_fn(4, 4, |i, j| {
            if i == j {
                1.0 + i as f64
            } else {
                0.1 * (1 + i + j) as f64
            }
        });
        let t = Mat::from_fn(4, 4, |i, j| {
            if i < 3 && j < 3 {
                *coordinates.r.get(i, j)
            } else if i == j {
                1.0
            } else {
                0.0
            }
        });
        let v_b = &t * &v_beta * t.transpose();
        let recovered = coordinates.standardized_covariance(&v_b);
        for i in 0..4 {
            for j in 0..4 {
                assert!((recovered.get(i, j) - v_beta.get(i, j)).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn initial_params_are_the_ols_coefficients_in_q_coordinates() {
        let (x, names) = design(40);
        let (coordinates, q) = DesignCoordinates::new(&x, &x, &names).unwrap();
        let beta = [0.5, 0.25, -0.75];
        let y = Mat::from_fn(40, 1, |i, _| {
            (0..3).map(|j| *x.get(i, j) * beta[j]).sum::<f64>() + 0.1 * ((i * 3) % 5) as f64
        });
        let params = coordinates.initial_params(&q, &y);
        let back = coordinates.standardized_params(&params);
        // 通常のOLS（正規方程式）と一致する。
        let xtx = x.transpose() * &x;
        let xty = x.transpose() * &y;
        use faer::Side;
        use faer::prelude::Solve;
        let ols = xtx.llt(Side::Lower).unwrap().solve(&xty);
        for (j, value) in back.iter().take(3).enumerate() {
            assert!((value - ols.get(j, 0)).abs() < 1e-10, "j={j}");
        }
    }

    #[test]
    fn initial_params_fall_back_to_sigma_one_for_a_perfect_fit() {
        let (x, names) = design(40);
        let (coordinates, q) = DesignCoordinates::new(&x, &x, &names).unwrap();
        let y = Mat::from_fn(40, 1, |i, _| 2.0 * *x.get(i, 1) + 1.0);
        let params = coordinates.initial_params(&q, &y);
        assert_eq!(params[3], 0.0);
    }

    #[test]
    fn exact_collinearity_is_reported_with_the_columns_of_the_original_design() {
        let (mut x, names) = design(40);
        x = Mat::from_fn(40, 3, |i, j| {
            if j == 2 {
                2.0 * *x.get(i, 1)
            } else {
                *x.get(i, j)
            }
        });
        let err = DesignCoordinates::new(&x, &x, &names).unwrap_err();
        assert!(
            matches!(&err, MleError::SingularDesignMatrix { columns } if columns == &["a", "b"]),
            "{err:?}"
        );
    }

    #[test]
    fn a_design_beyond_the_condition_limit_is_near_singular_design_gram() {
        let (mut x, names) = design(40);
        let a: Vec<f64> = (0..40).map(|i| *x.get(i, 1)).collect();
        // `b = a + 1e-9·ノイズ`: 列を揃えた条件数が約`1e9`（`5e7`超、`1e12`以下）。
        x = Mat::from_fn(40, 3, |i, j| match j {
            2 => a[i] + 1e-9 * (((i * 3) % 7) as f64 - 3.0),
            _ => *x.get(i, j),
        });
        let err = DesignCoordinates::new(&x, &x, &names).unwrap_err();
        assert!(
            matches!(
                err,
                MleError::Common(CommonError::NearSingularMatrix {
                    matrix: MatrixKind::DesignGram
                })
            ),
            "{err:?}"
        );
    }
}
