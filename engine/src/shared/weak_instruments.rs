//! 弱操作変数の診断統計量(Cragg-Donald)。行列だけを受け取る。
//!
//! 2SLS(`iv::two_sls`)・GMM(`iv::gmm`)が`IvInput`から行列を組んで呼び、固定効果つきの2SLSも
//! within変換後のデータと吸収したパラメータ数`n_absorbed`で呼べるようにしてある。重みつきの回帰は、
//! 呼び出し側がすべての行列(定数列を含む)に`√w`を掛けて渡す。

use faer::Mat;
use faer::Side;
use faer::prelude::SolveLstsq;

use super::error::MatrixKind;
use super::linear_algebra::{
    checked_col_piv_qr, column_norm, ensure_well_conditioned_symmetric_matrix, forward_substitute,
    unit_columns,
};

/// Cragg-Donald Wald F統計量: 内生変数`Y`の第一段階の係数行列のランク統計量の最小固有値を、除外操作変数の
/// 数`q`で割って自由度を合わせたもの。Stock-Yogo(2005)が臨界値を表にした`g_min`統計量で、Stataの
/// `ivreg2`の"Cragg-Donald Wald F statistic"と同じ定義(数値はlinearmodelsの`cragg_donald_f`と照合)。内生変数が1個なら、除外操作変数の係数に対する
/// 古典的な部分F統計量に一致する。
///
/// 定義: 外生変数`X`で`Y`と`Z`を直交化した`Ỹ`・`Z̃`について、`Ỹ`を`Z̃`に回帰した当てはめ値`Ŷ`と残差`E`から
///
/// ```text
/// A = Ŷ'Ŷ = Ỹ'P Ỹ,   S = E'E = Ỹ'M Ỹ,   G = S^{-1/2}' A S^{-1/2}
/// CD = λ_min(G) · (n − k_exog − q − n_absorbed) / q
/// ```
///
/// (`S = LL'`のCholesky分解で`G = L⁻¹ A L⁻ᵀ`)。`n_absorbed`は固定効果で吸収したパラメータ数
/// (定数項を除く。固定効果が無ければ`0`)で、`S`の自由度から引く。
///
/// - `y_endog`: 内生変数(n×k)、`x_exog`: 外生変数(n×k_exog。定数項を使うなら定数列を含める。
///   平均を引かず、他の列と同じく偏回帰で落とす)、`z`: 除外操作変数(n×q)。
/// - 等分散の統計量で、`cov_type`には依存しない。p値は無い(検定ではない)。
/// - 統計量は列ごとのスケールによらない。ランクの判定の前に、各列をノルム1にそろえる。
///
/// 統計量が定義できないときは`None`を返す: 内生変数が無い、`q < k`(識別不足)、自由度が1未満、
/// `X`がフルランクでない、`Z`の列が`X`の張る空間にほぼ入っている(直交化で元のノルムの`√ε`未満に
/// なる)、`Z̃`がフルランクでない、`Y`の列が`X`の張る空間にほぼ入っている、`S`が数値的に特異
/// (内生変数が共線)、いずれかの内生変数について第一段階が完全フィット、入力に非有限値がある。
pub(crate) fn cragg_donald(
    y_endog: &Mat<f64>,
    x_exog: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
) -> Option<f64> {
    let (n, k, q, k_exog) = (y_endog.nrows(), y_endog.ncols(), z.ncols(), x_exog.ncols());
    debug_assert_eq!(z.nrows(), n);
    debug_assert_eq!(x_exog.nrows(), n);
    if k == 0 || q < k {
        return None;
    }
    let df = n.checked_sub(k_exog + q + n_absorbed).filter(|&d| d >= 1)?;

    let (y_tilde, z_tilde) = orthogonalize_unit(y_endog, x_exog, z)?;

    // Ỹ を Z̃ に回帰する。
    let qr_z = checked_col_piv_qr(&z_tilde).ok()?;
    let fitted = &z_tilde * qr_z.solve_lstsq(&y_tilde);
    let residuals = &y_tilde - &fitted;
    let a = fitted.transpose() * &fitted;
    let s = residuals.transpose() * &residuals;

    // Ỹはノルム1なので、`S_jj`は第一段階の残差平方和の割合。どれかの内生変数が完全フィット
    // (`SSR <= q·ε·SST`、`shared::goodness_of_fit::is_perfect_fit`と同じ基準)なら統計量は無意味。
    let tolerance = (q as f64) * f64::EPSILON;
    if (0..k).any(|j| s.get(j, j).is_nan() || *s.get(j, j) <= tolerance) {
        return None;
    }
    // 内生変数が共線なら S が特異になる。非ピボットのCholeskyは検出できないので、固有値で確かめる。
    ensure_well_conditioned_symmetric_matrix(&s, k, MatrixKind::DesignGram).ok()?;

    // G = L⁻¹ A L⁻ᵀ。T = L⁻¹A を前進代入で求め、G = L⁻¹ Tᵀ。
    let llt = s.llt(Side::Lower).ok()?;
    let l = llt.L();
    let t = forward_substitute(l, &a);
    let mut g = forward_substitute(l, &t.transpose().to_owned());
    // 丸めの非対称を平均して対称にする。
    g = Mat::from_fn(k, k, |i, j| 0.5 * (*g.get(i, j) + *g.get(j, i)));
    let eigenvalues = g.self_adjoint_eigenvalues(Side::Lower).ok()?;
    let min_eigenvalue = eigenvalues.into_iter().fold(f64::INFINITY, f64::min);
    if !min_eigenvalue.is_finite() {
        return None;
    }
    // `G`は半正定値。操作変数が無関係だと最小固有値が丸めで`-1e-17`程度の負になりうる。
    Some(min_eigenvalue.max(0.0) * (df as f64) / (q as f64))
}

/// `[Y | Z]`を外生変数で直交化し、`Ỹ`・`Z̃`の各列をノルム1にそろえて返す(定数列も通常の列として
/// 落とす)。ランクの判定がスケールに依存しないよう、外生変数の列もノルム1にそろえてからQRにかける。
///
/// 次のとき`None`: `X`がフルランクでない、`Y`または`Z`の列が`X`の張る空間にほぼ入っている
/// (直交化で元のノルムの`√ε`未満になる)、入力に非有限値がある。
pub(super) fn orthogonalize_unit(
    y_endog: &Mat<f64>,
    x_exog: &Mat<f64>,
    z: &Mat<f64>,
) -> Option<(Mat<f64>, Mat<f64>)> {
    let (k, q, k_exog) = (y_endog.ncols(), z.ncols(), x_exog.ncols());
    let w = hcat(y_endog, z);
    let w_tilde = if k_exog == 0 {
        w.clone()
    } else {
        let x_unit = unit_columns(x_exog);
        let qr_x = checked_col_piv_qr(&x_unit).ok()?;
        &w - &x_unit * qr_x.solve_lstsq(&w)
    };
    // 直交化で元のノルムの`√ε`未満にまで縮む列は、`X`の張る空間にほぼ入っている。
    let sqrt_eps = f64::EPSILON.sqrt();
    for j in 0..w.ncols() {
        let (remaining, original) = (column_norm(&w_tilde, j), column_norm(&w, j));
        // NaNは比較が常に偽になるので、明示的に弾く。
        if remaining.is_nan() || original.is_nan() || remaining <= sqrt_eps * original {
            return None;
        }
    }
    let y_tilde = unit_columns(&w_tilde.subcols(0, k).to_owned());
    let z_tilde = unit_columns(&w_tilde.subcols(k, q).to_owned());
    Some((y_tilde, z_tilde))
}

/// [`orthogonalize_unit`]の結果を、各列の二乗平均が1になるようにそろえたもの(`Q = Z̃'Z̃/n`の対角が1)。
/// ランク統計量(Kleibergen-Paap・有効F)は列のスケールに依らないので、Gram行列の条件が観測数に
/// 依らないようにそろえる。`None`の条件は`orthogonalize_unit`と同じ。
pub(super) fn rms_orthogonalized(
    y_endog: &Mat<f64>,
    x_exog: &Mat<f64>,
    z: &Mat<f64>,
) -> Option<(Mat<f64>, Mat<f64>)> {
    let (y_unit, z_unit) = orthogonalize_unit(y_endog, x_exog, z)?;
    let root_n = (y_endog.nrows() as f64).sqrt();
    let scale = |m: &Mat<f64>| Mat::from_fn(m.nrows(), m.ncols(), |i, j| *m.get(i, j) * root_n);
    Some((scale(&y_unit), scale(&z_unit)))
}

/// 完全フィット(最小の正準相関が1、または一次段階の`R²`が1)とみなす`1 − R²`の下限。
/// Kleibergen-Paapと有効Fが共有する。
pub(super) const EXACT_FIT_TOLERANCE: f64 = 1.4210854715202004e-14;

/// 列を横に並べる。
fn hcat(left: &Mat<f64>, right: &Mat<f64>) -> Mat<f64> {
    let k = left.ncols();
    Mat::from_fn(left.nrows(), k + right.ncols(), |i, j| {
        if j < k {
            *left.get(i, j)
        } else {
            *right.get(i, j - k)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::instrumental_variables::partial_f_statistic;
    use crate::shared::least_squares::{least_squares, residual_sum_of_squares};
    use faer::prelude::Solve;

    /// n=12。定数・外生`x`(`k_exog=2`)、除外操作変数`z1,z2,z3`(`q=3`)、内生変数`d1,d2`。
    fn data() -> (Mat<f64>, Mat<f64>, Mat<f64>) {
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

    fn column(m: &Mat<f64>, j: usize) -> Mat<f64> {
        Mat::from_fn(m.nrows(), 1, |i, _| *m.get(i, j))
    }

    #[test]
    fn with_one_endogenous_variable_it_equals_the_partial_f_statistic() {
        let (y, exog, z) = data();
        let d = column(&y, 0);
        let full = Mat::from_fn(12, 5, |i, j| {
            if j < 2 {
                *exog.get(i, j)
            } else {
                *z.get(i, j - 2)
            }
        });
        let unrestricted = least_squares(&full, &d).unwrap();
        let ssr_u = residual_sum_of_squares(&unrestricted.residuals);

        let cd = cragg_donald(&d, &exog, &z, 0).unwrap();

        let partial_f = partial_f_statistic(&d, &exog, ssr_u, 5, 3, 0).unwrap();
        assert!((cd - partial_f).abs() < 1e-9 * partial_f.abs());
    }

    #[test]
    fn with_two_endogenous_variables_it_matches_the_closed_form_of_the_smaller_root() {
        let (y, exog, z) = data();
        // 独立な組み立て: 偏回帰を正規方程式で行い、2×2の S⁻¹A の固有値を二次方程式で解く。
        let partial_out = |m: &Mat<f64>| {
            let xtx = exog.transpose() * &exog;
            let beta = xtx.partial_piv_lu().solve(exog.transpose() * m);
            m - &exog * beta
        };
        let (yt, zt) = (partial_out(&y), partial_out(&z));
        let ztz = zt.transpose() * &zt;
        let b = ztz.partial_piv_lu().solve(zt.transpose() * &yt);
        let fitted = &zt * b;
        let e = &yt - &fitted;
        let a = fitted.transpose() * &fitted;
        let s = e.transpose() * &e;
        let det_s = s.get(0, 0) * s.get(1, 1) - s.get(0, 1) * s.get(1, 0);
        // M = S⁻¹ A (2×2)
        let m = |i: usize, j: usize| {
            let inv = [[*s.get(1, 1), -*s.get(0, 1)], [-*s.get(1, 0), *s.get(0, 0)]];
            (inv[i][0] * a.get(0, j) + inv[i][1] * a.get(1, j)) / det_s
        };
        let (tr, det) = (m(0, 0) + m(1, 1), m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));
        let lambda_min = 0.5 * (tr - (tr * tr - 4.0 * det).sqrt());
        let df = 12.0 - 2.0 - 3.0;
        let expected = lambda_min * df / 3.0;

        let cd = cragg_donald(&y, &exog, &z, 0).unwrap();

        assert!(
            (cd - expected).abs() < 1e-8 * expected.abs(),
            "{cd} {expected}"
        );
        // 最小固有値なので、各内生変数単独の部分Fより大きくならない(ランク統計量の性質)。
        for j in 0..2 {
            let single = cragg_donald(&column(&y, j), &exog, &z, 0).unwrap();
            assert!(cd <= single * (1.0 + 1e-12), "cd={cd} single[{j}]={single}");
        }
    }

    #[test]
    fn n_absorbed_enters_only_the_degrees_of_freedom_factor() {
        let (y, exog, z) = data();
        let base = cragg_donald(&y, &exog, &z, 0).unwrap();

        let absorbed = cragg_donald(&y, &exog, &z, 2).unwrap();

        // df は 12-2-3=7 から 12-2-3-2=5 になる。
        assert!((absorbed - base * 5.0 / 7.0).abs() < 1e-10 * base);
    }

    /// Frisch-Waugh-Lovell: 個体ダミーを外生変数に入れた計算(`n_absorbed=0`)と、個体内で平均を引いた
    /// データに`n_absorbed=ダミーの数`を渡した計算が一致する。
    #[test]
    fn dummies_in_the_exogenous_variables_equal_the_demeaned_data_with_absorbed_parameters() {
        let (y, exog, z) = data();
        let (groups, per) = (3, 4);
        let dummies = Mat::from_fn(12, groups, |i, g| f64::from(i / per == g));
        let x_col = column(&exog, 1);
        let with_dummies = Mat::from_fn(12, groups + 1, |i, j| {
            if j < groups {
                *dummies.get(i, j)
            } else {
                *x_col.get(i, 0)
            }
        });
        let a = cragg_donald(&y, &with_dummies, &z, 0).unwrap();

        let demean = |m: &Mat<f64>| {
            Mat::from_fn(12, m.ncols(), |i, j| {
                let g = i / per;
                let mean: f64 =
                    (g * per..(g + 1) * per).map(|r| *m.get(r, j)).sum::<f64>() / per as f64;
                *m.get(i, j) - mean
            })
        };
        let x_only = demean(&x_col);
        let b = cragg_donald(&demean(&y), &x_only, &demean(&z), groups).unwrap();

        assert!(a > 1e-3);
        assert!((a - b).abs() < 1e-8 * a, "{a} {b}");
    }

    #[test]
    fn it_is_invariant_to_scaling_columns_and_to_the_scale_of_a_common_row_weight() {
        let (y, exog, z) = data();
        let base = cragg_donald(&y, &exog, &z, 0).unwrap();

        let scaled_y = Mat::from_fn(12, 2, |i, j| *y.get(i, j) * [3.0, 0.01][j]);
        let scaled_z = Mat::from_fn(12, 3, |i, j| *z.get(i, j) * [5.0, 1.0, 0.2][j]);
        let scaled = cragg_donald(&scaled_y, &exog, &scaled_z, 0).unwrap();

        assert!((scaled - base).abs() < 1e-8 * base);
    }

    #[test]
    fn it_returns_none_when_the_statistic_is_not_defined() {
        let (y, exog, z) = data();
        // 内生変数が無い。
        assert!(cragg_donald(&Mat::<f64>::zeros(12, 0), &exog, &z, 0).is_none());
        // 識別不足(q < k): 除外操作変数1個に内生変数2個。
        let z_one = column(&z, 0);
        assert!(cragg_donald(&y, &exog, &z_one, 0).is_none());
        // 自由度が1未満(12-2-3-7=0)。
        assert!(cragg_donald(&y, &exog, &z, 7).is_none());
        assert!(cragg_donald(&y, &exog, &z, 100).is_none());
        // 内生変数が共線。
        let collinear = Mat::from_fn(12, 2, |i, j| *y.get(i, 0) * [1.0, 2.0][j]);
        assert!(cragg_donald(&collinear, &exog, &z, 0).is_none());
        // 除外操作変数が外生変数と共線(直交化後に列が消える)。
        let z_in_span = Mat::from_fn(12, 3, |i, j| {
            if j == 0 {
                *exog.get(i, 1)
            } else {
                *z.get(i, j)
            }
        });
        assert!(cragg_donald(&y, &exog, &z_in_span, 0).is_none());
        // 第一段階が完全フィット(内生変数が外生変数と操作変数の厳密な線形結合)。
        let exact = Mat::from_fn(12, 1, |i, _| 2.0 * z.get(i, 0) - 0.5 * z.get(i, 1) + 1.0);
        assert!(cragg_donald(&exact, &exog, &z, 0).is_none());
    }

    // linearmodels 7.1の`cragg_donald_f`を、このモジュールの`data()`と同じ入力で実行した値
    // (`linearmodels.iv.results.cragg_donald_f(endog, instr, exog)`、定数列は`exog`に含める)。
    const LINEARMODELS_CD_F_TWO_ENDOGENOUS: f64 = 5.661334354194701;
    const LINEARMODELS_CD_F_ONE_ENDOGENOUS: f64 = 47.75015872855707;

    #[test]
    fn it_matches_the_linearmodels_values_for_the_same_data() {
        let (y, exog, z) = data();
        let two = cragg_donald(&y, &exog, &z, 0).unwrap();
        let one = cragg_donald(&column(&y, 0), &exog, &z, 0).unwrap();
        assert!(
            (two - LINEARMODELS_CD_F_TWO_ENDOGENOUS).abs() < 1e-10 * two,
            "{two}"
        );
        assert!(
            (one - LINEARMODELS_CD_F_ONE_ENDOGENOUS).abs() < 1e-10 * one,
            "{one}"
        );
    }

    /// 無関係な操作変数(弱い)でも、統計量は負にならない(丸めで最小固有値が負になりうる)。
    #[test]
    fn it_is_not_negative_when_the_instruments_are_irrelevant() {
        let (_, exog, z) = data();
        // 外生変数と操作変数のどれとも相関しない内生変数を、決定的な擬似乱数で作る。
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        for _ in 0..50 {
            let y = Mat::from_fn(12, 2, |_, _| next());
            if let Some(cd) = cragg_donald(&y, &exog, &z, 0) {
                assert!(cd >= 0.0 && cd.is_finite(), "{cd}");
            }
        }
    }

    /// 除外操作変数が1個で、それが外生変数の張る空間にある(直交化で丸め誤差しか残らない)。
    #[test]
    fn it_returns_none_for_a_single_excluded_instrument_in_the_span_of_the_exogenous_variables() {
        let (y, exog, _) = data();
        let d = column(&y, 0);
        let z_in_span = Mat::from_fn(12, 1, |i, _| 3.0 * exog.get(i, 1) - 2.0 * exog.get(i, 0));
        assert!(cragg_donald(&d, &exog, &z_in_span, 0).is_none());
        // 内生変数が外生変数の張る空間にある。
        let d_in_span = Mat::from_fn(12, 1, |i, _| 0.5 * exog.get(i, 1) + 4.0);
        let (_, _, z) = data();
        assert!(cragg_donald(&d_in_span, &exog, &z, 0).is_none());
    }

    /// 列のスケールは、ランクの判定にも効かない(単位が極端に違う操作変数)。
    #[test]
    fn it_is_invariant_to_extreme_column_units() {
        let (y, exog, z) = data();
        let base = cragg_donald(&y, &exog, &z, 0).unwrap();
        let scaled_z = Mat::from_fn(12, 3, |i, j| *z.get(i, j) * [1e-14, 1.0, 1e9][j]);
        let scaled_exog = Mat::from_fn(12, 2, |i, j| *exog.get(i, j) * [1.0, 1e-7][j]);
        let scaled = cragg_donald(&y, &scaled_exog, &scaled_z, 0).unwrap();
        assert!((scaled - base).abs() < 1e-7 * base, "{scaled} {base}");
    }

    /// 丁度識別(`q == k`)でも定義できる。
    #[test]
    fn it_is_defined_when_just_identified() {
        let (y, exog, z) = data();
        let two_instruments = Mat::from_fn(12, 2, |i, j| *z.get(i, j));
        let cd = cragg_donald(&y, &exog, &two_instruments, 0).unwrap();
        assert!(cd > 0.0 && cd.is_finite());
    }

    /// 内生変数の一方だけが第一段階で完全フィットでも、`None`になる(トレース全体では見逃す)。
    #[test]
    fn it_returns_none_when_only_one_of_the_endogenous_variables_fits_perfectly() {
        let (y, exog, z) = data();
        let mixed = Mat::from_fn(12, 2, |i, j| {
            if j == 0 {
                0.4 * z.get(i, 0) - 0.7 * z.get(i, 2) + 1.5
            } else {
                *y.get(i, 1)
            }
        });
        assert!(cragg_donald(&mixed, &exog, &z, 0).is_none());
        // 小さいスケールの内生変数が完全フィットでも同じ。
        let tiny = Mat::from_fn(12, 2, |i, j| {
            if j == 0 {
                1e-9 * (0.4 * z.get(i, 0) - 0.7 * z.get(i, 2))
            } else {
                *y.get(i, 1)
            }
        });
        assert!(cragg_donald(&tiny, &exog, &z, 0).is_none());
    }

    #[test]
    fn it_returns_none_for_non_finite_inputs() {
        let (y, exog, z) = data();
        for bad in [f64::NAN, f64::INFINITY] {
            let mut y_bad = y.clone();
            *y_bad.get_mut(3, 0) = bad;
            assert!(cragg_donald(&y_bad, &exog, &z, 0).is_none(), "y {bad}");
            let mut z_bad = z.clone();
            *z_bad.get_mut(5, 1) = bad;
            assert!(cragg_donald(&y, &exog, &z_bad, 0).is_none(), "z {bad}");
            let mut x_bad = exog.clone();
            *x_bad.get_mut(7, 1) = bad;
            assert!(cragg_donald(&y, &x_bad, &z, 0).is_none(), "x {bad}");
        }
    }

    #[test]
    fn it_works_without_exogenous_variables() {
        let (y, _, z) = data();
        let none = Mat::<f64>::zeros(12, 0);
        let d = column(&y, 0);
        let ssr_u = residual_sum_of_squares(&least_squares(&z, &d).unwrap().residuals);

        let cd = cragg_donald(&d, &none, &z, 0).unwrap();

        let partial_f = partial_f_statistic(&d, &none, ssr_u, 3, 3, 0).unwrap();
        assert!((cd - partial_f).abs() < 1e-9 * partial_f.abs());
    }
}
