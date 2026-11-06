//! 操作変数法の行列を受け取る部品（2SLSの点推定・Sargan過剰識別検定・弱操作変数の部分F統計量・
//! Wu-Hausman検定の拡張回帰の列の組み立て）。
//!
//! 2SLS（`iv::two_sls`）が使い、固定効果つきの2SLS（within変換後のデータに同じ計算を当てはめる）も
//! 同じ関数を呼べるようにしてある。
//! `IvInput`・`OlsEstimator`には依存せず、行列と数だけを受け取る。呼び出し側が決めるもの:
//! 定数項の扱い（定数列を行列に含めるか）、Sarganの`σ̂²`の分母、固定効果で吸収したパラメータ数
//! （`n_absorbed`。自由度から引く）、推定の失敗を自系統のエラー型に直すこと。

use faer::prelude::Solve;
use faer::{Mat, Side};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::error::CommonError;
use super::least_squares::{least_squares, residual_sum_of_squares};
use super::linear_algebra::RankDeficient;

/// [`two_sls_point_estimate`]の結果。
#[derive(Debug)]
pub(crate) struct TwoSlsPointEstimate {
    /// 係数 (k, 1)
    pub(crate) params: Mat<f64>,
    /// 第二段階の設計行列`X̂`についての`(X̂'X̂)⁻¹` (k, k)。標準誤差のサンドイッチの「パン」に使う。
    pub(crate) xtx_hat_inv: Mat<f64>,
    /// 構造残差 `e = y - Xβ̂` (n, 1)。`X`は内生変数の実測値で、第二段階の残差`y - X̂β̂`ではない。
    pub(crate) residuals: Mat<f64>,
    /// 構造残差の二乗和 `Σ eᵢ²`
    pub(crate) ssr: f64,
}

/// 2SLSの点推定。`y`を`X̂`（第一段階の予測値で内生変数を置き換えた設計行列）に回帰して係数を求め
/// （列ピボットQR）、残差は元の設計行列`X`で作る。
///
/// `x`と`x_hat`は同じ列の並びで、`x`の内生変数の列が`x_hat`では予測値に置き換わっている
/// （外生変数の列は同じ）。
///
/// # Errors
/// `x_hat`がフルカラムランクでない、または`X̂'X̂`のCholesky分解が丸めで失敗した場合に
/// [`RankDeficient`]を返す。
pub(crate) fn two_sls_point_estimate(
    x: &Mat<f64>,
    x_hat: &Mat<f64>,
    y: &Mat<f64>,
) -> Result<TwoSlsPointEstimate, RankDeficient> {
    let fit = least_squares(x_hat, y)?;
    let residuals = y - x * &fit.params;
    let ssr = residual_sum_of_squares(&residuals);
    Ok(TwoSlsPointEstimate {
        params: fit.params,
        xtx_hat_inv: fit.xtx_inv,
        residuals,
        ssr,
    })
}

/// Sargan過剰識別検定の統計量とp値。`df`はχ²分布の自由度。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SarganTest {
    pub(crate) statistic: f64,
    pub(crate) p_value: f64,
}

/// Sargan過剰識別検定: `e'Z(Z'Z)⁻¹Z'e / σ̂²`を`χ²(df)`と比べる（`Z`は外生変数と除外操作変数を
/// 並べた行列）。
///
/// `sigma2`（分母）は呼び出し側が決める。2SLSは`e'e/n`（`fixest`の`sargan`と一致）。
/// `df = 除外操作変数の数 - 内生変数の数`は`1`以上であること（丁度識別は検定が無いので、
/// 呼び出し側が先に分岐する）。
///
/// # Errors
/// `Z'Z`のCholesky分解が失敗（呼び出し側で第一段階が同じ`Z'Z`を分解済みなら到達しない）、
/// または`df == 0`のとき`CommonError::Internal`。
pub(crate) fn sargan_test(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    sigma2: f64,
    df: usize,
) -> Result<SarganTest, CommonError> {
    let ztz = z.transpose() * z;
    let zte = z.transpose() * residuals;
    let llt_ztz = ztz
        .llt(Side::Lower)
        .map_err(|_| CommonError::internal("Z'Z could not be factorized in the Sargan test"))?;
    let ztz_inv_zte = llt_ztz.solve(&zte);
    let quad: f64 = (0..z.ncols())
        .map(|i| (*zte.get(i, 0)) * (*ztz_inv_zte.get(i, 0)))
        .sum();
    let statistic = quad / sigma2;
    let chi2 = ChiSquared::new(df as f64).map_err(|e| CommonError::internal(e.to_string()))?;
    let p_value = chi2.sf(statistic);
    Ok(SarganTest { statistic, p_value })
}

/// 弱操作変数診断の部分F統計量: 除外操作変数`q`個の係数がゼロという制限を検定する、常に等分散前提の
/// 古典的ネストF検定 `F = [(SSR_r - SSR_u)/q] / [SSR_u/(n - k_u - n_absorbed)]`。
///
/// - `ssr_unrestricted`・`k_unrestricted`: 制限なしモデル（内生変数を外生変数と除外操作変数に
///   回帰した第一段階）の残差平方和と回帰変数の数。呼び出し側が第一段階で既に求めているので
///   受け取る（回帰を繰り返さない）。
/// - `x_restricted`: 制限モデルの設計行列（外生変数だけ。定数項を使うなら定数列を含める）。
///   列が無くてよい（回帰変数が1つも無い制限モデルは、残差が`y`そのもの）。
/// - `n_absorbed`: 固定効果で吸収したパラメータ数（定数項を除く）。残差の自由度から引く。
///   固定効果が無ければ`0`。
///
/// 呼び出し側が保証すること: `q >= 1`、`ssr_unrestricted > 0`（完全フィットは先に弾く）、
/// `y`と`x_restricted`の行数が一致する（debugビルドでは検査する）。
///
/// # Errors
/// - `n <= k_unrestricted + n_absorbed`: `CommonError::InsufficientObservations`
///   （`k`には`k_unrestricted + n_absorbed`を入れる）
/// - 制限モデルの設計行列がフルカラムランクでない: `CommonError::Internal`。制限なしモデルの
///   設計行列の列の部分集合なので、制限なしモデルがフルランクなら到達しない。
pub(crate) fn partial_f_statistic(
    y: &Mat<f64>,
    x_restricted: &Mat<f64>,
    ssr_unrestricted: f64,
    k_unrestricted: usize,
    q: usize,
    n_absorbed: usize,
) -> Result<f64, CommonError> {
    debug_assert!(
        q >= 1,
        "the partial F statistic needs at least one excluded instrument"
    );
    debug_assert!(
        ssr_unrestricted > 0.0,
        "the unrestricted model must not fit perfectly"
    );
    debug_assert_eq!(y.nrows(), x_restricted.nrows());
    let n = y.nrows();
    let k_total = k_unrestricted + n_absorbed;
    if n <= k_total {
        return Err(CommonError::InsufficientObservations { n, k: k_total });
    }
    let df_u = n - k_total;
    let restricted = least_squares(x_restricted, y).map_err(|_| {
        CommonError::internal("the restricted model of the partial F statistic is rank deficient")
    })?;
    let ssr_r = residual_sum_of_squares(&restricted.residuals);
    Ok(((ssr_r - ssr_unrestricted) / (q as f64)) / (ssr_unrestricted / (df_u as f64)))
}

/// Wu-Hausman検定（回帰ベース）の拡張回帰に使う列と名前を組み立てる:
/// `[x_exog, x_endog, 第一段階残差]`。構造式に第一段階残差を足した回帰で、足した列の係数が
/// ゼロという仮説を検定する。回帰とWald検定は呼び出し側の回帰エンジンで行う
/// （2SLSは`OlsEstimator`、固定効果つきは`FeEstimator`）。
///
/// `x_exog`は定数列を含めない（回帰側が定数項を足す）。`first_stage_residuals`は`x_endog`と
/// 同じ順序・同じ個数。
pub(crate) fn wu_hausman_columns(
    x_exog: &[Vec<f64>],
    x_exog_names: &[String],
    x_endog: Vec<Vec<f64>>,
    x_endog_names: &[String],
    first_stage_residuals: Vec<Vec<f64>>,
) -> (Vec<Vec<f64>>, Vec<String>) {
    debug_assert_eq!(x_endog.len(), first_stage_residuals.len());
    debug_assert_eq!(x_endog.len(), x_endog_names.len());
    let mut columns: Vec<Vec<f64>> = x_exog.to_vec();
    columns.extend(x_endog);
    columns.extend(first_stage_residuals);

    let mut names: Vec<String> = x_exog_names.to_vec();
    names.extend(x_endog_names.iter().cloned());
    names.extend(
        x_endog_names
            .iter()
            .map(|endog_name| format!("{endog_name}_first_stage_resid")),
    );
    (columns, names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(values: &[f64]) -> Mat<f64> {
        Mat::from_fn(values.len(), 1, |i, _| values[i])
    }

    /// 6観測、定数・外生`x`・内生`d`・除外操作変数`z1, z2`。
    fn data() -> (Mat<f64>, Mat<f64>, Mat<f64>, Mat<f64>) {
        let x = [0.3, -1.2, 0.8, 2.0, -0.5, 1.1];
        let z1 = [1.0, 0.2, -0.7, 1.5, 0.9, -1.1];
        let z2 = [0.4, -0.9, 1.3, 0.1, -0.2, 0.7];
        let d = [0.9, -0.4, 0.2, 2.6, 0.3, 0.5];
        let y = [1.2, -0.3, 0.9, 3.1, 0.4, 1.0];
        let n = 6;
        let exog = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x[i] });
        let z = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x[i],
            2 => z1[i],
            _ => z2[i],
        });
        (exog, z, column(&d), column(&y))
    }

    /// 第一段階`d ~ z`の予測値`d̂`と残差、制限なしモデルの`ssr`。
    fn first_stage(z: &Mat<f64>, d: &Mat<f64>) -> (Mat<f64>, f64) {
        let fit = least_squares(z, d).unwrap();
        let fitted = d - &fit.residuals;
        (fitted, residual_sum_of_squares(&fit.residuals))
    }

    #[test]
    fn two_sls_point_estimate_matches_the_normal_equations_of_the_projected_regressors() {
        let (exog, z, d, y) = data();
        let (d_hat, _) = first_stage(&z, &d);
        let x = Mat::from_fn(
            6,
            3,
            |i, j| if j < 2 { *exog.get(i, j) } else { *d.get(i, 0) },
        );
        let x_hat = Mat::from_fn(6, 3, |i, j| {
            if j < 2 {
                *exog.get(i, j)
            } else {
                *d_hat.get(i, 0)
            }
        });

        let est = two_sls_point_estimate(&x, &x_hat, &y).unwrap();

        // 閉形式: β = (X̂'X̂)⁻¹ X̂'y。残差は元のXで作る。
        let gram = x_hat.transpose() * &x_hat;
        let expected = gram
            .llt(Side::Lower)
            .unwrap()
            .solve(&(x_hat.transpose() * &y));
        for j in 0..3 {
            assert!(
                (*est.params.get(j, 0) - *expected.get(j, 0)).abs() < 1e-10,
                "j={j}"
            );
        }
        let e = &y - &x * &expected;
        for i in 0..6 {
            assert!(
                (*est.residuals.get(i, 0) - *e.get(i, 0)).abs() < 1e-10,
                "i={i}"
            );
        }
        let ssr: f64 = (0..6).map(|i| e.get(i, 0).powi(2)).sum();
        assert!((est.ssr - ssr).abs() < 1e-10);
        // 構造残差は第二段階の残差（X̂での残差）と異なる。
        let second_stage_resid = &y - &x_hat * &expected;
        assert!((*est.residuals.get(0, 0) - *second_stage_resid.get(0, 0)).abs() > 1e-6);
        // (X̂'X̂)⁻¹
        let identity = &gram * &est.xtx_hat_inv;
        for i in 0..3 {
            for j in 0..3 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((*identity.get(i, j) - target).abs() < 1e-9, "({i},{j})");
            }
        }
    }

    #[test]
    fn two_sls_point_estimate_rejects_a_rank_deficient_projected_design() {
        let (exog, _, _, y) = data();
        // X̂の最後の列が定数列と同一。
        let x_hat = Mat::from_fn(6, 3, |i, j| if j < 2 { *exog.get(i, j) } else { 1.0 });
        let x = x_hat.clone();
        assert!(matches!(
            two_sls_point_estimate(&x, &x_hat, &y),
            Err(RankDeficient::Columns)
        ));
    }

    #[test]
    fn sargan_test_matches_the_quadratic_form_and_scales_with_sigma2() {
        let (_, z, _, e) = data();
        let sigma2 = 0.37;

        let test = sargan_test(&z, &e, sigma2, 2).unwrap();

        let ztz = z.transpose() * &z;
        let zte = z.transpose() * &e;
        let solved = ztz.llt(Side::Lower).unwrap().solve(&zte);
        let quad: f64 = (0..4).map(|i| zte.get(i, 0) * solved.get(i, 0)).sum();
        assert!((test.statistic - quad / sigma2).abs() < 1e-12);
        let expected_p = ChiSquared::new(2.0).unwrap().sf(test.statistic);
        assert!((test.p_value - expected_p).abs() < 1e-15);

        // 分母を2倍にすると統計量は半分になる（分母は呼び出し側が決める）。
        let doubled = sargan_test(&z, &e, 2.0 * sigma2, 2).unwrap();
        assert!((doubled.statistic - 0.5 * test.statistic).abs() < 1e-12);
    }

    #[test]
    fn sargan_test_rejects_zero_degrees_of_freedom_and_a_singular_gram_matrix() {
        let (_, z, _, e) = data();
        assert!(matches!(
            sargan_test(&z, &e, 1.0, 0),
            Err(CommonError::Internal { .. })
        ));
        let singular = Mat::from_fn(6, 2, |i, _| i as f64);
        assert!(matches!(
            sargan_test(&singular, &e, 1.0, 1),
            Err(CommonError::Internal { .. })
        ));
    }

    #[test]
    fn partial_f_statistic_matches_the_nested_f_formula() {
        let (exog, z, d, _) = data();
        let (_, ssr_u) = first_stage(&z, &d);
        let ssr_r = residual_sum_of_squares(&least_squares(&exog, &d).unwrap().residuals);
        let (n, k_u, q) = (6.0, 4.0, 2.0);

        let f = partial_f_statistic(&d, &exog, ssr_u, 4, 2, 0).unwrap();

        assert!((f - ((ssr_r - ssr_u) / q) / (ssr_u / (n - k_u))).abs() < 1e-12);
    }

    #[test]
    fn partial_f_statistic_subtracts_the_absorbed_parameters_from_the_residual_degrees_of_freedom()
    {
        let (exog, z, d, _) = data();
        let (_, ssr_u) = first_stage(&z, &d);
        let base = partial_f_statistic(&d, &exog, ssr_u, 4, 2, 0).unwrap();

        let absorbed = partial_f_statistic(&d, &exog, ssr_u, 4, 2, 1).unwrap();

        // 分母の自由度が 6-4=2 から 6-4-1=1 になり、Fは (2/1) の比で小さくなる。
        assert!((absorbed - base * 1.0 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn partial_f_statistic_treats_a_restricted_model_without_regressors_as_the_raw_outcome() {
        let (_, z, d, _) = data();
        let (_, ssr_u) = first_stage(&z, &d);
        let none = Mat::<f64>::zeros(6, 0);
        let ssr_r: f64 = (0..6).map(|i| d.get(i, 0).powi(2)).sum();

        let f = partial_f_statistic(&d, &none, ssr_u, 4, 4, 0).unwrap();

        assert!((f - ((ssr_r - ssr_u) / 4.0) / (ssr_u / 2.0)).abs() < 1e-12);
    }

    #[test]
    fn partial_f_statistic_rejects_too_few_observations_for_the_absorbed_parameters() {
        let (exog, z, d, _) = data();
        let (_, ssr_u) = first_stage(&z, &d);
        // n=6, k_u=4, n_absorbed=2 → 残差の自由度が0。
        assert_eq!(
            partial_f_statistic(&d, &exog, ssr_u, 4, 2, 2),
            Err(CommonError::InsufficientObservations { n: 6, k: 6 })
        );
    }

    /// Frisch-Waugh-Lovell: 個体ダミーを外生変数と操作変数に含めた計算（`n_absorbed = 0`）と、
    /// 個体内で平均を引いたデータに`n_absorbed = ダミーの数`を渡した計算は、2SLSの係数・構造残差・
    /// Sargan統計量・部分F統計量が一致する（FEIVが頼る恒等式。古典的な統計量で成り立つ）。
    #[test]
    fn dummies_in_the_regressors_equal_the_demeaned_data_with_absorbed_parameters() {
        let (groups, per_group) = (3, 4);
        let n = groups * per_group;
        let x = [
            0.3, -1.2, 0.8, 2.0, -0.5, 1.1, 0.7, -0.4, 1.6, 0.2, -1.0, 0.9,
        ];
        let z1 = [
            1.0, 0.2, -0.7, 1.5, 0.9, -1.1, 0.3, 0.8, -0.6, 1.2, -0.3, 0.5,
        ];
        let z2 = [
            0.4, -0.9, 1.3, 0.1, -0.2, 0.7, -1.4, 0.6, 0.0, -0.8, 1.1, 0.35,
        ];
        let u = [
            0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.2, 0.05, -0.45, 0.3, 0.15, -0.1,
        ];
        let d: Vec<f64> = (0..n)
            .map(|i| 0.8 * z1[i] - 0.5 * z2[i] + 0.3 * x[i] + 0.7 * u[i])
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.5 * d[i] - 0.4 * x[i] + u[i] + (i / per_group) as f64)
            .collect();
        let demean = |v: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let g = i / per_group;
                    let mean: f64 = v[g * per_group..(g + 1) * per_group].iter().sum::<f64>()
                        / per_group as f64;
                    v[i] - mean
                })
                .collect()
        };
        let mat = |cols: &[&[f64]]| Mat::from_fn(n, cols.len(), |i, j| cols[j][i]);
        let dummy =
            |g: usize| -> Vec<f64> { (0..n).map(|i| f64::from(i / per_group == g)).collect() };
        let dummies: Vec<Vec<f64>> = (0..groups).map(dummy).collect();
        let (d0, d1, d2) = (&dummies[0][..], &dummies[1][..], &dummies[2][..]);

        // (A) ダミー入り。外生変数 [D, x]、全操作変数 Z = [D, x, z1, z2]、内生変数 d。
        let first_a = least_squares(&mat(&[d0, d1, d2, &x, &z1, &z2]), &mat(&[&d])).unwrap();
        let d_hat_a = &mat(&[&d]) - &first_a.residuals;
        let x_a = mat(&[d0, d1, d2, &x, &d]);
        let x_hat_a = Mat::from_fn(n, 5, |i, j| {
            if j < 4 {
                *x_a.get(i, j)
            } else {
                *d_hat_a.get(i, 0)
            }
        });
        let est_a = two_sls_point_estimate(&x_a, &x_hat_a, &mat(&[&y])).unwrap();
        let z_a = mat(&[d0, d1, d2, &x, &z1, &z2]);
        let sargan_a = sargan_test(&z_a, &est_a.residuals, est_a.ssr / n as f64, 1).unwrap();
        let ssr_u_a = residual_sum_of_squares(&first_a.residuals);
        let f_a =
            partial_f_statistic(&mat(&[&d]), &mat(&[d0, d1, d2, &x]), ssr_u_a, 6, 2, 0).unwrap();

        // (B) 個体内で平均を引き、定数もダミーも無し。吸収したパラメータは3個。
        let (xt, z1t, z2t, dt, yt) = (demean(&x), demean(&z1), demean(&z2), demean(&d), demean(&y));
        let z_b = mat(&[&xt, &z1t, &z2t]);
        let first_b = least_squares(&z_b, &mat(&[&dt])).unwrap();
        let d_hat_b = &mat(&[&dt]) - &first_b.residuals;
        let x_b = mat(&[&xt, &dt]);
        let x_hat_b = Mat::from_fn(n, 2, |i, j| if j == 0 { xt[i] } else { *d_hat_b.get(i, 0) });
        let est_b = two_sls_point_estimate(&x_b, &x_hat_b, &mat(&[&yt])).unwrap();
        let sargan_b = sargan_test(&z_b, &est_b.residuals, est_b.ssr / n as f64, 1).unwrap();
        let ssr_u_b = residual_sum_of_squares(&first_b.residuals);
        let f_b = partial_f_statistic(&mat(&[&dt]), &mat(&[&xt]), ssr_u_b, 3, 2, groups).unwrap();

        // 0近傍だと相対許容誤差の比較が意味を持たないので、統計量が十分大きいことを確かめる。
        assert!(
            sargan_a.statistic > 1e-3 && f_a > 1e-3,
            "{} {f_a}",
            sargan_a.statistic
        );
        // 傾き（x, d）は一致し、構造残差も一致する（ダミーの係数は残差を個体内で中心化する）。
        assert!((*est_a.params.get(3, 0) - *est_b.params.get(0, 0)).abs() < 1e-9);
        assert!((*est_a.params.get(4, 0) - *est_b.params.get(1, 0)).abs() < 1e-9);
        for i in 0..n {
            assert!(
                (*est_a.residuals.get(i, 0) - *est_b.residuals.get(i, 0)).abs() < 1e-9,
                "i={i}"
            );
        }
        assert!((sargan_a.statistic - sargan_b.statistic).abs() < 1e-8 * sargan_a.statistic.abs());
        assert!((sargan_a.p_value - sargan_b.p_value).abs() < 1e-9);
        assert!((f_a - f_b).abs() < 1e-8 * f_a.abs());
    }

    #[test]
    fn wu_hausman_columns_orders_exogenous_endogenous_and_first_stage_residual_columns() {
        let x_exog = vec![vec![1.0, 2.0]];
        let x_endog = vec![vec![3.0, 4.0], vec![5.0, 6.0]];
        let resid = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

        let (columns, names) = wu_hausman_columns(
            &x_exog,
            &["x1".to_string()],
            x_endog.clone(),
            &["d1".to_string(), "d2".to_string()],
            resid.clone(),
        );

        assert_eq!(
            columns,
            [
                x_exog[0].clone(),
                x_endog[0].clone(),
                x_endog[1].clone(),
                resid[0].clone(),
                resid[1].clone()
            ]
        );
        assert_eq!(
            names,
            [
                "x1",
                "d1",
                "d2",
                "d1_first_stage_resid",
                "d2_first_stage_resid"
            ]
        );
    }
}
