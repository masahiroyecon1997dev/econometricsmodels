//! 係数分散共分散行列（classical / HC0-3 / HAC / cluster）の計算部品。
//!
//! サンドイッチ型の分散`bread · meat · bread`のうち、`meat`を観測ごとのスコア`s_i`
//! （最小二乗なら`s_i = e_i x_i`、MLEなら尤度のスコア）から作る関数と、`bread`
//! （最小二乗なら`(X'X)⁻¹`、MLEなら観測情報行列の逆行列）で挟む[`sandwich`]に分けてある。
//! HCは`S'S`の行列積のためスコア行列（n×k）を作るが、HACとクラスターはスコアを返す
//! クロージャ`score(i, a)`（`i`は元データの行、`a`は係数の列）で受け取り、n×kの行列を
//! 追加で確保しない。

use faer::linalg::matmul::matmul;
use faer::prelude::Solve;
use faer::{Accum, Mat, Par, Side};

use super::cluster::group_indices;
use crate::linear_algebra::RankDeficient;

/// `(X'X)⁻¹`を求める。classical・HC0-3・HAC・クラスターのいずれの標準誤差でも必要になる。
///
/// `X'X`は対称正定値であることが`checked_col_piv_qr`（Xの特異性検出）で既に保証されている
/// ため、Cholesky分解（`Llt`）で逆行列を求める。理論上ここで`LltError`は発生しないはずだが、
/// 浮動小数点演算の丸めにより境界的なケースで失敗しうるため、`RankDeficient`として扱う
/// （各系統が自系統のエラーへ`map_err`で変換する）。
pub(crate) fn xtx_inverse(x: &Mat<f64>) -> Result<Mat<f64>, RankDeficient> {
    let k = x.ncols();
    let xtx = x.transpose() * x;
    let llt = xtx.llt(Side::Lower).map_err(|_| RankDeficient)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// サンドイッチ `bread · meat · bread`。
pub(crate) fn sandwich(bread: &Mat<f64>, meat: &Mat<f64>) -> Mat<f64> {
    bread * meat * bread
}

/// classical（等分散前提）の係数分散共分散行列: `σ̂²(X'X)⁻¹`（k×k）。
pub(crate) fn classical_cov_params(sigma2: f64, xtx_inv: &Mat<f64>, k: usize) -> Mat<f64> {
    Mat::from_fn(k, k, |i, j| sigma2 * (*xtx_inv.get(i, j)))
}

/// HC0〜HC3の種類。`CovType`はclassicalも含む上位概念のため、HC計算専用の分岐であることを
/// 型で明確にする（`CovType::Classical`が紛れ込まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HcVariant {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// 行ごとのレバレッジ `h_ii = x_i'(X'X)⁻¹x_i`。
///
/// `(X (X'X)⁻¹ X')_ii`をn×nの行列を作らずに行ごとの内積で求める。
fn leverages(x: &Mat<f64>, xtx_inv: &Mat<f64>) -> Vec<f64> {
    let xh = x * xtx_inv; // (n, k)
    (0..x.nrows())
        .map(|i| {
            (0..x.ncols())
                .map(|j| (*xh.get(i, j)) * (*x.get(i, j)))
                .sum()
        })
        .collect()
}

/// HCの種類ごとの行スケール `scale_i`（`Ψ̂ = Σ_i scale_i² x_i x_i'`となるように、残差
/// `ε̂_i`に重みの平方根をかけたもの。符号は二乗で相殺されるため`ε̂_i`のままでよい）。
///
/// - HC0: `ε̂_i`
/// - HC1: `ε̂_i sqrt(n/(n-k))`
/// - HC2: `ε̂_i / sqrt(1-h_ii)`
/// - HC3: `ε̂_i / (1-h_ii)`
///
/// レバレッジはHC2/HC3でのみ必要なため、それ以外では計算しない。
fn hc_row_scales(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    variant: HcVariant,
) -> Vec<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let resid = |i: usize| *residuals.get(i, 0);
    match variant {
        HcVariant::Hc0 => (0..n).map(resid).collect(),
        HcVariant::Hc1 => {
            let hc1_correction = ((n as f64) / ((n - k) as f64)).sqrt();
            (0..n).map(|i| resid(i) * hc1_correction).collect()
        }
        HcVariant::Hc2 => {
            let h = leverages(x, xtx_inv);
            (0..n).map(|i| resid(i) / (1.0 - h[i]).sqrt()).collect()
        }
        HcVariant::Hc3 => {
            let h = leverages(x, xtx_inv);
            (0..n).map(|i| resid(i) / (1.0 - h[i])).collect()
        }
    }
}

/// HC0〜HC3ロバストな係数分散共分散行列: `(X'X)⁻¹Ψ̂(X'X)⁻¹`（k×k）。
///
/// `Ψ̂`は各行を[`hc_row_scales`]でスケールしたスコア行列`S`（`S[i,j] = scale_i x_ij`）の
/// `S'S`。`x_i x_i'`の外積を行ごとに手動で積み上げるより、既存の行列積を再利用できて簡潔なため。
pub(crate) fn hc_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    variant: HcVariant,
) -> Mat<f64> {
    let row_scale = hc_row_scales(x, residuals, xtx_inv, variant);
    let scores = Mat::from_fn(x.nrows(), x.ncols(), |i, j| row_scale[i] * (*x.get(i, j)));
    let psi_hat = scores.transpose() * &scores;
    sandwich(xtx_inv, &psi_hat)
}

/// `time_order`から、時系列の昇順に並べたときの行インデックス列を求める。
///
/// 比較は`total_cmp`（全順序）を使うためNaNがあってもパニックしない（NaNは最後に並ぶ）。
/// `time_order`の値はNaN/無限大を含まないことが`engine_pybind::column_extraction`側で
/// 既に保証されている前提（本関数は`engine`の責務境界の内側であり、クリーンな値しか
/// 受け取らない）。同様に、値が互いに異なる（`engine_pybind`が昇順の位置＝順位に変換済みで、同値は
/// `ValidationError`として弾かれている）ことも前提にする。この関数自身は同値を検出せず、
/// 同値があれば安定ソートにより行順で並べるだけ。
pub(crate) fn time_ordering(time_order: &[f64], n: usize) -> Vec<usize> {
    debug_assert_eq!(time_order.len(), n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| time_order[a].total_cmp(&time_order[b]));
    order
}

/// Newey-West HAC（Bartlettカーネル）の`meat`: `Ŝ = Ŝ₀ + Σ_{l=1}^{L} w_l (Ŝ_l + Ŝ_l')`（k×k）。
///
/// Bartlett重み`w_l = 1 - l/(L+1)`、`Ŝ_l = Σ_{t=l+1}^{n} s_t s_{t-l}'`。`order`で指定された
/// 時系列順に並べ替えたスコアを使ってラグ付き自己共分散を計算する。`score(i, a)`は
/// 元データの`i`行目の`a`列目のスコア。
///
/// 時系列順に並べたスコア行列`Xe`（`Xe[t,a] = score(order[t], a)`）を使うと、
/// `Ŝ₀ = Xe'Xe`、`Ŝ_l = Xe[l:,:]'Xe[:n-l,:]`という行列積に落とし込める（`Ŝ_l'`は転置を
/// 取るだけで再計算不要）。手書きの三重ループ（ラグ×観測×`k²`）よりfaerの行列積を使う方が
/// 大幅に高速（計測方法論は`docs/performance/ols.md`、実測値は`docs/guide/performance-results.md`参照）。
///
/// **`Par::Seq`を明示指定する理由**: `Ŝ_l`の行列積はラグの数だけ繰り返し呼ぶことになるが、
/// 1回あたりの行列積は`k×k`という小さい出力サイズのため、faer既定の並列実行（グローバル
/// スレッドプールへのディスパッチ）のオーバーヘッドが計算本体を上回り、**三重ループより
/// 遅くなる**ことを実測済み（n=10,000, k=2で0.13倍＝約6倍の悪化）。この関数
/// 内だけ`Par::Seq`にスコープを切ることで、他のcov_type計算・将来手法のグローバル並列化
/// 設定に影響を与えずにこの罠を回避している。
pub(crate) fn hac_meat(
    n: usize,
    k: usize,
    lags: usize,
    order: &[usize],
    score: impl Fn(usize, usize) -> f64,
) -> Mat<f64> {
    let xe = Mat::<f64>::from_fn(n, k, |t, a| score(order[t], a));

    // l=0項: Ŝ₀ = Xe'Xe（HC0のΨ̂と同形）
    let mut s_hat = Mat::<f64>::zeros(k, k);
    matmul(
        s_hat.as_mut(),
        Accum::Replace,
        xe.transpose(),
        xe.as_ref(),
        1.0,
        Par::Seq,
    );

    // l=1..=lags項: w_l * (Ŝ_l + Ŝ_l')
    let mut s_l = Mat::<f64>::zeros(k, k);
    for l in 1..=lags {
        let weight = 1.0 - (l as f64) / ((lags + 1) as f64);
        let xe_top = xe.as_ref().subrows(l, n - l);
        let xe_bot = xe.as_ref().subrows(0, n - l);
        matmul(
            s_l.as_mut(),
            Accum::Replace,
            xe_top.transpose(),
            xe_bot,
            1.0,
            Par::Seq,
        );

        for a in 0..k {
            for b in 0..k {
                // (Ŝ_l + Ŝ_l')[a,b] = Ŝ_l[a,b] + Ŝ_l[b,a]
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    s_hat
}

/// Newey-West HACの係数分散共分散行列: `(X'X)⁻¹Ŝ(X'X)⁻¹`（k×k）。スコアは`s_i = ε̂_i x_i`
/// （`docs/spec/ols-spec.md`「標準誤差」のHAC）。
pub(crate) fn hac_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    let meat = hac_meat(x.nrows(), x.ncols(), lags, order, |i, a| {
        (*residuals.get(i, 0)) * (*x.get(i, a))
    });
    sandwich(xtx_inv, &meat)
}

/// クラスターロバストの`meat`: `Ŝ = Σ_{g=1}^{G} S_g S_g'`（k×k）と、クラスター数`G`。
///
/// `S_g = Σ_{i∈g} s_i`（クラスター内のスコアの合計。クラスター内の観測を先に合計してから
/// 外積を取ることで、クラスター内の相関を許容する）。`score(i, a)`は`i`行目の`a`列目の
/// スコア。グループの反復順序は[`group_indices`]（辞書順）で決定的。
pub(crate) fn cluster_meat(
    groups: &[String],
    k: usize,
    score: impl Fn(usize, usize) -> f64,
) -> (Mat<f64>, usize) {
    let indices = group_indices(groups);
    let n_groups = indices.len();

    let mut s_hat = Mat::<f64>::zeros(k, k);
    for rows in indices.values() {
        let mut s_g = vec![0.0_f64; k];
        for &i in rows {
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += score(i, a);
            }
        }
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }
    (s_hat, n_groups)
}

/// クラスターロバストの小標本補正 `G/(G-1) * (n-1)/(n-k)`（Stata方式。常に適用する。
/// `docs/spec/ols-spec.md`「標準誤差」のクラスター参照）。
pub(crate) fn cluster_correction(n_groups: usize, n: usize, k: usize) -> f64 {
    (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64))
}

/// クラスターロバストな係数分散共分散行列: `(X'X)⁻¹Ŝ(X'X)⁻¹ * correction`（k×k）。
/// スコアは`s_i = ε̂_i x_i`。
///
/// `groups`が2種類以上の値を持つこと（`G >= 2`）は呼び出し側（`validate_cluster_groups`）で
/// 検証済みの前提とする。
pub(crate) fn cluster_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    groups: &[String],
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let (s_hat, n_groups) = cluster_meat(groups, k, |i, a| (*residuals.get(i, 0)) * (*x.get(i, a)));

    let correction = cluster_correction(n_groups, n, k);
    let cov_uncorrected = sandwich(xtx_inv, &s_hat);
    Mat::from_fn(k, k, |i, j| correction * (*cov_uncorrected.get(i, j)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    fn assert_matrix_close(actual: &Mat<f64>, expected: &Mat<f64>, tol: f64) {
        assert_eq!(actual.nrows(), expected.nrows());
        assert_eq!(actual.ncols(), expected.ncols());
        for i in 0..actual.nrows() {
            for j in 0..actual.ncols() {
                let (a, e) = (*actual.get(i, j), *expected.get(i, j));
                assert!(
                    (a - e).abs() <= tol * e.abs().max(1.0),
                    "[{i},{j}]: {a} vs {e}"
                );
            }
        }
    }

    /// 切片なし・1説明変数の最小例。`x = [1, 2, 3]`、残差 `e = [1, -1, 2]`。
    fn single_regressor_case() -> (Mat<f64>, Mat<f64>, Mat<f64>) {
        let x = matrix(&[&[1.0], &[2.0], &[3.0]]);
        let residuals = matrix(&[&[1.0], &[-1.0], &[2.0]]);
        let xtx_inv = xtx_inverse(&x).unwrap();
        (x, residuals, xtx_inv)
    }

    #[test]
    fn xtx_inverse_returns_the_inverse_gram_matrix() {
        let x = matrix(&[&[1.0, 0.0], &[1.0, 1.0], &[1.0, 3.0], &[1.0, 4.0]]);
        let inv = xtx_inverse(&x).unwrap();

        let identity = &(x.transpose() * &x) * &inv;
        assert_matrix_close(&identity, &Mat::<f64>::identity(2, 2), 1e-12);
    }

    #[test]
    fn xtx_inverse_rejects_a_zero_matrix() {
        assert_eq!(
            xtx_inverse(&Mat::<f64>::zeros(3, 2)).unwrap_err(),
            RankDeficient
        );
    }

    #[test]
    fn time_ordering_sorts_ascending_and_keeps_nan_last_without_panicking() {
        assert_eq!(time_ordering(&[3.0, 1.0, 2.0], 3), vec![1, 2, 0]);
        // NaNは`engine_pybind`で弾かれるが、`engine`単体でもパニックしない（最後に並ぶ）。
        assert_eq!(time_ordering(&[2.0, f64::NAN, 1.0], 3), vec![2, 0, 1]);
    }

    #[test]
    fn hc_cov_params_matches_the_sandwich_with_variant_specific_weights() {
        let (x, residuals, xtx_inv) = single_regressor_case();
        let (n, k) = (3.0_f64, 1.0_f64);
        let inv = *xtx_inv.get(0, 0);

        // Ψ̂ = Σ_i w_i e_i² x_i²。w_i は HC0: 1、HC1: n/(n-k)、HC2: 1/(1-h)、HC3: 1/(1-h)²。
        let psi = |weight: &dyn Fn(f64) -> f64| -> f64 {
            (0..3)
                .map(|i| {
                    let (xi, ei) = (*x.get(i, 0), *residuals.get(i, 0));
                    let h = xi * xi * inv;
                    weight(h) * ei * ei * xi * xi
                })
                .sum()
        };
        let cases: [(HcVariant, f64); 4] = [
            (HcVariant::Hc0, psi(&|_| 1.0)),
            (HcVariant::Hc1, psi(&|_| n / (n - k))),
            (HcVariant::Hc2, psi(&|h| 1.0 / (1.0 - h))),
            (HcVariant::Hc3, psi(&|h| 1.0 / ((1.0 - h) * (1.0 - h)))),
        ];
        for (variant, psi_hat) in cases {
            let cov = hc_cov_params(&x, &residuals, &xtx_inv, variant);
            let expected = inv * psi_hat * inv;
            assert!(
                (*cov.get(0, 0) - expected).abs() < 1e-12 * expected.abs(),
                "{variant:?}: {} vs {expected}",
                *cov.get(0, 0)
            );
        }
    }

    #[test]
    fn hac_with_zero_lags_equals_the_hc0_covariance() {
        let (x, residuals, xtx_inv) = single_regressor_case();
        let order = time_ordering(&[0.0, 1.0, 2.0], 3);

        let hac = hac_cov_params(&x, &residuals, &xtx_inv, 0, &order);
        let hc0 = hc_cov_params(&x, &residuals, &xtx_inv, HcVariant::Hc0);

        assert_matrix_close(&hac, &hc0, 1e-14);
    }

    #[test]
    fn hac_meat_adds_bartlett_weighted_autocovariances() {
        // 1変数・スコア s = [1, 2, 3]（時系列順）、lags=1: Ŝ₀ = 14、Ŝ₁ = s₂s₁ + s₃s₂ = 2 + 6 = 8、
        // w₁ = 1 - 1/2 = 0.5 → Ŝ = 14 + 0.5 × (8 + 8) = 22。
        let s = [1.0, 2.0, 3.0];
        let meat = hac_meat(3, 1, 1, &[0, 1, 2], |i, _| s[i]);
        assert!((*meat.get(0, 0) - 22.0).abs() < 1e-12);

        // 並べ替え順を逆にしても、自己共分散は対称なため同じ値になる。
        let reversed = hac_meat(3, 1, 1, &[2, 1, 0], |i, _| s[i]);
        assert!((*reversed.get(0, 0) - 22.0).abs() < 1e-12);
    }

    #[test]
    fn cluster_meat_sums_scores_within_clusters_before_taking_outer_products() {
        // クラスター a = {行0, 行2}、b = {行1}。スコア s = [1, 2, 3] → S_a = 4、S_b = 2。
        let groups = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        let s = [1.0, 2.0, 3.0];

        let (meat, n_groups) = cluster_meat(&groups, 1, |i, _| s[i]);

        assert_eq!(n_groups, 2);
        assert!((*meat.get(0, 0) - (16.0 + 4.0)).abs() < 1e-12);
    }

    #[test]
    fn cluster_meat_with_one_observation_per_cluster_equals_the_hc0_outer_product() {
        let (x, residuals, _) = single_regressor_case();
        let groups: Vec<String> = (0..3).map(|i| format!("g{i}")).collect();

        let (meat, n_groups) =
            cluster_meat(&groups, 1, |i, a| (*residuals.get(i, 0)) * (*x.get(i, a)));

        let outer: f64 = (0..3)
            .map(|i| ((*residuals.get(i, 0)) * (*x.get(i, 0))).powi(2))
            .sum();
        assert_eq!(n_groups, 3);
        assert!((*meat.get(0, 0) - outer).abs() < 1e-12);
    }

    #[test]
    fn cluster_correction_is_the_stata_small_sample_factor() {
        // G/(G-1) * (n-1)/(n-k) = (4/3) * (9/7)
        let expected = (4.0 / 3.0) * (9.0 / 7.0);
        assert!((cluster_correction(4, 10, 3) - expected).abs() < 1e-15);
    }

    #[test]
    fn cluster_cov_params_applies_the_correction_to_the_sandwich() {
        let (x, residuals, xtx_inv) = single_regressor_case();
        let groups = vec!["a".to_string(), "b".to_string(), "a".to_string()];

        let cov = cluster_cov_params(&x, &residuals, &xtx_inv, &groups);

        // S_a = e₀x₀ + e₂x₂ = 1 + 6 = 7、S_b = e₁x₁ = -2 → Ŝ = 49 + 4 = 53。
        let inv = *xtx_inv.get(0, 0);
        let expected = cluster_correction(2, 3, 1) * inv * 53.0 * inv;
        assert!((*cov.get(0, 0) - expected).abs() < 1e-12 * expected);
    }
}
