//! 係数分散共分散行列（classical / HC0-3 / HAC / cluster）の計算。
//!
//! 引数は設計行列・残差・`(X'X)⁻¹`だけで、推定量の型には依存しない。

use faer::linalg::matmul::matmul;
use faer::prelude::Solve;
use faer::{Accum, Mat, Par, Side};

use crate::linear::common::LeastSquaresError;

/// `(X'X)⁻¹`を求める。classical・HC0-3いずれの標準誤差計算でも共通して必要になる。
///
/// `X'X`は対称正定値であることが`ensure_full_rank`（Xの特異性検出）で既に保証されている
/// ため、Cholesky分解（`Llt`）で逆行列を求める。理論上ここで`LltError`は発生しないはずだが、
/// 浮動小数点演算の丸めにより境界的なケースで失敗しうるため、`SingularMatrix`として扱う。
pub(crate) fn xtx_inverse(x: &Mat<f64>, k: usize) -> Result<Mat<f64>, LeastSquaresError> {
    let xtx = x.transpose() * x;
    let llt = xtx
        .llt(Side::Lower)
        .map_err(|_| LeastSquaresError::SingularMatrix)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// classical（等分散前提）の係数分散共分散行列: `σ̂²(X'X)⁻¹`（k×k）。
pub(crate) fn classical_cov_params(sigma2: f64, xtx_inv: &Mat<f64>, k: usize) -> Mat<f64> {
    Mat::from_fn(k, k, |i, j| sigma2 * (*xtx_inv.get(i, j)))
}

/// HC0〜HC3ロバストな係数分散共分散行列: `(X'X)⁻¹Ψ̂(X'X)⁻¹`（k×k）。
///
/// `Ψ̂ = Σ_i w_i ε̂_i² x_i x_i'`（`w_i`はHCの種類ごとの重み）を、各行を
/// `scale_i = sqrt(w_i) * ε̂_i`でスケーリングした行列`Xw`を使って`Ψ̂ = Xw'Xw`として計算する
/// （符号は二乗で相殺されるため、`scale_i`の符号自体は`ε̂_i`のままでよい）。
/// `x_i x_i'`の外積を行ごとに手動で積み上げるより、既存の行列積を再利用できて簡潔なため。
///
/// - HC0: `w_i = 1`
/// - HC1: `w_i = n/(n-k)`（定数）
/// - HC2: `w_i = 1/(1-h_ii)`（`h_ii`はレバレッジ）
/// - HC3: `w_i = 1/(1-h_ii)²`
///
/// レバレッジ`h_ii = x_i'(X'X)⁻¹x_i`はHC2/HC3でのみ必要なため、それ以外では計算しない。
pub(crate) fn hc_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    variant: HcVariant,
) -> Mat<f64> {
    let leverage: Option<Vec<f64>> = match variant {
        HcVariant::Hc2 | HcVariant::Hc3 => {
            // h_ii = (X (X'X)⁻¹ X')_ii を、n×n の行列を作らずに行ごとの内積で求める。
            let xh = x * xtx_inv; // (n, k)
            Some(
                (0..n)
                    .map(|i| (0..k).map(|j| (*xh.get(i, j)) * (*x.get(i, j))).sum())
                    .collect(),
            )
        }
        HcVariant::Hc0 | HcVariant::Hc1 => None,
    };

    let hc1_correction = ((n as f64) / ((n - k) as f64)).sqrt();

    let x_scaled = Mat::from_fn(n, k, |i, j| {
        let resid = *residuals.get(i, 0);
        let scale = match variant {
            HcVariant::Hc0 => resid,
            HcVariant::Hc1 => resid * hc1_correction,
            HcVariant::Hc2 => {
                let h = leverage.as_ref().expect("Hc2はleverage計算済み")[i];
                resid / (1.0 - h).sqrt()
            }
            HcVariant::Hc3 => {
                let h = leverage.as_ref().expect("Hc3はleverage計算済み")[i];
                resid / (1.0 - h)
            }
        };
        scale * (*x.get(i, j))
    });

    let psi_hat = x_scaled.transpose() * &x_scaled;
    xtx_inv * &psi_hat * xtx_inv
}

/// `hc_cov_params`の内部でのみ使う、HCの種類。`CovType`はclassicalも含む上位概念のため、
/// HC計算専用の分岐であることを型で明確にする（`CovType::Classical`が紛れ込まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HcVariant {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// `CovType::Hac`の`time_order`から、時系列の昇順に並べたときの行インデックス列を求める。
///
/// 比較は`total_cmp`（全順序）を使うためNaNがあってもパニックしない（NaNは最後に並ぶ）。
/// `time_order`の値はNaN/無限大を含まないことが`engine_pybind::column_extraction`側で
/// 既に保証されている前提（本関数は`engine`の責務境界の内側であり、クリーンな値しか
/// 受け取らない。モジュール冒頭のdocコメント参照）。同様に、値が互いに異なる（`engine_pybind`が昇順の位置＝順位に変換済みで、同値は
/// `ValidationError`として弾かれている）ことも前提にする。この関数自身は同値を検出せず、
/// 同値があれば安定ソートにより行順で並べるだけ。
pub(crate) fn time_ordering(time_order: &[f64], n: usize) -> Vec<usize> {
    debug_assert_eq!(time_order.len(), n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| time_order[a].total_cmp(&time_order[b]));
    order
}

/// Newey-West HACの係数分散共分散行列: `(X'X)⁻¹Ŝ(X'X)⁻¹`（k×k）。
///
/// `Ŝ = Ŝ₀ + Σ_{l=1}^{L} w_l (Ŝ_l + Ŝ_l')`（Bartlett重み `w_l = 1 - l/(L+1)`）、
/// `Ŝ_l = Σ_{t=l+1}^{n} ε̂_t ε̂_{t-l} x_t x_{t-l}'`（`docs/spec/ols-spec.md`
/// 「標準誤差」のHAC）。`order`で指定された時系列順に並べ替えた残差・行を使ってラグ付き自己共分散を計算する。
///
/// 残差でスケールした行列`Xe`（`Xe[t,a] = ε̂_t・x_t[a]`、`order`の時系列順）を使うと、
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
pub(crate) fn hac_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    let xe = Mat::<f64>::from_fn(n, k, |t, a| {
        let i = order[t];
        (*residuals.get(i, 0)) * (*x.get(i, a))
    });

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

    xtx_inv * &s_hat * xtx_inv
}

/// クラスターロバストな係数分散共分散行列: `(X'X)⁻¹Ŝ(X'X)⁻¹ * correction`（k×k）。
///
/// `Ŝ = Σ_{g=1}^{G} S_g S_g'`、`S_g = Σ_{i∈g} ε̂_i x_i`（クラスター内の`x_i ε̂_i`の合計。
/// クラスター内の観測を先に合計してから外積を取ることで、クラスター内の相関を許容する）。
/// `correction = G/(G-1) * (n-1)/(n-k)`（Stata方式の小標本補正。常に適用する。
/// `docs/spec/ols-spec.md`「標準誤差」のクラスター参照）。
///
/// `groups`が2種類以上の値を持つこと（`G >= 2`）は呼び出し側（`validate_cluster_groups`）で
/// 検証済みの前提とする。
pub(crate) fn cluster_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &[String],
) -> Mat<f64> {
    // `HashMap`は反復順序がプロセスごとのハッシュシードに依存し非決定的なため、`Σ_g S_g S_g'`
    // の加算順序（延いては浮動小数点丸め誤差）が実行のたびに変わりうる。`BTreeMap`（クラスター
    // 名の辞書順）を使い、同じ入力に対して常に同じ合計順序・同じ結果になるようにする。
    let mut group_indices: std::collections::BTreeMap<&str, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        group_indices.entry(g.as_str()).or_default().push(i);
    }
    let n_groups = group_indices.len();

    let mut s_hat = Mat::<f64>::zeros(k, k);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; k];
        for &i in indices {
            let e = *residuals.get(i, 0);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*x.get(i, a));
            }
        }
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }

    let correction =
        (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| correction * (*cov_uncorrected.get(i, j)))
}
