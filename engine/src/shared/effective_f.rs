//! Montiel Olea and Pflueger(2013)の有効F統計量とPatnaik臨界値。行列だけを受け取る。
//!
//! 内生変数が1個のときの、ロバスト(不均一分散・クラスター・自己相関)な弱操作変数の診断。有効F統計量と、
//! 単純版TSLSの臨界値(`x = 1/τ`)、被説明変数を渡したときは一般化版(非単純版)TSLSの臨界値(`x = B_TSLS/τ`)。
//! `(Y, X_exog, Z)`と吸収したパラメータ数`n_absorbed`を受け取り、重みつきの回帰は呼び出し側が
//! `√w`を掛けた行列(定数列を含む)を渡す。共分散は[`ScoreCovariance`]で選び、meatと小標本の係数は
//! `shared::score_covariance`が定める(Kleibergen-Paapと同じ規約)。
//!
//! 参考文献:
//! - Montiel Olea, J. L. and Pflueger, C. (2013). A Robust Test for Weak Instruments. Journal of
//!   Business & Economic Statistics, 31(3), 358-369.

use faer::{Mat, Side};

use super::linear_algebra::inverse_cholesky_factor;
use super::noncentral_chi_squared::noncentral_chi_squared_quantile;
use super::score_covariance::{ScoreCovariance, covariance_multiplier, hc_row_scales, score_meat};
use super::weak_instruments::{EXACT_FIT_TOLERANCE, orthogonalize_unit, rms_orthogonalized};

/// 臨界値を出す相対バイアスの閾値`τ`(Nagarバイアスの、ベンチマークに対する割合の上限)。
pub(crate) const BIAS_THRESHOLDS: [f64; 4] = [0.05, 0.10, 0.20, 0.30];

/// `B_TSLS`の最大化で、`β`の格子の点数(`β = β₀·sinh u`、`u ∈ [−U, U]`の等間隔)。`sinh`は0の近くでは
/// 線形、大きい`|β|`では対数間隔になり、どちらのスケールにある山も見落とさない。
const BIAS_GRID_POINTS: usize = 3_000;

/// 格子上の局所最大のうち、黄金分割で詰める個数(値の大きい順)。
const MAX_REFINED_PEAKS: usize = 16;

/// 格子の端`U`(`|β|`の上限は`β₀·sinh U ≈ 5e12·β₀`。これより先は極限の値と区別できない)。
const BIAS_GRID_HALF_WIDTH: f64 = 30.0;

/// 格子上で値が左隣よりこれだけ大きい点だけを局所最大とみなす。`|β|`が大きい平坦な区間では、丸め誤差の
/// 細かい波が並ぶので、それらを局所最大に数えない。
const PEAK_RIPPLE_TOLERANCE: f64 = 1e-12;

/// 格子の評価点のうち、評価に失敗した点がこの割合を超えたら`B_TSLS`は未定義とする(少数の失敗は飛ばす)。
const MAX_FAILED_GRID_FRACTION: f64 = 0.01;

/// 黄金分割探索を止める区間幅(`u`の)。`u`の絶対値は最大30で、区間幅は約30 ulp。最大値の誤差は区間幅の
/// 2乗のオーダー。
const GOLDEN_SECTION_WIDTH: f64 = 1e-13;

/// `min_β tr(S1(β)) = tr(W1) − tr(W12)²/tr(W2)`が`tr(W1)`のこの割合以下なら、被説明変数の誘導形の誤差が
/// 内生変数の誘導形の誤差とほぼ比例しており(`v1 = βv2`)、ベンチマークが0になる`β`があるので未定義とする。
const DEGENERATE_BENCHMARK_TOLERANCE: f64 = 1e-12;

/// `Ŵ2`の最小固有値が、最大固有値のこの割合より負なら、丸め誤差ではなく不定値とみなす。
const NEGATIVE_EIGENVALUE_TOLERANCE: f64 = 1e-10;

/// Patnaik臨界値と、その計算に使った量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatnaikCriticalValue {
    /// 相対バイアスの閾値`τ`。
    pub tau: f64,
    /// 非心度の入力`x`(単純版は`1/τ`、一般化版は`B_TSLS/τ`)。
    pub x: f64,
    /// 有効自由度`K_eff`(小数)。
    pub effective_df: f64,
    /// 臨界値(`χ²_{K_eff}(x·K_eff)`の上側`α`点を`K_eff`で割ったもの)。有効Fがこれを超えれば、
    /// 水準`α`で「Nagarバイアスが閾値`τ`を超える」を棄却する。
    pub critical_value: f64,
}

/// [`effective_f`]の結果。
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveF {
    /// 有効F統計量。
    pub statistic: f64,
    /// 除外操作変数の数`K`。
    pub n_instruments: usize,
    /// 臨界値の水準`α`。
    pub alpha: f64,
    /// 単純版TSLSの臨界値(`x = 1/τ`)。`τ`は[`BIAS_THRESHOLDS`]の順。
    pub simplified_tsls: [PatnaikCriticalValue; 4],
    /// 一般化版(非単純版)TSLSの臨界値(`x = B_TSLS/τ`)。被説明変数を渡し、`B_TSLS`が定義できたときだけ。
    /// 単純版より小さいか等しい(`B_TSLS ≤ 1`)。
    pub tsls: Option<[PatnaikCriticalValue; 4]>,
    /// Nagarバイアスの最悪ケースに対する上限`B_TSLS`(論文 Eq. (21))。`0 ≤ B_TSLS ≤ 1`。
    pub bias_bound: Option<f64>,
}

/// Montiel Olea-Pfluegerの有効F統計量と、単純版・一般化版TSLSのPatnaik臨界値を計算する。
///
/// 定義: 外生変数で`Y`と`Z`を直交化し、`Z̃'Z̃/n = I`となるように直交正規化する(統計量はこの変換に
/// 依らない)。`π̂ = Z̃'Y/n`、`V̂2`を`π̂`の共分散の推定値(`covariance`による。`Ŵ2 = n·V̂2`)として
///
/// ```text
/// F_eff = π̂'π̂ / tr(V̂2)                                           [論文 Eq. (3)]
/// K_eff(x) = tr(Ŵ2)²(1 + 2x) / [tr(Ŵ2²) + 2x·tr(Ŵ2)·λ_max(Ŵ2)]    [同 Eq. (4), (33)]
/// 臨界値 = χ²_{K_eff}(x·K_eff)の上側α点 / K_eff                    [同 Eq. (32)]
/// ```
///
/// 単純版TSLSは`x = 1/τ`。`K_eff`は`Ŵ2`のスケールに依らない。`Classical`では`Ŵ2 = ω̂_22·I`で
/// `K_eff = K`、`F_eff`はCragg-Donaldの`F`に一致する。
///
/// - `y_endog`: 内生変数(n×1。1個だけ)、`x_exog`: 外生変数(n×k_exog。定数項を使うなら定数列を含め、
///   平均は引かず他の列と同じく偏回帰で落とす)、`z`: 除外操作変数(n×K)、`alpha`: 臨界値の水準(0〜1)。
/// - `y_dep`: 被説明変数(n×1)。渡すと、被説明変数の誘導形の残差もスコアに加えた`2K×2K`の`Ŵ`から、一般化版
///   TSLSの臨界値(`x = B_TSLS(Ŵ)/τ`、論文 Eq. (5), (21))も計算する。`None`なら単純版だけ。**`y_dep`は
///   `y_endog`・`x_exog`・`z`と同じ変換(`√w`の重み、within変換など)を済ませた形で渡す**(変換が違うと、被説明変数の
///   誘導形と一次段階の残差が別の変換になり、`W1`・`W12`と`B_TSLS`が誤る)。
/// - `Ŵ2`は半正定値でよい(クラスターや観測が少なくてランク落ちでも、トレースと固有値だけを使う)。
///
/// 次のとき全体が`None`: 内生変数が1個でない、`K = 0`、`alpha`が(0,1)の外、`X`がフルランクでない、`Z`の列が
/// `X`の張る空間にほぼ入っている、`Z̃`が共線、一次段階が完全フィット(`1 − R² ≤ 1.4e-14`)、HC2・HC3で
/// レバレッジが1、`n_absorbed > 0`のHC2・HC3、クラスターが2未満、`Hac`の時間順の長さが行数と違う・`lags >= n`、
/// 自由度`n − K − k_exog − n_absorbed`が1未満、`Ŵ2`が不定値(最小固有値が最大固有値の`-1e-10`倍より小さい)かゼロ、
/// 入力に非有限値がある。一般化版(`tsls`・`bias_bound`)だけが`None`になるのは: `y_dep`の形が不正・外生変数の
/// 張る空間にほぼ入っている・非有限値、`2K×2K`の`Ŵ`が不定値、`v1 = βv2`のほぼ比例(`B_TSLS`が未定義)、
/// `B_TSLS`の格子の評価の失敗が1%を超える。統計量と単純版は返す。
pub(crate) fn effective_f(
    y_endog: &Mat<f64>,
    x_exog: &Mat<f64>,
    z: &Mat<f64>,
    n_absorbed: usize,
    covariance: ScoreCovariance<'_>,
    alpha: f64,
    y_dep: Option<&Mat<f64>>,
) -> Option<EffectiveF> {
    let (n, k) = (y_endog.nrows(), z.ncols());
    debug_assert_eq!(z.nrows(), n);
    debug_assert_eq!(x_exog.nrows(), n);
    if y_endog.ncols() != 1 || k == 0 || n == 0 || !(alpha > 0.0 && alpha < 1.0) {
        return None;
    }
    let first_stage = FirstStage::new(y_endog, x_exog, z)?;
    let n_f = n as f64;
    let n_params = k + x_exog.ncols() + n_absorbed;

    // スコア`ψ_i = (ũ_i e1_i, ũ_i e2_i)`(`e1`は被説明変数の誘導形の残差、`e2`は一次段階の残差)から
    // 共分散の推定値`Ŵ = c · meat / n`を作る。被説明変数が無い(または誘導形が作れない)ときは`e2`だけ。
    let reduced_form = y_dep.and_then(|y| first_stage.reduced_form_residuals(y, x_exog));
    let series: Vec<&[f64]> = match &reduced_form {
        Some(e1) => vec![e1.as_slice(), first_stage.residuals.as_slice()],
        None => vec![first_stage.residuals.as_slice()],
    };
    let width = series.len() * k;
    let psi = Mat::from_fn(n, width, |i, col| {
        *first_stage.u.get(i, col % k) * series[col / k][i]
    });
    let (meat, n_groups) = score_meat(
        &psi,
        covariance,
        |variant| hc_row_scales(variant, &first_stage.u, x_exog, n_params, n_absorbed),
        || {
            // 共分散がクロネッカー積なら`Ŵ ∝ Ω̂⊗I`。`Ũ'Ũ/n = I`なので、`meat`のブロック`(s, t)`は
            // `(Σ e_s e_t)·I_K`。
            Mat::from_fn(width, width, |a, b| {
                if a % k != b % k {
                    return 0.0;
                }
                series[a / k]
                    .iter()
                    .zip(series[b / k])
                    .map(|(x, y)| x * y)
                    .sum()
            })
        },
    )?;
    let multiplier = covariance_multiplier(covariance, n, n_params, n_groups)?;
    let w = Mat::from_fn(width, width, |a, b| {
        multiplier * 0.5 * (*meat.get(a, b) + *meat.get(b, a)) / n_f
    });
    let last = width - k;
    let w2 = Mat::from_fn(k, k, |a, b| *w.get(last + a, last + b));

    let eigenvalues = psd_eigenvalues(&w2)?;
    let trace: f64 = eigenvalues.iter().sum();
    // `V̂2 = Ŵ2/n`なので`tr(V̂2) = tr(Ŵ2)/n`。
    let statistic = first_stage.coefficients.iter().map(|p| p * p).sum::<f64>() / (trace / n_f);
    if !statistic.is_finite() {
        return None;
    }

    let simplified_tsls = critical_value_set(&eigenvalues, alpha, |tau| 1.0 / tau)?;
    let generalized = reduced_form
        .as_ref()
        .and_then(|_| generalized_critical_values(&w, k, &eigenvalues, alpha));
    Some(EffectiveF {
        statistic,
        n_instruments: k,
        alpha,
        simplified_tsls,
        bias_bound: generalized.as_ref().map(|(bound, _)| *bound),
        tsls: generalized.map(|(_, values)| values),
    })
}

/// `τ`ごとのPatnaik臨界値の組(`x`は`x_of_tau(τ)`)。分位点が求まらないとき`None`。
fn critical_value_set(
    eigenvalues: &[f64],
    alpha: f64,
    x_of_tau: impl Fn(f64) -> f64,
) -> Option<[PatnaikCriticalValue; 4]> {
    let mut values = Vec::with_capacity(BIAS_THRESHOLDS.len());
    for tau in BIAS_THRESHOLDS {
        let x = x_of_tau(tau);
        let (effective_df, critical_value) = patnaik_critical_value(eigenvalues, x, alpha)?;
        values.push(PatnaikCriticalValue {
            tau,
            x,
            effective_df,
            critical_value,
        });
    }
    values.try_into().ok()
}

/// 一般化版TSLSの`(B_TSLS, 臨界値の組)`。`w`は`2K×2K`の`Ŵ`(前半が被説明変数、後半が一次段階)で、
/// `w2_eigenvalues`は`Ŵ2`(`w`の右下ブロック)の固有値。`w`が不定値のとき(`B_TSLS ≤ 1`が保証されない)や、
/// `B_TSLS`が定義できないとき`None`。
fn generalized_critical_values(
    w: &Mat<f64>,
    k: usize,
    w2_eigenvalues: &[f64],
    alpha: f64,
) -> Option<(f64, [PatnaikCriticalValue; 4])> {
    psd_eigenvalues(w)?;
    let block =
        |row: usize, col: usize| Mat::from_fn(k, k, |a, b| *w.get(row * k + a, col * k + b));
    let bound = tsls_bias_bound(&block(0, 0), &block(0, 1), &block(1, 1))?;
    let values = critical_value_set(w2_eigenvalues, alpha, |tau| bound / tau)?;
    Some((bound, values))
}

/// 半正定値の`Ŵ2`(対称)の固有値。丸め誤差の分の負の固有値は0にそろえる。最大固有値が正でない、
/// または最小固有値が最大固有値の`-1e-10`倍より小さい(不定値)とき`None`。
fn psd_eigenvalues(w2: &Mat<f64>) -> Option<Vec<f64>> {
    let values = w2.self_adjoint_eigenvalues(Side::Lower).ok()?;
    let largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
    if !(largest.is_finite() && largest > 0.0)
        || smallest < -NEGATIVE_EIGENVALUE_TOLERANCE * largest
    {
        return None;
    }
    Some(values.into_iter().map(|v| v.max(0.0)).collect())
}

/// `B_TSLS(Ŵ)`(論文 Theorem 1, Eq. (19), (21))。`w1`・`w12`・`w2`は`Ŵ`の`K×K`ブロック(被説明変数の
/// 誘導形・その交差・内生変数の誘導形)。
///
/// `β`ごとに`S1 = W1 − 2βW12 + β²W2`、`S12 = W12 − βW2`、`S2 = W2`として、`C0`(単位球面)についての
/// 上限は`S12`の対称部分の最小・最大固有値で解析的に取れる(Remark 5):
///
/// ```text
/// f(β) = max_{λ ∈ {λ_min, λ_max}} |tr(S12) − 2λ| / √(tr(S1)·tr(S2)),   B_TSLS = sup_β f(β)
/// ```
///
/// `β → ±∞`の極限は`max_{λ ∈ {λ_min(W2), λ_max(W2)}} |1 − 2λ/tr(W2)|`。`β`の上限は、`β = c + s·sinh u`の
/// `u`の等間隔の格子(`c`の近くは線形、遠くは対数間隔)で局所最大を探し、各局所最大の両隣の間を黄金分割探索で
/// 詰める(`f`は区間内で単峰)。`(c, s)`は2通り: `(0, β₀)`(`β₀ = √(tr(W1)/tr(W2))`)と、`tr(S1(β))`の
/// 最小点まわり`(β*, √(min tr(S1)/tr(W2)))`(誤差がほぼ比例するときの鋭い構造を拾う)。端の極限も候補に入れる。
///
/// 次のとき`None`: `tr(W2) <= 0`、`tr(W1) <= 0`、`min_β tr(S1(β))`が`tr(W1)`の1e-12倍以下
/// (被説明変数と内生変数の誘導形の誤差がほぼ比例)、非有限値。
fn tsls_bias_bound(w1: &Mat<f64>, w12: &Mat<f64>, w2: &Mat<f64>) -> Option<f64> {
    let k = w2.nrows();
    let trace = |m: &Mat<f64>| (0..k).map(|a| *m.get(a, a)).sum::<f64>();
    let (tr1, tr12, tr2) = (trace(w1), trace(w12), trace(w2));
    if !(tr1.is_finite() && tr1 > 0.0 && tr12.is_finite() && tr2.is_finite() && tr2 > 0.0) {
        return None;
    }
    // `tr(S1(β)) = min_tr_s1 + tr(W2)(β − β*)²`、`β* = tr(W12)/tr(W2)`。この形なら`β*`の近くで桁落ちしない。
    let min_tr_s1 = tr1 - tr12 * tr12 / tr2;
    let beta_star = tr12 / tr2;
    if min_tr_s1 <= DEGENERATE_BENCHMARK_TOLERANCE * tr1 {
        return None;
    }
    let sym12 = Mat::from_fn(k, k, |a, b| 0.5 * (*w12.get(a, b) + *w12.get(b, a)));
    let objective = |beta: f64| -> Option<f64> {
        let s12 = Mat::from_fn(k, k, |a, b| *sym12.get(a, b) - beta * *w2.get(a, b));
        let eigenvalues = s12.self_adjoint_eigenvalues(Side::Lower).ok()?;
        let smallest = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let largest = eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let tr_s12 = tr12 - beta * tr2;
        let tr_s1 = min_tr_s1 + tr2 * (beta - beta_star) * (beta - beta_star);
        // NaNは比較が偽になるので、`is_nan`も弾く。
        if tr_s1.is_nan() || tr_s1 <= 0.0 {
            return None;
        }
        let numerator = (tr_s12 - 2.0 * smallest)
            .abs()
            .max((tr_s12 - 2.0 * largest).abs());
        let value = numerator / (tr_s1 * tr2).sqrt();
        value.is_finite().then_some(value)
    };

    // `β → ±∞`の極限。
    let w2_eigenvalues = w2.self_adjoint_eigenvalues(Side::Lower).ok()?;
    let w2_smallest = w2_eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let w2_largest = w2_eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let limit = (1.0 - 2.0 * w2_smallest / tr2)
        .abs()
        .max((1.0 - 2.0 * w2_largest / tr2).abs());

    // 2通りのスケールで格子を張る。(1) `β = β₀ sinh u`(`β₀ = √(tr W1/tr W2)`): `W`全体のスケールの
    // 構造。(2) `β = β* + δ sinh u`(`β* = tr W12/tr W2`、`δ = √(min tr S1/tr W2)`): 被説明変数と内生変数の
    // 誘導形の誤差がほぼ比例する(`min tr S1`が小さい)と、`tr S1(β)`が最小になる`β*`の近くに幅`δ`の鋭い
    // 構造ができ、(1)の格子の刻みでは取りこぼす。
    let beta_scale = (tr1 / tr2).sqrt();
    let near_minimizer_scale = (min_tr_s1 / tr2).sqrt();
    let by_overall_scale = search_beta_grid(&|u: f64| objective(beta_scale * u.sinh()))?;
    let by_minimizer_scale =
        search_beta_grid(&|u: f64| objective(beta_star + near_minimizer_scale * u.sinh()))?;
    let best = limit.max(by_overall_scale).max(by_minimizer_scale);
    // 論文のTheorem 1.3で`B_TSLS ≤ 1`。超える値は丸め誤差の範囲のはず。
    debug_assert!(best <= 1.0 + 1e-6, "B_TSLS exceeds 1: {best}");
    Some(best.min(1.0))
}

/// `u ∈ [−U, U]`の等間隔の格子で`objective(u)`の局所最大を探し、上位だけを黄金分割で詰めて、最大値を返す。
/// 格子の評価が失敗した点は飛ばす(1%を超えたら`None`)。端の極限は呼び出し側が候補に入れる。
fn search_beta_grid(objective: &impl Fn(f64) -> Option<f64>) -> Option<f64> {
    let at_or_lowest = |u: f64| objective(u).unwrap_or(f64::NEG_INFINITY);
    let u_at = |j: usize| BIAS_GRID_HALF_WIDTH * (2.0 * j as f64 / BIAS_GRID_POINTS as f64 - 1.0);
    let mut values = Vec::with_capacity(BIAS_GRID_POINTS + 1);
    let mut failures = 0usize;
    for j in 0..=BIAS_GRID_POINTS {
        let value = at_or_lowest(u_at(j));
        failures += usize::from(value == f64::NEG_INFINITY);
        values.push(value);
    }
    if failures as f64 > MAX_FAILED_GRID_FRACTION * (BIAS_GRID_POINTS + 1) as f64 {
        return None;
    }

    // 内側の局所最大(左より丸め誤差を超えて大きく、右以下)は、値の大きい順に上位だけ、両隣の間を
    // 黄金分割で詰める(平坦な区間の丸め誤差の波は数えない)。
    let mut best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut peaks: Vec<usize> = (1..BIAS_GRID_POINTS)
        .filter(|&j| {
            values[j] > values[j - 1] + PEAK_RIPPLE_TOLERANCE && values[j] >= values[j + 1]
        })
        .collect();
    peaks.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    for &j in peaks.iter().take(MAX_REFINED_PEAKS) {
        best = best.max(maximize_golden(&at_or_lowest, u_at(j - 1), u_at(j + 1)));
    }
    best.is_finite().then_some(best)
}

/// 区間`[lo, hi]`での`f`の最大値(黄金分割探索。`f`が区間内で単峰のとき最大値を返す)。
/// 評価した点の最大値を返すので、単峰でなくても過大にはならない。
fn maximize_golden(f: &impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut x1 = hi - ratio * (hi - lo);
    let mut x2 = lo + ratio * (hi - lo);
    let (mut f1, mut f2) = (f(x1), f(x2));
    let mut best = f1.max(f2);
    while hi - lo > GOLDEN_SECTION_WIDTH {
        if f1 < f2 {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + ratio * (hi - lo);
            f2 = f(x2);
            best = best.max(f2);
        } else {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - ratio * (hi - lo);
            f1 = f(x1);
            best = best.max(f1);
        }
    }
    best
}

/// 有効自由度`K_eff`とPatnaik臨界値(論文 Eq. (33), (32))。`eigenvalues`は`Ŵ2`の固有値(0以上)。
/// `x`は非心度の入力、`alpha`は水準。`Ŵ2`のスケールに依らない(最大固有値で割ってから計算する)。
/// 固有値が正でないとき、分位点が求まらないときは`None`。`alpha`が1e-8より小さいと、分位点の計算の
/// 精度が水準に対して足りない(下側確率`1 − α`が1に近すぎる)。
fn patnaik_critical_value(eigenvalues: &[f64], x: f64, alpha: f64) -> Option<(f64, f64)> {
    let largest = eigenvalues.iter().copied().fold(0.0, f64::max);
    if !(largest > 0.0 && largest.is_finite() && x >= 0.0 && x.is_finite()) {
        return None;
    }
    let scaled: Vec<f64> = eigenvalues.iter().map(|v| v / largest).collect();
    let trace: f64 = scaled.iter().sum();
    let trace_of_square: f64 = scaled.iter().map(|v| v * v).sum();
    // 最大固有値は1に正規化してある。
    let effective_df = trace * trace * (1.0 + 2.0 * x) / (trace_of_square + 2.0 * x * trace);
    let quantile = noncentral_chi_squared_quantile(1.0 - alpha, effective_df, x * effective_df)?;
    Some((effective_df, quantile / effective_df))
}

/// 外生変数で直交化した内生変数と、直交正規化した操作変数、一次段階の係数と残差。
struct FirstStage {
    /// 直交正規化した操作変数(n×K。`Ũ'Ũ/n = I`)。
    u: Mat<f64>,
    /// 一次段階の係数`π̂ = Ũ'Ỹ/n`。
    coefficients: Vec<f64>,
    /// 一次段階の残差`e = Ỹ − Ũπ̂`。
    residuals: Vec<f64>,
}

impl FirstStage {
    fn new(y_endog: &Mat<f64>, x_exog: &Mat<f64>, z: &Mat<f64>) -> Option<Self> {
        let (n, k) = (y_endog.nrows(), z.ncols());
        let (y, z) = rms_orthogonalized(y_endog, x_exog, z)?;

        let n_f = n as f64;
        let gram = z.transpose() * &z;
        let q_zz = Mat::from_fn(k, k, |a, b| *gram.get(a, b) / n_f);
        // `Ũ = Z̃ C⁻ᵀ`(`Q_zz = CC'`)。`Ũ'Ũ/n = C⁻¹ Q_zz C⁻ᵀ = I`。
        let c_inv = inverse_cholesky_factor(&q_zz)?;
        let u = &z * c_inv.transpose();

        let u_y = u.transpose() * &y;
        let coefficients: Vec<f64> = (0..k).map(|a| *u_y.get(a, 0) / n_f).collect();
        let residuals: Vec<f64> = (0..n)
            .map(|i| {
                let fitted: f64 = (0..k).map(|a| *u.get(i, a) * coefficients[a]).sum();
                *y.get(i, 0) - fitted
            })
            .collect();
        // `Ỹ`の二乗平均は1なので、残差の二乗平均は`1 − R²`。Cragg-Donald・Kleibergen-Paapと同じ基準で、
        // 完全フィットなら統計量は無意味。
        let mean_square_residual = residuals.iter().map(|e| e * e).sum::<f64>() / n_f;
        if !mean_square_residual.is_finite() || mean_square_residual <= EXACT_FIT_TOLERANCE {
            return None;
        }
        if coefficients.iter().any(|p| !p.is_finite()) {
            return None;
        }
        Some(Self {
            u,
            coefficients,
            residuals,
        })
    }

    /// 被説明変数`y`の誘導形(外生変数で直交化した`ỹ`を`Ũ`に回帰)の残差`e1`。`y`が1列でない・行数が
    /// 合わない、`y`が外生変数の張る空間にほぼ入っている、非有限値があるとき`None`。
    fn reduced_form_residuals(&self, y_dep: &Mat<f64>, x_exog: &Mat<f64>) -> Option<Vec<f64>> {
        let n = self.u.nrows();
        if y_dep.ncols() != 1 || y_dep.nrows() != n {
            return None;
        }
        // 操作変数の列は要らない(`y`だけを外生変数で直交化する)ので、列数0の行列を渡す。
        let (y_unit, _) = orthogonalize_unit(y_dep, x_exog, &Mat::<f64>::zeros(n, 0))?;
        let root_n = (n as f64).sqrt();
        let n_f = n as f64;
        let k = self.u.ncols();
        let y: Vec<f64> = (0..n).map(|i| *y_unit.get(i, 0) * root_n).collect();
        let coefficients: Vec<f64> = (0..k)
            .map(|a| (0..n).map(|i| *self.u.get(i, a) * y[i]).sum::<f64>() / n_f)
            .collect();
        let residuals: Vec<f64> = (0..n)
            .map(|i| {
                y[i] - (0..k)
                    .map(|a| *self.u.get(i, a) * coefficients[a])
                    .sum::<f64>()
            })
            .collect();
        residuals.iter().all(|e| e.is_finite()).then_some(residuals)
    }
}

#[cfg(test)]
mod tests;
