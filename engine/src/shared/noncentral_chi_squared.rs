//! 非心カイ二乗分布のCDFと分位点。statrsに無いので、ポアソン混合で自前に実装する。
//!
//! 弱操作変数診断(`effective_f`)の臨界値が、自由度が小数の非心カイ二乗分布の分位点を使う。
//!
//! `X ~ χ²_df(ncp)`のCDFは、`J ~ Poisson(ncp/2)`として`P(X ≤ x) = Σ_j P(J = j) · P(χ²_{df + 2j} ≤ x)`。
//! ポアソンの重みは、最頻値の重みを1とした相対値を漸化式で両側にたどり、十分小さくなったところで
//! 打ち切って、残した重みの合計が1になるように正規化する(`ln Γ`を使わず、`ncp`が大きくても桁落ち・
//! アンダーフローが重み全体に共通の係数になって正規化で消える)。重みは`ncp`だけで決まるので、
//! 分位点の探索では1回だけ作って使い回す。分位点はCDFが`x`について単調増加であることを使い、
//! 区間を広げて挟んでから二分法で求める。
//!
//! 精度の目安: 水準`1 − p`が1e-8より小さい分位点は、CDFの丸め誤差(約1e-16)が水準に対して無視できない。

use statrs::distribution::{ChiSquared, ContinuousCDF};

/// 最頻値に対する相対重みがこれ未満になったら、残りの項を無視する(各項のCDFは1以下)。
const WEIGHT_CUTOFF: f64 = 1e-18;

/// 項の数の上限(ポアソン平均が極端に大きいときの無限ループの防止)。
const MAX_TERMS: usize = 1_000_000;

/// ポアソン平均`ncp/2`の上限。これを超えると項の数が上限を超えるので`None`にする(`17√λ`個ほど要る)。
const MAX_POISSON_MEAN: f64 = 1e9;

/// 分位点の二分法で、区間幅が上側の端の何倍以下になったら止めるか(答えに対する相対幅)。
const QUANTILE_RELATIVE_WIDTH: f64 = 1e-14;

/// 二分法の反復回数の上限(極端に小さい分位点でも相対幅が収束するのに十分)。
const MAX_BISECTIONS: usize = 2_000;

/// 正規化したポアソン重み。`weights[i]`は`J = first + i`の確率。
struct PoissonWeights {
    first: usize,
    weights: Vec<f64>,
}

impl PoissonWeights {
    /// 平均`lambda`(0以上)の重み。`lambda`が上限を超える・項が多すぎるとき`None`。
    fn new(lambda: f64) -> Option<Self> {
        if lambda == 0.0 {
            return Some(Self {
                first: 0,
                weights: vec![1.0],
            });
        }
        if !(lambda > 0.0 && lambda <= MAX_POISSON_MEAN) {
            return None;
        }
        let mode = lambda.floor() as usize;
        let mut above = Vec::new();
        let mut weight = 1.0;
        let mut j = mode;
        loop {
            weight *= lambda / (j as f64 + 1.0);
            j += 1;
            if weight < WEIGHT_CUTOFF {
                break;
            }
            above.push(weight);
            if above.len() > MAX_TERMS {
                return None;
            }
        }
        let mut below = Vec::new();
        let mut weight = 1.0;
        let mut j = mode;
        while j > 0 {
            weight *= (j as f64) / lambda;
            j -= 1;
            if weight < WEIGHT_CUTOFF {
                break;
            }
            below.push(weight);
            if below.len() > MAX_TERMS {
                return None;
            }
        }
        let first = mode - below.len();
        below.reverse();
        below.push(1.0);
        below.extend(above);
        let total: f64 = below.iter().sum();
        let weights = below.into_iter().map(|w| w / total).collect();
        Some(Self { first, weights })
    }

    /// 混合分布のCDF `Σ_j w_j · P(χ²_{df + 2j} ≤ x)`(`x > 0`)。
    fn cdf(&self, x: f64, df: f64) -> Option<f64> {
        let mut total = 0.0;
        for (i, weight) in self.weights.iter().enumerate() {
            let central = ChiSquared::new(df + 2.0 * (self.first + i) as f64).ok()?;
            total += weight * central.cdf(x);
        }
        Some(total.clamp(0.0, 1.0))
    }
}

fn valid_parameters(df: f64, ncp: f64) -> bool {
    df.is_finite() && df > 0.0 && ncp.is_finite() && ncp >= 0.0
}

/// 非心カイ二乗分布`χ²_df(ncp)`のCDF `P(X ≤ x)`。
///
/// `df`は正の小数でよい。`ncp`は0以上(0なら中心カイ二乗分布)。`df`・`ncp`・`x`が有限でない、
/// `df <= 0`、`ncp < 0`のときと、`ncp`が大きすぎて(`ncp/2`が1e9超)項の数が上限を超えるとき`None`。
/// `x <= 0`は0。
///
/// 分位点は内部で重みを使い回すので、この関数は検証用(テストから分位点と突き合わせる)。
#[cfg(test)]
pub(crate) fn noncentral_chi_squared_cdf(x: f64, df: f64, ncp: f64) -> Option<f64> {
    if !valid_parameters(df, ncp) || x.is_nan() {
        return None;
    }
    if x <= 0.0 {
        return Some(0.0);
    }
    if x == f64::INFINITY {
        return Some(1.0);
    }
    PoissonWeights::new(ncp / 2.0)?.cdf(x, df)
}

/// 非心カイ二乗分布`χ²_df(ncp)`の下側`p`分位点(`P(X ≤ q) = p`となる`q`)。
///
/// `p`は0より大きく1より小さい値。`df`・`ncp`の条件は[`noncentral_chi_squared_cdf`]と同じ。
/// 区間が見つからないとき(64回倍にしても`p`に届かない)や、`ncp`が大きすぎるときは`None`。
pub(crate) fn noncentral_chi_squared_quantile(p: f64, df: f64, ncp: f64) -> Option<f64> {
    if !(p > 0.0 && p < 1.0) || !valid_parameters(df, ncp) {
        return None;
    }
    let weights = PoissonWeights::new(ncp / 2.0)?;
    // 平均`df + ncp`と標準偏差`√(2(df + 2ncp))`を目安に上限を置き、CDFが`p`を超えるまで倍にする。
    let mean = df + ncp;
    let sd = (2.0 * (df + 2.0 * ncp)).sqrt();
    let mut hi = mean + 10.0 * sd + 10.0;
    let mut doublings = 0;
    while weights.cdf(hi, df)? < p {
        hi *= 2.0;
        doublings += 1;
        if doublings > 64 {
            return None;
        }
    }
    let mut lo = 0.0;
    for _ in 0..MAX_BISECTIONS {
        if hi - lo <= QUANTILE_RELATIVE_WIDTH * hi {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if weights.cdf(mid, df)? < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 期待値はscipy 1.18の`scipy.stats.ncx2`(`cdf`・`ppf`)。
    // (df, ncp, x, cdf)。xは`ppf(0.5)`・`ppf(0.9)`。
    const CDF_CASES: [(f64, f64, f64, f64); 18] = [
        (1.0, 0.0, 0.454936423119572, 0.5),
        (1.0, 0.0, 2.705543454095404, 0.9),
        (1.0, 1.0, 1.1036433113367594, 0.5),
        (1.0, 1.0, 5.218794098621895, 0.9),
        (1.0, 10.0, 10.000000002013074, 0.5),
        (1.0, 10.0, 19.74761818730153, 0.9),
        (2.5, 3.3, 4.873629587756065, 0.5),
        (2.5, 3.3, 11.576735610574797, 0.9),
        (4.0, 40.0, 43.0122250989059, 0.5),
        (4.0, 40.0, 61.14711395955812, 0.9),
        (7.3, 60.0, 66.31678533187808, 0.5),
        (7.3, 60.0, 88.30084194900323, 0.9),
        (30.0, 600.0, 629.0078731138419, 0.5),
        (30.0, 600.0, 694.17450490644, 0.9),
        (0.4, 2.0, 1.3248537295502205, 0.5),
        (0.4, 2.0, 6.401297871371592, 0.9),
        (12.7, 150.0, 161.71255999022708, 0.5),
        (12.7, 150.0, 195.33253227128057, 0.9),
    ];

    // (p, df, ncp, quantile)
    const QUANTILE_CASES: [(f64, f64, f64, f64); 27] = [
        (0.05, 1.0, 0.0, 0.003932140000019522),
        (0.95, 1.0, 0.0, 3.841458820694124),
        (0.99, 1.0, 0.0, 6.6348966010212145),
        (0.05, 1.0, 1.0, 0.010674708241848553),
        (0.95, 1.0, 1.0, 7.002086262339715),
        (0.99, 1.0, 1.0, 11.066480529979517),
        (0.05, 1.0, 10.0, 2.302617962488756),
        (0.95, 1.0, 10.0, 23.108511211606643),
        (0.99, 1.0, 10.0, 30.125010254773347),
        (0.05, 2.5, 3.3, 0.7250018806305892),
        (0.95, 2.5, 3.3, 14.054386798597669),
        (0.99, 2.5, 3.3, 19.397248567306672),
        (0.05, 4.0, 40.0, 24.489705498804945),
        (0.95, 4.0, 40.0, 66.8798504297499),
        (0.99, 4.0, 40.0, 78.339565834942),
        (0.05, 7.3, 60.0, 42.83174933104483),
        (0.95, 7.3, 60.0, 95.12221797444636),
        (0.99, 7.3, 60.0, 108.6212052509591),
        (0.05, 30.0, 600.0, 550.1427575711551),
        (0.95, 30.0, 600.0, 713.241479450483),
        (0.99, 30.0, 600.0, 749.7149156796679),
        (0.05, 0.4, 2.0, 6.0523096404272264e-05),
        (0.95, 0.4, 2.0, 8.450482603083147),
        (0.99, 0.4, 2.0, 13.013918809356978),
        (0.05, 12.7, 150.0, 123.31406726046991),
        (0.95, 12.7, 150.0, 205.4542150461167),
        (0.99, 12.7, 150.0, 225.1455009113921),
    ];

    // 非心度が大きい・水準が極端な分位点(scipy 1.18の`ppf`)。(p, df, ncp, quantile)
    const EXTREME_QUANTILE_CASES: [(f64, f64, f64, f64); 14] = [
        (0.999, 1.0, 2000.0, 2285.9483158203648),
        (0.9999, 1.0, 2000.0, 2346.4700304556322),
        (0.99999999, 1.0, 2000.0, 2533.447207200726),
        (0.999, 5.5, 5000.0, 5451.171444509995),
        (0.9999, 5.5, 5000.0, 5544.395889174869),
        (0.99999999, 5.5, 5000.0, 5829.825534512858),
        (0.999, 30.0, 10000.0, 10657.039854551052),
        (0.9999, 30.0, 10000.0, 10787.16736254972),
        (0.99999999, 30.0, 10000.0, 11183.69392795866),
        (0.999, 3.0, 30.0, 75.88812878739952),
        (0.9999, 3.0, 30.0, 87.1392355641182),
        (0.99999999, 3.0, 30.0, 125.76291847874887),
        (0.0001, 1.0, 10.0, 0.00034563240783530623),
        (1e-08, 1.0, 10.0, 3.4599091562396696e-12),
    ];

    #[test]
    fn quantile_matches_scipy_for_large_noncentrality_and_extreme_levels() {
        for (p, df, ncp, expected) in EXTREME_QUANTILE_CASES {
            let q = noncentral_chi_squared_quantile(p, df, ncp).unwrap();
            assert!(
                (q - expected).abs() <= 1e-7 * expected,
                "p={p} df={df} ncp={ncp}: {q} vs {expected}"
            );
        }
    }

    /// 小さい分位点でも、答えに対する相対精度で求まる(区間の幅ではなく)。
    #[test]
    fn small_quantiles_have_relative_accuracy() {
        let q = noncentral_chi_squared_quantile(0.05, 0.4, 2.0).unwrap();
        assert!((q - 6.0523096404272264e-05).abs() <= 1e-12 * 6.0523096404272264e-05);
    }

    #[test]
    fn an_excessive_noncentrality_returns_none() {
        assert!(noncentral_chi_squared_cdf(1e10, 2.0, 4e9).is_none());
        assert!(noncentral_chi_squared_quantile(0.5, 2.0, 4e9).is_none());
        assert!(noncentral_chi_squared_cdf(1e10, 2.0, 1e30).is_none());
    }

    #[test]
    fn cdf_matches_scipy() {
        for (df, ncp, x, expected) in CDF_CASES {
            let cdf = noncentral_chi_squared_cdf(x, df, ncp).unwrap();
            assert!(
                (cdf - expected).abs() < 1e-10,
                "df={df} ncp={ncp} x={x}: {cdf} vs {expected}"
            );
        }
    }

    #[test]
    fn quantile_matches_scipy() {
        for (p, df, ncp, expected) in QUANTILE_CASES {
            let q = noncentral_chi_squared_quantile(p, df, ncp).unwrap();
            assert!(
                (q - expected).abs() <= 1e-9 * expected,
                "p={p} df={df} ncp={ncp}: {q} vs {expected}"
            );
        }
    }

    #[test]
    fn without_noncentrality_it_is_the_central_distribution() {
        for (df, x) in [(1.0, 0.7), (3.0, 2.2), (6.5, 9.0)] {
            let expected = ChiSquared::new(df).unwrap().cdf(x);
            assert_eq!(noncentral_chi_squared_cdf(x, df, 0.0).unwrap(), expected);
        }
    }

    #[test]
    fn cdf_is_monotone_and_the_quantile_inverts_it() {
        let (df, ncp) = (3.7, 25.0);
        let mut previous = 0.0;
        for k in 1..=60 {
            let cdf = noncentral_chi_squared_cdf(k as f64, df, ncp).unwrap();
            assert!(cdf >= previous);
            previous = cdf;
        }
        for p in [0.01, 0.2, 0.5, 0.8, 0.999] {
            let q = noncentral_chi_squared_quantile(p, df, ncp).unwrap();
            let back = noncentral_chi_squared_cdf(q, df, ncp).unwrap();
            assert!((back - p).abs() < 1e-11, "p={p} back={back}");
        }
    }

    #[test]
    fn it_handles_the_support_boundaries() {
        assert_eq!(noncentral_chi_squared_cdf(0.0, 2.0, 5.0), Some(0.0));
        assert_eq!(noncentral_chi_squared_cdf(-1.0, 2.0, 5.0), Some(0.0));
        assert_eq!(
            noncentral_chi_squared_cdf(f64::INFINITY, 2.0, 5.0),
            Some(1.0)
        );
    }

    #[test]
    fn invalid_arguments_return_none() {
        assert!(noncentral_chi_squared_cdf(1.0, 0.0, 1.0).is_none());
        assert!(noncentral_chi_squared_cdf(1.0, -1.0, 1.0).is_none());
        assert!(noncentral_chi_squared_cdf(1.0, 1.0, -0.1).is_none());
        assert!(noncentral_chi_squared_cdf(1.0, f64::NAN, 1.0).is_none());
        assert!(noncentral_chi_squared_cdf(1.0, 1.0, f64::INFINITY).is_none());
        assert!(noncentral_chi_squared_cdf(f64::NAN, 1.0, 1.0).is_none());
        for p in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
            assert!(noncentral_chi_squared_quantile(p, 2.0, 1.0).is_none());
        }
        assert!(noncentral_chi_squared_quantile(0.5, 0.0, 1.0).is_none());
        assert!(noncentral_chi_squared_quantile(0.5, 1.0, -1.0).is_none());
    }

    /// ポアソン平均が大きくても、最頻値からたどるのでアンダーフローしない。
    #[test]
    fn it_works_for_a_large_noncentrality() {
        let cdf = noncentral_chi_squared_cdf(10_050.0, 5.0, 10_000.0).unwrap();
        assert!(cdf > 0.0 && cdf < 1.0);
        let q = noncentral_chi_squared_quantile(0.95, 5.0, 10_000.0).unwrap();
        let back = noncentral_chi_squared_cdf(q, 5.0, 10_000.0).unwrap();
        assert!((back - 0.95).abs() < 1e-10);
    }
}
