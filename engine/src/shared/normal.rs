//! 標準正規分布の累積分布関数`Φ`・生存関数`1-Φ`（`libm::erfc`ベース）と、それを使う分布型
//! [`StandardNormal`]。
//!
//! `statrs`の`Normal::cdf`・`sf`は`erfc`の近似の精度が低く、相対誤差が最大約`1e-10`
//! （`z=-0.75`付近）ある。`libm::erfc`は約`1e-14`（130桁の参照値との比較。`Φ`は`[-8.2, 8.2]`、
//! 生存関数は`[0, 10]`）で、`statrs`より10〜25%速い。引数`z/√2`の丸めが`erfc`で`z²`倍に増幅されるので、
//! 相対誤差は約`z²·1e-16`（`|z|≲10`で`1e-14`程度。`|z|≳37.5`は`erfc`が非正規化数・0になる）。Probit・Tobitの尤度・予測・限界効果と、
//! z検定のp値（Logit・Probit・Tobit・GMM）はここを通す。密度`φ`は`statrs`の`pdf`が`3e-16`で
//! 正確なので置き換えない。臨界値（`inverse_cdf`）も`statrs`の値が`2e-16`以内なのでそのまま使う。

use std::f64::consts::SQRT_2;

use statrs::distribution::{ContinuousCDF, Normal};
use statrs::statistics::{Max, Min};

/// 標準正規分布の累積分布関数`Φ(z)`。`z`が`NaN`なら`NaN`。
pub(crate) fn normal_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z / SQRT_2)
}

/// 標準正規分布の生存関数`1-Φ(z)=Φ(-z)`。`1 - Φ(z)`と違い、`z`が大きい裾でも桁落ちしない
/// （`z=10`で`7.6e-24`）。`z`が`NaN`なら`NaN`。
pub(crate) fn normal_sf(z: f64) -> f64 {
    0.5 * libm::erfc(z / SQRT_2)
}

/// 標準正規分布。`cdf`・`sf`を[`normal_cdf`]・[`normal_sf`]で評価する、z検定のp値と臨界値用の分布型
/// （[`crate::shared::inference`]の関数が`ContinuousCDF`を要求するため）。`statrs`の`Normal`と違い
/// 構築が失敗しない。
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StandardNormal;

impl Min<f64> for StandardNormal {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for StandardNormal {
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCDF<f64, f64> for StandardNormal {
    fn cdf(&self, x: f64) -> f64 {
        normal_cdf(x)
    }

    fn sf(&self, x: f64) -> f64 {
        normal_sf(x)
    }

    /// 分位点は`statrs`の値をそのまま使う（`0.90`・`0.95`・`0.975`・`0.995`で相対誤差`2e-16`以内）。
    fn inverse_cdf(&self, p: f64) -> f64 {
        Normal::standard().inverse_cdf(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 130桁の10進演算（`Φ(z)=1/2+φ(z)Σ z^{2n+1}/(2n+1)!!`）で求めた参照値。
    const CDF_REFERENCE: [(f64, f64); 9] = [
        (-8.0, 6.220960574271784e-16),
        (-5.0, 2.866515718791939e-7),
        (-3.0, 0.0013498980316300946),
        (-0.75, 0.2266273523768682),
        (-0.5, 0.3085375387259869),
        (1.0, 0.8413447460685429),
        (2.0, 0.9772498680518208),
        (3.5, 0.9997673709209645),
        (8.0, 0.9999999999999993),
    ];

    /// `Φ`は参照値と相対`1e-14`以内で一致する（`statrs`の`Normal::cdf`は`z=-0.75`付近で`1e-10`）。
    #[test]
    fn normal_cdf_matches_high_precision_reference_values() {
        for (z, expected) in CDF_REFERENCE {
            let got = normal_cdf(z);
            assert!(
                ((got - expected) / expected).abs() < 1e-14,
                "z={z}: got {got:e}, expected {expected:e}"
            );
        }
    }

    /// `statrs`の`Normal::cdf`が最もずれる`z=-1`付近（相対`8.6e-11`）で、`Φ`が機械精度で一致する。
    #[test]
    fn normal_cdf_is_accurate_at_minus_one() {
        let expected = 0.15865525393145707; // Φ(-1)
        let err = ((normal_cdf(-1.0) - expected) / expected).abs();
        assert!(err < 1e-15, "relative error {err:e}");
    }

    /// 生存関数は`Φ(-z)`と一致し、裾で`1 - Φ(z)`のように0に潰れない。
    #[test]
    fn normal_sf_mirrors_cdf_and_keeps_far_tail() {
        for (z, _) in CDF_REFERENCE {
            // `sf(z) = Φ(-z)`と、`Φ(z) + sf(z) = 1`。
            let mirrored = normal_cdf(-z);
            assert!(
                ((normal_sf(z) - mirrored) / mirrored).abs() < 1e-14,
                "z={z}"
            );
            assert!((normal_cdf(z) + normal_sf(z) - 1.0).abs() < 1e-15, "z={z}");
        }
        let sf10 = normal_sf(10.0);
        assert!(((sf10 - 7.619853024160525e-24) / 7.619853024160525e-24).abs() < 1e-13);
        assert_eq!(
            1.0 - normal_cdf(10.0),
            0.0,
            "premise: 1 - cdf loses the tail"
        );
    }

    #[test]
    fn normal_cdf_and_sf_handle_special_values() {
        assert_eq!(normal_cdf(0.0), 0.5);
        assert_eq!(normal_sf(0.0), 0.5);
        assert_eq!(normal_cdf(f64::INFINITY), 1.0);
        assert_eq!(normal_cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(normal_sf(f64::INFINITY), 0.0);
        assert_eq!(normal_sf(f64::NEG_INFINITY), 1.0);
        assert!(normal_cdf(f64::NAN).is_nan());
        assert!(normal_sf(f64::NAN).is_nan());
    }

    #[test]
    fn standard_normal_delegates_to_the_accurate_functions() {
        let dist = StandardNormal;
        assert_eq!(dist.cdf(-0.75), normal_cdf(-0.75));
        assert_eq!(dist.sf(0.75), normal_sf(0.75));
        assert_eq!(dist.min(), f64::NEG_INFINITY);
        assert_eq!(dist.max(), f64::INFINITY);
        // 分位点は`statrs`と同じで、`cdf`で往復できる。
        let z = dist.inverse_cdf(0.975);
        assert!(((z - 1.959963984540054) / 1.959963984540054).abs() < 1e-15);
        assert!((dist.cdf(z) - 0.975).abs() < 1e-15);
    }

    /// `inference`の関数に`StandardNormal`を通した結果。遠い裾のp値が0に潰れず（`1 - cdf`だと0）、
    /// 臨界値は`scipy.stats.norm.ppf`の値と一致する（`confidence_level`は0.8・0.95・0.99・0.999）。
    #[test]
    fn standard_normal_works_with_the_inference_functions() {
        use crate::shared::inference::{compute_inference_stat, critical_value};

        let dist = StandardNormal;
        let crit = critical_value(&dist, 0.95);
        let stat = compute_inference_stat(&dist, 10.0, 1.0, crit);
        let expected_p = 2.0 * 7.619853024160525e-24;
        assert!(((stat.p_value - expected_p) / expected_p).abs() < 1e-13);
        assert_eq!(stat.stat, 10.0);

        for (level, expected) in [
            (0.8, 1.2815515655446004),
            (0.95, 1.959963984540054),
            (0.99, 2.5758293035489004),
            (0.999, 3.2905267314919255),
        ] {
            let got = critical_value(&dist, level);
            assert!(
                ((got - expected) / expected).abs() < 1e-14,
                "level={level}: {got} vs {expected}"
            );
        }
    }
}
