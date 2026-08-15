//! 系統横断で共有するt/z検定の後処理ロジック。
//!
//! OLS（t分布）・Logit（z分布）で、係数から`std_err`/`stat`/`p_value`/
//! `conf_low`/`conf_high`を計算する処理がほぼ同型のまま独立実装されていたため、
//! `statrs::distribution::ContinuousCDF`をジェネリックに取る形でここに集約する
//! （Issue #152、`docs/planning/specs/panel-api-design.md` 4.2節）。
//! FE/RE・IVの2SLS（t分布）・GMM（z分布）でも同じ関数を使う想定。

use statrs::distribution::ContinuousCDF;

/// 単一の係数に対する検定統計量（t統計量またはz統計量）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InferenceStat {
    /// t統計量またはz統計量（`coef / se`）
    pub stat: f64,
    /// 両側p値
    pub p_value: f64,
    /// 信頼区間の下限
    pub conf_low: f64,
    /// 信頼区間の上限
    pub conf_high: f64,
}

/// 信頼区間の両側臨界値を計算する。
///
/// `confidence_level`は`(0, 1)`の範囲（例: `0.95`）であることを呼び出し側で
/// 事前に検証済みであることを前提とする。複数係数で使い回すため、この関数は
/// 一度だけ呼び出し、結果を`compute_inference_stat`に渡す。
pub fn critical_value<D>(dist: &D, confidence_level: f64) -> f64
where
    D: ContinuousCDF<f64, f64>,
{
    let alpha = 1.0 - confidence_level;
    dist.inverse_cdf(1.0 - alpha / 2.0)
}

/// 係数と標準誤差からt統計量/z統計量・両側p値・信頼区間を計算する。
///
/// `crit`は`critical_value`で事前に計算した臨界値を渡す。
pub fn compute_inference_stat<D>(dist: &D, coef: f64, se: f64, crit: f64) -> InferenceStat
where
    D: ContinuousCDF<f64, f64>,
{
    let stat = coef / se;
    let p_value = 2.0 * (1.0 - dist.cdf(stat.abs()));
    InferenceStat {
        stat,
        p_value,
        conf_low: coef - crit * se,
        conf_high: coef + crit * se,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{Normal, StudentsT};

    #[test]
    fn critical_value_matches_known_normal_quantile() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let crit = critical_value(&normal, 0.95);
        assert!((crit - 1.959_963_984_540_054).abs() < 1e-9);
    }

    #[test]
    fn compute_inference_stat_matches_manual_normal_calculation() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let crit = critical_value(&normal, 0.95);
        let coef = 2.0;
        let se = 0.5;
        let result = compute_inference_stat(&normal, coef, se, crit);

        let expected_stat = coef / se;
        let expected_p = 2.0 * (1.0 - normal.cdf(expected_stat.abs()));

        assert!((result.stat - expected_stat).abs() < 1e-12);
        assert!((result.p_value - expected_p).abs() < 1e-12);
        assert!((result.conf_low - (coef - crit * se)).abs() < 1e-12);
        assert!((result.conf_high - (coef + crit * se)).abs() < 1e-12);
    }

    #[test]
    fn compute_inference_stat_works_with_students_t() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let crit = critical_value(&t_dist, 0.95);
        let coef = -1.0;
        let se = 0.25;
        let result = compute_inference_stat(&t_dist, coef, se, crit);

        let expected_stat = coef / se;
        let expected_p = 2.0 * (1.0 - t_dist.cdf(expected_stat.abs()));

        assert!((result.stat - expected_stat).abs() < 1e-12);
        assert!((result.p_value - expected_p).abs() < 1e-12);
        assert!((result.conf_low - (coef - crit * se)).abs() < 1e-12);
        assert!((result.conf_high - (coef + crit * se)).abs() < 1e-12);
    }
}
