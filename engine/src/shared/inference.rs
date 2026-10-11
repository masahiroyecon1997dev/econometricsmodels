//! 系統横断で共有するt/z検定の後処理ロジック。
//!
//! OLS（t分布）・Logit（z分布）で、係数から`std_err`/`stat`/`p_value`/
//! `conf_low`/`conf_high`を計算する処理がほぼ同型のまま独立実装されていたため、
//! `statrs::distribution::ContinuousCDF`をジェネリックに取る形でここに集約する
//! （`docs/spec/panel-common.md` 4.2節）。
//! FE/RE・IVの2SLS（t分布）・GMM（z分布）でも同じ関数を使う想定。

use statrs::distribution::ContinuousCDF;

/// 検定統計量の従う分布（`test_stats`がt統計量かz統計量か、および自由度）。
///
/// t分布なら自由度を持ち、正規分布なら持たない、という関係を型で保証する。
/// 自由度は`df_resid`と一致するとは限らない（例: OLSの`cov_type=Cluster`は`G-1`）ため、
/// 利用者が`test_stats`からp値・信頼区間を再計算できるよう、実際に使った値を保持する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatDist {
    /// t分布（自由度`df`）。
    T { df: usize },
    /// 標準正規分布。
    Normal,
}

impl StatDist {
    /// 分布名（`"t"`または`"normal"`）。
    pub fn name(&self) -> &'static str {
        match self {
            StatDist::T { .. } => "t",
            StatDist::Normal => "normal",
        }
    }

    /// t分布のときの自由度。正規分布なら`None`。
    pub fn df(&self) -> Option<usize> {
        match self {
            StatDist::T { df } => Some(*df),
            StatDist::Normal => None,
        }
    }
}

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
///
/// `coef`・`se`が両方ちょうど0のとき`stat = coef / se = 0.0 / 0.0`はNaNになる
/// （`se`のみ0で`coef`が非0の場合は`stat = ±∞`になり、`ContinuousCDF::cdf`は無限大を
/// 正しく処理できるため問題にならない）。`statrs`の`StudentsT::cdf`はNaN入力で内部の
/// `beta_reg`が`Result::unwrap()`しているbetaの不完全関数の定義域チェックに失敗し
/// panicする（実際に踏む経路: `panel::re::swamy_arora_variance_components`の
/// between回帰で、全エンティティのグループ平均が完全一致し値が0の退化データ）。`stat`が
/// NaNのときは`cdf`を呼ばず`p_value`もNaNとする（統計的に不定であることをそのまま返す。
/// `OlsEstimator::fit`の`df_model==0`分岐——検定対象が無いモデルでNaNを返す——と同じ思想）。
pub fn compute_inference_stat<D>(dist: &D, coef: f64, se: f64, crit: f64) -> InferenceStat
where
    D: ContinuousCDF<f64, f64>,
{
    let stat = coef / se;
    let p_value = if stat.is_nan() {
        f64::NAN
    } else {
        // `1.0 - cdf`は裾でcdfが1.0に丸められp値が0になるため、生存関数`sf`を使う。
        2.0 * dist.sf(stat.abs())
    };
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
    fn compute_inference_stat_keeps_precision_in_the_far_tail() {
        // `1.0 - cdf`では裾でcdfが1.0に丸められp値が0.0になる（相対誤差が劣化する）ため
        // 生存関数`sf`を使う。R: `2*pt(-40, 20)` = 1.45746965543107e-20、
        // `2*pnorm(-20)` = 5.50724823721247e-89。
        let t_dist = StudentsT::new(0.0, 1.0, 20.0).unwrap();
        let crit = critical_value(&t_dist, 0.95);
        let t_result = compute_inference_stat(&t_dist, 40.0, 1.0, crit);
        assert!((t_result.p_value / 1.457_469_655_431_07e-20 - 1.0).abs() < 1e-8);
        // 符号が負でも対称（`stat.abs()`を使う）。
        let t_neg = compute_inference_stat(&t_dist, -40.0, 1.0, crit);
        assert!((t_neg.p_value / t_result.p_value - 1.0).abs() < 1e-12);

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_result = compute_inference_stat(&normal, 20.0, 1.0, critical_value(&normal, 0.95));
        assert!((z_result.p_value / 5.507_248_237_212_47e-89 - 1.0).abs() < 1e-8);
    }

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
    fn compute_inference_stat_returns_nan_p_value_without_panicking_when_coef_and_se_are_both_zero()
    {
        // coef=0.0, se=0.0だとstat=0.0/0.0=NaNになり、修正前はdist.cdf(NaN)が
        // panicしていた（statrsのbeta_reg内部でResult::unwrap()）。
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let crit = critical_value(&t_dist, 0.95);

        let result = compute_inference_stat(&t_dist, 0.0, 0.0, crit);

        assert!(result.stat.is_nan());
        assert!(result.p_value.is_nan());
        assert_eq!(result.conf_low, 0.0);
        assert_eq!(result.conf_high, 0.0);
    }

    #[test]
    fn compute_inference_stat_still_handles_infinite_stat_when_only_se_is_zero() {
        // se=0.0だがcoef!=0.0のときはstat=±∞になりNaNではないため、修正後もcdfを
        // 呼ぶ経路のまま（statrsは無限大を正しく処理できる）。
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let crit = critical_value(&t_dist, 0.95);

        let result = compute_inference_stat(&t_dist, 2.0, 0.0, crit);

        assert!(result.stat.is_infinite() && result.stat > 0.0);
        assert_eq!(result.p_value, 0.0);
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
