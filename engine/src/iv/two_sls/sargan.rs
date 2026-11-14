//! Sargan過剰識別検定。

use crate::iv::common::{IvError, IvInput, hcat};
use crate::shared::instrumental_variables;
use faer::Mat;

/// [`sargan_test`]の戻り値。
pub(super) struct SarganTest {
    pub(super) statistic: f64,
    pub(super) p_value: f64,
    pub(super) df: usize,
}

/// Sargan過剰識別検定（docs/spec/iv-spec.md 3.5節）。構造残差`e`を全操作変数
/// `Z = [x_exog, instruments]`に回帰した際の`n*R²`に相当する
/// `e'Z(Z'Z)⁻¹Z'e / σ̂²`（`σ̂² = e'e/n`）を計算する。自由度は
/// `len(instruments) - len(x_endog)`（`docs/spec/iv-spec.md`1.1節の`instruments`
/// ＝除外操作変数のみという定義に対応、`fit()`冒頭で`k_instruments() >= k_endog()`を
/// 検証済みのため常に0以上）。丁度識別（自由度0）では`None`（`docs/spec/iv-spec.md`
/// 1.2節・3.5節）。
///
/// **常に等分散（古典的）前提で計算し、`cov_type`には依存しない**（弱操作変数診断と
/// 同じ判断。Sargan検定はその定義自体が
/// 等分散前提の検定であり、不均一分散に頑健な版が欲しい場合はGMM＋Hansen J検定
/// （`gmm/hansen_j.rs`）を使うのが標準的な使い分けのため、`engine/src/iv/CLAUDE.md`参照）。
///
/// 計算は共有の`shared::instrumental_variables::sargan_test`で、`σ̂²`の分母（`n`）と自由度を
/// ここで決めて渡す。`Z = [x_exog, instruments]`は`OlsInput::from_columns`を経由せず行列代数
/// （`Z'Z`等）で使うため、定数項の扱いは通常の回帰変数と同じで良く、定数項込みの
/// `x_exog`をそのまま使う。`Z'Z`は第一段階回帰（同じ列を設計行列とする`OlsEstimator::fit`）が
/// `col_piv_qr`で既にfull column rankを検証済みのため、ここでの特異性は理論上到達不能
/// （`xtx_inverse`と同じ防御的`Result`化）。
pub(super) fn sargan_test(
    input: &IvInput,
    residuals: &Mat<f64>,
    ssr: f64,
) -> Result<Option<SarganTest>, IvError> {
    if input.k_instruments() == input.k_endog() {
        return Ok(None);
    }
    let df = input.k_instruments() - input.k_endog();
    let z = hcat(input.x_exog(), input.instruments());
    let sigma2 = ssr / (input.nobs() as f64);
    let test = instrumental_variables::sargan_test(&z, residuals, sigma2, df)?;
    Ok(Some(SarganTest {
        statistic: test.statistic,
        p_value: test.p_value,
        df,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iv::two_sls::TwoSlsEstimator;
    use crate::iv::two_sls::test_support::*;
    use crate::linear::ols::CovType;
    use faer::Side;
    use faer::prelude::Solve;

    /// 丁度識別（`len(instruments) == len(x_endog)`）ではSargan過剰識別検定の自由度が0の
    /// ため`None`になる（`docs/spec/iv-spec.md`1.2節・3.5節）。
    #[test]
    fn fit_sets_sargan_statistic_to_none_when_just_identified() {
        let estimator =
            TwoSlsEstimator::fit(just_identified_input(), CovType::Classical, 0.95).unwrap();
        assert_eq!(estimator.sargan_statistic(), None);
        assert_eq!(estimator.sargan_p_value(), None);
    }

    /// Sargan過剰識別検定の統計量を、`TwoSlsEstimator::fit`とは独立に構造残差`e`を
    /// 全操作変数`Z=[const, x1, z1, z2]`に回帰する形で手計算したオラクルと数値照合する
    /// （`fit()`のdocコメント「Sargan過剰識別検定」参照）。
    #[test]
    fn fit_computes_sargan_statistic_matching_manual_formula() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        let (_x_hat, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let (x1, _x_endog, z1, z2, _y) = nontrivial_x_exog_columns();
        let n = x1.len();

        let z = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let ztz = z.transpose() * &z;
        let zte = z.transpose() * &e;
        let ztz_inv_zte = ztz.llt(Side::Lower).unwrap().solve(&zte);
        let quad: f64 = (0..4)
            .map(|i| (*zte.get(i, 0)) * (*ztz_inv_zte.get(i, 0)))
            .sum();
        let ssr: f64 = (0..n).map(|i| (*e.get(i, 0)).powi(2)).sum();
        let expected_stat = quad / (ssr / (n as f64));
        let expected_p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(expected_stat);

        assert!((estimator.sargan_statistic().unwrap() - expected_stat).abs() < 1e-8);
        assert!((estimator.sargan_p_value().unwrap() - expected_p_value).abs() < 1e-8);
    }
}
