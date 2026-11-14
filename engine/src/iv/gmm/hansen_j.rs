//! Hansen J過剰識別検定。定義は`gmm.rs`のモジュールdoc「Hansen J過剰識別検定」参照。

use faer::Mat;

use crate::iv::common::{IvError, IvInput};
use crate::shared::gmm;

/// [`hansen_j_test`]の戻り値。
pub(super) struct HansenJTest {
    pub(super) statistic: f64,
    pub(super) p_value: f64,
    pub(super) df: usize,
}

/// Hansen J過剰識別検定（docs/spec/iv-spec.md 3.5節）。
/// `J = (Z'ê)'S⁻¹(Z'ê)`（`n`で割らない、`gmm.rs`のモジュールdoc
/// 「Hansen J過剰識別検定」参照。`ê`は最終推定に基づく残差、`S`は最終推定`beta`に
/// 実際に使った重み行列`s_used`）。自由度は`len(instruments) - len(x_endog)`
/// （`two_sls/sargan.rs`のSargan検定と同じ、`docs/spec/iv-spec.md`1.1節の`instruments`＝
/// 除外操作変数のみという定義に対応）。丁度識別（自由度0）では`None`
/// （`docs/spec/iv-spec.md`1.2節・3.5節）。
///
/// `s_used`は`beta`計算時（`gmm_point_estimate`内の`llt`、または`classical_s`
/// 自体が正定値`Z'Z`の正のスカラー倍）で既に反転成功済み・正定値性が保証された
/// 行列のため、ここでの特異性は理論上到達不能（`two_sls/cov.rs`の`xtx_inverse`と
/// 同じ防御的`Result`化）。計算は共有の`shared::gmm::hansen_j_test`で、IVは自由度を
/// ここで決めて渡す。
pub(super) fn hansen_j_test(
    input: &IvInput,
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    s_used: &Mat<f64>,
) -> Result<Option<HansenJTest>, IvError> {
    let q = input.k_instruments();
    let k_endog = input.k_endog();
    if q == k_endog {
        return Ok(None);
    }
    let df = q - k_endog;
    let test = gmm::hansen_j_test(z, residuals, s_used, df)?;
    Ok(Some(HansenJTest {
        statistic: test.statistic,
        p_value: test.p_value,
        df,
    }))
}

#[cfg(test)]
mod tests {
    use crate::iv::common::IvInput;
    use crate::iv::gmm::GmmEstimator;
    use crate::iv::gmm::GmmType;
    use crate::iv::gmm::WeightType;
    use crate::iv::gmm::test_support::*;
    use crate::linear::ols::CovType;
    use crate::linear::ols::CovType as OlsCovType;
    use faer::prelude::Solve;
    use faer::{Mat, Side};

    /// 丁度識別（`len(instruments) == len(x_endog)`）ではHansen J過剰識別検定の自由度が
    /// 0のため`None`になる（`docs/spec/iv-spec.md`1.2節・3.5節、`two_sls/sargan.rs`のSargan検定と
    /// 同じ扱い）。
    #[test]
    fn fit_sets_hansen_j_statistic_to_none_when_just_identified() {
        let estimator = GmmEstimator::fit(
            just_identified_input(),
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.hansen_j_statistic(), None);
        assert_eq!(estimator.hansen_j_p_value(), None);
    }

    /// 2-step efficient GMM（`TwoStep`）のHansen J統計量を、`GmmEstimator::fit`とは
    /// 独立に「W₀=(Z'Z)⁻¹で初期推定→残差→S=Σêᵢ²zᵢzᵢ'（=最終推定に使った重み行列）→
    /// 最終推定の残差でJ=(Z'ê)'S⁻¹(Z'ê)/nを計算」という同じ手順を再現した手計算オラクルと
    /// 数値照合する（`fit()`のdocコメント「Hansen J過剰識別検定」参照。
    /// `fit_computes_robust_weighted_estimate_matching_manual_formula`と同じデータ・
    /// 同じ`S`構築だが、点推定ではなくJ統計量を検証する点が異なる）。
    #[test]
    fn fit_computes_hansen_j_statistic_matching_manual_formula_with_two_step_gmm() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s = z_scaled.transpose() * &z_scaled;
        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let beta1 = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        let e1 = &y_mat - &x * &beta1;
        let zte1 = z.transpose() * &e1;
        let s_inv_zte1 = s.llt(Side::Lower).unwrap().solve(&zte1);
        // `J = (Z'ê)'S⁻¹(Z'ê)`（`n`で割らない、`fit()`冒頭のdocコメント
        // 「Hansen J過剰識別検定」参照。オラクルが本番コードと同じ`/n`を含むと
        // 独立検証にならないため、ここでは`n`で割らない形で組み立てる）。
        let expected_stat: f64 = (0..3)
            .map(|i| (*zte1.get(i, 0)) * (*s_inv_zte1.get(i, 0)))
            .sum();
        let expected_p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(expected_stat);

        assert!((estimator.hansen_j_statistic().unwrap() - expected_stat).abs() < 1e-8);
        assert!((estimator.hansen_j_p_value().unwrap() - expected_p_value).abs() < 1e-8);
    }

    /// 1-step GMM（`OneStep`）のHansen J統計量は、`weight_type`によらず
    /// `σ̂²₀・Z'Z`（`σ̂²₀`はstep-0＝最終残差の分散）を重み行列として計算する
    /// （`fit()`冒頭のdocコメント「Hansen J過剰識別検定」参照）。
    #[test]
    fn fit_computes_hansen_j_statistic_matching_manual_formula_with_one_step_gmm() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator =
            GmmEstimator::fit(input, GmmType::OneStep, true, CovType::Classical, 0.95).unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        // 1-step GMMのHansen J用の重みは`σ̂²₀・Z'Z`（`weight_type`に関わらず）。
        let sigma2_0: f64 = (0..n).map(|i| (*e0.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
        let s = Mat::from_fn(3, 3, |i, j| sigma2_0 * (*ztz.get(i, j)));
        let zte0 = z.transpose() * e0;
        let s_inv_zte0 = s.llt(Side::Lower).unwrap().solve(&zte0);
        let expected_stat: f64 = (0..3)
            .map(|i| (*zte0.get(i, 0)) * (*s_inv_zte0.get(i, 0)))
            .sum();
        let expected_p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(expected_stat);

        assert!((estimator.hansen_j_statistic().unwrap() - expected_stat).abs() < 1e-8);
        assert!((estimator.hansen_j_p_value().unwrap() - expected_p_value).abs() < 1e-8);
    }

    /// `weight_type=Classical`かつ`TwoStep`のHansen J統計量は、`two_sls/sargan.rs`の
    /// Sargan統計量と数値的に一致するはず（`fit()`冒頭のdocコメント「Hansen J過剰識別検定」
    /// 参照）。点推定側の`fit_matches_two_sls_point_estimate_when_weight_type_is_classical`
    /// と対になる不変条件チェック（同じデータセットを使う）。`GmmEstimator`単体の
    /// 手計算オラクルでは検出できない`σ̂²`スケーリング漏れを、
    /// 独立実装である`TwoSlsEstimator`との数値一致という形で検証する。
    #[test]
    fn fit_computes_hansen_j_statistic_matching_two_sls_sargan_when_weight_type_is_classical() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];

        let build_input = || {
            IvInput::from_columns(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let gmm = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let two_sls =
            crate::iv::two_sls::TwoSlsEstimator::fit(build_input(), OlsCovType::Classical, 0.95)
                .unwrap();

        let gmm_j = gmm.hansen_j_statistic().unwrap();
        let sargan = two_sls.sargan_statistic().unwrap();
        assert!(
            (gmm_j - sargan).abs() < 1e-8,
            "hansen_j={gmm_j}, sargan={sargan}"
        );
        let gmm_p = gmm.hansen_j_p_value().unwrap();
        let sargan_p = two_sls.sargan_p_value().unwrap();
        assert!(
            (gmm_p - sargan_p).abs() < 1e-8,
            "hansen_j_p={gmm_p}, sargan_p={sargan_p}"
        );
    }
}
