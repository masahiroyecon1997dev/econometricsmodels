//! property-basedテスト。`two_sls/fit/tests/proptests.rs`と同じケース生成器
//! （`common::proptest_support`）・許容誤差（`RTOL=1e-6`）・`prop_assume!`によるフルランク
//! 安全弁を使う。GMM固有の不変条件（`weight_type=Classical`と2SLSの一致、`S`のスカラー倍
//! に対する点推定の不変性、丁度識別での`W`への非依存）を検証する。

use crate::iv::common::IvError;
use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::common::proptest_support::IvCase;
use crate::iv::common::proptest_support::assert_approx_eq;
use crate::iv::common::proptest_support::iv_case_strategy;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::WeightType;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::linear::ols::CovType as OlsCovType;
use crate::shared::gmm_iteration::gmm_point_estimate;
use faer::Mat;
use proptest::prelude::*;
use std::collections::HashMap;

fn fit_gmm(
    input: IvInput,
    weight_type: WeightType,
    cov_type: OlsCovType,
) -> Result<GmmEstimator, IvError> {
    GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: weight_type,
        },
        true,
        cov_type,
        0.95,
    )
}

fn by_name(est: &GmmEstimator) -> HashMap<String, (f64, f64)> {
    est.param_names()
        .iter()
        .enumerate()
        .map(|(i, name)| {
            (
                name.clone(),
                (*est.params().get(i, 0), *est.std_errors().get(i, 0)),
            )
        })
        .collect()
}

/// `gmm_point_estimate`に渡す`(Z, X, y)`（`Z=[const, x_exog, z]`、`X=[const, x_exog,
/// x_endog]`）を`IvCase`から直接組み立てる。
fn design_matrices(case: &IvCase) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let n = case.nobs();
    let z_cols: Vec<&Vec<f64>> = case.x_exog.iter().chain(case.z.iter()).collect();
    let x_cols: Vec<&Vec<f64>> = case.x_exog.iter().chain(case.x_endog.iter()).collect();
    let z = Mat::from_fn(n, z_cols.len() + 1, |i, j| {
        if j == 0 { 1.0 } else { z_cols[j - 1][i] }
    });
    let x = Mat::from_fn(n, x_cols.len() + 1, |i, j| {
        if j == 0 { 1.0 } else { x_cols[j - 1][i] }
    });
    let y = Mat::from_fn(n, 1, |i, _| case.y[i]);
    (z, x, y)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `weight_type=Classical`のGMM点推定は2SLSの点推定と一致する（`W₁∝(Z'Z)⁻¹`）。
    #[test]
    fn unadjusted_gmm_matches_two_sls_point_estimate(
        case in iv_case_strategy(false),
    ) {
        let gmm = fit_gmm(case.input().unwrap(), WeightType::Classical, OlsCovType::Classical);
        let two_sls = TwoSlsEstimator::fit(case.input().unwrap(), OlsCovType::Classical, 0.95);
        prop_assume!(gmm.is_ok() && two_sls.is_ok());
        let (gmm, two_sls) = (gmm.unwrap(), two_sls.unwrap());
        prop_assert_eq!(gmm.param_names(), two_sls.param_names());
        for j in 0..gmm.params().nrows() {
            assert_approx_eq(*gmm.params().get(j, 0), *two_sls.params().get(j, 0), &format!("beta[{j}]"));
        }
    }

    /// 重み行列の`S`を正のスカラー倍しても、点推定`β̂(W)`は変わらない。
    #[test]
    fn point_estimate_is_invariant_to_positive_scaling_of_s(
        case in iv_case_strategy(false),
        c in 0.01f64..100.0,
    ) {
        let (z, x, y) = design_matrices(&case);
        let s = z.transpose() * &z;
        let s_scaled = Mat::from_fn(s.nrows(), s.ncols(), |i, j| c * *s.get(i, j));
        let unscaled = gmm_point_estimate(&z, &x, &y, &s);
        let scaled = gmm_point_estimate(&z, &x, &y, &s_scaled);
        prop_assume!(unscaled.is_ok() && scaled.is_ok());
        let (unscaled, scaled) = (unscaled.unwrap(), scaled.unwrap());
        for j in 0..unscaled.nrows() {
            assert_approx_eq(*scaled.get(j, 0), *unscaled.get(j, 0), &format!("beta[{j}]"));
        }
    }

    /// `x_exog`の列順序を入れ替えても、名前で対応付けた係数・SEは変わらない
    /// （`weight_type=Robust`の2-step GMM、`cov_type=Hc1`）。
    #[test]
    fn coefficients_and_se_are_invariant_to_x_exog_column_order(
        case in iv_case_strategy(false),
    ) {
        let base = fit_gmm(case.input().unwrap(), WeightType::Robust, OlsCovType::Hc1);
        let permuted = case
            .build_input(&case.shuffled_order(), &case.z, &case.y)
            .map(|input| fit_gmm(input, WeightType::Robust, OlsCovType::Hc1));
        prop_assume!(base.is_ok());
        prop_assume!(permuted.as_ref().is_ok_and(|r| r.is_ok()));
        let (base, permuted) = (by_name(&base.unwrap()), by_name(&permuted.unwrap().unwrap()));
        for (name, (beta, se)) in &base {
            let (p_beta, p_se) = permuted[name];
            assert_approx_eq(p_beta, *beta, &format!("beta[{name}]"));
            assert_approx_eq(p_se, *se, &format!("se[{name}]"));
        }
    }

    /// 丁度識別では、点推定は重み行列`W`（`weight_type`）に依存しない。
    #[test]
    fn just_identified_point_estimate_does_not_depend_on_weight_type(
        case in iv_case_strategy(true),
    ) {
        let classical = fit_gmm(case.input().unwrap(), WeightType::Classical, OlsCovType::Classical);
        let robust = fit_gmm(case.input().unwrap(), WeightType::Robust, OlsCovType::Classical);
        prop_assume!(classical.is_ok() && robust.is_ok());
        let (classical, robust) = (classical.unwrap(), robust.unwrap());
        for j in 0..classical.params().nrows() {
            assert_approx_eq(*robust.params().get(j, 0), *classical.params().get(j, 0), &format!("beta[{j}]"));
        }
    }
}
