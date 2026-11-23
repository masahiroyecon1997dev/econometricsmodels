//! property-basedテスト。`ols/fit/tests/proptests.rs`と同型の設計（配置・許容誤差`RTOL=1e-6`・
//! `prop_assume!`によるフルランク安全弁）だが、IV固有の不変条件（操作変数の張る空間のみへの
//! 依存・丁度識別の閉形式解）を検証する。ケース生成は`common::proptest_support`。

use crate::iv::common::IvEstimate;
use crate::iv::common::proptest_support::assert_approx_eq;
use crate::iv::common::proptest_support::iv_case_strategy;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::linear::ols::CovType;
use proptest::prelude::*;
use std::collections::HashMap;

fn by_name(est: &TwoSlsEstimator) -> HashMap<String, (f64, f64)> {
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

/// 部分ピボット付きガウス消去で`a * x = b`（`a`は`m x m`）を解く。`gmm_point_estimate`等の
/// 実装コード（Cholesky）とは独立な経路で閉形式解のオラクルを作るための自前実装。
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let m = b.len();
    for col in 0..m {
        let pivot =
            (col..m).max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())?;
        if a[pivot][col].abs() < 1e-10 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..m {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col].clone();
            for (target, src) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *target -= f * src;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; m];
    for row in (0..m).rev() {
        let tail: f64 = ((row + 1)..m).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    Some(x)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `x_exog`の列順序を入れ替えても、名前で対応付けた係数・SEは変わらない。
    #[test]
    fn coefficients_and_se_are_invariant_to_x_exog_column_order(
        case in iv_case_strategy(false),
        use_hc1 in any::<bool>(),
    ) {
        let cov = || if use_hc1 { CovType::Hc1 } else { CovType::Classical };
        let base = TwoSlsEstimator::fit(case.input().unwrap(), cov(), 0.95);
        let permuted = case
            .build_input(&case.shuffled_order(), &case.z, &case.y)
            .map(|input| TwoSlsEstimator::fit(input, cov(), 0.95));
        prop_assume!(base.is_ok());
        prop_assume!(permuted.as_ref().is_ok_and(|r| r.is_ok()));
        let (base, permuted) = (by_name(&base.unwrap()), by_name(&permuted.unwrap().unwrap()));
        for (name, (beta, se)) in &base {
            let (p_beta, p_se) = permuted[name];
            assert_approx_eq(p_beta, *beta, &format!("beta[{name}]"));
            assert_approx_eq(p_se, *se, &format!("se[{name}]"));
        }
    }

    /// 操作変数を可逆な線形変換（単位上三角行列）で置き換えても、2SLSの点推定・SEは
    /// 変わらない（第一段階の射影は操作変数の張る空間にのみ依存するため）。
    #[test]
    fn estimates_are_invariant_to_invertible_instrument_transform(
        case in iv_case_strategy(false),
    ) {
        let base = TwoSlsEstimator::fit(case.input().unwrap(), CovType::Classical, 0.95);
        let mixed_z = case.mixed_instruments();
        let mixed = case
            .build_input(&case.identity_order(), &mixed_z, &case.y)
            .map(|input| TwoSlsEstimator::fit(input, CovType::Classical, 0.95));
        prop_assume!(base.is_ok());
        prop_assume!(mixed.as_ref().is_ok_and(|r| r.is_ok()));
        let (base, mixed) = (base.unwrap(), mixed.unwrap().unwrap());
        for j in 0..base.params().nrows() {
            assert_approx_eq(*mixed.params().get(j, 0), *base.params().get(j, 0), &format!("beta[{j}]"));
            assert_approx_eq(*mixed.std_errors().get(j, 0), *base.std_errors().get(j, 0), &format!("se[{j}]"));
        }
    }

    /// 丁度識別（`len(z) == len(x_endog)`）では、2SLSの点推定は閉形式の操作変数推定量
    /// `(Z'X)⁻¹Z'y`（`Z=[const, x_exog, z]`、`X=[const, x_exog, x_endog]`）と一致する。
    #[test]
    fn just_identified_matches_closed_form_iv_estimator(
        case in iv_case_strategy(true),
    ) {
        let est = TwoSlsEstimator::fit(case.input().unwrap(), CovType::Classical, 0.95);
        prop_assume!(est.is_ok());
        let est = est.unwrap();

        let n = case.nobs();
        let ones = vec![1.0; n];
        let regressors: Vec<&Vec<f64>> = std::iter::once(&ones)
            .chain(case.x_exog.iter())
            .chain(case.x_endog.iter())
            .collect();
        let instruments: Vec<&Vec<f64>> = std::iter::once(&ones)
            .chain(case.x_exog.iter())
            .chain(case.z.iter())
            .collect();
        let dot = |a: &Vec<f64>, b: &Vec<f64>| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let zx: Vec<Vec<f64>> = instruments
            .iter()
            .map(|zc| regressors.iter().map(|xc| dot(zc, xc)).collect())
            .collect();
        let zy: Vec<f64> = instruments.iter().map(|zc| dot(zc, &case.y)).collect();
        let solved = solve_linear_system(zx, zy);
        prop_assume!(solved.is_some());
        for (j, expected) in solved.unwrap().iter().enumerate() {
            assert_approx_eq(*est.params().get(j, 0), *expected, &format!("beta[{j}]"));
        }
    }

    /// `y`を定数倍すると係数も同じ定数倍になる（SEは`|c|`倍）。
    #[test]
    fn coefficients_scale_linearly_with_y(
        case in iv_case_strategy(false),
        c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
    ) {
        let scaled_y: Vec<f64> = case.y.iter().map(|v| c * v).collect();
        let base = TwoSlsEstimator::fit(case.input().unwrap(), CovType::Classical, 0.95);
        let scaled = case
            .build_input(&case.identity_order(), &case.z, &scaled_y)
            .map(|input| TwoSlsEstimator::fit(input, CovType::Classical, 0.95));
        prop_assume!(base.is_ok());
        prop_assume!(scaled.as_ref().is_ok_and(|r| r.is_ok()));
        let (base, scaled) = (base.unwrap(), scaled.unwrap().unwrap());
        for j in 0..base.params().nrows() {
            assert_approx_eq(*scaled.params().get(j, 0), c * *base.params().get(j, 0), &format!("beta[{j}]"));
            assert_approx_eq(*scaled.std_errors().get(j, 0), c.abs() * *base.std_errors().get(j, 0), &format!("se[{j}]"));
        }
    }
}
