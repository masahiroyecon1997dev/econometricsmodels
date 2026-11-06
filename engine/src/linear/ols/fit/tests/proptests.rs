//! property-basedテスト。他のテーマ（固定シナリオでの値の一致確認）とは別に、
//! OLSが満たすべき不変条件をランダムなデータで検証する（`testing-policy.md`
//! 「property-basedテスト」参照）。

use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use proptest::collection;
use proptest::prelude::*;

const MAX_K: usize = 20;

/// `(n, k, y, x_cols, keys)`を生成する共通ストラテジ。
///
/// `n = k+10..=60`という十分なマージンを取り、値は独立な連続一様分布
/// （`-100.0..100.0`）からサンプリングする。この条件下では生成される設計行列は
/// 実務上ほぼ確実にフルランクになるため、各プロパティ側では追加の制約は課さず、
/// `prop_assume!(result.is_ok())`で非フルランクになるレアケース（丸め誤差起因の
/// 境界事例等）のみを除外する（「ランダムに生成する設計行列は
/// SingularMatrixにならない範囲に制約する」という方針に対応）。
/// `MAX_K=20`（旧4）はbenchmarkの高次元シナリオ`many_regressors`と揃えた値
/// （列数依存バグ・高kでの数値的挙動の検証）。
/// `k`が最大でも`n-k>=10`のマージンは保たれる。
///
/// `keys`は列順序入れ替えテスト専用の補助データ（他のプロパティでは未使用）。
fn ols_case_strategy() -> impl Strategy<Value = (usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<u64>)> {
    (1..=MAX_K).prop_flat_map(|k| {
        (k + 10..=60usize).prop_flat_map(move |n| {
            (
                Just(n),
                Just(k),
                collection::vec(-100.0f64..100.0, n),
                collection::vec(collection::vec(-100.0f64..100.0, n), k),
                collection::vec(any::<u64>(), k),
            )
        })
    })
}

fn x_names(k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("x{i}")).collect()
}

/// 相対誤差ベースの近似比較（絶対誤差フロア込み）。乱数生成で値のスケールが
/// 揃わないため、固定フィクスチャ比較（`testing-policy.md`のRTOL=1e-8）より
/// 緩めた閾値にしている（col_piv_qrの数値誤差の蓄積を考慮）。
fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
    let tol = 1e-6 * expected.abs().max(1.0);
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{msg}: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// 定数項ありOLSでの残差和は常に0（正規方程式`X'e=0`のうち切片列に対応する
    /// 行から従う）。
    #[test]
    fn residuals_sum_to_zero_when_intercept_included(
        (n, k, y, x_cols, _keys) in ols_case_strategy()
    ) {
        let input = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);
        prop_assume!(result.is_ok());
        let est = result.unwrap();

        let sum: f64 = (0..n).map(|i| *est.residuals().get(i, 0)).sum();
        let scale = y.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
        prop_assert!(
            sum.abs() <= 1e-6 * scale * (n as f64),
            "residual sum should be ~0, got {sum} (scale={scale}, n={n})"
        );
    }

    /// yをc倍すると、切片を含む全ての係数がc倍にスケールする
    /// （OLSはyに関して線形なため、切片も例外ではない）。
    #[test]
    fn coefficients_scale_linearly_with_y(
        (_n, k, y, x_cols, _keys) in ols_case_strategy(),
        c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
    ) {
        let input1 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
        let result1 = OlsEstimator::fit(input1, CovType::Classical, 0.95);
        prop_assume!(result1.is_ok());
        let est1 = result1.unwrap();

        let y_scaled: Vec<f64> = y.iter().map(|v| v * c).collect();
        let input2 =
            OlsInput::from_columns(&y_scaled, &x_cols, x_names(k), true, "y".to_string()).unwrap();
        let result2 = OlsEstimator::fit(input2, CovType::Classical, 0.95);
        prop_assume!(result2.is_ok());
        let est2 = result2.unwrap();

        for i in 0..=k {
            let expected = c * *est1.params().get(i, 0);
            let actual = *est2.params().get(i, 0);
            assert_approx_eq(actual, expected, &format!("param[{i}] scaled by c={c}"));
        }
    }

    /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
    #[test]
    fn coefficients_and_se_are_invariant_to_column_order(
        (_n, k, y, x_cols, keys) in ols_case_strategy()
            .prop_filter("need >=2 columns to permute", |(_, k, _, _, _)| *k >= 2)
    ) {
        let names = x_names(k);
        let input1 =
            OlsInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
        let result1 = OlsEstimator::fit(input1, CovType::Classical, 0.95);
        prop_assume!(result1.is_ok());
        let est1 = result1.unwrap();

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&i| keys[i]);
        let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
        let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

        let input2 =
            OlsInput::from_columns(&y, &permuted_x, permuted_names, true, "y".to_string()).unwrap();
        let result2 = OlsEstimator::fit(input2, CovType::Classical, 0.95);
        prop_assume!(result2.is_ok());
        let est2 = result2.unwrap();

        let names1 = est1.input().param_names().to_vec();
        let names2 = est2.input().param_names().to_vec();
        for (idx1, name) in names1.iter().enumerate() {
            let idx2 = names2
                .iter()
                .position(|n| n == name)
                .expect("name should exist in permuted result");
            let p1 = *est1.params().get(idx1, 0);
            let p2 = *est2.params().get(idx2, 0);
            assert_approx_eq(p2, p1, &format!("param[{name}] under column permutation"));
            let se1 = *est1.std_errors().get(idx1, 0);
            let se2 = *est2.std_errors().get(idx2, 0);
            assert_approx_eq(se2, se1, &format!("std_error[{name}] under column permutation"));
        }
    }

    /// HC0の標準誤差は常にHC1以下（`HC1 = HC0 * n/(n-k)`で`n/(n-k) >= 1`のため）。
    #[test]
    fn hc0_std_errors_are_at_most_hc1_std_errors(
        (_n, k, y, x_cols, _keys) in ols_case_strategy()
    ) {
        let input1 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
        let result1 = OlsEstimator::fit(input1, CovType::Hc0, 0.95);
        prop_assume!(result1.is_ok());
        let est_hc0 = result1.unwrap();

        let input2 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
        let result2 = OlsEstimator::fit(input2, CovType::Hc1, 0.95);
        prop_assume!(result2.is_ok());
        let est_hc1 = result2.unwrap();

        for i in 0..=k {
            let se_hc0 = *est_hc0.std_errors().get(i, 0);
            let se_hc1 = *est_hc1.std_errors().get(i, 0);
            // 浮動小数点の丸め誤差の余地として小さな絶対許容誤差を加える。
            prop_assert!(
                se_hc0 <= se_hc1 + 1e-9,
                "HC0 se[{i}]={se_hc0} should be <= HC1 se[{i}]={se_hc1}"
            );
        }
    }
}
