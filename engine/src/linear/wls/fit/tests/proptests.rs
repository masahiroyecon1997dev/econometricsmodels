//! property-basedテスト。`ols/fit/tests/proptests.rs`と同型の設計
//! （`testing-policy.md`「property-basedテスト」参照）。
//!
//! プロパティの有効性検証（バグ注入→検出確認→元に戻す）は、新規プロパティ
//! `weights_all_one_matches_ols`（`weighted_fit_statistics`のAIC計算式にk+1の
//! バグを注入）・`weighted_residuals_sum_to_zero_when_intercept_included`
//! （`original_scale_fitted_and_residuals`の切片寄与を半分にするバグを注入）・
//! `coefficients_scale_linearly_with_y`（`OlsInput::from_columns_impl`の
//! y変換に定数オフセットのバグを注入）の3件で実施済み。残り2件
//! （`coefficients_and_se_are_invariant_to_column_order`・
//! `hc0_std_errors_are_at_most_hc1_std_errors`）は`ols/fit/tests/proptests.rs`側で検証済みの
//! 同一ロジック（`OlsEstimator::fit`）をWLSの重み付き変換経由で呼ぶのみのため
//! 個別のバグ注入は省略した。

use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::linear::wls::WlsEstimator;
use proptest::collection;
use proptest::prelude::*;

// OLSのMAX_K拡張と揃えた値。
const MAX_K: usize = 20;

/// `wls_case_strategy`が生成するタプル: `(n, k, y, x_cols, weights, keys)`。
type WlsCase = (usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<u64>);

/// `(n, k, y, x_cols, weights, keys)`を生成する共通ストラテジ。
///
/// `y`/`x_cols`の生成方針は`ols/fit/tests/proptests.rs`の`ols_case_strategy`と同じ
/// （`n=k+10..=60`のマージン、独立な連続一様分布）。`weights`は正値
/// （`0.1..10.0`）のみ生成する（`OlsInput::from_columns_weighted`が非正の重みを
/// 拒否するため）。
fn wls_case_strategy() -> impl Strategy<Value = WlsCase> {
    (1..=MAX_K).prop_flat_map(|k| {
        (k + 10..=60usize).prop_flat_map(move |n| {
            (
                Just(n),
                Just(k),
                collection::vec(-100.0f64..100.0, n),
                collection::vec(collection::vec(-100.0f64..100.0, n), k),
                collection::vec(0.1f64..10.0, n),
                collection::vec(any::<u64>(), k),
            )
        })
    })
}

fn x_names(k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("x{i}")).collect()
}

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

    /// 切片ありなら重み付き残差和は常に0（`Σ w_i・residuals_i ≈ 0`）。
    ///
    /// `ols/fit/tests/proptests.rs`の`residuals_sum_to_zero_when_intercept_included`の単純移植では
    /// **ない**点に注意: WLSの正規方程式は変換後データ（切片列も`sqrt(w_i)`倍）の
    /// `X'e=0`から従うため、元スケールに戻すと単純な残差和ゼロではなく重み付き和が
    /// ゼロになる（`Σ sqrt(w_i)・sqrt(w_i)・residuals_i = Σ w_i・residuals_i = 0`）。
    #[test]
    fn weighted_residuals_sum_to_zero_when_intercept_included(
        (n, k, y, x_cols, weights, _keys) in wls_case_strategy()
    ) {
        let names = x_names(k);
        let result = WlsEstimator::fit(
            &y, &x_cols, names, true, "y".to_string(),
            &weights, CovType::Classical, 0.95,
        );
        prop_assume!(result.is_ok());
        let wls = result.unwrap();

        let weighted_sum: f64 = wls.residuals().iter().zip(&weights).map(|(r, w)| w * r).sum();
        let scale = y.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
        prop_assert!(
            weighted_sum.abs() <= 1e-6 * scale * (n as f64),
            "weighted residual sum should be ~0, got {weighted_sum} (scale={scale}, n={n})"
        );
    }

    /// 重みが全て1のとき、WLSはOLSと（丸め誤差の範囲内で）完全一致する
    /// （`wls-spec.md`「sqrt(w)変換」の構造的保証、固定値テスト
    /// `fit_with_all_weights_one_matches_ols`のproperty-based版）。
    /// `r_squared`等は`WlsEstimator`側で再計算する専用ロジック
    /// （`weighted_fit_statistics`）を通るため、変換パイプライン
    /// （`OlsInput::from_columns_weighted`）だけでなくこの再計算ロジックも
    /// 検証対象に含む。
    #[test]
    fn weights_all_one_matches_ols(
        (n, k, y, x_cols, _weights, _keys) in wls_case_strategy()
    ) {
        let weights = vec![1.0; n];
        let names = x_names(k);

        let wls_result = WlsEstimator::fit(
            &y, &x_cols, names.clone(), true, "y".to_string(),
            &weights, CovType::Classical, 0.95,
        );
        let ols_input = OlsInput::from_columns(&y, &x_cols, names, true, "y".to_string()).unwrap();
        let ols_result = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
        prop_assume!(wls_result.is_ok() && ols_result.is_ok());
        let wls = wls_result.unwrap();
        let ols = ols_result.unwrap();

        for i in 0..=k {
            assert_approx_eq(
                *wls.estimator().params().get(i, 0),
                *ols.params().get(i, 0),
                &format!("param[{i}]"),
            );
            assert_approx_eq(
                *wls.estimator().std_errors().get(i, 0),
                *ols.std_errors().get(i, 0),
                &format!("std_error[{i}]"),
            );
        }
        assert_approx_eq(wls.r_squared(), ols.r_squared(), "r_squared");
        assert_approx_eq(wls.adj_r_squared(), ols.adj_r_squared(), "adj_r_squared");
        assert_approx_eq(wls.log_likelihood(), ols.log_likelihood(), "log_likelihood");
        assert_approx_eq(wls.aic(), ols.aic(), "aic");
        assert_approx_eq(wls.bic(), ols.bic(), "bic");
    }

    /// yをc倍すると、切片を含む全ての係数がc倍にスケールする
    /// （WLSも重み付き最小二乗解がyに関して線形なため、`ols/fit/tests/proptests.rs`の
    /// `coefficients_scale_linearly_with_y`と同じ理由で成り立つ）。
    #[test]
    fn coefficients_scale_linearly_with_y(
        (_n, k, y, x_cols, weights, _keys) in wls_case_strategy(),
        c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
    ) {
        let names = x_names(k);
        let result1 = WlsEstimator::fit(
            &y, &x_cols, names.clone(), true, "y".to_string(),
            &weights, CovType::Classical, 0.95,
        );
        prop_assume!(result1.is_ok());
        let wls1 = result1.unwrap();

        let y_scaled: Vec<f64> = y.iter().map(|v| v * c).collect();
        let result2 = WlsEstimator::fit(
            &y_scaled, &x_cols, names, true, "y".to_string(),
            &weights, CovType::Classical, 0.95,
        );
        prop_assume!(result2.is_ok());
        let wls2 = result2.unwrap();

        for i in 0..=k {
            let expected = c * *wls1.estimator().params().get(i, 0);
            let actual = *wls2.estimator().params().get(i, 0);
            assert_approx_eq(actual, expected, &format!("param[{i}] scaled by c={c}"));
        }
    }

    /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
    #[test]
    fn coefficients_and_se_are_invariant_to_column_order(
        (_n, k, y, x_cols, weights, keys) in wls_case_strategy()
            .prop_filter("need >=2 columns to permute", |(_, k, _, _, _, _)| *k >= 2)
    ) {
        let names = x_names(k);
        let result1 = WlsEstimator::fit(
            &y, &x_cols, names.clone(), true, "y".to_string(),
            &weights, CovType::Classical, 0.95,
        );
        prop_assume!(result1.is_ok());
        let wls1 = result1.unwrap();

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&i| keys[i]);
        let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
        let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

        let result2 = WlsEstimator::fit(
            &y, &permuted_x, permuted_names, true, "y".to_string(),
            &weights, CovType::Classical, 0.95,
        );
        prop_assume!(result2.is_ok());
        let wls2 = result2.unwrap();

        let names1 = wls1.estimator().input().param_names().to_vec();
        let names2 = wls2.estimator().input().param_names().to_vec();
        for (idx1, name) in names1.iter().enumerate() {
            let idx2 = names2.iter().position(|n| n == name)
                .expect("name should exist in permuted result");
            let p1 = *wls1.estimator().params().get(idx1, 0);
            let p2 = *wls2.estimator().params().get(idx2, 0);
            assert_approx_eq(p2, p1, &format!("param[{name}] under column permutation"));
            let se1 = *wls1.estimator().std_errors().get(idx1, 0);
            let se2 = *wls2.estimator().std_errors().get(idx2, 0);
            assert_approx_eq(se2, se1, &format!("std_error[{name}] under column permutation"));
        }
    }

    /// HC0の標準誤差は常にHC1以下（`HC1 = HC0 * n/(n-k)`で`n/(n-k) >= 1`のため）。
    #[test]
    fn hc0_std_errors_are_at_most_hc1_std_errors(
        (_n, k, y, x_cols, weights, _keys) in wls_case_strategy()
    ) {
        let names = x_names(k);
        let result1 = WlsEstimator::fit(
            &y, &x_cols, names.clone(), true, "y".to_string(),
            &weights, CovType::Hc0, 0.95,
        );
        prop_assume!(result1.is_ok());
        let wls_hc0 = result1.unwrap();

        let result2 = WlsEstimator::fit(
            &y, &x_cols, names, true, "y".to_string(),
            &weights, CovType::Hc1, 0.95,
        );
        prop_assume!(result2.is_ok());
        let wls_hc1 = result2.unwrap();

        for i in 0..=k {
            let se_hc0 = *wls_hc0.estimator().std_errors().get(i, 0);
            let se_hc1 = *wls_hc1.estimator().std_errors().get(i, 0);
            prop_assert!(
                se_hc0 <= se_hc1 + 1e-9,
                "HC0 se[{i}]={se_hc0} should be <= HC1 se[{i}]={se_hc1}"
            );
        }
    }
}
