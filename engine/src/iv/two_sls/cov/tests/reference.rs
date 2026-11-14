//! 手計算の公式との照合・整合性のテスト。

use super::assert_slices_close;
use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::iv::two_sls::test_support::*;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use crate::shared::group_codes::GroupCodes;
use faer::Mat;

/// HC0〜HC3の標準誤差を、行列積のショートカット（`x_scaled.transpose() * x_scaled`）を
/// 使わず素朴な三重ループ（外積`x_i x_i'`の直接積み上げ）で独立計算する
/// （`TwoSlsEstimator::fit`内部の`hc_cov_params`とは別経路の実装）。`weight(i)`は
/// 観測`i`の`e_i²`に掛ける係数（HC0: 1、HC1: `n/(n-k)`、HC2: `1/(1-h_ii)`、
/// HC3: `1/(1-h_ii)²`）。
fn manual_hc_std_errors_with_weight(
    x: &Mat<f64>,
    e: &Mat<f64>,
    n: usize,
    k: usize,
    weight: impl Fn(usize) -> f64,
) -> Vec<f64> {
    let xtx_inv = manual_xtx_inverse(x, k);
    let mut psi = Mat::<f64>::zeros(k, k);
    for i in 0..n {
        let scaled_e2 = weight(i) * (*e.get(i, 0)).powi(2);
        for a in 0..k {
            for b in 0..k {
                *psi.get_mut(a, b) += scaled_e2 * (*x.get(i, a)) * (*x.get(i, b));
            }
        }
    }
    let cov = &xtx_inv * &psi * &xtx_inv;
    (0..k).map(|j| (*cov.get(j, j)).sqrt()).collect()
}

/// レバレッジ `h_ii = x_i'(X'X)⁻¹x_i` を素朴な二重ループで独立計算する（HC2/HC3用）。
fn manual_leverage(x: &Mat<f64>, xtx_inv: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut h = 0.0;
            for a in 0..k {
                for b in 0..k {
                    h += (*x.get(i, a)) * (*xtx_inv.get(a, b)) * (*x.get(i, b));
                }
            }
            h
        })
        .collect()
}

fn manual_hc0_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
    manual_hc_std_errors_with_weight(x, e, n, k, |_| 1.0)
}

fn manual_hc1_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
    let correction = (n as f64) / ((n - k) as f64);
    manual_hc_std_errors_with_weight(x, e, n, k, move |_| correction)
}

fn manual_hc2_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
    let xtx_inv = manual_xtx_inverse(x, k);
    let leverage = manual_leverage(x, &xtx_inv, n, k);
    manual_hc_std_errors_with_weight(x, e, n, k, move |i| 1.0 / (1.0 - leverage[i]))
}

fn manual_hc3_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
    let xtx_inv = manual_xtx_inverse(x, k);
    let leverage = manual_leverage(x, &xtx_inv, n, k);
    manual_hc_std_errors_with_weight(x, e, n, k, move |i| 1.0 / (1.0 - leverage[i]).powi(2))
}

/// クラスターロバスト標準誤差を素朴なループ（クラスターごとの手動集約）で独立計算する。
fn manual_cluster_std_errors(
    x: &Mat<f64>,
    e: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &[String],
) -> Vec<f64> {
    let xtx_inv = manual_xtx_inverse(x, k);
    let mut group_sums: std::collections::BTreeMap<&str, Vec<f64>> =
        std::collections::BTreeMap::new();
    for (i, group) in groups.iter().enumerate().take(n) {
        let s_g = group_sums.entry(group.as_str()).or_insert(vec![0.0; k]);
        for (a, s_g_a) in s_g.iter_mut().enumerate() {
            *s_g_a += (*e.get(i, 0)) * (*x.get(i, a));
        }
    }
    let n_groups = group_sums.len();
    let mut s_hat = Mat::<f64>::zeros(k, k);
    for s_g in group_sums.values() {
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }
    let correction =
        (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
    let cov = &xtx_inv * &s_hat * &xtx_inv;
    (0..k)
        .map(|j| (correction * (*cov.get(j, j))).sqrt())
        .collect()
}

/// classicalの標準誤差が構造残差（`y - Xβ̂`、第二段階回帰自身の残差`y - X̂β̂`ではない）
/// を使って計算されていることを確認する（`two_sls.rs`のモジュールdoc「標準誤差・
/// 適合度統計量」参照。誤って第二段階の回帰（`y ~ X̂`）をそのままOLSとして扱ったSEを使っていないことの
/// 回帰テスト）。
#[test]
fn fit_computes_classical_std_errors_using_structural_residuals() {
    let estimator =
        TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
    let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
    let n = x.nrows();
    let k = x.ncols();

    let expected_se = manual_classical_std_errors(&x, &e, n, k);
    let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
    assert_slices_close(&actual_se, &expected_se);

    // `residuals()`が構造残差（`y - Xβ̂`）そのものであることを確認する
    // （第二段階回帰自身の残差`y - X̂β̂`ではない）。
    assert_eq!(estimator.residuals().nrows(), n);
    for i in 0..n {
        assert!((*estimator.residuals().get(i, 0) - *e.get(i, 0)).abs() < 1e-8);
    }

    // 構造残差は第二段階回帰自身の残差（`y - X̂β̂`）とは異なる値になるはず
    // （`x_endog`が操作変数に完全予測されない、内生性が残る非退化ケースのため）。
    let (x1, _x_endog, _z1, _z2, y) = nontrivial_x_exog_columns();
    let x_hat = second_stage_design(&estimator, &x1);
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
    let second_stage_residuals = &y_mat - &x_hat * estimator.params();
    let mut any_differs = false;
    for i in 0..n {
        if (*estimator.residuals().get(i, 0) - *second_stage_residuals.get(i, 0)).abs() > 1e-8 {
            any_differs = true;
        }
    }
    assert!(
        any_differs,
        "expected structural residuals to differ from the second-stage OLS's own residuals"
    );
}

#[test]
fn fit_computes_hc0_std_errors_matching_manual_sandwich_formula() {
    let estimator = TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc0, 0.95).unwrap();
    let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
    let n = x.nrows();
    let k = x.ncols();

    let expected_se = manual_hc0_std_errors(&x, &e, n, k);
    let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
    assert_slices_close(&actual_se, &expected_se);
    assert_eq!(estimator.cov_type(), &CovType::Hc0);
}

#[test]
fn fit_computes_hc1_std_errors_matching_manual_sandwich_formula() {
    let estimator = TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc1, 0.95).unwrap();
    let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
    let n = x.nrows();
    let k = x.ncols();

    let expected_se = manual_hc1_std_errors(&x, &e, n, k);
    let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
    assert_slices_close(&actual_se, &expected_se);
}

/// HC2はレバレッジ`h_ii`によるスケーリングを要する（`X̂`のみからレバレッジを計算する
/// 拡張、`docs/spec/iv-spec.md`3.1節参照）。このテストは本実装が意図した式（下記
/// `manual_hc2_std_errors`と同一の式）通りに計算されていることを固定する、
/// 数式レベルの細粒度回帰確認。真に独立した参照実装（R `ivreg`+
/// `sandwich::vcovHC(type="HC2"/"HC3")`）との数値一致は
/// `tests/iv/test_iv_crosscheck.py`で別途検証済み
/// （`linearmodels`は引き続きhc2/hc3非対応のため対象外）。
#[test]
fn fit_computes_hc2_std_errors_matching_manual_sandwich_formula() {
    let estimator = TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc2, 0.95).unwrap();
    let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
    let n = x.nrows();
    let k = x.ncols();

    let expected_se = manual_hc2_std_errors(&x, &e, n, k);
    let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
    assert_slices_close(&actual_se, &expected_se);
}

/// HC3も同様（`fit_computes_hc2_std_errors_matching_manual_sandwich_formula`のdoc
/// コメント参照）。
#[test]
fn fit_computes_hc3_std_errors_matching_manual_sandwich_formula() {
    let estimator = TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc3, 0.95).unwrap();
    let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
    let n = x.nrows();
    let k = x.ncols();

    let expected_se = manual_hc3_std_errors(&x, &e, n, k);
    let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
    assert_slices_close(&actual_se, &expected_se);
}

#[test]
fn fit_computes_cluster_std_errors_matching_manual_formula() {
    let (input, groups) = cluster_test_input_and_groups();
    let cov_type = CovType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&groups)),
    };
    let estimator = TwoSlsEstimator::fit(input, cov_type, 0.95).unwrap();

    let y = [5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
    let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_endog = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
    let n = y.len();
    let x_structural = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => x1[i],
        _ => x_endog[i],
    });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
    let e = &y_mat - &x_structural * estimator.params();
    let x_hat = second_stage_design(&estimator, &x1);
    let k = x_hat.ncols();

    let expected_se = manual_cluster_std_errors(&x_hat, &e, n, k, &groups);
    let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
    assert_slices_close(&actual_se, &expected_se);
}

/// 全`cov_type`で、`cov_params()`が対称で、対角の平方根が`std_errors()`と一致する
/// （`std_errors`は`cov_params`の対角から作るため、公開している行列が標準誤差の元そのもの
/// であることの確認）。次元は`(k, k)`。
#[test]
fn cov_params_is_symmetric_and_diagonal_matches_std_errors_for_every_cov_type() {
    let z1: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let z2: Vec<f64> = z1.iter().map(|&v| (v * 1.7).sin() * 5.0 + v).collect();
    let noise = [
        -3.0, 2.5, -1.0, 4.0, -5.0, 6.5, -2.0, 7.0, -8.0, 3.0, -9.5, 10.0, -4.0, 11.0, -12.5, 6.0,
    ];
    let x_endog: Vec<f64> = (0..16)
        .map(|i| z1[i] + 0.5 * z2[i] + noise[i] * (1.0 + 0.3 * z1[i]))
        .collect();
    let x_exog: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.9).cos() * 3.0).collect();
    let y: Vec<f64> = (0..16)
        .map(|i| 2.0 + 1.5 * x_endog[i] + 0.5 * x_exog[i] + noise[i] * 0.6)
        .collect();
    let labels: Vec<String> = (0..16).map(|i| format!("g{}", i / 4)).collect();

    let cov_types = [
        CovType::Classical,
        CovType::Hc0,
        CovType::Hc1,
        CovType::Hc2,
        CovType::Hc3,
        CovType::Hac {
            lags: Some(2),
            time_order: row_time_order(16),
        },
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labels)),
        },
    ];
    for cov_type in cov_types {
        let name = format!("{cov_type:?}");
        let input = IvInput::from_columns(
            &y,
            std::slice::from_ref(&x_exog),
            vec!["x_exog".to_string()],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = TwoSlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let cov = estimator.cov_params();
        let k = estimator.params().nrows();
        assert_eq!((cov.nrows(), cov.ncols()), (k, k), "{name}: shape");
        for i in 0..k {
            for j in 0..k {
                let scale = (*cov.get(i, i) * *cov.get(j, j)).sqrt();
                assert!(
                    (*cov.get(i, j) - *cov.get(j, i)).abs() <= 1e-12 * scale,
                    "{name}: cov_params not symmetric at ({i},{j})"
                );
            }
            let se = *estimator.std_errors().get(i, 0);
            assert!(
                (cov.get(i, i).sqrt() - se).abs() <= 1e-12 * se,
                "{name}: sqrt(cov_params[{i}][{i}]) != std_errors[{i}]"
            );
        }
    }
}
