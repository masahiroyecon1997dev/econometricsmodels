//! HAC共分散（ラグ・時間順序）のテスト。

use super::assert_slices_close;
use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::iv::two_sls::test_support::*;
use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use faer::Mat;

/// Newey-West HAC標準誤差を、`hac_cov_params`の`matmul`ベースの実装とは別経路
/// （素朴な二重ループでの外積の積み上げ）で独立計算する（`manual_hc_std_errors_with_weight`
/// と同じ方針）。`order`は`time_ordering`の出力（時系列順の行インデックス列）。
fn manual_hac_std_errors(
    x: &Mat<f64>,
    e: &Mat<f64>,
    n: usize,
    k: usize,
    lags: usize,
    order: &[usize],
) -> Vec<f64> {
    let xtx_inv = manual_xtx_inverse(x, k);
    let mut s_hat = Mat::<f64>::zeros(k, k);
    for &i in order {
        let ei = *e.get(i, 0);
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += ei * ei * (*x.get(i, a)) * (*x.get(i, b));
            }
        }
    }
    for l in 1..=lags {
        let weight = 1.0 - (l as f64) / ((lags + 1) as f64);
        for t in l..n {
            let i_t = order[t];
            let i_tl = order[t - l];
            let e_t = *e.get(i_t, 0);
            let e_tl = *e.get(i_tl, 0);
            for a in 0..k {
                for b in 0..k {
                    let term = weight
                        * e_t
                        * e_tl
                        * ((*x.get(i_t, a)) * (*x.get(i_tl, b))
                            + (*x.get(i_tl, a)) * (*x.get(i_t, b)));
                    *s_hat.get_mut(a, b) += term;
                }
            }
        }
    }
    let cov = &xtx_inv * &s_hat * &xtx_inv;
    (0..k).map(|j| (*cov.get(j, j)).sqrt()).collect()
}

/// `cov_type=Hac{lags: Some(0), ..}`は自己相関項（`l=1..=lags`のループ）が空になり、
/// `l=0`項（HC0のΨ̂と同形）のみが残るため、HC0と数値的に一致するはず
/// （`ols/cov_params/tests/hc_hac.rs`の`fit_hac_with_zero_lags_matches_hc0`と同じ内部整合性テスト）。
#[test]
fn fit_hac_with_zero_lags_matches_hc0() {
    let hac_estimator = TwoSlsEstimator::fit(
        nontrivial_x_exog_input(),
        CovType::Hac {
            lags: Some(0),
            time_order: row_time_order(8),
        },
        0.95,
    )
    .unwrap();
    let hc0_estimator =
        TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc0, 0.95).unwrap();

    for j in 0..hac_estimator.k() {
        assert!(
            (*hac_estimator.std_errors().get(j, 0) - *hc0_estimator.std_errors().get(j, 0)).abs()
                < 1e-8
        );
    }
}

/// `lags=Some(2)`（自己相関項が実際に加算されるケース、`lags=0`の
/// `fit_hac_with_zero_lags_matches_hc0`とは異なりHC0とは一致しないはず）を
/// `manual_hac_std_errors`（独立実装）と照合する。
#[test]
fn fit_computes_hac_std_errors_with_explicit_lags_matching_manual_formula() {
    let estimator = TwoSlsEstimator::fit(
        nontrivial_x_exog_input(),
        CovType::Hac {
            lags: Some(2),
            time_order: row_time_order(8),
        },
        0.95,
    )
    .unwrap();
    let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
    let n = x.nrows();
    let k = x.ncols();
    let order: Vec<usize> = (0..n).collect();

    let expected_se = manual_hac_std_errors(&x, &e, n, k, 2, &order);
    let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
    assert_slices_close(&actual_se, &expected_se);
}

/// `lags=None`（経験則自動計算 `L = floor(4*(n/100)^(2/9))`）が、`n=8`では`L=2`と
/// 一致するため、`lags=Some(2)`を明示指定した場合と数値的に一致するはず。
#[test]
fn fit_computes_hac_std_errors_with_auto_lags_matching_explicit_lags() {
    let auto_estimator = TwoSlsEstimator::fit(
        nontrivial_x_exog_input(),
        CovType::Hac {
            lags: None,
            time_order: row_time_order(8),
        },
        0.95,
    )
    .unwrap();
    let explicit_estimator = TwoSlsEstimator::fit(
        nontrivial_x_exog_input(),
        CovType::Hac {
            lags: Some(2),
            time_order: row_time_order(8),
        },
        0.95,
    )
    .unwrap();

    for j in 0..auto_estimator.k() {
        assert!(
            (*auto_estimator.std_errors().get(j, 0) - *explicit_estimator.std_errors().get(j, 0))
                .abs()
                < 1e-8
        );
    }
}

/// `hac_lags_used()`: `lags=None`なら経験則（`n=8`→2）、明示指定ならその値、Hac以外は
/// `None`。第一段階のOLSも同じ`cov_type`で推定されるため同じ値になる。
#[test]
fn hac_lags_used_reflects_resolved_lags() {
    let fit = |cov_type: CovType| {
        TwoSlsEstimator::fit(nontrivial_x_exog_input(), cov_type, 0.95).unwrap()
    };

    let auto = fit(CovType::Hac {
        lags: None,
        time_order: row_time_order(8),
    });
    assert_eq!(auto.hac_lags_used(), Some(2));
    for (_, first_stage) in auto.first_stage_estimators() {
        assert_eq!(first_stage.hac_lags_used(), Some(2));
    }

    let explicit = fit(CovType::Hac {
        lags: Some(3),
        time_order: row_time_order(8),
    });
    assert_eq!(explicit.hac_lags_used(), Some(3));

    assert_eq!(fit(CovType::Classical).hac_lags_used(), None);
    assert_eq!(fit(CovType::Hc1).hac_lags_used(), None);
}

/// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
/// ラグ付き自己共分散を計算することを確認する（`ols/cov_params/tests/hc_hac.rs`の
/// `fit_computes_hac_std_errors_respecting_time_order`と同じ検証方針）。データは
/// `nontrivial_x_exog_columns()`と同一の内容を、時系列順の逆転を含む順序でシャッフル
/// して与える。
#[test]
fn fit_computes_hac_std_errors_respecting_time_order() {
    let (x1, x_endog, z1, z2, y) = nontrivial_x_exog_columns();
    // 元の時系列順=[0..n)を、時系列順の逆転を含む順序でシャッフルする。
    let shuffle: Vec<usize> = vec![3, 1, 6, 0, 5, 2, 7, 4];
    assert_eq!(shuffle.len(), y.len());
    let shuffled_time: Vec<f64> = shuffle.iter().map(|&i| i as f64).collect();
    let shuffled_y: Vec<f64> = shuffle.iter().map(|&i| y[i]).collect();
    let shuffled_x1: Vec<f64> = shuffle.iter().map(|&i| x1[i]).collect();
    let shuffled_x_endog: Vec<f64> = shuffle.iter().map(|&i| x_endog[i]).collect();
    let shuffled_z1: Vec<f64> = shuffle.iter().map(|&i| z1[i]).collect();
    let shuffled_z2: Vec<f64> = shuffle.iter().map(|&i| z2[i]).collect();

    let shuffled_input = IvInput::from_columns(
        &shuffled_y,
        &[shuffled_x1],
        vec!["x1".to_string()],
        &[shuffled_x_endog],
        vec!["endog1".to_string()],
        &[shuffled_z1, shuffled_z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let shuffled_estimator = TwoSlsEstimator::fit(
        shuffled_input,
        CovType::Hac {
            lags: Some(2),
            time_order: shuffled_time,
        },
        0.95,
    )
    .unwrap();

    let unshuffled_estimator = TwoSlsEstimator::fit(
        nontrivial_x_exog_input(),
        CovType::Hac {
            lags: Some(2),
            time_order: row_time_order(8),
        },
        0.95,
    )
    .unwrap();

    for j in 0..shuffled_estimator.k() {
        assert!(
            (*shuffled_estimator.std_errors().get(j, 0)
                - *unshuffled_estimator.std_errors().get(j, 0))
            .abs()
                < 1e-8
        );
    }
}
