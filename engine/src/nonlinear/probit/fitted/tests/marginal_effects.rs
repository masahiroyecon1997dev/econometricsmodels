//! `marginal_effects`のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MarginalEffectsAt;
use crate::nonlinear::common::MleError;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::probit::ProbitEstimator;
use crate::nonlinear::probit::ProbitInput;
use crate::nonlinear::probit::test_support::*;
use crate::shared::error::CommonError;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// 切片のみモデル（k=1、`k_constant=1`）は、限界効果の出力対象となる説明変数が
/// 存在しない（定数項は出力から除外するため）。`marginal_effects`が空の結果を
/// 返す（パニックしない）ことを確認する境界ケース。
#[test]
fn marginal_effects_returns_empty_result_for_intercept_only_model() {
    let estimator = ProbitEstimator::fit(
        intercept_only_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let effects = estimator
        .marginal_effects(MarginalEffectsAt::Overall, 0.95)
        .unwrap();
    assert!(effects.param_names().is_empty());
    assert!(effects.effect().is_empty());
}

/// `marginal_effects(at="overall")`の`effect`を、実装の内部ヘルパー（`overall_w_and_s`/
/// `dydx_and_jacobian`）とは別に、定義式`dy/dx_j = (1/n)Σᵢφ(zᵢ)θⱼ`を`Normal::standard()`
/// から直接計算する式で独立に再計算し、突き合わせる（`LogitEstimator`の対応する
/// テストと同じ技法。標準誤差は`dydx_j`自体をfit済みパラメータの周りで数値微分した
/// ヤコビアンで独立に求め、解析的な標準誤差と突き合わせる）。
#[test]
fn marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_method_se() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    let k = 3;
    let n = 4;
    let x = estimator.input().x();

    let effects = estimator
        .marginal_effects(MarginalEffectsAt::Overall, 0.95)
        .unwrap();
    assert_eq!(effects.param_names(), ["x1".to_string(), "x2".to_string()]);

    // effectの独立再計算（`Normal::standard().pdf`から直接、`overall_w_and_s`とは別の式）
    let normal = Normal::standard();
    let dydx_j = |params: &[f64], j: usize| -> f64 {
        (0..n)
            .map(|i| {
                let z: f64 = (0..k).map(|m| *x.get(i, m) * params[m]).sum();
                normal.pdf(z)
            })
            .sum::<f64>()
            / (n as f64)
            * params[j]
    };
    let params = estimator.params();
    for (idx, j) in (1..k).enumerate() {
        assert!((effects.effect()[idx] - dydx_j(params, j)).abs() < 1e-9);
    }

    // 標準誤差の独立検証: `dydx_j`をfit済みパラメータの周りで数値微分して
    // ヤコビアン行（j行目）を求め、`cov_params`との二次形式で分散を計算する。
    let h = 1e-6;
    for (idx, j) in (1..k).enumerate() {
        let mut jac_row = vec![0.0; k];
        for m in 0..k {
            let mut plus = params.to_vec();
            plus[m] += h;
            let mut minus = params.to_vec();
            minus[m] -= h;
            jac_row[m] = (dydx_j(&plus, j) - dydx_j(&minus, j)) / (2.0 * h);
        }
        let mut var_j = 0.0;
        for a in 0..k {
            for b in 0..k {
                var_j += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
            }
        }
        let expected_se = var_j.sqrt();
        assert!(
            (effects.std_errors()[idx] - expected_se).abs() < 1e-6,
            "idx={idx}, actual={}, expected={expected_se}",
            effects.std_errors()[idx]
        );
    }

    // z値・p値・信頼区間の内部整合性（既存のSE系テストと同じ検算パターン）
    let z_crit = normal.inverse_cdf(0.975);
    for idx in 0..2 {
        let se = effects.std_errors()[idx];
        assert!((effects.test_stats()[idx] - effects.effect()[idx] / se).abs() < 1e-9);
        let expected_p = 2.0 * (1.0 - normal.cdf(effects.test_stats()[idx].abs()));
        assert!((effects.p_values()[idx] - expected_p).abs() < 1e-9);
        assert!(
            (effects.conf_upper()[idx] - effects.conf_lower()[idx] - 2.0 * z_crit * se).abs()
                < 1e-9
        );
    }
}

/// `at="mean"`は`at="overall"`と異なる代表点（標本平均）で評価するため、一般には
/// 異なる値になる。実装がこの違いを正しく反映していること（`at`の分岐が機能して
/// いること）を確認する。`effect`を`column_means`から独立に再計算した値とも突き合わせる。
#[test]
fn marginal_effects_at_mean_differs_from_overall_and_matches_independent_recomputation() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let overall = estimator
        .marginal_effects(MarginalEffectsAt::Overall, 0.95)
        .unwrap();
    let at_mean = estimator
        .marginal_effects(MarginalEffectsAt::Mean, 0.95)
        .unwrap();

    assert!((overall.effect()[0] - at_mean.effect()[0]).abs() > 1e-9);

    // 独立再計算: x̄=[1, 25, 1]（定数項1、x1の平均25、x2の平均1）でφ(z̄)を評価
    let params = estimator.params();
    let x_bar = [1.0, 25.0, 1.0];
    let z_bar: f64 = (0..3).map(|m| x_bar[m] * params[m]).sum();
    let w = Normal::standard().pdf(z_bar);
    for (idx, j) in (1..3).enumerate() {
        assert!((at_mean.effect()[idx] - w * params[j]).abs() < 1e-9);
    }
}

/// `at="median"`が`at="mean"`/`at="overall"`と異なる代表点で評価されること
/// （非対称なデータセットで平均・中央値が異なる値になるよう構成）、および
/// `effect`を`column_medians`から独立に再計算した値と突き合わせる。
#[test]
fn marginal_effects_at_median_differs_from_mean_and_overall_and_matches_independent_recomputation()
{
    // x1: 平均40（=200/5）・中央値30（sorted: 10,20,30,40,100）と非対称にずらす
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
    let x_columns = vec![
        vec![10.0, 20.0, 30.0, 40.0, 100.0],
        vec![-5.0, 2.0, 8.0, -1.0, 50.0],
    ];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let overall = estimator
        .marginal_effects(MarginalEffectsAt::Overall, 0.95)
        .unwrap();
    let at_mean = estimator
        .marginal_effects(MarginalEffectsAt::Mean, 0.95)
        .unwrap();
    let at_median = estimator
        .marginal_effects(MarginalEffectsAt::Median, 0.95)
        .unwrap();

    assert!((at_median.effect()[0] - at_mean.effect()[0]).abs() > 1e-9);
    assert!((at_median.effect()[0] - overall.effect()[0]).abs() > 1e-9);

    // 独立再計算: x̄=[1, 30, 2]（定数項1、x1の中央値30、x2の中央値2）でφ(z̄)を評価
    let params = estimator.params();
    let x_bar = [1.0, 30.0, 2.0];
    let z_bar: f64 = (0..3).map(|m| x_bar[m] * params[m]).sum();
    let w = Normal::standard().pdf(z_bar);
    for (idx, j) in (1..3).enumerate() {
        assert!((at_median.effect()[idx] - w * params[j]).abs() < 1e-9);
    }
}

#[test]
fn marginal_effects_returns_invalid_confidence_level_error_out_of_range() {
    let estimator = ProbitEstimator::fit(
        intercept_only_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let result = estimator.marginal_effects(MarginalEffectsAt::Overall, 1.5);
    assert_eq!(
        result.unwrap_err(),
        MleError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5
        })
    );
}
