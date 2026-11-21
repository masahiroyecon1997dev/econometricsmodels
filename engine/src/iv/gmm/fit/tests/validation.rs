//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use crate::iv::common::IvError;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::WeightType;
use crate::iv::gmm::test_support::*;
use crate::linear::ols::CovType;
use crate::shared::error::CommonError;
use crate::shared::error::MatrixKind;

#[test]
fn fit_returns_perfect_fit_when_the_structural_residuals_are_zero() {
    // 構造式が厳密に成り立つと、初期推定の残差が全て0で重み行列が0になる。後段の逆行列が
    // 特異になる前に、原因の分かる完全フィットとして止める。
    let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_endog = vec![1.2, 1.9, 3.3, 3.8, 5.1];
    let y: Vec<f64> = x_endog.iter().map(|&x| 1.0 + 2.0 * x).collect();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &[x_endog],
        vec!["x_endog".to_string()],
        &[z],
        vec!["z".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Robust,
        },
        true,
        CovType::Classical,
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::PerfectFit)
    );
}

#[test]
fn fit_returns_outcome_constant_for_a_constant_outcome() {
    let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_endog = vec![1.2, 1.9, 3.3, 3.8, 5.1];
    let input = IvInput::from_columns(
        &[3.0; 5],
        &[],
        vec![],
        &[x_endog],
        vec!["x_endog".to_string()],
        &[z],
        vec!["z".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    );

    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::OutcomeConstant)
    );
}

#[test]
fn fit_returns_insufficient_instruments_error_when_under_identified() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_endog = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![1.0, 1.0, 2.0, 2.0, 3.0]];
    let instruments = vec![vec![2.0, 1.0, 4.0, 3.0, 6.0]];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &x_endog,
        vec!["endog1".to_string(), "endog2".to_string()],
        &instruments,
        vec!["z1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::InsufficientInstruments {
            n_instruments: 1,
            n_endog: 2,
        }
    );
}

/// 操作変数が完全な線形従属（`z2 = 2*z1`）だと`Z'Z`が特異になり
/// `CommonError::NearSingularMatrix { matrix: GmmWeight }`（`MATRIX_NEAR_SINGULAR`）
/// になることを確認する（`.claude/rules/testing-policy.md`「テストの3系統」の
/// `ComputationError`パス）。
#[test]
fn fit_returns_computation_error_when_instruments_are_perfectly_collinear() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_endog = vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0];
    let z1 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
    let z2: Vec<f64> = z1.iter().map(|v| 2.0 * v).collect();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &[z1, z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight
        })
    );
}

/// `confidence_level`が`(0, 1)`の範囲外なら`InvalidConfidenceLevel`
/// （`OlsEstimator::fit`/`TwoSlsEstimator::fit`と同じ規約）。
#[test]
fn fit_returns_invalid_confidence_level_error_out_of_range() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    for invalid in [0.0_f64, 1.0, -0.1, 1.5] {
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

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Classical,
            invalid,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: invalid
            }),
            "confidence_level={invalid}"
        );
    }
}

/// 観測数`n`が`k`（構造方程式の係数の数）以下なら`InsufficientObservations`
/// （`n<=k`のままだと`df_resid=n-k`による除算
/// 等がNaN/Infinityを静かに生成しうる、`gmm_hc_omega`のdocコメント参照）。
#[test]
fn fit_returns_insufficient_observations_error_when_n_is_at_most_k() {
    // n=2, k=2（const + endog1）: 丁度識別だが観測数が係数の数と等しい境界。
    let y = vec![1.0, 2.0];
    let x_endog = vec![2.0, 4.0];
    let z = vec![1.0, 3.0];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        std::slice::from_ref(&z),
        vec!["z1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
    );
}

/// 構造方程式は推定できる（`n > k`）が全操作変数の数`l`以下の観測数なら、
/// `InsufficientObservationsForFirstStage`で拒否する（`n < l`では`Z'Z`が特異、
/// `n = l`では`W₀ = (Z'Z)⁻¹`のGMMがOLSへ縮退する）。`2sls`と同じ検証を共有する。
#[test]
fn fit_returns_insufficient_observations_for_first_stage_when_n_is_at_most_l() {
    // n=3、k=2（切片+endog1）、l=3（切片+z1+z2）。
    let y = vec![1.0, 2.0, 4.0];
    let x_endog = vec![1.0, 2.5, 2.0];
    let z = vec![vec![1.0, 2.0, 4.0], vec![2.0, 1.0, 5.0]];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &z,
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::InsufficientObservationsForFirstStage { n: 3, l: 3 }
    );
}

/// `n < l`（`k < n`）でも同じ専用エラーで拒否する（検証が無ければ`Z'Z`が特異になり
/// `NearSingularMatrix`の`ComputationError`になる経路）。
#[test]
fn fit_returns_insufficient_observations_for_first_stage_when_n_is_below_l() {
    // n=3、k=2（切片+endog1）、l=4（切片+z1..z3）。
    let y = vec![1.0, 2.0, 4.0];
    let x_endog = vec![1.0, 2.5, 2.0];
    let z = vec![
        vec![1.0, 2.0, 4.0],
        vec![2.0, 1.0, 5.0],
        vec![0.5, 3.0, 1.0],
    ];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &z,
        vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Classical,
        },
        true,
        CovType::Classical,
        0.95,
    );
    assert_eq!(
        result.unwrap_err(),
        IvError::InsufficientObservationsForFirstStage { n: 3, l: 4 }
    );
}
