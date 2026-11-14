//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleError;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::probit::ProbitEstimator;
use crate::nonlinear::probit::ProbitInput;
use crate::nonlinear::probit::test_support::*;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

#[test]
fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Cluster { groups: None },
            confidence_level: 0.95,
        },
    );

    assert_eq!(
        result.unwrap_err(),
        MleError::Common(CommonError::MissingClusterColumn {
            setting: ClusterSetting::CovType,
        })
    );
}

#[test]
fn fit_returns_insufficient_clusters_error_when_only_one_group() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let groups = vec!["a".to_string(); 4];

    let result = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            confidence_level: 0.95,
        },
    );

    assert_eq!(
        result.unwrap_err(),
        MleError::Common(CommonError::InsufficientClusters {
            setting: ClusterSetting::CovType,
            g: 1
        })
    );
}

/// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下だと、
/// クラスターロバスト共分散`Ŝ`はクラスター寄与スコアの総和がゼロ（MLEの一次条件
/// `Σ_i s_i = 0`）で`rank(Ŝ) ≤ g - 1`となり退化する。`fit()`冒頭（Newton反復の前）の
/// バリデーションで`CommonError::InsufficientClustersForInference`として弾く
/// （Logitの同名テストと対、Logit/Probitでは新規制約）。説明変数2個
/// （`q = 3 - 1 = 2`）に対し`g = 2`。
#[test]
fn fit_returns_validation_error_when_cluster_count_at_most_slopes() {
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let x_columns = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    ];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let groups: Vec<String> = (0..6)
        .map(|i| if i < 3 { "a" } else { "b" }.to_string())
        .collect();

    let result = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            confidence_level: 0.95,
        },
    );

    assert_eq!(
        result.unwrap_err(),
        MleError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
    );
}

#[test]
fn fit_returns_invalid_confidence_level_error_out_of_range() {
    let result = ProbitEstimator::fit(
        intercept_only_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 1.5,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5
        })
    );
}

#[test]
fn fit_returns_invalid_max_iter_error_for_non_positive_max_iter() {
    let result = ProbitEstimator::fit(
        intercept_only_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 0,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::InvalidMaxIter { max_iter: 0 }
    );
}

#[test]
fn fit_returns_invalid_tol_error_for_non_positive_tol() {
    for tol in [0.0, -1.0] {
        let result = ProbitEstimator::fit(
            intercept_only_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(result.unwrap_err(), MleError::InvalidTol { tol });
    }
}

#[test]
fn fit_returns_invalid_binary_y_error_for_non_binary_y() {
    let y = vec![0.0, 1.0, 0.5, 1.0];
    let input = ProbitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap();
    let result = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::InvalidBinaryY { row: 2, value: 0.5 }
    );
}

#[test]
fn fit_returns_no_regressors_error_when_k_is_zero() {
    // `include_intercept=false`かつ説明変数も無い（k=0）病的な入力。`n<=k`チェック
    // （`n>=1`なら常に通過してしまう）をすり抜けて後段の分散共分散行列計算で
    // panicすることをLogit実装で経験済みのため、Probitでは
    // 当初からk==0を明示的に検証している（`probit.rs`のモジュールdocコメント・`fit`のdocコメント参照）。
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
    let input = ProbitInput::from_columns(&y, &[], vec![], false, "y".to_string()).unwrap();
    assert_eq!(input.k(), 0);

    let result = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::Common(CommonError::NoRegressors { n: 5 })
    );
}

#[test]
fn fit_returns_insufficient_observations_error_when_n_less_equal_k() {
    let y = vec![0.0, 1.0];
    let x_columns = vec![vec![1.0, 2.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
    );
}

/// 標準偏差0の列（定数列）は、切片ありでは標準化で平均が引かれて全ゼロ列になり、
/// 設計行列がランク落ちして`SingularDesignMatrix`で弾かれる（`solver`に依らない）。
#[test]
fn fit_returns_singular_design_matrix_error_for_constant_column_with_intercept() {
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let constant = vec![0.1; 8];

    for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
        let input = ProbitInput::from_columns(
            &y,
            &[x1.clone(), constant.clone()],
            vec!["x1".to_string(), "c".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let result = ProbitEstimator::fit(
            input,
            MleFitOptions {
                solver,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularDesignMatrix { .. })),
            "solver={solver:?}, result={result:?}"
        );
    }
}

/// 完全な多重共線性（`x2 = 2·x1`）の設計行列は、`fit()`冒頭の列ピボットQR
/// ランクチェック（`nonlinear::common::checked_design_matrix_qr`）で`solver`・
/// `cov_type`に関わらず単一経路で`SingularDesignMatrix`として弾かれる
/// （`LogitEstimator`の対応するテストと同じ設計・同じ経緯）。
///
/// 前段QRへの一本化で`solver`依存の検出漏れバグクラスを構造的に排除したため、
/// `solver`×`cov_type`を網羅していた旧5テスト（`..._with_bfgs_and_lbfgs` /
/// `..._with_hc0_and_hc1` / `fit_returns_singular_opg_matrix_error_...` /
/// `..._with_cluster`）を本1テストへ集約した。`x2`は`x1`から生成する。
///
/// 旧5テストが検証していた「`fit()`の各`cov_type`分岐での`SingularHessian`/
/// `SingularOpgMatrix`の`?`伝播」経路のカバレッジは、`common/cov.rs`の関数レベルテストと
/// `tobit/fit/tests/validation.rs`の`fit()`レベルテスト（`cov_params`計算は3手法で同一コード）が担う
/// （`LogitEstimator`の対応するテストのdocコメント参照）。
///
/// `Cluster`は`G=3 > q=2`にして`fit()`冒頭の`InsufficientClustersForInference`
/// （`G <= q`）より手前を通す。
#[test]
fn fit_returns_singular_design_matrix_error_for_perfectly_collinear_design_matrix() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let x2: Vec<f64> = x1.iter().map(|v| v * 2.0).collect();
    let groups = vec![
        "g1".to_string(),
        "g1".to_string(),
        "g2".to_string(),
        "g3".to_string(),
    ];

    for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
        for cov_type in [
            CovType::Classical,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Opg,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        ] {
            let input = ProbitInput::from_columns(
                &y,
                &[x1.clone(), x2.clone()],
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result = ProbitEstimator::fit(
                input,
                MleFitOptions {
                    solver,
                    max_iter: 100,
                    tol: 1e-6,
                    raise_on_non_convergence: true,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            );
            assert!(
                matches!(result, Err(MleError::SingularDesignMatrix { .. })),
                "solver={solver:?}, cov_type={cov_type:?}, result={result:?}"
            );
        }
    }
}
