//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleError;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::logit::LogitEstimator;
use crate::nonlinear::logit::LogitInput;
use crate::nonlinear::logit::test_support::*;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

#[test]
fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let input = LogitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = LogitEstimator::fit(
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
    let input = LogitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let groups = vec!["a".to_string(); 4];

    let result = LogitEstimator::fit(
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
/// `Σ_i s_i = 0`）で`rank(Ŝ) ≤ g - 1`となり退化する。Logit/Probitは全体検定がLR
/// のため`q×q`反転こそ通らないが、縮退した共分散から読んだSEを無警告で返すのは
/// 実質バグのため、`fit()`冒頭（Newton反復の前）のバリデーションで
/// `CommonError::InsufficientClustersForInference`として弾く（
/// Logit/Probitでは新規制約）。説明変数2個（`q = 3 - 1 = 2`）に対し`g = 2`。
#[test]
fn fit_returns_validation_error_when_cluster_count_at_most_slopes() {
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let x_columns = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    ];
    let input = LogitInput::from_columns(
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

    let result = LogitEstimator::fit(
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
    let result = LogitEstimator::fit(
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
    let result = LogitEstimator::fit(
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
fn fit_returns_insufficient_observations_error_when_n_less_equal_k() {
    let y = vec![0.0, 1.0];
    let x_columns = vec![vec![1.0, 2.0]];
    let input = LogitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = LogitEstimator::fit(
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

#[test]
fn fit_returns_no_regressors_error_when_k_is_zero() {
    // `include_intercept=false`かつ説明変数も無い（k=0）病的な入力。
    // 以前はこの経路が`fit()`冒頭の`n<=k`チェック（n>=1なら常に通過してしまう）を
    // すり抜け、後段の`cov_params`計算（0×0行列に対する
    // `ensure_well_conditioned_symmetric_matrix`）で`faer`がpanicしていた
    // すり抜けていた。`k==0`を明示的に検証することでグレースフルなエラーに
    // なることを確認する。`n<=k`（観測数不足）とは
    // 別の不正のため、`InsufficientObservations`ではなく専用の`NoRegressors`を返す
    // （`k=0`だと`n>k`は常に成立してしまい、`InsufficientObservations`のメッセージ
    // だと「条件を満たしているのにエラーになる」という誤解を招くため）。
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
    let input = LogitInput::from_columns(&y, &[], vec![], false, "y".to_string()).unwrap();
    assert_eq!(input.k(), 0);

    let result = LogitEstimator::fit(
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
fn fit_returns_invalid_tol_error_for_non_positive_tol() {
    for tol in [0.0, -1.0] {
        let result = LogitEstimator::fit(
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
fn fit_returns_invalid_binary_y_error_for_non_binary_values() {
    // 0.5（連続値）・2.0（0/1ではない2値）・statsmodelsなら許容する単位区間内の
    // 中間値も含め、{0.0, 1.0}の完全一致以外は全て弾かれることを確認する
    // （statsmodelsより厳格な`{0.0, 1.0}`のみ許容という設計判断）。
    for (bad_value, bad_row) in [(0.5, 1), (2.0, 2)] {
        let mut y = vec![0.0, 1.0, 0.0, 1.0];
        y[bad_row] = bad_value;
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = LogitEstimator::fit(
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
            MleError::InvalidBinaryY {
                row: bad_row,
                value: bad_value
            }
        );
    }
}

/// 標準偏差0の列（定数列）は、切片ありでは標準化で平均が引かれて全ゼロ列になり、
/// 設計行列がランク落ちして`SingularDesignMatrix`で弾かれる（`solver`に依らない）。
#[test]
fn fit_returns_singular_design_matrix_error_for_constant_column_with_intercept() {
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let constant = vec![0.1; 8];

    for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
        let input = LogitInput::from_columns(
            &y,
            &[x1.clone(), constant.clone()],
            vec!["x1".to_string(), "c".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let result = LogitEstimator::fit(
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
/// `cov_type`に関わらず単一経路で`SingularDesignMatrix`として弾かれる。
///
/// 従来はゼロベクトル初期値で、`newton`は`newton_step`内の列ピボットQR、`bfgs`/`lbfgs`は
/// 収束後の`observed_information_cov_params`（`cov_type=Opg`なら`opg_cov_params`）と
/// いう`solver`/`cov_type`依存の別経路で検出しており、過去に`bfgs`だけ検出漏れして
/// 桁違いに巨大なSEを含む`Ok`が返る実バグがあった（`tests/nonlinear/
/// test_logit_validation.py`の履歴参照）。前段QRへの一本化でそのバグクラスを
/// 構造的に排除したため、`solver`×`cov_type`を網羅していた旧5テスト
/// （`..._with_bfgs_and_lbfgs` / `..._with_hc0_and_hc1` / `fit_returns_singular_opg_
/// matrix_error_...` / `..._with_cluster`）を本1テストへ集約した。`x2`は`x1`から
/// 生成し関係を自明にする。
///
/// 旧5テストが検証していた「`fit()`の各`cov_type`分岐で`SingularHessian`/
/// `SingularOpgMatrix`が`?`で伝播する」経路は、本入力では前段QRで先に弾かれるため
/// もう通らない。この伝播経路のカバレッジは (1) `common/cov.rs`の関数レベルテスト
/// （`observed_information_cov_params_returns_singular_hessian_error_*` /
/// `opg_cov_params_*` / `sandwich_cov_params_*` / `cluster_cov_params_*`）と、
/// (2) 共有インフラを`fit()`レベルで踏む`tobit/fit/tests/validation.rs`の
/// `fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_
/// truncated_point_with_hc0_and_hc1` / `..._with_cluster` が担う（`cov_params`計算は
/// Logit/Probit/Tobitで同一コード）。`?`自体は分岐ロジックを持たないため
/// `testing-policy.md`のカバレッジ方針上これで足りる。
///
/// `Cluster`は`G=3 > q=2`（`q = k - k_constant`）にして`fit()`冒頭の
/// `InsufficientClustersForInference`（`G <= q`）より手前を通す。
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
            let input = LogitInput::from_columns(
                &y,
                &[x1.clone(), x2.clone()],
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result = LogitEstimator::fit(
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
