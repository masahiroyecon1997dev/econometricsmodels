//! `fit`の`cov_type`別（Classical・HC・Cluster・HAC・Driscoll-Kraay など）の共分散・検定統計量のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SandwichVariant;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::common::cluster_cov_params;
use crate::nonlinear::common::destandardize_cov_params;
use crate::nonlinear::common::opg_cov_params;
use crate::nonlinear::common::sandwich_cov_params;
use crate::nonlinear::common::standardize_columns;
use crate::nonlinear::probit::ProbitEstimator;
use crate::nonlinear::probit::ProbitInput;
use crate::nonlinear::probit::problem::ProbitProblem;
use crate::shared::group_codes::GroupCodes;
use argmin::core::Hessian;
use faer::Mat;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// 多変量（説明変数が2つ以上）の場合、標準誤差に閉じた形の解析解は無いため、
/// `cov_params`の対称性・各種統計量の内部整合性（z値・信頼区間の定義式通りの関係）を
/// 検証する回帰テスト。特に`destandardize_cov_params`が非対角成分も含めて
/// 正しく元のスケールへ逆変換できているかを確認する（対角成分だけでは
/// `destandardize_cov_params`のヤコビアン`J`の非対角（切片行）の取り違えや転置ミス等を検出できない）。
#[test]
fn fit_cov_params_is_symmetric_and_stats_are_internally_consistent() {
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

    for i in 0..k {
        for j in 0..k {
            assert!(
                (*estimator.cov_params().get(i, j) - *estimator.cov_params().get(j, i)).abs()
                    < 1e-9,
                "cov_params is not symmetric at ({i},{j})"
            );
        }
        assert!(
            *estimator.cov_params().get(i, i) > 0.0,
            "diagonal[{i}] <= 0"
        );
    }

    let normal = Normal::standard();
    let z_crit = normal.inverse_cdf(0.975);
    for j in 0..k {
        let se = estimator.std_errors()[j];
        assert!((se * se - *estimator.cov_params().get(j, j)).abs() < 1e-9);
        assert!((estimator.test_stats()[j] - estimator.params()[j] / se).abs() < 1e-9);
        assert!(
            (estimator.conf_upper()[j] - estimator.conf_lower()[j] - 2.0 * z_crit * se).abs()
                < 1e-9
        );
    }
}

/// `cov_type=Opg`/`Hc0`/`Hc1`が返す`cov_params`を、`fit()`と同じ手順
/// （標準化→収束点でのscores/Hessian評価→`common/cov.rs`の共通行列演算→
/// `destandardize_cov_params`）をテスト側で独立に再現した値と突き合わせる。
/// 多変量（k=3）データセットを使う理由: 切片のみモデルでは情報行列の等式
/// （`Σᵢsᵢsᵢ' = -H`）が有限標本で厳密に成り立ってしまい、`classical`/`opg`/`hc0`
/// が偶然同じ値になるため、`fit()`の`cov_type`振り分け（`cov_params_for_type`への配線）のミスを検出できない
/// （`fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ
/// データセットを再利用する。Logitの対応するテストと同じ構成）。
#[test]
fn fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let make_input = || {
        ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };
    let k = 3;
    let n = 4;

    let classical = ProbitEstimator::fit(
        make_input(),
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

    // `fit()`と同じ手順を独立に再現する: 標準化→θ_stdへ変換→収束点でのscores/
    // Hessian評価→destandardize_cov_params。`ProbitProblem::hessian`（argminトレイト）は
    // コスト関数（負の対数尤度）のHessianを返す符号規約のため、`run_solver`が
    // `SolverOutput.hessian`に格納する対数尤度そのもののHessianに合わせて1回符号反転する
    // （`run_solver`のdocコメント「Hessianトレイトの符号規約」と同じ変換）。
    let input_for_reconstruction = make_input();
    let (x_std, scale) = standardize_columns(
        input_for_reconstruction.x(),
        input_for_reconstruction.has_intercept(),
    );
    let params_std = scale.standardize_params(classical.params());
    let problem_std = ProbitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
    let scores_std = problem_std.scores(&params_std);
    let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
    let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

    let expected_opg = destandardize_cov_params(&opg_cov_params(&scores_std, k).unwrap(), &scale);
    let expected_hc0 = destandardize_cov_params(
        &sandwich_cov_params(&hessian_std, &scores_std, n, k, SandwichVariant::Hc0).unwrap(),
        &scale,
    );
    let expected_hc1 = destandardize_cov_params(
        &sandwich_cov_params(&hessian_std, &scores_std, n, k, SandwichVariant::Hc1).unwrap(),
        &scale,
    );

    let cases = [
        (CovType::Opg, &expected_opg),
        (CovType::Hc0, &expected_hc0),
        (CovType::Hc1, &expected_hc1),
    ];
    for (cov_type, expected) in cases {
        let estimator = ProbitEstimator::fit(
            make_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: cov_type.clone(),
                confidence_level: 0.95,
            },
        )
        .unwrap();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *expected.get(i, j)).abs() < 1e-8,
                    "cov_type={:?}, ({i},{j}): actual={}, expected={}",
                    cov_type,
                    *estimator.cov_params().get(i, j),
                    *expected.get(i, j)
                );
            }
        }
    }
}

/// `cov_type=Cluster`が返す`cov_params`を、`fit()`と同じ手順をテスト側で独立に
/// 再現した値と突き合わせる（上の`fit_cov_type_opg_hc0_hc1_match_independently_
/// recomputed_values`と同じ技法・同じ多変量データセット。情報行列の等式が
/// 厳密に成り立つ切片のみモデルでは配線ミスを検出できないため。Logitの
/// 対応するテストと同じ構成）。`G=3 > q=2`（`G <= q`は
/// `InsufficientClustersForInference`で弾かれる）ため`G=3`（2:1:1）。
#[test]
fn fit_cov_type_cluster_matches_independently_recomputed_values() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let make_input = || {
        ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };
    let k = 3;
    let n = 4;
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
    ];

    let classical = ProbitEstimator::fit(
        make_input(),
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

    let input_for_reconstruction = make_input();
    let (x_std, scale) = standardize_columns(
        input_for_reconstruction.x(),
        input_for_reconstruction.has_intercept(),
    );
    let params_std = scale.standardize_params(classical.params());
    let problem_std = ProbitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
    let scores_std = problem_std.scores(&params_std);
    let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
    let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

    let expected_cluster = destandardize_cov_params(
        &cluster_cov_params(
            &hessian_std,
            &scores_std,
            n,
            k,
            &crate::shared::group_codes::GroupCodes::from_labels(&groups),
        )
        .unwrap(),
        &scale,
    );

    let estimator = ProbitEstimator::fit(
        make_input(),
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
    )
    .unwrap();
    for i in 0..k {
        for j in 0..k {
            assert!(
                (*estimator.cov_params().get(i, j) - *expected_cluster.get(i, j)).abs() < 1e-8,
                "({i},{j}): actual={}, expected={}",
                *estimator.cov_params().get(i, j),
                *expected_cluster.get(i, j)
            );
        }
    }
}

/// 上のテストは均等サイズのグループのみを検証しているが、`testing-policy.md`が
/// 指摘する通り均等サイズのみのテストは実務で起こりやすい偏った分布のグループサイズ
/// を見逃しうる。OLS/Logit側の対応するテストに倣い、3:1:1の不均衡なグループでも
/// 同じ独立再計算の技法で検証する（`G=3 > q=2`）。
#[test]
fn fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups() {
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
    let x_columns = vec![
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![-5.0, 2.0, 8.0, -1.0, 3.0],
    ];
    let make_input = || {
        ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };
    let k = 3;
    let n = 5;
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
    ];

    let classical = ProbitEstimator::fit(
        make_input(),
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

    let input_for_reconstruction = make_input();
    let (x_std, scale) = standardize_columns(
        input_for_reconstruction.x(),
        input_for_reconstruction.has_intercept(),
    );
    let params_std = scale.standardize_params(classical.params());
    let problem_std = ProbitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
    let scores_std = problem_std.scores(&params_std);
    let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
    let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

    let expected_cluster = destandardize_cov_params(
        &cluster_cov_params(
            &hessian_std,
            &scores_std,
            n,
            k,
            &crate::shared::group_codes::GroupCodes::from_labels(&groups),
        )
        .unwrap(),
        &scale,
    );

    let estimator = ProbitEstimator::fit(
        make_input(),
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
    )
    .unwrap();
    for i in 0..k {
        for j in 0..k {
            assert!(
                (*estimator.cov_params().get(i, j) - *expected_cluster.get(i, j)).abs() < 1e-8,
                "({i},{j}): actual={}, expected={}",
                *estimator.cov_params().get(i, j),
                *expected_cluster.get(i, j)
            );
        }
    }
}

/// `solver`（`bfgs`/`lbfgs`）と`cov_type`（`Opg`/`Hc0`/`Hc1`/`Cluster`）の組み合わせが
/// 正しく機能することを確認する（Logitの対応するテストと同じ理由: `solver`横断が
/// `CovType::Classical`のみ、`cov_type`横断が`SolverType::Newton`のみのテストでは、
/// 両方を同時に変える組み合わせが未検証になる）。`scores_std`の評価は収束点の
/// パラメータにのみ依存し最適化アルゴリズムの種類に依存しない設計のため、
/// `newton`で計算した`cov_params`（既に上のテストで正しさを検証済み）と
/// `bfgs`/`lbfgs`の結果が一致するはず。
#[test]
fn fit_non_classical_cov_types_work_with_bfgs_and_lbfgs() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let make_input = || {
        ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };
    let k = 3;
    // `G=3 > q=2`（`G <= q`は`InsufficientClustersForInference`で弾かれる）。
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
    ];

    for cov_type in [
        CovType::Opg,
        CovType::Hc0,
        CovType::Hc1,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
    ] {
        let newton = ProbitEstimator::fit(
            make_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: cov_type.clone(),
                confidence_level: 0.95,
            },
        )
        .unwrap();

        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let estimator = ProbitEstimator::fit(
                make_input(),
                MleFitOptions {
                    solver,
                    max_iter: 200,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            )
            .unwrap();

            assert!(
                estimator.converged(),
                "cov_type={:?}, {:?}",
                cov_type,
                solver
            );
            for i in 0..k {
                for j in 0..k {
                    assert!(
                        (*estimator.cov_params().get(i, j) - *newton.cov_params().get(i, j)).abs()
                            < 1e-4,
                        "cov_type={:?}, solver={:?}, ({i},{j}): actual={}, newton={}",
                        cov_type,
                        solver,
                        *estimator.cov_params().get(i, j),
                        *newton.cov_params().get(i, j)
                    );
                }
            }
        }
    }
}
