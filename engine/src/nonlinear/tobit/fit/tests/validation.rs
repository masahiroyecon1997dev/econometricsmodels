//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleError;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::tobit::TobitEstimator;
use crate::nonlinear::tobit::TobitInput;
use crate::nonlinear::tobit::test_support::*;
use crate::shared::error::ClusterSetting;
use crate::shared::error::CommonError;
use crate::shared::error::ErrorInfo;
use crate::shared::group_codes::GroupCodes;

/// `cov_type=Cluster`のクラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`は、クラスター寄与
/// スコアの総和がゼロ（MLEの一次条件`Σ_i s_i = 0`）になるため`rank(Ŝ) ≤ G - 1`。
/// `multivariate_censored_input`（傾き係数`q=2`）に対し`G=2`（`G=q`ちょうど）を
/// 組み合わせると`q×q`部分行列は`rank ≤ 1 < 2`で構造的に特異になる。`G`・`q`は
/// 入力だけから判定できるため、`fit()`冒頭のバリデーション
/// （`validate_cluster_cov_type` → `validate_cluster_count_covers_slopes`）が
/// `CommonError::InsufficientClustersForInference`で弾く（
/// 検証が無ければ`wald_quadratic_form`内の`NearSingularMatrix`になる）。その
/// `ensure_well_conditioned_symmetric_matrix`側のbackstop（`g > q`だが悪条件で
/// 数値的にほぼ特異なケース）は、`wald_f_test`と共有する`wald_quadratic_form`の内部
/// （純粋な線形代数ユーティリティ）であり、`shared/linear_algebra.rs`の単体テスト・OLSの
/// `fit_returns_computation_failed_for_extreme_scale_difference_in_f_test`で担保する
/// （Tobitは`TobitScaling`が`x`を内部で列標準化するため、極端なスケール差による
/// 悪条件をTobitの`fit()`経由で再現するのが困難で、Tobit固有テストは設けない）。
#[test]
fn fit_returns_validation_error_when_cluster_count_equals_slopes() {
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
    ];
    let result = TobitEstimator::fit(
        multivariate_censored_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
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

/// `fit`は非収束（`raise_on_non_convergence=false`）でも常に収束点（打ち切り点）の
/// Hessianから`cov_params`を計算しようとするため、その打ち切り点でHessianが
/// 不定符号・特異な場合は`MleError::SingularHessian`が返る（Newtonの最適化過程
/// 自体は成功したが、SE計算のための逆行列計算が失敗するケース。`ols_initial_params`
/// 由来の`SingularDesignMatrix`（設計行列`X`自体の特異性、最適化前に発生）とは
/// 異なるエラー経路であることに注意）。`censored_regression_input`で`max_iter=1`
/// にすると、Newton初回ステップ後の打ち切り点で実際にこれが発生することを実測で
/// 確認済み（`max_iter=2`以降は`cov_params`計算が安定する。真の収束は`max_iter=11`、
/// `fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false`
/// のコメント参照）。
#[test]
fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point() {
    let input = censored_regression_input();
    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 1,
            tol: 1e-12,
            raise_on_non_convergence: false,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert!(
        matches!(result, Err(MleError::SingularHessian)),
        "{result:?}"
    );
}

/// 上のテストは`cov_type=Classical`（`observed_information_cov_params`）のみで
/// `SingularHessian`エラー伝播を検証しているが、`sandwich_cov_params`（`Hc0`/`Hc1`）も
/// 内部で同じHessianの逆行列計算を行うため、同じ打ち切り点で同じエラーが伝播する
/// はず。Logit/Probitの`fit_returns_singular_design_matrix_error_for_perfectly_
/// collinear_design_matrix`（`solver`×`cov_type`を1テストに集約）と同じ
/// 「cov_type分岐ごとのエラー伝播`?`」のギャップパターンを
/// Tobitでも確認する（`cargo llvm-cov`で発覚）。
#[test]
fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point_with_hc0_and_hc1()
 {
    for cov_type in [CovType::Hc0, CovType::Hc1] {
        let result = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 1,
                tol: 1e-12,
                raise_on_non_convergence: false,
                cov_type: cov_type.clone(),
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "cov_type={cov_type:?}, result={result:?}"
        );
    }
}

/// `cluster_cov_params`（`Cluster`）も同じ理由で`SingularHessian`が伝播するはず
/// （上記テストのdocコメント参照）。
#[test]
fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point_with_cluster()
 {
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
    ];
    let result = TobitEstimator::fit(
        censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 1,
            tol: 1e-12,
            raise_on_non_convergence: false,
            cov_type: CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            confidence_level: 0.95,
        },
    );
    assert!(
        matches!(result, Err(MleError::SingularHessian)),
        "{result:?}"
    );
}

#[test]
fn fit_returns_invalid_confidence_level_error_out_of_range() {
    let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
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
fn fit_returns_invalid_max_iter_error() {
    let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 0,
            tol: 1e-8,
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
fn fit_returns_invalid_tol_error() {
    let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 0.0,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert_eq!(result.unwrap_err(), MleError::InvalidTol { tol: 0.0 });
}

#[test]
fn fit_returns_insufficient_observations_error() {
    // n=2, 総パラメータ数k+1=2(切片1+logσ1) → n<=k+1でエラー
    // （`validate_sufficient_observations`にx列数ではなくk+1を渡す設計）。
    let input = intercept_only_uncensored_input(&[1.0, 2.0]);
    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    let err = result.unwrap_err();
    assert_eq!(
        err,
        MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
    );
    // `n_params`は`β`の数に`log σ`を足した値になる。
    let report = err.report();
    assert_eq!(
        report.message(),
        "Not enough observations: 2 observation(s) for 2 parameter(s); at least 3 are needed."
    );
}

#[test]
fn fit_returns_singular_design_matrix_error_for_perfectly_collinear_data() {
    // x2=2*x1（完全な多重共線性）。ols_initial_paramsのQR分解が特異性を検出し、
    // Newton法が一度も反復する前にエラーを返す（SingularHessian
    // とは区別されるバリアント`SingularDesignMatrix`の検証）。
    // n=6 > 総パラメータ数k+1=4（切片1+x1+x2+logσ1）で`InsufficientObservations`を
    // 回避しつつ、x2=2*x1の構造的な多重共線性を検証する。
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_columns = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    ];
    let input = TobitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
        Some(-1000.0),
        Some(1000.0),
    )
    .unwrap();

    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert!(
        matches!(result, Err(MleError::SingularDesignMatrix { .. })),
        "{:?}",
        result
    );
}

#[test]
fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
    let result = TobitEstimator::fit(
        multivariate_censored_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
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
    let groups = vec!["a".to_string(); 8];
    let result = TobitEstimator::fit(
        multivariate_censored_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
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

/// Logit（`fit_returns_singular_opg_matrix_error_for_perfectly_collinear_
/// design_matrix`）はxの完全な多重共線性（x2=2*x1）でOPG行列`Σᵢsᵢsᵢ'`を特異にして
/// いるが、Tobitでは`ols_initial_params`が最適化前にxのQRベースの階数検定を行うため
/// （`fit_returns_singular_design_matrix_error_for_perfectly_collinear_data`参照）、
/// xが完全に多重共線だと最適化に入る前に`SingularDesignMatrix`になり同じ手法は使えない。
///
/// このテストのデータ（x1・x2は互いに無関係、多重共線性なし）は実測で見つけた
/// 別の退化パターンで、`SingularOpgMatrix`の再現に多重共線性は必須ではないことが
/// 分かっている（独立検証で、x2をx1と無関係な値に変えても収束点での
/// OPG行列の最小固有値が機械精度オーダーまで落ちる同じ現象を確認済み）。原因は
/// 観測数`n=6`が総パラメータ数`k+1=4`に対して少なく（`validate_sufficient_
/// observations`の下限ぎりぎり）、かつ左打ち切り観測が2件という小標本特有の
/// 収束点の数値配置に依存すると推測されるが、厳密な理論的必然性までは未解明
/// （`n=7`など観測数を増やすと再現しなくなることを確認済み）。「なぜこの具体的な
/// データで起こるか」を将来の保守者が誤って多重共線性のせいだと読み違えないよう、
/// ここに正直に記録する。
#[test]
fn fit_returns_singular_opg_matrix_error_for_small_sample_degenerate_case() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85];
    let x2: Vec<f64> = vec![5.0, -3.0, 7.0, 1.0, -2.0, 9.0];
    let input = TobitInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        None,
    )
    .unwrap();

    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Opg,
            confidence_level: 0.95,
        },
    );
    assert!(
        matches!(result, Err(MleError::SingularOpgMatrix)),
        "{result:?}"
    );
}

/// 全件が下限（`lower`）で打ち切られている（非打ち切り観測が1件も無い）場合は
/// `MleError::NoUncensoredObservations`を返す。実測で確認済み: このバリデーションが
/// 無いと`fit()`は`converged=true`のまま統計的に無意味な巨大SE（100万倍オーダー）を
/// 返す退化収束を起こしていた（参照実装`survival::survreg`は同種のデータで
/// エラーを返すことをdevcontainer内で実際に確認している）。
#[test]
fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_at_lower() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![0.0; 8];
    let input = TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        None,
    )
    .unwrap();

    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::NoUncensoredObservations {
            lower: Some(0.0),
            upper: None
        }
    );
}

/// 上のテストは`lower`側のみの片側打ち切りを検証しているが、`at_lower`/`at_upper`の
/// 判定ロジックが対称的に実装されていることを、`upper`側のみの片側打ち切りでも
/// 独立に確認する。
#[test]
fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_at_upper() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![6.0; 8];
    let input = TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        None,
        Some(6.0),
    )
    .unwrap();

    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::NoUncensoredObservations {
            lower: None,
            upper: Some(6.0)
        }
    );
}

/// 両側打ち切りで、非打ち切り観測が1件も無い（全観測が`lower`または`upper`ちょうど）
/// 場合も同じエラーになることを確認する（`lower`側だけの判定に留まらないことの検証）。
#[test]
fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_two_sided() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0];
    let input = TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        Some(6.0),
    )
    .unwrap();

    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        MleError::NoUncensoredObservations {
            lower: Some(0.0),
            upper: Some(6.0)
        }
    );
}

/// 非打ち切り観測がちょうど1件ある場合は`NoUncensoredObservations`を返さない
/// （「1件以上」という条件の境界値、`validate_sufficient_observations`と同じ
/// `>=`境界の考え方）。このデータは病理的に近い（実際にNewtonは収束しない）ため
/// `raise_on_non_convergence=false`で受け、`NonConvergence`ではなく
/// バリデーション自体は通過することだけを確認する。
#[test]
fn fit_does_not_return_no_uncensored_observations_error_when_exactly_one_observation_is_uncensored()
{
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.95];
    let input = TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        None,
    )
    .unwrap();

    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: false,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert!(
        !matches!(result, Err(MleError::NoUncensoredObservations { .. })),
        "{result:?}"
    );
}
