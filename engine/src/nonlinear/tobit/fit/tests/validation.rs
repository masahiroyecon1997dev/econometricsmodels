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
/// `ensure_well_conditioned_symmetric_matrix`側のbackstop（`g > q`だが準多重共線性で
/// 数値的にほぼ特異なケース）は、`wald_f_test`と共有する`wald_quadratic_form`の内部
/// （相関スケール化した行列の判定）であり、`shared/wald.rs`の単体テスト
/// （`wald_f_test_rejects_a_near_singular_covariance_submatrix`）・OLSの
/// `fit_with_optional_f_test_does_not_hide_other_cluster_errors`で担保する。
/// 列の単位だけでは失敗しない（`tests/cross_method/test_unit_invariance.py`）。
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

/// 打ち切りの有無が`x`で完全に決まり、非打ち切り観測の当てはまりも完全なデータ
/// （`y = max(0, 100·x1 + 0.5·x2)`）。`σ→0`へ退化するので、どの`solver`・`max_iter`でも
/// 打ち切り点のHessianが特異になる。
fn noise_free_separation_input() -> TobitInput {
    let n = 120;
    let x1: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * (i as f64) / ((n - 1) as f64))
        .collect();
    let x2: Vec<f64> = (0..n).map(|i| (0.7 * i as f64).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (100.0 * x1[i] + 0.5 * x2[i]).max(0.0))
        .collect();
    TobitInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        None,
    )
    .unwrap()
}

/// `fit`は非収束（`raise_on_non_convergence=false`）でも常に打ち切り点のHessianから
/// `cov_params`を計算しようとするため、その打ち切り点でHessianが特異なら
/// `MleError::SingularHessian`が返る（最適化自体は成功したが、SE計算のための逆行列計算が
/// 失敗するケース。設計行列`X`自体の特異性（`SingularDesignMatrix`、最適化前に発生）とは
/// 異なるエラー経路）。`noise_free_separation_input`は`σ→0`に退化するので、`max_iter`を
/// 小さくした打ち切り点で実際にこれが起きる。
#[test]
fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point() {
    let input = noise_free_separation_input();
    let result = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 5,
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
            noise_free_separation_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 5,
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
    let groups: Vec<String> = (0..120).map(|i| format!("g{}", i % 4)).collect();
    let result = TobitEstimator::fit(
        noise_free_separation_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 5,
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
    // x2=2*x1（完全な多重共線性）。`DesignCoordinates::new`のランク判定が特異性を検出し、
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
/// いるが、Tobitでは`DesignCoordinates::new`が最適化前にxのQRベースの階数検定を行うため
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

/// 設定値（`confidence_level` → `max_iter` → `tol`）、打ち切り境界、観測数の順に報告される
/// （設定値の検査をデータに依る検査より先に行う順序は、Logit・Probitと同じ）。
#[test]
fn fit_reports_settings_before_the_censoring_bounds_and_the_observation_count() {
    // `n = 2 <= k + 1 = 2`（定数項のみ）で、`y`が下限`0.0`を下回る行を含み、境界も不正。
    let build = |lower: Option<f64>, upper: Option<f64>| {
        TobitInput::from_columns(
            &[-1.0, 1.0],
            &[],
            vec![],
            true,
            "y".to_string(),
            lower,
            upper,
        )
        .unwrap()
    };
    let options = |confidence_level: f64, max_iter: i64, tol: f64| MleFitOptions {
        solver: SolverType::Newton,
        max_iter,
        tol,
        raise_on_non_convergence: true,
        cov_type: CovType::Classical,
        confidence_level,
    };
    let fit = |lower: Option<f64>, upper: Option<f64>, options: MleFitOptions| {
        TobitEstimator::fit(build(lower, upper), options).unwrap_err()
    };

    // 設定値が不正なら、境界が不正でも設定値のエラー（`confidence_level` → `max_iter` → `tol`）。
    assert!(matches!(
        fit(None, None, options(1.5, 0, -1.0)),
        MleError::Common(CommonError::InvalidConfidenceLevel { .. })
    ));
    assert!(matches!(
        fit(None, None, options(0.95, 0, -1.0)),
        MleError::InvalidMaxIter { .. }
    ));
    assert!(matches!(
        fit(None, None, options(0.95, 10, -1.0)),
        MleError::InvalidTol { .. }
    ));
    // 設定値が有効なら、観測数より先に境界。
    assert_eq!(
        fit(None, None, options(0.95, 10, 1e-8)),
        MleError::CensoringBoundsMissing
    );
    assert!(matches!(
        fit(Some(0.0), None, options(0.95, 10, 1e-8)),
        MleError::YOutOfCensoringBounds { row: 0, .. }
    ));
    // 境界が有効で`y`も範囲内なら、観測数。
    assert_eq!(
        TobitEstimator::fit(
            TobitInput::from_columns(
                &[0.0, 1.0],
                &[],
                vec![],
                true,
                "y".to_string(),
                Some(0.0),
                None
            )
            .unwrap(),
            options(0.95, 10, 1e-8),
        )
        .unwrap_err(),
        MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
    );
}
