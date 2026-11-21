//! `fit`の`cov_type`別（Classical・HC・Cluster・HAC・Driscoll-Kraay など）の共分散・検定統計量のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SandwichVariant;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::common::cluster_cov_params;
use crate::nonlinear::common::opg_cov_params;
use crate::nonlinear::common::sandwich_cov_params;
use crate::nonlinear::tobit::TobitEstimator;
use crate::nonlinear::tobit::TobitInput;
use crate::nonlinear::tobit::problem::TobitProblem;
use crate::nonlinear::tobit::scaling::TobitScaling;
use crate::nonlinear::tobit::test_support::*;
use crate::shared::group_codes::GroupCodes;
use argmin::core::Hessian;
use faer::Mat;
use statrs::distribution::ChiSquared;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// `fit()`と同じ手順（`TobitScaling`による`x`のセンタリング＋スケーリングと`y`の
/// スケーリング→収束点でのscores/Hessian評価→`common/cov.rs`の`cov_type`別の共通行列
/// 演算→`TobitScaling::destandardize_cov_params`）をテスト側で独立に再現し、
/// `cov_params_std_fn`が計算する標準化空間の分散共分散行列から最終的な`(β,σ)`空間の
/// `cov_params`を得る。`Logit`の対応テストと同じ技法（`fit()`本体のロジックを
/// 再利用せず、独立に再現することで配線ミスを検出する。`TobitScaling`の各メソッドは
/// 元々の`standardize_columns`/`destandardize_cov_params`と同じく共有ヘルパー扱い）。
fn expected_cov_params(
    input: &TobitInput,
    classical: &TobitEstimator,
    cov_params_std_fn: impl FnOnce(&Mat<f64>, &Mat<f64>) -> Mat<f64>,
) -> Mat<f64> {
    let k = input.k();
    let scaling = TobitScaling::fit(input.x(), input.y(), input.has_intercept());
    let x_std = scaling.standardize_x(input.x());
    let y_std = scaling.standardize_y(input.y());
    let lower_std = scaling.standardize_bound(input.lower());
    let upper_std = scaling.standardize_bound(input.upper());

    // 元のスケールの`(β, σ)`から標準化空間の内部パラメータ`(β̃, s̃=logσ̃)`を再構成する
    // （`destandardize_params`の逆写像。`TobitScaling`のdocコメントの式を反転）。
    let c = scaling.y_scale;
    // `β/c`を`ColumnScale::standardize_params`（`x`側の逆写像）で標準化空間へ戻す。
    let beta_over_c: Vec<f64> = classical.params()[..k].iter().map(|b| b / c).collect();
    let mut params_std = scaling.x.standardize_params(&beta_over_c);
    params_std.push(classical.sigma().ln() - c.ln());

    let problem_std = TobitProblem::from_standardized(x_std, y_std, lower_std, upper_std);
    let scores_std = problem_std.scores(&params_std);
    let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
    let hessian_std = Mat::from_fn(k + 1, k + 1, |i, j| -cost_hessian_std[i][j]);

    scaling.destandardize_cov_params(
        &cov_params_std_fn(&hessian_std, &scores_std),
        classical.sigma(),
    )
}

fn assert_cov_params_close(actual: &Mat<f64>, expected: &Mat<f64>, k_plus_1: usize, tol: f64) {
    for i in 0..k_plus_1 {
        for j in 0..k_plus_1 {
            assert!(
                (*actual.get(i, j) - *expected.get(i, j)).abs() < tol,
                "({i},{j}): actual={}, expected={}",
                *actual.get(i, j),
                *expected.get(i, j)
            );
        }
    }
}

/// `df_model=1`（傾き係数が1個）のとき、`W=z²`（カイ二乗(1)統計量は標準正規のz値の
/// 2乗に一致するという代数的恒等式）が成り立つことを検証する。`fit_wald_statistic_
/// and_p_value_match_independently_recomputed_values`（`q=2`、手動2×2逆行列）とは
/// 別角度の検算（OLSの`wald_test_last_columns_matches_squared_t_statistic_for_
/// single_column`と同じ発想）。`censored_regression_input`（`q=1`）を使う。
#[test]
fn fit_wald_statistic_matches_squared_z_statistic_for_single_slope() {
    let estimator = TobitEstimator::fit(
        censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    assert_eq!(estimator.df_model(), 1);
    let z_slope = estimator.test_stats()[1];
    assert!((estimator.wald_statistic() - z_slope.powi(2)).abs() < 1e-9);

    let chi2 = ChiSquared::new(1.0).unwrap();
    let expected_p = 1.0 - chi2.cdf(estimator.wald_statistic());
    assert!((estimator.wald_p_value() - expected_p).abs() < 1e-9);
}

/// 多変量（説明変数が1つ、切片込みで`k=2`・`s`を含め合計3パラメータ）の場合、
/// 閉じた形の解析解は無いため、`cov_params`の対称性・各種統計量の内部整合性
/// （z値・信頼区間の定義式通りの関係）を検証する回帰テスト。Logitの
/// `fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ設計。
/// 特に`TobitScaling::destandardize_cov_params`（`J·Cov·Jᵀ`）の適用が非対角成分も
/// 含めて正しく機能しているかを確認する（対角成分だけでは転置ミス等の一部のバグを
/// 検出できない）。
#[test]
fn fit_cov_params_is_symmetric_and_stats_are_internally_consistent() {
    let estimator = TobitEstimator::fit(
        censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let k_plus_1 = estimator.params().len() + 1;
    for i in 0..k_plus_1 {
        for j in 0..k_plus_1 {
            assert!(
                (*estimator.cov_params().get(i, j) - *estimator.cov_params().get(j, i)).abs()
                    < 1e-9,
                "cov_params not symmetric at ({i},{j})"
            );
        }
        assert!(
            *estimator.cov_params().get(i, i) > 0.0,
            "diagonal[{i}] <= 0"
        );
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_crit = normal.inverse_cdf(0.975);
    let mut coefs = estimator.params().to_vec();
    coefs.push(estimator.sigma());

    for (j, &coef) in coefs.iter().enumerate().take(k_plus_1) {
        let se = estimator.std_errors()[j];
        assert!((se - estimator.cov_params().get(j, j).sqrt()).abs() < 1e-9);
        assert!((estimator.test_stats()[j] - coef / se).abs() < 1e-9);
        assert!(
            (estimator.p_values()[j] - 2.0 * (1.0 - normal.cdf(estimator.test_stats()[j].abs())))
                .abs()
                < 1e-9
        );
        assert!((estimator.conf_lower()[j] - (coef - z_crit * se)).abs() < 1e-9);
        assert!((estimator.conf_upper()[j] - (coef + z_crit * se)).abs() < 1e-9);
    }
}

#[test]
fn fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values() {
    let k_plus_1 = 4;
    let n = 8;

    let classical = TobitEstimator::fit(
        multivariate_censored_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let expected_opg = expected_cov_params(&multivariate_censored_input(), &classical, |_h, s| {
        opg_cov_params(s, k_plus_1).unwrap()
    });
    let expected_hc0 = expected_cov_params(&multivariate_censored_input(), &classical, |h, s| {
        sandwich_cov_params(h, s, n, k_plus_1, SandwichVariant::Hc0).unwrap()
    });
    let expected_hc1 = expected_cov_params(&multivariate_censored_input(), &classical, |h, s| {
        sandwich_cov_params(h, s, n, k_plus_1, SandwichVariant::Hc1).unwrap()
    });

    let cases = [
        (CovType::Opg, &expected_opg),
        (CovType::Hc0, &expected_hc0),
        (CovType::Hc1, &expected_hc1),
    ];
    for (cov_type, expected) in cases {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert_cov_params_close(estimator.cov_params(), expected, k_plus_1, 1e-6);
    }
}

/// `solver`（`bfgs`/`lbfgs`）と`cov_type`（`Opg`/`Hc0`/`Hc1`）の組み合わせが正しく
/// 機能することを確認する（Logitの`fit_non_classical_cov_types_work_with_bfgs_and_
/// lbfgs`と同じ理由。`scores_std`の評価は収束点のパラメータにのみ依存し最適化
/// アルゴリズムの種類に依存しない設計のため、`newton`で計算した`cov_params`
/// （上のテストで既に正しさを検証済み）と`bfgs`/`lbfgs`の結果が一致するはず）。
///
/// クラスターのグループ数は`G=4`（2件ずつ）にする。`multivariate_censored_input`は
/// 傾き係数`q=2`（`x1`・`x2`、切片を除く）を持ち、Wald検定が`fit()`に
/// 常時組み込まれたことで、クラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`の構造的な制約
/// （`rank(Ŝ)≤G`、`engine/src/linear/CLAUDE.md`「クラスター数`G`と傾き係数の数`q`の
/// 関係」参照）がWald検定の`q×q`部分行列にも及ぶことが判明した。`G=2`（`q`と同数）
/// では実測でこの部分行列が特異になり`fit()`全体が`NearSingularMatrix`
/// （`ComputationError`）になったため、
/// `G>q`を満たす`G=4`に変更した（OLSの既存ガイドライン「境界の成功パスのテストでは
/// `q`を`G`以下に保つ」をTobitでも踏襲）。
#[test]
fn fit_non_classical_cov_types_work_with_bfgs_and_lbfgs() {
    let k_plus_1 = 4;
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
        "c".to_string(),
        "c".to_string(),
        "d".to_string(),
        "d".to_string(),
    ];

    for cov_type in [
        CovType::Opg,
        CovType::Hc0,
        CovType::Hc1,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
    ] {
        let newton = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: cov_type.clone(),
                confidence_level: 0.95,
            },
        )
        .unwrap();

        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let estimator = TobitEstimator::fit(
                multivariate_censored_input(),
                MleFitOptions {
                    solver,
                    max_iter: 300,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            assert!(estimator.converged(), "cov_type={cov_type:?}, {solver:?}");
            assert_cov_params_close(estimator.cov_params(), newton.cov_params(), k_plus_1, 1e-4);
        }
    }
}

/// `multivariate_censored_input`ではなく`censored_regression_input`（傾き係数
/// `q=1`、`x1`のみ）を使う。Wald検定が`fit()`に常時組み込まれたことで、
/// クラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`の構造的な制約（`rank(Ŝ)≤G`、
/// `engine/src/linear/CLAUDE.md`「クラスター数`G`と傾き係数の数`q`の関係」参照）が
/// Wald検定の`q×q`部分行列にも及ぶことが判明した。`multivariate_censored_input`
/// （`q=2`）に対し`G=2`（G=2の境界値）を組み合わせると`q`と
/// 同数になり、実測でこの部分行列が特異になり`fit()`全体が`NearSingularMatrix`
/// （`ComputationError`）になった。`q=1`のデータセットなら`G=2>q=1`を満たしたまま「G=2の境界値」を検証できる
/// ため、こちらに切り替えた（OLSの既存ガイドライン「境界の成功パスのテストでは`q`を
/// `G`以下に保つ」をTobitでも踏襲）。
#[test]
fn fit_cov_type_cluster_matches_independently_recomputed_values() {
    let k_plus_1 = 3;
    let n = 8;
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

    let classical = TobitEstimator::fit(
        censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let groups_for_expected = groups.clone();
    let expected_cluster = expected_cov_params(&censored_regression_input(), &classical, |h, s| {
        cluster_cov_params(
            h,
            s,
            n,
            k_plus_1,
            &crate::shared::group_codes::GroupCodes::from_labels(&groups_for_expected),
        )
        .unwrap()
    });

    let estimator = TobitEstimator::fit(
        censored_regression_input(),
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
    )
    .unwrap();
    assert_cov_params_close(estimator.cov_params(), &expected_cluster, k_plus_1, 1e-6);
}

/// 上のテストは4:4の均等サイズのグループのみを検証しているが、
/// `testing-policy.md`が指摘する通り均等サイズのみのテストは実務で起こりやすい
/// 偏った分布のグループサイズを見逃しうる。5:3の不均衡なグループ（G=2の境界値を
/// 含む）でも同じ独立再計算の
/// 技法で検証する（`fit_cov_type_cluster_matches_independently_recomputed_values`と
/// 同じデータセット・同じ理由でq=1のデータセットを使う、グループ分割のみ変更）。
#[test]
fn fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups() {
    let k_plus_1 = 3;
    let n = 8;
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
        "b".to_string(),
    ];

    let classical = TobitEstimator::fit(
        censored_regression_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 100,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    let groups_for_expected = groups.clone();
    let expected_cluster = expected_cov_params(&censored_regression_input(), &classical, |h, s| {
        cluster_cov_params(
            h,
            s,
            n,
            k_plus_1,
            &crate::shared::group_codes::GroupCodes::from_labels(&groups_for_expected),
        )
        .unwrap()
    });

    let estimator = TobitEstimator::fit(
        censored_regression_input(),
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
    )
    .unwrap();
    assert_cov_params_close(estimator.cov_params(), &expected_cluster, k_plus_1, 1e-6);
}
