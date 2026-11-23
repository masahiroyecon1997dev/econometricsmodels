//! `fit`のソルバー・収束・非収束・分離・スケールに関するテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleError;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::common::standardize_columns;
use crate::nonlinear::logit::LogitEstimator;
use crate::nonlinear::logit::LogitInput;
use crate::nonlinear::logit::problem::LogitProblem;
use crate::nonlinear::logit::test_support::*;
use argmin::core::Hessian;

/// `beta1`（`x1`の真の係数）を指定して、ほぼ分離したデータを生成する。`beta1`が大きいほど
/// `y`が`x1`の符号だけで決まりやすくなり、`beta1=100`では`x1`の符号で全観測が完全に分類
/// できる（完全分離、`fit_returns_separation_detected_error_for_separated_data`）。
/// 疑似乱数生成は`benchmark/nonlinear/datasets.py`の`near_separation`シナリオと同じ設計
/// 思想（真のlogit DGPで生成）。小さい`beta1`（例: `20`）は、そのシナリオと同様に意図的に
/// 境界ケースに留めた、正常に収束するデータになる。
fn near_separation_input_with_beta1(beta1: f64) -> LogitInput {
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    let n = 200;
    let mut seed = 42u64;
    let beta = [0.0, beta1, 0.5];
    let x1: Vec<f64> = (0..n).map(|_| lcg(&mut seed) * 4.0 - 2.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| lcg(&mut seed) * 2.0 - 1.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let z = beta[0] + beta[1] * x1[i] + beta[2] * x2[i];
            let p = 1.0 / (1.0 + (-z).exp());
            if lcg(&mut seed) < p { 1.0 } else { 0.0 }
        })
        .collect();

    LogitInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

fn near_separation_input() -> LogitInput {
    near_separation_input_with_beta1(100.0)
}

#[test]
fn fit_newton_converges_to_closed_form_solution_for_intercept_only_model() {
    let input = intercept_only_input();
    let estimator = LogitEstimator::fit(
        input,
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

    let y_bar: f64 = 4.0 / 7.0;
    let expected = (y_bar / (1.0 - y_bar)).ln();

    assert!(estimator.converged());
    assert_eq!(estimator.params().len(), 1);
    assert!(
        (estimator.params()[0] - expected).abs() < 1e-6,
        "params={:?}, expected={}",
        estimator.params(),
        expected
    );
    // 切片のみの1次元凹関数のNewton法は数回で収束するはず
    assert!(estimator.n_iter() <= 10, "n_iter={}", estimator.n_iter());
}

/// `newton`と同じデータセット（既知の解析解を持つ切片のみモデル）で`bfgs`/`lbfgs`を
/// 実行し、いずれも同じ解析解へ収束することを検証する。
#[test]
fn fit_bfgs_and_lbfgs_converge_to_same_solution_as_newton() {
    let y_bar: f64 = 4.0 / 7.0;
    let expected = (y_bar / (1.0 - y_bar)).ln();

    for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            MleFitOptions {
                solver,
                max_iter: 100,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        assert!(estimator.converged(), "{:?}", solver);
        assert!(
            (estimator.params()[0] - expected).abs() < 1e-4,
            "solver={:?}, params={:?}, expected={}",
            solver,
            estimator.params(),
            expected
        );
    }
}

/// 切片のみモデル（`intercept_only_input`）は`x`が定数列（切片）だけのため、
/// `standardize_columns`のスケーリングが実質no-op（`stds`が全て`1.0`のまま）になり、
/// 標準化・逆標準化の往復ロジックを通らない。このテストは
/// 非自明なスケール（`std`が1から離れた値）を持つ説明変数を含むデータセットで
/// `newton`/`bfgs`/`lbfgs`を実行し、3手法が同じ解へ収束することを検証する
/// （閉じた形の解析解は存在しないため、`newton`の結果を参照値として使う
/// クロスメソッド一致検証。標準化空間でのBFGSの初期逆Hessian
/// （`identity_matrix(k)`）・`destandardize_params`が正しく機能していることの
/// 間接的な検証になる）。
#[test]
fn fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0]];
    let make_input = || {
        LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };

    let newton = LogitEstimator::fit(
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
    assert!(newton.converged());

    for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
        let estimator = LogitEstimator::fit(
            make_input(),
            MleFitOptions {
                solver,
                max_iter: 200,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        assert!(estimator.converged(), "{:?}", solver);
        for j in 0..2 {
            assert!(
                (estimator.params()[j] - newton.params()[j]).abs() < 1e-4,
                "solver={:?}, j={j}, params={:?}, newton_params={:?}",
                solver,
                estimator.params(),
                newton.params()
            );
        }
    }
}

// `max_iter`打ち切りのテストは`intercept_only_input()`を使わない: warm start
// （`ols_based_initial_params`）は切片のみモデルでは初期値がそのまま
// 厳密なMLE（`η₀=ln(ȳ/(1-ȳ))`）になり1反復以内で収束してしまうため。代わりに
// 多変量かつ収束に多反復を要する`near_separation_input_with_beta1(20.0)`を使う。
// `beta1=20`への暗黙の依存が2つある:
//   1. 収束の遅さ: `beta1`が大きいほど収束に多反復かかる。`20`は
//      `fit_converges_normally_for_mild_near_separation_data_across_all_methods`
//      （`max_iter=35`で3手法とも収束）で正常収束が保証されており、かつ warm start
//      からでも1反復では`tol=1e-12`に到底届かない。
//   2. 分離チェックとの相互作用: 分離の検出（`verify_no_separation`）は収束した場合にだけ
//      走る。`max_iter=1`では未収束のため必ず`NonConvergence`（`raise=true`）／
//      `converged=false`（`raise=false`）になる。`beta1=100`
//      （`fit_returns_separation_detected_error_for_separated_data`）は完全分離になる
//      ため、境界側の`20`である必要がある。
#[test]
fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
    let result = LogitEstimator::fit(
        near_separation_input_with_beta1(20.0),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 1,
            tol: 1e-12,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert!(
        matches!(result, Err(MleError::NonConvergence { .. })),
        "{:?}",
        result
    );
}

#[test]
fn fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false() {
    let estimator = LogitEstimator::fit(
        near_separation_input_with_beta1(20.0),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 1,
            tol: 1e-12,
            raise_on_non_convergence: false,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    assert!(!estimator.converged());
}

#[test]
fn fit_returns_separation_detected_error_for_separated_data() {
    for (solver, max_iter) in [
        (SolverType::Newton, 100),
        (SolverType::Bfgs, 100),
        (SolverType::Lbfgs, 100),
    ] {
        let result = LogitEstimator::fit(
            near_separation_input(),
            MleFitOptions {
                solver,
                max_iter,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SeparationDetected)),
            "solver={:?}, result={:?}",
            solver,
            result
        );
    }
}

#[test]
fn fit_returns_unconverged_result_for_separated_data_without_raising() {
    let estimator = LogitEstimator::fit(
        near_separation_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: false,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    assert!(!estimator.converged());
}

/// 回帰変数どうしの相関が強い（等相関`ρ=0.9`・`k=20`）データでも、`lbfgs`が既定の
/// `max_iter=35`以内に`newton`と同じ解へ収束する回帰テスト。標準化後Hessianの条件数が
/// 数百になり、`γI`を初期行列とするL-BFGSは40反復前後を要し既定を超えた。warm start点の
/// 逆Hessianを初期行列にすると4反復前後になる（centeringでは直らない、相関による悪条件）。
#[test]
fn fit_lbfgs_converges_within_default_max_iter_for_strongly_correlated_regressors() {
    let (y, columns) =
        crate::nonlinear::common::test_support::equicorrelated_binary_data(1000, 20, 0.9, 1);
    let names: Vec<String> = (1..=20).map(|j| format!("x{j}")).collect();
    let fit = |solver: SolverType| {
        let input =
            LogitInput::from_columns(&y, &columns, names.clone(), true, "y".to_string()).unwrap();
        LogitEstimator::fit(
            input,
            MleFitOptions {
                solver,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap()
    };

    let newton = fit(SolverType::Newton);
    let lbfgs = fit(SolverType::Lbfgs);

    assert!(lbfgs.converged());
    // 初期行列が`γI`だと40反復前後、warm start点の逆Hessianだと4反復前後（実測）。
    assert!(lbfgs.n_iter() <= 15, "n_iter={}", lbfgs.n_iter());
    for (a, b) in lbfgs.params().iter().zip(newton.params()) {
        assert!(
            (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
            "lbfgs={:?}, newton={:?}",
            lbfgs.params(),
            newton.params()
        );
    }
}

/// 平均が標準偏差に比べて大きい列（平均÷標準偏差が約12〜16）を持つ設計行列でも、標準化後の
/// Hessianは良条件（条件数が小さい）になる。標準化で平均を引かないと列が切片列とほぼ共線に
/// なり、条件数が数千になる（Wooldridge appleで約7,300）。ソルバーの初期化（`FaerBfgs`/
/// `FaerLbfgs`の逆Hessian）はこの悪条件を反復数の面で隠すため、反復数ではなく条件数を直接見る。
#[test]
fn standardized_hessian_is_well_conditioned_for_columns_with_large_means() {
    let (y, columns) = crate::nonlinear::common::test_support::large_means_binary_data();
    let names: Vec<String> = (1..=columns.len()).map(|j| format!("x{j}")).collect();
    let input = LogitInput::from_columns(&y, &columns, names, true, "y".to_string()).unwrap();
    let k = input.k();

    let (x_std, _) = standardize_columns(input.x(), input.has_intercept());
    let problem = LogitProblem::from_standardized(x_std, input.y().clone());
    let hessian = problem.hessian(&vec![0.0; k]).unwrap();

    let cond = crate::nonlinear::common::test_support::condition_number(&hessian);
    // 平均を引く標準化では列が互いにほぼ直交し1〜2程度、引かないと数百〜数千（実測）。
    assert!(cond < 10.0, "condition number={cond}");
}

/// `beta1=20`（分離と誤検知してはいけない境界ケース、`tests/
/// api_tests/fixtures/benchmarks/data/logit_near_separation.csv`と同じ設計思想）で
/// 3手法とも正常に収束することを固定するリグレッションテスト。
/// 分離検出の変更が既存の合格ケースを誤検知させないことを保証する。
#[test]
fn fit_converges_normally_for_mild_near_separation_data_across_all_methods() {
    for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
        let result = LogitEstimator::fit(
            near_separation_input_with_beta1(20.0),
            MleFitOptions {
                solver,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(result.is_ok(), "solver={:?}, result={:?}", solver, result);
        assert!(result.unwrap().converged(), "solver={:?}", solver);
    }
}
