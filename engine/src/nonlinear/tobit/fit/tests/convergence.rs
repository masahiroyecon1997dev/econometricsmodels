//! `fit`のソルバー・収束・非収束・分離・スケールに関するテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleError;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::tobit::TobitEstimator;
use crate::nonlinear::tobit::TobitInput;
use crate::nonlinear::tobit::test_support::*;
use crate::shared::group_codes::GroupCodes;

/// 打ち切りが重い（60観測中51観測＝85%が0で左打ち切り）合成データ。外部の乱数に依存しない
/// 線形合同法で作る（`tests/support/_nonlinear_helpers.py`の`heavily_censored_tobit_dataset`と同じ手順・
/// 同じ値。Python側の同名テストと同じ入力で挙動を固定する）。潜在変数は
/// `y* = -1.8 + x1 + 0.5·x2 + e`（`x1`・`x2`・`e`は12個の一様乱数の和−6で近似した正規乱数）。
fn heavily_censored_input() -> TobitInput {
    let mut state: u64 = 167;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut normal = move || (0..12).map(|_| unit()).sum::<f64>() - 6.0;
    let n = 60;
    // 引数の評価順（`x1`→`x2`→`e`）がPython側と同じになるよう、1つのクロージャを
    // 3回続けて回す。
    let x1: Vec<f64> = (0..n).map(|_| normal()).collect();
    let x2: Vec<f64> = (0..n).map(|_| normal()).collect();
    let e: Vec<f64> = (0..n).map(|_| normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (-1.8 + x1[i] + 0.5 * x2[i] + e[i]).max(0.0))
        .collect();
    assert_eq!(y.iter().filter(|&&v| v == 0.0).count(), 51);
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

fn heavily_censored_groups() -> Vec<String> {
    (0..60).map(|i| (i % 4).to_string()).collect()
}

fn truncated_fit_options(solver: SolverType, cov_type: CovType) -> MleFitOptions {
    MleFitOptions {
        solver,
        max_iter: 1,
        tol: 1e-12,
        raise_on_non_convergence: false,
        cov_type,
        confidence_level: 0.95,
    }
}

fn all_cov_types_for_heavily_censored_input() -> Vec<CovType> {
    let groups = heavily_censored_groups();
    vec![
        CovType::Classical,
        CovType::Opg,
        CovType::Hc0,
        CovType::Hc1,
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
    ]
}

/// 切片のみ（説明変数なし）・打ち切りなしのTobitは、通常の正規分布の最尤推定
/// （`β̂=ȳ`・`σ̂²=Σ(y-ȳ)²/n`という閉じた形の解析解、OLSの不偏推定量`n-1`除算とは
/// 異なる`n`除算のML推定量）に一致するはず（「打ち切りが極端に
/// 少ないデータで、Newton法がOLSの閉形式解に近い値に収束すること」という完了条件の境界ケース）。
#[test]
fn fit_newton_converges_to_closed_form_solution_for_intercept_only_uncensored_data() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
    let input = intercept_only_uncensored_input(&y);

    let n = y.len() as f64;
    let y_bar: f64 = y.iter().sum::<f64>() / n;
    let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
    let expected_sigma = (sse / n).sqrt();

    let estimator = TobitEstimator::fit(
        input,
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

    assert!(estimator.converged());
    assert_eq!(estimator.params().len(), 1);
    assert!(
        (estimator.params()[0] - y_bar).abs() < 1e-5,
        "params={:?}, expected={}",
        estimator.params(),
        y_bar
    );
    assert!(
        (estimator.sigma() - expected_sigma).abs() < 1e-5,
        "sigma={}, expected={}",
        estimator.sigma(),
        expected_sigma
    );
}

/// 説明変数ありのモデルでも、打ち切りが実質発生しないデータではNewton法がOLSの
/// 閉じた形の解（正規方程式）に近い値に収束するはず（完了条件の本体、
/// `expected_*`はOLSの公式から本テスト内で独立に計算する）。
#[test]
fn fit_newton_converges_near_ols_closed_form_when_censoring_is_negligible() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
    let input = TobitInput::from_columns(
        &y,
        std::slice::from_ref(&x),
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        Some(-1000.0),
        Some(1000.0),
    )
    .unwrap();

    let n = y.len() as f64;
    let x_bar: f64 = x.iter().sum::<f64>() / n;
    let y_bar: f64 = y.iter().sum::<f64>() / n;
    let sxy: f64 = x
        .iter()
        .zip(&y)
        .map(|(xi, yi)| (xi - x_bar) * (yi - y_bar))
        .sum();
    let sxx: f64 = x.iter().map(|xi| (xi - x_bar).powi(2)).sum();
    let expected_slope = sxy / sxx;
    let expected_intercept = y_bar - expected_slope * x_bar;
    let expected_sse: f64 = x
        .iter()
        .zip(&y)
        .map(|(xi, yi)| (yi - (expected_intercept + expected_slope * xi)).powi(2))
        .sum();
    let expected_sigma = (expected_sse / n).sqrt();

    let estimator = TobitEstimator::fit(
        input,
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

    assert!(estimator.converged());
    assert!(
        (estimator.params()[0] - expected_intercept).abs() < 1e-4,
        "intercept={}, expected={}",
        estimator.params()[0],
        expected_intercept
    );
    assert!(
        (estimator.params()[1] - expected_slope).abs() < 1e-4,
        "slope={}, expected={}",
        estimator.params()[1],
        expected_slope
    );
    assert!(
        (estimator.sigma() - expected_sigma).abs() < 1e-4,
        "sigma={}, expected={}",
        estimator.sigma(),
        expected_sigma
    );
}

#[test]
fn fit_newton_converges_for_data_with_actual_censoring() {
    let input = censored_regression_input();
    let estimator = TobitEstimator::fit(
        input,
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

    assert!(estimator.converged());
    assert_eq!(estimator.params().len(), 2);
    assert!(estimator.sigma() > 0.0 && estimator.sigma().is_finite());
    for &p in estimator.params() {
        assert!(p.is_finite());
    }
    // `n_iter()`が他のどのテストでも未使用だった（`cargo llvm-cov`で発覚）。
    // 収束時は`0 < n_iter <= max_iter`のはず。
    assert!(estimator.n_iter() > 0 && estimator.n_iter() <= 100);
}

#[test]
fn fit_bfgs_and_lbfgs_converge_to_similar_solution_as_newton_for_censored_data() {
    let newton = TobitEstimator::fit(
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

    for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
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
        assert!(estimator.converged(), "solver={:?}", solver);
        for (a, b) in estimator.params().iter().zip(newton.params()) {
            assert!((a - b).abs() < 1e-3, "solver={:?}, a={a}, b={b}", solver);
        }
        assert!(
            (estimator.sigma() - newton.sigma()).abs() < 1e-3,
            "solver={:?}, sigma={}, newton_sigma={}",
            solver,
            estimator.sigma(),
            newton.sigma()
        );
    }
}

/// `censored_regression_input`の`x`（`1..8`）は`TobitScaling`の列標準化が実質no-opに
/// なるほど自明なスケールではないが、`fit_bfgs_and_lbfgs_agree_with_newton_when_design_
/// matrix_has_nontrivial_scale`（`logit/fit/tests/convergence.rs`/`probit/fit/tests/convergence.rs`）と同じ観点で、桁が大きく離れた
/// スケール（`std`が1から大きく離れた値）でも標準化・逆標準化の往復が壊れないことを
/// 明示的に検証する。`x`を100倍しつつ係数を1/100にスケールして同じ潜在変数
/// `y*=-5+2x+noise`を再現しているため、期待される`β`・`σ`は`censored_regression_input`と
/// 同一だが、ここではLogit/Probitの当該テストと同じくNewtonの結果を参照値とする
/// クロスメソッド一致検証として書く（標準化空間での差異のみを見るのが目的のため）。
#[test]
fn fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale() {
    let x = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
    let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
    let make_input = || {
        TobitInput::from_columns(
            &y,
            std::slice::from_ref(&x),
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap()
    };

    let newton = TobitEstimator::fit(
        make_input(),
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
    assert!(newton.converged());

    for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
        let estimator = TobitEstimator::fit(
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
        assert!(estimator.converged(), "solver={:?}", solver);
        for (a, b) in estimator.params().iter().zip(newton.params()) {
            assert!((a - b).abs() < 1e-3, "solver={:?}, a={a}, b={b}", solver);
        }
        assert!(
            (estimator.sigma() - newton.sigma()).abs() < 1e-3,
            "solver={:?}, sigma={}, newton_sigma={}",
            solver,
            estimator.sigma(),
            newton.sigma()
        );
    }
}

/// `y`（および打ち切り境界）を大きなスケールに引き伸ばしても、`fit()`が生スケールの
/// データと数学的に同値な推定値に収束することを確認する回帰テスト。
///
/// `y`を標準化せず最適化していた頃は、`y`のスケールが大きいと健全なMLE解でも標準化
/// パラメータ空間のL2ノルムが大きくなり、当時あった標準化パラメータのノルムによる
/// (準)分離ヒューリスティックが誤発火して失敗していた（Wooldridge mroz `hours`の
/// 生スケールTobitで発覚）。`TobitScaling`が`y`を2の冪に丸めたスケールでスケーリング
/// してから最適化し収束後に逆変換することで解消した（本テストは`TobitScaling`による
/// スケール同値性の回帰テストとして維持する）。
///
/// `y' = c·y`・`lower' = c·lower`（`c`は2の冪）とすると`β' = c·β`・`σ' = c·σ`・
/// `SE' = c·SE`が厳密に成り立ち（`TobitScaling`が`y_scale`も`c`倍の2の冪に丸めるため、
/// 両者は完全に同一の標準化問題を解く）、z統計量・p値・Wald統計量はスケール不変。
#[test]
fn fit_converges_for_large_response_scale_without_false_separation() {
    let c = 1024.0;
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_base = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
    let y_scaled: Vec<f64> = y_base.iter().map(|v| v * c).collect();

    let make = |y: &[f64], lower: f64| {
        TobitInput::from_columns(
            y,
            std::slice::from_ref(&x),
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(lower),
            None,
        )
        .unwrap()
    };
    let fit = |y: &[f64], lower: f64| {
        TobitEstimator::fit(
            make(y, lower),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap()
    };

    let base = fit(&y_base, 0.0);
    let scaled = fit(&y_scaled, 0.0);

    assert!(scaled.converged());

    let rtol = |a: f64, b: f64| (a - b).abs() <= 1e-8 * b.abs().max(1.0);
    for (b, s) in base.params().iter().zip(scaled.params()) {
        assert!(rtol(s / c, *b), "beta: base={b}, scaled/c={}", s / c);
    }
    assert!(
        rtol(scaled.sigma() / c, base.sigma()),
        "sigma: base={}, scaled/c={}",
        base.sigma(),
        scaled.sigma() / c
    );
    for (b, s) in base.std_errors().iter().zip(scaled.std_errors()) {
        assert!(rtol(s / c, *b), "se: base={b}, scaled/c={}", s / c);
    }
    for (b, s) in base.test_stats().iter().zip(scaled.test_stats()) {
        assert!(rtol(*s, *b), "z: base={b}, scaled={s}");
    }
    assert!(
        rtol(scaled.wald_statistic(), base.wald_statistic()),
        "wald: base={}, scaled={}",
        base.wald_statistic(),
        scaled.wald_statistic()
    );
    // 対数尤度は変数変換のヤコビアン分ずれる（非打ち切り観測ごとに `-ln c`）。
    let n_uncensored = y_base.iter().filter(|&&v| v != 0.0).count() as f64;
    assert!(
        rtol(
            scaled.log_likelihood() + n_uncensored * c.ln(),
            base.log_likelihood()
        ),
        "llf: base={}, scaled+adj={}",
        base.log_likelihood(),
        scaled.log_likelihood() + n_uncensored * c.ln()
    );
}

/// 上のテストは切片あり・`c`が2の冪（＝標準化問題が両者で完全一致）だが、こちらは
/// (a) **切片なし**モデル、(b) **`y`の標準偏差が2の冪ちょうどではない**大スケール、で
/// `fit()`が既定Newtonで収束し真の傾き（生成係数）を回復することを確認する
/// （`TobitScaling`の切片なし経路の end-to-end カバレッジと、`y_scale`が恒等変換で
/// ない一般ケースの分離ヒューリスティック誤発火回避のリグレッションガード）。
#[test]
fn fit_converges_for_large_non_power_of_two_response_scale_without_intercept() {
    // 潜在変数 y* = 40·x（切片なし）、生成 σ ≈ 300 → y の std は数百オーダーで
    // 2の冪（256 / 512）ちょうどにはならない。左打ち切り 0。
    let x: Vec<f64> = (1..=40).map(|i| i as f64).collect();
    let mut rng_state = 0x1234_5678_u64;
    let mut next = || {
        // 決定的な xorshift でノイズを作る（テストの再現性、外部 rng 非依存）。
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 600.0
    };
    let y: Vec<f64> = x.iter().map(|&xi| (40.0 * xi + next()).max(0.0)).collect();
    let input = TobitInput::from_columns(
        &y,
        std::slice::from_ref(&x),
        vec!["x1".to_string()],
        false,
        "y".to_string(),
        Some(0.0),
        None,
    )
    .unwrap();

    let est = TobitEstimator::fit(
        input,
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

    assert!(est.converged());
    // 傾きは真値 40 の近傍（打ち切り＋ノイズがあるため緩め）。以前は`y`の
    // 大スケールで当時のノルムによる分離ヒューリスティックが誤発火し`.unwrap()`が
    // panicしていた（現在はこのヒューリスティック自体が無い）。
    assert!(
        (est.params()[0] - 40.0).abs() < 5.0,
        "slope={}, expected≈40",
        est.params()[0]
    );
    assert!(est.sigma() > 0.0 && est.sigma().is_finite());
}

#[test]
fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
    let input = censored_regression_input();
    let result = TobitEstimator::fit(
        input,
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

/// Tobitの(準)完全分離は`MleError::SeparationDetected`ではなく
/// `MleError::NonConvergence`として現れることを固定する。
///
/// `fit()`はLogit/Probit用の分離検出（`nonlinear::separation`）を呼ばない。Tobitの分離は
/// 係数の発散ではなく`σ→0`への退化として現れるため、この失敗モードが`NonConvergence`で
/// あること自体を、Tobitの分離の現れ方の正本として固定する（`SeparationDetected`と
/// `NonConvergence`はどちらも`ComputationError`にマップされるため、Python API境界の
/// テストでは区別できない）。
///
/// DGP: `y* = 100·x1 + 0.5·x2`（ノイズ皆無）を`lower=0`で左打ち切り。非打ち切り
/// 観測は存在する（`NoUncensoredObservations`は通過する）が、分離方向に沿って
/// `σ→0`へ退化し、Newtonは`max_iter`まで収束しない。
#[test]
fn fit_returns_non_convergence_not_separation_detected_for_noise_free_separation() {
    let n = 120;
    let x1: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * (i as f64) / ((n - 1) as f64))
        .collect();
    let x2: Vec<f64> = (0..n).map(|i| (0.7 * i as f64).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (100.0 * x1[i] + 0.5 * x2[i]).max(0.0))
        .collect();
    // 非打ち切り観測が1件以上あることを前提とする（無ければ
    // `NoUncensoredObservations`が先に返り、このテストの意図がずれる）。
    assert!(y.iter().any(|&v| v > 0.0));

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
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    );
    assert!(
        matches!(result, Err(MleError::NonConvergence { .. })),
        "expected NonConvergence, got {result:?}"
    );
}

#[test]
fn fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false() {
    // `max_iter=1`（このテストの元々の値）だと、`censored_regression_input`の
    // 打ち切り点（Newtonの初回ステップ、まだ真の最尤推定点から遠い）でHessianが
    // 不定符号になり、`fit`が非収束時でも`cov_params`を計算するようになったことで
    // `SingularHessian`が先に発生してしまう（実測で確認、
    // `max_iter=1`は`SingularHessian`、`max_iter=2`以降で`cov_params`計算が
    // 安定し`converged=false`が返るようになる。真の収束は`max_iter=11`）。
    // `max_iter=3`のまま（`max_iter=2`でも通るが余裕を持たせる）、「非収束だが
    // cov_params計算は成功する」ケースを踏む。
    let input = censored_regression_input();
    let estimator = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 3,
            tol: 1e-12,
            raise_on_non_convergence: false,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();
    assert!(!estimator.converged());
}

/// `max_iter=1`・`raise_on_non_convergence=false`で打ち切った点は、重く打ち切られたデータでは
/// `(β, log σ)`尤度が凹でない領域の近くにある。`Newton`（LM減衰で凹な領域に留まる）・`Bfgs`・`Lbfgs`
/// （初期点のHessianが正定値ならその厳密な逆行列から出発するため、1手目がNewtonステップと一致）は、
/// 全`cov_type`で未収束の結果を返す（Hessianの逆行列を使う`Classical`/`Hc0`/`Hc1`/`Cluster`でも
/// `SingularHessian`にならない）。かつて`Lbfgs`は初期点のHessianを使わず1手目が勾配方向だったため、
/// このデータでは凹でない領域（費用関数Hessianの最小固有値が約-0.6）に着地し、Hessianの逆行列を使う
/// `cov_type`が`SingularHessian`になっていた。
#[test]
fn fit_returns_unconverged_result_for_every_solver_and_cov_type_at_truncated_point() {
    for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
        for cov_type in all_cov_types_for_heavily_censored_input() {
            let result = TobitEstimator::fit(
                heavily_censored_input(),
                truncated_fit_options(solver, cov_type.clone()),
            );
            let estimator = result
                .unwrap_or_else(|e| panic!("solver={solver:?}, cov_type={cov_type:?}: {e:?}"));
            assert!(
                !estimator.converged(),
                "solver={solver:?}, cov_type={cov_type:?}"
            );
        }
    }
}
