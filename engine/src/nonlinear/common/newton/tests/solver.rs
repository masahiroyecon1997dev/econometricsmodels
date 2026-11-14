//! `run_solver`の収束・非収束・特異Hessianのテスト。

use crate::nonlinear::common::error::MleError;
use crate::nonlinear::common::numerics::l2_norm;
use crate::nonlinear::common::options::SolverType;
use crate::nonlinear::common::solver::run_solver;
use crate::nonlinear::common::test_support::*;
use argmin::core::CostFunction;
use argmin::core::Error as OptimizerError;
use argmin::core::Gradient;
use argmin::core::Hessian;

/// `Hessian`が常にゼロ行列（特異）を返すダミー問題。`newton_step`の特異性検出を
/// `run_solver`経由で（`FaerNewton`・`convert_optimizer_error`のダウンキャストを含めて）
/// 検証する。
#[derive(Clone)]
struct SingularHessianProblem;

impl CostFunction for SingularHessianProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, _param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        Ok(0.0)
    }
}

impl Gradient for SingularHessianProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, _param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        Ok(vec![1.0])
    }
}

impl Hessian for SingularHessianProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        Ok(vec![vec![0.0]])
    }
}

/// `FaerNewton`の収束判定が「実際に返却するパラメータでの勾配」を正しく見ているかを
/// 検証する回帰テスト（「更新前のパラメータ」の勾配で
/// 収束判定すると誤る）。
///
/// `|θ|<1`の領域は勾配に比べてHessianが極端に小さく、Newtonステップが大きくジャンプする。
/// ジャンプ先（`|θ|>=1`）では勾配が大きい。`converged=true`が返るなら、実際に返却された
/// `params`での勾配（このテスト内で独立に再計算する）がtol未満でなければならない
/// （旧実装はこの不変条件を満たさないケースがあった）。
#[derive(Clone)]
struct IllConditionedProblem;

impl CostFunction for IllConditionedProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, _param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        Ok(0.0)
    }
}

impl Gradient for IllConditionedProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        Ok(vec![if param[0].abs() < 1.0 { 1e-8 } else { 500.0 }])
    }
}

impl Hessian for IllConditionedProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        Ok(vec![vec![if param[0].abs() < 1.0 { 1e-12 } else { 1.0 }]])
    }
}

#[test]
fn run_solver_newton_converges_to_known_minimum() {
    let output = run_solver(
        quadratic_problem(),
        SolverType::Newton,
        vec![0.0, 0.0],
        35,
        1e-6,
        1,
        true,
    )
    .unwrap();

    assert!(output.converged);
    assert!((output.params[0] - 3.0).abs() < 1e-8, "{:?}", output.params);
    assert!(
        (output.params[1] - (-2.0)).abs() < 1e-8,
        "{:?}",
        output.params
    );
    // `QuadraticProblem::hessian`は`diag_a`（コスト関数のHessian）をそのまま返すが、
    // `SolverOutput.hessian`は`run_solver`内部で符号反転した「対数尤度そのもの」
    // 相当の値になる（`run_solver`のdocコメント「Hessianトレイトの符号規約」参照）。
    assert!((output.hessian[0][0] - (-2.0)).abs() < 1e-9);
    assert!((output.hessian[1][1] - (-5.0)).abs() < 1e-9);
    // Newtonは2次関数を1ステップで解くため、収束までの反復回数は小さいはず
    assert!(output.n_iter <= 2, "n_iter={}", output.n_iter);
}

#[test]
fn run_solver_newton_returns_non_convergence_error_when_max_iter_is_too_small() {
    let result = run_solver(
        quadratic_problem(),
        SolverType::Newton,
        vec![1000.0, -1000.0],
        0,
        1e-12,
        1,
        true,
    );

    assert!(matches!(result, Err(MleError::NonConvergence { .. })));
}

#[test]
fn run_solver_newton_returns_result_without_raising_when_raise_on_non_convergence_is_false() {
    let output = run_solver(
        quadratic_problem(),
        SolverType::Newton,
        vec![1000.0, -1000.0],
        0,
        1e-12,
        1,
        false,
    )
    .unwrap();

    assert!(!output.converged);
    assert_eq!(output.n_iter, 0);
}

#[test]
fn run_solver_newton_returns_singular_hessian_error() {
    let result = run_solver(
        SingularHessianProblem,
        SolverType::Newton,
        vec![0.0],
        35,
        1e-6,
        1,
        true,
    );

    assert!(
        matches!(result, Err(MleError::SingularHessian)),
        "{:?}",
        result
    );
}

#[test]
fn faer_newton_terminate_reflects_gradient_at_returned_params_not_previous_params() {
    // 開始点[0.5]は|θ|<1の領域（勾配1e-8 < tol、ただしHessianは1e-12とさらに小さい）。
    // 修正前の実装は「更新前のparam」の勾配で収束判定していたため、init()相当の処理が
    // 無く、この開始点でも1回next_iterを実行してから収束判定していた: 勾配1e-8/Hessian
    // 1e-12でNewtonステップが[0.5]から遠く離れた点（|θ|>=1の領域、真の勾配500）へ
    // オーバーシュートし、そこで「更新前(=[0.5])の勾配1e-8」を使って誤ってconverged=true
    // を返していた（実際に返すparamsの真の勾配は500で全く収束していない）。
    //
    // 修正後はinit()が開始点[0.5]の勾配をあらかじめstateに格納するため、最初の
    // terminate()チェック（next_iter実行前）で正しく収束と判定され、[0.5]から一歩も
    // 動かずにconverged=trueを返す。
    let output = run_solver(
        IllConditionedProblem,
        SolverType::Newton,
        vec![0.5],
        10,
        1e-6,
        1,
        false,
    )
    .unwrap();

    assert!(output.converged, "{:?}", output);
    let actual_grad = IllConditionedProblem.gradient(&output.params).unwrap();
    assert!(
        l2_norm(&actual_grad) < 1e-6,
        "converged=true was reported but the actual gradient at the returned params {:?} is {:?}",
        output.params,
        actual_grad
    );
}
