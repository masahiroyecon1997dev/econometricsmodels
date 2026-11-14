//! コストの下限・停滞での収束判定のテスト。

use crate::nonlinear::common::error::MleError;
use crate::nonlinear::common::numerics::l2_norm;
use crate::nonlinear::common::options::SolverType;
use crate::nonlinear::common::solver::run_solver;
use crate::nonlinear::common::test_support::*;
use argmin::core::CostFunction;
use argmin::core::Error as OptimizerError;
use argmin::core::Gradient;
use argmin::core::Hessian;
use std::cell::Cell;

/// 大標本のNewton収束判定で問題になる状況を模した問題。コスト関数は`θ = target`で最小になる素直な2次関数
/// だが、(1) 大きな定数オフセットによりコストのULPが粗く（`≈1.5e-11`）、`target`
/// 近傍ではコストがそれ以上減少しない浮動小数点の底に達する。(2) 勾配は`target`
/// 近傍でも`grad_floor`（`> tol`）で下げ止まる（勾配の評価誤差の床が
/// `tol`を上回る状況の模擬）。Hessianは常に正定値（`H_DIAG`）で、生のNewtonステップ
/// `grad_floor / H_DIAG`はパラメータのスケールに対して無視できる。
#[derive(Clone)]
struct FloatingPointFloorProblem {
    target: f64,
    /// 勾配の下げ止まり値（`gradient`が返す絶対値の下限）。
    grad_floor: f64,
}

impl FloatingPointFloorProblem {
    const COST_OFFSET: f64 = 1.0e5;
    const H_DIAG: f64 = 1.0e6;
}

impl CostFunction for FloatingPointFloorProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        let d = param[0] - self.target;
        Ok(Self::COST_OFFSET + 0.5 * Self::H_DIAG * d * d)
    }
}

impl Gradient for FloatingPointFloorProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        let raw = Self::H_DIAG * (param[0] - self.target);
        let floored = if raw.abs() < self.grad_floor {
            self.grad_floor.copysign(if raw == 0.0 { 1.0 } else { raw })
        } else {
            raw
        };
        Ok(vec![floored])
    }
}

impl Hessian for FloatingPointFloorProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        Ok(vec![vec![Self::H_DIAG]])
    }
}

/// コスト関数が平坦（どのステップでも減少しない）だが勾配ノルムが大きい問題。
/// `λ=0`のHessianは可逆なので`regularized_newton_step`は`RegularizedStep::NoProgress`
/// を返すが、勾配ノルムが`NEWTON_STALL_GRAD_FACTOR * tol`を大きく超えるため
/// `next_iter`は`stalled_at_optimum`を立てない（上記の最適点停滞判定のガード条件その2:
/// 最適点から遠い場所での停滞を収束と誤判定しない）。結果は`NonConvergence`。
#[derive(Clone)]
struct FlatCostLargeGradientProblem;

impl CostFunction for FlatCostLargeGradientProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, _param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        Ok(0.0)
    }
}

impl Gradient for FlatCostLargeGradientProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, _param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        Ok(vec![1.0])
    }
}

impl Hessian for FlatCostLargeGradientProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        Ok(vec![vec![1.0]])
    }
}

/// `λ=0`のHessianは可逆だが**不定符号**（鞍点）で、勾配ノルムは小さく（条件(1)(2)は
/// 満たす）コスト関数は平坦。`regularized_newton_step`は`RegularizedStep::NoProgress`を
/// 返すが、`next_iter`のコストHessian正定値チェック（条件(3)、`cost_hessian_is_
/// positive_definite`）が偽になるため`stalled_at_optimum`を立てない（上記の最適点停滞判定の
/// 2階条件ガード）。結果は`NonConvergence`——鞍点を「収束」として
/// 黙って推定値を返さない。
#[derive(Clone)]
struct IndefiniteHessianStallProblem;

impl CostFunction for IndefiniteHessianStallProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, _param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        Ok(0.0)
    }
}

impl Gradient for IndefiniteHessianStallProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, _param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        // `tol=1e-6 < 5e-5 < NEWTON_STALL_GRAD_FACTOR·tol = 1e-2`（条件(2)を満たす）。
        Ok(vec![5.0e-5, 0.0])
    }
}

impl Hessian for IndefiniteHessianStallProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        // 可逆（`det = -1`）だが不定符号（固有値 `+1`, `-1`）。`col_piv_qr`は可逆と
        // 判定するが`llt`（Cholesky）は失敗する。
        Ok(vec![vec![1.0, 0.0], vec![0.0, -1.0]])
    }
}

/// 収束点近傍でコストの評価値に丸め誤差程度のノイズが乗る問題（大標本の対数尤度の総和の
/// 模擬）。コストは`COST_OFFSET + ½·H_TRUE·‖θ-target‖²`に`NoisyCostFloorProblem::jitter`
/// （振幅`1e-10`。`NEWTON_COST_RESOLUTION_FACTOR·ε·COST_OFFSET≈2.2e-10`未満）を加えたもの、
/// 勾配はノイズ無しで正確。`hessian`は真の値の`HESSIAN_SCALE`倍を返し、Newtonステップの
/// 誤差の縮小を毎反復`1-1/HESSIAN_SCALE`倍の1次収束にする。これにより、予測減少量が
/// コストのノイズを下回ってからも勾配が`tol`に届くまで複数反復が必要になり、コスト比較で
/// ステップを検証する実装ではLMラダーの空回りが毎反復起きる。
/// `cost_evaluations`は`cost`の呼び出し回数（`run_solver`が所有権を取るため`Rc`で共有）。
#[derive(Clone)]
struct NoisyCostExactGradientProblem {
    target: Vec<f64>,
    cost_evaluations: std::rc::Rc<Cell<u64>>,
}

impl NoisyCostExactGradientProblem {
    const COST_OFFSET: f64 = 1.0e5;
    const H_TRUE: f64 = 1.0e6;
    const HESSIAN_SCALE: f64 = 1.5;

    fn new() -> Self {
        Self {
            target: vec![2.0, -1.0],
            cost_evaluations: std::rc::Rc::new(Cell::new(0)),
        }
    }
}

impl CostFunction for NoisyCostExactGradientProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        self.cost_evaluations.set(self.cost_evaluations.get() + 1);
        let quadratic: f64 = param
            .iter()
            .zip(self.target.iter())
            .map(|(p, t)| 0.5 * Self::H_TRUE * (p - t).powi(2))
            .sum();
        Ok(Self::COST_OFFSET + quadratic + NoisyCostFloorProblem::jitter(param))
    }
}

impl Gradient for NoisyCostExactGradientProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        Ok(param
            .iter()
            .zip(self.target.iter())
            .map(|(p, t)| Self::H_TRUE * (p - t))
            .collect())
    }
}

impl Hessian for NoisyCostExactGradientProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        let h = Self::HESSIAN_SCALE * Self::H_TRUE;
        Ok(vec![vec![h, 0.0], vec![0.0, h]])
    }
}

/// コストが一定値`0`（分解能の閾値も`0`になり`BelowCostResolution`に入らない）、Hessianは
/// 正定値、勾配は`tol < GRAD < NEWTON_STALL_GRAD_FACTOR·tol`で下げ止まる問題。
/// LMラダーが全段失敗して`NoProgress`になり、停滞3条件がそろうため`stalled_at_optimum`
/// 経由で**収束**する（`BelowCostResolution`の導入後は`FloatingPointFloorProblem`の
/// テストがこの経路を通らなくなったため、`NoProgress`経由の停滞収束を単独で押さえる）。
#[derive(Clone)]
struct FlatCostGradientFloorProblem;

impl FlatCostGradientFloorProblem {
    const GRAD: f64 = 5.0e-5;
}

impl CostFunction for FlatCostGradientFloorProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, _param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        Ok(0.0)
    }
}

impl Gradient for FlatCostGradientFloorProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, _param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        Ok(vec![Self::GRAD])
    }
}

impl Hessian for FlatCostGradientFloorProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        Ok(vec![vec![1.0]])
    }
}

/// 大標本Newton収束判定の回帰テスト: 勾配ノルムが`tol`の床（`grad_floor > tol`）で下げ止まり、
/// かつコスト関数が浮動小数点の底に達する問題で、`FaerNewton`が`SingularHessian`にも
/// `NonConvergence`にもならず**収束**する（予測減少量`≈1.25e-15`がコストの分解能
/// `≈2.2e-10`以下のため`regularized_newton_step`が`RegularizedStep::BelowCostResolution`を
/// 返し、`next_iter`が勾配の停滞を確認して`stalled_at_optimum`を立てる経路。LMラダー全段
/// 失敗の`NoProgress`経由の停滞収束は
/// `run_solver_newton_converges_via_stall_after_lm_ladder_exhausted`が担う）。
/// 停滞収束の導入前は`regularized_newton_step`が`MAX_LM_ATTEMPTS`回すべて失敗して
/// `Err(MleError::SingularHessian)`を返していた。
#[test]
fn run_solver_newton_converges_when_cost_hits_floating_point_floor_above_gradient_tol() {
    let output = run_solver(
        FloatingPointFloorProblem {
            target: 2.0,
            grad_floor: 5.0e-5,
        },
        SolverType::Newton,
        vec![0.0],
        50,
        1e-6,
        1,
        true,
    )
    .unwrap();

    assert!(output.converged, "{output:?}");
    assert!(
        (output.params[0] - 2.0).abs() < 1e-6,
        "params={:?}",
        output.params
    );
    // `stalled_at_optimum`経由の収束は最適点近傍に到達してから数反復以内に起きる。
    assert!(output.n_iter <= 5, "n_iter={}", output.n_iter);
}

/// `raise_on_non_convergence=false`でも、`stalled_at_optimum`経由の収束は
/// `converged=true`（`SolverConverged`）として返る（`NonConvergence`の
/// `converged=false`降格ではない）。
#[test]
fn run_solver_newton_stalled_at_optimum_reports_converged_even_without_raising() {
    let output = run_solver(
        FloatingPointFloorProblem {
            target: -1.5,
            grad_floor: 5.0e-5,
        },
        SolverType::Newton,
        vec![10.0],
        50,
        1e-6,
        1,
        false,
    )
    .unwrap();

    assert!(output.converged, "{output:?}");
    assert!(
        (output.params[0] - (-1.5)).abs() < 1e-6,
        "params={:?}",
        output.params
    );
}

#[test]
fn run_solver_newton_does_not_treat_flat_cost_with_large_gradient_as_converged() {
    let result = run_solver(
        FlatCostLargeGradientProblem,
        SolverType::Newton,
        vec![0.0],
        20,
        1e-6,
        1,
        true,
    );

    assert!(
        matches!(result, Err(MleError::NonConvergence { .. })),
        "{result:?}"
    );
}

#[test]
fn run_solver_newton_does_not_treat_indefinite_hessian_stall_as_converged() {
    let result = run_solver(
        IndefiniteHessianStallProblem,
        SolverType::Newton,
        vec![0.0, 0.0],
        20,
        1e-6,
        1,
        true,
    );

    assert!(
        matches!(result, Err(MleError::NonConvergence { .. })),
        "{result:?}"
    );
}

/// 予測減少量がコストの分解能を下回った後は、コスト比較とLMラダーを省いてraw Newton
/// ステップを採り、勾配基準（`tol`）で**収束**する。cost評価は1反復あたりほぼ1回
/// （`next_iter`冒頭の現在点の評価）に収まる。`BelowCostResolution`の分岐を無効化すると、
/// ノイズに埋もれたコスト比較でraw ステップが棄却されLMラダーが空回りし、cost評価が
/// 150回超に膨れる（実測: 26反復で156回）。
#[test]
fn run_solver_newton_skips_cost_comparison_below_cost_resolution() {
    let problem = NoisyCostExactGradientProblem::new();
    let cost_evaluations = std::rc::Rc::clone(&problem.cost_evaluations);
    let output = run_solver(
        problem,
        SolverType::Newton,
        vec![0.0, 0.0],
        100,
        1e-6,
        1,
        true,
    )
    .unwrap();

    assert!(output.converged, "{output:?}");
    let grad = NoisyCostExactGradientProblem::new()
        .gradient(&output.params)
        .unwrap();
    assert!(l2_norm(&grad) < 1e-6, "{grad:?}");
    assert!(
        cost_evaluations.get() <= 2 * output.n_iter as u64,
        "{} cost evaluations in {} iterations",
        cost_evaluations.get(),
        output.n_iter
    );
}

#[test]
fn run_solver_newton_converges_via_stall_after_lm_ladder_exhausted() {
    let output = run_solver(
        FlatCostGradientFloorProblem,
        SolverType::Newton,
        vec![0.0],
        20,
        1e-6,
        1,
        true,
    )
    .unwrap();

    assert!(output.converged, "{output:?}");
    // 停滞と判定した反復では生ステップを適用せず、現在点（初期値）にとどまる。
    assert_eq!(output.params, vec![0.0]);
    assert_eq!(output.n_iter, 1);
}
