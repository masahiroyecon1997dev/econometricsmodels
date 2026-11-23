//! LMラダーの1ステップ（`regularized_newton_step`・`predicted_decrease`）のテスト。

use crate::nonlinear::common::newton::MAX_LM_ATTEMPTS;
use crate::nonlinear::common::newton::NEWTON_COST_RESOLUTION_FACTOR;
use crate::nonlinear::common::newton::RegularizedStep;
use crate::nonlinear::common::newton::predicted_decrease_is_below_cost_resolution;
use crate::nonlinear::common::newton::regularized_newton_step;
use argmin::core::CostFunction;
use argmin::core::Error as OptimizerError;
use argmin::core::Problem;
use std::cell::Cell;

/// `regularized_newton_step`を直接検証するための、`cost`が常に同じ値を返す問題
/// （どのステップもコストを狭義に減少させない＝LMラダーは必ず全段失敗する）。
/// `evaluations`は`cost`の呼び出し回数。
struct ConstantCostProblem {
    cost: f64,
    evaluations: Cell<u64>,
}

impl CostFunction for ConstantCostProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, _param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        self.evaluations.set(self.evaluations.get() + 1);
        Ok(self.cost)
    }
}

fn constant_cost_problem(cost: f64) -> Problem<ConstantCostProblem> {
    Problem::new(ConstantCostProblem {
        cost,
        evaluations: Cell::new(0),
    })
}

fn constant_cost_evaluations(problem: &mut Problem<ConstantCostProblem>) -> u64 {
    problem.take_problem().map_or(0, |p| p.evaluations.get())
}

/// Hessianが正定値で予測減少量`½gᵀH⁻¹g`（ここでは`½·2·(1e-3)²/1e6=1e-12`）が
/// `NEWTON_COST_RESOLUTION_FACTOR·ε·|cost|`（`≈2.2e-10`）以下なら、costを1回も評価せずに
/// raw Newtonステップ`θ - H⁻¹g`を`BelowCostResolution`で返す。
#[test]
fn regularized_newton_step_returns_raw_step_without_cost_comparison_below_resolution() {
    let mut problem = constant_cost_problem(1.0e5);
    let param = [2.0, -1.0];
    let grad = [1.0e-3, -1.0e-3];
    let hessian = vec![vec![1.0e6, 0.0], vec![0.0, 1.0e6]];

    let step = regularized_newton_step(&mut problem, &param, &grad, &hessian, 1.0e5).unwrap();

    let RegularizedStep::BelowCostResolution(candidate) = step else {
        panic!("expected BelowCostResolution");
    };
    assert!(
        (candidate[0] - (2.0 - 1.0e-9)).abs() < 1e-15,
        "{candidate:?}"
    );
    assert!(
        (candidate[1] - (-1.0 + 1.0e-9)).abs() < 1e-15,
        "{candidate:?}"
    );
    assert_eq!(constant_cost_evaluations(&mut problem), 0);
}

/// 予測減少量が分解能以下でも、Hessianが不定符号（鞍点）ならコスト比較を省かず、従来どおり
/// LMラダーで検証する（ここではコストが一定のため全段失敗して`NoProgress`になる）。
#[test]
fn regularized_newton_step_keeps_ladder_for_indefinite_hessian_below_resolution() {
    let mut problem = constant_cost_problem(1.0e5);
    let hessian = vec![vec![1.0e6, 0.0], vec![0.0, -1.0e6]];

    let step = regularized_newton_step(&mut problem, &[0.0, 0.0], &[1.0e-3, 0.0], &hessian, 1.0e5)
        .unwrap();

    assert!(matches!(step, RegularizedStep::NoProgress(_)));
    assert_eq!(
        constant_cost_evaluations(&mut problem),
        MAX_LM_ATTEMPTS as u64
    );
}

/// 予測減少量が分解能を上回る（ここでは`½·1²/1=0.5`）ならコスト比較が意味を持つため、
/// Hessianが正定値でも従来どおりLMラダーで検証する。
#[test]
fn regularized_newton_step_keeps_ladder_when_predicted_decrease_exceeds_resolution() {
    let mut problem = constant_cost_problem(1.0e5);
    let hessian = vec![vec![1.0]];

    let step = regularized_newton_step(&mut problem, &[0.0], &[1.0], &hessian, 1.0e5).unwrap();

    assert!(matches!(step, RegularizedStep::NoProgress(_)));
    assert_eq!(
        constant_cost_evaluations(&mut problem),
        MAX_LM_ATTEMPTS as u64
    );
}

#[test]
fn predicted_decrease_is_below_cost_resolution_compares_half_gtd_with_scaled_epsilon() {
    let cost = 1.0e5;
    let resolution = NEWTON_COST_RESOLUTION_FACTOR * f64::EPSILON * cost;
    // `½gᵀΔθ = ½·2·resolution = resolution`（境界上は分解能以下とみなす）。
    assert!(predicted_decrease_is_below_cost_resolution(
        &[1.0],
        &[2.0 * resolution],
        cost
    ));
    assert!(!predicted_decrease_is_below_cost_resolution(
        &[1.0],
        &[4.0 * resolution],
        cost
    ));
    // コストの符号によらず`|cost|`で比較する（負の対数尤度は負にもなりうる）。
    assert!(predicted_decrease_is_below_cost_resolution(
        &[1.0],
        &[2.0 * resolution],
        -cost
    ));
    // 非有限のコストでは分解能を定義できないため常に偽（LMラダーに任せる）。
    assert!(!predicted_decrease_is_below_cost_resolution(
        &[1.0],
        &[0.0],
        f64::INFINITY
    ));
    assert!(!predicted_decrease_is_below_cost_resolution(
        &[1.0],
        &[0.0],
        f64::NAN
    ));
}

/// 予測減少量が負（Hessianは正定値だが求解の丸め誤差で上り方向になった場合の模擬）なら、
/// 分解能以下でも検証なしに採らずLMラダーに回す。
#[test]
fn predicted_decrease_is_below_cost_resolution_is_false_for_negative_decrease() {
    assert!(!predicted_decrease_is_below_cost_resolution(
        &[1.0],
        &[-1.0e-20],
        1.0e5
    ));
}
