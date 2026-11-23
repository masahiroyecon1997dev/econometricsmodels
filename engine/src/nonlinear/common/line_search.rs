//! `FaerBfgs`・`FaerLbfgs`のline search（`MoreThuenteLineSearch`）の定数と、Wolfe条件・収束点近傍の停滞の判定。

use argmin::core::{Error as OptimizerError, State, TerminationReason};
use argmin::solver::linesearch::MoreThuenteLineSearch;

use super::error::MleError;
use super::numerics::{dot, l2_norm};
use crate::shared::error::CommonError;

/// `FaerLbfgs::next_iter`が内側line search用`Executor`に設定する反復回数の上限。
///
/// **背景**: argminの`Executor::run()`は`max_iters`に到達しても`Err`を返さず、
/// `TerminationStatus::Terminated(TerminationReason::MaxItersReached)`を伴う`Ok`として
/// 打ち切り時点のパラメータをそのまま返す（収束判定を満たしたわけではない不正な
/// ステップになりうる）。`BudgetedProblem`は「評価回数の総枠を使い
/// 切ったら`Err`」という複数回のline search呼び出しを横断する粗い安全網で、この種の
/// 暴走を有限時間で検出するが、「1回のline searchが収束せず打ち切られた」ことを
/// 明示的には教えてくれず、黙って不正な点を採用しうる問題は残る。このため
/// `FaerLbfgs`では内側`Executor`に明示的な`max_iters`を設定した上で、`.run()`後に
/// 収束判定（`TerminationReason::SolverConverged`）を満たしていない場合は明示的に
/// エラー化する（`FaerBfgs`には無いチェックだが、この実装では`FaerLbfgs`のみを対象とし
/// `FaerBfgs`側は変更しない）。
///
/// **値の根拠**: 正常収束するケースのline searchは数回〜数十回程度で収束する
/// （`MAX_EVALUATIONS_PER_ITER`のdocコメント参照、公式ハーネスの実測でbfgsが6〜7
/// 反復・lbfgsが17反復で収束）。`100`はこれに対して数倍〜1桁のマージンを残しつつ、
/// `BudgetedProblem`の総枠（1outer反復あたり`2000`）よりずっと小さい専用の値とし、
/// 退化したline search単体を総枠の消費を待たずに早期検出できるようにする。
pub(super) const LINE_SEARCH_MAX_ITERS: u64 = 100;

/// `FaerBfgs`/`FaerLbfgs`のline search（`MoreThuenteLineSearch`）に渡すstrong Wolfe条件の
/// 係数`c1`（十分減少条件）。argminの既定値と同じ値だが、[`line_search_step_satisfies_wolfe`]
/// の再判定と必ず同じ値を使うため、[`wolfe_line_search`]で明示的に設定する。
const LINE_SEARCH_C1: f64 = 1e-4;

/// strong Wolfe条件の係数`c2`（曲率条件）。[`LINE_SEARCH_C1`]と同じ理由で明示する。
const LINE_SEARCH_C2: f64 = 0.9;

/// `FaerBfgs`/`FaerLbfgs`で、line searchがstrong Wolfe条件を満たすステップを見つけられずに
/// 終了した（[`line_search_step_satisfies_wolfe`]が偽）とき、それを「最適点で停滞」と
/// 解釈するための条件: 勾配ノルムが実効的な収束目標`tol`（`n_obs`で正規化済みの値）の
/// この倍数未満であること（`NEWTON_STALL_GRAD_FACTOR`と同じ位置づけのガード）。
///
/// **背景**: 収束点近傍では1ステップあたりのコスト減少量（`≈‖g‖²/2H`）が、
/// コスト（`n`個の対数尤度の和）の浮動小数点の丸め誤差と同程度まで小さくなる。すると
/// line searchは十分減少条件を正しく判定できず（より良い点も誤差で棄却される）、
/// 極小ステップで評価を数十回繰り返してから異常終了する。到達可能な勾配ノルムの下限は
/// `√(2Hδ)`（`H`・丸め誤差`δ`とも`O(n)`）で`n`に比例し、実効閾値`tol·n`（既定`1e-8·n`）と
/// ほぼ同じ大きさのため、勾配ノルム基準に届くかはデータ次第になる（実測:
/// `generate_binary_choice_dataset("baseline", link="probit", n=1_000_000, k=5, seed=42)`
/// で勾配ノルムが`1.2·tol`で頭打ちになり、bfgsが26反復・cost評価587回・43秒を要した）。
/// この実測はコストを素朴な逐次和で計算していた時点のもので、当時の`δ`はコストのULPの
/// 約100倍（`n=1e6`で`≈1e-8`）あった。現在はコストを補償和（[`super::numerics::compensated_sum`]）で計算し
/// `δ`は約1 ULPまで下がっているため、頭打ちはより小さい勾配ノルムで起きる（停滞検出は
/// その安全網として引き続き必要）。
/// scipyの`fmin_bfgs`がline search失敗を「precision loss」として即座に打ち切るのと
/// 同じ発想で、この状態を検出して終了する。
///
/// **値の根拠**: 実測の頭打ち点は`tol`の約1〜3倍（Probit `n=100_000`・`1_000_000`）。
/// `1e2`はこれに十分な余裕を持たせつつ、「観測あたり平均勾配`1e-6`」相当で、statsmodelsの
/// 既定（平均勾配の最大成分`gtol=1e-5`）より厳しい。最適化初期の勾配ノルム（`tol`の
/// `1e4`倍以上）での一時的なline search失敗は従来どおり反復を継続する。
const QUASI_NEWTON_STALL_GRAD_FACTOR: f64 = 1e2;

/// `FaerBfgs`/`FaerLbfgs`で、勾配ノルムが収束目標の近傍（[`is_near_convergence_target`]）に
/// あるときのline searchの反復上限。上限に達したら（`TerminationReason::MaxItersReached`）
/// 打ち切り時点の試行点は採用せず、直前の反復点で「最適点で停滞」として終了する。
///
/// **背景**: [`QUASI_NEWTON_STALL_GRAD_FACTOR`]のdocコメントの状況では、line searchは
/// 区間幅が`xtol`（argmin既定`1e-10`）まで縮むまで諦めないため、異常終了までに
/// 1回で約45回（cost＋勾配で約90回）の評価を消費していた（実測: Probit
/// `n=1_000_000`で停滞検出後もbfgs 4.8秒、statsmodelsの約5倍）。収束目標の近傍では
/// 準Newton方向の`α=1`がほぼそのまま受理される（実測で1〜3回）ため、それを大きく
/// 超える試行は失敗の兆候とみなしてよい。
///
/// **値の根拠**: `5`と`10`で精度・既存テスト結果に差が無く（実行時間の差も計測の
/// ばらつきの範囲内）、正常なline searchを誤って打ち切る余地が小さい`10`を採った
/// （Probit `n=1_000_000`でbfgs 43秒→約2秒・lbfgs 47秒→約2秒）。近傍以外の反復には
/// 適用しない（`FaerBfgs`は上限無し、`FaerLbfgs`は[`LINE_SEARCH_MAX_ITERS`]のまま）。
pub(super) const NEAR_TARGET_LINE_SEARCH_MAX_ITERS: u64 = 10;

/// 勾配ノルムが実効的な収束目標`tol`の[`QUASI_NEWTON_STALL_GRAD_FACTOR`]倍未満か
/// （`FaerBfgs`/`FaerLbfgs`の停滞判定が適用される範囲）。
pub(super) fn is_near_convergence_target(grad: &[f64], tol: f64) -> bool {
    l2_norm(grad) < QUASI_NEWTON_STALL_GRAD_FACTOR * tol
}

/// 内側line searchの`Executor`が`TerminationReason::SolverConverged`で終了したか。
/// argminの`MoreThuenteLineSearch`は異常終了もこの理由で返すため、条件を満たしたかどうかは
/// 別途[`line_search_step_satisfies_wolfe`]で確かめる必要がある。
pub(super) fn line_search_converged<S: State>(state: &S) -> bool {
    matches!(
        state.get_termination_reason(),
        Some(TerminationReason::SolverConverged)
    )
}

/// [`LINE_SEARCH_C1`]/[`LINE_SEARCH_C2`]を設定した`MoreThuenteLineSearch`を構築する。
pub(super) fn wolfe_line_search() -> Result<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, MleError>
{
    MoreThuenteLineSearch::new()
        .with_c(LINE_SEARCH_C1, LINE_SEARCH_C2)
        .map_err(convert_optimizer_error)
}

/// line searchが受理したステップ`step`（`s=θ_{k+1}-θ_k`）が、strong Wolfe条件
/// （十分減少条件`f₁ ≤ f₀ + c1·g₀ᵀs`と曲率条件`|g₁ᵀs| ≤ c2·|g₀ᵀs|`）を満たすかを判定する。
///
/// argminの`MoreThuenteLineSearch`は、条件を満たして終了した場合（MINPACKの`info=1`）も、
/// 区間幅が下限に達した・丸め誤差で進めない等の異常終了（`info=2,4,5,6`）も、区別せず
/// `TerminationReason::SolverConverged`として返す。呼び出し側からは失敗を見分けられない
/// ため、受理されたステップ自体から条件を再判定する（`step`は探索方向`d`とステップ幅
/// `α`の積なので、`α>0`である限り`d`基準の判定と丸め誤差を除いて同値）。
///
/// `g₀ᵀs ≥ 0`（降下しないステップ、`s=0`を含む）は失敗とみなす。`cost0`が
/// `f64::INFINITY`（1回目の反復、`init`がコストを評価しないため）の場合、十分減少条件は
/// 常に真になり曲率条件のみで判定する。
pub(super) fn line_search_step_satisfies_wolfe(
    cost0: f64,
    grad0: &[f64],
    cost1: f64,
    grad1: &[f64],
    step: &[f64],
) -> bool {
    let slope0 = dot(grad0, step);
    slope0 < 0.0
        && cost1 <= cost0 + LINE_SEARCH_C1 * slope0
        && dot(grad1, step).abs() <= LINE_SEARCH_C2 * (-slope0)
}

/// argminの内部エラー（`anyhow::Error`）を`MleError`に変換する。
/// `FaerNewton::next_iter`内で`MleError`から`?`により変換された値は`downcast`で復元し、
/// それ以外のうち、argminのline searchが入力（探索方向・目的関数の値）に応じて返す
/// `ArgminError::ConditionViolated`/`InvalidParameter`（「降下方向でない」「NaN/Infに遭遇した」等）は
/// `LineSearchFailed`、残り（argmin自体の内部エラー等、実装の誤りでだけ起きるもの）は
/// `Internal`にする。
///
/// 前者は`bfgs`/`lbfgs`の退化した目的関数で実際に起こりうる（`MoreThuenteLineSearch`が
/// `Executor::run()`の`Err`として返す）。後者の`Internal`は、本プロジェクトが制御する全ての
/// エラー経路が`MleError`を返すため実測ではカバーされていない防御的なフォールバックで、
/// 意図的にargminの内部を破壊するテストは書かない（`.claude/rules/testing-policy.md`
/// 「engine（Rust）のカバレッジ方針」の理論上到達不能な防御的エラーパスと同じ性質）。
pub(super) fn convert_optimizer_error(e: OptimizerError) -> MleError {
    match e.downcast::<MleError>() {
        Ok(mle_error) => mle_error,
        Err(other) => match other.downcast_ref::<argmin::core::ArgminError>() {
            Some(
                argmin::core::ArgminError::ConditionViolated { .. }
                | argmin::core::ArgminError::InvalidParameter { .. },
            ) => MleError::LineSearchFailed {
                max_iters: LINE_SEARCH_MAX_ITERS,
            },
            _ => CommonError::internal(other.to_string()).into(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::super::bfgs::FaerBfgs;

    use super::super::lbfgs::FaerLbfgs;
    use super::super::lbfgs::LBFGS_HISTORY_SIZE;

    use super::super::options::SolverType;

    use super::super::solver::run_solver;
    use super::super::test_support::*;
    use super::*;

    use argmin::core::Executor;
    use argmin::core::Gradient;

    use std::collections::VecDeque;

    /// `line_search_step_satisfies_wolfe`: 十分減少条件・曲率条件の両方を満たすステップ。
    /// `f(θ)=θ²`、`θ₀=1`（`f₀=1`、`g₀=2`）から`s=-1`（`θ₁=0`、`f₁=0`、`g₁=0`）:
    /// `g₀ᵀs=-2<0`、`f₁=0 ≤ 1+1e-4·(-2)`、`|g₁ᵀs|=0 ≤ 0.9·2`。
    #[test]
    fn line_search_step_satisfies_wolfe_accepts_exact_minimizer_step() {
        assert!(line_search_step_satisfies_wolfe(
            1.0,
            &[2.0],
            0.0,
            &[0.0],
            &[-1.0]
        ));
    }

    /// 十分減少条件違反: 同じ問題で`s=-2`（`θ₁=-1`、`f₁=1`）。コストが減っていない
    /// （`1 > 1+1e-4·(-4)`）ため偽（曲率条件`|(-2)(-2)|=4 > 0.9·4=3.6`も満たさない）。
    /// 浮動小数点の底で丸め誤差によりコストが見かけ上増えた状況に相当する。
    #[test]
    fn line_search_step_satisfies_wolfe_rejects_step_without_sufficient_decrease() {
        assert!(!line_search_step_satisfies_wolfe(
            1.0,
            &[2.0],
            1.0,
            &[-2.0],
            &[-2.0]
        ));
    }

    /// 曲率条件違反: 同じ問題で`s=-0.01`（`θ₁=0.99`、`f₁=0.9801`、`g₁=1.98`）。
    /// 十分減少条件は満たすが、`|g₁ᵀs|=0.0198 > 0.9·0.02=0.018`で極小ステップのため偽
    /// （区間幅が下限に達したline searchが返す、ほとんど進まないステップに相当する）。
    #[test]
    fn line_search_step_satisfies_wolfe_rejects_tiny_step_violating_curvature_condition() {
        assert!(!line_search_step_satisfies_wolfe(
            1.0,
            &[2.0],
            0.9801,
            &[1.98],
            &[-0.01]
        ));
    }

    /// 降下しないステップ（`s=0`を含む）は、両条件が形式的に満たされても偽。
    #[test]
    fn line_search_step_satisfies_wolfe_rejects_zero_or_ascent_step() {
        assert!(!line_search_step_satisfies_wolfe(
            1.0,
            &[2.0],
            1.0,
            &[2.0],
            &[0.0]
        ));
        assert!(!line_search_step_satisfies_wolfe(
            1.0,
            &[2.0],
            4.0,
            &[4.0],
            &[1.0]
        ));
    }

    /// `cost0`が`f64::INFINITY`（1回目の反復）なら十分減少条件は常に真で、曲率条件のみで
    /// 判定される。
    #[test]
    fn line_search_step_satisfies_wolfe_uses_only_curvature_condition_when_initial_cost_is_infinite()
     {
        assert!(line_search_step_satisfies_wolfe(
            f64::INFINITY,
            &[2.0],
            1.0e300,
            &[0.0],
            &[-1.0]
        ));
        assert!(!line_search_step_satisfies_wolfe(
            f64::INFINITY,
            &[2.0],
            0.9801,
            &[1.98],
            &[-0.01]
        ));
    }

    /// bfgs/lbfgsは、勾配ノルムが床（`GRAD_FLOOR·√2≈7.1e-2`）で頭打ちになり
    /// 実効閾値`tol=1e-2`に届かなくても、line searchの異常終了を停滞として検出し、評価回数を
    /// 浪費せずに**収束**する（`stalled_at_optimum`経路）。修正前は極小ステップの
    /// line searchを繰り返して評価回数が膨れていた（Probit `n=1_000_000`で26反復・評価約1200回）。
    #[test]
    fn run_solver_quasi_newton_stops_at_noisy_cost_floor_near_gradient_target() {
        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let problem = NoisyCostFloorProblem::new();
            let evaluations = std::rc::Rc::clone(&problem.evaluations);
            let output = run_solver(problem, solver, vec![0.0, 0.0], 100, 1e-2, 1, true).unwrap();

            assert!(output.converged, "{solver:?}: {output:?}");
            let grad = NoisyCostFloorProblem::new()
                .gradient(&output.params)
                .unwrap();
            assert!(
                l2_norm(&grad) < QUASI_NEWTON_STALL_GRAD_FACTOR * 1e-2,
                "{solver:?}: {grad:?}"
            );
            assert!(
                evaluations.get() < 60,
                "{solver:?}: {} evaluations",
                evaluations.get()
            );
        }
    }

    /// Wolfe再判定経路の統合テスト: line searchが反復上限（`NEAR_TARGET_LINE_SEARCH_MAX_ITERS`）
    /// より先に異常終了し、条件を満たさないステップを`SolverConverged`で返す状況で、
    /// `line_search_step_satisfies_wolfe`による再判定だけで停滞を検出する。区間幅の許容値
    /// （`xtol`）を大きくしたline searchを直接渡し、区間が少し縮んだだけで異常終了（MINPACKの
    /// `info=2`）させて再現する（`run_solver`経由では既定`xtol=1e-10`のため先に反復上限に
    /// 達し、この経路を切り分けられない）。`stalled_at_optimum`を直接確認する。
    #[test]
    fn quasi_newton_detects_stall_via_wolfe_recheck_when_line_search_terminates_abnormally() {
        let linesearch = || {
            wolfe_line_search()
                .unwrap()
                .with_width_tolerance(0.5)
                .unwrap()
        };

        let bfgs = FaerBfgs {
            linesearch: linesearch(),
            tol: 1e-2,
            stalled_at_optimum: false,
            h0_from_hessian: false,
        };
        let result = Executor::new(NoisyCostFloorProblem::new(), bfgs)
            .configure(|state| state.param(vec![0.0, 0.0]).max_iters(100))
            .run()
            .unwrap();
        assert!(result.solver.stalled_at_optimum, "Bfgs: {:?}", result.state);

        let lbfgs = FaerLbfgs {
            linesearch: linesearch(),
            tol: 1e-2,
            stalled_at_optimum: false,
            s_history: VecDeque::with_capacity(LBFGS_HISTORY_SIZE),
            y_history: VecDeque::with_capacity(LBFGS_HISTORY_SIZE),
            h0: None,
        };
        let result = Executor::new(NoisyCostFloorProblem::new(), lbfgs)
            .configure(|state| state.param(vec![0.0, 0.0]).max_iters(100))
            .run()
            .unwrap();
        assert!(
            result.solver.stalled_at_optimum,
            "Lbfgs: {:?}",
            result.state
        );
    }

    /// 停滞判定のガード（`is_near_convergence_target`）: 勾配ノルムが
    /// `QUASI_NEWTON_STALL_GRAD_FACTOR·tol`未満のときだけ真。最適化初期の大きな勾配での
    /// line search失敗を停滞（収束）と誤判定しないための境界を直接検証する（統合テストでは、
    /// 既存のコスト変化ベースの副次判定が先に発火しうるため、このガード単体の効果を
    /// 切り分けられない）。
    #[test]
    fn is_near_convergence_target_is_true_only_below_stall_grad_factor_times_tol() {
        let tol = 1e-2;
        let bound = QUASI_NEWTON_STALL_GRAD_FACTOR * tol;
        assert!(is_near_convergence_target(&[0.99 * bound], tol));
        assert!(!is_near_convergence_target(&[bound], tol));
        assert!(!is_near_convergence_target(
            &[0.8 * bound, 0.8 * bound],
            tol
        ));
    }
}
