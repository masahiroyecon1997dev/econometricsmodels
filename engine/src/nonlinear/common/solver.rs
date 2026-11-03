//! ソルバーの共通入口`run_solver`と、評価回数の総枠を設ける`BudgetedProblem`。

use argmin::core::{
    CostFunction, Error as OptimizerError, Executor, Gradient, Hessian, Problem, State,
    TerminationReason,
};
use std::cell::Cell;
use std::collections::VecDeque;

use super::bfgs::FaerBfgs;
use super::error::MleError;
use super::lbfgs::{FaerLbfgs, LBFGS_HISTORY_SIZE};
use super::line_search::{convert_optimizer_error, wolfe_line_search};
use super::newton::FaerNewton;
use super::options::SolverType;
use crate::shared::error::CommonError;

/// `run_solver`の出力。
#[derive(Debug, Clone)]
pub struct SolverOutput {
    /// 収束点（`raise_on_non_convergence=false`で未収束の場合は打ち切り時点）のパラメータ。
    pub params: Vec<f64>,
    /// 収束点で解析的に評価した**対数尤度そのもの**のHessian（k×k、真の最大点では負定値）。
    /// `solver`の選択に関わらず常に評価する。`cov_type`共通行列演算（`neg_hessian_inverse`
    /// 等）はこの符号（対数尤度のHessian）を前提とする。モデルの`Hessian`トレイト実装
    /// 自体は`CostFunction`/`Gradient`と同じ符号（コスト関数＝負の対数尤度のHessian）を
    /// 返す契約になっており、ここに格納する値は`run_solver`内部で1回符号反転したもの
    /// （`docs/spec/nonlinear-common.md`1.2節「Hessianトレイトの符号規約」参照）。
    pub hessian: Vec<Vec<f64>>,
    /// 収束したかどうか。
    pub converged: bool,
    /// 実際の反復回数。
    pub n_iter: usize,
}

/// [`run_solver`]が`problem`に許す目的関数・勾配・Hessian評価の総回数を、外側の
/// `max_iter`（1回のfit呼び出し全体を通した反復上限）に対してこの倍率でスケールする
/// （[`BudgetedProblem`]参照）。
///
/// **値の根拠**: `newton`は1反復あたり評価回数がO(1)（`cost`/`gradient`各1回＋
/// `regularized_newton_step`のLM ラダー、`MAX_LM_ATTEMPTS`で頭打ち）のため、この倍率が
/// どれだけ大きくてもまず消費し切らない。`bfgs`/`lbfgs`は1反復あたりのline search評価
/// 回数が問題依存で変動するが、正常に収束するケースでは数回〜数十回程度（実測:
/// `docs/performance/logit.md`でn=1,000,000・classicalのbfgsが6〜7反復、lbfgsが17反復で
/// 収束）。`2000`は、極端に打ち切りが強い・条件数が悪いなど「正常だが遅い」ケースにも
/// 数百倍のマージンを残しつつ、退化した無限ループ相当のケースを有限時間（実測で問題に
/// なった小標本のTobitでは秒オーダー）で打ち切れる値として選んだ。既定`max_iter=35`
/// （Logit/Probit/Tobit共通）に対する実効バジェットは`(35+1)*2000=72,000`で、上記の
/// 実測反復回数に対して十分なマージンがある。
const MAX_EVALUATIONS_PER_ITER: u64 = 2000;

/// [`run_solver`]に渡す`problem`をラップし、`CostFunction`/`Gradient`/`Hessian`の呼び出し
/// 回数に総枠（バジェット）を設ける（`nonlinear::tobit::fit::tests::proptests`が
/// devプロファイルで異常に長時間実行される問題の根本対応）。
///
/// **背景（当時の状況）**: `SolverType::Bfgs`（自前実装`FaerBfgs`、`FaerBfgs::next_iter`）・
/// `SolverType::Lbfgs`（当時はargmin組み込み`LBFGS`だったが、現在は`FaerLbfgs`に置き換え済み）は
/// いずれも、1回の外側反復ごとに`MoreThuenteLineSearch`を内側`Executor`で走らせるが、
/// この内側`Executor`には`max_iters`が設定されておらず（argminの`IterState`既定値
/// `u64::MAX`）、`MoreThuenteLineSearch`自体もステップ幅の上限（`stpmax`）を設定していない
/// （argmin側のデフォルトは`f64::INFINITY`）。探索方向・勾配の組み合わせが退化し、
/// line search内部の収束判定（`info`フラグ）もNaN/Infガードも一度も発火しない状況に
/// 嵌ると、この内側ループは理論上終了しない（[`MleError::EvaluationBudgetExceeded`]の
/// docコメント参照）。
///
/// **なぜここに1箇所実装すれば全solverを保護できるか**: `MoreThuenteLineSearch`は
/// 呼び出し元（`FaerBfgs`・`FaerLbfgs`いずれの自前実装も）を問わず、必ず`Problem<O>`経由で
/// `problem.cost()`/`problem.gradient()`を呼ぶ。argminの全traitメソッドは`Result`を返す
/// 設計のため、ここでエラーを返せば`?`演算子で呼び出し元（`MoreThuenteLineSearch::next_iter`
/// →内側`Executor::run()`→`FaerBfgs::next_iter`または`FaerLbfgs::next_iter`→外側
/// `Executor::run()`）を素通りしてそのまま伝播する。`FaerLbfgs`が
/// `LINE_SEARCH_MAX_ITERS`による個別の反復上限＋明示的なエラー化を持つようになった後も、
/// この`BudgetedProblem`は複数回のline search呼び出しを横断する総枠として引き続き
/// 全solverに一律適用する（個々のline search呼び出しは正常に収束を繰り返しても、
/// 外側反復自体が`max_iter`に対して過剰に多い場合の粗い安全網、[`FaerNewton`]の
/// `MAX_LM_ATTEMPTS`と同じ位置づけ）。
///
/// `SolverType::Newton`（`FaerNewton`）はこの経路を通らず自前のLMラダー
/// （`MAX_LM_ATTEMPTS`）で既に有限回に抑えられているが、[`run_solver`]は3method共通で
/// このラッパーを適用する（将来追加されるsolverも含め、個別のsolverが独自の反復上限を
/// 正しく実装しているかに依存しない、一律の安全網とするため）。
struct BudgetedProblem<O> {
    inner: O,
    budget: u64,
    remaining: Cell<u64>,
}

impl<O> BudgetedProblem<O> {
    fn new(inner: O, budget: u64) -> Self {
        Self {
            inner,
            budget,
            remaining: Cell::new(budget),
        }
    }

    /// ラップした`O`を取り出す（バジェットは破棄する）。`run_solver`が収束後に1回だけ
    /// 行う最終的なHessian評価（`cov_type`共通行列演算が使う「対数尤度そのもの」の
    /// Hessian）は、line search内側ループの暴走防止という本来の目的とは無関係な呼び出し
    /// のため、このバジェットの対象に含めない（含めると、バジェットが
    /// ちょうど収束と同時に枯渇した場合、正常に収束した結果に対しても
    /// `EvaluationBudgetExceeded`を誤って返しうる）。
    fn into_inner(self) -> O {
        self.inner
    }

    /// 呼び出し1回分を消費する。残り枠が無ければ
    /// [`MleError::EvaluationBudgetExceeded`]を返す。`MleError`は`std::error::Error`を
    /// 実装する（thiserror）ため、`anyhow::Error`（argminの`Error`型）へは`?`でそのまま
    /// 変換される（[`convert_optimizer_error`]のdocコメント「`FaerNewton::next_iter`内で
    /// `MleError`から`?`により変換された値はdowncastで復元し」と同じ仕組み）。
    fn tick(&self) -> Result<(), MleError> {
        let remaining = self.remaining.get();
        if remaining == 0 {
            return Err(MleError::EvaluationBudgetExceeded {
                budget: self.budget,
            });
        }
        self.remaining.set(remaining - 1);
        Ok(())
    }
}

impl<O> CostFunction for BudgetedProblem<O>
where
    O: CostFunction<Param = Vec<f64>, Output = f64>,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        self.tick()?;
        self.inner.cost(param)
    }
}

impl<O> Gradient for BudgetedProblem<O>
where
    O: Gradient<Param = Vec<f64>, Gradient = Vec<f64>>,
{
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        self.tick()?;
        self.inner.gradient(param)
    }
}

impl<O> Hessian for BudgetedProblem<O>
where
    O: Hessian<Param = Vec<f64>, Hessian = Vec<Vec<f64>>>,
{
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        self.tick()?;
        self.inner.hessian(param)
    }
}

/// `newton`/`bfgs`/`lbfgs`のいずれかでモデルの負の対数尤度を最小化し、収束点のパラメータ・
/// Hessian・収束フラグ・反復回数を返す。
///
/// `problem`は`CostFunction`（負の対数尤度）/`Gradient`/`Hessian`（いずれも`Param=Vec<f64>`）を
/// 実装したモデル固有の型（Logit/Probit/Tobitがそれぞれ実装する）。標準化された設計行列を使うかは
/// 呼び出し側（各モデルの`fit()`）の責務であり、この関数は`Vec<f64>`のパラメータ空間のみを扱う
/// （`standardize_columns`/`destandardize_params`参照）。
///
/// **`Hessian`トレイトの符号規約**: `CostFunction`/`Gradient`と同じ符号（コスト関数＝
/// 負の対数尤度のHessian）で実装すること。`FaerNewton`のNewtonステップ（`Δθ = H⁻¹g`、
/// `g`は`Gradient`が返す「スコアの符号反転」）が正しい方向に進むために必要。
/// `SolverOutput.hessian`（対数尤度そのもののHessian、符号が逆）への変換はこの関数が
/// 内部で1回だけ行う（呼び出し側・各モデルの実装は意識しなくてよい）。
///
/// (準)完全分離の検出はこの関数では行わない。Logit/Probitの`fit()`が収束後に
/// `nonlinear::separation::verify_no_separation`を呼ぶ（[`MleError::SeparationDetected`]参照）。
///
/// **`tol`の意味論はsolverにより異なる**: `Newton`は総和勾配に対する
/// 絶対閾値`‖∇ℓ(θ)‖ < tol`のまま（2次収束のため`n`依存性の影響をほとんど受けない、
/// `docs/spec/logit-spec.md`3.2節参照）。`Bfgs`/`Lbfgs`は`n_obs`（観測数）で正規化した
/// 「観測あたり平均勾配」基準`‖∇ℓ(θ)‖ / n_obs < tol`を使う（実装上は`tol * n_obs`を
/// 実効的な絶対閾値としてソルバーへ渡す形。statsmodels（scipy）が対数尤度・スコア・
/// Hessianを`nobs`で割ってから最適化する設計に倣い、大標本での`bfgs`/`lbfgs`の
/// 実行時間がNewtonから桁違いに遅れる問題を解消する）。`n_obs`は各モデルの`fit()`が
/// 観測数（`y.nrows()`）をそのまま渡す。
///
/// # Errors
/// - 収束点のHessianが特異（`SingularHessian`）
/// - `raise_on_non_convergence=true`かつ`max_iter`回で収束しなかった（`NonConvergence`）
/// - `bfgs`/`lbfgs`のline searchが収束判定を満たさないまま評価回数の総枠を使い切った
///   （`EvaluationBudgetExceeded`、`BudgetedProblem`参照）
/// - その他ソルバー内部でのエラー（`CommonError::Internal`）。line searchの失敗は`LineSearchFailed`
pub fn run_solver<O>(
    problem: O,
    solver: SolverType,
    initial_params: Vec<f64>,
    max_iter: u64,
    tol: f64,
    n_obs: usize,
    raise_on_non_convergence: bool,
) -> Result<SolverOutput, MleError>
where
    O: CostFunction<Param = Vec<f64>, Output = f64>
        + Gradient<Param = Vec<f64>, Gradient = Vec<f64>>
        + Hessian<Param = Vec<f64>, Hessian = Vec<Vec<f64>>>,
{
    // line search（bfgs/lbfgs）の内側ループが収束判定を満たさないまま暴走するのを防ぐ
    // 安全網（`BudgetedProblem`のdocコメント参照）。`newton`はこの経路を
    // 通らず自前のLMラダーで既に有限回に抑えられているが、将来のsolver追加も含めた
    // 一律の安全網として3method共通で適用する。
    // `max_iter.saturating_add(1)`: `max_iter=0`（`raise_on_non_convergence`の未収束経路を
    // 検証するテスト等で使われる）でも、`Executor::init()`が最初の`cost`/`gradient`評価を
    // 必ず1回行う分の枠は残す（`budget=0`のままだと初回評価から即座に
    // `EvaluationBudgetExceeded`になり、本来の`NonConvergence`を隠してしまう）。
    let budget = max_iter
        .saturating_add(1)
        .saturating_mul(MAX_EVALUATIONS_PER_ITER);
    let problem = BudgetedProblem::new(problem, budget);

    let (params, converged, n_iter, model) = match solver {
        SolverType::Newton => {
            let optimizer = FaerNewton {
                tol,
                stalled_at_optimum: false,
            };
            let result = Executor::new(problem, optimizer)
                .configure(|state| state.param(initial_params).max_iters(max_iter))
                .run()
                .map_err(convert_optimizer_error)?;
            extract_outcome(result.state, result.problem)?
        }
        SolverType::Bfgs => {
            // `n_obs`で正規化した「観測あたり平均勾配」基準（run_solverのdocコメント
            // 「tolの意味論はsolverにより異なる」参照）。`FaerBfgs`自体は正規化を知らず、
            // 実効的な絶対閾値を受け取るだけでよい。
            let optimizer = FaerBfgs {
                linesearch: wolfe_line_search()?,
                tol: tol * n_obs as f64,
                stalled_at_optimum: false,
                h0_from_hessian: false,
            };
            let result = Executor::new(problem, optimizer)
                .configure(|state| state.param(initial_params).max_iters(max_iter))
                .run()
                .map_err(convert_optimizer_error)?;
            extract_outcome(result.state, result.problem)?
        }
        SolverType::Lbfgs => {
            // Bfgsと同じ正規化（`n_obs`で正規化した「観測あたり平均勾配」基準）。
            // `FaerLbfgs`自体は正規化を知らず、実効的な絶対閾値を受け取るだけでよい
            // （`FaerBfgs`と同じ設計）。
            let optimizer = FaerLbfgs {
                linesearch: wolfe_line_search()?,
                tol: tol * n_obs as f64,
                stalled_at_optimum: false,
                s_history: VecDeque::with_capacity(LBFGS_HISTORY_SIZE),
                y_history: VecDeque::with_capacity(LBFGS_HISTORY_SIZE),
                h0: None,
            };
            let result = Executor::new(problem, optimizer)
                .configure(|state| state.param(initial_params).max_iters(max_iter))
                .run()
                .map_err(convert_optimizer_error)?;
            extract_outcome(result.state, result.problem)?
        }
    };

    if !converged && raise_on_non_convergence {
        return Err(MleError::NonConvergence { n_iter });
    }

    // Hessianはsolverの選択に関わらず、収束点で常に解析的に評価する
    // （bfgs/lbfgsの内部近似Hessianは使い回さない）。
    //
    // `Hessian`トレイトの契約は「`CostFunction`/`Gradientと同じ符号（コスト関数=負の
    // 対数尤度のHessian）」（`FaerNewton`のNewtonステップが正しい方向に進むために必要、
    // `CostFunction`のdocコメント参照）。一方`SolverOutput.hessian`は`cov_type`共通行列
    // 演算（`neg_hessian_inverse`等）が前提とする「対数尤度そのもののHessian」（真の
    // 最大点で負定値）でなければならない。両者は符号が逆（`-loglik`のHessian＝
    // `loglik`のHessianの符号反転）なので、ここで1回だけ符号反転して契約を合わせる
    // （`docs/spec/nonlinear-common.md`1.2節「Hessianトレイトの符号規約」参照）。
    // `into_inner()`でバジェットの対象から外す（`BudgetedProblem::into_inner`のdoc
    // コメント参照）。この1回の呼び出しはline search内側ループの暴走防止という
    // バジェット本来の目的とは無関係なため。
    let cost_hessian = model
        .into_inner()
        .hessian(&params)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let hessian: Vec<Vec<f64>> = cost_hessian
        .iter()
        .map(|row| row.iter().map(|v| -v).collect())
        .collect();

    Ok(SolverOutput {
        params,
        hessian,
        converged,
        n_iter,
    })
}

/// `Executor::run()`の結果から`(params, converged, n_iter, model)`を取り出す。
/// `SolverType`の3分岐で共通の後処理のため、`run_solver`から切り出している。
/// `I`はソルバーごとに異なる状態型（`FaerLbfgs`はHessianスロットを使わないため`H=()`）だが、
/// いずれも`State`トレイト経由で同じ形で取り出せる。
///
/// **`TerminationReason::SolverExit`を専用に検出していた分岐は削除済み**:
/// 元々は、`SolverType::Lbfgs`が当時使っていたargmin組み込み`LBFGS::next_iter`が、line search用
/// 内側`Executor::run()`の`Err`を`?`で伝播せず`Ok(state.terminate_with(SolverExit(msg)))`と
/// して握りつぶす挙動（`argmin-0.11.0/src/solver/quasinewton/lbfgs.rs`）を持っていたため、
/// この分岐で早期検出し元のエラー内容を保っていた。その後
/// `SolverType::Lbfgs`を自前実装`FaerLbfgs`（[`FaerBfgs`]・[`FaerNewton`]と同じく`?`で
/// そのまま`Err`を伝播する設計）に置き換えたことで、3method全てが`SolverExit`を
/// 二度と発生させなくなった（`MoreThuenteLineSearch`自体もこの`TerminationReason`は
/// 使わない）ため、分岐ごと削除した。
///
/// **残る`ok_or_else`の2箇所は理論上到達不能**（`.claude/rules/rust-style.md`「テスト」の
/// カバレッジ方針、Logitのカバレッジ確認時に判明・受け入れ済み）:
/// - `state.get_best_param()`が`None`になるのは`Executor::run()`が`init()`/`next_iter()`を
///   一度も呼ばずに終了した場合のみだが、`init()`が必ず初期パラメータを`state`に設定する
///   （`FaerNewton::init`・`FaerBfgs::init`・`FaerLbfgs::init`のいずれも）ため起こり得ない。
/// - `problem.take_problem()`が`None`になるのは既に一度`take_problem()`を呼んだ後に
///   再度呼んだ場合のみだが、`run_solver`はこの関数を`Executor::run()`の結果に対して
///   1回しか呼ばない。
fn extract_outcome<O, I>(
    state: I,
    mut problem: Problem<O>,
) -> Result<(Vec<f64>, bool, usize, O), MleError>
where
    I: State<Param = Vec<f64>>,
{
    let converged = matches!(
        state.get_termination_reason(),
        Some(TerminationReason::SolverConverged)
    );
    let n_iter = state.get_iter() as usize;
    let params = state
        .get_best_param()
        .cloned()
        .ok_or_else(|| CommonError::internal("the solver did not produce a parameter estimate"))?;
    let model = problem
        .take_problem()
        .ok_or_else(|| CommonError::internal("the optimization problem could not be recovered"))?;
    Ok((params, converged, n_iter, model))
}

#[cfg(test)]
mod tests {

    use super::super::numerics::identity_matrix;
    use super::super::test_support::*;
    use super::*;

    use argmin::core::Error as OptimizerError;

    /// [`BudgetedProblem`]が実際にバジェットを共有カウンタで管理し、
    /// 使い切ると`MleError::EvaluationBudgetExceeded`を返すことを直接検証する
    /// （`run_solver`経由の統合テストとは別に、この低レベルの building block 自体を
    /// 単体で検証する。`bfgs_rank2_update_matches_hand_computed_values_...`と同じ方針）。
    #[test]
    fn budgeted_problem_returns_evaluation_budget_exceeded_error_after_budget_is_exhausted() {
        let wrapped = BudgetedProblem::new(quadratic_problem(), 3);
        let param = vec![0.0, 0.0];

        assert!(wrapped.cost(&param).is_ok());
        assert!(wrapped.cost(&param).is_ok());
        assert!(wrapped.cost(&param).is_ok());

        let err = wrapped.cost(&param).unwrap_err();
        let mle_error = err.downcast::<MleError>().unwrap();
        assert_eq!(mle_error, MleError::EvaluationBudgetExceeded { budget: 3 });
    }

    /// `cost`と`gradient`は独立のカウンタではなく、同じバジェットを共有して消費する
    /// （実際のline searchは1反復でcost・gradient双方を呼ぶため、この共有が前提）。
    #[test]
    fn budgeted_problem_shares_the_same_budget_across_cost_and_gradient_calls() {
        let wrapped = BudgetedProblem::new(quadratic_problem(), 2);
        let param = vec![0.0, 0.0];

        assert!(wrapped.cost(&param).is_ok());
        assert!(wrapped.gradient(&param).is_ok());
        assert!(wrapped.cost(&param).is_err());
    }

    /// `f(θ) = -θ₀`（`θ₁`以降は無視）という、下に有界でない（最小点が存在しない）
    /// ダミー問題。`gradient`は`cost`の真の勾配`[-1, 0, ...]`と整合しており、退化した
    /// 細工（NaN・不整合な勾配等）は一切無い、正当な降下方向を持つ問題であるにも
    /// 関わらず、argmin`MoreThuenteLineSearch`は**理論上無限にループする**:
    /// `θ₀`方向の勾配が常に厳密に`-1`（一定）のため、strong Wolfe条件の曲率条件
    /// `|dg| <= gtol*|dginit|`（既定`gtol=0.9`）が`|dg|=|dginit|=1`である限り
    /// **どのステップ幅でも構造的に満たされない**（`1<=0.9`は恒偽）。一方
    /// 十分減少条件（Armijo）は`cost`が真に無限に減少し続けるため常に満たされ、
    /// line searchは「ブラケットせず`stpmax`方向へ延々と外挿し続ける」`cstep`の
    /// 分岐（4番目のケース、`info=4`）に永久に留まる。ステップ幅が実際に`f64::INFINITY`
    /// に達した後も`(stp-stpmax).abs()`が`NaN`になり比較が常に偽になるため、
    /// 収束判定（`info`フラグ1〜6）もNaN/Infガードも一度も発火しない
    /// （実際に踏んだ暴走と同型の構造。手計算でのトレース、および
    /// devビルドで実際にハングさせてから[`BudgetedProblem`]の除去がハングを再現し
    /// 導入が解消することを確認済み）。
    ///
    /// **単純な定数コスト（`cost`が`param`を無視して常に同じ値を返す）では再現しない**
    /// ことに注意（既に試して確認済み）: `FaerBfgs::terminate`はargmin組み込みBFGSと
    /// 同じコスト変化ベースの副次判定（`|prev_cost-cost|<f64::EPSILON`、
    /// `FaerBfgs`のdocコメント参照）を持つため、コストが真に一定だと外側のBFGS反復が
    /// line searchに入る前に「収束」と誤判定してしまい、内側のline searchが暴走する
    /// 状況を再現できない。この問題は`cost`が`param`に応じて真に（無限に）変化し続ける
    /// ことで外側の誤判定を避けつつ、内側line searchの曲率条件だけを恒久的に破る設計。
    #[derive(Clone)]
    struct UnboundedBelowProblem;

    impl CostFunction for UnboundedBelowProblem {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
            Ok(-param[0])
        }
    }

    impl Gradient for UnboundedBelowProblem {
        type Param = Vec<f64>;
        type Gradient = Vec<f64>;

        fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
            Ok(vec![-1.0; param.len()])
        }
    }

    impl Hessian for UnboundedBelowProblem {
        type Param = Vec<f64>;
        type Hessian = Vec<Vec<f64>>;

        fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
            Ok(identity_matrix(param.len()))
        }
    }

    #[test]
    fn run_solver_bfgs_returns_evaluation_budget_exceeded_error_instead_of_hanging_on_a_stalled_line_search()
     {
        let result = run_solver(
            UnboundedBelowProblem,
            SolverType::Bfgs,
            vec![0.0],
            1,
            1e-6,
            1,
            true,
        );

        assert!(matches!(
            result,
            Err(MleError::EvaluationBudgetExceeded { .. })
        ));
    }

    #[test]
    fn run_solver_lbfgs_returns_a_bounded_error_instead_of_hanging_on_a_stalled_line_search() {
        // `FaerLbfgs`（自前実装に置き換え済み）は、`FaerBfgs`と違い内側line
        // search用`Executor`に明示的な`max_iters`（[`LINE_SEARCH_MAX_ITERS`]=100）を
        // 設定しているため、[`BudgetedProblem`]の総枠（このテストでは
        // `(1+1)*2000=4000`評価分）を使い切るより先にこちらの上限に到達し、
        // `MleError::LineSearchFailed`を返す（`SolverType::Bfgs`、上のテストとは異なり
        // `EvaluationBudgetExceeded`という具体的なバリアントにはならない。`FaerBfgs`には
        // この個別チェックが無いため）。
        let result = run_solver(
            UnboundedBelowProblem,
            SolverType::Lbfgs,
            vec![0.0],
            1,
            1e-6,
            1,
            true,
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::LineSearchFailed { max_iters: 100 }
        );
    }

    /// `SolverType::Bfgs`は`tol * n_obs`を実効的な絶対閾値として使うはず（
    /// `run_solver`のdocコメント「tolの意味論はsolverにより異なる」参照）。`n_obs`と`tol`を
    /// 別々に振っても積が同じなら同じ収束点・反復回数になることを直接検証する
    /// （既存のBFGS/LBFGSテストは全て`n_obs=1`で呼んでおり、この正規化ロジック自体は
    /// このテストで検証する）。
    #[test]
    fn run_solver_bfgs_scales_effective_tol_by_n_obs() {
        let via_n_obs = run_solver(
            quadratic_problem(),
            SolverType::Bfgs,
            vec![0.0, 0.0],
            100,
            1e-9,
            1000,
            true,
        )
        .unwrap();
        let via_tol = run_solver(
            quadratic_problem(),
            SolverType::Bfgs,
            vec![0.0, 0.0],
            100,
            1e-6,
            1,
            true,
        )
        .unwrap();

        assert_eq!(via_n_obs.n_iter, via_tol.n_iter);
        assert!((via_n_obs.params[0] - via_tol.params[0]).abs() < 1e-12);
        assert!((via_n_obs.params[1] - via_tol.params[1]).abs() < 1e-12);
    }

    /// `SolverType::Lbfgs`も`Bfgs`と同じ正規化を使うはず（同じ理由）。
    #[test]
    fn run_solver_lbfgs_scales_effective_tol_by_n_obs() {
        let via_n_obs = run_solver(
            quadratic_problem(),
            SolverType::Lbfgs,
            vec![0.0, 0.0],
            100,
            1e-9,
            1000,
            true,
        )
        .unwrap();
        let via_tol = run_solver(
            quadratic_problem(),
            SolverType::Lbfgs,
            vec![0.0, 0.0],
            100,
            1e-6,
            1,
            true,
        )
        .unwrap();

        assert_eq!(via_n_obs.n_iter, via_tol.n_iter);
        assert!((via_n_obs.params[0] - via_tol.params[0]).abs() < 1e-12);
        assert!((via_n_obs.params[1] - via_tol.params[1]).abs() < 1e-12);
    }

    /// `SolverType::Newton`は`n_obs`を無視し、`tol`をそのまま絶対閾値として使うはず
    /// （`newton`は正規化の対象外、`docs/spec/logit-spec.md`3.2節参照）。
    #[test]
    fn run_solver_newton_ignores_n_obs() {
        let small_n_obs = run_solver(
            quadratic_problem(),
            SolverType::Newton,
            vec![0.0, 0.0],
            35,
            1e-6,
            1,
            true,
        )
        .unwrap();
        let large_n_obs = run_solver(
            quadratic_problem(),
            SolverType::Newton,
            vec![0.0, 0.0],
            35,
            1e-6,
            1_000_000,
            true,
        )
        .unwrap();

        assert_eq!(small_n_obs.n_iter, large_n_obs.n_iter);
        assert_eq!(small_n_obs.params, large_n_obs.params);
    }

    // `max_iter=0`: 初期逆Hessianが厳密なため、二次関数は`max_iter=1`でも1反復で収束する。
    #[test]
    fn run_solver_returns_non_convergence_error_when_max_iter_is_too_small() {
        let result = run_solver(
            quadratic_problem(),
            SolverType::Bfgs,
            vec![1000.0, -1000.0],
            0,
            1e-12,
            1,
            true,
        );

        assert!(matches!(result, Err(MleError::NonConvergence { .. })));
    }

    // `max_iter=0`: 初期逆Hessianが厳密なため、二次関数は`max_iter=1`でも1反復で収束する。
    #[test]
    fn run_solver_returns_result_without_raising_when_raise_on_non_convergence_is_false() {
        let output = run_solver(
            quadratic_problem(),
            SolverType::Bfgs,
            vec![1000.0, -1000.0],
            0,
            1e-12,
            1,
            false,
        )
        .unwrap();

        assert!(!output.converged);
    }
}
