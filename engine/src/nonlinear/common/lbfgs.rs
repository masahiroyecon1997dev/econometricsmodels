//! 自前のL-BFGS（`FaerLbfgs`）とtwo-loop recursion。

use argmin::core::TerminationStatus;
use argmin::core::{
    CostFunction, Error as OptimizerError, Executor, Gradient, Hessian, IterState, KV, LineSearch,
    Problem, Solver, State, TerminationReason,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use std::collections::VecDeque;

use super::error::MleError;
use super::line_search::{
    LINE_SEARCH_MAX_ITERS, NEAR_TARGET_LINE_SEARCH_MAX_ITERS, is_near_convergence_target,
    line_search_converged, line_search_step_satisfies_wolfe,
};
use super::numerics::{dot, inverse_if_positive_definite, l2_norm};

/// [`FaerLbfgs`]のtwo-loop recursion（[`two_loop_recursion`]）が保持する直近secantペア
/// `(s,y)`の件数（limited-memoryの由来）。従来のargmin組み込み
/// `LBFGS::new(linesearch, 7)`と同じ既定値をそのまま踏襲する（変更する積極的な理由が
/// 無いため）。
pub(super) const LBFGS_HISTORY_SIZE: usize = 7;

/// argmin組み込みの`LBFGS`（`argmin::solver::quasinewton::LBFGS`）を、`FaerBfgs`と同型の
/// パターンで自前実装したもの。
///
/// **経緯**: `FaerBfgs`（上記docコメント参照）は「1回目の反復専用のline
/// search初期ステップ幅`min(1,1/‖g₀‖)`」を適用することで、単位行列の逆Hessianによる
/// 最初の探索方向の暴走を防いだが、argmin組み込み`LBFGS`にはこの制御点を適用できな
/// かった（`linesearch`フィールドがprivateで、`self.linesearch.clone()`を反復ごとに
/// 使い捨てるだけの内部実装のため、`FaerBfgs`のように`self`が`linesearch`を所有し
/// 反復間で初期ステップ幅を切り替える制御ができない）。two-loop recursion自体が使う
/// `γ=(s_{k-1}ᵀy_{k-1})/(y_{k-1}ᵀy_{k-1})`によるself-scalingは、argmin組み込み`LBFGS`も
/// 2回目以降の反復では既に内部で行っている（履歴`s`/`y`から動的に計算、Nocedal &
/// Wright *Numerical Optimization* 6.1節そのもの）ため、`FaerBfgs`のケースとは異なり
/// two-loop recursionの数式自体に欠陥があるわけではない。**欠けていたのは「1回目の
/// 反復（履歴が空で`γ=1.0`固定）専用のline search初期ステップ幅」という、`FaerBfgs`と
/// 全く同じ制御点**であり、argmin組み込みの公開APIにはこれが無かった。
///
/// `argmin::core::Solver`トレイトを直接実装し、two-loop recursion（[`two_loop_recursion`]）を
/// 自前で持つことで、この制御点を獲得する（`FaerNewton`/`FaerBfgs`と同じ理由での
/// 自前化）。line search自体は`FaerBfgs`と同様、引き続き`MoreThuenteLineSearch`を流用する。
///
/// あわせて、内側line search用`Executor`に明示的な`max_iters`（[`LINE_SEARCH_MAX_ITERS`]）を
/// 設定し、収束判定を満たさないまま打ち切られた場合は明示的にエラー化する
/// （[`LINE_SEARCH_MAX_ITERS`]のdocコメント参照。`max_iters`到達は
/// `Err`にならず不正なステップを黙って採用してしまう問題への対応。`FaerBfgs`には無い
/// チェックだが、この実装では`FaerLbfgs`のみを対象とし`FaerBfgs`側は変更しない）。
///
/// `FaerBfgs`と異なり逆Hessian近似の全体は更新せず、直近[`LBFGS_HISTORY_SIZE`]件の
/// secantペア`(s,y)`だけを保持する（limited-memoryの由来）。初期行列だけは`k×k`の行列`h0`を持つ。
///
/// **初期行列（`h0`）**: two-loop recursionの初期行列には、標準の`γI`（`γ=sᵀy/yᵀy`）ではなく、
/// warm start点の厳密Hessianの逆行列`H₀⁻¹`を使う（`FaerBfgs::init`と同じ
/// `inverse_if_positive_definite`で正定値かつ可逆のときのみ）。`γI`は曲率のスケールしか表さず、
/// 回帰変数間の相関による悪条件の形を履歴から学習するのに多くの反復を要するため。`H₀⁻¹`は直近の
/// ペアで`sᵀy/(yᵀH₀⁻¹y)`倍に再スケールして使う（warm start点と収束点でHessianが大きく違う
/// データで、素の`H₀⁻¹`が`γI`より悪化するのを避ける）。
///
/// **正定値でない場合（Tobitのように尤度が大域凹でない手法のwarm start点）は、`H₀⁻¹`が降下方向を
/// 作らないため従来の初期化（`γI`と1回目のline search初期ステップ幅の調整）へ戻す。**
/// `h0`が`Some`のとき1回目のステップは既にNewtonステップに相当するスケールなので、初期ステップ幅は
/// 標準の`1.0`のままにする。実測値と経緯は`engine/src/nonlinear/CLAUDE.md`参照。
pub(super) struct FaerLbfgs {
    /// line search。`self`が所有し反復間で使い回す（`FaerBfgs::linesearch`と同じ理由）。
    pub(super) linesearch: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>,
    pub(super) tol: f64,
    /// `FaerBfgs::stalled_at_optimum`と同じ役割・同じ判定。
    pub(super) stalled_at_optimum: bool,
    /// 直近`s`（パラメータ差分`θ_{k+1}-θ_k`）の履歴。古い順（先頭が最古）、
    /// [`LBFGS_HISTORY_SIZE`]件を超えたら最古のペアから破棄する（limited-memoryの
    /// 由来）。`y_history`と常に同じ長さ・同じ順序で対応する。
    pub(super) s_history: VecDeque<Vec<f64>>,
    /// 直近`y`（勾配差分`∇f_{k+1}-∇f_k`）の履歴。`s_history`のdocコメント参照。
    pub(super) y_history: VecDeque<Vec<f64>>,
    /// warm start点の厳密Hessianの逆行列（正定値のときのみ`Some`。構造体docコメント参照）。
    /// `None`のときは従来の`γI`初期化。
    pub(super) h0: Option<Vec<Vec<f64>>>,
}

type LbfgsState = IterState<Vec<f64>, Vec<f64>, (), (), (), f64>;

impl<O> Solver<O, LbfgsState> for FaerLbfgs
where
    O: CostFunction<Param = Vec<f64>, Output = f64>
        + Gradient<Param = Vec<f64>, Gradient = Vec<f64>>
        + Hessian<Param = Vec<f64>, Hessian = Vec<Vec<f64>>>,
{
    /// `argmin::core::Solver`トレイトの必須メソッド（`FaerNewton::name`と同じ理由で
    /// 未カバーでも振る舞いの正しさに影響しない）。
    fn name(&self) -> &str {
        "L-BFGS (faer-backed)"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: LbfgsState,
    ) -> Result<(LbfgsState, Option<KV>), OptimizerError> {
        let param = state.take_param().ok_or_else(|| {
            OptimizerError::msg(
                "FaerLbfgs requires an initial parameter vector via Executor's configure method",
            )
        })?;
        let grad = problem.gradient(&param)?;

        // 初期点（warm start）の厳密Hessianが正定値なら、その逆行列を初期行列にする
        // （`FaerLbfgs`のdocコメント「初期行列」参照）。1回目のステップがほぼNewtonステップに
        // なるため、line searchの初期ステップ幅は標準の`1.0`のままでよい。
        if let Some(inv_hessian) = inverse_if_positive_definite(&problem.hessian(&param)?) {
            self.h0 = Some(inv_hessian);
            return Ok((state.param(param).gradient(grad), None));
        }

        // 1回目の反復専用のline search初期ステップ幅（`FaerBfgs::init`と同じ式・同じ
        // 理由）。履歴が空の1回目は`two_loop_recursion`のγが既定`1.0`のままで探索方向
        // `-g₀`をスケールできないため、単位行列の逆Hessianを使う`FaerBfgs`の1回目と
        // 同型の暴走リスクがある。
        //
        // `n_obs`で正規化する案（`min(1, n_obs/‖g₀‖)`、`tol`の正規化と同じ発想）は、
        // Tobit退化ケース（`fit_lbfgs_converges_for_a_previously_
        // stalling_case`）が再び`LINE_SEARCH_MAX_ITERS`超過で
        // 失敗する回帰を確認したため不採用とした（`‖g₀‖`と`n_obs`の関係は単純な
        // 比例関係ではなく、データセットごとに異なるため）。
        //
        // 無正規化のこの式は`generate_binary_choice_dataset("baseline", link="probit",
        // n=100_000, k=5, seed=42)`で`alpha0≈7.5e-5`と小さくなるが、同データで
        // lbfgsが遅かった（23反復・約3秒）主因はこれではなく、収束点近傍でコストが
        // 丸め誤差の床に達しline searchが極小ステップを繰り返していたことだった
        // （`QUASI_NEWTON_STALL_GRAD_FACTOR`のdocコメント参照）。停滞検出の導入後は
        // 9反復・約0.18秒（argmin組み込みLBFGS時代の7反復・0.16秒と同程度）。
        let alpha0 = (1.0 / l2_norm(&grad)).min(1.0);
        if alpha0.is_finite() && alpha0 > 0.0 {
            self.linesearch.initial_step_length(alpha0)?;
        }

        let state = state.param(param).gradient(grad);
        Ok((state, None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: LbfgsState,
    ) -> Result<(LbfgsState, Option<KV>), OptimizerError> {
        let param = state
            .take_param()
            .ok_or_else(|| OptimizerError::msg("FaerLbfgs: parameter vector in state not set"))?;
        let cur_cost = state.get_cost();
        let prev_grad = state
            .take_gradient()
            .ok_or_else(|| OptimizerError::msg("FaerLbfgs: gradient in state not set"))?;
        // `FaerBfgs::next_iter`の`is_first_iter`と同じ役割（履歴追加前に判定する）。
        let is_first_iter = state.get_iter() == 0;

        let direction = two_loop_recursion(
            &self.s_history,
            &self.y_history,
            &prev_grad,
            self.h0.as_deref(),
        );
        self.linesearch.search_direction(direction);

        // `take_problem()`が`None`になるのは既に一度取り出した後に再度取り出した場合
        // のみだが、`next_iter`はこの箇所でしか呼ばない（`FaerBfgs::next_iter`と同じ
        // 契約）。
        let inner_problem = problem.take_problem().ok_or_else(|| {
            OptimizerError::msg("FaerLbfgs: failed to recover the optimization problem")
        })?;
        // `FaerBfgs::next_iter`と同じく、収束目標の近傍ではline searchの反復上限を絞る。
        let near_target = is_near_convergence_target(&prev_grad, self.tol);
        let line_search_max_iters = if near_target {
            NEAR_TARGET_LINE_SEARCH_MAX_ITERS
        } else {
            LINE_SEARCH_MAX_ITERS
        };
        let result = Executor::new(inner_problem, self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .gradient(prev_grad.clone())
                    .cost(cur_cost)
                    .max_iters(line_search_max_iters)
            })
            .ctrlc(false)
            .run()?;
        let mut sub_state = result.state;
        let line_problem = result.problem;

        if near_target && !line_search_converged(&sub_state) {
            // `FaerBfgs::next_iter`と同じく、直前の反復点に留まり停滞として終了する。
            problem.consume_problem(line_problem);
            self.stalled_at_optimum = true;
            return Ok((state.param(param).cost(cur_cost).gradient(prev_grad), None));
        }

        // `LINE_SEARCH_MAX_ITERS`のdocコメント参照: `max_iters`到達は`Err`にならず
        // 打ち切り時点のパラメータをそのまま`Ok`で返すため、収束判定を明示的に
        // 確認する（`FaerBfgs`には無いチェック）。
        if !line_search_converged(&sub_state) {
            let mle_error = MleError::LineSearchFailed {
                max_iters: LINE_SEARCH_MAX_ITERS,
            };
            return Err(mle_error.into());
        }

        let xk1 = sub_state.take_param().ok_or_else(|| {
            OptimizerError::msg("FaerLbfgs: no parameters returned by line search")
        })?;
        let next_cost = sub_state.get_cost();
        problem.consume_problem(line_problem);

        let grad = problem.gradient(&xk1)?;

        // 1回目の反復専用のline search初期ステップ幅（`init`で設定）は、2回目以降は
        // 標準の1.0に戻す（`FaerBfgs::next_iter`と同じ理由）。
        if is_first_iter {
            self.linesearch.initial_step_length(1.0)?;
        }

        let sk: Vec<f64> = xk1.iter().zip(param.iter()).map(|(a, b)| a - b).collect();
        let yk: Vec<f64> = grad
            .iter()
            .zip(prev_grad.iter())
            .map(|(a, b)| a - b)
            .collect();

        // `FaerBfgs::next_iter`と同じ停滞判定。
        self.stalled_at_optimum =
            !line_search_step_satisfies_wolfe(cur_cost, &prev_grad, next_cost, &grad, &sk)
                && is_near_convergence_target(&grad, self.tol);

        // secant条件`yᵀs>0`のチェックは行わず、argmin組み込み`LBFGS::next_iter`と同じく
        // 常にペアを履歴に追加する（`FaerBfgs`のrank-2更新スキップとは異なる設計）。
        // `MoreThuenteLineSearch`はstrong Wolfe条件（曲率条件
        // `|φ'(α)|≤c2|φ'(0)|`）を満たすステップのみ受理するため、受理されたペアは
        // 理論上`yᵀs>0`を自然に満たす（負曲率を追加で弾く必要性が薄い）。
        //
        // 当初は`FaerBfgs`に倣い絶対閾値`f64::EPSILON`でペアを弾くガードを入れていたが、
        // `generate_binary_choice_dataset("baseline", link="probit", n=100_000, k=5,
        // seed=42)`で実測したところ、収束点近傍でコスト関数が浮動小数点の底に達すると
        // （Newtonの収束判定で述べた、コスト関数が浮動小数点の底に達する現象と同根）
        // `yᵀs`が`f64::EPSILON`をわずかに下回る値になり続け、
        // このガードが新しいペアの追加を拒否し続けて履歴が古いまま凍結される結果、
        // 収束までの反復回数が9→31、実行時間が0.27s→5.1sに悪化することが判明した。
        // ガードを外す（常に追加する）と反復回数は12まで戻る。`rho=1/(yᵀs)`が極端に
        // 大きくなるリスクはあるが、limited-memoryのため悪いペアの影響は
        // `LBFGS_HISTORY_SIZE`反復で自然にローテーションアウトされる（argmin組み込み
        // 版が長年この設計で問題なく動いてきたこととも整合する）。
        if self.s_history.len() >= LBFGS_HISTORY_SIZE {
            self.s_history.pop_front();
            self.y_history.pop_front();
        }
        self.s_history.push_back(sk);
        self.y_history.push_back(yk);

        Ok((state.param(xk1).cost(next_cost).gradient(grad), None))
    }

    fn terminate(&mut self, state: &LbfgsState) -> TerminationStatus {
        if let Some(g) = state.get_gradient()
            && l2_norm(g) < self.tol
        {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        if self.stalled_at_optimum {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        // コストが（ほぼ）変化しなくなった場合も収束扱いにする（`FaerBfgs::terminate`と
        // 同じ理由・同じ基準。argmin組み込み`LBFGS`の既定`tol_cost=F::epsilon()`も踏襲）。
        if (state.get_prev_cost() - state.get_cost()).abs() < f64::EPSILON {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

/// L-BFGSのtwo-loop recursion（Nocedal & Wright *Numerical Optimization* Algorithm 7.4）:
/// 直近`m`件のsecantペア`(sᵢ,yᵢ)`（`s_history`/`y_history`、古い順）と現在の勾配`grad`から
/// 近似逆Hessian×勾配`H_k∇f_k`を計算し、降下方向`-H_k∇f_k`を返す。
///
/// `h0`は初期行列（`FaerLbfgs`のdocコメント「初期行列」参照）。`Some(H₀⁻¹)`のときは
/// `H₀⁻¹`を直近のペアで`sᵀy/(yᵀH₀⁻¹y)`倍に再スケールして使い、履歴が空（最適化1回目の反復）なら
/// 再スケールせず`-H₀⁻¹grad`を返す。`yᵀH₀⁻¹y`・`sᵀy`が実質ゼロ以下（縮退）のときも再スケール
/// しない（`1.0`）。
///
/// `None`のときは標準の`γI`: 履歴が空なら`γ=1.0`（スケーリング無し）として単位行列近似にフォール
/// バックし`-grad`を返す（`FaerLbfgs::init`が設定するline search初期ステップ幅でこの反復の
/// スケールを別途補う）。`γ=(s_lastᵀy_last)/(y_lastᵀy_last)`（直近のペアのみを使うself-scaling、
/// argmin組み込み`LBFGS::next_iter`の`gamma`計算と同じ式）で、`y_lastᵀy_last`が実質ゼロ（縮退）
/// の場合はゼロ除算を避け`γ=1.0`にフォールバックする（`bfgs_updated_inv_hessian`の
/// `ykyk > f64::EPSILON`ガードと同じ発想）。
fn two_loop_recursion(
    s_history: &VecDeque<Vec<f64>>,
    y_history: &VecDeque<Vec<f64>>,
    grad: &[f64],
    h0: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    let m = s_history.len();
    let mut q = grad.to_vec();
    let mut alpha = vec![0.0; m];
    let mut rho = vec![0.0; m];

    // `yᵀs`が実質ゼロ（丸め誤差で負・ほぼゼロになりうる）なペアは、そのペアの寄与を
    // 素通り（`rho_i=alpha_i=0.0`、`q`を更新しない）させて安全に無視する
    // `gamma`計算の`yy > f64::EPSILON`ガードと同じ絶対閾値・
    // 同じ発想だが、`gamma`が履歴の最後のペアだけを見るのに対し、こちらは
    // two-loop recursionが参照する**全ペア**に適用する必要がある——`FaerLbfgs::
    // next_iter`は（`FaerBfgs`のrank-2更新スキップとは異なり）secant条件を満たさない
    // ペアも履歴にそのまま追加する設計（このファイル内の`FaerLbfgs`のdocコメント
    // 「実装時に踏んだ罠その1」参照）のため、`rho_i=1/(yᵀs)`のゼロ除算が理論上
    // 履歴中の任意のペアで起こりうる。`rho[i]=alpha[i]=0.0`のままにしておけば、
    // 後続の前向きループでも`beta_i=rho[i]*..=0`・`coeff=alpha[i]-beta_i=0`となり
    // 自動的に同じペアの寄与が無視される（前向きループ側に別途ガードを足す必要はない）。
    for (i, (s, y)) in s_history.iter().zip(y_history.iter()).enumerate().rev() {
        let yksk = dot(y, s);
        if yksk <= f64::EPSILON {
            continue;
        }
        let rho_i = 1.0 / yksk;
        let alpha_i = rho_i * dot(s, &q);
        for (q_j, y_j) in q.iter_mut().zip(y.iter()) {
            *q_j -= alpha_i * y_j;
        }
        rho[i] = rho_i;
        alpha[i] = alpha_i;
    }

    let mut r: Vec<f64> = match h0 {
        Some(h0) => {
            // 初期行列`H₀⁻¹`を直近のペアで`sᵀy/(yᵀH₀⁻¹y)`倍に再スケールして`H₀⁻¹q`に掛ける
            // （`FaerLbfgs`のdocコメント「初期行列」参照）。ペアが無い（1回目の反復）、または
            // `yᵀH₀⁻¹y`・`sᵀy`が実質ゼロ以下（縮退）のときは再スケールしない（`1.0`）。
            let scale = s_history
                .back()
                .zip(y_history.back())
                .and_then(|(s, y)| {
                    let h0_y: Vec<f64> = h0.iter().map(|row| dot(row, y)).collect();
                    let y_h0_y = dot(y, &h0_y);
                    let s_y = dot(s, y);
                    (y_h0_y > f64::EPSILON && s_y > f64::EPSILON).then(|| s_y / y_h0_y)
                })
                .unwrap_or(1.0);
            h0.iter().map(|row| scale * dot(row, &q)).collect()
        }
        None => {
            let gamma = s_history
                .back()
                .zip(y_history.back())
                .map(|(s, y)| {
                    let yy = dot(y, y);
                    if yy > f64::EPSILON {
                        dot(s, y) / yy
                    } else {
                        1.0
                    }
                })
                .unwrap_or(1.0);
            q.iter().map(|v| v * gamma).collect()
        }
    };

    for (i, (s, y)) in s_history.iter().zip(y_history.iter()).enumerate() {
        let beta_i = rho[i] * dot(y, &r);
        let coeff = alpha[i] - beta_i;
        for (r_j, s_j) in r.iter_mut().zip(s.iter()) {
            *r_j += coeff * s_j;
        }
    }

    r.iter().map(|v| -v).collect()
}

#[cfg(test)]
mod tests {
    use super::super::bfgs::bfgs_rank2_update;

    use super::super::options::SolverType;
    use super::super::solver::run_solver;
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn two_loop_recursion_returns_negative_gradient_when_history_is_empty() {
        // 履歴が空（最適化1回目の反復）の場合はγ=1.0（スケーリング無し）で単位行列
        // 近似にフォールバックし、`-grad`をそのまま返すはず。
        let s_history = VecDeque::new();
        let y_history = VecDeque::new();
        let grad = vec![3.0, -4.0];

        let direction = two_loop_recursion(&s_history, &y_history, &grad, None);
        assert_eq!(direction, vec![-3.0, 4.0]);
    }

    #[test]
    fn two_loop_recursion_matches_hand_computed_values_for_one_history_pair() {
        // s=[1,0]、y=[0,1]、grad=[2,3]の1件履歴（互いに直交、γ・rhoの手計算を単純化
        // するため意図的に選んだ値）。
        //   rho = 1/(y・s) = 1/0 ... y・s=0だと特異になるため、s=[1,0]・y=[1,2]を使う。
        // rho=1/(y・s)=1/1=1、alpha=rho*(s・grad)=1*2=2、q=grad-alpha*y=[2,3]-2*[1,2]=[0,-1]。
        // γ=(s・y)/(y・y)=1/5=0.2、r=γ*q=[0,-0.2]。
        // beta=rho*(y・r)=1*(1*0+2*(-0.2))=-0.4、r+=s*(alpha-beta)=[1,0]*(2-(-0.4))=[2.4,0]
        // → r=[2.4,-0.2]。方向=-r=[-2.4,0.2]。
        let mut s_history = VecDeque::new();
        s_history.push_back(vec![1.0, 0.0]);
        let mut y_history = VecDeque::new();
        y_history.push_back(vec![1.0, 2.0]);
        let grad = vec![2.0, 3.0];

        let direction = two_loop_recursion(&s_history, &y_history, &grad, None);
        assert!(
            (direction[0] - (-2.4)).abs() < 1e-10,
            "direction={direction:?}"
        );
        assert!(
            (direction[1] - 0.2).abs() < 1e-10,
            "direction={direction:?}"
        );
    }

    /// `dot(y,s)<=f64::EPSILON`（ゼロ除算につながる退化ペア）を履歴の**最古**の位置に
    /// 混ぜても、そのペアの寄与が黙って無視され、残りの正常なペアだけで
    /// `two_loop_recursion_matches_hand_computed_values_for_one_history_pair`と
    /// 完全に同じ結果になるはず（`FaerLbfgs`は`FaerBfgs`と異なり
    /// secant条件を満たさないペアも履歴にそのまま追加する設計のため、
    /// two-loop recursion自体がこの種のペアに対して頑健である必要がある）。
    #[test]
    fn two_loop_recursion_skips_degenerate_pair_with_near_zero_curvature() {
        let mut s_history = VecDeque::new();
        // 最古のペア: y=[0,0]なので dot(y,s)=0（退化、無視されるはず）。
        s_history.push_back(vec![5.0, -3.0]);
        // 最新のペア: 上の単一ペアテストと同じ値。
        s_history.push_back(vec![1.0, 0.0]);
        let mut y_history = VecDeque::new();
        y_history.push_back(vec![0.0, 0.0]);
        y_history.push_back(vec![1.0, 2.0]);
        let grad = vec![2.0, 3.0];

        let direction = two_loop_recursion(&s_history, &y_history, &grad, None);
        assert!(
            (direction[0] - (-2.4)).abs() < 1e-10,
            "degenerate pair should be fully transparent: direction={direction:?}"
        );
        assert!(
            (direction[1] - 0.2).abs() < 1e-10,
            "degenerate pair should be fully transparent: direction={direction:?}"
        );
    }

    #[test]
    fn two_loop_recursion_falls_back_to_gamma_one_when_y_norm_is_too_small_to_scale() {
        // yᵀy=1e-20はf64::EPSILON以下（ゼロ除算ガードが発火する）ため、γ=1.0
        // （スケーリング無し）にフォールバックするはず。`bfgs_updated_inv_hessian_
        // first_iteration_uses_prior_when_y_norm_is_too_small_to_scale`と同じ発想の
        // 退化ケース。
        //
        // s=[1,0]・y=[1e-10,0]・grad=[2,3]（第2成分は`y`と直交させ、丸め誤差の影響を
        // 受けない値で検証できるようにしている）。手計算（γ=1.0フォールバック前提）:
        // rho=1/(y・s)=1e10、alpha=rho*(s・grad)=2e10、q=grad-alpha*y=[2,3]-2e10*[1e-10,0]
        // ≈[0,3]（第1成分は桁落ちが乗るが第2成分は`y[1]=0`のため厳密に不変）。
        // γ=1.0（フォールバック）、r=γ*q≈[0,3]。beta=rho*(y・r)=1e10*(1e-10*r[0]+0*3)、
        // r[1]+=s[1]*(alpha-beta)=0*(...)=0 → r[1]=3（厳密）。方向[1]=-r[1]=-3。
        let mut s_history = VecDeque::new();
        s_history.push_back(vec![1.0, 0.0]);
        let mut y_history = VecDeque::new();
        y_history.push_back(vec![1e-10, 0.0]);
        let grad = vec![2.0, 3.0];

        let direction = two_loop_recursion(&s_history, &y_history, &grad, None);
        assert_eq!(
            direction[1], -3.0,
            "gamma=1.0 fallback should leave the y-orthogonal component exact: {direction:?}"
        );
    }

    /// `FaerLbfgs`も`FaerBfgs`と同じく、初期点のHessianが正定値ならその厳密な逆行列を初期行列
    /// にするため、二次関数は初期スケールの大きな食い違い（初期点`(1000,-1000)`）があっても
    /// 1〜2反復で収束する（`γI`からの出発では、履歴がHessianの形を学習するのに`k`反復以上要る）。
    #[test]
    fn run_solver_lbfgs_uses_exact_inverse_hessian_at_initial_point_when_positive_definite() {
        let output = run_solver(
            quadratic_problem(),
            SolverType::Lbfgs,
            vec![1000.0, -1000.0],
            100,
            1e-9,
            1,
            true,
        )
        .unwrap();

        assert!(output.converged);
        assert!(output.n_iter <= 2, "n_iter={}", output.n_iter);
    }

    /// 初期点のHessianが正定値でない場合は従来の初期化（`γI`と1回目のline search初期ステップ幅の
    /// 調整）に戻り、`H⁻¹`が降下方向を作らずline searchが失敗する、という事態を避けて極小点`x=1`へ
    /// 収束する（`run_solver_bfgs_falls_back_when_hessian_at_initial_point_is_indefinite`の`lbfgs`版）。
    #[test]
    fn run_solver_lbfgs_falls_back_when_hessian_at_initial_point_is_indefinite() {
        let output = run_solver(Quartic, SolverType::Lbfgs, vec![0.1], 100, 1e-9, 1, true).unwrap();

        assert!(output.converged);
        assert!((output.params[0] - 1.0).abs() < 1e-6, "{:?}", output.params);
    }

    /// 初期行列`H₀⁻¹`を渡したtwo-loop recursionの手計算。履歴が空（1回目の反復）なら
    /// 再スケールせず`-H₀⁻¹g`。
    #[test]
    fn two_loop_recursion_with_initial_matrix_returns_minus_h0_times_gradient_when_history_is_empty()
     {
        let h0 = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let direction = two_loop_recursion(
            &VecDeque::new(),
            &VecDeque::new(),
            &[1.0, 1.0],
            Some(h0.as_slice()),
        );
        assert_eq!(direction, vec![-2.0, -4.0]);
    }

    /// 1組のペアがあるとき、`H₀⁻¹`を`sᵀy/(yᵀH₀⁻¹y)`倍に再スケールして使う。
    /// `s=[1,0]`・`y=[1,2]`・`H₀⁻¹=diag(2,1)`・`g=[3,1]`:
    /// `ρ=1/sᵀy=1`、`α=ρsᵀg=3`、`q=g-αy=[0,-5]`、`yᵀH₀⁻¹y=2+4=6`なのでスケール`1/6`、
    /// `r=(1/6)H₀⁻¹q=[0,-5/6]`、`β=ρyᵀr=-5/3`、`r+=s(α-β)`で`r=[14/3,-5/6]`、方向`-r`。
    #[test]
    fn two_loop_recursion_with_initial_matrix_rescales_it_by_the_latest_pair() {
        let h0 = vec![vec![2.0, 0.0], vec![0.0, 1.0]];
        let s_history = VecDeque::from(vec![vec![1.0, 0.0]]);
        let y_history = VecDeque::from(vec![vec![1.0, 2.0]]);

        let direction =
            two_loop_recursion(&s_history, &y_history, &[3.0, 1.0], Some(h0.as_slice()));

        assert!((direction[0] + 14.0 / 3.0).abs() < 1e-12, "{direction:?}");
        assert!((direction[1] - 5.0 / 6.0).abs() < 1e-12, "{direction:?}");
    }

    /// 割線方程式`H_new y = s`は初期行列のスケールによらず成り立つ（再スケールが
    /// two-loop recursionの割線条件を壊さないことの確認）。`g=y`なら`-direction=s`。
    #[test]
    fn two_loop_recursion_with_initial_matrix_satisfies_the_secant_equation() {
        let h0 = vec![vec![3.0, 1.0], vec![1.0, 2.0]];
        let s = vec![0.5, -1.0];
        let y = vec![1.5, 0.25];
        let s_history = VecDeque::from(vec![s.clone()]);
        let y_history = VecDeque::from(vec![y.clone()]);

        let direction = two_loop_recursion(&s_history, &y_history, &y, Some(h0.as_slice()));

        for (d, s_j) in direction.iter().zip(&s) {
            assert!((d + s_j).abs() < 1e-12, "{direction:?} vs {s:?}");
        }
    }

    /// 履歴`pairs`（古い順）と初期行列`h0`（再スケール済み）から、密な逆Hessian近似を
    /// `bfgs_rank2_update`で古いペアから順に更新して`-H g`を返す（two-loop recursionの独立な参照実装）。
    fn dense_lbfgs_direction(
        h0: &[Vec<f64>],
        pairs: &[(Vec<f64>, Vec<f64>)],
        g: &[f64],
    ) -> Vec<f64> {
        let mut h = h0.to_vec();
        for (s, y) in pairs {
            h = bfgs_rank2_update(&h, s, y, 1.0 / dot(y, s));
        }
        h.iter().map(|row| -dot(row, g)).collect()
    }

    fn scaled(h0: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
        h0.iter()
            .map(|row| row.iter().map(|v| v * scale).collect())
            .collect()
    }

    fn assert_close_relative(actual: &[f64], expected: &[f64], rtol: f64) {
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= rtol * (1.0 + e.abs()),
                "{actual:?} vs {expected:?}"
            );
        }
    }

    /// 履歴が2ペア以上のとき、再スケールは**直近のペア**（`sᵀy/(yᵀH₀⁻¹y)`）から決まり、結果は
    /// 「再スケールした`H₀⁻¹`に古いペアから順にBFGSのrank-2更新を適用した密な行列」×勾配と一致する。
    /// 直近のペアではなく最古のペアから決めると、このデータでは別の値になる。
    #[test]
    fn two_loop_recursion_with_initial_matrix_matches_dense_bfgs_update_with_several_pairs() {
        // 3次元で2ペア: 次元がペア数以下だと割線方程式が逆Hessianを一意に決め、初期行列の
        // スケールが結果に現れない（2次元・2ペアでは区別できない）。
        let h0 = vec![
            vec![3.0, 1.0, 0.0],
            vec![1.0, 2.0, 0.5],
            vec![0.0, 0.5, 1.5],
        ];
        let pairs = vec![
            (vec![1.0, 0.0, 0.5], vec![2.0, 1.0, 0.3]),
            (vec![0.5, -1.0, 0.2], vec![1.5, 0.25, -0.1]),
        ];
        let g = [0.7, -1.3, 0.4];
        let s_history: VecDeque<Vec<f64>> = pairs.iter().map(|(s, _)| s.clone()).collect();
        let y_history: VecDeque<Vec<f64>> = pairs.iter().map(|(_, y)| y.clone()).collect();

        let scale_from = |(s, y): &(Vec<f64>, Vec<f64>)| {
            let h0_y: Vec<f64> = h0.iter().map(|row| dot(row, y)).collect();
            dot(s, y) / dot(y, &h0_y)
        };
        let latest_scale = scale_from(&pairs[1]);
        let oldest_scale = scale_from(&pairs[0]);
        assert!(
            (latest_scale - oldest_scale).abs() > 1e-3,
            "テストデータの前提"
        );

        let direction = two_loop_recursion(&s_history, &y_history, &g, Some(h0.as_slice()));
        let expected = dense_lbfgs_direction(&scaled(&h0, latest_scale), &pairs, &g);
        assert_close_relative(&direction, &expected, 1e-12);

        let wrong = dense_lbfgs_direction(&scaled(&h0, oldest_scale), &pairs, &g);
        assert!(
            direction
                .iter()
                .zip(&wrong)
                .any(|(a, w)| (a - w).abs() > 1e-6),
            "最古のペアから決めたスケールと区別できない"
        );
    }

    /// 直近のペアだけが縮退（`sᵀy≤ε`）し、それ以前に有効なペアがあるとき、縮退したペアは
    /// two-loop recursionが無視し、再スケールもしない（`1.0`）。結果は有効な古いペアだけを
    /// `H₀⁻¹`に適用した密な行列と一致する。
    #[test]
    fn two_loop_recursion_with_initial_matrix_ignores_a_degenerate_latest_pair() {
        let h0 = vec![vec![3.0, 1.0], vec![1.0, 2.0]];
        let valid = (vec![1.0, 0.0], vec![2.0, 1.0]);
        let degenerate = (vec![1.0, 0.0], vec![-1.0, 0.0]);
        let g = [0.7, -1.3];
        let s_history = VecDeque::from(vec![valid.0.clone(), degenerate.0.clone()]);
        let y_history = VecDeque::from(vec![valid.1.clone(), degenerate.1.clone()]);

        let direction = two_loop_recursion(&s_history, &y_history, &g, Some(h0.as_slice()));
        let expected = dense_lbfgs_direction(&h0, &[valid], &g);

        assert_close_relative(&direction, &expected, 1e-12);
    }

    /// `sᵀy`は十分正だが`yᵀH₀⁻¹y≤ε`（`y`が極端に小さい）の場合も再スケールしない（`1.0`）。
    /// `sᵀy>ε`のガードとは独立に、`yᵀH₀⁻¹y`側のガード（ゼロ除算回避）を通る。
    #[test]
    fn two_loop_recursion_with_initial_matrix_does_not_rescale_when_y_h0_y_is_negligible() {
        let h0 = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let pair = (vec![1.0, 1.0], vec![5e-9, 5e-9]);
        assert!(dot(&pair.0, &pair.1) > f64::EPSILON, "テストデータの前提");
        let h0_y: Vec<f64> = h0.iter().map(|row| dot(row, &pair.1)).collect();
        assert!(dot(&pair.1, &h0_y) <= f64::EPSILON, "テストデータの前提");
        let g = [1.0, -2.0];
        let s_history = VecDeque::from(vec![pair.0.clone()]);
        let y_history = VecDeque::from(vec![pair.1.clone()]);

        let direction = two_loop_recursion(&s_history, &y_history, &g, Some(h0.as_slice()));
        let expected = dense_lbfgs_direction(&h0, &[pair], &g);

        assert_close_relative(&direction, &expected, 1e-8);
    }

    /// `run_solver_lbfgs_falls_back_...`の補足: 初期点のHessianは正定値だが`f''`がほぼ0で
    /// 最初の`H₀⁻¹`が巨大になる初期点（`x=0.6`、`f''=3x²-1≈0.08`）でも、再スケールと強Wolfe条件の
    /// 下で極小点`x=1`へ収束する（固定の`H₀⁻¹`が反復途中で不適切になっても破綻しない）。
    #[test]
    fn run_solver_lbfgs_converges_from_a_start_where_the_initial_inverse_hessian_is_huge() {
        for x0 in [0.6, 0.7, -0.6, 3.0] {
            let output =
                run_solver(Quartic, SolverType::Lbfgs, vec![x0], 100, 1e-9, 1, true).unwrap();

            assert!(output.converged, "x0={x0}");
            assert!(
                (output.params[0].abs() - 1.0).abs() < 1e-6,
                "x0={x0}, {:?}",
                output.params
            );
        }
    }

    /// 直近のペアの曲率が実質ゼロ以下（`sᵀy≤ε`）なら再スケールせず（`1.0`）、そのペアは
    /// two-loop recursion自体が無視する（`two_loop_recursion_skips_degenerate_pair_...`と同じ
    /// ガード）ため、方向は`-H₀⁻¹g`になる。
    #[test]
    fn two_loop_recursion_with_initial_matrix_does_not_rescale_for_a_degenerate_pair() {
        let h0 = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let s_history = VecDeque::from(vec![vec![1.0, 0.0]]);
        let y_history = VecDeque::from(vec![vec![-1.0, 0.0]]);

        let direction =
            two_loop_recursion(&s_history, &y_history, &[1.0, 1.0], Some(h0.as_slice()));

        assert_eq!(direction, vec![-2.0, -4.0]);
    }

    #[test]
    fn run_solver_lbfgs_converges_to_known_minimum() {
        let output = run_solver(
            quadratic_problem(),
            SolverType::Lbfgs,
            vec![0.0, 0.0],
            100,
            1e-6,
            1,
            true,
        )
        .unwrap();

        assert!(output.converged);
        assert!((output.params[0] - 3.0).abs() < 1e-4, "{:?}", output.params);
        assert!(
            (output.params[1] - (-2.0)).abs() < 1e-4,
            "{:?}",
            output.params
        );
    }
}
