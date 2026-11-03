//! 自前のBFGS（`FaerBfgs`）とrank-2更新。

use argmin::core::TerminationStatus;
use argmin::core::{
    CostFunction, Error as OptimizerError, Executor, Gradient, Hessian, IterState, KV, LineSearch,
    Problem, Solver, State, TerminationReason,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;

use super::line_search::{
    NEAR_TARGET_LINE_SEARCH_MAX_ITERS, is_near_convergence_target, line_search_converged,
    line_search_step_satisfies_wolfe,
};
use super::numerics::{
    dot, identity_matrix, inverse_if_positive_definite, l2_norm, mat_mat, mat_vec, scaled_identity,
    transpose,
};

/// argmin組み込みのBFGS（`argmin::solver::quasinewton::BFGS`）は、外側反復をまたぐ
/// line searchの初期ステップ幅・逆Hessianの初期スケールを実行中に調整する手段を
/// 公開していない（`linesearch`フィールドはprivateで、`initial_step_length`は
/// 一度設定すると全反復で同じ固定値が使われ続ける）。単位行列を初期逆Hessianにすると、
/// Logit/Probit等の尤度Hessianのスケール（観測数`n`個のスコアの和で`O(n)`）との乖離が
/// `n`が大きいほど桁違いに開き、最初の探索方向`d=-Ig`が暴走してline searchが1反復
/// あたり多数の関数評価を消費する（実測: n=100,000→1,000,000でNewtonの
/// 反復回数は4のまま一定なのに対し、BFGSは15→22と増加）。
///
/// 標準的な対策（Nocedal & Wright *Numerical Optimization* 6.1節のself-scaling初期化）
/// は、「1回目の反復でline searchが実際に受理したステップ」から秘密方程式ペア
/// `(s₀,y₀)`を作り、`γ=(y₀ᵀs₀)/(y₀ᵀy₀)`で初期逆Hessianをスケーリングしてから通常の
/// BFGS rank-2更新を適用する、というもの。これはargmin自身の`BFGS::next_iter`に
/// コメントアウトされた形で存在する（`self.inv_hessian`という現存しないフィールドを
/// 参照しており、リファクタリングの過程で壊れたまま放置されている）。この手法を
/// 使うには「1回目の反復の終わり（rank-2更新の直前）」というタイミングにフックする
/// 必要があり、argminの公開APIには存在しないため、`FaerNewton`と同じ理由（組み込み
/// ソルバーが必要な制御点を公開していない）で自前実装する。
///
/// あわせて、1回目の反復専用のline search初期ステップ幅も`min(1, 1/‖g₀‖)`に調整する
/// （単位行列の逆Hessianによる探索方向`-g₀`は`n`が大きいほど大きくなるため、
/// `alpha=1`から始めて大きくバックトラックする代わりに、あらかじめ妥当な大きさへ
/// 抑える）。2回目以降の反復は、既にrank-2更新でスケールが補正された逆Hessianにより
/// 探索方向自体が適切な大きさになっているはずなので、標準の`alpha=1.0`に戻す
/// （固定値のまま反復間で使い回すと、スケール補正済みの反復まで不必要に小さい
/// ステップから始めることになり逆効果になりうるため）。
///
/// **warm startとの相互作用（初期逆Hessianを初期点の厳密Hessianの逆行列にする）**:
/// Logit/Probitはwarm start（`ols_based_initial_params`）で最適点の近くから始まるが、
/// 上記の`min(1,1/‖g₀‖)`は1回目のステップ長を初期点の位置によらず常に`‖s₀‖=1`
/// （標準化パラメータ空間）にする。ゼロ初期値（最適点まで距離およそ1）には適切だが、
/// warm startでは過大で、最初のsecantペアが最適点から遠い領域の曲率を拾うため`γ`が
/// 真の逆曲率から外れ、超1次収束に入るまでの反復が増える（実測: 反復数が
/// Logit n=100,000で8→13、Probit n=1,000,000で8→10。line searchの評価回数は
/// 変わらず原因ではない）。初期点で`H`を1回評価して`H⁻¹`を初期逆Hessianにすると、
/// 1回目のステップはほぼNewtonステップになりこの問題が消える（同条件で13→6、10→7）。
/// `H`が正定値でない場合（Tobitのように尤度が大域凹でない手法）は`H⁻¹`が降下方向を
/// 作らないため、従来の単位行列＋self-scalingに戻す（[`inverse_if_positive_definite`]）。
/// `H`を「nullモデルの`w·X'X`」で代用する案は、Hessianの評価が不要になる反面、反復数が
/// 改善せず（Logit n=1,000,000では悪化）不採用。
///
/// **`SolverType::Lbfgs`は当初対象外だったが、現在は解消済み**: 導入当初は
/// argmin 0.11.0の組み込みLBFGS実装が`s`/`y`履歴・初期`γ`を外部から注入する公開APIを
/// 持たず（privateフィールド、対応するビルダーメソッド無し）、同じ手法を適用できな
/// かった。その後`SolverType::Lbfgs`自体を自前実装`FaerLbfgs`に置き換え、
/// `FaerBfgs`と同じ「`self`がline searchを所有し1回目の反復だけ初期ステップ幅を
/// 切り替える」制御を獲得したことで解消した（詳細は`FaerLbfgs`のdocコメント参照）。
pub(super) struct FaerBfgs {
    /// line search。`self`が所有し反復間で使い回す（`initial_step_length`で1回目の
    /// 反復だけ特別なステップ幅を設定し、2回目以降は標準値に戻す、という制御を行う
    /// ため。built-inのBFGSは`self.linesearch.clone()`を反復ごとに使い捨てるだけで、
    /// `self.linesearch`自体を反復間で更新する経路が無い）。
    pub(super) linesearch: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>,
    pub(super) tol: f64,
    /// `next_iter`でline searchがstrong Wolfe条件を満たせずに終了し、かつ勾配ノルムが
    /// 収束目標の近傍にあった（コストが浮動小数点の底に達し、これ以上改善できない）ことを
    /// 示すフラグ。`terminate`で収束として扱う（`QUASI_NEWTON_STALL_GRAD_FACTOR`の
    /// docコメント参照、`FaerNewton::stalled_at_optimum`と同じ位置づけ）。
    pub(super) stalled_at_optimum: bool,
    /// `init`で初期逆Hessianをwarm start点の厳密Hessianの逆行列にしたか。`true`のとき
    /// 1回目の反復のself-scaling（`bfgs_updated_inv_hessian`の`is_first_iter`）は行わない
    /// （既にスケール・形状とも曲率に合っているため。再スケールすると情報を捨てる）。
    pub(super) h0_from_hessian: bool,
}

type BfgsState = IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>;

impl<O> Solver<O, BfgsState> for FaerBfgs
where
    O: CostFunction<Param = Vec<f64>, Output = f64>
        + Gradient<Param = Vec<f64>, Gradient = Vec<f64>>
        + Hessian<Param = Vec<f64>, Hessian = Vec<Vec<f64>>>,
{
    /// `argmin::core::Solver`トレイトの必須メソッド（`FaerNewton::name`と同じ理由で
    /// 未カバーでも振る舞いの正しさに影響しない）。
    fn name(&self) -> &str {
        "BFGS (faer-backed)"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: BfgsState,
    ) -> Result<(BfgsState, Option<KV>), OptimizerError> {
        let param = state.take_param().ok_or_else(|| {
            OptimizerError::msg(
                "FaerBfgs requires an initial parameter vector via Executor's configure method",
            )
        })?;
        let grad = problem.gradient(&param)?;

        // 初期点（warm start）の厳密Hessianが正定値なら、その逆行列を初期逆Hessianにする
        // （`FaerBfgs`のdocコメント「warm startとの相互作用」参照）。1回目のステップは
        // ほぼNewtonステップになるため、初期ステップ幅は標準の`1.0`のままでよい。
        if let Some(inv_hessian) = inverse_if_positive_definite(&problem.hessian(&param)?) {
            self.h0_from_hessian = true;
            let state = state.param(param).gradient(grad).inv_hessian(inv_hessian);
            return Ok((state, None));
        }

        // 初期点のHessianが正定値でない（Tobit等、尤度が大域凹でない手法のwarm start点）
        // 場合の従来の初期化。1回目の反復専用のline search初期ステップ幅（`FaerBfgs`の
        // docコメント参照）。`g0_norm`が実質ゼロ（既に停留点付近）なら`1.0/g0_norm`は
        // `f64::INFINITY`になり`min(1.0)`で1.0（＝標準値）に収まるため、この場合は特別扱い
        // 不要で自然に既定動作へ落ちる。
        let alpha0 = (1.0 / l2_norm(&grad)).min(1.0);
        if alpha0.is_finite() && alpha0 > 0.0 {
            self.linesearch.initial_step_length(alpha0)?;
        }

        let k = param.len();
        let state = state
            .param(param)
            .gradient(grad)
            .inv_hessian(identity_matrix(k));
        Ok((state, None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: BfgsState,
    ) -> Result<(BfgsState, Option<KV>), OptimizerError> {
        let param = state
            .take_param()
            .ok_or_else(|| OptimizerError::msg("FaerBfgs: parameter vector in state not set"))?;
        let cur_cost = state.get_cost();
        let prev_grad = state
            .take_gradient()
            .ok_or_else(|| OptimizerError::msg("FaerBfgs: gradient in state not set"))?;
        let inv_hessian = state
            .take_inv_hessian()
            .ok_or_else(|| OptimizerError::msg("FaerBfgs: inverse Hessian in state not set"))?;
        // `terminate`が「1回目の反復だけの特別扱い」を後段で正しく戻せるよう、
        // rank-2更新前に判定しておく（`next_iter`本体の途中でstateの反復カウントが
        // 変わることはない）。
        let is_first_iter = state.get_iter() == 0;

        let direction: Vec<f64> = mat_vec(&inv_hessian, &prev_grad)
            .into_iter()
            .map(|d| -d)
            .collect();
        self.linesearch.search_direction(direction);

        // `take_problem()`が`None`になるのは既に一度取り出した後に再度取り出した場合
        // のみだが、`next_iter`はこの箇所でしか呼ばない（`FaerNewton`と同様、argmin
        // 組み込みソルバー自体も同じ契約に依存して`.unwrap()`している）。
        let inner_problem = problem.take_problem().ok_or_else(|| {
            OptimizerError::msg("FaerBfgs: failed to recover the optimization problem")
        })?;
        // 収束目標の近傍ではline searchの反復上限を絞り、上限到達は停滞とみなす
        // （`NEAR_TARGET_LINE_SEARCH_MAX_ITERS`のdocコメント参照）。近傍以外では従来どおり
        // 上限を設けない（`BudgetedProblem`の総枠のみ）。
        let near_target = is_near_convergence_target(&prev_grad, self.tol);
        let result = Executor::new(inner_problem, self.linesearch.clone())
            .configure(|config| {
                let config = config
                    .param(param.clone())
                    .gradient(prev_grad.clone())
                    .cost(cur_cost);
                if near_target {
                    config.max_iters(NEAR_TARGET_LINE_SEARCH_MAX_ITERS)
                } else {
                    config
                }
            })
            .ctrlc(false)
            .run()?;
        let mut sub_state = result.state;
        let line_problem = result.problem;
        problem.consume_problem(line_problem);

        if near_target && !line_search_converged(&sub_state) {
            // 打ち切り時点の試行点は採用せず、直前の反復点に留まる。
            self.stalled_at_optimum = true;
            return Ok((
                state
                    .param(param)
                    .cost(cur_cost)
                    .gradient(prev_grad)
                    .inv_hessian(inv_hessian),
                None,
            ));
        }

        let xk1 = sub_state.take_param().ok_or_else(|| {
            OptimizerError::msg("FaerBfgs: no parameters returned by line search")
        })?;
        let next_cost = sub_state.get_cost();

        let grad = problem.gradient(&xk1)?;

        // 1回目の反復専用のline search初期ステップ幅（`init`で設定）は、2回目以降は
        // 標準の1.0に戻す（`FaerBfgs`のdocコメント参照）。
        if is_first_iter {
            self.linesearch.initial_step_length(1.0)?;
        }
        // 初期逆Hessianを厳密Hessianの逆行列にした場合は、1回目もself-scalingしない。
        let apply_self_scaling = is_first_iter && !self.h0_from_hessian;

        let sk: Vec<f64> = xk1.iter().zip(param.iter()).map(|(a, b)| a - b).collect();
        let yk: Vec<f64> = grad
            .iter()
            .zip(prev_grad.iter())
            .map(|(a, b)| a - b)
            .collect();

        // line searchが条件を満たさないまま`SolverConverged`で返った（異常終了）場合の
        // 停滞判定（`line_search_step_satisfies_wolfe`のdocコメント参照）。
        self.stalled_at_optimum =
            !line_search_step_satisfies_wolfe(cur_cost, &prev_grad, next_cost, &grad, &sk)
                && is_near_convergence_target(&grad, self.tol);

        let updated_inv_hessian =
            bfgs_updated_inv_hessian(inv_hessian, &sk, &yk, apply_self_scaling);

        Ok((
            state
                .param(xk1)
                .cost(next_cost)
                .gradient(grad)
                .inv_hessian(updated_inv_hessian),
            None,
        ))
    }

    fn terminate(&mut self, state: &BfgsState) -> TerminationStatus {
        if let Some(g) = state.get_gradient()
            && l2_norm(g) < self.tol
        {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        // line searchが最適点近傍で進めなくなった（`stalled_at_optimum`のdocコメント参照）。
        if self.stalled_at_optimum {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        // コストが（ほぼ）変化しなくなった場合も収束扱いにする。built-inのBFGS
        // （`argmin::solver::quasinewton::BFGS`）が持つ副次的な収束判定
        // （既定`tol_cost=f64::EPSILON`）と同じ基準。通常のデータでは勾配ノルム基準
        // より先に発火することは無いが、near-separation等の退化した尤度面では
        // line searchがこれ以上進めずコストが完全に変化しなくなる一方、勾配ノルムが
        // まだ`tol`を下回らないことがある（実測: この基準を落とすとnear-separation
        // ・Tobitの一部データで`NonConvergence`になる回帰を確認したため、built-inと
        // 同じ基準を明示的に踏襲する）。
        if (state.get_prev_cost() - state.get_cost()).abs() < f64::EPSILON {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

/// BFGSの逆Hessian近似のrank-2更新: `H₊ = (I-ρsyᵀ)H(I-ρysᵀ)+ρssᵀ`
/// （`ρ=1/(yᵀs)`は呼び出し側が渡す）。`k`は小さい（Logit/Probit/Tobitのパラメータ数、
/// 実務的には数〜数十）ため、faerを介さず素朴な行列積で十分（`O(k³)`は無視できる
/// コスト）。
pub(super) fn bfgs_rank2_update(h: &[Vec<f64>], s: &[f64], y: &[f64], rho: f64) -> Vec<Vec<f64>> {
    let k = s.len();
    let v: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|m| (if i == m { 1.0 } else { 0.0 }) - rho * s[i] * y[m])
                .collect()
        })
        .collect();
    let vh = mat_mat(&v, h);
    let vt = transpose(&v);
    let mut result = mat_mat(&vh, &vt);
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell += rho * s[i] * s[j];
        }
    }
    result
}

/// `FaerBfgs::next_iter`が1回の反復の終わりに呼ぶ、逆Hessian近似の更新ロジック本体。
/// `next_iter`のargmin`Solver`実装（`Problem`/`IterState`の出し入れ、line searchの
/// 実行）から分離した純粋関数にすることで、rank-2更新・self-scaling・secant条件の
/// 安全策を単体テストで直接検証できるようにしている（`mod tests`の
/// `bfgs_updated_inv_hessian_*`参照）。
///
/// secant条件`yᵀs>0`を満たさない場合はrank-2更新を行わず、`inv_hessian`をそのまま
/// 返す（更新すると正定値性が壊れうるため。標準的なBFGSの安全策）。
///
/// **閾値は絶対値`f64::EPSILON`のまま（`‖s‖‖y‖`に対する相対値は不採用）**: 当初は
/// rust-style.md「線形代数」節の特異性判定の方針（データのスケールに依存しない相対閾値を使う）に倣い`yksk <= f64::EPSILON *
/// l2_norm(sk) * l2_norm(yk)`を試したが、`generate_binary_choice_dataset("baseline",
/// link="logit", n=1_000_000, k=5, seed=42)`で実測したところ、絶対閾値では反復
/// 6回目以降のrank-2更新が正常に適用され続けるのに対し、相対閾値では途中の反復で
/// 更新がスキップされる頻度が増え、収束までの反復回数・実行時間がかえって悪化した
/// （n_iter 17→23、実行時間 15.0s→20.7s）。secant条件は「行列の特異性判定」ではなく
/// 「BFGS更新の正定値性を保つための降下方向チェック」であり、既存の特異性判定
/// （QR分解のR対角成分等）とは性質が異なるため、この方針をそのまま転用するのは
/// 適切でなかったと判断し、絶対閾値に戻した。
///
/// `is_first_iter`が`true`の場合のみ、Nocedal & Wright *Numerical Optimization*
/// 6.1節のself-scaling初期化（`γ=(yᵀs)/(yᵀy)`で単位行列の代わりに`γI`を
/// 「更新前の逆Hessian」としてrank-2更新に使う）を行う（`FaerBfgs`のdocコメント
/// 参照）。`yᵀy`が実質ゼロ（縮退、勾配がほとんど変化しなかった）の場合は
/// `inv_hessian`をそのまま「更新前の逆Hessian」として使う（スケーリングを諦める
/// フォールバック。ゼロ除算を避けるためだけの絶対閾値ガード）。
fn bfgs_updated_inv_hessian(
    inv_hessian: Vec<Vec<f64>>,
    sk: &[f64],
    yk: &[f64],
    is_first_iter: bool,
) -> Vec<Vec<f64>> {
    let yksk = dot(yk, sk);
    if yksk <= f64::EPSILON {
        return inv_hessian;
    }
    let rho = 1.0 / yksk;
    let prior_h = if is_first_iter {
        let ykyk = dot(yk, yk);
        if ykyk > f64::EPSILON {
            scaled_identity(yksk / ykyk, sk.len())
        } else {
            inv_hessian
        }
    } else {
        inv_hessian
    };
    bfgs_rank2_update(&prior_h, sk, yk, rho)
}

#[cfg(test)]
mod tests {

    use super::super::options::SolverType;
    use super::super::solver::run_solver;
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn run_solver_bfgs_converges_to_known_minimum() {
        let output = run_solver(
            quadratic_problem(),
            SolverType::Bfgs,
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

    #[test]
    fn bfgs_rank2_update_matches_hand_computed_values_for_non_diagonal_case() {
        // H=[[2,1],[1,3]]（対称正定値・非対角）、s=[1,0]、y=[1,1]、rho=1/(y・s)=1。
        // V=I-rho*s*y^T=[[0,-1],[0,1]]、V*H=[[-1,-3],[1,3]]、V*H*V^T=[[3,-3],[-3,3]]、
        // +rho*s*s^T=[[1,0],[0,0]] で H+=[[4,-3],[-3,3]]（手計算で検証済み）。
        let h = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let s = vec![1.0, 0.0];
        let y = vec![1.0, 1.0];
        let rho = 1.0 / dot(&y, &s);

        let updated = bfgs_rank2_update(&h, &s, &y, rho);
        let expected = vec![vec![4.0, -3.0], vec![-3.0, 3.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (updated[i][j] - expected[i][j]).abs() < 1e-10,
                    "updated={updated:?}, expected={expected:?}"
                );
            }
        }
    }

    #[test]
    fn bfgs_updated_inv_hessian_skips_update_when_secant_condition_violated() {
        let inv_hessian = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // yᵀs = -1 < 0（secant条件`yᵀs>0`を満たさない）ため、rank-2更新をせず
        // `inv_hessian`をそのまま返すはず。
        let s = vec![1.0, 0.0];
        let y = vec![-1.0, 0.0];

        let updated = bfgs_updated_inv_hessian(inv_hessian.clone(), &s, &y, false);
        assert_eq!(updated, inv_hessian);
    }

    /// 初期点のHessianが正定値なら初期逆Hessianが厳密になり、二次関数は初期スケールの
    /// 大きな食い違い（初期点`(1000,-1000)`）があっても1〜2反復で収束する
    /// （単位行列からの出発では、rank-2更新がHessianを学習するのに`k`反復以上要る）。
    #[test]
    fn run_solver_bfgs_uses_exact_inverse_hessian_at_initial_point_when_positive_definite() {
        let output = run_solver(
            quadratic_problem(),
            SolverType::Bfgs,
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

    /// 初期点のHessianが正定値でない（`f(x)=x⁴/4-x²/2`は`|x|<1/√3`で`f''<0`）場合は
    /// 従来の初期化（単位行列＋self-scaling）に戻り、`H⁻¹`が降下方向を作らず
    /// line searchが失敗する、という事態を避けて極小点`x=1`へ収束する。
    #[test]
    fn run_solver_bfgs_falls_back_when_hessian_at_initial_point_is_indefinite() {
        let output = run_solver(Quartic, SolverType::Bfgs, vec![0.1], 100, 1e-9, 1, true).unwrap();

        assert!(output.converged);
        assert!((output.params[0] - 1.0).abs() < 1e-6, "{:?}", output.params);
    }

    #[test]
    fn bfgs_updated_inv_hessian_applies_self_scaling_gamma_on_first_iteration() {
        // yᵀs=1・yᵀy=2 → γ=0.5。1回目の反復では`inv_hessian`（ここでは意図的に
        // γIとは異なる値にしている）を使わず、`scaled_identity(0.5, 2)`を
        // 「更新前の逆Hessian」としてrank-2更新するはず。
        let inv_hessian = vec![vec![7.0, 0.0], vec![0.0, 7.0]];
        let s = vec![1.0, 0.0];
        let y = vec![1.0, 1.0];

        let updated = bfgs_updated_inv_hessian(inv_hessian, &s, &y, true);
        let expected = bfgs_rank2_update(&scaled_identity(0.5, 2), &s, &y, 1.0);
        assert_eq!(updated, expected);
    }

    #[test]
    fn bfgs_updated_inv_hessian_does_not_apply_self_scaling_after_first_iteration() {
        // 同じ`(s,y)`でも`is_first_iter=false`なら、γIではなく渡された`inv_hessian`
        // をそのまま「更新前の逆Hessian」として使うはず。
        let inv_hessian = vec![vec![7.0, 0.0], vec![0.0, 7.0]];
        let s = vec![1.0, 0.0];
        let y = vec![1.0, 1.0];

        let updated = bfgs_updated_inv_hessian(inv_hessian.clone(), &s, &y, false);
        let expected = bfgs_rank2_update(&inv_hessian, &s, &y, 1.0 / dot(&y, &s));
        assert_eq!(updated, expected);
    }

    #[test]
    fn bfgs_updated_inv_hessian_first_iteration_uses_prior_when_y_norm_is_too_small_to_scale() {
        // yᵀy=1e-20はf64::EPSILON以下（self-scalingのゼロ除算ガードが発火する）が、
        // yᵀs=1e-10はsecant条件の絶対閾値`f64::EPSILON`は上回る（rank-2更新自体は
        // スキップされない）。この場合`inv_hessian`をスケーリングせずそのまま
        // 「更新前の逆Hessian」として使うはず。
        let inv_hessian = vec![vec![9.0, 0.0], vec![0.0, 9.0]];
        let s = vec![1.0, 0.0];
        let y = vec![1e-10, 0.0];

        let updated = bfgs_updated_inv_hessian(inv_hessian.clone(), &s, &y, true);
        let yksk = dot(&y, &s);
        let expected = bfgs_rank2_update(&inv_hessian, &s, &y, 1.0 / yksk);
        assert_eq!(updated, expected);
    }
}
