//! 自前のNewton法（`FaerNewton`）。Levenberg-Marquardt型の正則化と、収束点近傍の停滞収束判定を持つ。

use argmin::core::TerminationStatus;
use argmin::core::{
    CostFunction, Error as OptimizerError, Gradient, Hessian, IterState, KV, Problem, Solver,
    TerminationReason,
};
use faer::Mat;
use faer::prelude::SolveLstsq;

use super::error::MleError;
use super::numerics::{cost_hessian_is_positive_definite, l2_norm};
use crate::shared::linear_algebra::checked_col_piv_qr;

/// argmin組み込みの`Newton`ソルバーは`H: ArgminInv<H>`（Hessianの逆行列）を要求するが、
/// `argmin-math`の`vec`機能（`Vec<Vec<f64>>`向け）には`ArgminInv`の実装が存在しない
/// （faer/nalgebra/ndarrayの行列型にしか実装されていない）ため使えない。Newton法は独自の
/// `Solver`実装とし、ステップの求解はfaer（列ピボットQR、`linear_algebra::checked_col_piv_qr`の
/// 特異性検出）で行う（`docs/spec/nonlinear-common.md`1.2節参照）。
pub(super) struct FaerNewton {
    pub(super) tol: f64,
    /// `regularized_newton_step`が[`RegularizedStep::NoProgress`]または
    /// [`RegularizedStep::BelowCostResolution`]を返し（コスト比較ではステップの良否を
    /// 判定できなかった）、かつ`next_iter`が「最適点に到達しこれ以上進めない」と判定した
    /// ことを`terminate`へ伝えるフラグ。
    ///
    /// 主たる収束判定（`terminate`の`l2_norm(gradient) < tol`）は**総和勾配に対する絶対
    /// 閾値**であり観測数`n`でスケールしない。勾配の評価誤差の床が`tol`（既定`1e-6`）を
    /// 上回るほど大きな標本では、コスト比較でステップを検証できない状態のまま勾配基準が
    /// 永久に発火しないことがありうる。その状態で勾配の停滞・目標近傍・コストHessianの
    /// 正定値性（`next_iter`の3条件）がそろえば内点最大に到達しているため、このフラグ経由で
    /// 収束として扱う。
    ///
    /// **実データでは現状この経路は安全網**: かつてTobit `n=1_000_000`で「勾配の床≈`36·tol`」
    /// と記録していた張り付きは、勾配の床ではなかった。コストの素朴な逐次和の丸め誤差
    /// （ULPの約100倍）でraw Newtonステップが棄却され、LMラダーがノイズで受理した不完全な
    /// ステップの結果だった（raw Newtonステップを適用すれば勾配は`≈2e-10`まで下がる）。
    /// コストの補償和と[`RegularizedStep::BelowCostResolution`]の導入後は、Probit/Tobit
    /// `n=1_000_000`とも生ステップで勾配基準に到達し、このフラグは立たない。
    ///
    /// `next_iter`が毎反復書き換え、`terminate`が読む。argminの`Executor`が各反復で
    /// `next_iter`→`terminate`の順に呼ぶ契約に依存している（`terminate`が`next_iter`で
    /// 立てたフラグを同じ反復内で見られる前提。argminバージョン更新時はこの順序を要確認）。
    /// `FaerNewton`は`run_solver`内で`solver`ごとに毎回新規構築されるため、`run_solver`
    /// 呼び出しをまたぐフラグの持ち越しは無い。
    pub(super) stalled_at_optimum: bool,
}

type NewtonState = IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>;

/// [`regularized_newton_step`]がコスト減少の候補が見つからない場合に諦めるまでの
/// 最大試行回数（`INITIAL_LM_LAMBDA`から`LM_LAMBDA_GROWTH`倍ずつ`λ`を増やしながら試す）。
///
/// 値の根拠: `λ`は`INITIAL_LM_LAMBDA * LM_LAMBDA_GROWTH^(n-1)`と幾何級数的に増加するため、
/// `n=40`回目には`λ≈1e-3*4^39≈10^20`という天文学的な値に達する。十分大きな`λ`では
/// ステップが最急降下方向`-g/λ`に漸近し、非零の勾配に対しては理論上必ず降下方向になる
/// （`regularized_newton_step`のdocコメント参照）ため、実務的にはこれよりずっと少ない
/// 回数（Tobitの打ち切りデータでの実測では1反復あたり最大9回程度）で受理される。
/// 実測比較による厳密な根拠があるわけではなく、「理論上十分すぎるほど大きい」という
/// 設計上の安全マージンとして選んだ値。
const MAX_LM_ATTEMPTS: usize = 40;

/// `λ=0`（生のNewtonステップ）がコストを減少させなかった場合の初期正則化係数。
const INITIAL_LM_LAMBDA: f64 = 1e-3;

/// 各試行で`λ`を増やす倍率。
const LM_LAMBDA_GROWTH: f64 = 4.0;

/// [`RegularizedStep::NoProgress`]（LMラダーがコスト減少ステップを見つけられなかったが
/// `λ=0`のHessianは可逆）・[`RegularizedStep::BelowCostResolution`]（予測減少量がコストの
/// 分解能以下でコスト比較を省いた）を「最適点で停滞」と解釈するための条件その1:
/// 勾配ノルムがこの反復で「実質的に減っていない」とみなす比。
/// `‖g_new‖ ≥ NEWTON_STALL_GRAD_RATIO · ‖g_prev‖`のとき停滞とみなす。健全な二次収束では
/// 勾配が反復ごとに桁で減る（実測でも停滞前は比≈0.01）ため、この条件は
/// 停滞後（実測で連続する勾配ノルムの比≈0.93〜1.0）でしか満たされない。値`0.9`は
/// その2つの領域の間で、停滞側に十分な余裕を持たせた閾値。
const NEWTON_STALL_GRAD_RATIO: f64 = 0.9;

/// `NoProgress`/`BelowCostResolution`を「最適点で停滞」と解釈するための条件その2:
/// 収束点の勾配ノルムが収束目標`tol`のこの倍数未満であること。最適点から遠い場所での
/// 停滞・発散（勾配ノルムが桁違いに大きい）を収束と誤判定しないためのガード。
/// `1e4`は、勾配の評価誤差の床が`tol`を大きく上回る大標本でもカバーできる余裕を持たせつつ、
/// 最適化初期の勾配ノルム（実測で`1e2`〜`1e6`オーダー）は確実に除外できる値。
/// （導入時は膠着点の勾配ノルムを`n=200_000`で約`1·tol`、`n=1_000_000`で約`36·tol`と
/// 実測し「勾配の床」とみなしていたが、これはコストの丸め誤差でLMラダーがノイズを
/// 受理した結果で、真の床ではなかった。[`FaerNewton::stalled_at_optimum`]のdocコメント参照。）
///
/// この「初期の勾配ノルムは`1e2`〜`1e6`」という前提は、`run_solver`の呼び出し元
/// （Logit/Probitの`standardize_columns`、Tobitの`TobitScaling`。どちらも`ColumnScale`で
/// `x`を標準化する）が**設計行列を標準化した
/// 空間で最適化する**ことに依存している。非標準化スケール（`x`が極端に大きい等）で
/// `run_solver`を呼ぶ手法を将来追加する場合、この定数の妥当性は再検証が必要
/// （より根本的にはNewton減少量`√(gᵀH⁻¹g)`のようなスケール不変な
/// 停止基準への置き換えが望ましい。`docs/spec/nonlinear-common.md`9章）。
const NEWTON_STALL_GRAD_FACTOR: f64 = 1e4;

/// raw Newtonステップの予測コスト減少量`½gᵀH⁻¹g`（局所2次モデルでの減少量）が
/// `NEWTON_COST_RESOLUTION_FACTOR · ε · |cost|`以下なら、コスト比較ではステップの良否を
/// 判定できない（コストの評価値の分解能以下）とみなす
/// （[`RegularizedStep::BelowCostResolution`]、`regularized_newton_step`のdocコメント参照）。
///
/// 値の根拠: 対数尤度の総和を補償和（[`super::numerics::compensated_sum`]）で計算すると、コストの評価誤差は
/// 実測で1 ULP（`ε·|cost|`）程度になる（Logit/Probitの対数尤度の項は全て同符号のため。
/// 項ごとの評価誤差の寄与も`ε·|cost|/√n`程度）。`10`はその誤差に対する余裕。これより
/// 大きい予測減少量ではコスト比較が意味を持つため、従来どおりLMラダーで検証する。
/// Tobitのように項の符号が混在し`|cost| ≪ Σ|ℓᵢ|`（強い相殺）になる場合は`ε·|cost|`が
/// 実際の評価誤差を過小評価するが、その場合はこの分岐が発火しにくくなって従来の
/// LMラダーに戻るだけで、安全側に倒れる。
///
/// コスト比較を省いてもよい理由（「予測減少量が小さい＝2次モデルが正確」ではない。
/// 2次モデルの精度はステップ長で決まるため）:
/// - (a) この分岐が発火するのはNewton減少量`δ² = gᵀH⁻¹g ≤ 2·NEWTON_COST_RESOLUTION_FACTOR·
///   ε·|cost|`の領域だけで、`H`（＝情報行列）ノルムで測ったステップ長は`δ`以下、つまり
///   各パラメータの標準誤差の`√(20ε|cost|)`倍程度（`|cost|=4e5`で約`4e-5`SE）に収まる。
/// - (b) 採ったステップは無検証ではない。`FaerNewton::next_iter`がステップ後の勾配で判定し、
///   `terminate`は勾配基準か停滞3条件でしか収束を宣言しないため、偽の収束は生じない。
/// - (c) 悪いステップで`δ²`が閾値を超えれば、次反復では従来どおりLMラダーの検証に戻る
///   （自己制限的）。
const NEWTON_COST_RESOLUTION_FACTOR: f64 = 10.0;

impl<O> Solver<O, NewtonState> for FaerNewton
where
    O: CostFunction<Param = Vec<f64>, Output = f64>
        + Gradient<Param = Vec<f64>, Gradient = Vec<f64>>
        + Hessian<Param = Vec<f64>, Hessian = Vec<Vec<f64>>>,
{
    /// `argmin::core::Solver`トレイトの必須メソッド（ロギング・エラーメッセージ表示等、
    /// argmin内部が使う識別名）。本プロジェクトはargminの`Observer`（進捗ロギング機構）を
    /// 使っておらず、`run_solver`のテストでも実行結果（`SolverOutput`）のみを検証するため
    /// 呼ばれない。実装自体は`argmin::solver::newton::Newton`等の組み込みソルバーに倣った
    /// 定型実装で、分岐を持たない単純な文字列リテラルの返却のため、未カバーでも
    /// 振る舞いの正しさに影響しない（判明・受け入れ済み）。
    fn name(&self) -> &str {
        "Newton (faer-backed)"
    }

    /// 初期パラメータでの勾配をあらかじめ`state`に格納する。これにより`terminate()`は
    /// 常に「`state.get_param()`と対応する勾配」を見られる（`next_iter`実行前の最初の
    /// `terminate()`呼び出しも含む。初期値が既に収束条件を満たす場合を正しく扱うため）。
    ///
    /// **`ok_or_else`分岐は理論上到達不能**（判明・受け入れ済み）:
    /// `run_solver`は必ず`Executor::configure(|state| state.param(initial_params)...)`
    /// で`init()`が呼ばれる前に初期パラメータを`state`へ設定しており、argminの
    /// `Executor::run()`はこの設定後に`init()`を呼ぶ契約のため、`state.take_param()`が
    /// `None`になることはない。
    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: NewtonState,
    ) -> Result<(NewtonState, Option<KV>), OptimizerError> {
        let param = state.take_param().ok_or_else(|| {
            OptimizerError::msg(
                "FaerNewton requires an initial parameter vector via Executor's configure method",
            )
        })?;
        let grad = problem.gradient(&param)?;
        let state = state.param(param).gradient(grad);
        Ok((state, None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: NewtonState,
    ) -> Result<(NewtonState, Option<KV>), OptimizerError> {
        let param = state
            .take_param()
            .ok_or_else(|| OptimizerError::msg("FaerNewton: parameter vector in state not set"))?;
        // `init()`（初回）または前回の`next_iter`（2回目以降）で、この`param`に対応する
        // 勾配が既に`state`に格納されている前提（`terminate()`が常に「返却されるparamと
        // 対応する勾配」を見られるよう、param/gradientを常にペアで更新する設計）。
        let grad = state
            .take_gradient()
            .ok_or_else(|| OptimizerError::msg("FaerNewton: gradient in state not set"))?;
        let hessian = problem.hessian(&param)?;
        let cost = problem.cost(&param)?;
        // 収束判定（terminate）が「更新後のparamに対応する勾配」を見られるよう、
        // 更新前のgradを使い回さずnew_paramで改めて評価する。
        let (new_param, new_grad) =
            match regularized_newton_step(problem, &param, &grad, &hessian, cost)? {
                RegularizedStep::Accepted(next) => {
                    let next_grad = problem.gradient(&next)?;
                    (next, next_grad)
                }
                RegularizedStep::NoProgress(raw_candidate)
                | RegularizedStep::BelowCostResolution(raw_candidate) => {
                    // コスト比較ではraw Newtonステップの良否を判定できなかった
                    // （`NoProgress`: LMラダーがコストを減少させるステップを1つも
                    // 見つけられなかった。`BelowCostResolution`: 予測減少量がコストの
                    // 評価値の分解能以下で、比較自体を省いた）。どちらも`λ=0`の
                    // Hessianは可逆なので、真に特異な問題（`SingularHessian`）ではない。
                    // 代わりに勾配で判定し、次の3条件がそろったとき「最適点に到達し
                    // これ以上進めない」とみなして収束を通知する（`terminate`の
                    // フォールバック）:
                    //   (1) 生のNewtonステップを1回進めても勾配ノルムが実質的に減らない
                    //       （＝これ以上詰められない）。
                    //   (2) 勾配ノルムが収束目標`tol`の近傍にある（`NEWTON_STALL_GRAD_FACTOR`
                    //       倍未満）。最適点から遠い場所での停滞は (2) で除外される。
                    //   (3) コスト関数（負の対数尤度）のHessianが正定値
                    //       （＝内点最大の2階条件）。`newton_step`の可逆性判定だけでは
                    //       鞍点（可逆だが不定符号）を弾けないため（Tobitの`(β, logσ)`
                    //       尤度は大域凹ではない、`docs/spec/tobit-spec.md`3.1節）、
                    //       ここで`llt`により明示的に確認する。
                    let raw_grad = problem.gradient(&raw_candidate)?;
                    let raw_grad_norm = l2_norm(&raw_grad);
                    let stalled = raw_grad_norm >= NEWTON_STALL_GRAD_RATIO * l2_norm(&grad)
                        && raw_grad_norm < NEWTON_STALL_GRAD_FACTOR * self.tol
                        && cost_hessian_is_positive_definite(&hessian);
                    if stalled {
                        // 最適点に到達しており、これ以上動かさない（現在点をそのまま返す。
                        // 生のNewtonステップは丸め誤差オーダーでコストを狭義には減少
                        // させないため適用しない）。
                        self.stalled_at_optimum = true;
                        (param, grad)
                    } else {
                        // 生のNewtonステップを適用して次反復へ進む。収束点近傍の通常の
                        // ケース（生ステップで勾配が桁で減る、条件(1)が偽）はここで
                        // 勾配が`tol`を下回り、`terminate`の勾配基準で収束する。
                        // 鞍点や勾配がまだ目標から遠い場合もここに来て、降下方向が
                        // 最後まで見つからなければ`max_iter`到達で`NonConvergence`になる。
                        (raw_candidate, raw_grad)
                    }
                }
            };
        let state = state.param(new_param).gradient(new_grad);
        Ok((state, None))
    }

    fn terminate(&mut self, state: &NewtonState) -> TerminationStatus {
        // 勾配ノルム基準が発火しないまま最適点で停滞したケース
        // （`FaerNewton::stalled_at_optimum`のdocコメント参照）。`next_iter`が直前に
        // 判定済みで、通常の勾配基準より先に確認する。
        if self.stalled_at_optimum {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        if let Some(g) = state.get_gradient()
            && l2_norm(g) < self.tol
        {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

/// `Δθ = (H+λI)⁻¹g`（`λ=0`なら生のNewtonステップと同じ）で候補パラメータを求め、
/// `cost(θ-Δθ) < cost(θ)`となるまで`λ`を`0→INITIAL_LM_LAMBDA→×LM_LAMBDA_GROWTH→…`と
/// 段階的に増やす（Levenberg-Marquardt**型**のHessian修正、Nocedal & Wright
/// *Numerical Optimization*の"Hessian modification"に相当）。Levenberg-Marquardt原法
/// （非線形最小二乗のGauss-Newton近似Hessianに対する信頼領域法）とは目的関数の形が
/// 異なる一般のMLE最適化への適用であり、名称はあくまで「`H+λI`による正則化」という
/// 手法的な類似性を指す。
///
/// **導入経緯**: Logit/Probitのように尤度が大域凹（Hessianが半正定値）な
/// 問題では、収束点に向かう正常な軌道上は`λ=0`の生のNewtonステップが最初の試行で
/// 受理される（収束点近傍の最後の1〜2反復は、下記の「予測減少量がコストの分解能以下の
/// 場合」の分岐で同じ生のステップが比較なしに採られる）ため、収束の挙動（反復回数・
/// 収束点）は変わらない。
///
/// **設計行列が構造的に特異な入力（完全な多重共線性等）について**:
/// Logit/Probit/Tobitの`fit()`は`run_solver`を呼ぶ前に`checked_design_matrix_qr`で
/// 設計行列を列ピボットQR分解し、ランク落ちを`SingularDesignMatrix`として弾くように
/// なった。このためLogit/Probitのこの種の入力は`regularized_newton_step`にそもそも
/// 到達しない。`newton_step`が`SingularHessian`を返す経路自体は残しており
/// （収束点近傍で丸め誤差により不定符号Hessianになる等）、その回帰ガードは合成問題
/// `SingularHessianProblem`（実データではなく`hessian()`が常にゼロ行列を返すモデル）を
/// 使う`run_solver_newton_returns_singular_hessian_error`が担う。
///
/// 一方Tobitは`(β, logσ)`パラメータ化で大域凹性が保証されず
/// （`docs/spec/tobit-spec.md`3.1節参照。Olsen(1978)の`(β/σ, 1/σ)`変換は不採用）、
/// Hessianが不定符号になる領域では
/// **生のNewtonステップが降下方向ですらなくなる**ことが実測で判明した（OLS推定値を
/// 初期値にしても、実際に打ち切りが発生するデータで一貫して再現。ステップをどれだけ
/// 小さくスケールしても`cost`が改善しないケースを確認済み）。`λI`を加えて
/// Hessianを正定値に近づけることで、十分大きな`λ`では最急降下方向（`-g/λ`、非零の
/// 勾配に対して必ず降下方向）に漸近するため、通常は有限回の試行でコスト減少方向が
/// 見つかる。**例外はNewtonステップによる真のコスト減少量が
/// コストの評価誤差を下回る収束点近傍**で、この場合`cost`の比較は丸め誤差の符号で決まり、
/// どの`λ`でも`cost`を狭義に減少させられないことがある（以前は誤って`SingularHessian`を
/// 返していた）。現在は`λ=0`のHessianが可逆かどうかで
/// [`RegularizedStep::NoProgress`]（収束扱い）と`SingularHessian`（真に特異）を分ける
/// （下記参照）。
///
/// `newton_step`が`MleError::SingularHessian`を返した場合（`λ`を加えても数値的に
/// 特異なまま）は、そのまま次の`λ`を試す（即座にエラーを伝播しない）。
///
/// **予測減少量がコストの分解能以下の場合（ラダーに入らない）**: Hessianが正定値で、
/// raw Newtonステップの予測コスト減少量`½gᵀH⁻¹g`が
/// [`NEWTON_COST_RESOLUTION_FACTOR`]`·ε·|cost|`以下なら、コスト比較もLMラダーも行わず
/// [`RegularizedStep::BelowCostResolution`]を返す。収束点近傍（2次収束の最後の1〜2反復）
/// ではNewtonステップによる真のコスト減少量がコストの評価誤差を下回り、
/// `cost(θ-Δθ) < cost(θ)`の判定は丸め誤差の符号で決まる。この比較のままだと、
/// 勾配を`tol`未満まで下げる正しいステップを棄却し、LMラダーを数十段進めて偶然
/// コストが下がった無意味な極小ステップ（または不完全に正則化されたステップ）を受理し、
/// 大標本でcost評価と反復を大幅に浪費する（Probit `n=1_000_000`で実測: 10反復・cost評価
/// 155回。この分岐と補償和の導入後は5反復・各1回）。良否は呼び出し元が勾配で判定する。
///
/// **`MAX_LM_ATTEMPTS`回すべて失敗した場合の分岐**: `λ=0`（正則化前）の
/// `newton_step`が成功していた（Hessianが数値的に可逆）かどうかで結果を分ける。
/// - 可逆だった場合 → [`RegularizedStep::NoProgress`]。これは「Hessianは可逆だが、
///   `λ`をどれだけ増やしてもコストを狭義に減少させられない」状態。典型的には収束点近傍で
///   真のコスト減少量がコストの評価誤差を下回ったケース（Hessianが正定値なら通常は上記の
///   `BelowCostResolution`の分岐で先に扱われ、ここに来るのは`|cost|≈0`で分解能の閾値が
///   ほぼ0になる場合や、Hessianが正定値でない場合など）。呼び出し元
///   （`FaerNewton::next_iter`）が勾配の停滞・目標近傍・コストHessianの正定値性を
///   追加確認し、そろえば収束、そうでなければ（鞍点等）生ステップを適用して反復を続け、
///   最終的に`NonConvergence`になる。
/// - `λ=0`で`newton_step`が特異だった場合 → `SingularHessian`（完全な多重共線性等、
///   `run_solver_newton_returns_singular_hessian_error`のケース。既存のエラー型・意味を
///   変えない）。
enum RegularizedStep {
    /// コストを狭義に減少させる有限のステップが見つかった。中身は更新後のパラメータ
    /// `θ - Δθ`。
    Accepted(Vec<f64>),
    /// `MAX_LM_ATTEMPTS`回試してもコストを減少させるステップが無かったが、`λ=0`の
    /// Hessianは可逆だった（＝真に特異ではない）。中身は`λ=0`の生の
    /// Newtonステップを適用した候補パラメータ`θ - H⁻¹g`（最適点近傍なら差は丸め誤差
    /// オーダー）。`FaerNewton::next_iter`がこの候補で停滞条件を判定し、収束と判断した
    /// 場合は候補を適用せず現在点`θ`にとどまる。
    NoProgress(Vec<f64>),
    /// Hessianが正定値で、raw Newtonステップの予測コスト減少量`½gᵀH⁻¹g`がコストの
    /// 評価値の分解能（[`NEWTON_COST_RESOLUTION_FACTOR`]`·ε·|cost|`）以下のため、
    /// コスト比較とLMラダーを省いた。中身は`NoProgress`と同じくraw Newtonステップを適用した
    /// 候補`θ - H⁻¹g`で、`FaerNewton::next_iter`も`NoProgress`と同じ勾配ベースの判定で扱う。
    BelowCostResolution(Vec<f64>),
}

fn regularized_newton_step<O>(
    problem: &mut Problem<O>,
    param: &[f64],
    grad: &[f64],
    hessian: &[Vec<f64>],
    cost: f64,
) -> Result<RegularizedStep, OptimizerError>
where
    O: CostFunction<Param = Vec<f64>, Output = f64>,
{
    let k = grad.len();
    // ステップ`Δθ`を現在のパラメータに適用して候補`θ - Δθ`を作る。
    let apply_step =
        |step: &[f64]| -> Vec<f64> { param.iter().zip(step).map(|(p, s)| p - s).collect() };

    // `λ=0`（正則化前）のNewtonステップが可逆だったかを、`MAX_LM_ATTEMPTS`回すべて
    // 失敗したときの分岐（`NoProgress` vs `SingularHessian`）のために覚えておく。
    // 可逆なら適用後のパラメータ`θ - H⁻¹g`も控える（`NoProgress`で返す候補。生の局所
    // 2次モデルの最良推定で、最適点との差は丸め誤差オーダー）。ループ初回（`λ=0`）は
    // この値を使い回し、`newton_step`の二重計算を避ける。
    let raw_newton_step = newton_step(hessian, grad).ok();

    // 予測減少量がコストの分解能以下なら、コスト比較は丸め誤差の符号を見ているだけになる
    // （正しいステップを棄却し、LMラダーが無意味な極小ステップを偶然受理して反復とcost評価を
    // 浪費する）。Hessianが正定値（局所2次モデルが凸）なら比較を省いてraw Newtonステップを
    // 返し、良否の判定は呼び出し元が勾配で行う。
    if let Some(step) = &raw_newton_step
        && predicted_decrease_is_below_cost_resolution(grad, step, cost)
        && cost_hessian_is_positive_definite(hessian)
    {
        return Ok(RegularizedStep::BelowCostResolution(apply_step(step)));
    }

    let raw_newton_candidate = raw_newton_step.map(|step| apply_step(&step));

    let mut lambda = 0.0_f64;
    for attempt in 0..MAX_LM_ATTEMPTS {
        let candidate = if attempt == 0 {
            raw_newton_candidate.clone()
        } else {
            let regularized: Vec<Vec<f64>> = (0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| hessian[i][j] + if i == j { lambda } else { 0.0 })
                        .collect()
                })
                .collect();
            newton_step(&regularized, grad)
                .ok()
                .map(|step| apply_step(&step))
        };
        if let Some(candidate) = candidate
            && let Ok(candidate_cost) = problem.cost(&candidate)
            && candidate_cost.is_finite()
            && candidate_cost < cost
        {
            return Ok(RegularizedStep::Accepted(candidate));
        }
        lambda = if lambda == 0.0 {
            INITIAL_LM_LAMBDA
        } else {
            lambda * LM_LAMBDA_GROWTH
        };
    }
    // コストを減少させるステップが1つも見つからなかった。`λ=0`のHessianが可逆だったなら
    // 真に特異な問題ではなく、コスト比較がステップの良否を判定できない状態
    // （`BelowCostResolution`の導入前は`n=200_000, seed=1` / `n=1_000_000, seed=42`の
    // `moderate_censoring`Tobitでここに到達していた）。
    // `λ=0`でも特異だった場合は従来どおり`SingularHessian`
    // （完全な多重共線性等、`SingularHessianProblem`）。
    match raw_newton_candidate {
        Some(candidate) => Ok(RegularizedStep::NoProgress(candidate)),
        None => Err(MleError::SingularHessian.into()),
    }
}

/// raw Newtonステップ`Δθ = H⁻¹g`の予測コスト減少量`½gᵀΔθ`（局所2次モデル
/// `cost(θ-Δθ) ≈ cost(θ) - gᵀΔθ + ½ΔθᵀHΔθ = cost(θ) - ½gᵀH⁻¹g`）が、コストの評価値の
/// 分解能[`NEWTON_COST_RESOLUTION_FACTOR`]`·ε·|cost|`以下か。`cost`が非有限なら偽
/// （分解能を定義できないため、従来のLMラダーに任せる）。
///
/// 予測減少量が負なら偽: Hessianが正定値なら理論上`gᵀH⁻¹g > 0`だが、悪条件なHessianでは
/// 求解の丸め誤差で負になりうる（＝上り方向のステップ）。その場合は検証なしに採らず、
/// 従来どおりLMラダーで検証する。
fn predicted_decrease_is_below_cost_resolution(grad: &[f64], step: &[f64], cost: f64) -> bool {
    if !cost.is_finite() {
        return false;
    }
    let predicted_decrease = 0.5 * grad.iter().zip(step).map(|(g, s)| g * s).sum::<f64>();
    (0.0..=NEWTON_COST_RESOLUTION_FACTOR * f64::EPSILON * cost.abs()).contains(&predicted_decrease)
}

/// Newtonステップ`Δθ = H⁻¹g`を求める。`H`は対称とは限らない（収束点から離れた場所では
/// 正定値でないこともある、Probit等）ため、列ピボットQR（`linear_algebra::checked_col_piv_qr`の
/// 相対閾値での特異性検出）を使う。
fn newton_step(hessian: &[Vec<f64>], grad: &[f64]) -> Result<Vec<f64>, MleError> {
    let k = grad.len();
    let h = Mat::from_fn(k, k, |i, j| hessian[i][j]);
    let g = Mat::from_fn(k, 1, |i, _| grad[i]);

    let qr = checked_col_piv_qr(&h).map_err(|_| MleError::SingularHessian)?;

    let step = qr.solve_lstsq(&g);
    Ok((0..k).map(|i| *step.get(i, 0)).collect())
}

#[cfg(test)]
mod tests {
    use super::super::options::SolverType;
    use super::super::solver::run_solver;
    use super::super::test_support::*;
    use super::*;

    use argmin::core::Error as OptimizerError;

    use std::cell::Cell;

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

        let step =
            regularized_newton_step(&mut problem, &[0.0, 0.0], &[1.0e-3, 0.0], &hessian, 1.0e5)
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
}
