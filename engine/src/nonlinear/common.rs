//! nonlinear系統（Logit/Probit/Tobit）で共有するエラー型。
//!
//! OLSは1手法1エラー型（`LeastSquaresError`、旧`OlsError`）だったが、nonlinear系統は`raise_on_non_convergence`
//! 未収束・観測数不足・`confidence_level`範囲外等、3手法でほぼ共通のバリアントが多いため、
//! `<系統>/common.rs`に共有型として定義する（`.claude/rules/rust-style.md`「ファイル・
//! ディレクトリ構成」参照）。Tobit専用のバリアント（`InvalidCensoringBounds`）も、別のenumに
//! 分離せず同じ`MleError`に含める（Logit/Probitの`fit()`はそのバリアントを構築しないだけで、
//! 型を分ける必要はない。OLSの`CovType::Hac`/`CovType::Cluster`がフィールド付きバリアントとして
//! 共存しているのと同じ考え方）。
//!
//! バリアント一覧・Python例外との対応表は`docs/planning/specs/nonlinear-implementation-notes.md`
//! 「エラー型: nonlinear系統で共有（MleError）」を参照。
//!
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`は、linear系統の
//! `LeastSquaresError`と文言まで完全に重複していたため`engine::error::CommonError`に
//! 切り出し、`Common`バリアント経由で保持する（Issue #113）。

use argmin::core::TerminationStatus;
use argmin::core::{
    CostFunction, Error as OptimizerError, Executor, Gradient, Hessian, IterState, KV, Problem,
    Solver, State, TerminationReason,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::{BFGS, LBFGS};
use faer::prelude::{Solve, SolveLstsq};
use faer::{Mat, Side};
use thiserror::Error;

use crate::error::CommonError;
use crate::linear_algebra::ensure_well_conditioned_symmetric_matrix;

/// Logit/Probit/Tobitの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
#[derive(Debug, Error, PartialEq)]
pub enum MleError {
    /// 系統をまたいで共通のバリデーション・計算エラー（`CommonError`参照）。
    #[error(transparent)]
    Common(#[from] CommonError),

    /// `raise_on_non_convergence=true`（既定）かつ`max_iter`回で収束しなかった。
    #[error(
        "failed to converge after {n_iter} iterations. Set raise_on_non_convergence=False \
         to receive the result anyway, or increase max_iter"
    )]
    NonConvergence { n_iter: usize },

    /// `max_iter`が0以下。
    #[error("max_iter must be a positive integer, got {max_iter}")]
    InvalidMaxIter { max_iter: i64 },

    /// `tol`が0以下。勾配ノルムに基づく収束判定`‖∇ℓ(θ)‖ < tol`が理論上満たされないため、
    /// 常に`max_iter`まで反復して`NonConvergence`（または`converged=false`）になる
    /// （Issue #118、`InvalidMaxIter`と同じ形の早期バリデーション）。
    #[error("tol must be a positive number, got {tol}")]
    InvalidTol { tol: f64 },

    /// Hessianが特異で逆行列が計算できない。Newton法のステップ求解中（収束前の任意の点）、
    /// および収束点での観測情報行列・サンドイッチ型・クラスターロバストSE計算
    /// （いずれもHessianの逆行列を使う）の両方で発生しうる。
    #[error("the Hessian is singular and cannot be inverted")]
    SingularHessian,

    /// OPG（`Σᵢ sᵢsᵢ'`、outer product of gradients）行列が特異で逆行列が計算できない。
    /// `cov_type="opg"`（BHHH）でのみ発生しうる。Hessianではなくスコアの外積和が
    /// 特異という別の原因のため、`SingularHessian`とは区別する。
    #[error("the outer-product-of-gradients (OPG) matrix is singular and cannot be inverted")]
    SingularOpgMatrix,

    /// Tobit専用: 打ち切り境界（下限/上限）の指定が不正（下限≧上限等）。
    #[error(
        "invalid censoring bounds: lower={lower:?}, upper={upper:?} \
         (at least one bound must be set, and lower must be less than upper when both are set)"
    )]
    InvalidCensoringBounds {
        lower: Option<f64>,
        upper: Option<f64>,
    },

    /// Logit/Probit専用: `y`（被説明変数）が`{0.0, 1.0}`の2値でない値を含む。
    ///
    /// statsmodelsの`Logit`はコンストラクタ時点で単位区間`[0,1]`を要求する
    /// （比率データ／frequency weights的な用途も許容する設計）が、本実装は
    /// 常に真の2値アウトカムのみを想定するため、より厳格に`{0.0, 1.0}`の完全一致を
    /// 要求する（ユーザー確認済み、Issue #135）。`InvalidCensoringBounds`（Tobit専用）
    /// と対称的に、Tobit（連続な打ち切り被説明変数）の`fit()`はこのバリアントを
    /// 構築しない（`MleError`モジュールdocコメント「別のenumに分離せず同じ`MleError`に
    /// 含める」の方針通り）。
    #[error("y at row {row} must be coded as 0.0 or 1.0 (binary outcome), got {value}")]
    InvalidBinaryY { row: usize, value: f64 },
}

/// `y`が`{0.0, 1.0}`の2値でない値を含む場合にエラーを返す（Logit/Probit専用、
/// `MleError::InvalidBinaryY`のdocコメント参照）。statsmodelsは`Logit`のコンストラクタ
/// 時点でこの検証を行うが、本実装では`fit()`冒頭（`LogitInput::from_columns`の
/// 次元検証とは別、`nonlinear-implementation-notes.md`参照）で行う。O(n)の単純走査
/// （既にengine_pybind側で行っているNaN/無限大チェックと同オーダー）で、
/// 反復最適化本体（O(n·k²)を`max_iter`回）に対して計算コストは無視できる
/// （Issue #135、実測: n=1,000,000で`fit()`全体の約0.16%）。
pub fn validate_binary_y(y: &Mat<f64>) -> Result<(), MleError> {
    for i in 0..y.nrows() {
        let value = *y.get(i, 0);
        if value != 0.0 && value != 1.0 {
            return Err(MleError::InvalidBinaryY { row: i, value });
        }
    }
    Ok(())
}

/// 数値最適化ソルバーの種類。文字列パース（Python文字列 → この型への変換）は
/// `engine_pybind`側の責務（OLSの`CovType`と同じ設計。`.claude/rules/rust-style.md`参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Newton-Raphson法（既定）。解析的Hessianを使う。
    Newton,
    /// BFGS法（準ニュートン法、Hessianを近似的に更新する）。
    Bfgs,
    /// L-BFGS法（限定記憶BFGS）。
    Lbfgs,
}

/// 標準誤差（係数分散共分散行列）の種別。文字列パース（Python文字列 → この型への変換、
/// `"classical"`/`"nonrobust"`のエイリアス化を含む）は`engine_pybind`側の責務
/// （OLSの`CovType`と同じ設計。`.claude/rules/rust-style.md`参照）。
///
/// Logit/Probit/Tobitで共通のバリアント（`nonlinear-api-design.md`4章）のため
/// `nonlinear/common.rs`に定義する（`Method`と同じ理由）。
///
/// `Cluster`のみ、他のバリアントと異なり追加データ（グループキー）を持つため
/// フィールド付きバリアントにしている（OLSの`CovType::Cluster`と同じ設計パターン。
/// `groups`が`None`の場合、モデルの`fit()`は`CommonError::MissingClusterColumn`を返す）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CovType {
    /// 観測情報行列（`"classical"`/`"nonrobust"`、既定）: `Σ = -H⁻¹`
    Classical,
    /// OPG/BHHH（`"opg"`）: `Σ = (Σᵢ sᵢsᵢ')⁻¹`
    Opg,
    /// サンドイッチ型（misspecification-robust、`"hc0"`）: `Σ = H⁻¹(Σᵢ sᵢsᵢ')H⁻¹`
    Hc0,
    /// サンドイッチ型+小標本補正（`"hc1"`）: `hc0`の`Σ`に`n/(n-k)`を乗じる
    Hc1,
    /// クラスターロバスト（`"cluster"`）: `Σ = correction * H⁻¹(Σ_g S_gS_g')H⁻¹`
    /// （`docs/planning/specs/nonlinear-implementation-notes.md`「標準誤差の技術仕様」参照）。
    Cluster {
        /// クラスターのグループキー。モデルの入力データの行と対応する長さnの配列。
        /// `None`の場合、モデルの`fit()`は`CommonError::MissingClusterColumn`を返す。
        groups: Option<Vec<String>>,
    },
}

/// 限界効果（`marginal_effects`）をどの代表点で評価するか。文字列パース（Python文字列 →
/// この型への変換）は`engine_pybind`側の責務（`Method`/`CovType`と同じ設計。
/// `.claude/rules/rust-style.md`参照）。
///
/// Logit/Probit/Tobitで共通の概念（`nonlinear-api-design.md`6章）のため`nonlinear/
/// common.rs`に定義する（`Method`/`CovType`と同じ理由）。実際の限界効果・デルタ法
/// ヤコビアンの計算式はリンク関数（`Λ`/`Φ`等）に依存するため、モデルごとの実装
/// （`logit.rs`等）に置く。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginalEffectsAt {
    /// 平均限界効果（AME、average marginal effects）。全観測での限界効果の平均。既定。
    Overall,
    /// 各説明変数の標本平均からなる代表点での限界効果（MEM、marginal effects at the mean）。
    Mean,
    /// 各説明変数の標本中央値からなる代表点での限界効果。
    Median,
}

/// `run_solver`の出力。
#[derive(Debug, Clone)]
pub struct SolverOutput {
    /// 収束点（`raise_on_non_convergence=false`で未収束の場合は打ち切り時点）のパラメータ。
    pub params: Vec<f64>,
    /// 収束点で解析的に評価した**対数尤度そのもの**のHessian（k×k、真の最大点では負定値）。
    /// `method`の選択に関わらず常に評価する。`cov_type`共通行列演算（`neg_hessian_inverse`
    /// 等）はこの符号（対数尤度のHessian）を前提とする。モデルの`Hessian`トレイト実装
    /// 自体は`CostFunction`/`Gradient`と同じ符号（コスト関数＝負の対数尤度のHessian）を
    /// 返す契約になっており、ここに格納する値は`run_solver`内部で1回符号反転したもの
    /// （`docs/planning/specs/nonlinear-implementation-notes.md`「engine内のtrait設計」参照）。
    pub hessian: Vec<Vec<f64>>,
    /// 収束したかどうか。
    pub converged: bool,
    /// 実際の反復回数。
    pub n_iter: usize,
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
/// # Errors
/// - 収束点のHessianが特異（`SingularHessian`）
/// - `raise_on_non_convergence=true`かつ`max_iter`回で収束しなかった（`NonConvergence`）
/// - その他ソルバー内部でのエラー（`ComputationFailed`）
pub fn run_solver<O>(
    problem: O,
    method: Method,
    initial_params: Vec<f64>,
    max_iter: u64,
    tol: f64,
    raise_on_non_convergence: bool,
) -> Result<SolverOutput, MleError>
where
    O: CostFunction<Param = Vec<f64>, Output = f64>
        + Gradient<Param = Vec<f64>, Gradient = Vec<f64>>
        + Hessian<Param = Vec<f64>, Hessian = Vec<Vec<f64>>>,
{
    let k = initial_params.len();

    let (params, converged, n_iter, model) = match method {
        Method::Newton => {
            let solver = FaerNewton { tol };
            let result = Executor::new(problem, solver)
                .configure(|state| state.param(initial_params).max_iters(max_iter))
                .run()
                .map_err(convert_optimizer_error)?;
            extract_outcome(result.state, result.problem)?
        }
        Method::Bfgs => {
            let linesearch = MoreThuenteLineSearch::new();
            let solver = BFGS::new(linesearch)
                .with_tolerance_grad(tol)
                .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
            let result = Executor::new(problem, solver)
                .configure(|state| {
                    state
                        .param(initial_params)
                        .inv_hessian(identity_matrix(k))
                        .max_iters(max_iter)
                })
                .run()
                .map_err(convert_optimizer_error)?;
            extract_outcome(result.state, result.problem)?
        }
        Method::Lbfgs => {
            let linesearch = MoreThuenteLineSearch::new();
            let solver = LBFGS::new(linesearch, 7)
                .with_tolerance_grad(tol)
                .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
            let result = Executor::new(problem, solver)
                .configure(|state| state.param(initial_params).max_iters(max_iter))
                .run()
                .map_err(convert_optimizer_error)?;
            extract_outcome(result.state, result.problem)?
        }
    };

    if !converged && raise_on_non_convergence {
        return Err(MleError::NonConvergence { n_iter });
    }

    // Hessianはmethodの選択に関わらず、収束点で常に解析的に評価する
    // （bfgs/lbfgsの内部近似Hessianは使い回さない）。
    //
    // `Hessian`トレイトの契約は「`CostFunction`/`Gradientと同じ符号（コスト関数=負の
    // 対数尤度のHessian）」（`FaerNewton`のNewtonステップが正しい方向に進むために必要、
    // `CostFunction`のdocコメント参照）。一方`SolverOutput.hessian`は`cov_type`共通行列
    // 演算（`neg_hessian_inverse`等）が前提とする「対数尤度そのもののHessian」（真の
    // 最大点で負定値）でなければならない。両者は符号が逆（`-loglik`のHessian＝
    // `loglik`のHessianの符号反転）なので、ここで1回だけ符号反転して契約を合わせる
    // （Issue #55着手時に発覚、`docs/planning/specs/nonlinear-implementation-notes.md`
    // 参照）。
    let cost_hessian = model
        .hessian(&params)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
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
/// `Method`の3分岐で共通の後処理のため、`run_solver`から切り出している。
/// `I`はソルバーごとに異なる状態型（LBFGSはHessianスロットを使わないため`H=()`）だが、
/// いずれも`State`トレイト経由で同じ形で取り出せる。
///
/// **`ok_or_else`の2箇所は理論上到達不能**（`.claude/rules/rust-style.md`「テスト」の
/// カバレッジ方針、Issue #64でLogitのカバレッジ確認時に判明・受け入れ済み）:
/// - `state.get_best_param()`が`None`になるのは`Executor::run()`が`init()`/`next_iter()`を
///   一度も呼ばずに終了した場合のみだが、`init()`が必ず初期パラメータを`state`に設定する
///   （`FaerNewton::init`、BFGS/LBFGSも同様に組み込みソルバーが初期化時に設定する）ため
///   起こり得ない。
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
    let params = state.get_best_param().cloned().ok_or_else(|| {
        CommonError::ComputationFailed("solver did not produce a parameter estimate".to_string())
    })?;
    let model = problem.take_problem().ok_or_else(|| {
        CommonError::ComputationFailed("failed to recover the optimization problem".to_string())
    })?;
    Ok((params, converged, n_iter, model))
}

/// argminの内部エラー（`anyhow::Error`）を`MleError`に変換する。
/// `FaerNewton::next_iter`内で`MleError`から`?`により変換された値は`downcast`で復元し、
/// それ以外（argmin自体の内部エラー等）は`ComputationFailed`にまとめる。
///
/// **`Err(other)`分岐は実測ではカバーされていない**（Issue #64で判明）:
/// 本プロジェクトが制御する全てのエラー経路（`FaerNewton`・モデルの`CostFunction`/
/// `Gradient`/`Hessian`実装）は`MleError`（`?`経由で`anyhow::Error`に変換されたもの）
/// のみを返すため、`downcast`は常に成功する。この分岐はargmin自体の内部（`Executor`の
/// 状態管理等）が予期しない`anyhow::Error`を生成した場合に備えた防御的なフォールバックで、
/// 意図的にargminの内部を破壊するようなテストを書くのは実装の振る舞いというより
/// argmin内部実装への依存になるため見送っている（OLSの`ols-implementation-notes.md`
/// 「理論上到達不能な防御的エラーパス」と同じ性質）。
fn convert_optimizer_error(e: OptimizerError) -> MleError {
    match e.downcast::<MleError>() {
        Ok(mle_error) => mle_error,
        Err(other) => CommonError::ComputationFailed(other.to_string()).into(),
    }
}

fn identity_matrix(k: usize) -> Vec<Vec<f64>> {
    (0..k)
        .map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn gradient_norm(g: &[f64]) -> f64 {
    g.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// argmin組み込みの`Newton`ソルバーは`H: ArgminInv<H>`（Hessianの逆行列）を要求するが、
/// `argmin-math`の`vec`機能（`Vec<Vec<f64>>`向け）には`ArgminInv`の実装が存在しない
/// （faer/nalgebra/ndarrayの行列型にしか実装されていない）ため使えない。Newton法は独自の
/// `Solver`実装とし、ステップの求解はfaer（列ピボットQR、OLSの`ensure_full_rank`と同じ
/// 特異性検出パターン）で行う（`docs/planning/specs/nonlinear-implementation-notes.md`参照）。
struct FaerNewton {
    tol: f64,
}

type NewtonState = IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>;

impl<O> Solver<O, NewtonState> for FaerNewton
where
    O: Gradient<Param = Vec<f64>, Gradient = Vec<f64>>
        + Hessian<Param = Vec<f64>, Hessian = Vec<Vec<f64>>>,
{
    /// `argmin::core::Solver`トレイトの必須メソッド（ロギング・エラーメッセージ表示等、
    /// argmin内部が使う識別名）。本プロジェクトはargminの`Observer`（進捗ロギング機構）を
    /// 使っておらず、`run_solver`のテストでも実行結果（`SolverOutput`）のみを検証するため
    /// 呼ばれない。実装自体は`argmin::solver::newton::Newton`等の組み込みソルバーに倣った
    /// 定型実装で、分岐を持たない単純な文字列リテラルの返却のため、未カバーでも
    /// 振る舞いの正しさに影響しない（Issue #64で判明・受け入れ済み）。
    fn name(&self) -> &str {
        "Newton (faer-backed)"
    }

    /// 初期パラメータでの勾配をあらかじめ`state`に格納する。これにより`terminate()`は
    /// 常に「`state.get_param()`と対応する勾配」を見られる（`next_iter`実行前の最初の
    /// `terminate()`呼び出しも含む。初期値が既に収束条件を満たす場合を正しく扱うため）。
    ///
    /// **`ok_or_else`分岐は理論上到達不能**（Issue #64で判明・受け入れ済み）:
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
        let step = newton_step(&hessian, &grad)?;
        let new_param: Vec<f64> = param.iter().zip(step.iter()).map(|(p, s)| p - s).collect();
        // 収束判定（terminate）が「更新後のparamに対応する勾配」を見られるよう、
        // 更新前のgradを使い回さずnew_paramで改めて評価する。
        let new_grad = problem.gradient(&new_param)?;
        let state = state.param(new_param).gradient(new_grad);
        Ok((state, None))
    }

    fn terminate(&mut self, state: &NewtonState) -> TerminationStatus {
        if let Some(g) = state.get_gradient()
            && gradient_norm(g) < self.tol
        {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

/// Newtonステップ`Δθ = H⁻¹g`を求める。`H`は対称とは限らない（収束点から離れた場所では
/// 正定値でないこともある、Probit等）ため、列ピボットQR（OLSの`ensure_full_rank`と同じ
/// 相対閾値での特異性検出）を使う。
fn newton_step(hessian: &[Vec<f64>], grad: &[f64]) -> Result<Vec<f64>, MleError> {
    let k = grad.len();
    let h = Mat::from_fn(k, k, |i, j| hessian[i][j]);
    let g = Mat::from_fn(k, 1, |i, _| grad[i]);

    let qr = h.col_piv_qr();
    let r = qr.thin_R();
    let max_abs_diag = (0..k).map(|i| (*r.get(i, i)).abs()).fold(0.0_f64, f64::max);
    let threshold = (k as f64) * f64::EPSILON * max_abs_diag;
    for i in 0..k {
        let diag = (*r.get(i, i)).abs();
        // NaNを明示的にチェックする（`diag <= threshold`だとNaNとの比較は常にfalseになり
        // すり抜けてしまう）。全ゼロ行列のcol_piv_qrは列選択時の0除算によりRの対角がNaNに
        // なりうるため（faer 0.24.4で実機確認済み）、この形にしている。
        if diag.is_nan() || diag <= threshold {
            return Err(MleError::SingularHessian);
        }
    }

    let step = qr.solve_lstsq(&g);
    Ok((0..k).map(|i| *step.get(i, 0)).collect())
}

/// 設計行列の列ごとの標準化スケール（標準偏差のみ。平均は引かない）。
///
/// 分散1へのスケーリングのみ行い、平均センタリングは行わない。理由: `x_std = (x-mean)/std`と
/// 平均も引く標準化は、切片が「平均分のズレ」を吸収する前提の変換で、`include_intercept=false`
/// のとき逆変換の式が成立しない（吸収先の切片が存在しないため）。`x_std = x/std`のみなら
/// `θ_orig_j = θ_std_j/std_j`で完結し、切片の有無に関係なく成立する。当初の目的（勾配ノルムの
/// 絶対閾値がxのスケールに依存する問題への対処）もスケーリングのみで達成できる
/// （`docs/planning/specs/nonlinear-implementation-notes.md`参照）。
#[derive(Debug, Clone)]
pub struct ColumnScale {
    /// 列ごとの標準偏差。切片列（`has_intercept=true`の先頭列）は`1.0`のまま
    /// （スケーリング対象外）。標準偏差が0の列（定数列）も`1.0`のまま扱う（0除算回避）。
    stds: Vec<f64>,
}

impl ColumnScale {
    /// 列ごとの標準偏差（`standardize_columns`のdocコメント参照）。標準化はengine内部の
    /// 実装詳細であり`engine_pybind`に公開する必然性は無い（rust-reviewer指摘）が、
    /// `pub(crate)`にすると`cargo clippy --all-targets -- -D warnings`の`lib`ターゲット
    /// （テストコードを含まないビルド）で`dead_code`エラーになる（唯一の呼び出し元が
    /// `nonlinear/logit.rs`の`#[cfg(test)] mod tests`のみのため）。`pub`アイテムは
    /// dead_code検出の対象外という言語仕様上の扱いにより、`pub`のままにしている
    /// （テストが`fit()`と同じ標準化・逆標準化の手順を独立に再現するために必要）。
    pub fn stds(&self) -> &[f64] {
        &self.stds
    }
}

/// 設計行列`x`の各列を標準偏差でスケーリングする（切片列は除外）。
/// 収束後は`destandardize_params`で元のスケールのパラメータへ逆変換する。
pub fn standardize_columns(x: &Mat<f64>, has_intercept: bool) -> (Mat<f64>, ColumnScale) {
    let n = x.nrows();
    let k = x.ncols();
    let start = usize::from(has_intercept);

    let mut stds = vec![1.0; k];
    for (j, std) in stds.iter_mut().enumerate().take(k).skip(start) {
        let mean: f64 = (0..n).map(|i| *x.get(i, j)).sum::<f64>() / n as f64;
        let var: f64 = (0..n).map(|i| (*x.get(i, j) - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt();
        if sd > 0.0 {
            *std = sd;
        }
    }

    let x_std = Mat::from_fn(n, k, |i, j| *x.get(i, j) / stds[j]);
    (x_std, ColumnScale { stds })
}

/// `standardize_columns`で標準化した空間で最適化した`params_std`を、元のスケールの
/// パラメータへ逆変換する（`θ_orig_j = θ_std_j / std_j`）。
pub fn destandardize_params(params_std: &[f64], scale: &ColumnScale) -> Vec<f64> {
    params_std
        .iter()
        .zip(scale.stds.iter())
        .map(|(p, s)| p / s)
        .collect()
}

/// `standardize_columns`で標準化した空間で評価した係数分散共分散行列`cov_std`
/// （`observed_information_cov_params`等の`cov_type`共通行列演算の戻り値）を、
/// 元のスケールの係数分散共分散行列へ逆変換する。
///
/// `θ_orig = D⁻¹θ_std`（`D = diag(stds)`、`destandardize_params`と同じ変換）のとき、
/// 対数尤度を`θ_std`の関数とみなすと連鎖律により`H_std = D⁻¹ H_orig D⁻¹`
/// （`H`はHessian、`D`は対角行列なので`D⁻¹` の転置は`D⁻¹`自身）が成り立つ。
/// 分散共分散行列はHessianの逆行列に比例する（`Σ = -H⁻¹`等）ため、
/// `Σ_orig = -H_orig⁻¹ = -(D H_std D)⁻¹ = D⁻¹(-H_std⁻¹)D⁻¹ = D⁻¹ Σ_std D⁻¹`
/// （`H_std = D⁻¹H_origD⁻¹`の逆を取ると`H_orig = D H_std D`になることを使った）。
/// この関係はOPG（`Σᵢsᵢsᵢ'`の逆行列）・サンドイッチ・クラスターの各`cov_type`でも
/// 同様に成り立つ（いずれも`H⁻¹`を両側から掛ける、または`H⁻¹`の逆行列を取る形の式のため）。
/// `D`が対角行列であることから、要素ごとに`Σ_orig[i,j] = Σ_std[i,j] / (stds[i]*stds[j])`
/// という単純な計算に帰着する。
pub fn destandardize_cov_params(cov_std: &Mat<f64>, scale: &ColumnScale) -> Mat<f64> {
    let k = scale.stds.len();
    Mat::from_fn(k, k, |i, j| {
        *cov_std.get(i, j) / (scale.stds[i] * scale.stds[j])
    })
}

// `cov_type`ごとの係数分散共分散行列の共通計算。モデル固有の尤度計算には依存せず、
// 収束点で評価した`H`（対数尤度のHessian、k×k）と`scores`（観測ごとのスコア行列、n×k、
// 各行が観測`i`のスコアベクトル`sᵢ`）だけを受け取る（`docs/planning/specs/
// nonlinear-implementation-notes.md`「標準誤差の技術仕様」参照）。
//
// `"classical"`/`"nonrobust"`は同じ計算（観測情報行列）のエイリアスのため、
// engine側では区別せず`observed_information_cov_params`ひとつに統一する
// （文字列パースの分岐はOLSの`"classical"`/`"nonrobust"`と同じくengine_pybind側の責務）。

/// `-H`（Hessianの符号反転）のコレスキー分解による逆行列。
///
/// 真のMLE最大点では`-H`が正定値になるはず、という前提でコレスキー分解を使う
/// （OLSの`xtx_inverse`と同じ発想）。分解に失敗した場合は`MleError::SingularHessian`を返す。
/// この失敗は理論上到達不能な防御的分岐ではなく、悪条件な収束点（ほぼ特異なHessian等）
/// で実際に起こりうる（`MleError::SingularHessian`のdocコメント参照）。
///
/// 数学的に`-H⁻¹ = (-H)⁻¹`が成り立つため、観測情報行列によるΣ（`= -H⁻¹`）はこの関数の
/// 戻り値そのもの。さらに`H⁻¹ Ψ H⁻¹ = (-H)⁻¹ Ψ (-H)⁻¹`（符号が2回打ち消しあう）ため、
/// サンドイッチ型・クラスターロバストの計算でも同じ戻り値をそのまま両側から掛ければよく、
/// 追加の符号反転は不要（`sandwich_cov_params`・`cluster_cov_params`参照）。
///
/// **Cholesky分解の前に固有値ベースの悪条件検出を行う**（`crate::linear_algebra::
/// ensure_well_conditioned_symmetric_matrix`、OLSの`wald_f_test`と共有する
/// ユーティリティ）。非ピボットCholesky（`Llt`）のL因子対角成分は、構造的な
/// 特異性（完全な多重共線性等）を確実には検出できない（OLSで実測確認済み、
/// Issue #107）。Newton法は`newton_step`内の別の検出経路（ピボット付きQR）が
/// 最適化中に必ず通るためこの問題が表面化しなかったが、BFGS/L-BFGSは
/// `newton_step`を経由しないため、収束後のこの関数が唯一の検出経路になる
/// （Issue #129で発覚: `Method::Bfgs`で完全な多重共線性のあるデータセットを
/// 最適化すると、修正前はエラーにならず桁違いに巨大な値を返していた）。
fn neg_hessian_inverse(hessian: &Mat<f64>, k: usize) -> Result<Mat<f64>, MleError> {
    let neg_h = Mat::from_fn(k, k, |i, j| -(*hessian.get(i, j)));
    // `context`引数（"negated Hessian"）は`map_err`で`MleError::SingularHessian`に
    // 潰すため実際には使われない。呼び出し元ごとに専用のエラーバリアントを持つ
    // 系統（nonlinear）ではこれでよいが、`ensure_well_conditioned_symmetric_matrix`
    // 自体は`CommonError::ComputationFailed`のメッセージ用に使う汎用引数であることに注意。
    ensure_well_conditioned_symmetric_matrix(&neg_h, k, "negated Hessian")
        .map_err(|_| MleError::SingularHessian)?;
    let llt = neg_h
        .llt(Side::Lower)
        .map_err(|_| MleError::SingularHessian)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// 観測情報行列による係数分散共分散行列（`cov_type="classical"`/`"nonrobust"`、既定）:
/// `Σ = -H⁻¹`。
pub fn observed_information_cov_params(hessian: &Mat<f64>, k: usize) -> Result<Mat<f64>, MleError> {
    neg_hessian_inverse(hessian, k)
}

/// 観測ごとのスコア行列からOPG（outer product of gradients）行列
/// `Ψ = Σᵢ sᵢsᵢ' = scores' * scores`を計算する。OPG（BHHH）SE自体にも、
/// サンドイッチ型SE（hc0/hc1）にも使う共通の中間値。
fn opg_matrix(scores: &Mat<f64>) -> Mat<f64> {
    scores.transpose() * scores
}

/// OPG（BHHH）による係数分散共分散行列（`cov_type="opg"`）: `Σ = (Σᵢ sᵢsᵢ')⁻¹`。
///
/// `Ψ = Σᵢ sᵢsᵢ'`は外積の和のため半正定値であり、コレスキー分解で逆行列を求める
/// （`neg_hessian_inverse`と同じ理由でコレスキーを使うが、対象行列がHessianではなく
/// スコアの外積和のため、特異時のエラーは`MleError::SingularOpgMatrix`で区別する）。
/// `neg_hessian_inverse`と同じ理由で、Cholesky分解の前に`ensure_well_conditioned_
/// symmetric_matrix`による固有値ベースの悪条件検出を行う（Issue #129）。
pub fn opg_cov_params(scores: &Mat<f64>, k: usize) -> Result<Mat<f64>, MleError> {
    let psi = opg_matrix(scores);
    ensure_well_conditioned_symmetric_matrix(&psi, k, "OPG matrix")
        .map_err(|_| MleError::SingularOpgMatrix)?;
    let llt = psi
        .llt(Side::Lower)
        .map_err(|_| MleError::SingularOpgMatrix)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// `sandwich_cov_params`の内部でのみ使う、サンドイッチ型SEの種類。`opg_matrix`はOPGの
/// バリアントも含むより広い概念であるため、この列挙はHC0/HC1限定であることを型で明確にする
/// （OLSの`HcVariant`と同じ考え方）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandwichVariant {
    /// misspecification-robust（quasi-MLEサンドイッチ）: `Σ = H⁻¹ΨH⁻¹`
    Hc0,
    /// `hc0`のΣに小標本補正`n/(n-k)`を乗じる
    Hc1,
}

/// サンドイッチ型（misspecification-robust）による係数分散共分散行列
/// （`cov_type="hc0"`/`"hc1"`）: `Σ = H⁻¹ (Σᵢ sᵢsᵢ') H⁻¹`（`hc1`は`n/(n-k)`を追加で乗じる）。
pub fn sandwich_cov_params(
    hessian: &Mat<f64>,
    scores: &Mat<f64>,
    n: usize,
    k: usize,
    variant: SandwichVariant,
) -> Result<Mat<f64>, MleError> {
    let neg_h_inv = neg_hessian_inverse(hessian, k)?;
    let psi = opg_matrix(scores);
    let sandwich = &neg_h_inv * &psi * &neg_h_inv;
    Ok(match variant {
        SandwichVariant::Hc0 => sandwich,
        SandwichVariant::Hc1 => {
            let correction = (n as f64) / ((n - k) as f64);
            Mat::from_fn(k, k, |i, j| correction * (*sandwich.get(i, j)))
        }
    })
}

/// クラスターロバストな係数分散共分散行列（`cov_type="cluster"`）:
/// `Σ = correction * H⁻¹ (Σ_g S_gS_g') H⁻¹`、`S_g = Σ_{i∈g} sᵢ`、
/// `correction = G/(G-1) * (n-1)/(n-k)`（OLSと同じ小標本補正、常に適用する。無効化
/// オプションはない）。
///
/// `groups`が2種類以上の値を持つこと（クラスター数`G>=2`）の検証（`CommonError::
/// InsufficientClusters`）、および未指定時の`CommonError::MissingClusterColumn`は
/// モデルごとの`fit()`実装側の責務（OLSの`validate_cluster_groups`と同じ役割分担、
/// `docs/planning/specs/logit-probit-issue-breakdown.md`のB7/C7参照）。
/// `groups.len() != n`もモデル側の内部契約（`debug_assert_eq!`で検証）。
pub fn cluster_cov_params(
    hessian: &Mat<f64>,
    scores: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &[String],
) -> Result<Mat<f64>, MleError> {
    debug_assert_eq!(
        groups.len(),
        n,
        "groups length must match nobs (caller contract)"
    );

    let neg_h_inv = neg_hessian_inverse(hessian, k)?;

    // `HashMap`は反復順序がプロセスごとのハッシュシードに依存し非決定的なため、`BTreeMap`
    // （クラスター名の辞書順）を使う（OLSの`cluster_cov_params`と同じ理由）。
    let mut group_indices: std::collections::BTreeMap<&str, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        group_indices.entry(g.as_str()).or_default().push(i);
    }
    let n_groups = group_indices.len();

    let mut s_hat = Mat::<f64>::zeros(k, k);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; k];
        for &i in indices {
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += *scores.get(i, a);
            }
        }
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }

    let correction =
        (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
    let sandwich = &neg_h_inv * &s_hat * &neg_h_inv;
    Ok(Mat::from_fn(k, k, |i, j| {
        correction * (*sandwich.get(i, j))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用のダミー目的関数: `f(θ) = 0.5 (θ - target)' A (θ - target)`
    /// （`A`は対角行列、対角成分は正でAは正定値）。最小点は`θ = target`、
    /// そこでのHessianは`A`自身（既知の解析解で検証できる）。
    #[derive(Clone)]
    struct QuadraticProblem {
        target: Vec<f64>,
        diag_a: Vec<f64>,
    }

    impl CostFunction for QuadraticProblem {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
            let cost = param
                .iter()
                .zip(self.target.iter())
                .zip(self.diag_a.iter())
                .map(|((p, t), a)| 0.5 * a * (p - t).powi(2))
                .sum();
            Ok(cost)
        }
    }

    impl Gradient for QuadraticProblem {
        type Param = Vec<f64>;
        type Gradient = Vec<f64>;

        fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
            Ok(param
                .iter()
                .zip(self.target.iter())
                .zip(self.diag_a.iter())
                .map(|((p, t), a)| a * (p - t))
                .collect())
        }
    }

    impl Hessian for QuadraticProblem {
        type Param = Vec<f64>;
        type Hessian = Vec<Vec<f64>>;

        fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
            let k = self.diag_a.len();
            Ok((0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| if i == j { self.diag_a[i] } else { 0.0 })
                        .collect()
                })
                .collect())
        }
    }

    fn quadratic_problem() -> QuadraticProblem {
        QuadraticProblem {
            target: vec![3.0, -2.0],
            diag_a: vec![2.0, 5.0],
        }
    }

    #[test]
    fn run_solver_newton_converges_to_known_minimum() {
        let output = run_solver(
            quadratic_problem(),
            Method::Newton,
            vec![0.0, 0.0],
            35,
            1e-6,
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
    fn run_solver_bfgs_converges_to_known_minimum() {
        let output = run_solver(
            quadratic_problem(),
            Method::Bfgs,
            vec![0.0, 0.0],
            100,
            1e-6,
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
    fn run_solver_lbfgs_converges_to_known_minimum() {
        let output = run_solver(
            quadratic_problem(),
            Method::Lbfgs,
            vec![0.0, 0.0],
            100,
            1e-6,
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
    fn run_solver_returns_non_convergence_error_when_max_iter_is_too_small() {
        let result = run_solver(
            quadratic_problem(),
            Method::Bfgs,
            vec![1000.0, -1000.0],
            1,
            1e-12,
            true,
        );

        assert!(matches!(result, Err(MleError::NonConvergence { .. })));
    }

    #[test]
    fn run_solver_returns_result_without_raising_when_raise_on_non_convergence_is_false() {
        let output = run_solver(
            quadratic_problem(),
            Method::Bfgs,
            vec![1000.0, -1000.0],
            1,
            1e-12,
            false,
        )
        .unwrap();

        assert!(!output.converged);
    }

    #[test]
    fn run_solver_newton_returns_non_convergence_error_when_max_iter_is_too_small() {
        let result = run_solver(
            quadratic_problem(),
            Method::Newton,
            vec![1000.0, -1000.0],
            0,
            1e-12,
            true,
        );

        assert!(matches!(result, Err(MleError::NonConvergence { .. })));
    }

    #[test]
    fn run_solver_newton_returns_result_without_raising_when_raise_on_non_convergence_is_false() {
        let output = run_solver(
            quadratic_problem(),
            Method::Newton,
            vec![1000.0, -1000.0],
            0,
            1e-12,
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
            Method::Newton,
            vec![0.0],
            35,
            1e-6,
            true,
        );

        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    /// `FaerNewton`の収束判定が「実際に返却するパラメータでの勾配」を正しく見ているかを
    /// 検証する回帰テスト（rust-reviewerが指摘: 修正前は「更新前のパラメータ」の勾配で
    /// 収束判定していたバグがあった）。
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
            Method::Newton,
            vec![0.5],
            10,
            1e-6,
            false,
        )
        .unwrap();

        assert!(output.converged, "{:?}", output);
        let actual_grad = IllConditionedProblem.gradient(&output.params).unwrap();
        assert!(
            gradient_norm(&actual_grad) < 1e-6,
            "converged=true was reported but the actual gradient at the returned params {:?} is {:?}",
            output.params,
            actual_grad
        );
    }

    #[test]
    fn standardize_columns_scales_to_unit_variance_and_leaves_intercept_untouched() {
        let x = Mat::from_fn(4, 3, |i, j| {
            if j == 0 {
                1.0
            } else if j == 1 {
                [10.0, 20.0, 30.0, 40.0][i]
            } else {
                [1.0, 2.0, 3.0, 4.0][i]
            }
        });

        let (x_std, scale) = standardize_columns(&x, true);

        for i in 0..4 {
            assert_eq!(*x_std.get(i, 0), 1.0);
        }
        // 平均は引いていないため、分散（平均まわりの2次モーメント）で検証する
        // （2乗の平均=E[x²]ではなくVar(x)=E[x²]-E[x]²が1になるはず）。
        let variance = |col: usize| -> f64 {
            let values: Vec<f64> = (0..4).map(|i| *x_std.get(i, col)).collect();
            let mean = values.iter().sum::<f64>() / 4.0;
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0
        };
        assert!((variance(1) - 1.0).abs() < 1e-9);
        assert!((variance(2) - 1.0).abs() < 1e-9);
        assert_eq!(scale.stds[0], 1.0);
    }

    #[test]
    fn standardize_and_destandardize_round_trips_without_intercept() {
        let x = Mat::from_fn(4, 2, |i, j| {
            if j == 0 {
                [10.0, 20.0, 30.0, 40.0][i]
            } else {
                [100.0, 200.0, 150.0, 50.0][i]
            }
        });

        let (_, scale) = standardize_columns(&x, false);
        let params_std = vec![6.0, -3.0];
        let params_orig = destandardize_params(&params_std, &scale);

        // 線形予測子 x_std' * params_std == x_orig' * params_orig が成立するはず
        for i in 0..4 {
            let pred_std: f64 = (0..2)
                .map(|j| (*x.get(i, j) / scale.stds[j]) * params_std[j])
                .sum();
            let pred_orig: f64 = (0..2).map(|j| *x.get(i, j) * params_orig[j]).sum();
            assert!((pred_std - pred_orig).abs() < 1e-9);
        }
    }

    #[test]
    fn standardize_and_destandardize_round_trips_with_intercept() {
        let x = Mat::from_fn(4, 3, |i, j| {
            if j == 0 {
                1.0
            } else if j == 1 {
                [10.0, 20.0, 30.0, 40.0][i]
            } else {
                [100.0, 200.0, 150.0, 50.0][i]
            }
        });

        let (_, scale) = standardize_columns(&x, true);
        let params_std = vec![1.5, 6.0, -3.0];
        let params_orig = destandardize_params(&params_std, &scale);

        // 切片の係数はスケーリング対象外（scale.stds[0] == 1.0）のため、
        // 切片なしのケースと同様に線形予測子が一致するはず。切片項自体は
        // 平均センタリングをしていないため素通しでよい（destandardize_paramsの
        // 補正対象にならない）。
        assert_eq!(params_orig[0], params_std[0]);
        for i in 0..4 {
            let pred_std: f64 = params_std[0]
                + (1..3)
                    .map(|j| (*x.get(i, j) / scale.stds[j]) * params_std[j])
                    .sum::<f64>();
            let pred_orig: f64 =
                params_orig[0] + (1..3).map(|j| *x.get(i, j) * params_orig[j]).sum::<f64>();
            assert!((pred_std - pred_orig).abs() < 1e-9);
        }
    }

    #[test]
    fn destandardize_cov_params_divides_by_outer_product_of_stds() {
        let x = Mat::from_fn(4, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                [10.0, 20.0, 30.0, 40.0][i]
            }
        });
        let (_, scale) = standardize_columns(&x, true);
        assert_eq!(scale.stds[0], 1.0);
        assert!(scale.stds[1] > 1.0);

        let cov_std = Mat::from_fn(2, 2, |i, j| [[4.0, 6.0], [6.0, 9.0]][i][j]);
        let cov_orig = destandardize_cov_params(&cov_std, &scale);

        // 切片列は`stds[0]==1.0`なので影響を受けない。x1列は`stds[1]`で
        // 割った分だけ小さくなる（`Σ_orig[i,j] = Σ_std[i,j]/(stds[i]*stds[j])`）。
        let s1 = scale.stds[1];
        assert!((*cov_orig.get(0, 0) - 4.0).abs() < 1e-12);
        assert!((*cov_orig.get(0, 1) - 6.0 / s1).abs() < 1e-12);
        assert!((*cov_orig.get(1, 0) - 6.0 / s1).abs() < 1e-12);
        assert!((*cov_orig.get(1, 1) - 9.0 / (s1 * s1)).abs() < 1e-12);
    }

    #[test]
    fn mle_error_messages_are_human_readable() {
        // 6種の共通バリアント（DimensionMismatch等）のメッセージ検証は
        // `engine::error`側のテストに集約済み（Issue #113）。ここではnonlinear固有の
        // バリアントに加え、`Common`が`CommonError`のDisplayをtransparentに転送する
        // ことだけを確認する。
        assert_eq!(
            MleError::NonConvergence { n_iter: 35 }.to_string(),
            "failed to converge after 35 iterations. Set raise_on_non_convergence=False \
             to receive the result anyway, or increase max_iter"
        );
        assert_eq!(
            MleError::InvalidMaxIter { max_iter: 0 }.to_string(),
            "max_iter must be a positive integer, got 0"
        );
        assert_eq!(
            MleError::SingularHessian.to_string(),
            "the Hessian is singular and cannot be inverted"
        );
        assert_eq!(
            MleError::InvalidCensoringBounds {
                lower: Some(10.0),
                upper: Some(5.0),
            }
            .to_string(),
            "invalid censoring bounds: lower=Some(10.0), upper=Some(5.0) \
             (at least one bound must be set, and lower must be less than upper when both are set)"
        );
        assert_eq!(
            MleError::SingularOpgMatrix.to_string(),
            "the outer-product-of-gradients (OPG) matrix is singular and cannot be inverted"
        );
        assert_eq!(
            MleError::Common(CommonError::MissingClusterColumn).to_string(),
            "cov_type='cluster' requires cluster identifiers to be provided"
        );
    }

    #[test]
    fn mle_error_implements_partial_eq() {
        assert_eq!(MleError::SingularHessian, MleError::SingularHessian);
        assert_ne!(MleError::SingularHessian, MleError::SingularOpgMatrix);
        assert_ne!(
            MleError::Common(CommonError::InsufficientClusters { g: 1 }),
            MleError::Common(CommonError::InsufficientClusters { g: 0 })
        );
        assert_eq!(
            MleError::NonConvergence { n_iter: 35 },
            MleError::NonConvergence { n_iter: 35 }
        );
        assert_ne!(
            MleError::NonConvergence { n_iter: 35 },
            MleError::NonConvergence { n_iter: 10 }
        );
    }

    /// `cov_type`共通行列演算のテスト用データ: 対角Hessian`H = diag(-2, -5)`
    /// （`-H`が正定値、MLE最大点を模す）と、4観測分のスコア行列。スコアは列間で
    /// 「観測ごとに片方が常にゼロ」になるよう選んでおり（`(±2, 0)`・`(0, ±5)`）、
    /// `Σsᵢsᵢ'`が対角行列になるため手計算で期待値を検証できる。
    fn cov_test_hessian() -> Mat<f64> {
        Mat::from_fn(2, 2, |i, j| if i == j { [-2.0, -5.0][i] } else { 0.0 })
    }

    fn cov_test_scores() -> Mat<f64> {
        // 行: (2,0), (-2,0), (0,5), (0,-5)
        Mat::from_fn(4, 2, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (1, 0) => -2.0,
            (2, 1) => 5.0,
            (3, 1) => -5.0,
            _ => 0.0,
        })
    }

    fn assert_diag_close(m: &Mat<f64>, expected: [f64; 2], tol: f64) {
        assert!((*m.get(0, 0) - expected[0]).abs() < tol, "{:?}", m);
        assert!((*m.get(1, 1) - expected[1]).abs() < tol, "{:?}", m);
        assert!((*m.get(0, 1)).abs() < tol, "{:?}", m);
        assert!((*m.get(1, 0)).abs() < tol, "{:?}", m);
    }

    #[test]
    fn observed_information_cov_params_computes_negative_hessian_inverse() {
        // Σ = -H⁻¹ = -diag(-2,-5)⁻¹ = diag(0.5, 0.2)
        let cov = observed_information_cov_params(&cov_test_hessian(), 2).unwrap();
        assert_diag_close(&cov, [0.5, 0.2], 1e-9);
    }

    #[test]
    fn observed_information_cov_params_returns_singular_hessian_error_for_zero_hessian() {
        let zero_hessian = Mat::<f64>::zeros(2, 2);
        let result = observed_information_cov_params(&zero_hessian, 2);
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    #[test]
    fn opg_cov_params_computes_inverse_outer_product_of_gradients() {
        // Ψ = Σsᵢsᵢ' = diag(2²+(-2)², 5²+(-5)²) = diag(8, 50)
        // Σ = Ψ⁻¹ = diag(0.125, 0.02)
        let cov = opg_cov_params(&cov_test_scores(), 2).unwrap();
        assert_diag_close(&cov, [0.125, 0.02], 1e-9);
    }

    #[test]
    fn opg_cov_params_returns_singular_opg_matrix_error_for_zero_scores() {
        let zero_scores = Mat::<f64>::zeros(4, 2);
        let result = opg_cov_params(&zero_scores, 2);
        assert!(
            matches!(result, Err(MleError::SingularOpgMatrix)),
            "{:?}",
            result
        );
    }

    /// `opg_cov_params`が非ピボットCholeskyでは検出できない「ほぼ特異」（構造的な
    /// ゼロ行列ではなく、極端なスケール差による悪条件）も検出できることを確認する
    /// （Issue #129のチェックリスト「opg_cov_paramsが同じ問題を抱えているか実測で
    /// 確認する」に対応。`ensure_well_conditioned_symmetric_matrix`単体テストと同じ
    /// スケール差（1e6/1e-3）を`scores`側で再現し、`Ψ=scores'*scores`が
    /// `diag(1e12, 1e-6)`相当になるよう構成した）。
    #[test]
    fn opg_cov_params_returns_singular_opg_matrix_error_for_extreme_scale_difference() {
        let scores = Mat::from_fn(2, 2, |i, j| if i == j { [1e6, 1e-3][i] } else { 0.0 });
        let result = opg_cov_params(&scores, 2);
        assert!(
            matches!(result, Err(MleError::SingularOpgMatrix)),
            "{:?}",
            result
        );
    }

    #[test]
    fn sandwich_cov_params_hc0_computes_sandwich_formula() {
        // H⁻¹ΨH⁻¹ = diag(0.5,0.2) * diag(8,50) * diag(0.5,0.2) = diag(2.0, 2.0)
        let cov = sandwich_cov_params(
            &cov_test_hessian(),
            &cov_test_scores(),
            4,
            2,
            SandwichVariant::Hc0,
        )
        .unwrap();
        assert_diag_close(&cov, [2.0, 2.0], 1e-9);
    }

    #[test]
    fn sandwich_cov_params_hc1_applies_small_sample_correction() {
        // hc0(diag(2.0,2.0)) * n/(n-k) = * 4/2 = diag(4.0, 4.0)
        let cov = sandwich_cov_params(
            &cov_test_hessian(),
            &cov_test_scores(),
            4,
            2,
            SandwichVariant::Hc1,
        )
        .unwrap();
        assert_diag_close(&cov, [4.0, 4.0], 1e-9);
    }

    #[test]
    fn cluster_cov_params_computes_clustered_sandwich_with_small_sample_correction() {
        // グループ a = {観測0, 観測2}, b = {観測1, 観測3}（同符号の組にならないよう
        // 意図的にクロスさせている。同グループ内の合計が単純に打ち消し合わないようにするため）。
        // S_a = s0+s2 = (2,5), S_b = s1+s3 = (-2,-5)
        // S_hat = S_aS_a' + S_bS_b' = [[8,20],[20,50]]
        // correction = G/(G-1) * (n-1)/(n-k) = 2/1 * 3/2 = 3
        // H⁻¹S_hatH⁻¹ = [[2,2],[2,2]] (observed_information_cov_paramsの対角0.5,0.2で挟む)
        // Σ = 3 * [[2,2],[2,2]] = [[6,6],[6,6]]
        let groups = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "b".to_string(),
        ];
        let cov =
            cluster_cov_params(&cov_test_hessian(), &cov_test_scores(), 4, 2, &groups).unwrap();
        assert!((*cov.get(0, 0) - 6.0).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 1) - 6.0).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(0, 1) - 6.0).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 0) - 6.0).abs() < 1e-9, "{:?}", cov);
    }

    #[test]
    fn cluster_cov_params_returns_singular_hessian_error_for_zero_hessian() {
        let zero_hessian = Mat::<f64>::zeros(2, 2);
        let groups = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "b".to_string(),
        ];
        let result = cluster_cov_params(&zero_hessian, &cov_test_scores(), 4, 2, &groups);
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    /// `cov_test_scores`は列間で観測ごとに片方が常にゼロになるよう選んであり、`Ψ`が
    /// 対角行列になるため非対角成分の検証ができない。このフィクスチャは列間に相関を
    /// 持たせ（`Ψ`が非対角成分を持つ）、`opg_cov_params`/`sandwich_cov_params`が
    /// 転置・スケーリングの順序を誤っていないかを対角のみのテストより厳密に検証する。
    fn cov_test_scores_correlated() -> Mat<f64> {
        // 行: (1,2), (3,4), (-1,1), (2,-3)
        Mat::from_fn(4, 2, |i, j| {
            [[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0], [2.0, -3.0]][i][j]
        })
    }

    #[test]
    fn opg_cov_params_computes_inverse_for_correlated_scores() {
        // Ψ = scores'*scores = [[15,7],[7,30]]（手計算: 1²+3²+(-1)²+2²=15,
        // 2²+4²+1²+(-3)²=30, 1*2+3*4+(-1)*1+2*(-3)=7）。
        // 2×2逆行列の公式（本体実装が使うコレスキーとは独立な検算経路）で期待値を求める。
        let det = 15.0 * 30.0 - 7.0 * 7.0;
        let expected = [[30.0 / det, -7.0 / det], [-7.0 / det, 15.0 / det]];

        let cov = opg_cov_params(&cov_test_scores_correlated(), 2).unwrap();
        assert!((*cov.get(0, 0) - expected[0][0]).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 1) - expected[1][1]).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(0, 1) - expected[0][1]).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 0) - expected[1][0]).abs() < 1e-9, "{:?}", cov);
    }

    #[test]
    fn sandwich_cov_params_hc0_computes_off_diagonal_correctly() {
        // H⁻¹ΨH⁻¹ = diag(0.5,0.2) * [[15,7],[7,30]] * diag(0.5,0.2)
        //          = [[0.5*15*0.5, 0.5*7*0.2], [0.2*7*0.5, 0.2*30*0.2]]
        //          = [[3.75, 0.7], [0.7, 1.2]]
        let cov = sandwich_cov_params(
            &cov_test_hessian(),
            &cov_test_scores_correlated(),
            4,
            2,
            SandwichVariant::Hc0,
        )
        .unwrap();
        assert!((*cov.get(0, 0) - 3.75).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 1) - 1.2).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(0, 1) - 0.7).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 0) - 0.7).abs() < 1e-9, "{:?}", cov);
    }

    #[test]
    fn sandwich_cov_params_returns_singular_hessian_error_for_zero_hessian() {
        let zero_hessian = Mat::<f64>::zeros(2, 2);
        let result = sandwich_cov_params(
            &zero_hessian,
            &cov_test_scores(),
            4,
            2,
            SandwichVariant::Hc0,
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    #[test]
    fn validate_binary_y_ok_for_all_zero_and_one() {
        let y = Mat::from_fn(4, 1, |i, _| [0.0, 1.0, 0.0, 1.0][i]);
        assert_eq!(validate_binary_y(&y), Ok(()));
    }

    #[test]
    fn validate_binary_y_returns_invalid_binary_y_error_for_non_binary_value() {
        let y = Mat::from_fn(3, 1, |i, _| [0.0, 0.5, 1.0][i]);
        assert_eq!(
            validate_binary_y(&y),
            Err(MleError::InvalidBinaryY { row: 1, value: 0.5 })
        );
    }

    #[test]
    fn validate_binary_y_reports_first_violation_row() {
        let y = Mat::from_fn(3, 1, |i, _| [2.0, 0.0, -1.0][i]);
        assert_eq!(
            validate_binary_y(&y),
            Err(MleError::InvalidBinaryY { row: 0, value: 2.0 })
        );
    }
}
