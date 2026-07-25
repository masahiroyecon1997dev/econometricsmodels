//! nonlinear系統（Logit/Probit/Tobit）で共有するエラー型。
//!
//! OLSは1手法1エラー型（`OlsError`）だったが、nonlinear系統は`raise_on_non_convergence`
//! 未収束・観測数不足・`confidence_level`範囲外等、3手法でほぼ共通のバリアントが多いため、
//! `<系統>/common.rs`に共有型として定義する（`.claude/rules/rust-style.md`「ファイル・
//! ディレクトリ構成」参照）。Tobit専用のバリアント（`InvalidCensoringBounds`）も、別のenumに
//! 分離せず同じ`MleError`に含める（Logit/Probitの`fit()`はそのバリアントを構築しないだけで、
//! 型を分ける必要はない。OLSの`CovType::Hac`/`CovType::Cluster`がフィールド付きバリアントとして
//! 共存しているのと同じ考え方）。
//!
//! バリアント一覧・Python例外との対応表は`docs/planning/specs/nonlinear-implementation-notes.md`
//! 「エラー型: nonlinear系統で共有（MleError）」を参照。

use argmin::core::TerminationStatus;
use argmin::core::{
    CostFunction, Error as OptimizerError, Executor, Gradient, Hessian, IterState, KV, Problem,
    Solver, State, TerminationReason,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::{BFGS, LBFGS};
use faer::Mat;
use faer::prelude::SolveLstsq;
use thiserror::Error;

/// Logit/Probit/Tobitの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
#[derive(Debug, Error, PartialEq)]
pub enum MleError {
    /// `raise_on_non_convergence=true`（既定）かつ`max_iter`回で収束しなかった。
    #[error(
        "failed to converge after {n_iter} iterations. Set raise_on_non_convergence=False \
         to receive the result anyway, or increase max_iter"
    )]
    NonConvergence { n_iter: usize },

    /// 観測数nが説明変数の数k（定数項を含む）以下。
    #[error(
        "insufficient observations: n={n} must be greater than k={k} \
         (number of independent variables, including the intercept)"
    )]
    InsufficientObservations { n: usize, k: usize },

    /// `confidence_level`が`(0, 1)`の範囲外。
    #[error("confidence_level must be in the range (0, 1): {confidence_level}")]
    InvalidConfidenceLevel { confidence_level: f64 },

    /// `max_iter`が0以下。
    #[error("max_iter must be a positive integer, got {max_iter}")]
    InvalidMaxIter { max_iter: i64 },

    /// `cov_type="cluster"`なのにクラスターのグループキーが渡されていない。
    #[error("cov_type='cluster' requires cluster identifiers to be provided")]
    MissingClusterColumn,

    /// `cov_type="cluster"`のときのクラスター数が2未満。
    #[error("cov_type='cluster' requires at least 2 clusters, got {g}")]
    InsufficientClusters { g: usize },

    /// 収束点のHessianが特異で、観測情報行列（`cov_type="classical"`/`"nonrobust"`既定）の
    /// 逆行列が計算できない。
    #[error(
        "the Hessian at convergence is singular; cannot compute the observed information matrix"
    )]
    SingularHessian,

    /// 上記以外の計算過程での失敗（分布のCDF計算等）。
    #[error("computation failed: {0}")]
    ComputationFailed(String),

    /// Tobit専用: 打ち切り境界（下限/上限）の指定が不正（下限≧上限等）。
    #[error(
        "invalid censoring bounds: lower={lower:?}, upper={upper:?} \
         (at least one bound must be set, and lower must be less than upper when both are set)"
    )]
    InvalidCensoringBounds {
        lower: Option<f64>,
        upper: Option<f64>,
    },
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

/// `run_solver`の出力。
#[derive(Debug, Clone)]
pub struct SolverOutput {
    /// 収束点（`raise_on_non_convergence=false`で未収束の場合は打ち切り時点）のパラメータ。
    pub params: Vec<f64>,
    /// 収束点で解析的に評価したHessian（k×k）。`method`の選択に関わらず常に評価する
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
                .map_err(|e| MleError::ComputationFailed(e.to_string()))?;
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
                .map_err(|e| MleError::ComputationFailed(e.to_string()))?;
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
    let hessian = model
        .hessian(&params)
        .map_err(|e| MleError::ComputationFailed(e.to_string()))?;

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
        MleError::ComputationFailed("solver did not produce a parameter estimate".to_string())
    })?;
    let model = problem.take_problem().ok_or_else(|| {
        MleError::ComputationFailed("failed to recover the optimization problem".to_string())
    })?;
    Ok((params, converged, n_iter, model))
}

/// argminの内部エラー（`anyhow::Error`）を`MleError`に変換する。
/// `FaerNewton::next_iter`内で`MleError`から`?`により変換された値は`downcast`で復元し、
/// それ以外（argmin自体の内部エラー等）は`ComputationFailed`にまとめる。
fn convert_optimizer_error(e: OptimizerError) -> MleError {
    match e.downcast::<MleError>() {
        Ok(mle_error) => mle_error,
        Err(other) => MleError::ComputationFailed(other.to_string()),
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
    fn name(&self) -> &str {
        "Newton (faer-backed)"
    }

    fn next_iter(
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
        let hessian = problem.hessian(&param)?;
        let step = newton_step(&hessian, &grad)?;
        let new_param: Vec<f64> = param.iter().zip(step.iter()).map(|(p, s)| p - s).collect();
        let state = state.param(new_param).gradient(grad);
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
        if (*r.get(i, i)).abs() <= threshold {
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
        assert!((output.hessian[0][0] - 2.0).abs() < 1e-9);
        assert!((output.hessian[1][1] - 5.0).abs() < 1e-9);
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
    fn mle_error_messages_are_human_readable() {
        assert_eq!(
            MleError::NonConvergence { n_iter: 35 }.to_string(),
            "failed to converge after 35 iterations. Set raise_on_non_convergence=False \
             to receive the result anyway, or increase max_iter"
        );
        assert_eq!(
            MleError::InsufficientObservations { n: 2, k: 3 }.to_string(),
            "insufficient observations: n=2 must be greater than k=3 \
             (number of independent variables, including the intercept)"
        );
        assert_eq!(
            MleError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
            .to_string(),
            "confidence_level must be in the range (0, 1): 1.5"
        );
        assert_eq!(
            MleError::InvalidMaxIter { max_iter: 0 }.to_string(),
            "max_iter must be a positive integer, got 0"
        );
        assert_eq!(
            MleError::MissingClusterColumn.to_string(),
            "cov_type='cluster' requires cluster identifiers to be provided"
        );
        assert_eq!(
            MleError::InsufficientClusters { g: 1 }.to_string(),
            "cov_type='cluster' requires at least 2 clusters, got 1"
        );
        assert_eq!(
            MleError::SingularHessian.to_string(),
            "the Hessian at convergence is singular; cannot compute the observed information matrix"
        );
        assert_eq!(
            MleError::ComputationFailed("normal CDF did not converge".to_string()).to_string(),
            "computation failed: normal CDF did not converge"
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
    }

    #[test]
    fn mle_error_implements_partial_eq() {
        assert_eq!(MleError::SingularHessian, MleError::SingularHessian);
        assert_ne!(
            MleError::InsufficientClusters { g: 1 },
            MleError::InsufficientClusters { g: 0 }
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
}
