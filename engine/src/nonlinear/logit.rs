//! Logitの入力データ（被説明変数・設計行列）の型定義、および負の対数尤度・スコア・
//! Hessian（argminの`CostFunction`/`Gradient`/`Hessian`トレイト実装）。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `LogitInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`engine::linear::ols::OlsInput`と同型の設計
//! （`docs/planning/specs/nonlinear-api-design.md`参照）。
//!
//! OLSと異なり、Phase2（Logit/Probit/Tobit）では`weights`/`offset`を見送っているため
//! （`nonlinear-api-design.md`7章）、`from_columns_weighted`に相当するものはない。
//!
//! ## 数式（ロジスティック回帰）
//!
//! `p_i = Λ(x_i'θ) = 1/(1+exp(-x_i'θ))`（ロジスティック関数）。観測`i`の対数尤度への
//! 寄与は`ℓ_i(θ) = y_i log(p_i) + (1-y_i) log(1-p_i)`で、`z_i = x_i'θ`とおくと
//! `ℓ_i(θ) = y_i z_i - softplus(z_i)`（`softplus(z) = log(1+exp(z))`）という同値な形に
//! 書き換えられる（`log(p_i) = z_i - softplus(z_i)`、`log(1-p_i) = -softplus(z_i)`）。
//! `softplus`形の方が指数関数のオーバーフローを避けやすいため、`cost`の実装はこちらを使う。
//!
//! - スコア（対数尤度の1階微分）: `∂ℓ/∂θ = Σᵢ (yᵢ-pᵢ)xᵢ = X'(y-p)`
//! - Hessian（対数尤度の2階微分）: `∂²ℓ/∂θ∂θ' = -Σᵢ pᵢ(1-pᵢ)xᵢxᵢ' = -X'WX`
//!   （`W = diag(pᵢ(1-pᵢ))`。ロジットの対数尤度は大域的に凹なので`-X'WX`は
//!   （厳密な多重共線性がなければ）常に負定値、`X'WX`は常に正定値）
//!
//! `CostFunction`は`-ℓ(θ)`（argminは最小化フレームワークのため）。`run_solver`の
//! docコメント「`Hessian`トレイトの符号規約」の通り、`Gradient`/`Hessian`トレイトは
//! `CostFunction`と同じ符号（`-ℓ`の1階・2階微分）で実装する。`scores()`（`cov_type`
//! 共通行列演算向け）は符号反転しない生のスコア`sᵢ=(yᵢ-pᵢ)xᵢ`を返す
//! （`nonlinear/common.rs`の`SolverOutput.hessian`と同じく対数尤度そのものの符号）。

use crate::error::CommonError;
use crate::nonlinear::common::{
    Method, MleError, destandardize_cov_params, destandardize_params,
    observed_information_cov_params, run_solver, standardize_columns,
};
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;
use statrs::distribution::{ContinuousCDF, Normal};

/// Logitの被説明変数・設計行列を保持する入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
#[derive(Debug)]
pub struct LogitInput {
    /// 被説明変数 (n, 1)
    y: Mat<f64>,
    /// 設計行列 (n, k)。`include_intercept=true`の場合、先頭列が定数項（すべて1.0）
    x: Mat<f64>,
    /// 係数名（`include_intercept=true`なら先頭が"const"）。`x`の列と対応する
    param_names: Vec<String>,
    /// 被説明変数名
    dep_var_name: String,
    /// 定数項を含むか。`nonlinear/common.rs`の`standardize_columns`等で必要
    has_intercept: bool,
}

impl LogitInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `LogitInput`を組み立てる。`include_intercept=true`の場合、設計行列の先頭列に
    /// 定数項（すべて1.0）を自動追加する。
    ///
    /// `y`が単位区間`[0,1]`に収まることの検証は、このIssue（#54、次元検証のみがスコープ）では
    /// 行わない。statsmodelsの`Logit`はコンストラクタ時点でこの検証を行っている（`endog`が
    /// 範囲外だと`ValueError: endog must be in the unit interval.`）ため、本実装でも尤度・
    /// スコア・Hessianを実装するIssue（B2）で同等の検証を追加する予定
    /// （`docs/planning/specs/nonlinear-implementation-notes.md`参照）。
    ///
    /// # Errors
    /// `y`といずれかの`x_columns`の長さが一致しない場合は`CommonError::DimensionMismatch`を返す。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする。これは
    /// 呼び出し側（`engine_pybind`）の実装バグでしか起こり得ない内部契約であり、
    /// 実データに起因する`ValidationError`とは性質が異なるため区別している
    /// （`OlsInput::from_columns`と同じ方針）。
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
    ) -> Result<Self, MleError> {
        debug_assert_eq!(
            x_columns.len(),
            x_names.len(),
            "x_columns and x_names must have the same length"
        );
        for col in x_columns {
            if col.len() != y.len() {
                return Err(CommonError::DimensionMismatch {
                    y_rows: y.len(),
                    x_rows: col.len(),
                }
                .into());
            }
        }

        let n = y.len();
        let k = if include_intercept {
            x_columns.len() + 1
        } else {
            x_columns.len()
        };

        let x = Mat::from_fn(n, k, |i, j| {
            if include_intercept {
                if j == 0 { 1.0 } else { x_columns[j - 1][i] }
            } else {
                x_columns[j][i]
            }
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let mut param_names = Vec::with_capacity(k);
        if include_intercept {
            param_names.push("const".to_string());
        }
        param_names.extend(x_names);

        Ok(Self {
            y: y_mat,
            x,
            param_names,
            dep_var_name,
            has_intercept: include_intercept,
        })
    }

    pub fn y(&self) -> &Mat<f64> {
        &self.y
    }

    pub fn x(&self) -> &Mat<f64> {
        &self.x
    }

    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 定数項を含むか
    pub fn has_intercept(&self) -> bool {
        self.has_intercept
    }

    /// 観測数 n
    pub fn nobs(&self) -> usize {
        self.y.nrows()
    }

    /// 説明変数の数 k（定数項を含む）
    pub fn k(&self) -> usize {
        self.x.ncols()
    }
}

/// `z = log(1+exp(z))`（softplus）の数値的に安定な計算。`z`が大きい正の値でも
/// `exp(z)`のオーバーフローを起こさない（標準的な安定化式、`z.max(0)+log1p(exp(-|z|))`）。
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// ロジスティック関数`Λ(z) = 1/(1+exp(-z))`の数値的に安定な計算。
/// `z`の符号で分岐し、`exp`の引数が常に非正になるようにする（`exp`のオーバーフロー回避）。
fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Logitの負の対数尤度・スコア・Hessian（argminの`CostFunction`/`Gradient`/`Hessian`
/// トレイト実装）。`LogitInput`の`X`・`y`を保持する（`run_solver`が`problem`の所有権を
/// 必要とするため、`LogitInput`とは独立した所有データとして持つ。`Clone`は
/// `argmin::core::Executor`が要求する）。
#[derive(Debug, Clone)]
pub struct LogitProblem {
    x: Mat<f64>,
    y: Mat<f64>,
}

impl LogitProblem {
    pub fn new(input: &LogitInput) -> Self {
        Self {
            x: input.x().clone(),
            y: input.y().clone(),
        }
    }

    /// 観測`i`の線形予測子 `z_i = x_i'θ`。
    fn linear_predictor(&self, i: usize, params: &[f64]) -> f64 {
        (0..self.x.ncols())
            .map(|j| *self.x.get(i, j) * params[j])
            .sum()
    }

    /// 観測ごとのスコア行列（n×k）。各行が`sᵢ = (yᵢ-pᵢ)xᵢ`（対数尤度の1階微分そのもの、
    /// `Gradient`トレイトとは符号が逆）。OPG/サンドイッチ/クラスターSEの計算に使う
    /// （argminの`Gradient`は合計済みの1本のベクトルしか返さないため別途必要、
    /// `docs/planning/specs/nonlinear-implementation-notes.md`「engine内のtrait設計」参照）。
    pub fn scores(&self, params: &[f64]) -> Mat<f64> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        Mat::from_fn(n, k, |i, j| {
            let p = logistic(self.linear_predictor(i, params));
            (*self.y.get(i, 0) - p) * (*self.x.get(i, j))
        })
    }
}

impl CostFunction for LogitProblem {
    type Param = Vec<f64>;
    type Output = f64;

    /// 負の対数尤度 `-ℓ(θ) = Σᵢ [softplus(zᵢ) - yᵢzᵢ]`（モジュール冒頭の数式参照）。
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        let n = self.x.nrows();
        let cost: f64 = (0..n)
            .map(|i| {
                let z = self.linear_predictor(i, param);
                softplus(z) - (*self.y.get(i, 0)) * z
            })
            .sum();
        Ok(cost)
    }
}

impl Gradient for LogitProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// `-ℓ(θ)`の勾配 `Σᵢ (pᵢ-yᵢ)xᵢ = X'(p-y)`（対数尤度のスコアの符号反転）。
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let mut grad = vec![0.0; k];
        for i in 0..n {
            let p = logistic(self.linear_predictor(i, param));
            let diff = p - *self.y.get(i, 0);
            for (j, grad_j) in grad.iter_mut().enumerate() {
                *grad_j += diff * (*self.x.get(i, j));
            }
        }
        Ok(grad)
    }
}

impl Hessian for LogitProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// `-ℓ(θ)`のHessian `X'WX`（`W = diag(pᵢ(1-pᵢ))`、対数尤度のHessian`-X'WX`の符号反転）。
    /// `run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照。
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let mut h = vec![vec![0.0; k]; k];
        for i in 0..n {
            let p = logistic(self.linear_predictor(i, param));
            let w = p * (1.0 - p);
            for (a, row) in h.iter_mut().enumerate().take(k) {
                let xa = *self.x.get(i, a);
                for (b, cell) in row.iter_mut().enumerate().take(k) {
                    *cell += w * xa * (*self.x.get(i, b));
                }
            }
        }
        Ok(h)
    }
}

/// Logitの推定結果。`fit`でのバリデーション・最適化・観測情報行列によるSE計算を
/// 通過した状態を表す。
///
/// 適合度統計量・限界効果等は未実装。`docs/planning/specs/
/// logit-probit-issue-breakdown.md`のB6以降（OPG/サンドイッチ/クラスターSE等）で
/// `fit`に追加していく想定。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct LogitEstimator {
    input: LogitInput,
    /// 係数（元のスケール。`standardize_columns`で標準化した空間で最適化した後、
    /// `destandardize_params`で逆変換済み）。`input.param_names()`と対応する
    params: Vec<f64>,
    /// 係数の分散共分散行列（元のスケール、k×k）。現時点では常に観測情報行列
    /// （`Σ = -H⁻¹`、`cov_type="classical"`/`"nonrobust"`相当）。限界効果
    /// （デルタ法、`logit-probit-issue-breakdown.md`のB9）で再利用するため、
    /// 対角成分（`std_errors`）だけでなく行列そのものを保持する。
    cov_params: Mat<f64>,
    /// 標準誤差（k, 元のスケール）。`cov_params`の対角成分の平方根
    std_errors: Vec<f64>,
    /// z統計量（k）= `params / std_errors`
    z_stats: Vec<f64>,
    /// 両側p値（k）。標準正規分布に基づく
    p_values: Vec<f64>,
    /// 信頼区間の下限（k）
    conf_lower: Vec<f64>,
    /// 信頼区間の上限（k）
    conf_upper: Vec<f64>,
    /// 収束したかどうか
    converged: bool,
    /// 実際の反復回数
    n_iter: usize,
}

impl LogitEstimator {
    /// `method`（Newton-Raphson/BFGS/L-BFGS）で負の対数尤度を最小化し、Logitの係数・
    /// 観測情報行列によるSE・z値・p値・信頼区間を推定する。
    ///
    /// `method`の選択に関わらず、収束点でのHessian評価（SE計算用）は常に解析的に行う
    /// （`run_solver`の実装方針、`docs/planning/specs/nonlinear-implementation-notes.md`
    /// 「engine内のtrait設計」参照）。BFGS/L-BFGSが最適化中に内部で保持する近似Hessianは
    /// 使い回さない。
    ///
    /// 初期値は常にゼロベクトル（`start_params`によるユーザー指定は未対応。
    /// `nonlinear-api-design.md`7章では確定オプションだが、対応するIssueが存在しないため
    /// 本Issueのスコープ外とし、ユーザー確認の上で見送った）。
    ///
    /// 設計行列は`standardize_columns`で内部的に標準化してから最適化し（勾配ノルムに
    /// 基づく収束判定`tol`が設計行列のスケールに依存しないようにするため、
    /// `docs/planning/specs/nonlinear-implementation-notes.md`「収束判定のtol」参照）、
    /// 収束後のパラメータを`destandardize_params`で元のスケールへ逆変換する。
    /// `run_solver`が返すHessianは標準化空間（θ_std）で評価されたものであり、
    /// 分散共分散行列もいったん標準化空間で計算してから`destandardize_cov_params`で
    /// 元のスケールへ逆変換する（`destandardize_params`を先に適用してから逆算するのではなく、
    /// 標準化空間のcov_paramsを直接destandardizeする。数学的に等価だが後者の方が
    /// 中間にHessianの逆変換を挟まず単純、`destandardize_cov_params`のdocコメント参照）。
    ///
    /// `cov_type`は現時点では観測情報行列（`"classical"`/`"nonrobust"`相当）のみで、
    /// 選択オプションはまだ無い（OPG/サンドイッチ/クラスターは
    /// `logit-probit-issue-breakdown.md`のB6・B7で追加）。検定分布は標準正規分布
    /// （`nonlinear-api-design.md`5章、OLSのt分布とは異なる）。
    ///
    /// `n <= k`（観測数が説明変数の数、定数項を含む、以下）のとき`CommonError::
    /// InsufficientObservations`で弾く閾値はOLSと同じ式だが、根拠は異なる。OLSでは
    /// 残差自由度`n-k`が0以下だと分散推定が原理的に不可能という数学的必要条件だが、
    /// LogitのようなMLEベースのモデルでは`n<=k`はほぼ確実に完全分離
    /// （perfect separation。ある説明変数の値でyの値が完全に分かれてしまい、
    /// 尤度が発散してMLEが存在しない状態）を引き起こす経験則としての安全側の判断。
    /// 数学的な必要条件ではないが、後続のProbit/Tobit実装でもこの閾値をそのまま
    /// 踏襲する（他パッケージがこの水準で明示的な検証をしていない場合でも、
    /// 発散した推定量を黙って返すよりは早期にエラーにする方針を優先する）。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - `max_iter`が0以下: `MleError::InvalidMaxIter`
    /// - 観測数`n`が`k`（定数項を含む説明変数の数）以下: `CommonError::InsufficientObservations`
    /// - `raise_on_non_convergence=true`かつ`max_iter`回で未収束: `MleError::NonConvergence`
    /// - 収束点（または`raise_on_non_convergence=false`時の打ち切り点）のHessianが特異
    ///   （設計行列の完全な多重共線性等）: `MleError::SingularHessian`
    pub fn fit(
        input: LogitInput,
        method: Method,
        max_iter: i64,
        tol: f64,
        raise_on_non_convergence: bool,
        confidence_level: f64,
    ) -> Result<Self, MleError> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
        }
        if max_iter <= 0 {
            return Err(MleError::InvalidMaxIter { max_iter });
        }

        let n = input.nobs();
        let k = input.k();
        if n <= k {
            return Err(CommonError::InsufficientObservations { n, k }.into());
        }

        let (x_std, scale) = standardize_columns(input.x(), input.has_intercept());
        let problem = LogitProblem {
            x: x_std,
            y: input.y().clone(),
        };

        let output = run_solver(
            problem,
            method,
            vec![0.0; k],
            max_iter as u64,
            tol,
            raise_on_non_convergence,
        )?;

        let params = destandardize_params(&output.params, &scale);

        let hessian_std = Mat::from_fn(k, k, |i, j| output.hessian[i][j]);
        let cov_params_std = observed_information_cov_params(&hessian_std, k)?;
        let cov_params = destandardize_cov_params(&cov_params_std, &scale);

        // `Normal::new(0.0, 1.0)`は標準正規分布であり、標準偏差が正であることを
        // 要求するstatrsの検証を常に満たすため、この`map_err`分岐は理論上到達不能
        // （`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照）。
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let alpha = 1.0 - confidence_level;
        let z_crit = normal.inverse_cdf(1.0 - alpha / 2.0);

        let mut std_errors = vec![0.0; k];
        let mut z_stats = vec![0.0; k];
        let mut p_values = vec![0.0; k];
        let mut conf_lower = vec![0.0; k];
        let mut conf_upper = vec![0.0; k];

        for j in 0..k {
            let se = (*cov_params.get(j, j)).sqrt();
            let z = params[j] / se;

            std_errors[j] = se;
            z_stats[j] = z;
            p_values[j] = 2.0 * (1.0 - normal.cdf(z.abs()));
            conf_lower[j] = params[j] - z_crit * se;
            conf_upper[j] = params[j] + z_crit * se;
        }

        Ok(Self {
            input,
            params,
            cov_params,
            std_errors,
            z_stats,
            p_values,
            conf_lower,
            conf_upper,
            converged: output.converged,
            n_iter: output.n_iter,
        })
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &LogitInput {
        &self.input
    }

    /// 係数（元のスケール）
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// 係数の分散共分散行列（元のスケール、k×k）
    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// 標準誤差（k、元のスケール）
    pub fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }

    /// z統計量（k）
    pub fn z_stats(&self) -> &[f64] {
        &self.z_stats
    }

    /// 両側p値（k）
    pub fn p_values(&self) -> &[f64] {
        &self.p_values
    }

    /// 信頼区間の下限（k）
    pub fn conf_lower(&self) -> &[f64] {
        &self.conf_lower
    }

    /// 信頼区間の上限（k）
    pub fn conf_upper(&self) -> &[f64] {
        &self.conf_upper
    }

    /// 収束したかどうか
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// 実際の反復回数
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_columns_with_intercept_prepends_const_column() {
        let y = vec![0.0, 1.0, 0.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.nobs(), 3);
        assert_eq!(input.k(), 2);
        assert!(input.has_intercept());
        assert_eq!(input.param_names(), ["const".to_string(), "x1".to_string()]);
        assert_eq!(input.dep_var_name(), "y");
        for i in 0..3 {
            assert_eq!(*input.x().get(i, 0), 1.0);
            assert_eq!(*input.x().get(i, 1), x_columns[0][i]);
            assert_eq!(*input.y().get(i, 0), y[i]);
        }
    }

    #[test]
    fn from_columns_without_intercept_omits_const_column() {
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.k(), 2);
        assert!(!input.has_intercept());
        assert_eq!(input.param_names(), ["x1".to_string(), "x2".to_string()]);
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_column_length() {
        let y = vec![0.0, 1.0, 0.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let result = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2
            })
        );
    }

    #[test]
    #[should_panic(expected = "x_columns and x_names must have the same length")]
    fn from_columns_panics_on_mismatched_names_arity() {
        // x_names.len() != x_columns.len()はengine_pybind側の実装バグでしか
        // 起こり得ない内部契約違反のため、`debug_assert!`でパニックする
        // （`OlsInput::from_columns`と同じ方針）。
        let y = vec![0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let _ = LogitInput::from_columns(&y, &x_columns, vec![], true, "y".to_string());
    }

    /// n=4, k=2（切片+x1）の小規模データ。`θ=[0,0]`のとき`p_i=0.5`（全観測で共通）と
    /// なり、`cost`/`gradient`/`hessian`が指数関数の評価なしに閉じた形（`softplus(0)=ln2`、
    /// `Hessian=0.25*X'X`等）で手計算できる。
    fn small_input() -> LogitInput {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn cost_gradient_hessian_match_closed_form_at_zero_params() {
        let input = small_input();
        let problem = LogitProblem::new(&input);
        let params = vec![0.0, 0.0];

        // cost = Σ softplus(0) = 4*ln(2)（y_i*z_i=0のため）
        let cost = problem.cost(&params).unwrap();
        assert!(
            (cost - 4.0 * std::f64::consts::LN_2).abs() < 1e-12,
            "{cost}"
        );

        // grad = Σ(0.5-y_i)*x_i、y=[0,1,0,1]・x1=[1,2,3,4]・切片=1
        // 切片成分: 0.5-0.5+0.5-0.5=0、x1成分: 0.5*1-0.5*2+0.5*3-0.5*4=-1.0
        let grad = problem.gradient(&params).unwrap();
        assert!(grad[0].abs() < 1e-12, "{:?}", grad);
        assert!((grad[1] - (-1.0)).abs() < 1e-12, "{:?}", grad);

        // Hessian = 0.25 * X'X、X'X = [[4,10],[10,30]]（n=4, Σx1=10, Σx1²=30）
        let hessian = problem.hessian(&params).unwrap();
        assert!((hessian[0][0] - 1.0).abs() < 1e-12, "{:?}", hessian);
        assert!((hessian[0][1] - 2.5).abs() < 1e-12, "{:?}", hessian);
        assert!((hessian[1][0] - 2.5).abs() < 1e-12, "{:?}", hessian);
        assert!((hessian[1][1] - 7.5).abs() < 1e-12, "{:?}", hessian);
    }

    #[test]
    fn scores_match_closed_form_at_zero_params() {
        let input = small_input();
        let problem = LogitProblem::new(&input);
        let scores = problem.scores(&[0.0, 0.0]);

        // score_i = (y_i-0.5)*x_i
        let expected = [(-0.5, -0.5), (0.5, 1.0), (-0.5, -1.5), (0.5, 2.0)];
        for (i, (e0, e1)) in expected.iter().enumerate() {
            assert!((*scores.get(i, 0) - e0).abs() < 1e-12, "row {i}");
            assert!((*scores.get(i, 1) - e1).abs() < 1e-12, "row {i}");
        }
    }

    #[test]
    fn scores_sum_to_negative_gradient() {
        // scoresは対数尤度の生のスコア（符号反転なし）、gradientはCostFunction
        // （負の対数尤度）の勾配のため、観測方向に合計すると符号が逆になるはず。
        let input = small_input();
        let problem = LogitProblem::new(&input);
        let params = vec![0.3, -0.2];

        let scores = problem.scores(&params);
        let grad = problem.gradient(&params).unwrap();
        let n = scores.nrows();

        for j in 0..2 {
            let sum: f64 = (0..n).map(|i| *scores.get(i, j)).sum();
            assert!(
                (sum - (-grad[j])).abs() < 1e-9,
                "j={j}, sum={sum}, grad={:?}",
                grad
            );
        }
    }

    #[test]
    fn gradient_matches_numerical_differentiation_of_cost() {
        let input = small_input();
        let problem = LogitProblem::new(&input);
        let params = vec![0.3, -0.2];
        let h = 1e-6;

        let analytic = problem.gradient(&params).unwrap();
        for j in 0..2 {
            let mut plus = params.clone();
            plus[j] += h;
            let mut minus = params.clone();
            minus[j] -= h;
            let numeric =
                (problem.cost(&plus).unwrap() - problem.cost(&minus).unwrap()) / (2.0 * h);
            assert!(
                (analytic[j] - numeric).abs() < 1e-6,
                "j={j}, analytic={}, numeric={}",
                analytic[j],
                numeric
            );
        }
    }

    #[test]
    fn hessian_matches_numerical_differentiation_of_gradient() {
        let input = small_input();
        let problem = LogitProblem::new(&input);
        let params = vec![0.3, -0.2];
        let h = 1e-5;

        let analytic = problem.hessian(&params).unwrap();
        for j in 0..2 {
            let mut plus = params.clone();
            plus[j] += h;
            let mut minus = params.clone();
            minus[j] -= h;
            let grad_plus = problem.gradient(&plus).unwrap();
            let grad_minus = problem.gradient(&minus).unwrap();
            for i in 0..2 {
                let numeric = (grad_plus[i] - grad_minus[i]) / (2.0 * h);
                assert!(
                    (analytic[i][j] - numeric).abs() < 1e-4,
                    "i={i}, j={j}, analytic={}, numeric={}",
                    analytic[i][j],
                    numeric
                );
            }
        }
    }

    #[test]
    fn logistic_and_softplus_are_numerically_stable_for_large_inputs() {
        assert!((logistic(1000.0) - 1.0).abs() < 1e-12);
        assert!(logistic(-1000.0).abs() < 1e-12);
        assert!(softplus(1000.0).is_finite());
        assert!((softplus(1000.0) - 1000.0).abs() < 1e-9);
        assert!(softplus(-1000.0).abs() < 1e-12);
    }

    /// 切片のみ（説明変数なし）のLogitは、MLEの一階条件`Σy_i - n*p = 0`から
    /// `p = ȳ`、すなわち`θ̂ = ln(ȳ/(1-ȳ))`という閉じた形の解析解を持つ
    /// （`fit`が最適化ロジックを経ずに正しい値へ収束することを検証できる、
    /// 数値最適化を要する手法としては数少ない厳密な既知解のケース）。
    fn intercept_only_input() -> LogitInput {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        LogitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap()
    }

    #[test]
    fn fit_newton_converges_to_closed_form_solution_for_intercept_only_model() {
        let input = intercept_only_input();
        let estimator = LogitEstimator::fit(input, Method::Newton, 35, 1e-6, true, 0.95).unwrap();

        let y_bar: f64 = 4.0 / 7.0;
        let expected = (y_bar / (1.0 - y_bar)).ln();

        assert!(estimator.converged());
        assert_eq!(estimator.params().len(), 1);
        assert!(
            (estimator.params()[0] - expected).abs() < 1e-6,
            "params={:?}, expected={}",
            estimator.params(),
            expected
        );
        // 切片のみの1次元凹関数のNewton法は数回で収束するはず
        assert!(estimator.n_iter() <= 10, "n_iter={}", estimator.n_iter());
    }

    /// 切片のみモデルは観測情報行列も閉じた形で書ける: 全観測で`p_i=ȳ`（closed form）
    /// なので、Hessianは`-Σp_i(1-p_i) = -n*ȳ*(1-ȳ)`というスカラーになり、
    /// `Var(θ̂) = -H⁻¹ = 1/(n*ȳ*(1-ȳ))`。z値・p値・信頼区間はこの分散から
    /// 標準正規分布（統計独立に`statrs::Normal`で検算）で導出できる。
    #[test]
    fn fit_computes_std_errors_z_stats_p_values_and_ci_matching_closed_form_for_intercept_only_model()
     {
        let estimator =
            LogitEstimator::fit(intercept_only_input(), Method::Newton, 35, 1e-6, true, 0.95)
                .unwrap();

        let n: f64 = 7.0;
        let y_bar: f64 = 4.0 / 7.0;
        let expected_var = 1.0 / (n * y_bar * (1.0 - y_bar));
        let expected_se = expected_var.sqrt();

        // Newtonの収束判定（勾配ノルム`tol=1e-6`）による数値誤差があるため、
        // 他の閉じた形テスト（`fit_newton_converges_to_closed_form_solution_...`）と
        // 同じ桁の許容誤差（1e-6）を使う（1e-9のような厳しすぎる許容誤差は、収束点が
        // 解析解からわずかにズレることに起因する誤検出を招く）。
        assert!((*estimator.cov_params().get(0, 0) - expected_var).abs() < 1e-6);
        assert!((estimator.std_errors()[0] - expected_se).abs() < 1e-6);

        let expected_z = estimator.params()[0] / expected_se;
        assert!((estimator.z_stats()[0] - expected_z).abs() < 1e-6);

        // p値・信頼区間はstatrsのNormalで独立に検算する（本体実装と同じ計算式を
        // 繰り返すのではなく、標準正規分布の性質から直接導出する）。
        let normal = Normal::new(0.0, 1.0).unwrap();
        let expected_p = 2.0 * (1.0 - normal.cdf(expected_z.abs()));
        assert!((estimator.p_values()[0] - expected_p).abs() < 1e-6);

        let z_crit = normal.inverse_cdf(0.975);
        let expected_lower = estimator.params()[0] - z_crit * expected_se;
        let expected_upper = estimator.params()[0] + z_crit * expected_se;
        assert!((estimator.conf_lower()[0] - expected_lower).abs() < 1e-6);
        assert!((estimator.conf_upper()[0] - expected_upper).abs() < 1e-6);
    }

    /// 多変量（説明変数が2つ以上）の場合、標準誤差に閉じた形の解析解は無いため、
    /// `cov_params`の対称性・各種統計量の内部整合性（z値・信頼区間の定義式通りの関係）を
    /// 検証する回帰テスト。特に`destandardize_cov_params`が非対角成分も含めて
    /// 正しく元のスケールへ逆変換できているかを確認する（対角成分だけでは
    /// `destandardize_cov_params`の`stds[i]*stds[j]`の掛け違い（転置ミス等）を検出できない）。
    #[test]
    fn fit_cov_params_is_symmetric_and_stats_are_internally_consistent() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(input, Method::Newton, 35, 1e-8, true, 0.95).unwrap();
        let k = 3;

        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *estimator.cov_params().get(j, i)).abs()
                        < 1e-9,
                    "cov_params is not symmetric at ({i},{j})"
                );
            }
            assert!(
                *estimator.cov_params().get(i, i) > 0.0,
                "diagonal[{i}] <= 0"
            );
        }

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal.inverse_cdf(0.975);
        for j in 0..k {
            let se = estimator.std_errors()[j];
            assert!((se * se - *estimator.cov_params().get(j, j)).abs() < 1e-9);
            assert!((estimator.z_stats()[j] - estimator.params()[j] / se).abs() < 1e-9);
            assert!(
                (estimator.conf_upper()[j] - estimator.conf_lower()[j] - 2.0 * z_crit * se).abs()
                    < 1e-9
            );
        }
    }

    /// `newton`と同じデータセット（既知の解析解を持つ切片のみモデル）で`bfgs`/`lbfgs`を
    /// 実行し、いずれも同じ解析解へ収束することを検証する（Issue #57完了条件）。
    #[test]
    fn fit_bfgs_and_lbfgs_converge_to_same_solution_as_newton() {
        let y_bar: f64 = 4.0 / 7.0;
        let expected = (y_bar / (1.0 - y_bar)).ln();

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator =
                LogitEstimator::fit(intercept_only_input(), method, 100, 1e-6, true, 0.95).unwrap();

            assert!(estimator.converged(), "{:?}", method);
            assert!(
                (estimator.params()[0] - expected).abs() < 1e-4,
                "method={:?}, params={:?}, expected={}",
                method,
                estimator.params(),
                expected
            );
        }
    }

    /// 切片のみモデル（`intercept_only_input`）は`x`が定数列（切片）だけのため、
    /// `standardize_columns`のスケーリングが実質no-op（`stds`が全て`1.0`のまま）になり、
    /// 標準化・逆標準化の往復ロジックを通らない（rust-reviewer指摘）。このテストは
    /// 非自明なスケール（`std`が1から離れた値）を持つ説明変数を含むデータセットで
    /// `newton`/`bfgs`/`lbfgs`を実行し、3手法が同じ解へ収束することを検証する
    /// （閉じた形の解析解は存在しないため、`newton`の結果を参照値として使う
    /// クロスメソッド一致検証。標準化空間でのBFGSの初期逆Hessian
    /// （`identity_matrix(k)`）・`destandardize_params`が正しく機能していることの
    /// 間接的な検証になる）。
    #[test]
    fn fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0]];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let newton =
            LogitEstimator::fit(make_input(), Method::Newton, 35, 1e-8, true, 0.95).unwrap();
        assert!(newton.converged());

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator =
                LogitEstimator::fit(make_input(), method, 200, 1e-8, true, 0.95).unwrap();

            assert!(estimator.converged(), "{:?}", method);
            for j in 0..2 {
                assert!(
                    (estimator.params()[j] - newton.params()[j]).abs() < 1e-4,
                    "method={:?}, j={j}, params={:?}, newton_params={:?}",
                    method,
                    estimator.params(),
                    newton.params()
                );
            }
        }
    }

    #[test]
    fn fit_returns_invalid_confidence_level_error_out_of_range() {
        let result =
            LogitEstimator::fit(intercept_only_input(), Method::Newton, 35, 1e-6, true, 1.5);
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    #[test]
    fn fit_returns_invalid_max_iter_error_for_non_positive_max_iter() {
        let result =
            LogitEstimator::fit(intercept_only_input(), Method::Newton, 0, 1e-6, true, 0.95);
        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidMaxIter { max_iter: 0 }
        );
    }

    #[test]
    fn fit_returns_insufficient_observations_error_when_n_less_equal_k() {
        let y = vec![0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = LogitEstimator::fit(input, Method::Newton, 35, 1e-6, true, 0.95);
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
    }

    #[test]
    fn fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix() {
        // x2 = 2*x1（完全な多重共線性）。θ=0でのHessianは0.25*X'Xで、X'X自体が
        // 構造的に特異（yの値に関わらず常に特異）なので、Newtonの初回ステップで
        // 確実にnewton_stepの特異性検出に引っかかる（完全分離のような「収束の
        // 挙動に依存する」ケースと異なり、決定的に再現できる）。
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = LogitEstimator::fit(input, Method::Newton, 35, 1e-6, true, 0.95);
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    /// 同じ完全な多重共線性のデータセットを`bfgs`/`lbfgs`で最適化した場合の
    /// `SingularHessian`伝播経路（Issue #129）: `newton`は`newton_step`内の特異性検出
    /// （最適化のステップ計算中）で検出するが、`bfgs`/`lbfgs`は`newton_step`を
    /// 一切経由しない（準ニュートン法は内部の近似逆Hessianで降下方向を決めるため、
    /// モデルの解析的Hessianの特異性に依存しない）。この場合、収束後に
    /// `observed_information_cov_params`（`neg_hessian_inverse`）が呼ぶ
    /// `ensure_well_conditioned_symmetric_matrix`（固有値ベースの悪条件検出）が、
    /// `bfgs`/`lbfgs`にとって唯一の特異性検出経路になる。修正前（Issue #129発覚時点）
    /// はこのテストは失敗していた（非ピボットCholeskyが特異性を検出できず、
    /// 桁違いに巨大な値を含む`Ok`が返っていた）。両方のソルバーで同じコードパスを
    /// 通ることをそれぞれ独立に確認する。
    #[test]
    fn fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_bfgs_and_lbfgs()
     {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];

        for method in [Method::Bfgs, Method::Lbfgs] {
            let input = LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result = LogitEstimator::fit(input, method, 100, 1e-6, true, 0.95);
            assert!(
                matches!(result, Err(MleError::SingularHessian)),
                "method={:?}, result={:?}",
                method,
                result
            );
        }
    }

    #[test]
    fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
        let result =
            LogitEstimator::fit(intercept_only_input(), Method::Newton, 1, 1e-12, true, 0.95);
        assert!(
            matches!(result, Err(MleError::NonConvergence { .. })),
            "{:?}",
            result
        );
    }

    #[test]
    fn fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            1,
            1e-12,
            false,
            0.95,
        )
        .unwrap();
        assert!(!estimator.converged());
    }
}
