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
    CovType, MarginalEffectsAt, Method, MleError, SandwichVariant, cluster_cov_params,
    destandardize_cov_params, destandardize_params, observed_information_cov_params,
    opg_cov_params, run_solver, sandwich_cov_params, standardize_columns,
};
use crate::validation::validate_cluster_groups;
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

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

/// 対数尤度 `ℓ(θ) = Σᵢ [yᵢzᵢ - softplus(zᵢ)]`（`zᵢ=xᵢ'θ`、モジュール冒頭の数式参照）を
/// `x`・`y`・`params`から直接計算する。`LogitProblem::cost`（`-ℓ(θ)`、argminの
/// `CostFunction`）と同じ数式のΣ部分を共有する（`cost`はこの関数を符号反転して呼ぶ）。
/// argminのトレイトが要求する`Result`型を経由する必要が無い内部専用の計算
/// （適合度統計量向け、収束後のパラメータで1回だけ評価する）のため、独立した
/// 関数として切り出している。
fn log_likelihood(x: &Mat<f64>, y: &Mat<f64>, params: &[f64]) -> f64 {
    let n = x.nrows();
    (0..n)
        .map(|i| {
            let z: f64 = (0..x.ncols()).map(|j| *x.get(i, j) * params[j]).sum();
            (*y.get(i, 0)) * z - softplus(z)
        })
        .sum()
}

/// 切片のみモデルの対数尤度 `ℓ_null = n1*ln(ȳ) + n0*ln(1-ȳ)`（`ȳ=n1/n`の閉じた形の
/// 解析解、`LogitEstimator`の`log_likelihood_null`フィールドdocコメント参照）を`y`から
/// 直接計算する。`n1`または`n0`が0（全観測が同じ値）のときの`0*ln(0)`（NaN）を避けるため、
/// 該当項を明示的に0として扱う（情報理論の`0 log 0 = 0`規約）。`log_likelihood`と同じ理由
/// （`Result`を経由しない内部専用の計算、退化ケースを`fit()`の反復最適化を経由せず
/// 直接テストできるようにするため）で独立した関数として切り出している。
fn log_likelihood_null(y: &Mat<f64>) -> f64 {
    let n = y.nrows();
    let n1: f64 = (0..n).map(|i| *y.get(i, 0)).sum();
    let n0 = n as f64 - n1;
    let y_bar = n1 / n as f64;
    (if n1 > 0.0 { n1 * y_bar.ln() } else { 0.0 })
        + (if n0 > 0.0 {
            n0 * (1.0 - y_bar).ln()
        } else {
            0.0
        })
}

/// 限界効果（`LogitEstimator::marginal_effects`のdocコメント「数式（デルタ法）」参照）の
/// `at="overall"`（AME）における`w=(1/n)Σᵢpᵢ(1-pᵢ)`・`s_m=(1/n)Σᵢ(1-2pᵢ)pᵢ(1-pᵢ)xᵢₘ`を
/// 全観測を1回走査して計算する。
fn overall_w_and_s(x: &Mat<f64>, params: &[f64]) -> (f64, Vec<f64>) {
    let n = x.nrows();
    let k = x.ncols();
    let mut w = 0.0;
    let mut s = vec![0.0; k];
    for i in 0..n {
        let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        let p = logistic(z);
        let pq = p * (1.0 - p);
        w += pq;
        let coef = (1.0 - 2.0 * p) * pq;
        for (m, s_m) in s.iter_mut().enumerate() {
            *s_m += coef * (*x.get(i, m));
        }
    }
    let n_f = n as f64;
    w /= n_f;
    for s_m in s.iter_mut() {
        *s_m /= n_f;
    }
    (w, s)
}

/// 限界効果の`at="mean"`/`"median"`における、代表点`x_bar`（各説明変数の標本平均または
/// 中央値）で評価した`w=p̄(1-p̄)`・`s_m=(1-2p̄)p̄(1-p̄)x̄ₘ`（`p̄=Λ(x̄'θ)`）を計算する。
fn at_point_w_and_s(x_bar: &[f64], params: &[f64]) -> (f64, Vec<f64>) {
    let k = x_bar.len();
    let z: f64 = (0..k).map(|m| x_bar[m] * params[m]).sum();
    let p = logistic(z);
    let pq = p * (1.0 - p);
    let coef = (1.0 - 2.0 * p) * pq;
    let s: Vec<f64> = (0..k).map(|m| coef * x_bar[m]).collect();
    (pq, s)
}

/// `overall_w_and_s`/`at_point_w_and_s`が返す`(w,s)`から、限界効果`dydx_j=w*θⱼ`と
/// そのヤコビアン`jacobian[j][m]=∂dydx_j/∂θₘ=θⱼ*s_m + [j==m]*w`を計算する
/// （`LogitEstimator::marginal_effects`のdocコメント「数式（デルタ法）」参照。
/// AME・mean・medianのいずれも`g_j(θ)=w(θ)*θⱼ`という同じ形に帰着するため、
/// `w`・`s`の計算方法（`at`ごとに異なる）とこの式（`at`に依らず共通）を分離できる）。
fn dydx_and_jacobian(k: usize, params: &[f64], w: f64, s: &[f64]) -> (Vec<f64>, Mat<f64>) {
    let dydx: Vec<f64> = (0..k).map(|j| w * params[j]).collect();
    let jacobian = Mat::from_fn(k, k, |j, m| params[j] * s[m] + if j == m { w } else { 0.0 });
    (dydx, jacobian)
}

/// 説明変数ごとの標本平均（列ごと、`marginal_effects`の`at="mean"`用）。
fn column_means(x: &Mat<f64>) -> Vec<f64> {
    let n = x.nrows();
    let k = x.ncols();
    (0..k)
        .map(|j| (0..n).map(|i| *x.get(i, j)).sum::<f64>() / (n as f64))
        .collect()
}

/// 説明変数ごとの標本中央値（列ごと、`marginal_effects`の`at="median"`用）。`n`が偶数の
/// 場合は中央2値の平均。
///
/// `partial_cmp().unwrap()`について: `x`の値はNaN/無限大を含まないことが
/// `engine_pybind::column_extraction`側で既に保証されている前提（`engine`の責務境界の
/// 内側であり、クリーンな値しか受け取らない。OLSの`time_ordering`と同じ扱い、
/// `engine/src/linear/ols.rs`参照）。
fn column_medians(x: &Mat<f64>) -> Vec<f64> {
    let n = x.nrows();
    let k = x.ncols();
    (0..k)
        .map(|j| {
            let mut col: Vec<f64> = (0..n).map(|i| *x.get(i, j)).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if n % 2 == 1 {
                col[n / 2]
            } else {
                (col[n / 2 - 1] + col[n / 2]) / 2.0
            }
        })
        .collect()
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
        Ok(-log_likelihood(&self.x, &self.y, param))
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

/// `LogitEstimator::marginal_effects`の結果。`coef_table`と同じ行指向
/// （`dydx`/`std_err`/`z`/`p_value`/`conf_low`/`conf_high`、`nonlinear-api-design.md`
/// 6章）。定数項（切片）は行から除外する（切片の限界効果は経済学的に意味を持たない、
/// statsmodelsの`get_margeff()`と同じ扱い）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct MarginalEffects {
    /// 説明変数名（定数項を除く）。`LogitInput::param_names()`から定数項を除いたもの
    param_names: Vec<String>,
    /// 限界効果 `dy/dx`
    dydx: Vec<f64>,
    /// デルタ法標準誤差
    std_errors: Vec<f64>,
    /// z統計量
    z_stats: Vec<f64>,
    /// 両側p値
    p_values: Vec<f64>,
    /// 信頼区間の下限
    conf_lower: Vec<f64>,
    /// 信頼区間の上限
    conf_upper: Vec<f64>,
}

impl MarginalEffects {
    /// 説明変数名（定数項を除く）
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// 限界効果 `dy/dx`
    pub fn dydx(&self) -> &[f64] {
        &self.dydx
    }

    /// デルタ法標準誤差
    pub fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }

    /// z統計量
    pub fn z_stats(&self) -> &[f64] {
        &self.z_stats
    }

    /// 両側p値
    pub fn p_values(&self) -> &[f64] {
        &self.p_values
    }

    /// 信頼区間の下限
    pub fn conf_lower(&self) -> &[f64] {
        &self.conf_lower
    }

    /// 信頼区間の上限
    pub fn conf_upper(&self) -> &[f64] {
        &self.conf_upper
    }
}

/// Logitの推定結果。`fit`でのバリデーション・最適化・`cov_type`に応じたSE計算・
/// 適合度統計量の計算を通過した状態を表す。
///
/// `predict`/`pred_table`は未実装。`docs/planning/specs/
/// logit-probit-issue-breakdown.md`のB10で`fit`とは別のメソッドとして追加していく想定。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct LogitEstimator {
    input: LogitInput,
    /// 係数（元のスケール。`standardize_columns`で標準化した空間で最適化した後、
    /// `destandardize_params`で逆変換済み）。`input.param_names()`と対応する
    params: Vec<f64>,
    /// 係数の分散共分散行列（元のスケール、k×k）。`fit`に渡した`cov_type`に応じて
    /// 観測情報行列（`Classical`）・OPG（`Opg`）・サンドイッチ型（`Hc0`/`Hc1`）の
    /// いずれかで計算される。限界効果（デルタ法、`logit-probit-issue-breakdown.md`の
    /// B9）で再利用するため、対角成分（`std_errors`）だけでなく行列そのものを保持する。
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
    /// 対数尤度 `ℓ(θ̂)`（収束点で評価）
    log_likelihood: f64,
    /// 切片のみモデルの対数尤度 `ℓ(θ̂_null)`。`ȳ=n1/n`の閉じた形の解析解
    /// （`nᵢ log(ȳ) + (1-nᵢ) log(1-ȳ)`の総和）から直接計算する（ソルバーの
    /// 再フィットは経由しない。ユーザー確認済み、`log_likelihood`のdocコメント
    /// 「切片のみモデルのllf」参照）。`include_intercept`の値に関わらず常にこの
    /// 「切片のみ」モデルを参照する（`nonlinear-api-design.md`5章の定義通り、
    /// statsmodelsも`k_constant`の有無に関わらず同じ挙動）。
    ///
    /// **`include_intercept=false`のとき、この値が参照する「切片のみ」モデルは
    /// フィット対象のモデルの部分集合（入れ子）にならない**（statsmodelsで
    /// `Logit(y, X_without_const).fit()`をした場合と同じ状況、実測でも
    /// `llr`が負値・`llr_pvalue`がほぼ1.0になる例を確認済み）。この場合
    /// `lr_statistic`が負になったり`lr_p_value`が統計的に意味の薄い値になったり
    /// しうるが、これはstatsmodels準拠の仕様上の挙動でありバグではない
    /// （rust-reviewerの指摘を受けて明記）。
    log_likelihood_null: f64,
    /// 尤度比検定統計量 `2*(ℓ(θ̂)-ℓ(θ̂_null))`。`include_intercept=false`のときの
    /// 非入れ子性については`log_likelihood_null`のdocコメント参照
    lr_statistic: f64,
    /// 尤度比検定のp値（自由度`df_model`のカイ二乗分布、上側確率）。
    /// `df_model==0`（説明変数なし）のときはNaN（OLSの`f_p_value`と同じ扱い）
    lr_p_value: f64,
    /// McFadden疑似決定係数 `1 - ℓ(θ̂)/ℓ(θ̂_null)`
    pseudo_r_squared: f64,
    /// 赤池情報量規準 `-2ℓ(θ̂) + 2k`
    aic: f64,
    /// ベイズ情報量規準 `-2ℓ(θ̂) + ln(n)k`
    bic: f64,
    /// モデルの自由度 `k-1`（切片のみのnullモデルとのパラメータ数差。statsmodels準拠、
    /// `include_intercept`の値に関わらず常にこの式。ユーザー確認済み）
    df_model: usize,
    /// 残差自由度 `n-k`
    df_resid: usize,
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
    /// `cov_type`は観測情報行列（`Classical`）・OPG（`Opg`）・サンドイッチ型
    /// （`Hc0`/`Hc1`）・クラスターロバスト（`Cluster`）に対応する。`Cluster`の
    /// グループキー未指定・クラスター数不足は、最適化を実行する前（`fit()`冒頭）に
    /// 検証して早期に返す（OLSの`cov_type=Cluster`は閉形式解のため事後検証でも
    /// コストが変わらないが、Logitは反復最適化のため無駄な計算を避ける）。
    /// `Opg`/`Hc0`/`Hc1`/`Cluster`は収束点での観測ごとのスコア（`LogitProblem::
    /// scores`）が必要なため、標準化空間の設計行列を保持したまま`LogitProblem`を
    /// クローンしておき（`argmin::core::Executor`向けに元々`Clone`を要求しているため
    /// 追加コストは`Clone`実装自体のみ）、`run_solver`が返す収束点のパラメータで
    /// 評価する。検定分布は標準正規分布（`nonlinear-api-design.md`5章、OLSのt分布とは
    /// 異なる）。
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
    /// - `cov_type=Opg`でOPG行列（`Σᵢ sᵢsᵢ'`）が特異: `MleError::SingularOpgMatrix`
    /// - `cov_type=Cluster`でグループキー未指定: `CommonError::MissingClusterColumn`
    /// - `cov_type=Cluster`でクラスター数が2未満: `CommonError::InsufficientClusters`
    pub fn fit(
        input: LogitInput,
        method: Method,
        max_iter: i64,
        tol: f64,
        raise_on_non_convergence: bool,
        cov_type: CovType,
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
        if let CovType::Cluster { groups } = &cov_type {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
            validate_cluster_groups(groups, n)?;
        }

        let (x_std, scale) = standardize_columns(input.x(), input.has_intercept());
        let problem = LogitProblem {
            x: x_std,
            y: input.y().clone(),
        };
        // `cov_type`がOPG/サンドイッチ型の場合、収束点でのスコア評価に元の
        // `LogitProblem`（標準化空間のx_std）が必要になる。`run_solver`は`problem`の
        // 所有権を取り込む（内部で保持していたモデルを呼び出し元へ返さない設計）ため、
        // 事前にクローンしておく必要がある（`LogitProblem`は`argmin::core::Executor`
        // 向けに元々`Clone`を要求しているため、この用途のための追加のtraitではない）。
        // `Classical`はスコアを使わないため、無駄な複製（設計行列を含む）を避けるために
        // `cov_type`に応じて条件付きで行う（rust-reviewer指摘）。
        let problem_for_scores = match &cov_type {
            CovType::Classical => None,
            CovType::Opg | CovType::Hc0 | CovType::Hc1 | CovType::Cluster { .. } => {
                Some(problem.clone())
            }
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
        // `problem_for_scores.as_ref().expect(...)`は各非`Classical`分岐でのみ呼ばれ、
        // 直前の`match cov_type { CovType::Classical => None, _ => Some(...) }`により
        // 常に`Some`であることが保証されている内部契約（`cov_type`という同じ値で
        // 2回目のmatchを行うことになるが、パニックしないことをコンパイラの型システムでは
        // 表現できないため、`expect`のメッセージで契約を明記して防御的に扱う）。
        let cov_params_std = match &cov_type {
            CovType::Classical => observed_information_cov_params(&hessian_std, k)?,
            CovType::Opg => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Opg");
                opg_cov_params(&problem.scores(&output.params), k)?
            }
            CovType::Hc0 => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Hc0");
                sandwich_cov_params(
                    &hessian_std,
                    &problem.scores(&output.params),
                    n,
                    k,
                    SandwichVariant::Hc0,
                )?
            }
            CovType::Hc1 => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Hc1");
                sandwich_cov_params(
                    &hessian_std,
                    &problem.scores(&output.params),
                    n,
                    k,
                    SandwichVariant::Hc1,
                )?
            }
            CovType::Cluster { groups } => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Cluster");
                // `groups`のNone・クラスター数不足の検証はfit()冒頭で完了済み
                // （MissingClusterColumn/InsufficientClustersを最適化前に早期に返す
                // ため）。ここでの`expect`はその契約を明記する防御的な扱い。
                let groups = groups
                    .as_ref()
                    .expect("groups is validated as Some at the top of fit()");
                cluster_cov_params(&hessian_std, &problem.scores(&output.params), n, k, groups)?
            }
        };
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

        let llf = log_likelihood(input.x(), input.y(), &params);
        let llnull = log_likelihood_null(input.y());

        let lr_statistic = 2.0 * (llf - llnull);
        // `k.saturating_sub(1)`: `include_intercept=false`かつ説明変数も無い（`k=0`）という
        // 病的な入力（`fit()`冒頭の`n<=k`チェックは`n>=1`なら通過してしまう）でも
        // このフィールド単体はアンダーフローしないための防御。ただし`k=0`のとき
        // 実際には本箇所に到達する前（`cov_params`計算経路）で別の既知の問題により
        // 到達不能（トラッキング: 別issue、本Issueのスコープ外。ユーザー確認済み）。
        let df_model = k.saturating_sub(1);
        let lr_p_value = if df_model == 0 {
            // 説明変数が定数項のみ（傾き係数が無い）モデル。検定対象が存在しないため
            // OLSの`f_p_value`（`df_model=0`時にNaN）と同じ扱い（ユーザー確認済み）。
            f64::NAN
        } else {
            // `df_model>0`が保証されているため、`ChiSquared::new`の失敗
            // （自由度が正であることの検証）は理論上到達不能
            // （`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照）。
            let chi2 = ChiSquared::new(df_model as f64)
                .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
            1.0 - chi2.cdf(lr_statistic)
        };

        let pseudo_r_squared = 1.0 - llf / llnull;
        let aic = -2.0 * llf + 2.0 * (k as f64);
        let bic = -2.0 * llf + (n as f64).ln() * (k as f64);
        let df_resid = n - k;

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
            log_likelihood: llf,
            log_likelihood_null: llnull,
            lr_statistic,
            lr_p_value,
            pseudo_r_squared,
            aic,
            bic,
            df_model,
            df_resid,
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

    /// 対数尤度 `ℓ(θ̂)`
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// 切片のみモデルの対数尤度 `ℓ(θ̂_null)`
    pub fn log_likelihood_null(&self) -> f64 {
        self.log_likelihood_null
    }

    /// 尤度比検定統計量
    pub fn lr_statistic(&self) -> f64 {
        self.lr_statistic
    }

    /// 尤度比検定のp値（`df_model==0`のときNaN）
    pub fn lr_p_value(&self) -> f64 {
        self.lr_p_value
    }

    /// McFadden疑似決定係数
    pub fn pseudo_r_squared(&self) -> f64 {
        self.pseudo_r_squared
    }

    /// 赤池情報量規準
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// ベイズ情報量規準
    pub fn bic(&self) -> f64 {
        self.bic
    }

    /// 観測数。`self.input.nobs()`への委譲（`OlsEstimator`と同じパターン、
    /// `n`という同じ値の出どころを2つに分けない）
    pub fn n_obs(&self) -> usize {
        self.input.nobs()
    }

    /// モデルの自由度（`k-1`）
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 残差自由度（`n-k`）
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// 限界効果（`marginal_effects`）。`fit()`とは独立した別メソッド（`fit()`のReturn
    /// 本体には含めない、`nonlinear-api-design.md`6章で確定済み）。`fit()`時の
    /// `cov_params`を再利用するため再最適化は不要（`confidence_level`は`fit()`とは
    /// 独立したパラメータとして受け取り、`fit()`時の値に縛られず事後的に異なる
    /// CI幅を見られるようにする）。
    ///
    /// ## 数式（デルタ法）
    ///
    /// `p_i = Λ(x_i'θ)`のとき、変数`j`（連続変数として扱う。`dummy=False`が既定の
    /// statsmodelsの`get_margeff()`に倣い、離散変数の自動判定は行わない設計、
    /// `nonlinear-implementation-notes.md`「限界効果」参照）の限界効果は
    /// `dy/dx_j = p(1-p)θ_j`。
    ///
    /// - `at="overall"`（AME）: `g_j(θ) = w(θ)*θ_j`、`w(θ) = (1/n)Σᵢ pᵢ(1-pᵢ)`
    /// - `at="mean"`/`"median"`: `g_j(θ) = w(θ)*θ_j`、`w(θ) = p̄(1-p̄)`
    ///   （`p̄=Λ(x̄'θ)`、`x̄`は各説明変数の標本平均または中央値からなる代表点）
    ///
    /// いずれも同じ`g_j(θ)=w(θ)*θ_j`という形に帰着するため、`w`とその勾配
    /// `s_m=∂w/∂θ_m`さえ計算できれば、ヤコビアンは
    /// `∂g_j/∂θ_m = θ_j*s_m + [j==m]*w`という共通の式で書ける
    /// （`overall_w_and_s`/`at_point_w_and_s`が`(w,s)`を計算し、
    /// `dydx_and_jacobian`が上記の共通式を適用する）。
    ///
    /// 変数`j`の分散は`Var(g_j) = jac_j · Σ · jac_jᵀ`（`jac_j`はヤコビアンの`j`行目、
    /// `Σ=cov_params`）。標準誤差はこの平方根、検定分布は標準正規分布
    /// （`fit()`本体と同じ、`nonlinear-api-design.md`5章）。
    ///
    /// 定数項（切片）は出力から除外する（切片の限界効果は意味を持たない、
    /// statsmodelsも同様）。
    ///
    /// # Errors
    /// `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    pub fn marginal_effects(
        &self,
        at: MarginalEffectsAt,
        confidence_level: f64,
    ) -> Result<MarginalEffects, MleError> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
        }

        let x = self.input.x();
        let k = self.input.k();
        let (w, s) = match at {
            MarginalEffectsAt::Overall => overall_w_and_s(x, &self.params),
            MarginalEffectsAt::Mean => at_point_w_and_s(&column_means(x), &self.params),
            MarginalEffectsAt::Median => at_point_w_and_s(&column_medians(x), &self.params),
        };
        let (dydx, jacobian) = dydx_and_jacobian(k, &self.params, w, &s);

        // `Normal::new(0.0, 1.0)`は標準正規分布であり、標準偏差が正であることを
        // 要求するstatrsの検証を常に満たすため、この`map_err`分岐は理論上到達不能
        // （`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照、`fit()`と同じ扱い）。
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let alpha = 1.0 - confidence_level;
        let z_crit = normal.inverse_cdf(1.0 - alpha / 2.0);

        let k_constant = usize::from(self.input.has_intercept());
        let mut param_names = Vec::with_capacity(k - k_constant);
        let mut out_dydx = Vec::with_capacity(k - k_constant);
        let mut std_errors = Vec::with_capacity(k - k_constant);
        let mut z_stats = Vec::with_capacity(k - k_constant);
        let mut p_values = Vec::with_capacity(k - k_constant);
        let mut conf_lower = Vec::with_capacity(k - k_constant);
        let mut conf_upper = Vec::with_capacity(k - k_constant);

        for (j, &dydx_j) in dydx.iter().enumerate().skip(k_constant) {
            let jac_row: Vec<f64> = (0..k).map(|m| *jacobian.get(j, m)).collect();
            let mut var_j = 0.0;
            for a in 0..k {
                for b in 0..k {
                    var_j += jac_row[a] * (*self.cov_params.get(a, b)) * jac_row[b];
                }
            }
            let se = var_j.sqrt();
            let z = dydx_j / se;

            param_names.push(self.input.param_names()[j].clone());
            out_dydx.push(dydx_j);
            std_errors.push(se);
            z_stats.push(z);
            p_values.push(2.0 * (1.0 - normal.cdf(z.abs())));
            conf_lower.push(dydx_j - z_crit * se);
            conf_upper.push(dydx_j + z_crit * se);
        }

        Ok(MarginalEffects {
            param_names,
            dydx: out_dydx,
            std_errors,
            z_stats,
            p_values,
            conf_lower,
            conf_upper,
        })
    }

    /// 予測確率 `p_i = Λ(x_i'θ)` を、`fit()`に使った学習データ（`self.input.x()`）の
    /// 各行について返す（`fit()`のReturn本体には含めない別メソッド、
    /// `nonlinear-api-design.md`6章）。
    ///
    /// **新規データでの予測（out-of-sample）は未対応**（本Issueのスコープ外、
    /// 別issueでトラッキング。ユーザー確認済み）。
    pub fn predict(&self) -> Vec<f64> {
        let x = self.input.x();
        let n = x.nrows();
        let k = x.ncols();
        (0..n)
            .map(|i| {
                let z: f64 = (0..k).map(|j| *x.get(i, j) * self.params[j]).sum();
                logistic(z)
            })
            .collect()
    }

    /// 分類の的中表（2×2、`table[actual][predicted]`のカウント。行=実測クラス、
    /// 列=予測クラス）。`predict()`が返す予測確率のみを`threshold`で二値化し
    /// （`predicted = 1 if p > threshold else 0`）、実測`y`は`threshold`に関わらず
    /// 常に`0.5`で二値化する（`actual = 1 if y >= 0.5 else 0`）。
    ///
    /// statsmodelsの`pred_table(threshold)`（`BinaryResults.pred_table`）の実装を
    /// 数値照合の上で確認した挙動: `pred = (self.predict() > threshold)`で
    /// 予測確率のみを`threshold`で二値化した**後**、`histogram2d(actual, pred,
    /// bins=[0, 0.5, 1])`で固定の0.5分割によりクロス集計する。`actual`（生の`endog`）は
    /// この固定分割にしか通らず、`threshold`の影響を受けない（rust-reviewerの
    /// 指摘・Python数値照合で発覚した実装ミスを修正: 初版では`actual`も`threshold`で
    /// 二値化していたため、`threshold≠0.5`のときstatsmodelsと一致しなかった）。
    /// `y`が厳密に0/1でない場合（現状値域検証は未実装、`nonlinear-implementation-
    /// notes.md`参照）も、常に`0.5`分割になる点でstatsmodelsと同じ扱い。
    ///
    /// `threshold`の値域は検証しない（`[0,1]`の範囲外でも、`predicted`が単に自明な
    /// 分類結果（全て一方のクラスに分類される等）になるだけで計算上は破綻しない。
    /// statsmodelsも検証していない）。
    ///
    /// **新規データでの的中表（out-of-sample）は未対応**（本Issueのスコープ外、
    /// 別issueでトラッキング。ユーザー確認済み）。
    pub fn pred_table(&self, threshold: f64) -> Mat<f64> {
        let predicted = self.predict();
        let y = self.input.y();
        let n = y.nrows();

        let mut table = Mat::zeros(2, 2);
        for (i, &p_i) in predicted.iter().enumerate().take(n) {
            let actual = usize::from(*y.get(i, 0) >= 0.5);
            let pred = usize::from(p_i > threshold);
            *table.get_mut(actual, pred) += 1.0;
        }
        table
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
        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

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
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
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

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
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

    /// 切片のみモデルは`log_likelihood`と`log_likelihood_null`が定義上一致する
    /// （どちらも同じ「切片のみ」モデルを参照するため）。この性質を使い、
    /// `df_model=0`分岐（`lr_p_value=NaN`、OLSの`f_p_value`と同じ扱い）を含めた
    /// 適合度統計量一式を検証する。
    #[test]
    fn fit_computes_goodness_of_fit_statistics_for_intercept_only_model() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let n: f64 = 7.0;
        let y_bar: f64 = 4.0 / 7.0;
        // 切片のみモデルの対数尤度の閉じた形: n1*ln(ȳ) + n0*ln(1-ȳ)（n1=4, n0=3）
        let expected_ll = 4.0 * y_bar.ln() + 3.0 * (1.0 - y_bar).ln();

        assert!((estimator.log_likelihood() - expected_ll).abs() < 1e-6);
        assert!((estimator.log_likelihood_null() - expected_ll).abs() < 1e-9);
        assert!(estimator.lr_statistic().abs() < 1e-6);
        assert!(estimator.pseudo_r_squared().abs() < 1e-6);
        assert_eq!(estimator.df_model(), 0);
        assert!(estimator.lr_p_value().is_nan());
        assert_eq!(estimator.n_obs(), 7);
        assert_eq!(estimator.df_resid(), 6);

        let expected_aic = -2.0 * expected_ll + 2.0;
        let expected_bic = -2.0 * expected_ll + n.ln();
        assert!((estimator.aic() - expected_aic).abs() < 1e-6);
        assert!((estimator.bic() - expected_bic).abs() < 1e-6);
    }

    /// `log_likelihood_null`は`n1`（y=1の観測数）または`n0`（y=0の観測数）が0の
    /// 退化ケース（全観測が同じ値）で、`0*ln(0)`によるNaNを避ける分岐（`if n1 > 0.0
    /// {...} else {0.0}`等）を通る。この分岐は`fit()`経由（反復最適化）だと、全観測が
    /// 同じyの場合に完全分離が起きて収束の挙動が不安定になりうるため、`fit()`を
    /// 経由せず`log_likelihood_null`を直接呼んで検証する
    /// （`.claude/rules/rust-style.md`「テスト」のカバレッジ方針、Issue #64でこの分岐が
    /// 未カバーだったことが判明し、`fit()`内にインラインだった計算を独立関数に切り出した
    /// 上で追加したテスト）。
    #[test]
    fn log_likelihood_null_returns_zero_for_degenerate_all_same_y() {
        // 全観測y=1（n0=0）: n0側の`else{0.0}`分岐を通る。ℓ_null = n*ln(1) + 0 = 0
        let all_ones = Mat::from_fn(4, 1, |_, _| 1.0);
        assert!((log_likelihood_null(&all_ones) - 0.0).abs() < 1e-12);

        // 全観測y=0（n1=0）: n1側の`else{0.0}`分岐を通る。ℓ_null = 0 + n*ln(1) = 0
        let all_zeros = Mat::from_fn(4, 1, |_, _| 0.0);
        assert!((log_likelihood_null(&all_zeros) - 0.0).abs() < 1e-12);
    }

    /// 多変量（k=3）モデルでの適合度統計量を、実装（`softplus`ベース）とは異なる式
    /// （`logistic`から直接`Σ[y ln(p) + (1-y) ln(1-p)]`を計算するベルヌーイ対数尤度の
    /// 定義式そのもの）で独立に再計算し、突き合わせる。`fit_cov_params_is_symmetric_
    /// and_stats_are_internally_consistent`と同じデータセットを再利用する。
    #[test]
    fn fit_computes_goodness_of_fit_statistics_matching_independently_recomputed_values() {
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

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let n = 4usize;
        let k = 3usize;
        let x = estimator.input().x();
        let params = estimator.params();
        let expected_ll: f64 = (0..n)
            .map(|i| {
                let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                let p = logistic(z);
                y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln()
            })
            .sum();
        assert!((estimator.log_likelihood() - expected_ll).abs() < 1e-9);

        let y_bar: f64 = 2.0 / 4.0; // n1=2, n0=2
        let expected_llnull = 2.0 * y_bar.ln() + 2.0 * (1.0 - y_bar).ln();
        assert!((estimator.log_likelihood_null() - expected_llnull).abs() < 1e-9);

        let expected_lr = 2.0 * (expected_ll - expected_llnull);
        assert!((estimator.lr_statistic() - expected_lr).abs() < 1e-9);

        let expected_pseudo_r2 = 1.0 - expected_ll / expected_llnull;
        assert!((estimator.pseudo_r_squared() - expected_pseudo_r2).abs() < 1e-9);

        assert_eq!(estimator.df_model(), k - 1);
        assert_eq!(estimator.df_resid(), n - k);
        assert_eq!(estimator.n_obs(), n);

        let chi2 = ChiSquared::new((k - 1) as f64).unwrap();
        let expected_lr_p = 1.0 - chi2.cdf(expected_lr);
        assert!((estimator.lr_p_value() - expected_lr_p).abs() < 1e-9);

        let expected_aic = -2.0 * expected_ll + 2.0 * (k as f64);
        let expected_bic = -2.0 * expected_ll + (n as f64).ln() * (k as f64);
        assert!((estimator.aic() - expected_aic).abs() < 1e-9);
        assert!((estimator.bic() - expected_bic).abs() < 1e-9);
    }

    /// `include_intercept=false`のとき、`log_likelihood_null`が参照する「切片のみ」
    /// モデルはフィット対象のモデルの部分集合（入れ子）にならない
    /// （`LogitEstimator`の`log_likelihood_null`フィールドdocコメント参照）。
    /// この場合`lr_statistic`が負になりうる（statsmodels準拠の仕様上の挙動、
    /// rust-reviewer指摘・実測で確認済み）ことを回帰テストとして固定する。
    /// `df_model`/`df_resid`/`aic`/`bic`は`include_intercept`の値に関わらず
    /// 同じ式（`k-1`/`n-k`/`-2ℓ+2k`/`-2ℓ+ln(n)k`）で計算されることも確認する。
    #[test]
    fn fit_lr_statistic_can_be_negative_when_include_intercept_is_false() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let n = 4usize;
        let k = 2usize; // 切片なし、x1・x2のみ
        assert_eq!(estimator.df_model(), k - 1);
        assert_eq!(estimator.df_resid(), n - k);

        let expected_aic = -2.0 * estimator.log_likelihood() + 2.0 * (k as f64);
        let expected_bic = -2.0 * estimator.log_likelihood() + (n as f64).ln() * (k as f64);
        assert!((estimator.aic() - expected_aic).abs() < 1e-9);
        assert!((estimator.bic() - expected_bic).abs() < 1e-9);

        // 非入れ子のため`lr_statistic`が負になりうる（NaN/Infにはならない）。
        assert!(estimator.lr_statistic().is_finite());
        assert!(estimator.lr_p_value().is_finite());
    }

    /// `cov_type=Opg`/`Hc0`/`Hc1`が返す`cov_params`を、`fit()`と同じ手順
    /// （標準化→収束点でのscores/Hessian評価→`common.rs`の共通行列演算→
    /// `destandardize_cov_params`）をテスト側で独立に再現した値と突き合わせる。
    /// 多変量（k=3）データセットを使う理由: 切片のみモデルでは情報行列の等式
    /// （`Σᵢsᵢsᵢ' = -H`）が有限標本で厳密に成り立ってしまい、`classical`/`opg`/`hc0`
    /// が偶然同じ値になるため、`fit()`の`match cov_type`の配線ミス（例えば`Opg`の
    /// 枝で誤って`observed_information_cov_params`を呼んでいた場合等）を検出できない
    /// （`fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ
    /// データセットを再利用する）。
    #[test]
    fn fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 4;

        let classical = LogitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        // `fit()`と同じ手順を独立に再現する: 標準化→θ_stdへ変換→収束点でのscores/
        // Hessian評価→destandardize_cov_params。`LogitProblem::hessian`（argminトレイト）は
        // コスト関数（負の対数尤度）のHessianを返す符号規約のため、`run_solver`が
        // `SolverOutput.hessian`に格納する対数尤度そのもののHessianに合わせて1回符号反転する
        // （`run_solver`のdocコメント「Hessianトレイトの符号規約」と同じ変換）。
        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std: Vec<f64> = classical
            .params()
            .iter()
            .zip(scale.stds())
            .map(|(p, s)| p * s)
            .collect();
        let problem_std = LogitProblem {
            x: x_std,
            y: input_for_reconstruction.y().clone(),
        };
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_opg =
            destandardize_cov_params(&opg_cov_params(&scores_std, k).unwrap(), &scale);
        let expected_hc0 = destandardize_cov_params(
            &sandwich_cov_params(&hessian_std, &scores_std, n, k, SandwichVariant::Hc0).unwrap(),
            &scale,
        );
        let expected_hc1 = destandardize_cov_params(
            &sandwich_cov_params(&hessian_std, &scores_std, n, k, SandwichVariant::Hc1).unwrap(),
            &scale,
        );

        let cases = [
            (CovType::Opg, &expected_opg),
            (CovType::Hc0, &expected_hc0),
            (CovType::Hc1, &expected_hc1),
        ];
        for (cov_type, expected) in cases {
            let estimator = LogitEstimator::fit(
                make_input(),
                Method::Newton,
                35,
                1e-8,
                true,
                cov_type.clone(),
                0.95,
            )
            .unwrap();
            for i in 0..k {
                for j in 0..k {
                    assert!(
                        (*estimator.cov_params().get(i, j) - *expected.get(i, j)).abs() < 1e-8,
                        "cov_type={:?}, ({i},{j}): actual={}, expected={}",
                        cov_type,
                        *estimator.cov_params().get(i, j),
                        *expected.get(i, j)
                    );
                }
            }
        }
    }

    /// `cov_type=Cluster`が返す`cov_params`を、`fit()`と同じ手順をテスト側で独立に
    /// 再現した値と突き合わせる（上の`fit_cov_type_opg_hc0_hc1_match_independently_
    /// recomputed_values`と同じ技法・同じ多変量データセット。情報行列の等式が
    /// 厳密に成り立つ切片のみモデルでは配線ミスを検出できないため）。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 4;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        let classical = LogitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std: Vec<f64> = classical
            .params()
            .iter()
            .zip(scale.stds())
            .map(|(p, s)| p * s)
            .collect();
        let problem_std = LogitProblem {
            x: x_std,
            y: input_for_reconstruction.y().clone(),
        };
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_cluster = destandardize_cov_params(
            &cluster_cov_params(&hessian_std, &scores_std, n, k, &groups).unwrap(),
            &scale,
        );

        let estimator = LogitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        )
        .unwrap();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *expected_cluster.get(i, j)).abs() < 1e-8,
                    "({i},{j}): actual={}, expected={}",
                    *estimator.cov_params().get(i, j),
                    *expected_cluster.get(i, j)
                );
            }
        }
    }

    /// 上のテストは2:2の均等サイズのグループのみを検証しているが、
    /// `testing-policy.md`が指摘する通り均等サイズのみのテストは実務で起こりやすい
    /// 偏った分布のグループサイズ（クラスター内の観測数がクラスターごとに異なる場合）
    /// を見逃しうる。OLS側の対応するテスト（`fit_computes_cluster_std_errors_t_stats_
    /// p_values_conf_int_and_f_test`、2:3の不均衡）に倣い、3:2の不均衡なグループでも
    /// 同じ独立再計算の技法で検証する。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups() {
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let x_columns = vec![
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![-5.0, 2.0, 8.0, -1.0, 3.0],
        ];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 5;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        let classical = LogitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std: Vec<f64> = classical
            .params()
            .iter()
            .zip(scale.stds())
            .map(|(p, s)| p * s)
            .collect();
        let problem_std = LogitProblem {
            x: x_std,
            y: input_for_reconstruction.y().clone(),
        };
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_cluster = destandardize_cov_params(
            &cluster_cov_params(&hessian_std, &scores_std, n, k, &groups).unwrap(),
            &scale,
        );

        let estimator = LogitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        )
        .unwrap();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *expected_cluster.get(i, j)).abs() < 1e-8,
                    "({i},{j}): actual={}, expected={}",
                    *estimator.cov_params().get(i, j),
                    *expected_cluster.get(i, j)
                );
            }
        }
    }

    #[test]
    fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Cluster { groups: None },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::MissingClusterColumn)
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let groups = vec!["a".to_string(); 4];

        let result = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    /// `method`（`bfgs`/`lbfgs`）と`cov_type`（`Opg`/`Hc0`/`Hc1`/`Cluster`）の組み合わせが
    /// 正しく機能することを確認する（rust-reviewer指摘: 既存テストは`method`横断が
    /// `CovType::Classical`のみ、`cov_type`横断が`Method::Newton`のみで、両方を
    /// 同時に変える組み合わせが未検証だった）。`scores_std`の評価は収束点の
    /// パラメータにのみ依存し最適化アルゴリズムの種類に依存しない設計のため、
    /// `newton`で計算した`cov_params`（既に上のテストで正しさを検証済み）と
    /// `bfgs`/`lbfgs`の結果が一致するはず。
    #[test]
    fn fit_non_classical_cov_types_work_with_bfgs_and_lbfgs() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        for cov_type in [
            CovType::Opg,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Cluster {
                groups: Some(groups),
            },
        ] {
            let newton = LogitEstimator::fit(
                make_input(),
                Method::Newton,
                35,
                1e-8,
                true,
                cov_type.clone(),
                0.95,
            )
            .unwrap();

            for method in [Method::Bfgs, Method::Lbfgs] {
                let estimator = LogitEstimator::fit(
                    make_input(),
                    method,
                    200,
                    1e-8,
                    true,
                    cov_type.clone(),
                    0.95,
                )
                .unwrap();

                assert!(
                    estimator.converged(),
                    "cov_type={:?}, {:?}",
                    cov_type,
                    method
                );
                for i in 0..k {
                    for j in 0..k {
                        assert!(
                            (*estimator.cov_params().get(i, j) - *newton.cov_params().get(i, j))
                                .abs()
                                < 1e-4,
                            "cov_type={:?}, method={:?}, ({i},{j}): actual={}, newton={}",
                            cov_type,
                            method,
                            *estimator.cov_params().get(i, j),
                            *newton.cov_params().get(i, j)
                        );
                    }
                }
            }
        }
    }

    /// `newton`と同じデータセット（既知の解析解を持つ切片のみモデル）で`bfgs`/`lbfgs`を
    /// 実行し、いずれも同じ解析解へ収束することを検証する（Issue #57完了条件）。
    #[test]
    fn fit_bfgs_and_lbfgs_converge_to_same_solution_as_newton() {
        let y_bar: f64 = 4.0 / 7.0;
        let expected = (y_bar / (1.0 - y_bar)).ln();

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator = LogitEstimator::fit(
                intercept_only_input(),
                method,
                100,
                1e-6,
                true,
                CovType::Classical,
                0.95,
            )
            .unwrap();

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

        let newton = LogitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(newton.converged());

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator = LogitEstimator::fit(
                make_input(),
                method,
                200,
                1e-8,
                true,
                CovType::Classical,
                0.95,
            )
            .unwrap();

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
        let result = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            1.5,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    #[test]
    fn fit_returns_invalid_max_iter_error_for_non_positive_max_iter() {
        let result = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            0,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        );
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

        let result = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        );
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

        let result = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        );
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

            let result =
                LogitEstimator::fit(input, method, 100, 1e-6, true, CovType::Classical, 0.95);
            assert!(
                matches!(result, Err(MleError::SingularHessian)),
                "method={:?}, result={:?}",
                method,
                result
            );
        }
    }

    /// `sandwich_cov_params`（`cov_type=Hc0`/`Hc1`）も内部で`neg_hessian_inverse`を
    /// 呼ぶため、`Classical`と同じ完全な多重共線性のデータセットで`SingularHessian`に
    /// なるはずだが、Issue #64（カバレッジ確認）時点ではこの伝播経路
    /// （`fit()`の`CovType::Hc0`/`Hc1`分岐の`?`）を通るテストが無かった
    /// （`cargo-llvm-cov`で判明）。`Opg`/`Cluster`分岐は既存の`fit_cov_type_*`系
    /// テストが特異でないデータセットでの成功パスのみ検証しているのと対照的に、
    /// ここでは特異データセットでのエラー伝播を検証する。
    ///
    /// `method=Newton`は使わない: `newton_step`内の特異性検出（ピボット付きQR）が
    /// `cov_type`の分岐に到達する前（最適化中）に`SingularHessian`を返してしまうため
    /// （`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_
    /// with_bfgs_and_lbfgs`のdocコメントと同じ理由。当初`Method::Newton`で書いていて
    /// この経路を実際には通れていなかったことが`cargo-llvm-cov`の再計測で発覚し、
    /// `Method::Bfgs`に修正した）。
    #[test]
    fn fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_hc0_and_hc1() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];

        for cov_type in [CovType::Hc0, CovType::Hc1] {
            let input = LogitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result =
                LogitEstimator::fit(input, Method::Bfgs, 100, 1e-6, true, cov_type.clone(), 0.95);
            assert!(
                matches!(result, Err(MleError::SingularHessian)),
                "cov_type={:?}, result={:?}",
                cov_type,
                result
            );
        }
    }

    /// `cov_type=Opg`のエラー伝播（`opg_cov_params`が返す`SingularOpgMatrix`。
    /// `SingularHessian`とは別のエラー型、`common.rs`「OPG行列特異時のエラー型を分離」
    /// 参照）も、Hc0/Hc1と同じ完全な多重共線性データセットで検証する。
    /// `scores_i=(y_i-p_i)x_i`かつ`x2=2*x1`のため、スコア行列も`x1`と同じ構造的な
    /// 多重共線性を持ち（列2=2×列1）、OPG行列`Σsᵢsᵢ'`も特異になる。rust-reviewerの
    /// 指摘（Hc0/Hc1の修正時、同種のギャップがOpg/Clusterにも残っていることが
    /// `cargo-llvm-cov`のHTMLレポートで判明）を受けて追加。
    #[test]
    fn fit_returns_singular_opg_matrix_error_for_perfectly_collinear_design_matrix() {
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

        let result = LogitEstimator::fit(input, Method::Bfgs, 100, 1e-6, true, CovType::Opg, 0.95);
        assert!(
            matches!(result, Err(MleError::SingularOpgMatrix)),
            "{:?}",
            result
        );
    }

    /// `cov_type=Cluster`のエラー伝播（`cluster_cov_params`も内部で`neg_hessian_inverse`を
    /// 呼ぶため`SingularHessian`）も、Hc0/Hc1と同じ完全な多重共線性データセットで検証する
    /// （rust-reviewerの指摘、上記2テストと同じ経緯）。
    #[test]
    fn fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_cluster() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];
        let groups = vec![
            "g1".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g2".to_string(),
        ];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = LogitEstimator::fit(
            input,
            Method::Bfgs,
            100,
            1e-6,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    #[test]
    fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
        let result = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            1,
            1e-12,
            true,
            CovType::Classical,
            0.95,
        );
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
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(!estimator.converged());
    }

    /// `at="overall"`（AME）のヤコビアン（`dydx_and_jacobian`が`overall_w_and_s`の
    /// 出力から計算する`∂dydx_j/∂θ_m`）を、`overall_w_and_s`が返す`w`を`θ`の関数として
    /// 中心差分で数値微分した値と比較する。`hessian_matches_numerical_differentiation_
    /// of_gradient`と同じ技法（解析解が閉じた形で書けない一般の点で、実装から独立に
    /// 検証する）。
    #[test]
    fn dydx_and_jacobian_matches_numerical_differentiation_for_overall_w_and_s() {
        let x = Mat::from_fn(4, 3, |i, j| match j {
            0 => 1.0,
            1 => [10.0, 20.0, 30.0, 40.0][i],
            _ => [-5.0, 2.0, 8.0, -1.0][i],
        });
        let params = vec![0.3, -0.02, 0.05];
        let k = 3;
        let h = 1e-6;

        let (w, s) = overall_w_and_s(&x, &params);
        let (_, jacobian) = dydx_and_jacobian(k, &params, w, &s);

        for j in 0..k {
            for m in 0..k {
                let mut plus = params.clone();
                plus[m] += h;
                let mut minus = params.clone();
                minus[m] -= h;
                let (w_plus, _) = overall_w_and_s(&x, &plus);
                let (w_minus, _) = overall_w_and_s(&x, &minus);
                let dydx_plus_j = w_plus * plus[j];
                let dydx_minus_j = w_minus * minus[j];
                let numeric = (dydx_plus_j - dydx_minus_j) / (2.0 * h);
                assert!(
                    (*jacobian.get(j, m) - numeric).abs() < 1e-6,
                    "j={j}, m={m}, analytic={}, numeric={numeric}",
                    *jacobian.get(j, m)
                );
            }
        }
    }

    /// `at="mean"`/`"median"`（`at_point_w_and_s`、代表点`x_bar`固定）のヤコビアンについても、
    /// 上記と同じ数値微分による独立検証を行う。
    #[test]
    fn dydx_and_jacobian_matches_numerical_differentiation_for_at_point_w_and_s() {
        let x_bar = vec![1.0, 25.0, 1.0];
        let params = vec![0.3, -0.02, 0.05];
        let k = 3;
        let h = 1e-6;

        let (w, s) = at_point_w_and_s(&x_bar, &params);
        let (_, jacobian) = dydx_and_jacobian(k, &params, w, &s);

        for j in 0..k {
            for m in 0..k {
                let mut plus = params.clone();
                plus[m] += h;
                let mut minus = params.clone();
                minus[m] -= h;
                let (w_plus, _) = at_point_w_and_s(&x_bar, &plus);
                let (w_minus, _) = at_point_w_and_s(&x_bar, &minus);
                let dydx_plus_j = w_plus * plus[j];
                let dydx_minus_j = w_minus * minus[j];
                let numeric = (dydx_plus_j - dydx_minus_j) / (2.0 * h);
                assert!(
                    (*jacobian.get(j, m) - numeric).abs() < 1e-6,
                    "j={j}, m={m}, analytic={}, numeric={numeric}",
                    *jacobian.get(j, m)
                );
            }
        }
    }

    /// 切片のみモデル（k=1、`k_constant=1`）は、限界効果の出力対象となる説明変数が
    /// 存在しない（定数項は出力から除外するため）。`marginal_effects`が空の結果を
    /// 返す（パニックしない）ことを確認する境界ケース。
    #[test]
    fn marginal_effects_returns_empty_result_for_intercept_only_model() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let effects = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        assert!(effects.param_names().is_empty());
        assert!(effects.dydx().is_empty());
    }

    /// `marginal_effects(at="overall")`の`dydx`を、実装の内部ヘルパー（`overall_w_and_s`/
    /// `dydx_and_jacobian`）とは別に、定義式`dy/dx_j = (1/n)Σᵢpᵢ(1-pᵢ)θⱼ`を`logistic`
    /// から直接計算する式で独立に再計算し、突き合わせる。標準誤差は、デルタ法の
    /// ヤコビアンを`overall_w_and_s`経由ではなく数値微分（`dydx_j`自体をfit済みパラメータ
    /// の周りで直接数値微分したもの）で独立に求め、`marginal_effects`が返す解析的な
    /// 標準誤差と突き合わせることで、`dydx_and_jacobian`内の式（`θⱼ*s_m + [j==m]*w`）が
    /// 正しく実装に配線されていることを検証する（配線ミスがあればこのテストで検出できる）。
    #[test]
    fn marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_method_se() {
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

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let k = 3;
        let n = 4;
        let x = estimator.input().x();

        let effects = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        assert_eq!(effects.param_names(), ["x1".to_string(), "x2".to_string()]);

        // dydxの独立再計算（`logistic`から直接、`overall_w_and_s`とは別の式）
        let dydx_j = |params: &[f64], j: usize| -> f64 {
            (0..n)
                .map(|i| {
                    let z: f64 = (0..k).map(|m| *x.get(i, m) * params[m]).sum();
                    let p = logistic(z);
                    p * (1.0 - p)
                })
                .sum::<f64>()
                / (n as f64)
                * params[j]
        };
        let params = estimator.params();
        for (idx, j) in (1..k).enumerate() {
            assert!((effects.dydx()[idx] - dydx_j(params, j)).abs() < 1e-9);
        }

        // 標準誤差の独立検証: `dydx_j`をfit済みパラメータの周りで数値微分して
        // ヤコビアン行（j行目）を求め、`cov_params`との二次形式で分散を計算する。
        let h = 1e-6;
        for (idx, j) in (1..k).enumerate() {
            let mut jac_row = vec![0.0; k];
            for m in 0..k {
                let mut plus = params.to_vec();
                plus[m] += h;
                let mut minus = params.to_vec();
                minus[m] -= h;
                jac_row[m] = (dydx_j(&plus, j) - dydx_j(&minus, j)) / (2.0 * h);
            }
            let mut var_j = 0.0;
            for a in 0..k {
                for b in 0..k {
                    var_j += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
                }
            }
            let expected_se = var_j.sqrt();
            assert!(
                (effects.std_errors()[idx] - expected_se).abs() < 1e-6,
                "idx={idx}, actual={}, expected={expected_se}",
                effects.std_errors()[idx]
            );
        }

        // z値・p値・信頼区間の内部整合性（既存のSE系テストと同じ検算パターン）
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal.inverse_cdf(0.975);
        for idx in 0..2 {
            let se = effects.std_errors()[idx];
            assert!((effects.z_stats()[idx] - effects.dydx()[idx] / se).abs() < 1e-9);
            let expected_p = 2.0 * (1.0 - normal.cdf(effects.z_stats()[idx].abs()));
            assert!((effects.p_values()[idx] - expected_p).abs() < 1e-9);
            assert!(
                (effects.conf_upper()[idx] - effects.conf_lower()[idx] - 2.0 * z_crit * se).abs()
                    < 1e-9
            );
        }
    }

    /// `at="mean"`は`at="overall"`と異なる代表点（標本平均）で評価するため、一般には
    /// 異なる値になる。実装がこの違いを正しく反映していること（`at`の分岐が機能して
    /// いること）を確認する。`dydx`を`column_means`から独立に再計算した値とも突き合わせる。
    #[test]
    fn marginal_effects_at_mean_differs_from_overall_and_matches_independent_recomputation() {
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

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let overall = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        let at_mean = estimator
            .marginal_effects(MarginalEffectsAt::Mean, 0.95)
            .unwrap();

        assert!((overall.dydx()[0] - at_mean.dydx()[0]).abs() > 1e-9);

        // 独立再計算: x̄=[1, 25, 1]（定数項1、x1の平均25、x2の平均1）でp̄を評価
        let params = estimator.params();
        let x_bar = [1.0, 25.0, 1.0];
        let z_bar: f64 = (0..3).map(|m| x_bar[m] * params[m]).sum();
        let p_bar = logistic(z_bar);
        let w = p_bar * (1.0 - p_bar);
        for (idx, j) in (1..3).enumerate() {
            assert!((at_mean.dydx()[idx] - w * params[j]).abs() < 1e-9);
        }
    }

    /// `column_medians`（`marginal_effects`の`at="median"`が使う代表点の計算）を、
    /// 奇数・偶数それぞれの観測数で直接検証する。
    #[test]
    fn column_medians_matches_expected_for_odd_and_even_n() {
        // 奇数（n=5）: ソート後の中央1点がそのまま中央値
        let x_odd = Mat::from_fn(5, 2, |i, j| match j {
            0 => [10.0, 20.0, 30.0, 40.0, 100.0][i],
            _ => [-5.0, 2.0, 8.0, -1.0, 50.0][i],
        });
        let medians_odd = column_medians(&x_odd);
        assert!((medians_odd[0] - 30.0).abs() < 1e-12); // sorted: 10,20,30,40,100
        assert!((medians_odd[1] - 2.0).abs() < 1e-12); // sorted: -5,-1,2,8,50

        // 偶数（n=4）: 中央2値の平均
        let x_even = Mat::from_fn(4, 1, |i, _| [10.0, 20.0, 30.0, 40.0][i]);
        let medians_even = column_medians(&x_even);
        assert!((medians_even[0] - 25.0).abs() < 1e-12); // (20+30)/2
    }

    /// `at="median"`が`at="mean"`/`at="overall"`と異なる代表点で評価されること
    /// （非対称なデータセットで平均・中央値が異なる値になるよう構成）、および
    /// `dydx`を`column_medians`から独立に再計算した値と突き合わせる。
    #[test]
    fn marginal_effects_at_median_differs_from_mean_and_overall_and_matches_independent_recomputation()
     {
        // x1: 平均40（=200/5）・中央値30（sorted: 10,20,30,40,100）と非対称にずらす
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let x_columns = vec![
            vec![10.0, 20.0, 30.0, 40.0, 100.0],
            vec![-5.0, 2.0, 8.0, -1.0, 50.0],
        ];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let overall = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        let at_mean = estimator
            .marginal_effects(MarginalEffectsAt::Mean, 0.95)
            .unwrap();
        let at_median = estimator
            .marginal_effects(MarginalEffectsAt::Median, 0.95)
            .unwrap();

        assert!((at_median.dydx()[0] - at_mean.dydx()[0]).abs() > 1e-9);
        assert!((at_median.dydx()[0] - overall.dydx()[0]).abs() > 1e-9);

        // 独立再計算: x̄=[1, 30, 2]（定数項1、x1の中央値30、x2の中央値2）でp̄を評価
        let params = estimator.params();
        let x_bar = [1.0, 30.0, 2.0];
        let z_bar: f64 = (0..3).map(|m| x_bar[m] * params[m]).sum();
        let p_bar = logistic(z_bar);
        let w = p_bar * (1.0 - p_bar);
        for (idx, j) in (1..3).enumerate() {
            assert!((at_median.dydx()[idx] - w * params[j]).abs() < 1e-9);
        }
    }

    #[test]
    fn marginal_effects_returns_invalid_confidence_level_error_out_of_range() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let result = estimator.marginal_effects(MarginalEffectsAt::Overall, 1.5);
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    /// 切片のみモデルは全観測で`p_i=ȳ`（closed form、他のテストと同じ性質）なので、
    /// `predict()`が返す予測確率が`ȳ`と一致することを検証できる。
    #[test]
    fn predict_matches_closed_form_for_intercept_only_model() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let y_bar: f64 = 4.0 / 7.0;
        let predicted = estimator.predict();
        assert_eq!(predicted.len(), 7);
        for p in predicted {
            assert!((p - y_bar).abs() < 1e-6);
        }
    }

    /// 多変量モデルでは`predict()`に閉じた形の解析解が無いため、`logistic`から直接
    /// `p_i=Λ(x_i'θ)`を計算する式で独立に再計算し、突き合わせる。
    #[test]
    fn predict_matches_independently_recomputed_logistic_of_linear_predictor() {
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

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let params = estimator.params();
        let x = estimator.input().x();
        let n = 4;
        let k = 3;
        let predicted = estimator.predict();
        for (i, &p_i) in predicted.iter().enumerate().take(n) {
            let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let expected = 1.0 / (1.0 + (-z).exp());
            assert!((p_i - expected).abs() < 1e-12);
        }
    }

    /// 切片のみモデルは全観測で`p_i=ȳ=4/7≈0.571`（closed form）のため、`threshold`に
    /// よって全観測が一方のクラスに分類される自明なケースになる。この性質を使い、
    /// `pred_table`の的中表を手計算で検証する（`y=[0,0,0,1,1,1,1]`、実測は
    /// `y_i>threshold`で二値化）。
    #[test]
    fn pred_table_matches_hand_computed_counts_for_intercept_only_model() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        // threshold=0.5: p_i=4/7>0.5 なので全観測が予測クラス1。
        // 実測は y=[0,0,0,1,1,1,1] なので actual0が3件・actual1が4件。
        let table_low = estimator.pred_table(0.5);
        assert!((*table_low.get(0, 0) - 0.0).abs() < 1e-12);
        assert!((*table_low.get(0, 1) - 3.0).abs() < 1e-12);
        assert!((*table_low.get(1, 0) - 0.0).abs() < 1e-12);
        assert!((*table_low.get(1, 1) - 4.0).abs() < 1e-12);

        // threshold=0.99: p_i=4/7<0.99 なので全観測が予測クラス0。
        let table_high = estimator.pred_table(0.99);
        assert!((*table_high.get(0, 0) - 3.0).abs() < 1e-12);
        assert!((*table_high.get(0, 1) - 0.0).abs() < 1e-12);
        assert!((*table_high.get(1, 0) - 4.0).abs() < 1e-12);
        assert!((*table_high.get(1, 1) - 0.0).abs() < 1e-12);
    }

    /// `pred_table`が返すカウントの総和が観測数`n`と一致すること、および`predict()`の
    /// 出力から独立に再計算した分類結果（同じ`threshold`での二値化）と一致することを
    /// 多変量モデルで検証する（`pred_table`内部の配線ミスを検出できる設計、
    /// `fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values`と同じ技法）。
    #[test]
    fn pred_table_matches_independently_recomputed_classification() {
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

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        // `threshold≠0.5`にする（`actual`側は`threshold`に依存せず常に0.5固定である
        // ことを検出できるようにするため。`threshold=0.5`固定のテストでは、この
        // 区別がつかない、rust-reviewer指摘）。
        let threshold = 0.2;
        let predicted = estimator.predict();
        let table = estimator.pred_table(threshold);

        let mut expected = [[0.0; 2]; 2];
        for i in 0..4 {
            let actual = usize::from(y[i] >= 0.5);
            let pred = usize::from(predicted[i] > threshold);
            expected[actual][pred] += 1.0;
        }

        let mut total = 0.0;
        for (a, row) in expected.iter().enumerate() {
            for (p, &expected_count) in row.iter().enumerate() {
                assert!((*table.get(a, p) - expected_count).abs() < 1e-12);
                total += *table.get(a, p);
            }
        }
        assert!((total - 4.0).abs() < 1e-12);
    }

    /// `pred_table`の実測クラス（行方向の合計、`actual0`の件数+`actual1`の件数）は
    /// `threshold`の値に関わらず不変であるべき（statsmodelsの`pred_table`が実測`y`を
    /// 常に固定の0.5分割でバケット化し、`threshold`は予測確率側にのみ適用する仕様、
    /// `pred_table`のdocコメント参照）。この不変性が保たれているかを回帰テストとして
    /// 固定する（初版実装は`actual`も`threshold`で二値化しており、`threshold≠0.5`で
    /// この不変性が壊れていた。rust-reviewerの指摘・statsmodelsとの数値照合で発覚し
    /// 修正済み）。
    #[test]
    fn pred_table_actual_class_counts_are_invariant_to_threshold() {
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

        let estimator = LogitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let table = estimator.pred_table(threshold);
            let actual0 = *table.get(0, 0) + *table.get(0, 1);
            let actual1 = *table.get(1, 0) + *table.get(1, 1);
            // y=[0,1,0,1] → actual0=2件・actual1=2件（`threshold`に関わらず常に一定）
            assert!(
                (actual0 - 2.0).abs() < 1e-12,
                "threshold={threshold}, actual0={actual0}"
            );
            assert!(
                (actual1 - 2.0).abs() < 1e-12,
                "threshold={threshold}, actual1={actual1}"
            );
        }
    }
}
