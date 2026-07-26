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
use crate::nonlinear::common::MleError;
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;

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
}
