//! Probitの入力データ（被説明変数・設計行列）の型定義、および負の対数尤度・スコア・
//! Hessian（argminの`CostFunction`/`Gradient`/`Hessian`トレイト実装）。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `ProbitInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`engine::nonlinear::logit::LogitInput`とほぼ同型の設計
//! （`docs/planning/specs/nonlinear-api-design.md`参照）。
//!
//! OLS/Logitと同様、Phase2（Logit/Probit/Tobit）では`weights`/`offset`を見送っているため
//! （`nonlinear-api-design.md`7章）、`from_columns_weighted`に相当するものはない。
//!
//! ## 数式（プロビット回帰）
//!
//! `z_i = x_i'θ`、`Φ`・`φ`を標準正規分布のCDF・PDFとする。観測`i`の対数尤度への寄与は
//! `ℓ_i(θ) = y_i log Φ(z_i) + (1-y_i) log Φ(-z_i)`（`Φ(-z)=1-Φ(z)`を使用）。
//! `q_i = 2y_i-1 ∈ {-1,+1}`とおくと`ℓ_i(θ) = log Φ(q_i z_i)`という同値な形に書き換えられる
//! （`y_i=1`なら`q_i=1`で`logΦ(z_i)`、`y_i=0`なら`q_i=-1`で`logΦ(-z_i)`に一致）。
//!
//! `λ_i = q_i φ(q_i z_i)/Φ(q_i z_i)`（一般化残差、逆ミルズ比に相当）とおくと:
//!
//! - スコア: `∂ℓ/∂θ = Σᵢ λᵢxᵢ = X'λ`
//! - Hessian: `∂²ℓ/∂θ∂θ' = -Σᵢ λᵢ(λᵢ+zᵢ)xᵢxᵢ' = -X'WX`（`W = diag(λᵢ(λᵢ+zᵢ))`）
//!
//! （導出: `u=q_i z_i`とおき`g(u)=φ(u)/Φ(u)`とすると`λ_i=q_i g(u)`。
//! `g'(u) = -u g(u) - g(u)²`（`φ'(u)=-uφ(u)`より）を使うと
//! `dλ_i/dz_i = q_i² g'(u) = g'(u) = -λ_i(λ_i+z_i)`（`q_i²=1`、`u=q_i z_i`と`g(u)=q_i λ_i`を代入）。
//! Logitの`-X'WX`（`W=diag(pᵢ(1-pᵢ))`）より複雑だが、`λᵢ(λᵢ+zᵢ) > 0`は常に成り立つため
//! （プロビットの対数尤度が大域的に凹であることの根拠）、`X'WX`は常に正定値。）
//!
//! `CostFunction`は`-ℓ(θ)`（argminは最小化フレームワークのため）。`Gradient`/`Hessian`
//! トレイトは`CostFunction`と同じ符号（`-ℓ`の1階・2階微分）で実装する
//! （`run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照）。`scores()`
//! （`cov_type`共通行列演算向け）は符号反転しない生のスコア`sᵢ=λᵢxᵢ`を返す。
//!
//! ## 数値安定化について
//!
//! `Φ(q_i z_i)`・`φ(q_i z_i)`は`statrs::distribution::Normal`（`cdf`/`pdf`）をそのまま使う。
//! `1-Φ(z)`を手動計算せず常に`Φ(q_i z_i)`の形（`q_i`で符号を吸収）で評価するため、
//! `statrs`の`cdf`実装（`erfc`ベース）が両裾で提供する精度をそのまま活かせる
//! （手動で`1.0 - cdf(z)`を計算する場合に生じる桁落ちを避けられる）。
//! ただし、Logitの`log_likelihood`が使う`softplus`のような「対数を経由しても
//! アンダーフローしない」変形は用意していない（`statrs`に`Normal`用の`ln_cdf`が無いため）。
//! `z_i`が極端な値（完全分離に近いデータ等）で`Φ(q_i z_i)`が0にアンダーフローすると
//! `cost`が`+inf`になりうる。Logitの完全分離対応（`nonlinear-implementation-notes.md`
//! 参照、勾配ノルム収束判定のアンダーフロー対応は`LogitEstimator::fit`実装後の
//! 別Issueで対応した経緯がある）と同様、本Issue（#71）のスコープは尤度・スコア・
//! Hessianの数式が閉じた形・数値微分と一致することの検証までとし、極端な入力での
//! 頑健性は`ProbitEstimator::fit`実装以降の別Issueで必要に応じて対応する。

use crate::error::CommonError;
use crate::nonlinear::common::MleError;
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Probitの被説明変数・設計行列を保持する入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
#[derive(Debug)]
pub struct ProbitInput {
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

impl ProbitInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `ProbitInput`を組み立てる。`include_intercept=true`の場合、設計行列の先頭列に
    /// 定数項（すべて1.0）を自動追加する。
    ///
    /// `y`が{0.0, 1.0}の二値であることの検証は、このIssue（#70、次元検証のみがスコープ）では
    /// 行わない。`LogitInput::from_columns`と同じ方針で、尤度・スコア・Hessianを実装する
    /// 後続Issueで`validate_binary_y`（`nonlinear/common.rs`）による検証を追加する予定。
    ///
    /// # Errors
    /// `y`といずれかの`x_columns`の長さが一致しない場合は`CommonError::DimensionMismatch`を返す。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする。これは
    /// 呼び出し側（`engine_pybind`）の実装バグでしか起こり得ない内部契約であり、
    /// 実データに起因する`ValidationError`とは性質が異なるため区別している
    /// （`LogitInput::from_columns`と同じ方針）。
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

/// Probitの負の対数尤度・スコア・Hessian（argminの`CostFunction`/`Gradient`/`Hessian`
/// トレイト実装）。`ProbitInput`の`X`・`y`を保持する（`run_solver`が`problem`の所有権を
/// 必要とするため、`ProbitInput`とは独立した所有データとして持つ。`Clone`は
/// `argmin::core::Executor`が要求する。`LogitProblem`と同じ設計）。
#[derive(Debug, Clone)]
pub struct ProbitProblem {
    x: Mat<f64>,
    y: Mat<f64>,
}

impl ProbitProblem {
    /// `input`の`x`・`y`をそのまま（未標準化のスケールで）複製して構築する。
    /// 閉じた形の解と突き合わせる単体テスト専用（`LogitProblem::new`と同じ位置づけ）。
    #[cfg(test)]
    fn new(input: &ProbitInput) -> Self {
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

    /// 観測`i`の線形予測子`z_i`と一般化残差`λ_i = q_i φ(q_i z_i)/Φ(q_i z_i)`
    /// （`q_i=2y_i-1`、モジュール冒頭の数式参照）をまとめて計算する。`hessian`が
    /// `z_i`・`λ_i`の両方を必要とするため、個別に呼び出すより重複計算を避けられる。
    /// `normal`は呼び出し側（観測`n`件のループ全体）で1回だけ構築して渡す
    /// （`cost`/`gradient`/`hessian`いずれもn回ではなく1回の構築で済ませる）。
    fn linear_predictor_and_residual(
        &self,
        i: usize,
        params: &[f64],
        normal: &Normal,
    ) -> (f64, f64) {
        let z = self.linear_predictor(i, params);
        let q = 2.0 * (*self.y.get(i, 0)) - 1.0;
        let u = q * z;
        let lambda = q * normal.pdf(u) / normal.cdf(u);
        (z, lambda)
    }

    /// 観測ごとのスコア行列（n×k）。各行が`sᵢ = λᵢxᵢ`（対数尤度の1階微分そのもの、
    /// `Gradient`トレイトとは符号が逆）。OPG/サンドイッチ/クラスターSEの計算に使う
    /// （`LogitProblem::scores`と同じ役割）。
    pub fn scores(&self, params: &[f64]) -> Mat<f64> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let normal = Normal::standard();
        Mat::from_fn(n, k, |i, j| {
            let (_, lambda) = self.linear_predictor_and_residual(i, params, &normal);
            lambda * (*self.x.get(i, j))
        })
    }
}

impl CostFunction for ProbitProblem {
    type Param = Vec<f64>;
    type Output = f64;

    /// 負の対数尤度 `-ℓ(θ) = -Σᵢ log Φ(q_i z_i)`（モジュール冒頭の数式参照）。
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        let normal = Normal::standard();
        let n = self.x.nrows();
        let cost: f64 = (0..n)
            .map(|i| {
                let z = self.linear_predictor(i, param);
                let q = 2.0 * (*self.y.get(i, 0)) - 1.0;
                -normal.cdf(q * z).ln()
            })
            .sum();
        Ok(cost)
    }
}

impl Gradient for ProbitProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// `-ℓ(θ)`の勾配 `-Σᵢ λᵢxᵢ = -X'λ`（対数尤度のスコアの符号反転）。
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let normal = Normal::standard();
        let mut grad = vec![0.0; k];
        for i in 0..n {
            let (_, lambda) = self.linear_predictor_and_residual(i, param, &normal);
            for (j, grad_j) in grad.iter_mut().enumerate() {
                *grad_j += -lambda * (*self.x.get(i, j));
            }
        }
        Ok(grad)
    }
}

impl Hessian for ProbitProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// `-ℓ(θ)`のHessian `X'WX`（`W = diag(λᵢ(λᵢ+zᵢ))`、対数尤度のHessian`-X'WX`の
    /// 符号反転）。`run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照。
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let normal = Normal::standard();
        let mut h = vec![vec![0.0; k]; k];
        for i in 0..n {
            let (z, lambda) = self.linear_predictor_and_residual(i, param, &normal);
            let w = lambda * (lambda + z);
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
        let input = ProbitInput::from_columns(
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
        let input = ProbitInput::from_columns(
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
        let result = ProbitInput::from_columns(
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
        // （`LogitInput::from_columns`と同じ方針）。
        let y = vec![0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let _ = ProbitInput::from_columns(&y, &x_columns, vec![], true, "y".to_string());
    }

    /// n=4, k=2（切片+x1）の小規模データ。`θ=[0,0]`のとき`z_i=0`（全観測で共通）となり、
    /// `Φ(0)=0.5`・`φ(0)=1/√(2π)`という既知の値から`cost`/`gradient`/`hessian`が
    /// 閉じた形で手計算できる（`Φ(0)=0.5`はLogitの`logistic(0)=0.5`と同じ値になるため、
    /// `cost`は`LogitInput`の対応するテストと同じ`4*ln(2)`になる）。
    fn small_input() -> ProbitInput {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        ProbitInput::from_columns(
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
        let problem = ProbitProblem::new(&input);
        let params = vec![0.0, 0.0];

        // z_i=0のときΦ(0)=0.5、cost = -Σlog(0.5) = 4*ln(2)
        let cost = problem.cost(&params).unwrap();
        assert!(
            (cost - 4.0 * std::f64::consts::LN_2).abs() < 1e-12,
            "{cost}"
        );

        // λ_i = q_i*φ(0)/Φ(0) = q_i*c（c=φ(0)/0.5=√(2/π)）、y=[0,1,0,1]→q=[-1,1,-1,1]
        // grad(-ℓ) = -Σλ_i*x_i、切片成分: -c*(-1+1-1+1)=0、x1成分: -c*(-1+2-3+4)=-2c
        let c = (2.0 / std::f64::consts::PI).sqrt();
        let grad = problem.gradient(&params).unwrap();
        assert!(grad[0].abs() < 1e-9, "{:?}", grad);
        assert!((grad[1] - (-2.0 * c)).abs() < 1e-9, "{:?}", grad);

        // z_i=0なのでw_i=λ_i(λ_i+0)=λ_i²=c²=2/π（符号によらず全観測共通）
        // Hessian(-ℓ) = (2/π)*X'X、X'X=[[4,10],[10,30]]（n=4,Σx1=10,Σx1²=30、Logitと同じ設計行列）
        let w = c * c;
        let hessian = problem.hessian(&params).unwrap();
        assert!((hessian[0][0] - w * 4.0).abs() < 1e-9, "{:?}", hessian);
        assert!((hessian[0][1] - w * 10.0).abs() < 1e-9, "{:?}", hessian);
        assert!((hessian[1][0] - w * 10.0).abs() < 1e-9, "{:?}", hessian);
        assert!((hessian[1][1] - w * 30.0).abs() < 1e-9, "{:?}", hessian);
    }

    #[test]
    fn scores_match_closed_form_at_zero_params() {
        let input = small_input();
        let problem = ProbitProblem::new(&input);
        let scores = problem.scores(&[0.0, 0.0]);

        // score_i = λ_i*x_i = q_i*c*x_i（c=√(2/π)）、y=[0,1,0,1]→q=[-1,1,-1,1]
        let c = (2.0 / std::f64::consts::PI).sqrt();
        let q = [-1.0, 1.0, -1.0, 1.0];
        let x1 = [1.0, 2.0, 3.0, 4.0];
        for i in 0..4 {
            let lambda = q[i] * c;
            assert!((*scores.get(i, 0) - lambda).abs() < 1e-9, "row {i}");
            assert!((*scores.get(i, 1) - lambda * x1[i]).abs() < 1e-9, "row {i}");
        }
    }

    #[test]
    fn scores_sum_to_negative_gradient() {
        // scoresは対数尤度の生のスコア（符号反転なし）、gradientはCostFunction
        // （負の対数尤度）の勾配のため、観測方向に合計すると符号が逆になるはず。
        let input = small_input();
        let problem = ProbitProblem::new(&input);
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
        let problem = ProbitProblem::new(&input);
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
        let problem = ProbitProblem::new(&input);
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
}
