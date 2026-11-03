//! `LogitProblem`（負の対数尤度・スコア・Hessian。argminの`CostFunction`/`Gradient`/`Hessian`の実装）。

#[cfg(test)]
use super::input::LogitInput;
use super::model::{log_likelihood, logistic};
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;

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
    /// `input`の`x`・`y`をそのまま（未標準化のスケールで）複製して構築する。
    /// 閉じた形の解と突き合わせる単体テスト専用（`LogitEstimator::fit`は標準化後の
    /// 設計行列を使うため、こちらではなく`from_standardized`を使う。
    /// 構築経路を2種類の明示的なコンストラクタに統一）。
    #[cfg(test)]
    fn new(input: &LogitInput) -> Self {
        Self {
            x: input.x().clone(),
            y: input.y().clone(),
        }
    }

    /// `standardize_columns`で標準化済みの設計行列`x_std`と`y`から構築する。
    /// `LogitEstimator::fit`が最適化・収束点でのスコア評価に使う経路（
    /// フィールドリテラルでの直接構築を避け、両方の構築経路を明示的なコンストラクタに
    /// 統一する）。
    pub(super) fn from_standardized(x_std: Mat<f64>, y: Mat<f64>) -> Self {
        Self { x: x_std, y }
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
    /// `docs/spec/nonlinear-common.md`3章参照）。
    pub fn scores(&self, params: &[f64]) -> Mat<f64> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        // 行の残差`yᵢ-pᵢ`は`j`によらないため、観測あたり1回だけ計算する
        // （セルごとに計算すると`linear_predictor`のO(k)が行あたりk回になる）。
        let residuals: Vec<f64> = (0..n)
            .map(|i| *self.y.get(i, 0) - logistic(self.linear_predictor(i, params)))
            .collect();
        Mat::from_fn(n, k, |i, j| residuals[i] * (*self.x.get(i, j)))
    }
}

impl CostFunction for LogitProblem {
    type Param = Vec<f64>;
    type Output = f64;

    /// 負の対数尤度 `-ℓ(θ) = Σᵢ [softplus(zᵢ) - yᵢzᵢ]`（`logit.rs`のモジュールdocの数式参照）。
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

#[cfg(test)]
mod tests {

    use super::super::test_support::*;
    use super::*;

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

    /// `scores`は行ごとの残差を観測あたり1回だけ計算して行を埋める。セルごとに再計算する
    /// 素朴な式と、演算の順序まで含めてビット単位で一致することを確認する
    /// （極端なパラメータ点を含む）。
    /// 参照式は本体と同じヘルパーを呼ぶため、検出するのは行列に詰める配置・順序の退行で、
    /// ヘルパー側の式の変更は別のテスト（閉形式・数値微分）が担う。
    #[test]
    fn scores_match_per_cell_reference_bit_for_bit() {
        let input = small_input();
        let problem = LogitProblem::new(&input);
        // `[800.0, -1.0]`は`z≥796`で`logistic`が`1.0`に飽和する（`y-p`が`±0.0`・`-1.0`になる）
        for params in [
            vec![0.0, 0.0],
            vec![0.3, -0.7],
            vec![40.0, -25.0],
            vec![800.0, -1.0],
        ] {
            let scores = problem.scores(&params);
            for i in 0..scores.nrows() {
                for j in 0..scores.ncols() {
                    let p = logistic(problem.linear_predictor(i, &params));
                    let expected = (*problem.y.get(i, 0) - p) * (*problem.x.get(i, j));
                    assert_eq!(scores.get(i, j).to_bits(), expected.to_bits(), "({i}, {j})");
                }
            }
        }
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
}
