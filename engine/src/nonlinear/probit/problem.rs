//! `ProbitProblem`（負の対数尤度・スコア・Hessian。argminの`CostFunction`/`Gradient`/`Hessian`の実装）。

#[cfg(test)]
use super::input::ProbitInput;
use super::model::log_likelihood;
use crate::nonlinear::common::{U_CLAMP, clamped_pdf_cdf, linear_predictor};
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;
use statrs::distribution::Normal;

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

    /// `standardize_columns`で標準化済みの設計行列`x_std`と`y`から構築する。
    /// `ProbitEstimator::fit`が最適化・収束点でのスコア評価に使う経路
    /// （`LogitProblem::from_standardized`と同じ位置づけ）。
    pub(super) fn from_standardized(x_std: Mat<f64>, y: Mat<f64>) -> Self {
        Self { x: x_std, y }
    }

    /// 観測`i`の符号付き線形予測子 `(q_i, u_i)`（`q_i=2y_i-1`、`u_i=q_i z_i`）。
    /// 一般化残差`λ_i`・`hessian`の重みの両方がこの`u_i`から作られる。
    fn signed_index(&self, i: usize, params: &[f64]) -> (f64, f64) {
        let q = 2.0 * (*self.y.get(i, 0)) - 1.0;
        (q, q * linear_predictor(&self.x, i, params))
    }

    /// `signed_index`の`(q, u)`から一般化残差`λ = q φ(u)/Φ(u)`（`probit.rs`のモジュールdocの数式参照）を
    /// 計算する。`u`は`clamped_pdf_cdf`が`[-U_CLAMP, U_CLAMP]`にクランプしてから評価する。
    fn generalized_residual_at(normal: &Normal, q: f64, u: f64) -> f64 {
        let (phi, big_phi) = clamped_pdf_cdf(normal, u);
        q * phi / big_phi
    }

    /// 観測`i`の一般化残差`λ_i = q_i φ(q_i z_i)/Φ(q_i z_i)`（`q_i=2y_i-1`）。
    /// `gradient`/`scores`が使う（`cost`は対数尤度を`log_likelihood`で直接評価する）。
    /// `normal`は呼び出し側（観測`n`件のループ全体）で
    /// 1回だけ構築して渡す。
    fn generalized_residual(&self, i: usize, params: &[f64], normal: &Normal) -> f64 {
        let (q, u) = self.signed_index(i, params);
        Self::generalized_residual_at(normal, q, u)
    }

    /// 観測`i`の`hessian`の重み `w_i = λ_i(λ_i + z̃_i)`。`z̃_i = q_i·clamp(q_i z_i, -U_CLAMP,
    /// U_CLAMP)`は`λ_i`と同じクランプ済み引数から再構成した値で、**生の線形予測子`z_i`では
    /// ない**。`z̃_i`は`w_i`の中でしか意味を持たないため、単独では返さずこの関数の中に閉じている。
    ///
    /// `λ_i`は`clamped_pdf_cdf`で`u=q_i z_i`を`[-U_CLAMP, U_CLAMP]`にクランプした後の値を
    /// 使うため、`|u|>U_CLAMP`の領域では`λ_i`は`z_i`に対して事実上定数になる。この領域で
    /// 生の（非クランプの）`z_i`を`w_i`に混ぜると、`λ_i(λ_i+z_i)>0`という大域凹性の前提
    /// （`probit.rs`のモジュールdocの導出参照）が数学的に破れ、`w_i`が負になりうる（実測: 悪条件
    /// パラメータ点で`w<0`の観測を確認済み）。`λ_i`が実際に依存しているのと同じクランプ済み
    /// 引数から`z̃_i`を再構成することで、`λ_i(λ_i+z̃_i)>0`の恒等式がクランプ領域でも保たれる
    /// （クランプ後の`u`に対する同一の解析式を評価しているにすぎないため）。
    fn hessian_weight(&self, i: usize, params: &[f64], normal: &Normal) -> f64 {
        let (q, u) = self.signed_index(i, params);
        let lambda = Self::generalized_residual_at(normal, q, u);
        let clamped_z = q * u.clamp(-U_CLAMP, U_CLAMP);
        lambda * (lambda + clamped_z)
    }

    /// 観測ごとのスコア行列（n×k）。各行が`sᵢ = λᵢxᵢ`（対数尤度の1階微分そのもの、
    /// `Gradient`トレイトとは符号が逆）。OPG/サンドイッチ/クラスターSEの計算に使う
    /// （`LogitProblem::scores`と同じ役割）。
    pub fn scores(&self, params: &[f64]) -> Mat<f64> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let normal = Normal::standard();
        // `λᵢ`は`j`によらないため、観測あたり1回だけ計算する（セルごとに計算すると
        // `linear_predictor`のO(k)と`φ`/`Φ`の評価が行あたりk回になる）。
        let lambdas: Vec<f64> = (0..n)
            .map(|i| self.generalized_residual(i, params, &normal))
            .collect();
        Mat::from_fn(n, k, |i, j| lambdas[i] * (*self.x.get(i, j)))
    }
}

impl CostFunction for ProbitProblem {
    type Param = Vec<f64>;
    type Output = f64;

    /// 負の対数尤度 `-ℓ(θ) = -Σᵢ log Φ(q_i z_i)`（`probit.rs`のモジュールdocの数式参照）。
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        Ok(-log_likelihood(&self.x, &self.y, param))
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
            let lambda = self.generalized_residual(i, param, &normal);
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
    ///
    /// 重みは`hessian_weight`が返す、`λᵢ`と同じクランプ済み引数から`zᵢ`を
    /// 再構成して作った値（同関数のdocコメント参照）。これにより`λᵢ(λᵢ+zᵢ)>0`
    /// （`W`の正定値性、対数尤度の大域凹性の根拠）がクランプ領域でも保たれる。
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let normal = Normal::standard();
        let mut h = vec![vec![0.0; k]; k];
        for i in 0..n {
            let w = self.hessian_weight(i, param, &normal);
            for (a, row) in h.iter_mut().enumerate() {
                let xa = *self.x.get(i, a);
                for (b, cell) in row.iter_mut().enumerate() {
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

    use crate::nonlinear::common::SolverType;
    use crate::nonlinear::common::U_CLAMP;
    use crate::nonlinear::common::clamped_pdf_cdf;

    use crate::nonlinear::common::destandardize_params;

    use crate::nonlinear::common::ols_based_initial_params;

    use crate::nonlinear::common::run_solver;

    use crate::nonlinear::common::standardize_columns;

    use crate::nonlinear::probit::ProbitInput;

    use argmin::core::Error as OptimizerError;

    use statrs::distribution::Continuous;
    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;

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

    /// `scores`は行ごとの一般化残差`λᵢ`を観測あたり1回だけ計算して行を埋める。セルごとに
    /// 再計算する素朴な式と、ビット単位で一致することを確認する（`|z|`が`U_CLAMP`を
    /// 超える極端なパラメータ点を含む）。
    /// 参照式は本体と同じヘルパーを呼ぶため、検出するのは行列に詰める配置・順序の退行で、
    /// ヘルパー側の式の変更は別のテスト（閉形式・数値微分）が担う。
    #[test]
    fn scores_match_per_cell_reference_bit_for_bit() {
        let input = small_input();
        let problem = ProbitProblem::new(&input);
        let normal = Normal::standard();
        for params in [vec![0.0, 0.0], vec![0.3, -0.7], vec![40.0, -25.0]] {
            let scores = problem.scores(&params);
            for i in 0..scores.nrows() {
                for j in 0..scores.ncols() {
                    let expected =
                        problem.generalized_residual(i, &params, &normal) * (*problem.x.get(i, j));
                    assert_eq!(scores.get(i, j).to_bits(), expected.to_bits(), "({i}, {j})");
                }
            }
        }
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

    #[test]
    fn cost_gradient_hessian_stay_finite_for_extreme_linear_predictor() {
        // z_i=1000（クランプが無ければu=q*z=±1000でφ(u)/Φ(u)が0にアンダーフローし
        // NaNになる、`probit.rs`のモジュールdoc「数値安定化について」参照）。U_CLAMPでのクランプ
        // によりcost/gradient/hessianが常に有限であることを確認する。
        let y = vec![1.0, 0.0];
        let input = ProbitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap();
        let problem = ProbitProblem::new(&input);
        let params = vec![1000.0];

        let cost = problem.cost(&params).unwrap();
        assert!(cost.is_finite(), "{cost}");

        let grad = problem.gradient(&params).unwrap();
        assert!(grad[0].is_finite(), "{:?}", grad);

        let hessian = problem.hessian(&params).unwrap();
        assert!(hessian[0][0].is_finite(), "{:?}", hessian);

        let scores = problem.scores(&params);
        for i in 0..2 {
            assert!(scores.get(i, 0).is_finite(), "row {i}");
        }
    }

    /// `|z|`が`U_CLAMP`未満の通常領域では`z̃=z`なので、`hessian_weight`は素の
    /// `λ(λ+z)`と一致し、`generalized_residual`は`scores`の切片列（`λ·1`）と一致する。
    #[test]
    fn hessian_weight_and_generalized_residual_match_closed_form_in_the_unclamped_region() {
        let y = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![0.5, -1.0, 2.0, 0.3, -0.7]];
        let input =
            ProbitInput::from_columns(&y, &x_columns, vec!["x".to_string()], true, "y".to_string())
                .unwrap();
        let problem = ProbitProblem::new(&input);
        let params = vec![0.2, 0.4];
        let normal = Normal::standard();
        let scores = problem.scores(&params);

        for i in 0..y.len() {
            let z = params[0] + params[1] * x_columns[0][i];
            let q = 2.0 * y[i] - 1.0;
            let lambda = q * normal.pdf(q * z) / normal.cdf(q * z);

            let residual = problem.generalized_residual(i, &params, &normal);
            assert!((residual - lambda).abs() < 1e-12, "row {i}");
            assert!((*scores.get(i, 0) - residual).abs() < 1e-12, "row {i}");

            let weight = problem.hessian_weight(i, &params, &normal);
            assert!((weight - lambda * (lambda + z)).abs() < 1e-12, "row {i}");
        }
    }

    #[test]
    fn hessian_weight_is_non_negative_even_when_misclassified_observation_exceeds_u_clamp() {
        // クランプ済みλと生のzを混在させると、|u|>U_CLAMPかつ誤分類
        // （qz が大きく負）の観測でw=λ(λ+z)が負になりうる（λᵢ(λᵢ+zᵢ)>0という
        // 大域凹性の前提が数値的に破れる、`probit.rs`のモジュールdocの数値安定化についての節参照）。
        //
        // 上の`cost_gradient_hessian_stay_finite_for_extreme_linear_predictor`と
        // 同じデータ（切片のみ、z=1000）を使う。y=[1,0]の2件はどちらも同じ`z=1000`を
        // 共有するが、y=0の観測はq=-1でqz=-1000（大きく負、誤分類）となり、
        // これがまさに再現条件。修正前はh[0][0]が約-8177（手計算
        // 概算値≈-8064と整合）だったことを確認済み（バグ注入により再現）。
        let y = vec![1.0, 0.0];
        let input = ProbitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap();
        let problem = ProbitProblem::new(&input);
        let params = vec![1000.0];

        let hessian = problem.hessian(&params).unwrap();
        assert!(
            hessian[0][0] >= 0.0,
            "Hessian weight should stay non-negative (X'WX positive semi-definite), got {}",
            hessian[0][0]
        );
    }

    /// `U_CLAMP`領域での`cost()`/`gradient()`の数学的非整合（`docs/spec/probit-spec.md`
    /// 4章に未検証リスクとして記載されていた項目）が
    /// 実際にBFGS/L-BFGSのline searchを壊すかを、near-separationデータ（真の有限MLEは
    /// 存在するが収束点付近でも一部観測の`|u|`が`U_CLAMP`を大きく超える設計）で検証する。
    ///
    /// `ProbitEstimator::fit`を直接呼ばず、その内部パイプライン（標準化・
    /// `ols_based_initial_params`によるwarm start・`run_solver`）をこのテスト内で
    /// 再現している。`ProbitProblem`を`Instrumented`でラップし、`cost`/`gradient`/
    /// `hessian`が呼ばれるたび（line searchのトライアル評価も含む）に`|u|`の最大値を
    /// 記録することで、**このテストが実際に辿る経路自身で`|u|`が`U_CLAMP`を超えたこと**を
    /// 下のアサーションで検証する（fit()をブラックボックスのまま呼ぶだけでは、将来
    /// warm start/line searchの実装が変わり`|u|`が`U_CLAMP`を超えなくなった場合に、
    /// テスト名が主張する検証内容とテストの中身が乖離したまま気づけないリスクがある
    /// ため、この形にしている）。
    ///
    /// 実測（2026-09-13、調査時点のこのテスト自身のコード）: newton/bfgs/lbfgsいずれも
    /// `max|u|`が`U_CLAMP`（≈8.13）を大きく超える（bfgsは約36、lbfgsは約185）にもかかわらず
    /// 3手法とも収束し、destandardize後のパラメータが相互に相対誤差1e-3程度で一致した
    /// （`rtol=1e-2`はここから1桁のマージンを取った値）。より厳しい設定（コールドスタート
    /// `[0,0]`・より強い分離`beta1=50`）でも同様に一致することを別途手動で確認済みだが、
    /// このテストは最も基本的な「fit()と同じwarm startを使った通常の呼び出し」に絞って
    /// 固定する。line searchが受理可能なステップを見つけられない・不適切なステップを
    /// 受理するという理論上の懸念は、試した範囲では一度も顕在化しなかった。
    #[test]
    fn fit_bfgs_and_lbfgs_match_newton_despite_deep_u_clamp_excursions_in_near_separation_data() {
        use std::cell::Cell;
        use std::rc::Rc;

        #[derive(Clone)]
        struct Instrumented {
            inner: ProbitProblem,
            max_abs_u: Rc<Cell<f64>>,
        }

        impl Instrumented {
            fn record(&self, params: &[f64]) {
                let n = self.inner.x.nrows();
                let mut m = self.max_abs_u.get();
                for i in 0..n {
                    let z: f64 = (0..self.inner.x.ncols())
                        .map(|j| *self.inner.x.get(i, j) * params[j])
                        .sum();
                    let q = 2.0 * (*self.inner.y.get(i, 0)) - 1.0;
                    let u = (q * z).abs();
                    if u > m {
                        m = u;
                    }
                }
                self.max_abs_u.set(m);
            }
        }

        impl CostFunction for Instrumented {
            type Param = Vec<f64>;
            type Output = f64;
            fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
                self.record(param);
                self.inner.cost(param)
            }
        }
        impl Gradient for Instrumented {
            type Param = Vec<f64>;
            type Gradient = Vec<f64>;
            fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
                self.record(param);
                self.inner.gradient(param)
            }
        }
        impl Hessian for Instrumented {
            type Param = Vec<f64>;
            type Hessian = Vec<Vec<f64>>;
            fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
                self.record(param);
                self.inner.hessian(param)
            }
        }

        // 決定論的な疑似乱数（LCG、`std::collections::hash_map::DefaultHasher`は
        // 実装詳細でありバージョン間の安定性が保証されないため使わない。定数は
        // `logit/fit/tests/convergence.rs`の近傍分離テストの`lcg`と同じ乗数・増分に揃えている）。
        fn lcg_uniform(state: &mut u64) -> f64 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((*state >> 11) as f64) / ((1u64 << 53) as f64)
        }

        let n = 300usize;
        let beta1_true = 10.0;
        let normal = Normal::standard();
        // `x1`はLCGでランダム生成せず、決定論的な分位点格子（`(i+0.5)/n`を`inverse_cdf`に
        // 通す）にする。ランダム抽出だと裾（`|x1|`が大きい観測）が偶然含まれない
        // seedを引くリスクがあるが、格子にすることで両端（`u≈1/600`・`599.5/600`）が
        // 常に含まれ、`beta1_true=10`と組み合わせて`|z|`が`U_CLAMP`を大きく超える
        // 観測を確実に生成できる。`y`（分離を乱すノイズ）だけLCGで抽出する。
        let mut x1 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut rng_state = 42u64;
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            let xi = normal.inverse_cdf(u);
            let p = normal.cdf(beta1_true * xi);
            let draw = lcg_uniform(&mut rng_state);
            x1.push(xi);
            y.push(if draw < p { 1.0 } else { 0.0 });
        }

        let input =
            ProbitInput::from_columns(&y, &[x1], vec!["x1".to_string()], true, "y".to_string())
                .unwrap();
        let (x_std, scale) = standardize_columns(input.x(), input.has_intercept());
        let initial_params = ols_based_initial_params(
            &x_std,
            input.x(),
            input.y(),
            input.has_intercept(),
            &[],
            |p_bar| {
                let eta0 = normal.inverse_cdf(p_bar);
                (normal.pdf(eta0), eta0)
            },
        )
        .unwrap();

        let run = |solver: SolverType| {
            let inner = ProbitProblem::from_standardized(x_std.clone(), input.y().clone());
            let tracker = Rc::new(Cell::new(0.0_f64));
            let problem = Instrumented {
                inner,
                max_abs_u: tracker.clone(),
            };
            let output = run_solver(
                problem,
                solver,
                initial_params.clone(),
                200,
                1e-6,
                input.y().nrows(),
                true,
            )
            .unwrap();
            assert!(output.converged, "{solver:?} did not converge");
            (destandardize_params(&output.params, &scale), tracker.get())
        };

        let (newton_params, newton_max_u) = run(SolverType::Newton);
        let (bfgs_params, bfgs_max_u) = run(SolverType::Bfgs);
        let (lbfgs_params, lbfgs_max_u) = run(SolverType::Lbfgs);

        // このテストが検証したいのは「U_CLAMPを超える領域を実際に通過してもbfgs/lbfgsが
        // 壊れない」ことであり、通過しなければテストの主張自体が空虚になる。
        // ここで実際に超えたことを固定する。
        assert!(
            bfgs_max_u > U_CLAMP,
            "bfgs did not actually enter the U_CLAMP region (max|u|={bfgs_max_u}); \
             this test's premise no longer holds"
        );
        assert!(
            lbfgs_max_u > U_CLAMP,
            "lbfgs did not actually enter the U_CLAMP region (max|u|={lbfgs_max_u}); \
             this test's premise no longer holds"
        );

        for (name, params) in [("bfgs", &bfgs_params), ("lbfgs", &lbfgs_params)] {
            for j in 0..2 {
                let expected = newton_params[j];
                let actual = params[j];
                // 実測相対誤差は1e-3程度（2026-09-13時点）。そこから1桁のマージンを
                // 取った値。
                let rtol = 1e-2;
                assert!(
                    (actual - expected).abs() <= rtol * expected.abs().max(1.0),
                    "{name} param[{j}]={actual} diverged from newton's {expected} \
                     (newton_max_u={newton_max_u})"
                );
            }
        }
    }

    /// `hessian_weight`の非負性をクランプ領域まで振って検証するproperty-basedテスト。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        // `probit/fit/tests/proptests.rs`と同じ理由でMAX_Kは小さく保つ。
        const MAX_K: usize = 4;

        fn x_names(k: usize) -> Vec<String> {
            (1..=k).map(|i| format!("x{i}")).collect()
        }

        /// `hessian_weight_is_non_negative_at_extreme_params`が生成するケース:
        /// `(n, k, x_cols, y_flags, intercept, slopes, v)`。`intercept`・`slopes`は
        /// `(符号, 10の指数)`で、パラメータの大きさを対数一様に振る（`|z|`が`U_CLAMP`を
        /// 大きく超える領域まで）。`v`は二次形式の方向ベクトル。
        type ExtremeCase = (
            usize,
            usize,
            Vec<Vec<f64>>,
            Vec<bool>,
            (bool, f64),
            Vec<(bool, f64)>,
            Vec<f64>,
        );

        /// `fit()`を通さずHessianを直接評価する点のケース。通常の`probit_case_strategy`は
        /// `|z|`が`U_CLAMP≈8.13`に届くことがほとんどなく、クランプ領域の誤分類観測を作れない
        /// ため、切片を`|β₀|∈[10, 1000]`（`U_CLAMP`超）、傾きを`|βⱼ|∈[1, 1000]`で振る。
        fn extreme_param_case_strategy() -> impl Strategy<Value = ExtremeCase> {
            (1..=MAX_K).prop_flat_map(|k| {
                (k + 5..=60usize).prop_flat_map(move |n| {
                    (
                        Just(n),
                        Just(k),
                        collection::vec(collection::vec(-2.0f64..2.0, n), k),
                        collection::vec(any::<bool>(), n),
                        (any::<bool>(), 1.0f64..3.0),
                        collection::vec((any::<bool>(), 0.0f64..3.0), k),
                        collection::vec(-1.0f64..1.0, k + 1),
                    )
                })
            })
        }

        fn signed_power_of_ten((positive, exponent): (bool, f64)) -> f64 {
            let magnitude = 10f64.powf(exponent);
            if positive { magnitude } else { -magnitude }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// 任意のパラメータ点で、観測ごとのHessian重み`wᵢ=λᵢ(λᵢ+z̃ᵢ)`が非負で、Hessian
            /// `X'WX`が半正定値（対角成分が非負、任意の`v`で`v'Hv≥0`）になる。`A(u)=λ(λ+u)>0`
            /// （`probit.rs`のモジュールdocの導出）の数値版で、クランプ領域の誤分類観測（`qᵢzᵢ<-U_CLAMP`）
            /// で`λ`だけクランプして生の`z`を混ぜると`w`が負になる回帰を捕まえる。
            ///
            /// 符号だけだと`|w|`で隠す変更や`hessian()`の組み立て誤りを見逃すため、(1)0行目の重みが
            /// クランプ境界`u=-U_CLAMP`での閉形式`m(m-U_CLAMP)`（`m=φ/Φ`）に一致すること、
            /// (2)`hessian()`が`Σᵢwᵢxᵢₐxᵢᵦ`と一致することも確認する。保証するのは「クランプ後の
            /// 関数としての半正定値性」で、クランプしない真のHessianとの一致ではない
            /// （クランプ領域で`cost`/`gradient`とHessianが非整合な既知の課題は別、`probit-spec.md`4章）。
            ///
            /// 0行目は説明変数を全て0にして`z₀=β₀`（`|β₀|>U_CLAMP`）とし、`y₀`を符号と逆に
            /// 置く（誤分類、`q₀z₀=-|β₀|<-U_CLAMP`）ので、どのケースでもクランプ領域の誤分類
            /// 観測を少なくとも1件含む。`y₁`は`y₀`と逆の値にして`y`に変動を持たせる。
            ///
            /// 有効性確認済み: `hessian_weight`の`clamped_z`を生の線形予測子`z`に戻すと失敗する。
            #[test]
            fn hessian_weight_is_non_negative_at_extreme_params(
                (n, k, mut x_cols, y_flags, intercept, slopes, v) in extreme_param_case_strategy(),
            ) {
                for col in &mut x_cols {
                    col[0] = 0.0;
                }
                let mut params = vec![signed_power_of_ten(intercept)];
                params.extend(slopes.iter().map(|&s| signed_power_of_ten(s)));

                let mut y: Vec<f64> = y_flags.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
                y[0] = if params[0] > 0.0 { 0.0 } else { 1.0 };
                y[1] = 1.0 - y[0];

                let input = ProbitInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let problem = ProbitProblem::new(&input);
                let normal = Normal::standard();

                let (_, u0) = problem.signed_index(0, &params);
                prop_assert!(u0 < -U_CLAMP, "row 0 must be misclassified beyond U_CLAMP, got u={u0}");

                let weights: Vec<f64> = (0..n)
                    .map(|i| problem.hessian_weight(i, &params, &normal))
                    .collect();
                for (i, w) in weights.iter().enumerate() {
                    prop_assert!(*w >= 0.0, "w[{i}]={w} should be non-negative");
                }
                // 0行目はクランプ境界`u=-U_CLAMP`と同じ重み: `q²=1`なので`w=m(m-U_CLAMP)`
                let (phi, big_phi) = clamped_pdf_cdf(&normal, -U_CLAMP);
                let m = phi / big_phi;
                let boundary_weight = m * (m - U_CLAMP);
                prop_assert!(
                    (weights[0] - boundary_weight).abs() <= 1e-9 * (1.0 + boundary_weight),
                    "w[0]={} should equal the boundary weight {boundary_weight}", weights[0]
                );

                let hessian = problem.hessian(&params).unwrap();
                let scale: f64 = 1.0 + (0..=k).map(|a| hessian[a][a].abs()).sum::<f64>();
                for (a, row) in hessian.iter().enumerate() {
                    for (b, cell) in row.iter().enumerate() {
                        let expected: f64 = (0..n)
                            .map(|i| weights[i] * problem.x.get(i, a) * problem.x.get(i, b))
                            .sum();
                        prop_assert!(
                            (cell - expected).abs() <= 1e-9 * scale,
                            "H[{a}][{b}]={cell} should equal sum_i w_i x_ia x_ib = {expected}"
                        );
                    }
                }
                for (a, row) in hessian.iter().enumerate() {
                    prop_assert!(row[a] >= 0.0, "H[{a}][{a}]={} should be non-negative", row[a]);
                }
                let quad: f64 = (0..=k)
                    .flat_map(|a| (0..=k).map(move |b| (a, b)))
                    .map(|(a, b)| v[a] * hessian[a][b] * v[b])
                    .sum();
                prop_assert!(quad >= -1e-9 * scale, "v'Hv={quad} should be non-negative (scale={scale})");
            }
        }
    }
}
