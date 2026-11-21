//! `TobitProblem`（負の対数尤度・スコア・Hessian。argminの`CostFunction`/`Gradient`/`Hessian`の実装）と、
//! 観測ごとの尤度の寄与（非打ち切り・打ち切り）。

#[cfg(test)]
use super::input::TobitInput;
use crate::nonlinear::common::{U_CLAMP, clamped_pdf_cdf, compensated_sum, linear_predictor};
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;
use statrs::distribution::Normal;

/// `0.5 * ln(2π)`。正規分布の対数密度`log φ(v) = -0.5*ln(2π) - v²/2`の定数項
/// （コンパイル時定数としてハードコード。`ln`は非`const fn`のため`nonlinear/common/normal.rs`の
/// `U_CLAMP`と同じ方式）。
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_7;

/// 観測1件の対数尤度・スコア・Hessianへの寄与（`tobit.rs`のモジュールdocの数式表参照）。`cost`/
/// `gradient`/`hessian`/`scores`いずれも観測ごとに同じ分類（非打ち切り/左打ち切り/
/// 右打ち切り）を経由するため、重複を避けてここに集約する。
pub(super) struct Contribution {
    /// 対数尤度への寄与 `ℓᵢ(θ)`
    pub(super) log_lik: f64,
    /// 生のスコア`∂ℓᵢ/∂θ`のうち`β`成分の係数（実際の値は`coef * xᵢⱼ/σ`、`Gradient`
    /// トレイトとは符号が逆。`scores()`が返す値そのもの）
    score_beta_coef: f64,
    /// 生のスコアのうち`s`（`log σ`）成分 `∂ℓᵢ/∂s`
    score_s: f64,
    /// `-ℓᵢ`のHessianのうち`β`ブロックの係数（実際の値は`coef * xᵢⱼxᵢₘ/σ²`）
    h_beta_coef: f64,
    /// `-ℓᵢ`のHessianのうち`β`-`s`ブロックの係数（実際の値は`coef * xᵢⱼ/σ`）
    h_beta_s_coef: f64,
    /// `-ℓᵢ`のHessianのうち`s`-`s`要素
    h_ss: f64,
}

/// 非打ち切り観測（`tobit.rs`のモジュールdocの数式表「非打ち切り」列）。`v = (yᵢ-xᵢ'β)/σ`、
/// `s = log σ`。
fn uncensored_contribution(v: f64, s: f64) -> Contribution {
    Contribution {
        log_lik: -s - HALF_LN_2PI - 0.5 * v * v,
        score_beta_coef: v,
        score_s: v * v - 1.0,
        h_beta_coef: 1.0,
        h_beta_s_coef: 2.0 * v,
        h_ss: 2.0 * v * v,
    }
}

/// 打ち切り観測（左右いずれか、`tobit.rs`のモジュールdocの数式表参照）。`zeta`は打ち切り境界を
/// 標準化した値（左打ち切り: `(lower-xᵢ'β)/σ`、右打ち切り: `(xᵢ'β-upper)/σ`）、
/// `direction`は`β`成分の符号（左: `1.0`、右: `-1.0`。`s`成分・Hessianはいずれも
/// `direction`に依らず同じ式に帰着する、`tobit.rs`のモジュールdocコメント参照）。
///
/// **Hessian項（`a=A(u)`・`c=C(u)`・`h_ss`）が使う`u`は、`λ`と同じクランプ済み
/// 引数から再構成する**（Probitの`ProbitProblem`と同型のバグ）。
/// `λ`は`clamped_pdf_cdf`が`zeta`を`[-U_CLAMP, U_CLAMP]`にクランプした後の値を
/// 使うため、`|zeta|>U_CLAMP`の領域では`λ`は`zeta`に対して事実上定数になる。
/// ここで生の（非クランプの）`zeta`を`A(u)=λ(u+λ)`の計算に混ぜると、`A(u)>0`と
/// いう恒等式（`tobit.rs`のモジュールdocの数式表、Probitの大域凹性と同型の根拠）が数値的に
/// 破れ、`h_beta_coef`が負になりうる。`score_s`（勾配）は今回のスコープ外
/// （`U_CLAMP`領域でのcost/gradientの数学的非整合は別の既知の課題、
/// `docs/spec/probit-spec.md`4章参照）のため、`zeta`のまま変更しない。
fn censored_contribution(normal: &Normal, zeta: f64, direction: f64) -> Contribution {
    let (phi, big_phi) = clamped_pdf_cdf(normal, zeta);
    let lambda = phi / big_phi;
    let zeta_for_hessian = zeta.clamp(-U_CLAMP, U_CLAMP);
    let a = lambda * (zeta_for_hessian + lambda);
    let c = zeta_for_hessian * a - lambda;
    Contribution {
        log_lik: big_phi.ln(),
        score_beta_coef: -direction * lambda,
        score_s: -zeta * lambda,
        h_beta_coef: a,
        h_beta_s_coef: direction * c,
        h_ss: zeta_for_hessian * c,
    }
}

/// Tobitの負の対数尤度・スコア・Hessian（argminの`CostFunction`/`Gradient`/`Hessian`
/// トレイト実装）。`TobitInput`の`X`・`y`・打ち切り境界を保持する（`run_solver`が
/// `problem`の所有権を必要とするため、`TobitInput`とは独立した所有データとして持つ。
/// `Clone`は`argmin::core::Executor`が要求する。`LogitProblem`/`ProbitProblem`と
/// 同じ設計）。
#[derive(Debug, Clone)]
pub struct TobitProblem {
    x: Mat<f64>,
    y: Mat<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
}

impl TobitProblem {
    /// `input`の`x`・`y`・打ち切り境界をそのまま（未標準化のスケールで）複製して構築する。
    /// 閉じた形の解・OLSとの整合性と突き合わせる単体テスト専用（`LogitProblem::new`と
    /// 同じ位置づけ。`TobitEstimator::fit`は標準化後の設計行列を使うため、こちらではなく
    /// `from_standardized`を使う）。
    #[cfg(test)]
    pub(super) fn new(input: &TobitInput) -> Self {
        Self {
            x: input.x().clone(),
            y: input.y().clone(),
            lower: input.lower(),
            upper: input.upper(),
        }
    }

    /// `TobitScaling`で標準化済みの設計行列`x_std`・`y`（`fit()`では`y`もスケーリング
    /// 済み）・打ち切り境界（`y`と同じ`y_scale`でスケーリング済み）から構築する。
    /// `TobitEstimator::fit`が最適化に使う経路（`LogitProblem::from_standardized`と
    /// 同じ位置づけ）。名前に反して標準化済みかどうかは問わない（`x`・`y`・境界・
    /// `params`のスケールが対応してさえいればよい。単体テストは生スケールでも呼ぶ）。
    pub(super) fn from_standardized(
        x_std: Mat<f64>,
        y: Mat<f64>,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        Self {
            x: x_std,
            y,
            lower,
            upper,
        }
    }

    /// 説明変数の数`k`（定数項を含む、`params`の先頭`k`要素が`β`に対応）。
    fn k(&self) -> usize {
        self.x.ncols()
    }

    /// 観測`i`の分類（非打ち切り/左打ち切り/右打ち切り）に応じた[`Contribution`]を計算する。
    /// `TobitInput::from_columns`が`lower <= yᵢ <= upper`を保証しているため、境界値との
    /// 一致判定（`==`）で打ち切り観測を識別できる（`y*ᵢ`が境界値ちょうどになる確率は
    /// 連続分布の下で0のため、境界値ちょうどの観測は定義上すべて打ち切り観測）。
    pub(super) fn contribution(&self, i: usize, params: &[f64], normal: &Normal) -> Contribution {
        let k = self.k();
        let s = params[k];
        let sigma = s.exp();
        let xb = linear_predictor(&self.x, i, params);
        let y_i = *self.y.get(i, 0);

        if let Some(l) = self.lower
            && y_i == l
        {
            let zeta = (l - xb) / sigma;
            censored_contribution(normal, zeta, 1.0)
        } else if let Some(u) = self.upper
            && y_i == u
        {
            let zeta = (xb - u) / sigma;
            censored_contribution(normal, zeta, -1.0)
        } else {
            let v = (y_i - xb) / sigma;
            uncensored_contribution(v, s)
        }
    }

    /// 観測ごとのスコア行列（n×(k+1)、末尾の列が`s=logσ`成分）。各行が`sᵢ=∂ℓᵢ/∂θ`
    /// （対数尤度の1階微分そのもの、`Gradient`トレイトとは符号が逆）。OPG/サンドイッチ/
    /// クラスターSEの計算に使う（`LogitProblem::scores`/`ProbitProblem::scores`と同じ役割）。
    pub fn scores(&self, params: &[f64]) -> Mat<f64> {
        let n = self.x.nrows();
        let k = self.k();
        let sigma = params[k].exp();
        let normal = Normal::standard();
        // `contribution`（`linear_predictor`・`φ`/`Φ`・打ち切り分岐）は列によらないため、
        // 観測あたり1回だけ計算する。`Contribution`ごとは保持せず、使う2成分だけ集める。
        let mut beta_coefs = Vec::with_capacity(n);
        let mut score_s = Vec::with_capacity(n);
        for i in 0..n {
            let contribution = self.contribution(i, params, &normal);
            beta_coefs.push(contribution.score_beta_coef);
            score_s.push(contribution.score_s);
        }
        Mat::from_fn(n, k + 1, |i, col| {
            if col < k {
                beta_coefs[i] * (*self.x.get(i, col)) / sigma
            } else {
                score_s[i]
            }
        })
    }
}

impl CostFunction for TobitProblem {
    type Param = Vec<f64>;
    type Output = f64;

    /// 負の対数尤度 `-ℓ(θ) = -Σᵢ ℓᵢ(θ)`（`tobit.rs`のモジュールdocの数式表参照）。
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        let n = self.x.nrows();
        let normal = Normal::standard();
        // 補償和（`compensated_sum`のdocコメント参照。大標本のNewtonのコスト比較のため）。
        let ll = compensated_sum((0..n).map(|i| self.contribution(i, param, &normal).log_lik));
        Ok(-ll)
    }
}

impl Gradient for TobitProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// `-ℓ(θ)`の勾配（対数尤度のスコアの符号反転、`tobit.rs`のモジュールdocの数式表参照）。
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        let n = self.x.nrows();
        let k = self.k();
        let sigma = param[k].exp();
        let normal = Normal::standard();
        let mut grad = vec![0.0; k + 1];
        for i in 0..n {
            let contribution = self.contribution(i, param, &normal);
            // `grad`は長さ`k+1`（末尾が`s=logσ`）。`take(k)`でβ成分だけを回す
            // （外すと`x.get(i, k)`が範囲外になる）。
            for (j, grad_j) in grad.iter_mut().enumerate().take(k) {
                *grad_j += -contribution.score_beta_coef * (*self.x.get(i, j)) / sigma;
            }
            grad[k] += -contribution.score_s;
        }
        Ok(grad)
    }
}

impl Hessian for TobitProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// `-ℓ(θ)`のHessian（`(k+1)×(k+1)`、`tobit.rs`のモジュールdocの数式表参照。`Contribution`の
    /// `h_*`フィールドは既に`-ℓᵢ`のHessian成分として定義済みのため、追加の符号反転は
    /// 不要）。`run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照。
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        let n = self.x.nrows();
        let k = self.k();
        let sigma = param[k].exp();
        let sigma2 = sigma * sigma;
        let normal = Normal::standard();
        let mut h = vec![vec![0.0; k + 1]; k + 1];
        for i in 0..n {
            let contribution = self.contribution(i, param, &normal);
            // `h`は`(k+1)×(k+1)`（最後の行・列が`s=logσ`）。`take(k)`でβのブロック
            // （行`a<k`、列`b<k`）だけを回す。`take`を外すと`x.get(i, k)`が範囲外になる。
            for (a, row) in h.iter_mut().enumerate().take(k) {
                let xa = *self.x.get(i, a);
                for (b, cell) in row.iter_mut().enumerate().take(k) {
                    *cell += contribution.h_beta_coef * xa * (*self.x.get(i, b)) / sigma2;
                }
                // 対称性から`h[k][a]`も同じ値になるが、行`k`は`h.iter_mut()`の範囲外
                // （`take(k)`）のため、`row[k]`（`h[a][k]`）のみここで更新し、`h[k][a]`は
                // ループ後に一括で複製する（`h`全体を可変・不変で同時に借用できないため）。
                row[k] += contribution.h_beta_s_coef * xa / sigma;
            }
            h[k][k] += contribution.h_ss;
        }
        let cross_terms: Vec<f64> = h.iter().take(k).map(|row| row[k]).collect();
        h[k][..k].copy_from_slice(&cross_terms);
        Ok(h)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::nonlinear::common::U_CLAMP;

    use crate::nonlinear::common::clamped_pdf_cdf;

    use crate::nonlinear::tobit::TobitInput;

    use statrs::distribution::Continuous;
    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;

    /// n=4, k=2（切片+x1）。`y=[0.0, 2.0, 5.0, 3.0]`・`lower=Some(0.0)`・`upper=Some(5.0)`で、
    /// 観測0が左打ち切り・観測2が右打ち切り・観測1,3が非打ち切りという3分類すべてを含む
    /// （`Contribution`の3分岐・左右の符号反転を1つのテストデータで網羅的に検証するため）。
    fn mixed_censoring_input() -> TobitInput {
        let y = vec![0.0, 2.0, 5.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(5.0),
        )
        .unwrap()
    }

    #[test]
    fn gradient_matches_numerical_differentiation_of_cost() {
        let input = mixed_censoring_input();
        let problem = TobitProblem::new(&input);
        let params = vec![0.5, 0.3, -0.2];
        let h = 1e-6;

        let analytic = problem.gradient(&params).unwrap();
        for j in 0..3 {
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
        let input = mixed_censoring_input();
        let problem = TobitProblem::new(&input);
        let params = vec![0.5, 0.3, -0.2];
        let h = 1e-5;

        let analytic = problem.hessian(&params).unwrap();
        for j in 0..3 {
            let mut plus = params.clone();
            plus[j] += h;
            let mut minus = params.clone();
            minus[j] -= h;
            let grad_plus = problem.gradient(&plus).unwrap();
            let grad_minus = problem.gradient(&minus).unwrap();
            for i in 0..3 {
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

    /// `scores`は`contribution`を観測あたり1回だけ計算して行を埋める。セルごとに再計算する
    /// 素朴な式と、ビット単位で一致することを確認する（左・右・非打ち切りの観測と、
    /// `|zeta|`が`U_CLAMP`を超える極端なパラメータ点を含む）。
    /// 参照式は本体と同じヘルパーを呼ぶため、検出するのは行列に詰める配置・順序の退行で、
    /// ヘルパー側の式の変更は別のテスト（閉形式・数値微分）が担う。
    #[test]
    fn scores_match_per_cell_reference_bit_for_bit() {
        let input = mixed_censoring_input();
        let problem = TobitProblem::new(&input);
        let normal = Normal::standard();
        // 通常点に加え、`zeta`が両側でクランプされる点（`β₀`の符号違い）と、`σ=e^s`が極小・
        // 極大の点を含める。
        for params in [
            vec![0.5, 0.3, -0.2],
            vec![-1.0, 2.0, 0.4],
            vec![30.0, -9.0, -1.5],
            vec![-30.0, -9.0, -1.5],
            vec![0.5, 0.3, -6.0],
            vec![0.5, 0.3, 4.0],
        ] {
            let scores = problem.scores(&params);
            let (k, sigma) = (problem.k(), params[problem.k()].exp());
            for i in 0..scores.nrows() {
                for col in 0..=k {
                    let c = problem.contribution(i, &params, &normal);
                    let expected = if col < k {
                        c.score_beta_coef * (*problem.x.get(i, col)) / sigma
                    } else {
                        c.score_s
                    };
                    assert_eq!(
                        scores.get(i, col).to_bits(),
                        expected.to_bits(),
                        "({i}, {col})"
                    );
                }
            }
        }
    }

    #[test]
    fn scores_sum_to_negative_gradient() {
        // scoresは対数尤度の生のスコア（符号反転なし）、gradientはCostFunction
        // （負の対数尤度）の勾配のため、観測方向に合計すると符号が逆になるはず
        // （LogitProblem/ProbitProblemの同名テストと同じ検証）。
        let input = mixed_censoring_input();
        let problem = TobitProblem::new(&input);
        let params = vec![0.5, 0.3, -0.2];

        let scores = problem.scores(&params);
        let grad = problem.gradient(&params).unwrap();
        let n = scores.nrows();

        for j in 0..3 {
            let sum: f64 = (0..n).map(|i| *scores.get(i, j)).sum();
            assert!(
                (sum - (-grad[j])).abs() < 1e-9,
                "j={j}, sum={sum}, grad={:?}",
                grad
            );
        }
    }

    #[test]
    fn cost_matches_closed_form_normal_log_likelihood_when_no_observation_is_censored() {
        // 境界を極端に広く取り、全観測が非打ち切りになるデータ。この場合Tobitの対数尤度は
        // 通常の正規回帰の対数尤度に一致するはず（「打ち切りなし観測のみの
        // データで、cost/gradient/hessianがOLSの対数尤度と整合すること」という完了条件の境界ケース検算）。
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(-100.0),
            Some(100.0),
        )
        .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![0.5, 0.3, -0.2];

        let cost = problem.cost(&params).unwrap();

        let sigma = params[2].exp();
        let n = y.len() as f64;
        let sse: f64 = (0..y.len())
            .map(|i| {
                let xb = params[0] + params[1] * x_columns[0][i];
                (y[i] - xb).powi(2)
            })
            .sum();
        let expected_ll = -(n / 2.0) * (2.0 * std::f64::consts::PI).ln()
            - n * sigma.ln()
            - sse / (2.0 * sigma * sigma);

        assert!(
            (cost - (-expected_ll)).abs() < 1e-9,
            "cost={cost}, expected={}",
            -expected_ll
        );
    }

    #[test]
    fn cost_gradient_hessian_stay_finite_for_extreme_linear_predictor() {
        // x1=1000のとき、打ち切り観測のzeta=(lower-x'β)/σが極端な値になり、クランプが
        // 無ければclamped_pdf_cdf内のφ(u)/Φ(u)が0.0/0.0のNaNになりうる
        // （Probitのcost_gradient_hessian_stay_finite_for_extreme_linear_predictorと
        // 同じ数値安定化の検証、`tobit.rs`のモジュールdoc「数値安定化について」参照）。
        let y = vec![0.0, 5.0];
        let x_columns = vec![vec![1000.0, 1000.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(5.0),
        )
        .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![0.0, 1.0, 0.0];

        let cost = problem.cost(&params).unwrap();
        assert!(cost.is_finite(), "cost={cost}");

        let grad = problem.gradient(&params).unwrap();
        for (j, g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "grad[{j}]={g}");
        }

        let hessian = problem.hessian(&params).unwrap();
        for (i, row) in hessian.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!(v.is_finite(), "hessian[{i}][{j}]={v}");
            }
        }
    }

    /// 打ち切り分岐（`censored_contribution`）を、数値微分ではなく`φ`/`Φ`の
    /// プリミティブから独立に再計算した値と直接突き合わせる。
    /// 数値微分だけでは`log_lik`/`score_*`/`h_*`に共通するスケール・符号誤りを
    /// 検出できないため。`z=(lower-x'β)/σ=0`となるよう`β0=0`・`σ=1`（`s=0`）・
    /// `lower=0`を選ぶと、`Φ(0)=0.5`・`φ(0)=1/√(2π)`という暗算しやすい値になる。
    #[test]
    fn left_censored_cost_gradient_hessian_scores_match_closed_form_at_zeta_zero() {
        let y = vec![0.0];
        let input =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(0.0), None)
                .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![0.0, 0.0]; // β0=0, s=0(σ=1) → z=(0-0)/1=0

        let normal = Normal::standard();
        let phi0 = normal.pdf(0.0);
        let big_phi0 = normal.cdf(0.0);
        let lambda0 = phi0 / big_phi0;
        let a0 = lambda0 * (0.0 + lambda0);
        let c0 = 0.0 * a0 - lambda0;

        let cost = problem.cost(&params).unwrap();
        assert!((cost - (-big_phi0.ln())).abs() < 1e-12, "cost={cost}");

        // 左打ち切り: direction=1.0 → cost_grad_beta_coef=direction*lambda0=lambda0
        let grad = problem.gradient(&params).unwrap();
        assert!((grad[0] - lambda0).abs() < 1e-12, "grad[0]={}", grad[0]);
        assert!(grad[1].abs() < 1e-12, "grad[1]={}", grad[1]);

        let hessian = problem.hessian(&params).unwrap();
        assert!((hessian[0][0] - a0).abs() < 1e-12, "h00={}", hessian[0][0]);
        assert!((hessian[0][1] - c0).abs() < 1e-12, "h01={}", hessian[0][1]);
        assert!((hessian[1][0] - c0).abs() < 1e-12, "h10={}", hessian[1][0]);
        assert!(hessian[1][1].abs() < 1e-12, "h11={}", hessian[1][1]);

        // scoresは生スコア: score_beta_coef=-direction*lambda0=-lambda0
        let scores = problem.scores(&params);
        assert!((*scores.get(0, 0) - (-lambda0)).abs() < 1e-12);
        assert!((*scores.get(0, 1)).abs() < 1e-12);
    }

    /// `probit/problem.rs`と同型のバグ回帰ガード: `censored_contribution`の
    /// `A(u)=λ(u+λ)`計算で、クランプ済み`λ`と生の（非クランプの）`zeta`を混在させると、
    /// `|zeta|>U_CLAMP`の領域で`h_beta_coef`（`A(u)`）が負になりうる（`A(u)>0`という
    /// 恒等式が数値的に破れる、`tobit.rs`のモジュールdocの数式表参照）。左打ち切り（`lower=0.0`）・
    /// 切片のみで`β0=1000, σ=1`とすると`zeta=(0-1000)/1=-1000`となり、この経路を踏む。
    #[test]
    fn hessian_weight_is_non_negative_even_when_censored_observation_exceeds_u_clamp() {
        let y = vec![0.0];
        let input =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(0.0), None)
                .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![1000.0, 0.0]; // β0=1000, s=0（σ=1）→ zeta=(0-1000)/1=-1000

        let hessian = problem.hessian(&params).unwrap();
        assert!(
            hessian[0][0] >= 0.0,
            "Hessian weight A(u) should stay non-negative, got {}",
            hessian[0][0]
        );
    }

    /// `left_censored_cost_gradient_hessian_scores_match_closed_form_at_zeta_zero`の
    /// 右打ち切り版。`w=(x'β-upper)/σ=0`となるよう`upper=0`を選ぶ（`zeta`の値自体は
    /// 左打ち切り版と同じ0だが、`direction=-1.0`によりβ成分の符号が反転することを検証する）。
    #[test]
    fn right_censored_cost_gradient_hessian_scores_match_closed_form_at_zeta_zero() {
        let y = vec![0.0];
        let input =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), None, Some(0.0))
                .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![0.0, 0.0]; // β0=0, s=0(σ=1) → w=(0-0)/1=0

        let normal = Normal::standard();
        let phi0 = normal.pdf(0.0);
        let big_phi0 = normal.cdf(0.0);
        let lambda0 = phi0 / big_phi0;
        let a0 = lambda0 * (0.0 + lambda0);
        let c0 = 0.0 * a0 - lambda0;

        let cost = problem.cost(&params).unwrap();
        assert!((cost - (-big_phi0.ln())).abs() < 1e-12, "cost={cost}");

        // 右打ち切り: direction=-1.0 → cost_grad_beta_coef=direction*lambda0=-lambda0
        // （左打ち切り版とちょうど符号が反転する）
        let grad = problem.gradient(&params).unwrap();
        assert!((grad[0] - (-lambda0)).abs() < 1e-12, "grad[0]={}", grad[0]);
        assert!(grad[1].abs() < 1e-12, "grad[1]={}", grad[1]);

        let hessian = problem.hessian(&params).unwrap();
        assert!((hessian[0][0] - a0).abs() < 1e-12, "h00={}", hessian[0][0]);
        assert!(
            (hessian[0][1] - (-c0)).abs() < 1e-12,
            "h01={}",
            hessian[0][1]
        );
        assert!(
            (hessian[1][0] - (-c0)).abs() < 1e-12,
            "h10={}",
            hessian[1][0]
        );
        assert!(hessian[1][1].abs() < 1e-12, "h11={}", hessian[1][1]);

        let scores = problem.scores(&params);
        assert!((*scores.get(0, 0) - lambda0).abs() < 1e-12);
        assert!((*scores.get(0, 1)).abs() < 1e-12);
    }

    /// `fit()`を通さずHessianを直接評価するproperty-basedテスト。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        // 高kはbenchmarkのmany_regressorsシナリオでカバー済みのため小さく保つ
        // （logit・probitの`fit/tests/proptests.rs`と同じ理由）。
        const MAX_K: usize = 3;

        /// 左打ち切りの下限（打ち切りなしのTobitの意味が薄れない程度に、真の`y*`分布の
        /// 中心付近を狙う）。右打ち切りの上限はこの値と対称（`beta`の分布が0周りに対称な
        /// ため、左・右で打ち切り割合が同程度になる）。
        const LOWER_BOUND: f64 = 0.0;

        /// 打ち切り境界`(lower, upper)`。
        type Bounds = (Option<f64>, Option<f64>);

        /// 左打ち切り・右打ち切り・両側打ち切りをランダムに選ぶ。両側打ち切りの幅
        /// （`[-1, 1]`）は、`NoUncensoredObservations`等の`prop_assume`棄却が過剰にならない
        /// 程度に真の`y*`分布の中心付近を含める。
        fn censoring_strategy() -> impl Strategy<Value = Bounds> {
            prop_oneof![
                Just((Some(LOWER_BOUND), None)),
                Just((None, Some(LOWER_BOUND))),
                Just((Some(-1.0), Some(1.0))),
            ]
        }

        fn x_names(k: usize) -> Vec<String> {
            (1..=k).map(|i| format!("x{i}")).collect()
        }

        /// `censored_hessian_weight_is_non_negative_at_extreme_params`が生成するケース:
        /// `(n, k, x_cols, y_raw, censored_flags, bounds, intercept, slopes, s, v)`。
        /// `intercept`・`slopes`は`(符号, 10の指数)`、`s`は`logσ`、`v`は二次形式の方向。
        type ExtremeCase = (
            usize,
            usize,
            Vec<Vec<f64>>,
            Vec<f64>,
            Vec<bool>,
            Bounds,
            f64,
            Vec<(bool, f64)>,
            f64,
            Vec<f64>,
        );

        /// `fit()`を通さずHessianを直接評価する点のケース。`tobit_case_strategy`は
        /// `|zeta|`が`U_CLAMP≈8.13`に届くことがほとんどないため、`σ=e^s`（`s∈[-1,1]`）に
        /// 対して切片を`|β₀|∈[10^1.5, 1000]`で振り、0行目の`|zeta|=|β₀|/σ`が`U_CLAMP`を
        /// 超えるようにする。
        fn extreme_param_case_strategy() -> impl Strategy<Value = ExtremeCase> {
            (1..=MAX_K).prop_flat_map(|k| {
                (k + 5..=60usize).prop_flat_map(move |n| {
                    (
                        Just(n),
                        Just(k),
                        collection::vec(collection::vec(-2.0f64..2.0, n), k),
                        collection::vec(0.05f64..0.9, n),
                        collection::vec(any::<bool>(), n),
                        censoring_strategy(),
                        1.5f64..3.0,
                        collection::vec((any::<bool>(), 0.0f64..3.0), k),
                        -1.0f64..1.0,
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

            /// 任意のパラメータ点で、打ち切り観測の`A(u)=λ(λ+u)`（`h_beta_coef`）が非負で、
            /// Hessianの`β`ブロック（`A(u)`と非打ち切りの`1`を係数とする`Σ coef·xx'/σ²`）が
            /// 半正定値になる。`λ`だけクランプして生の`zeta`を混ぜると`|zeta|>U_CLAMP`の
            /// 領域で`A(u)`が負になる回帰を捕まえる（Probitの`hessian_weight_is_non_negative_
            /// at_extreme_params`と対）。`(β, logσ)`全体のHessianは大域凹でないため、符号の性質
            /// （非負・半正定値）は`β`ブロックだけに表明する。`β`-`s`・`s`-`s`ブロックは符号の
            /// 性質が無いので、代わりに(1)0行目の寄与がクランプ境界`zeta=-U_CLAMP`での閉形式
            /// （`A=m(m-U)`、`c=-U·A-m`、`h_beta_s=direction·c`、`h_ss=-U·c`、`m=φ/Φ`）に
            /// 一致すること、(2)`hessian()`が`Contribution`の和として組み立てられていること
            /// （全ブロック）を確認する。保証するのは「クランプ後の関数としての性質」で、クランプ
            /// しない真のHessianとの一致ではない（`score_s`は生の`zeta`を使うため、クランプ領域で
            /// `gradient`とHessianが非整合な既知の課題は別）。
            ///
            /// 0行目は説明変数を全て0にして`xb=β₀`とし、`β₀`の符号を打ち切り方向と逆に置く
            /// （左・両側は下限で打ち切り、`β₀>0`で`zeta=(l-β₀)/σ<-U_CLAMP`。右は
            /// `β₀<0`で`zeta=(β₀-u)/σ<-U_CLAMP`）ので、どのケースでもクランプ領域の打ち切り
            /// 観測を少なくとも1件含む。1行目は非打ち切りにして、打ち切りと非打ち切りの両方を含める。
            ///
            /// 有効性確認済み: `censored_contribution`の`zeta_for_hessian`を生の`zeta`に戻すと、
            /// 左・右どちらの分岐だけを戻しても失敗する（通常の`tobit_case_strategy`は`|zeta|`が
            /// `U_CLAMP`に届かず検出できない）。
            #[test]
            fn censored_hessian_weight_is_non_negative_at_extreme_params(
                (n, k, mut x_cols, y_raw, censored, bounds, intercept_exp, slopes, s, v)
                    in extreme_param_case_strategy(),
            ) {
                for col in &mut x_cols {
                    col[0] = 0.0;
                }
                let (lower, upper) = bounds;
                let beta0 = signed_power_of_ten((lower.is_some(), intercept_exp));
                let mut params = vec![beta0];
                params.extend(slopes.iter().map(|&slope| signed_power_of_ten(slope)));
                params.push(s);

                // 観測の値: 打ち切りなら境界値、そうでなければ区間内部の値。0行目は必ず
                // 打ち切り、1行目は必ず非打ち切り。0行目の境界は`β₀`の符号と逆側にそろえる
                // （下限があれば下限、なければ上限）。
                let y: Vec<f64> = (0..n)
                    .map(|i| {
                        let r = y_raw[i];
                        let is_censored = i == 0 || (i != 1 && censored[i]);
                        match (lower, upper) {
                            (Some(l), None) => if is_censored { l } else { l + r },
                            (None, Some(u)) => if is_censored { u } else { u - r },
                            (Some(l), Some(_)) if i == 0 => l,
                            (Some(l), Some(u)) => {
                                if !is_censored { l + (u - l) * r } else if r > 0.5 { u } else { l }
                            }
                            (None, None) => unreachable!("censoring_strategy always sets a bound"),
                        }
                    })
                    .collect();

                let input = TobitInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string(), lower, upper).unwrap();
                let problem = TobitProblem::new(&input);
                let normal = Normal::standard();
                let sigma = s.exp();

                let zeta0 = match lower {
                    Some(l) => (l - params[0]) / sigma,
                    None => (params[0] - upper.unwrap_or(0.0)) / sigma,
                };
                prop_assert!(zeta0 < -U_CLAMP, "row 0 must be censored beyond U_CLAMP, got zeta={zeta0}");

                let contributions: Vec<Contribution> =
                    (0..n).map(|i| problem.contribution(i, &params, &normal)).collect();
                for (i, c) in contributions.iter().enumerate() {
                    prop_assert!(c.h_beta_coef >= 0.0, "A(u)[{i}]={} should be non-negative", c.h_beta_coef);
                }
                // 0行目はクランプ境界`zeta=-U_CLAMP`と同じ寄与（0行目は`lower`があれば左、なければ右）
                let (phi, big_phi) = clamped_pdf_cdf(&normal, -U_CLAMP);
                let m = phi / big_phi;
                let a0 = m * (m - U_CLAMP);
                let c0 = -U_CLAMP * a0 - m;
                let direction = if lower.is_some() { 1.0 } else { -1.0 };
                let close = |actual: f64, expected: f64| (actual - expected).abs() <= 1e-9 * (1.0 + expected.abs());
                prop_assert!(close(contributions[0].h_beta_coef, a0), "A[0]={} vs {a0}", contributions[0].h_beta_coef);
                prop_assert!(close(contributions[0].h_beta_s_coef, direction * c0), "h_beta_s[0]={} vs {}", contributions[0].h_beta_s_coef, direction * c0);
                prop_assert!(close(contributions[0].h_ss, -U_CLAMP * c0), "h_ss[0]={} vs {}", contributions[0].h_ss, -U_CLAMP * c0);

                // `hessian()`は`Contribution`の和: 全ブロックを別ループで組み立て直して比べる
                let hessian = problem.hessian(&params).unwrap();
                let kb = k + 1; // 切片を含む`β`の本数
                let sigma2 = sigma * sigma;
                let scale: f64 = 1.0 + (0..=kb).map(|a| hessian[a][a].abs()).sum::<f64>();
                for (a, row) in hessian.iter().enumerate().take(kb) {
                    for (b, cell) in row.iter().enumerate().take(kb) {
                        let expected: f64 = (0..n)
                            .map(|i| contributions[i].h_beta_coef * problem.x.get(i, a) * problem.x.get(i, b) / sigma2)
                            .sum();
                        prop_assert!((cell - expected).abs() <= 1e-9 * scale, "H[{a}][{b}]={cell} vs {expected}");
                    }
                    let cross: f64 = (0..n)
                        .map(|i| contributions[i].h_beta_s_coef * problem.x.get(i, a) / sigma)
                        .sum();
                    prop_assert!((hessian[a][kb] - cross).abs() <= 1e-9 * scale, "H[{a}][s]={} vs {cross}", hessian[a][kb]);
                    prop_assert!((hessian[kb][a] - cross).abs() <= 1e-9 * scale, "H[s][{a}]={} vs {cross}", hessian[kb][a]);
                }
                let h_ss: f64 = contributions.iter().map(|c| c.h_ss).sum();
                prop_assert!((hessian[kb][kb] - h_ss).abs() <= 1e-9 * scale, "H[s][s]={} vs {h_ss}", hessian[kb][kb]);

                // `β`ブロックの半正定値性: 対角が非負、任意方向`v'H_ββv ≥ 0`
                for (a, row) in hessian.iter().enumerate().take(kb) {
                    prop_assert!(row[a] >= 0.0, "H[{a}][{a}]={} should be non-negative", row[a]);
                }
                let quad: f64 = (0..kb)
                    .flat_map(|a| (0..kb).map(move |b| (a, b)))
                    .map(|(a, b)| v[a] * hessian[a][b] * v[b])
                    .sum();
                prop_assert!(quad >= -1e-9 * scale, "v'H_bb v={quad} should be non-negative (scale={scale})");
            }
        }
    }
}
