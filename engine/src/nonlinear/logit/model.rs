//! Logitの数式部品: `softplus`・ロジスティック関数・対数尤度、限界効果の`w`と`s`。

use crate::nonlinear::common::{compensated_sum, linear_predictor};
use faer::Mat;

/// `softplus(z) = log(1+exp(z))`の数値的に安定な計算。`z`が大きい正の値でも
/// `exp(z)`のオーバーフローを起こさない（標準的な安定化式、`z.max(0)+log1p(exp(-|z|))`）。
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// ロジスティック関数`Λ(z) = 1/(1+exp(-z))`の数値的に安定な計算。
/// `z`の符号で分岐し、`exp`の引数が常に非正になるようにする（`exp`のオーバーフロー回避）。
pub(super) fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// 対数尤度 `ℓ(θ) = Σᵢ [yᵢzᵢ - softplus(zᵢ)]`（`zᵢ=xᵢ'θ`、`logit.rs`のモジュールdocの数式参照）を
/// `x`・`y`・`params`から直接計算する。`LogitProblem::cost`（`-ℓ(θ)`、argminの
/// `CostFunction`）と同じ数式のΣ部分を共有する（`cost`はこの関数を符号反転して呼ぶ）。
/// argminのトレイトが要求する`Result`型を経由する必要が無い内部専用の計算
/// （適合度統計量向け、収束後のパラメータで1回だけ評価する）のため、独立した
/// 関数として切り出している。
///
/// 総和は補償和（[`compensated_sum`]のdocコメント参照。素朴な逐次加算の丸め誤差が
/// 大標本のNewtonのコスト比較を狂わせるため）。
pub(super) fn log_likelihood(x: &Mat<f64>, y: &Mat<f64>, params: &[f64]) -> f64 {
    let n = x.nrows();
    compensated_sum((0..n).map(|i| {
        let z = linear_predictor(x, i, params);
        (*y.get(i, 0)) * z - softplus(z)
    }))
}

/// 限界効果（`LogitFitted::marginal_effects`のdocコメント「数式（デルタ法）」参照）の
/// `at="overall"`（AME）における`w=(1/n)Σᵢpᵢ(1-pᵢ)`・`s_m=(1/n)Σᵢ(1-2pᵢ)pᵢ(1-pᵢ)xᵢₘ`を
/// 全観測を1回走査して計算する。
pub(super) fn overall_w_and_s(x: &Mat<f64>, params: &[f64]) -> (f64, Vec<f64>) {
    let n = x.nrows();
    let k = x.ncols();
    let mut w = 0.0;
    let mut s = vec![0.0; k];
    for i in 0..n {
        let z = linear_predictor(x, i, params);
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
pub(super) fn at_point_w_and_s(x_bar: &[f64], params: &[f64]) -> (f64, Vec<f64>) {
    let k = x_bar.len();
    let z: f64 = (0..k).map(|m| x_bar[m] * params[m]).sum();
    let p = logistic(z);
    let pq = p * (1.0 - p);
    let coef = (1.0 - 2.0 * p) * pq;
    let s: Vec<f64> = (0..k).map(|m| coef * x_bar[m]).collect();
    (pq, s)
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::nonlinear::common::dydx_and_jacobian;

    #[test]
    fn logistic_and_softplus_are_numerically_stable_for_large_inputs() {
        assert!((logistic(1000.0) - 1.0).abs() < 1e-12);
        assert!(logistic(-1000.0).abs() < 1e-12);
        assert!(softplus(1000.0).is_finite());
        assert!((softplus(1000.0) - 1000.0).abs() < 1e-9);
        assert!(softplus(-1000.0).abs() < 1e-12);
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
}
