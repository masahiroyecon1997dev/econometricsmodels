//! Probitの数式部品: 対数尤度、限界効果の`w`と`s`。

use crate::nonlinear::common::{clamped_pdf_cdf, compensated_sum};
use faer::Mat;
use statrs::distribution::{Continuous, Normal};

// `clamped_pdf_cdf`（`φ(u)`・`Φ(u)`評価前のクランプ）は`nonlinear/common/normal.rs`へ移設した
// （Tobit実装時、打ち切り観測の尤度も同型の`λ=φ/Φ`を含むため共有。`probit.rs`のモジュールdoc
// 「数値安定化について」参照）。

/// 対数尤度 `ℓ(θ) = Σᵢ log Φ(qᵢzᵢ)`（`zᵢ=xᵢ'θ`、`qᵢ=2yᵢ-1`、`probit.rs`のモジュールdocの数式参照）を
/// `x`・`y`・`params`から直接計算する。`ProbitProblem::cost`（`-ℓ(θ)`、argminの
/// `CostFunction`）と同じ数式のΣ部分を共有する（`cost`はこの関数を符号反転して呼ぶ）。
/// argminのトレイトが要求する`Result`型を経由する必要が無い内部専用の計算
/// （適合度統計量向け、収束後のパラメータで1回だけ評価する）のため、独立した
/// 関数として切り出している（`LogitProblem`の`log_likelihood`と同じ位置づけ）。
pub(super) fn log_likelihood(x: &Mat<f64>, y: &Mat<f64>, params: &[f64]) -> f64 {
    let normal = Normal::standard();
    let n = x.nrows();
    // 補償和（`compensated_sum`のdocコメント参照。大標本のNewtonのコスト比較のため）。
    compensated_sum((0..n).map(|i| {
        let z: f64 = (0..x.ncols()).map(|j| *x.get(i, j) * params[j]).sum();
        let q = 2.0 * (*y.get(i, 0)) - 1.0;
        let (_, big_phi) = clamped_pdf_cdf(&normal, q * z);
        big_phi.ln()
    }))
}

/// 限界効果（`ProbitEstimator::marginal_effects`のdocコメント「数式（デルタ法）」参照）の
/// `at="overall"`（AME）における`w=(1/n)Σᵢφ(zᵢ)`・`s_m=(1/n)Σᵢ(-zᵢ)φ(zᵢ)xᵢₘ`を
/// 全観測を1回走査して計算する（`φ'(z)=-zφ(z)`、標準正規PDFの導関数。Logitの
/// `overall_w_and_s`の`w=p(1-p)`・`s_m=(1-2p)p(1-p)xᵢₘ`に相当するProbit版）。
///
/// `U_CLAMP`によるクランプは行わない: `φ(z)`単体は`z`が極端でも滑らかに0へ収束するのみで、
/// `λ=φ/Φ`のような`0.0/0.0`のNaN化リスクが無いため（`probit.rs`のモジュールdoc「数値安定化について」
/// 参照、クランプが必要なのは`Φ`で割る箇所のみ）。
///
/// `z`が有限であることを暗黙の前提にしている（`coef=-z*φ(z)`は`z=±∞`だと`0*∞`のNaNに
/// なりうる）。`fit()`を通過した収束済み・有限の`θ`と、`engine_pybind`側で既に有限性が
/// 保証された`x`から計算する`z`は実質常に有限のため到達不能だが、`logistic`/`softplus`
/// （Logit）が「有限の`z`では絶対にNaNを生まない」設計だったのとは異なる前提であることに
/// 注意。
pub(super) fn overall_w_and_s(x: &Mat<f64>, params: &[f64]) -> (f64, Vec<f64>) {
    let n = x.nrows();
    let k = x.ncols();
    let normal = Normal::standard();
    let mut w = 0.0;
    let mut s = vec![0.0; k];
    for i in 0..n {
        let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        let phi = normal.pdf(z);
        w += phi;
        let coef = -z * phi;
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
/// 中央値）で評価した`w=φ(z̄)`・`s_m=-z̄φ(z̄)x̄ₘ`（`z̄=x̄'θ`）を計算する
/// （Logitの`at_point_w_and_s`に相当するProbit版）。
pub(super) fn at_point_w_and_s(x_bar: &[f64], params: &[f64]) -> (f64, Vec<f64>) {
    let k = x_bar.len();
    let z: f64 = (0..k).map(|m| x_bar[m] * params[m]).sum();
    let phi = Normal::standard().pdf(z);
    let coef = -z * phi;
    let s: Vec<f64> = (0..k).map(|m| coef * x_bar[m]).collect();
    (phi, s)
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::nonlinear::common::dydx_and_jacobian;

    /// `at="overall"`（AME）のヤコビアン（`dydx_and_jacobian`が`overall_w_and_s`の
    /// 出力から計算する`∂dydx_j/∂θ_m`）を、`overall_w_and_s`が返す`w`を`θ`の関数として
    /// 中心差分で数値微分した値と比較する（`LogitEstimator`の対応するテストと同じ技法。
    /// Probitでは`w=φ(z)`ベースの式になる点が異なる）。
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
