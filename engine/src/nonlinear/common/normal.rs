//! 標準正規分布の`φ`・`Φ`の評価（`|u|`のクランプ）。ProbitとTobitの尤度が共有する。

use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// `φ(u)`・`Φ(u)`を評価する前に`u`をこの絶対値以下にクランプする閾値。`u`がこれより
/// 極端になると`λ=φ(u)/Φ(u)`（逆ミルズ比、一般化残差）が`0.0/0.0`のNaNになりうる
/// （実測では`|u|≳39`から発生。本閾値`≈8.126`はそれよりずっと手前で安全に倒す）。
///
/// 元はProbit専用（`ProbitProblem::generalized_residual_at`の一般化残差
/// `λ_i=q_iφ(q_iz_i)/Φ(q_iz_i)`向け）だったが、Tobitの打ち切り観測の尤度
/// （`logΦ(z)`・`log(1-Φ(z))`型）も同型の`λ=φ/Φ`を含み同じリスクを共有するため、
/// Tobit実装時にここへ移設して共有した。
///
/// R言語`stats::binomial(link="probit")$linkinv`の`thresh <- -qnorm(.Machine$double.eps)`
/// と同じ値（`-Φ⁻¹(f64::EPSILON)`）。`Normal::inverse_cdf`は反復計算のためホットパスで
/// 毎回呼ぶのを避け、コンパイル時定数としてハードコードしている（Rとscipyの両方で
/// `8.125890664701908`と算出されることを確認済み）。
pub const U_CLAMP: f64 = 8.125_890_664_701_908;

/// `u`を`[-U_CLAMP, U_CLAMP]`にクランプしてから`(φ(u), Φ(u))`を評価する
/// （`U_CLAMP`のdocコメント参照）。呼び出し側（`cost`/`gradient`/`hessian`/`scores`）が
/// すべて同じ関所を経由することで、`statsmodels`の`Probit`実装に見られる非対称性
/// （`score`/`loglike`はクリップするが`hessian`はしない）を避ける。
pub fn clamped_pdf_cdf(normal: &Normal, u: f64) -> (f64, f64) {
    let u = u.clamp(-U_CLAMP, U_CLAMP);
    (normal.pdf(u), normal.cdf(u))
}
