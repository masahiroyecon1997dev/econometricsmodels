//! FEIVのSargan過剰識別検定。

use faer::Mat;

use crate::shared::error::CommonError;
use crate::shared::instrumental_variables::sargan_test;

/// [`fe_sargan_test`]の戻り値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FeSarganTest {
    pub(super) statistic: f64,
    pub(super) p_value: f64,
    pub(super) df: usize,
}

/// within変換後のデータのSargan過剰識別検定。構造残差`e`を、変換後の全操作変数
/// `Z̃ = [x̃_exog, z̃]`に回帰した`e'Z̃(Z̃'Z̃)⁻¹Z̃'e / σ̂²`を`χ²(df)`と比べる。
///
/// **`σ̂² = e'e/(n − n_absorbed)`**。within変換後の残差の二乗和の期待値は`σ²(n − n_absorbed)`なので、
/// `n`で割ると統計量が`T/(T − 1)`倍（1-wayのとき）に膨らみ、小さい`T`で棄却に偏る
/// （名目5%に対し`T = 3`で約10〜12%）。`n_absorbed = 0`のIV本体は`e'e/n`で同じ式になる。
/// 構造係数`k`個の自由度は引かない（IV本体と同じ。漸近的には同じ）。
/// `fixest`（0.14.2）の`sargan`は`n(1 − SSR_r/SSR_null)`、つまり`σ̂² = e'e/n`の尺度なので、固定効果が
/// あると値が異なる。
///
/// `n_absorbed`は`absorbed_parameters`の数え方（連結成分が複数ある不均衡な2-wayでは実際のランクより
/// 多い）なので、その場合は統計量がわずかに小さい（保守的な）側にずれる。
///
/// 常に等分散（古典的）前提で、`cov_type`には依存しない。`df = len(instruments) − len(x_endog)`は
/// 呼び出し側が決める（丁度識別は検定が無いので呼ばない）。`z_tilde.nrows() > n_absorbed`は
/// 呼び出し側（`fit_fe_iv`の自由度の検査）が保証する。
///
/// # Errors
/// `Z̃'Z̃`が分解できない場合の`CommonError::Internal`。第一段階が同じ列を設計行列として
/// フルランクを検証済みなので到達しない。
pub(super) fn fe_sargan_test(
    z_tilde: &Mat<f64>,
    residuals: &Mat<f64>,
    ssr: f64,
    n_absorbed: usize,
    df: usize,
) -> Result<FeSarganTest, CommonError> {
    debug_assert!(z_tilde.nrows() > n_absorbed);
    let sigma2 = ssr / (z_tilde.nrows() - n_absorbed) as f64;
    let test = sargan_test(z_tilde, residuals, sigma2, df)?;
    Ok(FeSarganTest {
        statistic: test.statistic,
        p_value: test.p_value,
        df,
    })
}
